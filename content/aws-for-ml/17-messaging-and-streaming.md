# 17 — Messaging and Streaming for ML Data

The earlier modules mentioned queues and streams in passing — SQS buffering inference requests, EventBridge scheduling retraining — but an ML system lives or dies on how it moves data between components that run at different speeds. A model server that processes 50 requests per second cannot be wired directly to a producer that bursts 5,000; something has to absorb the shock, retry the failures, and fan the same event out to the several consumers that care about it. This module covers the async and streaming services that do exactly that — SQS, SNS, Kinesis, Data Firehose, and MSK — and gives you a clear rule for which one to reach for.

## Why decoupling matters for ML

Wire a producer directly to a consumer and you have built a fragile system: if the consumer is slow the producer blocks, if it is down the request is lost, and a traffic spike topples everything. A queue or stream sits between them and breaks the coupling — the producer writes and moves on, the consumer reads at whatever rate it can sustain. This buys you **buffering** (a burst of async inference requests piles up in a queue instead of overwhelming the model server), **backpressure** (the consumer pulls at its own pace rather than being pushed past its limit), **retries and durability** (a failed message is redelivered instead of dropped, and a poison message eventually lands in a dead-letter queue for inspection), and **fan-out** (one event — a new model registered — reaches several independent consumers that the producer need not know about). For streaming specifically, an ordered high-throughput stream is what feeds **near-real-time feature ingestion**: clickstream and telemetry events flow in continuously, get aggregated into online features, and are ready for the next inference within seconds. These are the difference between a pipeline that survives a spike and one that pages you at 2am.

## Amazon SQS: durable queues that absorb bursts

**Amazon SQS** is a fully managed message queue: a producer sends messages, a consumer polls and processes them, and SQS holds each message durably until it is deleted or expires. There are two flavors. **Standard queues** offer nearly unlimited throughput with at-least-once delivery and best-effort ordering — the right default. **FIFO queues** (name must end in `.fifo`) guarantee exact ordering and exactly-once processing within a message group, at a lower throughput ceiling — reach for them only when order or deduplication genuinely matters. The classic ML use is buffering: async inference or preprocessing requests land in the queue and a fleet of workers (or Lambda) drains them at a controlled rate, so a burst never overwhelms the model.

Two settings define correctness. **Visibility timeout** is how long a received message is hidden from other consumers while one worker processes it; set it longer than your worst-case processing time or the message reappears and gets processed twice. **Long polling** (`--wait-time-seconds` up to 20 on receive) makes the consumer wait for messages to arrive instead of returning empty immediately, which slashes empty-receive costs and latency.

```bash
# Standard queue with a 5-minute visibility timeout (long inference jobs)
aws sqs create-queue \
  --queue-name ml-inference-requests \
  --attributes VisibilityTimeout=300

# FIFO queue — name MUST end in .fifo — for strictly ordered event processing
aws sqs create-queue \
  --queue-name ml-events.fifo \
  --attributes FifoQueue=true,ContentBasedDeduplication=true
```

A **dead-letter queue (DLQ)** catches messages that fail repeatedly so they do not clog the main queue or retry forever. You create a second queue as the DLQ, then attach a `RedrivePolicy` to the source queue naming the DLQ's ARN and a `maxReceiveCount` — after that many failed receives, SQS moves the message to the DLQ where you can inspect the poison payloads.

```bash
# Wire a DLQ: after 5 failed processing attempts, park the message for inspection
aws sqs set-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/ml-inference-requests \
  --attributes '{
    "RedrivePolicy": "{\"deadLetterTargetArn\":\"arn:aws:sqs:us-east-1:123456789012:ml-inference-dlq\",\"maxReceiveCount\":\"5\"}"
  }'
```

The everyday loop is send, receive (with long polling), process, delete. A message you receive is *not* removed — you must explicitly delete it with the `ReceiptHandle` after successful processing, which is what makes redelivery-on-failure work.

```bash
aws sqs send-message \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/ml-inference-requests \
  --message-body '{"job_id":"j-42","s3_input":"s3://ml-data/pending/img-42.jpg"}'

# Long-poll for up to 20s, grab up to 10 messages
aws sqs receive-message \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/ml-inference-requests \
  --max-number-of-messages 10 --wait-time-seconds 20

aws sqs delete-message \
  --queue-url https://sqs.us-east-1.amazonaws.com/123456789012/ml-inference-requests \
  --receipt-handle <ReceiptHandle-from-receive>
```

One hard limit shapes ML payloads: **the maximum message size is 256 KB.** An image, an audio clip, or a large feature vector will not fit. The standard pattern is the **S3 pointer**: write the large payload to S3, and put only the S3 URI in the SQS message. The consumer reads the pointer, fetches the object, and processes it — the queue carries kilobytes of metadata while S3 carries the gigabytes.

The cleanest way to drain an SQS queue is a Lambda **event source mapping**: Lambda polls the queue for you and invokes your function with a batch. `--batch-size` controls how many messages per invocation. Critically, add `--function-response-types ReportBatchItemFailures` so a **partial batch response** is possible — your function returns the message IDs that failed, and only those are redelivered instead of the entire batch (without this, one bad message in a batch of ten forces all ten to retry).

```bash
aws lambda create-event-source-mapping \
  --event-source-arn arn:aws:sqs:us-east-1:123456789012:ml-inference-requests \
  --function-name drain-inference \
  --batch-size 10 \
  --function-response-types ReportBatchItemFailures
```

## Amazon SNS: pub/sub and fan-out

**Amazon SNS** is push-based publish/subscribe. A publisher sends a message to a **topic**, and SNS immediately pushes a copy to every **subscriber** — SQS queues, Lambda functions, HTTP endpoints, email, SMS. Where SQS is one-producer-to-one-pool-of-workers, SNS is one-message-to-many-consumers. This is **fan-out**: publish "a new model version was registered" once, and independent consumers each react — one kicks off a canary deployment, one refreshes a cache, one posts to Slack — none aware of the others.

```bash
aws sns create-topic --name model-registered

aws sns subscribe --topic-arn arn:aws:sns:us-east-1:123456789012:model-registered \
  --protocol lambda \
  --notification-endpoint arn:aws:lambda:us-east-1:123456789012:function:deploy-canary

aws sns publish --topic-arn arn:aws:sns:us-east-1:123456789012:model-registered \
  --message '{"model":"fraud-detector","version":"v7","artifact":"s3://models/fraud/v7/"}'
```

The most durable fan-out pattern in AWS is **SNS + SQS fan-out**: subscribe several SQS queues to one SNS topic. SNS handles the branching; each SQS queue gives its consumer independent buffering, retries, and a DLQ. If one consumer is down, its queue simply backs up while the others process normally — the message is never lost. This is the backbone of event-driven ML architectures.

```bash
# Fan a topic out to a durable per-consumer queue
aws sns subscribe --topic-arn arn:aws:sns:us-east-1:123456789012:model-registered \
  --protocol sqs \
  --notification-endpoint arn:aws:sqs:us-east-1:123456789012:cache-refresh
```

**Message filtering** lets a subscriber receive only the subset of messages it cares about, so you do not need a separate topic per event type. You attach a `FilterPolicy` — a JSON object matched against message attributes — to the subscription; SNS only delivers messages whose attributes match.

```bash
# This subscriber only wants messages where model_type == "vision"
aws sns set-subscription-attributes \
  --subscription-arn arn:aws:sns:us-east-1:123456789012:model-registered:abcd-1234 \
  --attribute-name FilterPolicy \
  --attribute-value '{"model_type":["vision"]}'
```

SNS also offers **FIFO topics** (paired with FIFO queues) when fan-out must preserve strict ordering, and it is the standard target for **alerting**: a CloudWatch alarm on model latency or a data-drift metric publishes to an SNS topic that emails the on-call engineer and pages PagerDuty.

## Amazon Kinesis Data Streams: ordered, high-throughput streaming

**Amazon Kinesis Data Streams** is for continuous, ordered, high-throughput event streams — think clickstreams, IoT telemetry, and application events flowing in by the thousands per second. A stream is divided into **shards**; each shard is an ordered sequence with a fixed capacity (1 MB/s or 1,000 records/s in, 2 MB/s out). Every record carries a **partition key**, and Kinesis hashes it to pick a shard — so all records with the same key (e.g. the same `user_id`) land on the same shard **in order**. This per-key ordering is the property that distinguishes Kinesis from SQS and makes it the right tool for real-time event streams feeding **feature computation, online features, and streaming inference**.

You pick a **capacity mode** at creation. **On-demand** (`StreamMode=ON_DEMAND`) auto-scales shards to your traffic — the safe default when load is unpredictable. **Provisioned** (`--shard-count`) is cheaper for steady, known throughput but you manage the shard count yourself. Records are retained 24 hours by default, extendable up to 365 days, so a consumer can replay recent history — valuable for backfilling a feature or recovering from a bad deploy.

```bash
# On-demand: Kinesis manages shards for you
aws kinesis create-stream \
  --stream-name events-firehose \
  --stream-mode-details StreamMode=ON_DEMAND

# Provisioned: you own the shard count
aws kinesis create-stream --stream-name events-prov --shard-count 4

aws kinesis describe-stream-summary --stream-name events-firehose
```

Producers call `put-record` (one record) or `put-records` (a batch, far more efficient); high-volume producers use the **Kinesis Producer Library (KPL)** to aggregate and batch automatically. Note the record `Data` is base64 on the wire.

```bash
aws kinesis put-record \
  --stream-name events-firehose \
  --partition-key user_123 \
  --data "$(echo -n '{"event":"click","user":"user_123"}' | base64)"
```

The classic consumer reads via a **shard iterator**: get an iterator for a shard (from `LATEST`, `TRIM_HORIZON`, or a sequence number), then loop `get-records`, following the returned `NextShardIterator`.

```bash
ITER=$(aws kinesis get-shard-iterator \
  --stream-name events-firehose \
  --shard-id shardId-000000000000 \
  --shard-iterator-type LATEST \
  --query 'ShardIterator' --output text)

aws kinesis get-records --shard-iterator "$ITER" --limit 100
```

Standard consumers *share* the shard's 2 MB/s read capacity — add a few consumers and they contend. **Enhanced Fan-Out (EFO)** gives each registered consumer its own dedicated 2 MB/s per shard and lower latency, so multiple independent feature pipelines can read the same stream at full speed without starving each other. You register a consumer against the stream ARN:

```bash
aws kinesis register-stream-consumer \
  --stream-arn arn:aws:kinesis:us-east-1:123456789012:stream/events-firehose \
  --consumer-name feature-pipeline
```

## Amazon Data Firehose: zero-code delivery into the lake

**Amazon Data Firehose** (formerly Kinesis Data Firehose) is the managed, no-code path from a stream to storage. You point a **delivery stream** at a destination — **S3, Redshift, OpenSearch, Splunk, or Iceberg tables** — and Firehose buffers incoming records and writes them out on a size or time threshold, with no consumer code to run or scale. For ML, this is how streaming data **lands in the S3 lake for training**: raw events flow into Firehose and accumulate as objects in S3, ready for Glue and Athena.

Firehose earns its keep with three built-in transforms. **Buffering hints** control the batch size and flush interval. **Format conversion** turns incoming JSON into columnar **Parquet** on the way in — so the lake is query-efficient from the start, no separate ETL. **Dynamic partitioning** routes records into date- or attribute-based S3 prefixes (e.g. `year=2026/month=07/`), which is exactly the partition layout Athena needs to scan cheaply (see module 08). Note the service CLI is `aws firehose`, not `aws kinesis`.

```bash
# DirectPut delivery stream landing into S3 with 128 MB / 60s buffering
aws firehose create-delivery-stream \
  --delivery-stream-name events-to-lake \
  --delivery-stream-type DirectPut \
  --extended-s3-destination-configuration '{
    "RoleARN":"arn:aws:iam::123456789012:role/firehose-delivery",
    "BucketARN":"arn:aws:s3:::ml-data-lake",
    "Prefix":"events/",
    "BufferingHints":{"SizeInMBs":128,"IntervalInSeconds":60},
    "CompressionFormat":"UNCOMPRESSED"
  }'

aws firehose put-record \
  --delivery-stream-name events-to-lake \
  --record '{"Data":"eyJldmVudCI6ImNsaWNrIn0="}'
```

A common wiring is **Kinesis Data Streams → Firehose → S3**: Kinesis gives you ordered real-time reads for the online feature path, and a Firehose consumer on the same stream simultaneously archives everything to the lake for training. One stream, two consumers, two purposes.

## Amazon MSK: managed Kafka when you already speak Kafka

**Amazon MSK (Managed Streaming for Apache Kafka)** is fully managed Apache Kafka. Reach for it when a team **already runs Kafka**, has existing producers/consumers and Kafka tooling (Kafka Connect, Streams, Schema Registry), or needs the broad Kafka ecosystem — not when starting greenfield on AWS, where Kinesis is simpler and requires no cluster thinking. **MSK Serverless** removes broker and capacity management, auto-scaling to your throughput, which narrows the operational gap with Kinesis considerably. The current CLI creates clusters with `create-cluster-v2`, which takes either a `--provisioned` or a `--serverless` block.

```bash
# MSK Serverless cluster with IAM auth
aws kafka create-cluster-v2 \
  --cluster-name ml-events \
  --serverless '{
    "VpcConfigs":[{"SubnetIds":["subnet-aaa","subnet-bbb"]}],
    "ClientAuthentication":{"Sasl":{"Iam":{"Enabled":true}}}
  }'

aws kafka list-clusters-v2
```

The decision between MSK and Kinesis usually comes down to team and ecosystem, not raw capability: choose **Kinesis** for AWS-native simplicity and tight integration with Firehose/Lambda/Analytics; choose **MSK** when Kafka is a hard requirement or you are lifting an existing Kafka workload into AWS.

## EventBridge vs SQS/SNS/Kinesis

These are easy to conflate, so the one-line distinction: **EventBridge** (covered in modules 07 and 11) is an **event bus** — it routes events to targets by content-matching *rules* and runs *schedules* (nightly retraining, cron-style triggers); **SQS/SNS** are your **queue** and **pub/sub** primitives — durable buffering and one-to-many fan-out of messages; **Kinesis** is an **ordered, high-throughput stream** you read and replay, for continuous data feeds. Rule of thumb: EventBridge for "route this event by its content or on a schedule," SQS for "buffer this work," SNS for "notify everyone," Kinesis for "process this ordered firehose of events."

## Choosing a messaging or streaming service

| Service | Model | Ordering | Throughput | Consumers | Reach for it when |
|---|---|---|---|---|---|
| **SQS** | Queue (pull) | Best-effort (FIFO opt-in) | Very high (Standard) | One pool of workers | Buffer bursty async work; decouple producer/consumer; retries + DLQ |
| **SNS** | Pub/sub (push) | Best-effort (FIFO opt-in) | Very high | Many, independent | Fan-out one event to many; alerting; SNS+SQS durable fan-out |
| **Kinesis Data Streams** | Ordered stream (pull) | Per-shard (per partition key) | High, shard-scaled | Many; replayable; EFO for dedicated | Real-time ordered event feeds; online-feature ingestion; replay |
| **Data Firehose** | Managed delivery (no code) | N/A | High | Delivers to S3/Redshift/OpenSearch/Iceberg | Zero-code landing of streaming data into the lake; Parquet conversion |
| **MSK (Kafka)** | Ordered log (Kafka) | Per-partition | Very high | Kafka consumer groups | Existing Kafka workload / ecosystem; Kafka is a requirement |

## How this fits the whole ML solution

Messaging and streaming are the shock absorbers and the arteries of the architecture. On the ingestion side, Kinesis carries real-time events and Firehose lands them in the S3 lake (module 04/08) where Glue and Athena turn them into training data. On the serving side, SQS buffers async inference so a traffic burst queues instead of crashing the model server, and SNS+SQS fan-out lets a single lifecycle event — a model registered, a drift alarm firing — reach every consumer that must react, each with its own durable queue and retry behavior. Where Lambda (module 07) is the connective glue that fires on events, these services are what make the events reliable, ordered, and buffered — the reason the pipeline degrades gracefully under load instead of losing data.

## Key takeaways

- Decoupling buys buffering, backpressure, retries, and fan-out — the difference between a pipeline that survives a spike and one that drops data.
- SQS is a durable pull queue: standard vs FIFO, visibility timeout sized above processing time, long polling, DLQ via `RedrivePolicy` + `maxReceiveCount`; 256 KB cap means large payloads go to S3 with only a pointer in the message.
- Drain SQS with a Lambda event source mapping using `--batch-size` and `--function-response-types ReportBatchItemFailures` for partial-batch retries.
- SNS is push pub/sub for fan-out; SNS+SQS is the durable fan-out pattern, and `FilterPolicy` delivers only matching messages.
- Kinesis is ordered per-shard streaming (partition key → shard) with on-demand or provisioned capacity and Enhanced Fan-Out for dedicated per-consumer throughput; Firehose is the zero-code delivery into S3/Redshift/OpenSearch/Iceberg with Parquet conversion and dynamic partitioning.
- Use MSK when Kafka is already in play; use EventBridge for content-based routing and schedules, not for buffering or high-throughput streams.

## CLI cheat-sheet

```bash
# --- SQS ---
aws sqs create-queue --queue-name ml-inference-requests --attributes VisibilityTimeout=300
aws sqs create-queue --queue-name ml-events.fifo --attributes FifoQueue=true,ContentBasedDeduplication=true
aws sqs send-message --queue-url <url> --message-body '{"job":"j-1"}'
aws sqs receive-message --queue-url <url> --max-number-of-messages 10 --wait-time-seconds 20  # long poll
aws sqs delete-message --queue-url <url> --receipt-handle <handle>
aws sqs get-queue-attributes --queue-url <url> --attribute-names All
aws sqs set-queue-attributes --queue-url <url> \
  --attributes '{"RedrivePolicy":"{\"deadLetterTargetArn\":\"<dlq-arn>\",\"maxReceiveCount\":\"5\"}"}'  # DLQ

# --- SNS ---
aws sns create-topic --name model-registered
aws sns subscribe --topic-arn <arn> --protocol sqs --notification-endpoint <queue-arn>   # fan-out
aws sns publish --topic-arn <arn> --message '{"model":"v7"}'
aws sns set-subscription-attributes --subscription-arn <arn> \
  --attribute-name FilterPolicy --attribute-value '{"model_type":["vision"]}'            # filtering

# --- Kinesis Data Streams ---
aws kinesis create-stream --stream-name events --stream-mode-details StreamMode=ON_DEMAND
aws kinesis create-stream --stream-name events --shard-count 4                            # provisioned
aws kinesis describe-stream-summary --stream-name events
aws kinesis put-record --stream-name events --partition-key user_123 --data <base64>
aws kinesis put-records --stream-name events --records file://records.json                # batch
aws kinesis get-shard-iterator --stream-name events --shard-id shardId-000000000000 --shard-iterator-type LATEST
aws kinesis get-records --shard-iterator <iter> --limit 100
aws kinesis register-stream-consumer --stream-arn <arn> --consumer-name feature-pipeline  # enhanced fan-out

# --- Data Firehose (service CLI is 'firehose') ---
aws firehose create-delivery-stream --delivery-stream-name events-to-lake \
  --delivery-stream-type DirectPut \
  --extended-s3-destination-configuration '{"RoleARN":"<role>","BucketARN":"<bucket>","Prefix":"events/","BufferingHints":{"SizeInMBs":128,"IntervalInSeconds":60},"CompressionFormat":"UNCOMPRESSED"}'
aws firehose put-record --delivery-stream-name events-to-lake --record '{"Data":"<base64>"}'

# --- MSK (managed Kafka) ---
aws kafka create-cluster-v2 --cluster-name ml-events \
  --serverless '{"VpcConfigs":[{"SubnetIds":["subnet-aaa","subnet-bbb"]}],"ClientAuthentication":{"Sasl":{"Iam":{"Enabled":true}}}}'
aws kafka list-clusters-v2

# --- Lambda event source mapping (drain a queue/stream) ---
aws lambda create-event-source-mapping --event-source-arn <sqs-or-kinesis-arn> \
  --function-name drain-inference --batch-size 10 \
  --function-response-types ReportBatchItemFailures                                        # partial batch
```

## Try it

Stand up the async-inference buffering pattern end to end. Create an SQS standard queue and a second queue to serve as its DLQ, then wire them together with a `RedrivePolicy` and `maxReceiveCount` of 3. Write a Lambda that reads a batch, treats messages whose body contains a `"fail"` flag as failures, and returns them via a partial batch response — then attach it with an event source mapping using `--function-response-types ReportBatchItemFailures`. Send a mix of good and bad messages, confirm the good ones are deleted and the bad ones land in the DLQ after three attempts. For the streaming half, create an on-demand Kinesis stream, `put-record` a few events with the same partition key, and read them back with a shard iterator to see that same-key ordering holds. Finally, point a Firehose delivery stream at an S3 bucket and confirm records land as objects in the lake — the moment streaming data becomes training data.
