# 16 — Streaming Data for ML: Pub/Sub and Dataflow

Module 08 introduced Pub/Sub and Dataflow briefly as members of the data plane; this module goes deep on both, because the streaming half of an ML system is where most of the operational complexity lives. Fraud does not arrive in nightly batches — a transaction happens now, and a useful feature is "how many times has this card been used in the last five minutes." Computing that means ingesting an unbounded event stream reliably, windowing it correctly in the face of late and out-of-order data, and landing the results where training and serving can both read them. Pub/Sub is the durable messaging backbone that absorbs the events; Dataflow is the managed Apache Beam runner that transforms them into features. This module keeps BigQuery in module 08's hands — cross-reference it for the warehouse — and focuses entirely on getting data into and through the stream.

## Pub/Sub: topics, subscriptions, and delivery semantics

A producer publishes messages to a **topic**; a consumer reads them through a **subscription**. The decoupling is the point: the fraud app publishes a transaction event and moves on, while any number of independent subscriptions — one landing raw events in BigQuery, one feeding a Dataflow feature pipeline, one triggering an alerting function — consume the same topic at their own pace. Each subscription gets its own copy of the stream and its own backlog.

There are four subscription **delivery types**, and choosing the right one removes a lot of code:

- **Pull** — your consumer explicitly pulls and acknowledges messages. Maximum control, used by Dataflow and custom workers.
- **Push** — Pub/Sub POSTs each message to an HTTPS `--push-endpoint` (a Cloud Run service, for example). Good for serverless consumers you do not want to run a poll loop in.
- **BigQuery** — messages are written *directly* into a BigQuery table with no intermediate compute (module 08's zero-code ingest pattern).
- **Cloud Storage** — messages are batched and written as files (Avro, text, or JSON) to a bucket, useful for cheap archival of the raw stream.

```bash
# Create a topic
gcloud pubsub topics create transactions
gcloud pubsub topics list
gcloud pubsub topics publish transactions \
  --message='{"txn_id":"t-9001","card":"c-42","amount":88.10}' \
  --attribute=source=pos,region=us

# A pull subscription with a longer ack deadline and 7-day retention
gcloud pubsub subscriptions create transactions-worker \
  --topic=transactions \
  --ack-deadline=60 \
  --message-retention-duration=7d \
  --retain-acked-messages \
  --expiration-period=never

# A push subscription delivering to a Cloud Run consumer
gcloud pubsub subscriptions create transactions-push \
  --topic=transactions \
  --push-endpoint=https://fraud-consumer-xxxx.run.app/events \
  --ack-deadline=30
```

A few of these flags carry real operational weight. `--ack-deadline` is how long Pub/Sub waits for an acknowledgment before redelivering — set it above your worst-case processing time or you will get duplicate work. `--message-retention-duration` (up to 31 days) controls how far back you can replay; `--retain-acked-messages` keeps already-acked messages within that window so you can seek backward through them. `--expiration-period` controls when an idle subscription is auto-deleted (`never` for production subscriptions you cannot afford to lose).

## Dead-letter topics, ordering, and exactly-once

Three subscription features turn Pub/Sub from "a queue" into a production-grade ingestion layer.

A **dead-letter topic** catches messages that fail repeatedly, so one poison message does not wedge your pipeline forever. You attach a separate topic and a maximum delivery count (5–100); after that many failed deliveries, the message is forwarded to the dead-letter topic where you can inspect and reprocess it.

**Message ordering** guarantees that messages sharing an *ordering key* are delivered in publish order. For fraud, ordering per card (`--ordering-key` at publish time, `--enable-message-ordering` on the subscription) means a "card blocked" event never arrives before the transaction that triggered it. Ordering is scoped per key, not globally, so throughput stays high.

**Exactly-once delivery** (`--enable-exactly-once-delivery`) guarantees no acknowledged message is redelivered within the ack window — within a single region. It does not absolve you of idempotency across the whole system, but it eliminates the most common source of duplicates.

```bash
# Dead-letter topic wired to a subscription
gcloud pubsub topics create transactions-dlq
gcloud pubsub subscriptions create transactions-worker \
  --topic=transactions \
  --dead-letter-topic=transactions-dlq \
  --max-delivery-attempts=5

# Ordered, exactly-once subscription (publish with an ordering key)
gcloud pubsub subscriptions create transactions-ordered \
  --topic=transactions \
  --enable-message-ordering \
  --enable-exactly-once-delivery
gcloud pubsub topics publish transactions \
  --message='{"txn_id":"t-9002","card":"c-42"}' \
  --ordering-key=c-42
```

The rule that survives all of this: **Pub/Sub delivery is at-least-once by default**, so downstream consumers must be **idempotent** — the same discipline you apply to event-driven functions. Exactly-once narrows the window; it does not let you skip the idempotency key on your feature writes.

## Schemas, replay, and direct-to-BigQuery

Attaching a **schema** to a topic validates every published message, so a malformed transaction is rejected at ingest rather than corrupting a feature table downstream. Pub/Sub supports **Avro** and **Protocol Buffer** schemas.

```bash
# Register an Avro schema and bind it to a topic
gcloud pubsub schemas create txn-schema \
  --type=AVRO \
  --definition-file=txn.avsc
gcloud pubsub topics create transactions-validated \
  --schema=txn-schema \
  --message-encoding=JSON
```

**Seeking and replay** let you rewind a subscription to a point in time or to a **snapshot**, redelivering everything after it. This is invaluable in ML: when you fix a bug in a Dataflow feature transform, you replay the last few days of events through the corrected pipeline to rebuild the features, without asking the source systems to re-emit anything.

```bash
# Snapshot a subscription's ack state, then seek back to it later
gcloud pubsub snapshots create pre-deploy --subscription=transactions-worker
gcloud pubsub subscriptions seek transactions-worker --snapshot=pre-deploy
# Or seek to a wall-clock time (requires retained messages)
gcloud pubsub subscriptions seek transactions-worker \
  --time=2026-07-01T00:00:00Z
```

For the zero-compute path, a **BigQuery subscription** writes straight into a table using the topic schema (module 08 covers the warehouse side):

```bash
gcloud pubsub subscriptions create transactions-to-bq \
  --topic=transactions-validated \
  --bigquery-table=myco-fraud-dev:fraud.raw_events \
  --use-topic-schema
```

One deprecation note: **Pub/Sub Lite is shut down** (end of service March 2026). Do not design new systems around it — standard Pub/Sub is the only choice, and it now covers the low-cost, high-throughput cases Lite was meant for.

## Dataflow and the Beam programming model

**Dataflow** is Google Cloud's fully managed runner for **Apache Beam** pipelines. Beam gives you one programming model for both **batch** (bounded data — a day of transactions in BigQuery) and **streaming** (unbounded data — the live Pub/Sub topic), and Dataflow provisions the workers, autoscales them, handles failures, and commits results exactly once.

The model has a small vocabulary. A **PCollection** is a distributed dataset flowing through the pipeline. A **PTransform** is an operation on it; the workhorse is **ParDo**, which applies a function to every element (a `DoFn`). For unbounded streams you add **windowing** — grouping elements by event time so aggregations are well-defined:

- **Fixed windows** — non-overlapping intervals (transactions per 60-second bucket).
- **Sliding windows** — overlapping intervals (a 5-minute count updated every minute — exactly the velocity feature fraud needs).
- **Session windows** — gap-based, grouping bursts of activity separated by idle time (a user's activity session).

Two more concepts make streaming correct. A **watermark** is Beam's estimate of "event time up to which we have seen all data"; it drives when a window is considered complete. **Triggers** decide when to emit a window's result — at the watermark, early on a timer, or late when stragglers arrive — and how to handle **late data** (elements that arrive after the watermark passed their window).

```python
import apache_beam as beam
from apache_beam.transforms.window import SlidingWindows

# Rolling 5-minute transaction count per card, updated every minute
with beam.Pipeline() as p:
    (p
     | "Read" >> beam.io.ReadFromPubSub(topic="projects/myco-fraud-dev/topics/transactions")
     | "Parse" >> beam.Map(parse_txn)                       # -> (card, 1)
     | "Window" >> beam.WindowInto(SlidingWindows(300, 60)) # 300s window, 60s period
     | "Count" >> beam.CombinePerKey(sum)
     | "Write" >> beam.io.WriteToBigQuery(
           "myco-fraud-dev:fraud.card_velocity_5m",
           write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))
```

That sliding-window count is a **streaming feature**: it lands in a BigQuery table that both trains the model (historical rows) and feeds the online store (freshest row per card). This is the streaming path into the Vertex AI Feature Store covered in module 17.

## Running Dataflow: templates, Flex Templates, and autoscaling

You rarely deploy a raw Beam pipeline in production; you package it. **Google-provided templates** cover the common shapes with no code — the Pub/Sub → BigQuery streaming template is the canonical one. **Flex Templates** package *your own* pipeline as a container image plus a metadata file, so CI/CD can launch it with parameters and no local Beam environment. (Older **classic templates**, which stage a serialized graph, still work but Flex is the current recommendation.)

```bash
# Launch a Google-provided template (Pub/Sub Subscription -> BigQuery)
gcloud dataflow jobs run txn-ingest \
  --gcs-location=gs://dataflow-templates-us-central1/latest/PubSub_Subscription_to_BigQuery \
  --region=us-central1 \
  --staging-location=gs://myco-fraud-staging/df \
  --parameters=inputSubscription=projects/myco-fraud-dev/subscriptions/transactions-worker,outputTableSpec=myco-fraud-dev:fraud.raw_events

# Build a Flex Template from your own pipeline image, then run it
gcloud dataflow flex-template build gs://myco-fraud-staging/templates/velocity.json \
  --image=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/velocity:v1 \
  --sdk-language=PYTHON \
  --metadata-file=metadata.json
gcloud dataflow flex-template run velocity-features \
  --template-file-gcs-location=gs://myco-fraud-staging/templates/velocity.json \
  --region=us-central1 \
  --parameters=inputTopic=projects/myco-fraud-dev/topics/transactions \
  --max-workers=10 \
  --worker-machine-type=n2-standard-4 \
  --enable-streaming-engine
```

Dataflow **autoscales** worker count to the backlog; `--max-workers` caps it and `--worker-machine-type` sizes each. **Streaming Engine** moves window and state storage off the workers into the Dataflow backend, so scaling is faster and workers are cheaper — enable it for any serious streaming job.

**Dataflow ML** adds a `RunInference` PTransform that runs a model over a PCollection inside the pipeline. For large, embarrassingly-parallel scoring, this is a real alternative to Vertex AI batch prediction: read from BigQuery or Pub/Sub, `RunInference` with your model handler, write results back — one pipeline, autoscaled, with the preprocessing and postprocessing in the same graph.

```python
from apache_beam.ml.inference.base import RunInference
# ... | "Score" >> RunInference(model_handler) | "Write" >> beam.io.WriteToBigQuery(...)
```

## Stopping streaming jobs, and the gotchas

A streaming job runs forever, so how you stop it matters. **Drain** stops ingesting new data but lets in-flight windows finish and commit — the safe way to deploy a new version without losing buffered data. **Cancel** halts immediately and may drop in-flight data — use it only when the job is broken.

```bash
gcloud dataflow jobs list --region=us-central1 --status=active
gcloud dataflow jobs drain JOB_ID --region=us-central1    # graceful
gcloud dataflow jobs cancel JOB_ID --region=us-central1   # immediate
```

The gotchas that bite teams:

- **Streaming cost is always-on.** Unlike a batch job that ends, a streaming pipeline holds workers 24/7. Right-size `--max-workers`, use Streaming Engine, and question whether a micro-batch on a schedule would be cheaper.
- **Late data silently drops** unless you configure allowed lateness and a trigger to handle it. Decide explicitly what happens to a transaction that arrives 20 minutes late.
- **Exactly-once in Beam** applies to Dataflow's internal processing and its sinks that support it; a non-idempotent external write can still double-count. Keep the idempotency key.
- **Hot keys** — one card with millions of events, or a null key — create a straggler worker that autoscaling cannot fix. Detect skew and rebalance (salting the key, or a combiner) rather than adding workers.

## How this fits the whole solution

This module is the live front edge of the end-to-end system from module 12. **Pub/Sub** ingests every transaction and fans it out: a **BigQuery subscription** lands raw events for analytics (module 08's warehouse), while a **Dataflow** streaming pipeline computes windowed features. Those features flow two ways — into **BigQuery feature tables** that become training data for **Vertex AI custom training** (module 09), and into the **Vertex AI Feature Store** online store (module 17) that serving reads at prediction time, which is how the same feature definition stays consistent between training and serving. Dataflow's `RunInference` offers a scalable batch-scoring path alongside Vertex batch prediction. Replay and snapshots let you rebuild features after a bug without touching upstream systems, and drain-not-cancel keeps deployments lossless. Get this streaming layer right and the rest of the pipeline inherits clean, fresh, correctly-windowed data.

## Key takeaways

- **Pub/Sub** decouples producers from consumers via **topics** and **subscriptions** (pull / push / BigQuery / Cloud Storage); delivery is **at-least-once**, so consumers must be **idempotent** even with `--enable-exactly-once-delivery`.
- Production subscriptions use **dead-letter topics** (`--dead-letter-topic` + `--max-delivery-attempts`), **ordering keys** (`--enable-message-ordering`), **schemas** (Avro/Proto), and **seek/snapshot replay** to rebuild features. **Pub/Sub Lite is shut down** — use standard Pub/Sub.
- **Dataflow** runs **Apache Beam** for batch and streaming; **windowing** (fixed/sliding/session), **watermarks**, and **triggers** make streaming feature computation correct, and **sliding windows** are how you build rolling velocity features.
- Package pipelines as **Flex Templates** for CI/CD, enable **Streaming Engine** and cap **`--max-workers`** to control always-on cost, use **`RunInference`** for in-pipeline scoring, and **drain** (not cancel) to redeploy without data loss.

## CLI cheat-sheet

```bash
# --- Pub/Sub: topics & publishing ---
gcloud pubsub topics create transactions
gcloud pubsub topics list
gcloud pubsub topics publish transactions --message='{...}' \
  --attribute=k=v --ordering-key=c-42

# --- Pub/Sub: subscriptions ---
gcloud pubsub subscriptions create SUB --topic=transactions \
  --ack-deadline=60 --message-retention-duration=7d --retain-acked-messages \
  --expiration-period=never
gcloud pubsub subscriptions create SUB --topic=T --push-endpoint=https://...
gcloud pubsub subscriptions create SUB --topic=T \
  --dead-letter-topic=transactions-dlq --max-delivery-attempts=5
gcloud pubsub subscriptions create SUB --topic=T \
  --enable-message-ordering --enable-exactly-once-delivery
gcloud pubsub subscriptions create SUB --topic=T \
  --bigquery-table=proj:ds.tbl --use-topic-schema
gcloud pubsub subscriptions pull SUB --auto-ack --limit=10

# --- Pub/Sub: schemas, snapshots, replay ---
gcloud pubsub schemas create txn-schema --type=AVRO --definition-file=txn.avsc
gcloud pubsub topics create T --schema=txn-schema --message-encoding=JSON
gcloud pubsub snapshots create SNAP --subscription=SUB
gcloud pubsub subscriptions seek SUB --snapshot=SNAP
gcloud pubsub subscriptions seek SUB --time=2026-07-01T00:00:00Z

# --- Dataflow: run templates ---
gcloud dataflow jobs run JOB \
  --gcs-location=gs://dataflow-templates-us-central1/latest/PubSub_Subscription_to_BigQuery \
  --region=us-central1 --staging-location=gs://myco-fraud-staging/df \
  --parameters=inputSubscription=...,outputTableSpec=proj:ds.tbl

# --- Dataflow: Flex Templates ---
gcloud dataflow flex-template build gs://.../tmpl.json \
  --image=REGION-docker.pkg.dev/PROJ/REPO/img:v1 --sdk-language=PYTHON \
  --metadata-file=metadata.json
gcloud dataflow flex-template run JOB \
  --template-file-gcs-location=gs://.../tmpl.json --region=us-central1 \
  --parameters=inputTopic=... --max-workers=10 \
  --worker-machine-type=n2-standard-4 --enable-streaming-engine

# --- Dataflow: manage jobs ---
gcloud dataflow jobs list --region=us-central1 --status=active
gcloud dataflow jobs drain  JOB_ID --region=us-central1   # graceful stop
gcloud dataflow jobs cancel JOB_ID --region=us-central1   # immediate stop
```

## Try it

Build a streaming feature pipeline for the fraud system:

1. Create a `transactions` topic and publish a handful of JSON transaction messages with `gcloud pubsub topics publish`, using `--ordering-key` set to the card id.
2. Add a **BigQuery subscription** (`--bigquery-table` + `--use-topic-schema`) and confirm raw events land in `fraud.raw_events` with zero code.
3. Write a small Beam pipeline that reads the topic, applies a **sliding window** (300s window, 60s period), counts transactions per card, and writes to a `fraud.card_velocity_5m` table; run it on Dataflow with `--enable-streaming-engine` and `--max-workers=5`.
4. Attach a **dead-letter topic** with `--max-delivery-attempts=5` to a pull subscription, publish a malformed message, and watch it land in the DLQ after retries.
5. Take a **snapshot**, publish more events, then `seek` back to the snapshot and confirm the events replay — you have just rebuilt features from history without re-emitting anything upstream. Finish by **draining** (not cancelling) the streaming job.
