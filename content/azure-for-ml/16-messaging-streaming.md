# 16 — Messaging and Streaming: Event Hubs, Service Bus, Event Grid

Real ML systems are event-driven: a transaction arrives and must be scored, a file lands and must be processed, a training job finishes and the next stage must start, a model drifts and someone must be paged. Azure has three distinct messaging services for these patterns, and they are *not* interchangeable — using the wrong one is a common architecture mistake. **Event Hubs** is a high-throughput streaming pipe for telemetry and events. **Service Bus** is an enterprise message broker for reliable, ordered, transactional work queues. **Event Grid** is a lightweight publish-subscribe router for reactive event notifications. The data-services module introduced Event Hubs as the streaming front door; this module puts all three side by side, shows when each fits an ML data pipeline, and covers the operations you actually run. In the end-to-end solution, these are the connective tissue that moves data and triggers actions between the ingestion, feature, training, and serving stages.

## Three services, three jobs

The single most useful thing to internalize is the division of labor:

- **Event Hubs** — *streaming ingestion.* Millions of events per second, partitioned for parallel consumers, with retention so consumers can replay. Speaks the **Apache Kafka** protocol. Think firehose: clickstream, transactions, sensor telemetry, model-inference logs. Consumers read the stream in order per partition and check point their position.
- **Service Bus** — *reliable work queue / broker.* Lower throughput than Event Hubs but far richer delivery guarantees: at-least-once or exactly-once processing, ordered **sessions** (FIFO per key), transactions, dead-lettering, scheduled and deferred messages. Think task queue: "score this batch," "retrain this model" — work items that must not be lost and often must be processed once, in order.
- **Event Grid** — *event router / notification bus.* Not for data payloads; for *facts that something happened*. It routes small events ("a blob was created," "a Key Vault secret is near expiry," "a model was registered") from a source to one or many handlers (a Function, a webhook, a Service Bus queue) with retries and dead-lettering. Think reactive glue.

A blunt heuristic: **Event Hubs for data streams, Service Bus for reliable task queues, Event Grid for reactive notifications.** Many ML pipelines use all three — Event Hubs ingests the stream, Event Grid reacts when curated data lands, Service Bus queues the resulting scoring jobs.

## Event Hubs: streaming ingestion

Event Hubs partitions an event stream so multiple consumers process it in parallel; **partition count is fixed at creation** for Standard tier (you cannot easily change it later), so size it for your peak parallelism up front. A **consumer group** is an independent view of the stream, letting several downstream systems (a Fabric real-time pipeline, an Azure Function, a Spark job) each read the whole stream at their own pace without interfering.

```bash
az eventhubs namespace create -g rg-mlx-dev -n ehns-mlx --sku Standard --location eastus2
az eventhubs eventhub create   -g rg-mlx-dev --namespace-name ehns-mlx -n transactions \
  --partition-count 8 --retention-time-in-hours 72
az eventhubs eventhub consumer-group create -g rg-mlx-dev --namespace-name ehns-mlx \
  --eventhub-name transactions -n features-pipeline
```

Because it speaks **Kafka**, existing producers connect with a bootstrap-server change and no code rewrite — decisive if your org already standardized on Kafka. **Event Hubs Capture** automatically lands the raw stream into the lake (bronze) as Avro/Parquet with zero code, giving you a durable, replayable archive alongside real-time consumption. For very high scale, the **Premium/Dedicated** tiers give reserved capacity and single-tenant isolation. Consumers authenticate with a managed identity granted **Azure Event Hubs Data Receiver** (read) or **Data Sender** (write) — no connection strings.

## Service Bus: reliable messaging and work queues

Service Bus is the broker for work that must not be lost. Its features are exactly what a robust ML job queue needs:

- **Queues** (point-to-point) and **topics/subscriptions** (publish-subscribe with per-subscription filters).
- **Sessions** for FIFO ordering within a key (process all events for one customer in order).
- **Dead-letter queues (DLQ)** — messages that repeatedly fail land in a DLQ instead of blocking the queue or vanishing, so you can inspect and replay poison messages.
- **Duplicate detection**, **scheduled delivery**, **message deferral**, and **transactions** across entities.

```bash
az servicebus namespace create -g rg-mlx-dev -n sbns-mlx --sku Standard --location eastus2
az servicebus queue create -g rg-mlx-dev --namespace-name sbns-mlx -n scoring-jobs \
  --max-size 5120 --default-message-time-to-live P7D \
  --enable-dead-lettering-on-message-expiration true --lock-duration PT5M
# Topic + a filtered subscription (fan-out with routing)
az servicebus topic create -g rg-mlx-dev --namespace-name sbns-mlx -n model-events
az servicebus topic subscription create -g rg-mlx-dev --namespace-name sbns-mlx \
  --topic-name model-events -n retrain-trigger
```

The `--lock-duration` is the window a consumer has to process and acknowledge a message before it becomes visible again for redelivery — set it longer than your worst-case scoring time or you will get duplicate processing. Pair Service Bus with an **Azure Functions Service Bus trigger** (from the serverless module) for automatic, retryable, scale-out consumption, and grant the consumer **Azure Service Bus Data Receiver**.

## Event Grid: reactive event routing

Event Grid connects "something happened" to "do something," without either side knowing about the other. Azure services emit events into Event Grid as **system topics** (Blob Storage `BlobCreated`, Key Vault `SecretNearExpiry`, Azure ML `ModelRegistered`/`RunCompleted`); you can also publish your own to a **custom topic**. Subscriptions route matching events to handlers with server-side filtering, retries, and a dead-letter destination.

```bash
# React to new curated data: route Blob "created" events under datasets/ to a Function
STORAGE_ID=$(az storage account show -n stmlxdata -g rg-mlx-dev --query id -o tsv)
az eventgrid system-topic create -g rg-mlx-dev -n st-storage \
  --source "$STORAGE_ID" --topic-type Microsoft.Storage.StorageAccounts --location eastus2

az eventgrid system-topic event-subscription create -n on-new-dataset \
  -g rg-mlx-dev --system-topic-name st-storage \
  --endpoint-type azurefunction \
  --endpoint "/subscriptions/$SUB_ID/resourceGroups/rg-mlx-dev/providers/Microsoft.Web/sites/func-mlx-scoring/functions/on_blob" \
  --included-event-types Microsoft.Storage.BlobCreated \
  --subject-begins-with "/blobServices/default/containers/datasets/"
```

Event Grid can also deliver **into** Service Bus — the durable pattern where a reactive notification enqueues a reliable work item that a scaled-out consumer processes with retries and dead-lettering:

```bash
SBQ_ID=$(az servicebus queue show -g rg-mlx-dev --namespace-name sbns-mlx -n scoring-jobs --query id -o tsv)
az eventgrid system-topic event-subscription create -n new-data-to-queue \
  -g rg-mlx-dev --system-topic-name st-storage \
  --endpoint-type servicebusqueue --endpoint "$SBQ_ID" \
  --included-event-types Microsoft.Storage.BlobCreated
```

## Choosing among the three

- New telemetry/event **stream**, high volume, replay, Kafka compatibility → **Event Hubs**.
- **Work items** that must not be lost, need ordering/exactly-once/DLQ/transactions → **Service Bus**.
- **React** to a fact (blob created, model registered, secret expiring) and fan it out to handlers → **Event Grid**.

They compose: Event Hubs ingests the raw stream; Event Grid fires when a pipeline produces curated data or registers a model; Service Bus carries the resulting scoring or retraining jobs to reliable, retryable consumers.

## How messaging fits the whole solution

Messaging is the event backbone threading the pipeline together. **Event Hubs** is the streaming front door (data-services module) — transactions and telemetry land, with **Capture** archiving raw events to bronze and a Fabric real-time pipeline computing features. **Event Grid** makes the platform reactive: `BlobCreated` on the curated container triggers scoring, `ModelRegistered` triggers a deployment pipeline, Key Vault `SecretNearExpiry` triggers rotation (secrets module). **Service Bus** carries the durable work — a queue of scoring jobs consumed by a scaled-out Functions or AKS worker with dead-lettering for poison messages, and a `model-events` topic fanning out retrain/notify subscriptions. Every namespace authenticates with the shared **managed identity** (Data Sender/Receiver roles) and sits behind **private endpoints**, so the event layer is as secure as the rest of the platform.

## Key takeaways

- Three services, three jobs: **Event Hubs** = high-throughput streaming (Kafka-compatible, partitioned, replayable); **Service Bus** = reliable work queue/broker (ordering, sessions, DLQ, transactions); **Event Grid** = lightweight reactive event routing (notifications, not payloads).
- **Event Hubs** partition count is fixed at creation — size for peak parallelism; use **consumer groups** for independent readers and **Capture** to archive raw events to the lake.
- **Service Bus** gives the delivery guarantees ML job queues need — **dead-letter queues** for poison messages, **sessions** for FIFO, `--lock-duration` longer than worst-case processing; pair with a Functions trigger.
- **Event Grid** routes `BlobCreated`, `ModelRegistered`, `SecretNearExpiry` and other facts to handlers; a robust pattern is Event Grid → Service Bus → scaled-out consumer.
- Authenticate every namespace with **managed identity** (Data Sender/Receiver roles) behind **private endpoints** — no connection strings.

## CLI cheat-sheet

```bash
# --- Event Hubs (streaming) ---
az eventhubs namespace create -g rg-mlx-dev -n ehns-mlx --sku Standard --location eastus2
az eventhubs eventhub create  -g rg-mlx-dev --namespace-name ehns-mlx -n transactions \
  --partition-count 8 --retention-time-in-hours 72
az eventhubs eventhub consumer-group create -g rg-mlx-dev --namespace-name ehns-mlx \
  --eventhub-name transactions -n features-pipeline
az eventhubs eventhub list -g rg-mlx-dev --namespace-name ehns-mlx -o table
# grant a managed identity read/write (no connection strings)
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Azure Event Hubs Data Receiver" --scope "$(az eventhubs namespace show -g rg-mlx-dev -n ehns-mlx --query id -o tsv)"

# --- Service Bus (reliable queues/topics) ---
az servicebus namespace create -g rg-mlx-dev -n sbns-mlx --sku Standard --location eastus2
az servicebus queue create -g rg-mlx-dev --namespace-name sbns-mlx -n scoring-jobs \
  --max-size 5120 --default-message-time-to-live P7D \
  --enable-dead-lettering-on-message-expiration true --lock-duration PT5M
az servicebus topic create        -g rg-mlx-dev --namespace-name sbns-mlx -n model-events
az servicebus topic subscription create -g rg-mlx-dev --namespace-name sbns-mlx \
  --topic-name model-events -n retrain-trigger
az servicebus queue list -g rg-mlx-dev --namespace-name sbns-mlx -o table
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Azure Service Bus Data Receiver" --scope "$(az servicebus namespace show -g rg-mlx-dev -n sbns-mlx --query id -o tsv)"

# --- Event Grid (reactive routing) ---
az eventgrid system-topic create -g rg-mlx-dev -n st-storage \
  --source "$STORAGE_ID" --topic-type Microsoft.Storage.StorageAccounts --location eastus2
az eventgrid system-topic event-subscription create -n on-new-dataset -g rg-mlx-dev \
  --system-topic-name st-storage --endpoint-type azurefunction --endpoint "$FUNC_ID" \
  --included-event-types Microsoft.Storage.BlobCreated \
  --subject-begins-with "/blobServices/default/containers/datasets/"
# deliver events straight into a Service Bus queue (durable fan-in)
az eventgrid system-topic event-subscription create -n new-data-to-queue -g rg-mlx-dev \
  --system-topic-name st-storage --endpoint-type servicebusqueue --endpoint "$SBQ_ID"
az eventgrid system-topic event-subscription list -g rg-mlx-dev --system-topic-name st-storage -o table
# a custom topic for your own events
az eventgrid topic create -g rg-mlx-dev -n egt-mlx --location eastus2
```

## Try it

Stand up all three. Create an Event Hubs namespace with a `transactions` hub (8 partitions) and a `features-pipeline` consumer group. Create a Service Bus namespace with a `scoring-jobs` queue that has dead-lettering enabled and a 5-minute lock. Create an Event Grid system topic on your storage account and subscribe `BlobCreated` events under `datasets/` to deliver into the `scoring-jobs` queue. Now trace the flow: drop a blob under `datasets/`, watch Event Grid route the event into the Service Bus queue, and (bonus) wire a Functions Service Bus trigger that consumes it. Then write down, for three parts of your own ML system, which of the three services each event should use and why.
