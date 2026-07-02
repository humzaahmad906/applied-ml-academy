# 08 — Data Services

The data lake holds files; a complete ML system also needs services that give data *structure, transactions, low-latency lookups, streaming, and orchestration*. This section covers the data-plane services that surround the lake — operational databases (Azure SQL, Cosmos DB), streaming ingestion (Event Hubs), the analytics platform (Microsoft Fabric, with Synapse as its predecessor), and pipeline orchestration (Data Factory) — and how they compose into the data half of an end-to-end ML solution. The recurring theme: use the right store for each access pattern, and let one governed lake (OneLake / ADLS Gen2) be the shared truth underneath.

## Operational databases: Azure SQL and Cosmos DB

**Azure SQL Database** is the managed relational engine — full T-SQL, ACID transactions, strong consistency, and mature tooling. In an ML system it holds the structured, transactional records that models are built from and act on: customer records, orders, labels, and the *outputs* your predictions feed into an application. A model's batch scores often land back in Azure SQL so a downstream app can query them with a normal SQL join. It is the right home for relational data with a known schema and strong integrity needs.

You provision a **logical server** first (the security and connection boundary) and then databases under it. The choice that matters for ML economics is the **compute model**: the **serverless** tier auto-scales vCores with load and auto-pauses to storage-only billing when idle — ideal for a dev feature-store DB or an intermittently queried scores table — while the **Hyperscale** tier scales to 100 TB with fast backups and read replicas for large training-adjacent datasets. `az sql db list-editions` is the underused command that tells you exactly which SKUs and vCore ranges a region offers before you commit:

```bash
az sql server create -g rg-mlx-dev -n sql-mlx --location eastus2 \
  --enable-ad-only-auth --external-admin-name id-mlplatform \
  --external-admin-principal-type Application     # Entra-only auth, no SQL password
az sql server firewall-rule create -g rg-mlx-dev -s sql-mlx \
  -n allow-azure --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0   # "allow Azure services"

# Serverless General Purpose DB that auto-pauses when idle (great for dev / intermittent use)
az sql db create -g rg-mlx-dev -s sql-mlx -n scores \
  -e GeneralPurpose --compute-model Serverless -f Gen5 \
  --min-capacity 0.5 --capacity 4 --auto-pause-delay 60

# Hyperscale for a large dataset with read replicas
az sql db create -g rg-mlx-dev -s sql-mlx -n warehouse \
  -e Hyperscale --compute-model Provisioned -f Gen5 -c 8 --read-replicas 1

az sql db list -g rg-mlx-dev -s sql-mlx -o table
az sql db show -g rg-mlx-dev -s sql-mlx -n scores --query "{tier:currentServiceObjectiveName, model:currentSku.name}"
az sql db list-editions -l eastus2 --edition Hyperscale -o table   # what SKUs exist in-region
```

Prefer **Entra-only authentication** (`--enable-ad-only-auth`) so the database has no SQL admin password to leak; the platform managed identity then connects with a token, consistent with every other service.

**Azure Cosmos DB** is the globally distributed, multi-model NoSQL database, built for single-digit-millisecond reads and writes at any scale, with tunable consistency and automatic partitioning. Its ML fit is **low-latency lookups in the request path** — the classic case being an **online feature store** or a serving cache: when an endpoint needs a user's precomputed features to score a request in a few milliseconds, Cosmos DB serves them far faster than a relational query. It is also a natural sink for high-volume telemetry and for storing embeddings/metadata in some RAG designs. Choose Cosmos when you need horizontal scale and predictable low latency more than relational joins and transactions.

```python
# Read precomputed online features for a user at request time (serving path)
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

client = CosmosClient("https://cosmos-mlx.documents.azure.com:443/",
                      credential=DefaultAzureCredential())
container = client.get_database_client("features").get_container_client("user")
features = container.read_item(item="user-42", partition_key="user-42")
```

On the provisioning side, Cosmos DB gives you a throughput decision that mirrors SQL's compute-model choice. A **serverless** account (`--capabilities EnableServerless`) bills per request-unit consumed with no floor — the right default for a bursty online feature store or a dev cache where you do not want to pay for idle RU/s. A **provisioned-throughput** account (autoscale or manual RU/s) suits steady, predictable serving load. Note that you cannot pass `--throughput` when creating a container in a serverless account — it will error — because serverless has no reserved capacity to set:

```bash
# Serverless account (pay per RU, no idle floor) with a database and a partitioned container
az cosmosdb create -g rg-mlx-dev -n cosmos-mlx --capabilities EnableServerless \
  --default-consistency-level Session --locations regionName=eastus2 failoverPriority=0
az cosmosdb sql database create  -g rg-mlx-dev -a cosmos-mlx -n features
az cosmosdb sql container create -g rg-mlx-dev -a cosmos-mlx -d features -n user \
  --partition-key-path /userId          # no --throughput on a serverless account

# A provisioned account instead would use autoscale RU/s:
# az cosmosdb sql container create ... --max-throughput 4000
az cosmosdb sql database list -g rg-mlx-dev -a cosmos-mlx -o table
```

The credential detail most people miss: Cosmos DB has a **separate data-plane RBAC** system, distinct from Azure control-plane roles. Assigning a control-plane role like *Contributor* lets an identity manage the account but does **not** grant permission to read or write documents. To let the serving code fetch features with `DefaultAzureCredential` (no keys), you assign a Cosmos DB data role — the built-in *Cosmos DB Built-in Data Reader* / *Data Contributor* definitions — to the managed identity:

```bash
# Grant the managed identity data-plane read/write (built-in Data Contributor = 00000000-...-0002)
az cosmosdb sql role assignment create -g rg-mlx-dev -a cosmos-mlx \
  --role-definition-id 00000000-0000-0000-0000-000000000002 \
  --principal-id $(az identity show -g rg-mlx-dev -n id-mlplatform --query principalId -o tsv) \
  --scope "/"
az cosmosdb sql role definition list -g rg-mlx-dev -a cosmos-mlx -o table   # built-in + custom roles
```

A useful rule: **Azure SQL for transactional, relational, strongly consistent data; Cosmos DB for globally distributed, low-latency, high-scale key lookups.** Many systems use both — SQL as the system of record, Cosmos as the hot cache in the serving path.

## Streaming ingestion: Event Hubs

Real-time ML starts with a stream. **Azure Event Hubs** is the managed, high-throughput event-ingestion service — millions of events per second, partitioned for parallelism, with retention so consumers can replay. It speaks the **Apache Kafka** protocol, so existing Kafka producers and consumers work against it with a connection-string change, which matters if your organization already standardized on Kafka. Event Hubs is the front door for streaming data into the lake and into real-time feature/scoring pipelines: producers push clickstreams, transactions, or sensor data; consumers (Functions with an Event Hubs trigger, Fabric real-time pipelines, or a Spark job) read and process them. It is the streaming counterpart to the batch ingestion that Data Factory handles.

```bash
az eventhubs namespace create -g rg-mlx-dev -n ehns-mlx --sku Standard
az eventhubs eventhub create -g rg-mlx-dev --namespace-name ehns-mlx \
  --name transactions --partition-count 8 --message-retention 3
```

Partition count is the knob that sets your **parallelism ceiling** — a consumer group can have at most one active reader per partition, so an 8-partition hub tops out at 8 parallel consumers. It is fixed at creation on Standard, so size it for peak fan-out up front. Each independent consumer needs its own **consumer group** to track its own offset without stepping on others (your real-time feature job and your archival job each get one). Access is granted through **authorization rules** (shared-access policies), but the cleaner pattern is a private endpoint plus managed-identity RBAC rather than passing a connection string:

```bash
# A dedicated consumer group per independent reader (feature pipeline vs. archiver)
az eventhubs eventhub consumer-group create -g rg-mlx-dev --namespace-name ehns-mlx \
  --eventhub-name transactions --name feature-pipeline

# A send-only authorization rule + its connection string (prefer MI RBAC where you can)
az eventhubs eventhub authorization-rule create -g rg-mlx-dev --namespace-name ehns-mlx \
  --eventhub-name transactions --name producer --rights Send
az eventhubs eventhub authorization-rule keys list -g rg-mlx-dev --namespace-name ehns-mlx \
  --eventhub-name transactions --name producer --query primaryConnectionString -o tsv

az eventhubs namespace list -g rg-mlx-dev -o table
az eventhubs eventhub list  -g rg-mlx-dev --namespace-name ehns-mlx -o table
```

Two features matter for the ML data spine. First, **Kafka compatibility**: the namespace exposes a Kafka endpoint on Standard and above, so existing Kafka producers/consumers connect by changing only the broker string and credentials — no code rewrite. Second, **Capture**, which continuously writes the raw stream to Blob/ADLS Gen2 in Avro, giving you the bronze landing of streaming data with zero consumer code — the streaming counterpart to a Data Factory batch load:

```bash
# Enable Capture: raw events land in ADLS Gen2 automatically (bronze landing for the stream)
az eventhubs eventhub update -g rg-mlx-dev --namespace-name ehns-mlx --name transactions \
  --enable-capture true --capture-interval 300 --storage-account stmlxdata \
  --blob-container bronze --archive-name-format \
  '{Namespace}/{EventHub}/{PartitionId}/{Year}/{Month}/{Day}/{Hour}/{Minute}/{Second}'
```

## The analytics platform: Microsoft Fabric (and Synapse)

For years, **Azure Synapse Analytics** was the unified analytics service — Spark pools, dedicated and serverless SQL pools, and pipelines over a data lake. It still runs, and plenty of production workloads use it, but Microsoft's strategic direction has clearly shifted: **Microsoft Fabric** is now the flagship, SaaS-first analytics platform, and new analytics capabilities are landing there first. Treat Synapse as the mature incumbent you may inherit, and Fabric as where greenfield analytics goes.

**Microsoft Fabric** unifies data engineering, data warehousing, data science, real-time analytics, and BI into one product, all sitting on **OneLake** — a single, tenant-wide data lake in open **Delta Parquet** format. The pieces that matter for ML:

- **Lakehouse** — Spark-based data engineering and a queryable table layer over OneLake; where feature pipelines run.
- **Data Warehouse** — full T-SQL warehousing for curated, modeled data.
- **Data Science** — notebooks, experiment tracking, and model development integrated with the lakehouse (it interoperates with Azure Machine Learning for heavier training).
- **Real-Time Intelligence** — ingest and analyze streams (fed by Event Hubs/Kafka) with low latency.
- **Mirroring** — continuously replicate operational databases (Azure SQL, Cosmos DB, PostgreSQL) into OneLake in Delta format with near-zero ETL. This is the current, recommended way to make operational data analytics-ready — it supersedes the older Synapse Link approach, which is no longer the path for new projects.

Because everything reads the same OneLake tables in Delta format, a feature table your pipeline writes in the lakehouse is immediately available to a training job, a warehouse query, and a Power BI report without copies. That shared, open format is the reason Fabric composes so well into an ML solution.

## Orchestration: Data Factory

**Azure Data Factory** (and its embedded equivalent inside Fabric/Synapse pipelines) is the managed data-integration and orchestration service — the tool that *moves* and *transforms* data on a schedule or trigger. It has 90+ connectors, so it pulls from on-prem databases, SaaS APIs, and other clouds into the lake, and it orchestrates multi-step data workflows as pipelines with dependencies, retries, and monitoring. In an ML system, Data Factory owns **batch ingestion** (nightly loads into bronze) and **scheduled data preparation** (bronze → silver → gold), handing off cleaned, curated data for training. It is the batch counterpart to Event Hubs' streaming ingestion.

The `az datafactory` extension manages the factory and its objects, though the pipeline/linked-service *definitions* themselves are JSON you supply with `--properties @file.json` (authoring is usually done in the visual Studio, then exported to source control). The commands you run in CI/CD and operations are create, run-on-demand, and the trigger lifecycle:

```bash
az datafactory create -g rg-mlx-dev -n adf-mlx --location eastus2

# Define a pipeline / trigger from JSON, kick a run, and read its status
az datafactory pipeline create   -g rg-mlx-dev --factory-name adf-mlx -n pl_bronze_to_silver --pipeline @pipeline.json
az datafactory pipeline create-run -g rg-mlx-dev --factory-name adf-mlx -n pl_bronze_to_silver \
  --parameters '{"runDate":"2026-07-02"}'                       # returns a runId to track
az datafactory pipeline-run show -g rg-mlx-dev --factory-name adf-mlx --run-id <runId>

# Schedule trigger: create, then start it (triggers are created stopped)
az datafactory trigger create -g rg-mlx-dev --factory-name adf-mlx -n nightly --properties @trigger.json
az datafactory trigger start  -g rg-mlx-dev --factory-name adf-mlx -n nightly
az datafactory trigger stop   -g rg-mlx-dev --factory-name adf-mlx -n nightly
az datafactory trigger list   -g rg-mlx-dev --factory-name adf-mlx -o table
```

A trigger is created in a stopped state, so the `trigger start` after `trigger create` is the step that actually schedules it — a common "why isn't my pipeline running" gotcha. **Microsoft Fabric**, by contrast, has no first-class `az` command group; you provision and manage Fabric capacity and items through the portal, the Fabric REST API, or the `fabric-cli`/PowerShell — plan your automation accordingly.

There is a natural division of orchestration labor: **Data Factory** for data movement and ELT/ETL DAGs; **Azure Machine Learning pipelines** for the training/eval/registration DAG; **Durable Functions** for lightweight event-driven reactions; and, if a team already standardized on it, **Apache Airflow** (available as a managed offering) for teams that want a code-first DAG scheduler across both data and ML steps. Pick per team maturity, but keep the boundary clear so you do not orchestrate the same step from two systems.

## Choosing a store: a quick map

- Structured, transactional, relational, strong consistency → **Azure SQL**.
- Global, low-latency key lookups / online feature serving / high-scale telemetry → **Cosmos DB**.
- High-throughput streaming ingestion (Kafka-compatible) → **Event Hubs**.
- Files, raw and curated datasets, model artifacts → **Blob / ADLS Gen2** (previous topic).
- Unified analytics, feature engineering, warehousing, BI on open Delta → **Microsoft Fabric** (Synapse if inherited).
- Scheduled data movement and ELT/ETL DAGs → **Data Factory**.

## How data services fit the whole solution

The data half of the reference architecture wires these together like this. **Event Hubs** ingests streams and **Data Factory** ingests batch, both landing raw data in the **bronze** layer of the lake. A **Fabric lakehouse** transforms bronze into curated **silver** and feature-rich **gold** tables in Delta on OneLake; operational data from **Azure SQL** and **Cosmos DB** is brought in via **mirroring** so it is analytics-ready without brittle ETL. Feature tables live in gold; the low-latency subset needed at serving time is materialized into **Cosmos DB** as the online feature store. Training jobs read gold datasets; batch predictions land back in **Azure SQL** for applications to consume. **Data Factory** (or Airflow, or Fabric pipelines) schedules the data DAG, while Azure ML pipelines own the training DAG. Every store authenticates with managed identity and, where sensitive, sits behind private endpoints — so the data platform is fast, governed, and secure end to end.

## Key takeaways

- Match the store to the access pattern: **Azure SQL** (relational/transactional), **Cosmos DB** (low-latency global lookups / online features), **Blob/ADLS** (files/datasets), **Event Hubs** (streaming, Kafka-compatible).
- **Microsoft Fabric** is the current flagship analytics platform on **OneLake / open Delta Parquet**; **Synapse** still runs but new work goes to Fabric. Use **mirroring** (not the older Synapse Link) to make operational data analytics-ready.
- **Data Factory** owns scheduled batch ingestion and ELT/ETL DAGs; keep a clear boundary with Azure ML pipelines (training DAG) and Durable Functions/Airflow (event-driven / code-first orchestration).
- An **online feature store** pattern pairs a batch feature table in the lake/warehouse with a **Cosmos DB** materialization for millisecond serving-time lookups.
- One governed lake in an open format is the shared truth that lets every data service — and the ML layer — compose without copies.

## CLI cheat-sheet

```bash
# --- Azure SQL: server + databases ---
az sql server create -g rg-mlx-dev -n sql-mlx -l eastus2 --enable-ad-only-auth \
  --external-admin-name id-mlplatform --external-admin-principal-type Application
az sql server firewall-rule create -g rg-mlx-dev -s sql-mlx -n allow-azure --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0
az sql db create -g rg-mlx-dev -s sql-mlx -n scores -e GeneralPurpose --compute-model Serverless -f Gen5 --min-capacity 0.5 --capacity 4 --auto-pause-delay 60
az sql db create -g rg-mlx-dev -s sql-mlx -n warehouse -e Hyperscale --compute-model Provisioned -f Gen5 -c 8 --read-replicas 1
az sql db list          -g rg-mlx-dev -s sql-mlx -o table
az sql db show          -g rg-mlx-dev -s sql-mlx -n scores
az sql db list-editions -l eastus2 --edition Hyperscale -o table

# --- Cosmos DB: account, database, container, data-plane RBAC ---
az cosmosdb create -g rg-mlx-dev -n cosmos-mlx --capabilities EnableServerless \
  --default-consistency-level Session --locations regionName=eastus2 failoverPriority=0
az cosmosdb sql database create  -g rg-mlx-dev -a cosmos-mlx -n features
az cosmosdb sql container create -g rg-mlx-dev -a cosmos-mlx -d features -n user --partition-key-path /userId
# (provisioned account instead: --max-throughput 4000 on the container)
az cosmosdb sql database list -g rg-mlx-dev -a cosmos-mlx -o table
az cosmosdb sql role definition list -g rg-mlx-dev -a cosmos-mlx -o table
az cosmosdb sql role assignment create -g rg-mlx-dev -a cosmos-mlx \
  --role-definition-id 00000000-0000-0000-0000-000000000002 \
  --principal-id <mi-principal-id> --scope "/"

# --- Event Hubs: namespace, hub, consumer group, auth, capture ---
az eventhubs namespace create -g rg-mlx-dev -n ehns-mlx --sku Standard
az eventhubs eventhub  create -g rg-mlx-dev --namespace-name ehns-mlx -n transactions --partition-count 8 --message-retention 3
az eventhubs eventhub consumer-group create -g rg-mlx-dev --namespace-name ehns-mlx --eventhub-name transactions --name feature-pipeline
az eventhubs eventhub authorization-rule create -g rg-mlx-dev --namespace-name ehns-mlx --eventhub-name transactions --name producer --rights Send
az eventhubs eventhub authorization-rule keys list -g rg-mlx-dev --namespace-name ehns-mlx --eventhub-name transactions --name producer
az eventhubs eventhub update -g rg-mlx-dev --namespace-name ehns-mlx -n transactions \
  --enable-capture true --capture-interval 300 --storage-account stmlxdata --blob-container bronze
az eventhubs eventhub list -g rg-mlx-dev --namespace-name ehns-mlx -o table

# --- Data Factory: factory, pipelines, triggers ---
az datafactory create -g rg-mlx-dev -n adf-mlx -l eastus2
az datafactory pipeline create     -g rg-mlx-dev --factory-name adf-mlx -n pl_bronze_to_silver --pipeline @pipeline.json
az datafactory pipeline create-run -g rg-mlx-dev --factory-name adf-mlx -n pl_bronze_to_silver
az datafactory pipeline-run show   -g rg-mlx-dev --factory-name adf-mlx --run-id <runId>
az datafactory trigger create -g rg-mlx-dev --factory-name adf-mlx -n nightly --properties @trigger.json
az datafactory trigger start  -g rg-mlx-dev --factory-name adf-mlx -n nightly    # triggers start stopped!
az datafactory trigger list   -g rg-mlx-dev --factory-name adf-mlx -o table
# NOTE: Microsoft Fabric has no `az` group — use the portal, Fabric REST API, or fabric-cli.
```

## Try it

Stand up the streaming front door and an online-lookup store: create an Event Hubs namespace with a `transactions` hub (8 partitions), and a Cosmos DB account with a `features/user` container. Write a tiny producer that pushes a few JSON events to the hub, and a reader that fetches a feature document from Cosmos by user id with `DefaultAzureCredential`. Then sketch, on paper, how a streamed transaction would flow from Event Hubs → bronze → Fabric feature pipeline → gold → Cosmos online store → a scoring endpoint. That flow is the data spine of the system you will assemble later.
