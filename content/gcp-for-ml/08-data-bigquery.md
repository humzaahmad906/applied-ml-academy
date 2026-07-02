# 08 — Data: BigQuery and Friends

Machine learning is only as good as the data feeding it, and on Google Cloud the center of gravity for structured data is **BigQuery** — a serverless, petabyte-scale data warehouse where your training data is stored, cleaned, joined, aggregated, and often where features (and even models) are produced. Around it sit a family of data services: **Cloud SQL** for transactional workloads, **Dataflow** for large-scale streaming and batch processing, and **Pub/Sub** for event ingestion. This module maps that data plane and shows how it feeds your ML pipelines.

## BigQuery: the serverless warehouse

BigQuery separates storage from compute. You load data into **tables** grouped in **datasets** (a dataset is regional and the unit of access control), and you run **SQL** queries that Google's engine parallelizes across thousands of workers with no cluster for you to size. For ML this means you can join a billion-row event log against a dimension table and materialize a training set in seconds, entirely in SQL.

```bash
# Create a dataset (co-located with your compute region)
bq --location=us-central1 mk --dataset myco-fraud-dev:fraud

# Load a Parquet file from Cloud Storage into a table
bq load --source_format=PARQUET \
  fraud.transactions \
  gs://myco-fraud-data/datasets/v1/transactions.parquet

# Query it
bq query --use_legacy_sql=false \
  'SELECT label, COUNT(*) FROM `myco-fraud-dev.fraud.transactions` GROUP BY label'
```

A critical, irreversible choice happens at `bq mk` time: the dataset's **location** (a region like `us-central1` or a multi-region like `US`). Location is **immutable** — you cannot move a dataset later, only recreate and copy — and a query can only join tables that live in the *same* location. Co-locate datasets with the compute region and the Cloud Storage buckets they load from, or you pay for cross-region copies and lose the ability to join. Standard `bq mk` options worth knowing: `--default_table_expiration` and `--default_partition_expiration` (auto-delete stale data to control storage cost), `--description`, and `--label` (attach the `team`/`env`/`component`/`model` labels that flow into billing export).

### Loading data: formats, autodetect, and replace

`bq load` ingests CSV, JSON (newline-delimited), Avro, Parquet, and ORC from Cloud Storage or a local file, and it is the workhorse for getting data in. Avro and Parquet carry their own schema, so no `--schema` is needed; for CSV/JSON either pass an explicit schema or let `--autodetect` infer one (fine for exploration, risky for production where you want a pinned schema). `--replace` truncates the table before loading (idempotent full refresh); the default appends. `--skip_leading_rows=1` drops a CSV header; `--time_partitioning_field` and `--clustering_fields` set physical layout at load time.

```bash
# Autodetected CSV load, replacing the table each run
bq load --autodetect --replace --source_format=CSV --skip_leading_rows=1 \
  fraud.transactions gs://myco-fraud-data/datasets/v1/transactions.csv

# Parquet load into a partitioned + clustered table (schema inferred from Parquet)
bq load --source_format=PARQUET \
  --time_partitioning_field=event_date --time_partitioning_type=DAY \
  --clustering_fields=merchant_category,account_id \
  fraud.transactions gs://myco-fraud-data/datasets/v1/transactions.parquet
```

**Pricing model.** Two axes: **storage** (cheap, per-GB, with active vs long-term rates — a partition untouched for 90 days automatically drops to the ~50%-cheaper long-term rate) and **compute**. Compute is either **on-demand** (you pay per **byte scanned** by each query — so `SELECT *` on a wide table is expensive and column pruning / partitioning is how you save money) or **capacity-based** (BigQuery editions with reserved **slots** for predictable, high-volume workloads). The single most important cost fact: on-demand billing is on bytes *scanned*, not bytes *returned* — a `LIMIT 10` on an unpartitioned billion-row table still scans (and bills for) the whole table. Two habits neutralize this. First, **never `SELECT *`** in a repeated query; name the columns you need, because BigQuery is columnar and only reads the columns you reference. Second, always **estimate before running** with `--dry_run` and cap blast radius with `--maximum_bytes_billed`:

```bash
# Show bytes that WOULD be scanned, without running (and without cost)
bq query --use_legacy_sql=false --dry_run \
  'SELECT account_id, amount FROM `myco-fraud-dev.fraud.transactions`
   WHERE event_date = "2026-06-01"'

# Hard-cap a query: it fails (free) if it would scan more than 10 GB
bq query --use_legacy_sql=false --maximum_bytes_billed=10000000000 \
  'SELECT ... '
```

For ML teams, partitioning tables by date and clustering by common filter columns dramatically cuts bytes scanned and cost — covered in depth below. For high-volume, predictable workloads, reserve capacity with `bq mk --reservation` (BigQuery editions/slots), and for interactive dashboards layered on BigQuery, **BI Engine** caches hot data in memory for sub-second, near-free repeat reads.

**BigQuery Studio** is the unified workspace (SQL, notebooks, and Python) in the Console. And **BigFrames** (`bigframes`) gives you a pandas- and scikit-learn-like Python API whose operations execute *inside* BigQuery — so you write familiar DataFrame code but the computation scales to the full warehouse without pulling data to a single machine. This is a key tool for ML feature engineering at scale.

```python
import bigframes.pandas as bpd

bpd.options.bigquery.project = "myco-fraud-dev"
df = bpd.read_gbq("myco-fraud-dev.fraud.transactions")
# Runs server-side in BigQuery, not in local pandas memory
features = df.groupby("account_id").agg({"amount": ["mean", "std", "count"]})
```

## Partitioning and clustering: the cost levers that matter

Partitioning and clustering are the two physical-layout decisions that determine whether a query scans a gigabyte or a terabyte. **Partitioning** splits a table into segments by a column — most commonly a date/timestamp (`DAY`, `HOUR`, `MONTH`, or `YEAR` granularity), but also an integer range or the special ingestion-time pseudo-column `_PARTITIONTIME`. When a query filters on the partition column, BigQuery reads only the matching partitions — **partition pruning** — and you pay for those bytes alone. The gotcha: pruning only works when the filter is on the raw partition column with a constant (`WHERE event_date >= '2026-06-01'`); wrap it in a function or compare it to a subquery and the optimizer often cannot prune, silently scanning everything.

**Clustering** sorts data within each partition by up to four columns, so filters and aggregations on those columns touch fewer blocks. Clustering is the right tool for high-cardinality filter columns (`account_id`, `merchant_id`) where partitioning would create too many tiny partitions (BigQuery caps a table at ~10,000 partitions). Partition by time, cluster by the columns you filter and group by most — that combination is the default for any large fact table feeding ML.

```bash
# Create an empty partitioned + clustered table with a partition expiration
bq mk --table \
  --time_partitioning_field=event_date \
  --time_partitioning_type=DAY \
  --time_partitioning_expiration=7776000 \
  --clustering_fields=merchant_category,account_id \
  --schema=event_date:DATE,account_id:STRING,amount:FLOAT,merchant_category:STRING,label:INT64 \
  myco-fraud-dev:fraud.transactions

# Require a partition filter so nobody can accidentally full-scan the table
bq update --require_partition_filter myco-fraud-dev:fraud.transactions
```

`--require_partition_filter` is a cheap guardrail: it rejects any query that omits a partition filter, turning "someone ran a full scan and burned the month's budget" into a fast error.

## Table operations: the everyday `bq` toolkit

Beyond load and query, a handful of `bq` subcommands cover almost all table management. `bq ls` lists datasets and tables; `bq show` prints schema, row count, size, partitioning, and clustering; `bq head` peeks at rows without a billed query; `bq extract` exports a table to Cloud Storage; `bq cp` copies (or snapshots) tables; `bq rm` deletes.

```bash
bq ls myco-fraud-dev:fraud                       # tables in a dataset
bq show --schema myco-fraud-dev:fraud.transactions
bq head -n 20 myco-fraud-dev:fraud.transactions  # free peek, no query cost
bq extract --destination_format=PARQUET \
  myco-fraud-dev:fraud.transactions gs://myco-fraud-data/export/txn-*.parquet
bq cp myco-fraud-dev:fraud.transactions myco-fraud-dev:fraud.transactions_bak
bq rm -f -t myco-fraud-dev:fraud.transactions_bak
```

Writing query results into a managed table (rather than eyeballing them) is the normal way to materialize a feature or training table. Pass `--destination_table` with `--replace` (overwrite) or `--append_table`:

```bash
bq query --use_legacy_sql=false --replace \
  --destination_table=myco-fraud-dev:fraud.training_set \
  'SELECT account_id, amount, merchant_category, label
   FROM `myco-fraud-dev.fraud.transactions`
   WHERE event_date BETWEEN "2026-01-01" AND "2026-06-30"'
```

**Parameterized queries** keep values out of the SQL string (safer, cacheable, scriptable). Pass `--parameter=name:TYPE:value` and reference `@name`:

```bash
bq query --use_legacy_sql=false \
  --parameter='cutoff:DATE:2026-06-01' \
  'SELECT COUNT(*) FROM `myco-fraud-dev.fraud.transactions` WHERE event_date >= @cutoff'
```

### External tables, views, and snapshots

Not all data needs to live inside BigQuery storage. An **external table** leaves the data as files in Cloud Storage (Parquet, CSV, ORC, or open table formats like Iceberg) and queries it in place — handy for a data lake you do not want to duplicate, at the cost of no clustering and slower scans. Define one with `bq mk --external_table_definition`:

```bash
bq mk --table \
  --external_table_definition=PARQUET=gs://myco-fraud-data/lake/txn/*.parquet \
  myco-fraud-dev:fraud.txn_external
```

**Authorized views** are the governance workhorse: you grant a team access to a *view* (which selects only the columns/rows they may see) without granting access to the underlying table, so PII stays locked down while analysts still self-serve. **Table snapshots** (`bq cp --snapshot`) capture a cheap, point-in-time, read-only copy of a table — you pay only for bytes that later diverge — which is ideal for freezing the exact rows a model trained on so an eval or audit is reproducible.

```bash
bq cp --snapshot --no_clobber \
  myco-fraud-dev:fraud.training_set \
  myco-fraud-dev:fraud.training_set_snap_2026_07_01
```

### Knowing what queries cost: INFORMATION_SCHEMA and scheduled queries

BigQuery records every job in `INFORMATION_SCHEMA.JOBS`, so you can attribute bytes scanned (and therefore cost) by user, query, or day — the query-level complement to billing export. This is how you find the `SELECT *` that is quietly costing thousands:

```sql
SELECT user_email,
       ROUND(SUM(total_bytes_billed)/POW(2,40), 2) AS tib_billed,
       COUNT(*) AS jobs
FROM `region-us-central1`.INFORMATION_SCHEMA.JOBS
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND job_type = 'QUERY'
GROUP BY user_email ORDER BY tib_billed DESC;
```

**Scheduled queries** run a SQL statement on a cadence (hourly nightly refresh of a feature table, for example) with no external orchestrator — `bq query --schedule` or the Console — which is the simplest way to keep a materialized training table fresh.

## BigQuery ML: models in SQL

**BigQuery ML** lets you train and run models with SQL, keeping the data in place — no export, no separate training infrastructure for a large class of problems (note it runs under the Enterprise editions, not the Standard tier). `CREATE MODEL` supports logistic and linear regression, boosted trees (XGBoost), random forests, DNNs, k-means clustering, matrix factorization, PCA/autoencoders, and ARIMA_PLUS time-series forecasting, and more. You can also **import** TensorFlow/TFLite/ONNX/XGBoost models to run inference in SQL, and — importantly for modern ML — define **remote models** backed by a Vertex AI endpoint or a Gemini model (via a Cloud Resource connection), invoked from SQL. The current generative functions are `AI.GENERATE_TEXT` (table-valued, recommended over the older `ML.GENERATE_TEXT`) and the scalar `AI.GENERATE`, alongside `AI.EMBED` and `AI.GENERATE_TABLE`:

```sql
-- Train a classifier entirely in the warehouse
CREATE OR REPLACE MODEL `fraud.txn_classifier`
OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', input_label_cols=['label']) AS
SELECT amount, merchant_category, hour_of_day, label
FROM `myco-fraud-dev.fraud.transactions`;

-- Predict
SELECT * FROM ML.PREDICT(MODEL `fraud.txn_classifier`,
  (SELECT * FROM `fraud.new_transactions`));

-- Evaluate on a holdout slice (returns precision, recall, AUC, log loss, ...)
SELECT * FROM ML.EVALUATE(MODEL `fraud.txn_classifier`,
  (SELECT * FROM `fraud.holdout`));

-- Explain a prediction: per-row top feature attributions (needs ENABLE_GLOBAL_EXPLAIN)
SELECT * FROM ML.EXPLAIN_PREDICT(MODEL `fraud.txn_classifier`,
  (SELECT * FROM `fraud.new_transactions`), STRUCT(3 AS top_k_features));

-- Define a remote model backed by a Gemini model on Vertex AI
CREATE OR REPLACE MODEL `fraud.gemini_flash`
REMOTE WITH CONNECTION `myco-fraud-dev.us-central1.vertex_conn`
OPTIONS (ENDPOINT = 'gemini-2.5-flash');

-- Call it directly from SQL over warehouse rows
SELECT *
FROM AI.GENERATE_TEXT(
  MODEL `fraud.gemini_flash`,
  (SELECT prompt FROM `fraud.review_prompts`),
  STRUCT(1024 AS max_output_tokens, 0.2 AS temperature));
```

The model catalog is broader than the classic tabular set. Beyond `BOOSTED_TREE_CLASSIFIER`/`_REGRESSOR`, `LOGISTIC_REG`, `LINEAR_REG`, `RANDOM_FOREST_CLASSIFIER`, `DNN_CLASSIFIER`/`_REGRESSOR`, and `KMEANS`, BigQuery ML covers `ARIMA_PLUS` (time-series forecasting with holidays and anomaly detection), `MATRIX_FACTORIZATION` (recommenders), and `PCA`/`AUTOENCODER` (dimensionality reduction and anomaly detection). Every model type shares the same lifecycle: `CREATE MODEL` → `ML.EVALUATE` for metrics → `ML.PREDICT` for scoring → `ML.EXPLAIN_PREDICT` / `ML.GLOBAL_EXPLAIN` for attributions. For embeddings, `ML.GENERATE_EMBEDDING` produces vectors — either via a PCA/autoencoder model or, through a remote model, via a Vertex text/image embedding endpoint — which you can then index with `VECTOR_SEARCH` for retrieval and similarity features, all inside the warehouse.

Remote and generative models require a **connection** — a Cloud Resource connection whose service account is granted the Vertex AI User role — created once with `bq mk --connection`, after which the connection's service account is what actually calls Vertex:

```bash
# Create a connection, then read its service-account id to grant it Vertex access
bq mk --connection --location=us-central1 --connection_type=CLOUD_RESOURCE vertex_conn
bq show --connection myco-fraud-dev.us-central1.vertex_conn
# grant that SA roles/aiplatform.user, then CREATE MODEL ... REMOTE WITH CONNECTION ...
```

BigQuery ML is the fastest route from data to a working baseline model, and its Gemini integration lets analysts run generative AI over warehouse data with no pipeline. Its tight coupling with Vertex AI (export a BQML model to the model registry, or call Vertex endpoints from SQL) makes it a first-class member of the ML stack, not a toy.

## Cloud SQL and external data

**Cloud SQL** is managed PostgreSQL, MySQL, or SQL Server — the transactional (OLTP) database behind your application. The rule of thumb: **Cloud SQL for operational reads/writes** (a live app updating rows), **BigQuery for analytics and ML** (scanning huge volumes). Data typically flows *from* Cloud SQL (or its events) *into* BigQuery for analysis. BigQuery can also query data in place via **external tables** (over Parquet/CSV/Iceberg in Cloud Storage) and **federated queries** (against Cloud SQL/Spanner), so you can join warehouse and operational data without a full copy.

## Dataflow: large-scale processing

**Dataflow** is Google Cloud's managed **Apache Beam** runner for both **batch** and **streaming** data processing. It autoscales workers, handles fault tolerance, and is the tool for transformations too heavy or too stateful for SQL or a function — parsing and validating raw files, deduplicating, windowing and aggregating a stream, and computing rolling features. For ML it plays two roles: large-scale **preprocessing** of training data, and **streaming feature computation** that keeps an online feature store fresh. Google provides ready-made **Dataflow templates** (for example, Pub/Sub → BigQuery) so common pipelines need no code. (Module 16 covers Pub/Sub and Dataflow streaming pipelines in depth; this module keeps them at the level needed to understand how data reaches BigQuery.)

## Pub/Sub: event ingestion

**Pub/Sub** is the globally-scalable, at-least-once messaging backbone. Producers publish messages to a **topic**; consumers read via **subscriptions**. It decouples the systems that generate events (an app emitting transactions, a device emitting telemetry) from the systems that process them. Two ingestion patterns dominate:

- **Pub/Sub → BigQuery subscription** — a special subscription writes messages *directly* into a BigQuery table with no intermediate compute. Zero-code, ideal when events arrive already in the right shape.
- **Pub/Sub → Dataflow → BigQuery** — when you need windowing, aggregation, enrichment, or feature engineering before landing the data. This is the standard streaming-feature pipeline.

```bash
# Create a topic and a direct-to-BigQuery subscription
gcloud pubsub topics create transactions
gcloud pubsub subscriptions create transactions-to-bq \
  --topic=transactions \
  --bigquery-table=myco-fraud-dev:fraud.raw_events \
  --use-topic-schema
```

Because delivery is at-least-once, downstream consumers must be **idempotent** — the same lesson as event-driven functions. One cost gotcha specific to BigQuery: the legacy **streaming inserts** API (`tabledata.insertAll`) bills per row inserted and is materially more expensive at volume than batch loads or the newer Storage Write API — prefer `bq load`, a direct Pub/Sub→BigQuery subscription, or the Storage Write API over row-by-row streaming when you can tolerate small latency.

## How this fits the whole solution

This data plane is the front half of the end-to-end system. **Pub/Sub** ingests real-time events; **Dataflow** transforms streams and batches (computing features); **Cloud Storage** holds raw and bulk files; and **BigQuery** is the warehouse where training sets are assembled and, via BigQuery ML, baseline models and Gemini calls run in SQL. Everything downstream — Vertex AI training reading a BigQuery-derived dataset, a feature store syncing from BigQuery tables, batch prediction writing results back to BigQuery — depends on this layer being clean, partitioned, and well-governed. Master the data plane and the modeling work becomes far easier.

## Key takeaways

- **BigQuery** is the serverless warehouse at the center of the data plane; control cost by **partitioning/clustering** and pruning columns (on-demand billing is per byte scanned) or reserving **slots** for heavy use.
- **BigFrames** runs pandas/scikit-learn-style Python inside BigQuery; **BigQuery ML** trains models and calls **Vertex/Gemini remote models** directly from SQL — the fastest path to a baseline.
- **Cloud SQL** is for transactional (OLTP) workloads; move/federate its data into BigQuery for analytics and ML. Use **external tables** to query Cloud Storage in place.
- **Dataflow** (managed Apache Beam) does heavy batch/streaming processing and feature computation; **Pub/Sub** ingests events, landing them **directly in BigQuery** or via **Dataflow** for transformation — keep consumers **idempotent**.

## CLI cheat-sheet

```bash
# --- Datasets & tables (location is IMMUTABLE; co-locate with compute) ---
bq --location=us-central1 mk --dataset --label=team:ml myco-fraud-dev:fraud
bq mk --table \
  --time_partitioning_field=event_date --time_partitioning_type=DAY \
  --time_partitioning_expiration=7776000 \
  --clustering_fields=merchant_category,account_id \
  --schema=event_date:DATE,account_id:STRING,amount:FLOAT,label:INT64 \
  myco-fraud-dev:fraud.transactions
bq update --require_partition_filter myco-fraud-dev:fraud.transactions

# --- Load (Parquet/Avro carry schema; CSV/JSON need --autodetect or --schema) ---
bq load --source_format=PARQUET fraud.transactions gs://myco-fraud-data/v1/*.parquet
bq load --autodetect --replace --source_format=CSV --skip_leading_rows=1 \
  fraud.transactions gs://myco-fraud-data/v1/transactions.csv

# --- Query (estimate first, cap the blast radius, name columns not *) ---
bq query --use_legacy_sql=false --dry_run 'SELECT amount FROM `...` WHERE event_date="2026-06-01"'
bq query --use_legacy_sql=false --maximum_bytes_billed=10000000000 \
  --destination_table=myco-fraud-dev:fraud.training_set --replace \
  'SELECT account_id, amount, label FROM `myco-fraud-dev.fraud.transactions`'
bq query --use_legacy_sql=false --parameter='cutoff:DATE:2026-06-01' \
  'SELECT COUNT(*) FROM `...` WHERE event_date >= @cutoff'

# --- Inspect / manage ---
bq ls myco-fraud-dev:fraud
bq show --schema myco-fraud-dev:fraud.transactions
bq head -n 20 myco-fraud-dev:fraud.transactions            # free peek
bq extract --destination_format=PARQUET fraud.transactions gs://myco-fraud-data/export/*.parquet
bq cp fraud.transactions fraud.transactions_bak
bq cp --snapshot fraud.training_set fraud.training_set_snap  # cheap point-in-time freeze
bq rm -f -t fraud.transactions_bak

# --- External tables, connections for remote/Vertex models ---
bq mk --table --external_table_definition=PARQUET=gs://myco-fraud-data/lake/*.parquet \
  myco-fraud-dev:fraud.txn_external
bq mk --connection --location=us-central1 --connection_type=CLOUD_RESOURCE vertex_conn

# --- Direct Pub/Sub -> BigQuery ingestion (no pipeline) ---
gcloud pubsub topics create transactions
gcloud pubsub subscriptions create transactions-to-bq --topic=transactions \
  --bigquery-table=myco-fraud-dev:fraud.raw_events --use-topic-schema
```

```sql
-- BigQuery ML lifecycle: train -> evaluate -> predict -> explain
CREATE OR REPLACE MODEL `fraud.txn_classifier`
OPTIONS(model_type='BOOSTED_TREE_CLASSIFIER', input_label_cols=['label']) AS
SELECT amount, merchant_category, label FROM `myco-fraud-dev.fraud.transactions`;
SELECT * FROM ML.EVALUATE(MODEL `fraud.txn_classifier`, (SELECT * FROM `fraud.holdout`));
SELECT * FROM ML.PREDICT(MODEL `fraud.txn_classifier`, (SELECT * FROM `fraud.new_transactions`));

-- Attribute query cost by user over the last 7 days
SELECT user_email, ROUND(SUM(total_bytes_billed)/POW(2,40),2) AS tib_billed
FROM `region-us-central1`.INFORMATION_SCHEMA.JOBS
WHERE creation_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND job_type='QUERY'
GROUP BY user_email ORDER BY tib_billed DESC;
```

## Try it

Build a small data plane and produce a training set:

1. Create a dataset and load a CSV/Parquet file from Cloud Storage into a partitioned, clustered BigQuery table.
2. Write a SQL query that assembles a labeled training table (joins, aggregates, a date filter) and observe the "bytes processed" estimate — then add a partition filter and watch it drop.
3. Train a `BOOSTED_TREE_CLASSIFIER` with BigQuery ML `CREATE MODEL` and run `ML.PREDICT` on a holdout slice.
4. Create a Pub/Sub topic with a direct BigQuery subscription, publish a few JSON messages with `gcloud pubsub topics publish`, and confirm they land in the table — you have now stood up streaming ingestion with zero pipeline code.
