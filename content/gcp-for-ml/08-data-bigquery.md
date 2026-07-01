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

**Pricing model.** Two axes: **storage** (cheap, per-GB, with active vs long-term rates) and **compute**. Compute is either **on-demand** (you pay per byte scanned by each query — so `SELECT *` on a wide table is expensive and column pruning / partitioning is how you save money) or **capacity-based** (BigQuery editions with reserved **slots** for predictable, high-volume workloads). For ML teams, partitioning tables by date and clustering by common filter columns dramatically cuts bytes scanned and cost.

**BigQuery Studio** is the unified workspace (SQL, notebooks, and Python) in the Console. And **BigFrames** (`bigframes`) gives you a pandas- and scikit-learn-like Python API whose operations execute *inside* BigQuery — so you write familiar DataFrame code but the computation scales to the full warehouse without pulling data to a single machine. This is a key tool for ML feature engineering at scale.

```python
import bigframes.pandas as bpd

bpd.options.bigquery.project = "myco-fraud-dev"
df = bpd.read_gbq("myco-fraud-dev.fraud.transactions")
# Runs server-side in BigQuery, not in local pandas memory
features = df.groupby("account_id").agg({"amount": ["mean", "std", "count"]})
```

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

BigQuery ML is the fastest route from data to a working baseline model, and its Gemini integration lets analysts run generative AI over warehouse data with no pipeline. Its tight coupling with Vertex AI (export a BQML model to the model registry, or call Vertex endpoints from SQL) makes it a first-class member of the ML stack, not a toy.

## Cloud SQL and external data

**Cloud SQL** is managed PostgreSQL, MySQL, or SQL Server — the transactional (OLTP) database behind your application. The rule of thumb: **Cloud SQL for operational reads/writes** (a live app updating rows), **BigQuery for analytics and ML** (scanning huge volumes). Data typically flows *from* Cloud SQL (or its events) *into* BigQuery for analysis. BigQuery can also query data in place via **external tables** (over Parquet/CSV/Iceberg in Cloud Storage) and **federated queries** (against Cloud SQL/Spanner), so you can join warehouse and operational data without a full copy.

## Dataflow: large-scale processing

**Dataflow** is Google Cloud's managed **Apache Beam** runner for both **batch** and **streaming** data processing. It autoscales workers, handles fault tolerance, and is the tool for transformations too heavy or too stateful for SQL or a function — parsing and validating raw files, deduplicating, windowing and aggregating a stream, and computing rolling features. For ML it plays two roles: large-scale **preprocessing** of training data, and **streaming feature computation** that keeps an online feature store fresh. Google provides ready-made **Dataflow templates** (for example, Pub/Sub → BigQuery) so common pipelines need no code.

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

Because delivery is at-least-once, downstream consumers must be **idempotent** — the same lesson as event-driven functions.

## How this fits the whole solution

This data plane is the front half of the end-to-end system. **Pub/Sub** ingests real-time events; **Dataflow** transforms streams and batches (computing features); **Cloud Storage** holds raw and bulk files; and **BigQuery** is the warehouse where training sets are assembled and, via BigQuery ML, baseline models and Gemini calls run in SQL. Everything downstream — Vertex AI training reading a BigQuery-derived dataset, a feature store syncing from BigQuery tables, batch prediction writing results back to BigQuery — depends on this layer being clean, partitioned, and well-governed. Master the data plane and the modeling work becomes far easier.

## Key takeaways

- **BigQuery** is the serverless warehouse at the center of the data plane; control cost by **partitioning/clustering** and pruning columns (on-demand billing is per byte scanned) or reserving **slots** for heavy use.
- **BigFrames** runs pandas/scikit-learn-style Python inside BigQuery; **BigQuery ML** trains models and calls **Vertex/Gemini remote models** directly from SQL — the fastest path to a baseline.
- **Cloud SQL** is for transactional (OLTP) workloads; move/federate its data into BigQuery for analytics and ML. Use **external tables** to query Cloud Storage in place.
- **Dataflow** (managed Apache Beam) does heavy batch/streaming processing and feature computation; **Pub/Sub** ingests events, landing them **directly in BigQuery** or via **Dataflow** for transformation — keep consumers **idempotent**.

## Try it

Build a small data plane and produce a training set:

1. Create a dataset and load a CSV/Parquet file from Cloud Storage into a partitioned, clustered BigQuery table.
2. Write a SQL query that assembles a labeled training table (joins, aggregates, a date filter) and observe the "bytes processed" estimate — then add a partition filter and watch it drop.
3. Train a `BOOSTED_TREE_CLASSIFIER` with BigQuery ML `CREATE MODEL` and run `ML.PREDICT` on a holdout slice.
4. Create a Pub/Sub topic with a direct BigQuery subscription, publish a few JSON messages with `gcloud pubsub topics publish`, and confirm they land in the table — you have now stood up streaming ingestion with zero pipeline code.
