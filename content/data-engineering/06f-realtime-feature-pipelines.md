# 06 — Advanced Topics: Real-Time ML Feature Pipelines — The Bridge to MLOps

This is a supplementary advanced topic that sits between the data-engineering track and the MLOps course. Everything you've learned about streaming (Kafka, Flink, Spark Structured Streaming), storage (Parquet, key-value stores), and correctness (idempotency, exactly-once) now gets pointed at one specific consumer: a machine-learning model that needs numbers computed from events seconds old, and needs them computed *the same way* at training time and at inference time. Get this wrong and your offline metrics lie to you. This is the single most under-taught link in the pipeline, and it's where DEs and ML engineers actually meet.

> **Date-stamped claims.** Product and API details below reflect the state as of mid-2026. Feast and Tecton APIs move; verify the current signature before you ship.

---

## The Core Problem — Train/Serve Skew

Start with a concrete model. A fraud model scores a card transaction. The signal it needs is behavioral and recent: *how many transactions has this card made in the last 60 seconds*, *is this merchant category new for this user*, *how far is this transaction from the last one geographically*. None of those are columns in a table. They're aggregations over an event stream, and their value at 12:00:03 is different from their value at 12:00:01. The model is worthless if it scores on features that are minutes stale.

Now the trap. You have two moments where these features must exist:

1. **Training.** You have a labeled history — millions of past transactions, each tagged fraud/not-fraud. For each one you need the feature values *as they were at the moment that transaction happened*.
2. **Inference.** A live transaction arrives. You need the same features computed against the live stream, in single-digit milliseconds, before you approve or decline.

If those two paths use *different code* — a Spark batch job computes "transactions in last 60s" for training, and a hand-written Python function computes it for the live scorer — they will disagree. Maybe the batch job counts inclusive of the current event and the online path doesn't. Maybe one uses UTC and the other local time. Maybe the batch job silently backfills a null as zero and the online path returns null. The model trained on the batch distribution now sees a subtly different distribution in production. This is **train/serve skew**, and it is the number-one cause of "the model looked great offline and tanked in prod."

The fix is architectural, not clever: **one definition of each feature, two execution contexts.** The feature store exists to enforce exactly that.

---

## Offline vs Online Store — Why You Need Both

A feature store separates two storage concerns that have opposite requirements:

- **Offline store** — holds the full history of feature values, keyed by entity and timestamp. This is your data warehouse or lakehouse: BigQuery, Snowflake, Redshift, or an Iceberg/Delta table on S3. It's optimized for large scans (you're pulling millions of rows to build a training set) and it keeps every historical value, not just the latest. Throughput matters; per-row latency does not.

- **Online store** — holds only the *latest* value per entity, keyed for point lookup. This is a low-latency KV store: Redis, DynamoDB, Bigtable, or Postgres for smaller scale. At inference you do `get(entity_key)` and expect sub-10ms. It's tiny by comparison — one row per entity, not one per event.

You need both because the two access patterns are irreconcilable in one system. Training reads "all values for entity X across all of last year" — a warehouse scan. Serving reads "the current value for entity X" a hundred thousand times a second — a KV lookup. No single store is good at both. The feature store's job is to keep them *consistent*: the value the online store serves for entity X must be the same value the offline store would have recorded for X at that timestamp. That consistency is what kills skew.

---

## Point-in-Time-Correct Joins — Not Leaking the Future

This is the subtle one, and it's pure data engineering. You already know the shape of it from SQL window functions: you learned `LAG`, `LEAD`, and windowed aggregates that respect ordering. Point-in-time correctness is that discipline applied to label construction.

Say you're building the training set. You have a table of labels — each row is `(entity_id, event_timestamp, is_fraud)`. You want to attach feature values to each label. The naive join is:

```sql
SELECT l.*, f.txn_count_60s
FROM labels l
JOIN features f ON l.card_id = f.card_id
```

This is catastrophically wrong. It joins the label to *whatever the current feature value is*, or to all historical feature rows with no time bound. Either way you leak: you attach a feature value computed *after* the label's timestamp, meaning your training data contains information from the future that will not exist at inference time. Your offline metrics become fantasy.

The correct join is a temporal **AS OF** join: for each label, pick the *most recent feature value whose timestamp is less than or equal to the label's event timestamp*.

```sql
SELECT l.card_id, l.event_timestamp, l.is_fraud, f.txn_count_60s
FROM labels l
LEFT JOIN LATERAL (
    SELECT txn_count_60s
    FROM features f
    WHERE f.card_id = l.card_id
      AND f.event_timestamp <= l.event_timestamp   -- no future leakage
    ORDER BY f.event_timestamp DESC
    LIMIT 1
) f ON true
```

That `event_timestamp <= label_timestamp` predicate is the entire game. It's the same "respect the arrow of time" instinct as a windowed `LAG`. Feature stores implement this AS-OF join for you across every feature view — Feast's `get_historical_features` and Databricks' point-in-time join take an *entity dataframe* (your labels + timestamps) and return a training frame with each feature filled in as-of. You should still understand what it's doing, because when a training set silently underperforms, a broken point-in-time join is the first thing to check. Add a TTL to features so a lookup that finds only a very old value returns null rather than a stale reading, and confirm your online path treats "no recent value" the same way training did.

---

## Streaming Feature Computation

Where do fresh feature values come from? For real-time features, a stream job computes them continuously and writes the latest value into the online store. The canonical pipeline is the one you already know how to build:

```
events → Kafka → Flink / Spark Structured Streaming → aggregate over windows → online store (Redis/DynamoDB)
                                                     ↘ append raw/aggregated to offline store (Iceberg/warehouse)
```

Kafka is the durable, partition-ordered log. Flink or Spark Structured Streaming does the stateful aggregation — counts, sums, and rates over sliding or tumbling windows keyed by entity. The result lands in two places: the online store (latest value, for serving) and the offline store (full history, for training and backfill). This dual-write is exactly the consistency the feature store cares about.

A Spark Structured Streaming sketch for "transactions per card in the last 60 seconds":

```python
from pyspark.sql.functions import window, count, col

agg = (
    spark.readStream.format("kafka")
        .option("subscribe", "transactions")
        .load()
        .select(from_json_expr("value"))          # parse event
        .withWatermark("event_time", "2 minutes")   # bound state, allow late data
        .groupBy(col("card_id"), window("event_time", "60 seconds", "10 seconds"))
        .agg(count("*").alias("txn_count_60s"))
)

# write the latest value per card into the online store
agg.writeStream.foreachBatch(write_to_redis).outputMode("update").start()
```

The `withWatermark` is doing real work: it bounds the state Spark must hold and defines how late an event can arrive and still be counted. In Flink you'd express the same thing with a keyed window and a watermark strategy; Flink tends to win when state is large and exactly-once and low latency both matter. SQL-first stream engines (RisingWave, Materialize) let you define these aggregations as standing SQL views if you'd rather not write a job.

**Freshness SLAs.** Every real-time feature has a freshness budget: the gap between an event happening and its effect being visible at inference. For fraud, that budget may be under a second; for a "products viewed this week" recommender feature, hours is fine. Freshness is not something you tighten by lowering a read staleness setting — the online store can only serve what's been written to it. If the stream job lags, the feature is stale no matter how you configure reads. So you tier SLAs by feature, monitor the write lag (event time vs. materialization time) per feature, and alert only when a breach persists past one processing cycle rather than on every transient hiccup. A stalled stream job is a silent model-degradation incident; treat it like one.

---

## On-Demand Features — Computed at Request Time

Some features can't be precomputed because they depend on data that only exists *in the request*. The classic example: `distance(transaction_location, user_home_location)`. The transaction location arrives in the scoring request; the home location is a stored feature. You can't materialize the distance ahead of time because you don't know the transaction location until it happens.

These are **on-demand** (or request-time) features: transformations executed at inference against a mix of already-stored features and request-time inputs. The critical rule is that the *same transformation code* runs during training-set construction and during online serving — otherwise you've reintroduced skew through the back door. Feast models this with the `@on_demand_feature_view` decorator, which declares its inputs as a combination of existing feature views and a `RequestSource`, and runs identically in `get_historical_features` and `get_online_features`. Keep the heavy aggregations in the stream job and reserve on-demand transforms for the genuinely request-dependent arithmetic; running expensive logic on every request eats your latency budget.

---

## Feast as the Orchestration Layer

Feast is the open-source reference implementation of these ideas, and it's worth knowing concretely even if you end up on Tecton, Databricks Feature Store, or a cloud-native store. Its architecture is the dual-store model: you bring your own compute (Spark, Flink, dbt) and your own stores, and Feast is the registry and consistency layer that ties one feature definition to both an offline and an online store. The core objects:

- **Entity** — the thing features are about (a card, a user, a driver). Defined by its join key.
- **FeatureView** — a named group of features, tied to an entity and a data source, with a schema and a TTL.
- **StreamFeatureView** — a feature view whose source is a stream (Kafka/Kinesis), populating the online store continuously.
- **Materialization** — the process of loading the latest values from the offline store into the online store, so serving reads are cheap.

Definitions look like this:

```python
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, KafkaSource, stream_feature_view
from feast.types import Int64, Float32

card = Entity(name="card", join_keys=["card_id"])

# Batch/offline-backed feature view (historical values live in the offline store)
card_stats = FeatureView(
    name="card_stats",
    entities=[card],
    schema=[
        Field(name="txn_count_60s", dtype=Int64),
        Field(name="avg_amount_1h", dtype=Float32),
    ],
    source=FileSource(path="s3://.../card_stats/", timestamp_field="event_timestamp"),
    ttl=timedelta(days=1),   # lookups older than this return null, not stale
)

# Streaming feature view: same schema, computed live off Kafka
@stream_feature_view(
    entities=[card],
    ttl=timedelta(days=1),
    mode="spark",
    schema=[Field(name="txn_count_60s", dtype=Int64)],
    timestamp_field="event_timestamp",
    online=True,
    source=KafkaSource(name="txn_stream", kafka_bootstrap_servers="...", topic="transactions",
                       timestamp_field="event_timestamp"),
)
def card_stats_stream(df):
    return df   # windowed aggregation logic
```

Building a training set uses the point-in-time join — you pass an entity dataframe (your labels and their timestamps) and Feast returns each feature filled in as-of:

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=labels_df,   # columns: card_id, event_timestamp, is_fraud
    features=["card_stats:txn_count_60s", "card_stats:avg_amount_1h"],
).to_df()
```

Then you push the latest values into the online store and serve them:

```python
# batch load latest values into the online store (or run this incrementally on a schedule)
store.materialize_incremental(end_date=datetime.utcnow())

# inference: single-digit-ms lookup by entity key
features = store.get_online_features(
    features=["card_stats:txn_count_60s", "card_stats:avg_amount_1h"],
    entity_rows=[{"card_id": 1001}],
).to_dict()
```

Note the symmetry: `get_historical_features` for training, `get_online_features` for serving, *the same feature references* (`"card_stats:txn_count_60s"`) in both. That shared reference is the anti-skew guarantee. Streaming values arrive either through the stream feature view or via `store.push(...)` from a push source, so a Flink/Spark job can shove computed values straight into the online store between materializations.

The managed alternatives trade flexibility for less operational burden. **Tecton** owns the feature compute itself — you point a transformation at a Kafka topic and it handles batch, streaming, and request-time execution plus freshness monitoring, rather than you wiring Spark and Redis together. **Databricks** and the cloud-native stores (Vertex, SageMaker) do point-in-time joins natively against their own warehouses. Feast is the right thing to *learn on* because it makes every seam visible; a managed platform is often the right thing to *run* when freshness SLAs are contractual.

---

## The Handoff to MLOps

This is where the data-engineering track hands the baton to the MLOps course. Everything up to the online store is data engineering: streams, windows, watermarks, consistency, point-in-time correctness. Everything after is MLOps: the model server calls `get_online_features`, assembles the vector, scores it, and — critically — *logs the exact features it served alongside the prediction*. That served-feature log closes the loop: it's how you later detect skew (compare served features to what the offline store says they should have been), how you build the next training set without recomputing, and how you monitor feature drift in production.

So the mental model to carry into MLOps: **the feature store is the contract between the two disciplines.** The DE owns feature freshness, correctness, and the pipelines that fill both stores. The ML engineer owns the model, the serving path, and the monitoring. The feature store is the shared surface where a single feature definition guarantees that what the model learned from is what the model sees. If you internalize point-in-time correctness and the one-definition/two-contexts principle, the rest of real-time ML serving is plumbing you already know how to build.

---

## You can now

- Explain train/serve skew as a concrete failure mode and name the architectural fix: one feature definition, two execution contexts behind a shared API.
- Justify the offline/online store split from first principles — warehouse-scan vs. point-lookup — and state why consistency between them is the whole point.
- Write a point-in-time-correct AS-OF join and recognize the future-leakage bug in a naive feature join, tying it back to windowed SQL.
- Sketch a Kafka → Flink/Spark Structured Streaming → online-store pipeline with watermarking, and reason about per-feature freshness SLAs and write-lag monitoring.
- Distinguish precomputed streaming features from request-time on-demand features and know why both must share transformation code.
- Read and write Feast entities, feature views, stream feature views, materialization, and `get_historical_features`/`get_online_features`, and articulate the handoff to the MLOps serving and monitoring layer.
