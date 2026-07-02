# 17 — Vertex AI Feature Store, Experiments, and Metadata

Module 09 covered training, Pipelines, and the Model Registry; module 13 covers model monitoring and drift. Between them sit the MLOps components that make a model *reproducible, comparable, and safe to serve* — and this module owns them: the **Vertex AI Feature Store** (so training and serving read the same features), **Vertex AI Experiments** (so "the model improved" is a measured claim, not folklore), **Vertex ML Metadata and Lineage** (so you can answer "which dataset produced this model"), and **Model Evaluation** (so the pipeline's eval gate has real numbers behind it). All of these use the current `from google.cloud import aiplatform` SDK, which is unaffected by the generative-AI SDK deprecation. We continue the fraud-scoring example throughout.

## Vertex AI Feature Store: the BigQuery-backed model

The current Feature Store is **BigQuery-backed**, which changed its shape from the older "featurestore" resource. There are three concepts:

- A **Feature Group** registers a BigQuery **table or view** as the **offline store**. The group points at the data; it does not copy it. This is where training reads features, straight from BigQuery, partitioned and joined with the full power of the warehouse (module 08).
- **Features** are the named columns within a group's source that you want to serve. You register the ones that matter, not every column.
- A **Feature Online Store** plus **Feature Views** provide the **online store** for low-latency serving. A feature view selects features (from feature groups or directly from BigQuery) and **syncs** them from BigQuery into the online store on a schedule, so serving reads the freshest values in single-digit milliseconds.

The reason this architecture exists is **training-serving skew**. If the training code computes "5-minute card velocity" one way and the serving code computes it another, the model sees different inputs in production than it trained on and silently degrades. By defining the feature once (a BigQuery column landed by the Dataflow pipeline in module 16) and reading it offline for training and online for serving, the definition is identical by construction.

**There is no `gcloud` surface for the BigQuery-backed Feature Store** — unlike most services in this course, feature groups, online stores, and feature views are managed only through the **Python SDK, the REST API, or Terraform** (the Cloud Console also exposes them). The idiomatic path is the `vertexai.resources.preview.feature_store` module:

```python
from google.cloud import aiplatform
from vertexai.resources.preview import feature_store

aiplatform.init(project="myco-fraud-dev", location="us-central1")

# 1) Register the BigQuery source as a feature group (offline store)
fg = feature_store.FeatureGroup.create(
    name="fraud_features",
    source=feature_store.utils.FeatureGroupBigQuerySource(
        uri="bq://myco-fraud-dev.fraud.card_velocity_5m",
        entity_id_columns=["card"],
    ),
)

# 2) Register the individual features you will serve
fg.create_feature(name="txn_count_5m")
fg.create_feature(name="amount_mean_5m")
```

The source table needs an **entity ID column** (here `card`) and, for point-in-time correctness, a feature timestamp column. The offline store is exactly your BigQuery table, so its **freshness is whatever your Dataflow pipeline writes** — a stale streaming job means stale features.

## The online store and feature views

For online serving you create a **Feature Online Store** and then a **Feature View** synced into it. Serving is **Bigtable-backed**: the online store runs on Bigtable nodes that you either fix or autoscale. (Note the "optimized" online serving option is deprecated as of May 2026 — Bigtable serving is the supported path for new deployments; do not build on optimized.)

```bash
# Create a Bigtable online store with autoscaling
gcloud ai feature-online-stores create fraud_online \
  --region=us-central1 \
  --min-node-count=1 \
  --max-node-count=3 \
  --cpu-utilization-target=60

# Create a feature view that pulls from the feature group and syncs hourly
gcloud ai feature-views create card_features \
  --region=us-central1 \
  --feature-online-store=fraud_online \
  --feature-groups=fraud_features:txn_count_5m,amount_mean_5m \
  --cron="0 * * * *"

# Trigger a sync now (rather than waiting for the schedule) and watch it
gcloud ai feature-views sync card_features \
  --feature-online-store=fraud_online --region=us-central1
gcloud ai feature-view-syncs list \
  --feature-view=card_features --feature-online-store=fraud_online \
  --region=us-central1
```

A feature view can source from **feature groups** (`--feature-groups`, tying serving to the registered offline definitions) or from a registry/BigQuery source directly. The `--cron` schedule controls how often the online store refreshes from BigQuery; for a fraud system you want this frequent, but each sync is a BigQuery read and a Bigtable write, so there is a cost/freshness tradeoff to tune.

Serving reads a feature vector by entity ID. In Python this is the online serving path that your prediction service calls per request:

```python
from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1.types import feature_online_store_service as fos

client = FeatureOnlineStoreServiceClient(
    client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"})
resp = client.fetch_feature_values(request=fos.FetchFeatureValuesRequest(
    feature_view=("projects/myco-fraud-dev/locations/us-central1/"
                  "featureOnlineStores/fraud_online/featureViews/card_features"),
    data_key=fos.FeatureViewDataKey(key="c-42"),
))
```

The critical gotcha: **a feature view must be synced before it can serve** — create it, run a sync, then read. And the online store is **always-on Bigtable nodes**, a real fixed cost, so size autoscaling to your actual QPS rather than provisioning for a peak you rarely hit.

The management operations you will actually run:

```bash
gcloud ai feature-views describe card_features \
  --feature-online-store=fraud_online --region=us-central1
gcloud ai feature-online-stores list --region=us-central1
gcloud ai feature-online-stores delete fraud_online --region=us-central1
```

## Point-in-time correctness for training data

The subtlest reason to use a feature store is **point-in-time correctness**. When you assemble a training set, each labeled example must see feature values *as they were at the time of that event* — never a value computed after the label was known. Reading "current" feature values into historical rows is **label leakage**: the model learns from information it will not have at inference time, and its offline metrics lie.

The feature store's offline retrieval performs a **point-in-time join**: given a set of entity IDs and event timestamps, it returns the feature values that were current at each timestamp. Your fraud training set — one row per historical transaction with its label — joins against the feature timestamp column so the 5-minute velocity is the value *at that transaction's moment*, not today's. Getting this right is the difference between a model that works and one that scores 0.99 offline and fails in production.

## Vertex AI Experiments

An **Experiment** groups related **runs** so you can compare them. Each run logs **parameters** (hyperparameters, config), **metrics** (accuracy, AUC, loss), **time-series metrics** (per-epoch curves), and **artifacts**. Instead of hunting through logs to remember which learning rate gave which AUC, you query and compare runs in one place — which is what makes the eval-gate philosophy from module 09 enforceable: you cannot gate on "improvement" you did not measure.

```python
from google.cloud import aiplatform

aiplatform.init(project="myco-fraud-dev", location="us-central1",
                experiment="fraud-classifier")

aiplatform.start_run("run-042")
aiplatform.log_params({"model": "xgboost", "max_depth": 6, "lr": 0.1})
for epoch, loss in enumerate(training_losses):
    aiplatform.log_time_series_metrics({"loss": loss}, step=epoch)
aiplatform.log_metrics({"auc": 0.947, "precision_at_1pct": 0.88})
aiplatform.end_run()

# Compare all runs in the experiment as a DataFrame
df = aiplatform.get_experiment_df("fraud-classifier")
```

Experiments integrate with **Vertex AI TensorBoard** for rich loss curves and support **autologging** for common frameworks, so you can capture params and metrics without hand-instrumenting every line. A practical gotcha: **metric cardinality**. Logging thousands of distinct metric keys, or a fresh run per trivial change, turns the comparison view into noise — log a stable, meaningful set.

## Vertex ML Metadata and Lineage

Every pipeline run, dataset, and model produces an entry in **Vertex ML Metadata** — a store of **artifacts** (a dataset, a model, an eval), **executions** (a training step that ran), and **contexts** (the pipeline run that grouped them). The payoff is **lineage**: because Vertex AI Pipelines (module 09) write to the metadata store automatically, you can trace which BigQuery dataset version and which training execution produced a given model version — and, in reverse, which downstream models a dataset fed. This is the backbone of reproducibility and governance: when a model misbehaves in production, lineage tells you exactly what went into it.

```python
from google.cloud import aiplatform

# Trace the lineage of a specific pipeline run's artifacts
job = aiplatform.PipelineJob.get("projects/myco-fraud-dev/locations/us-central1/"
                                 "pipelineJobs/fraud-pipeline-20260702")
for task in job.task_details:
    print(task.task_name, [o.uri for o in task.outputs.values()])
```

You rarely write metadata by hand — running pipelines populates it. The value is in *querying* it later.

## Model Evaluation

The Model Registry (module 09) holds versions; **Model Evaluation** attaches measured quality to a version. You run an evaluation job against a version and a labeled dataset to produce standard metrics (AUC, precision/recall, confusion matrix), and — importantly — **slice-based evaluation**: metrics broken out by a categorical feature (fraud rate by region, by merchant category) so you catch a model that is great overall but fails on a critical segment.

```python
from google.cloud import aiplatform

model = aiplatform.Model("projects/myco-fraud-dev/locations/us-central1/models/fraud-classifier")
eval_job = model.evaluate(
    prediction_type="classification",
    target_field_name="label",
    gcs_source_uris=["gs://myco-fraud-data/eval/frozen_v3.jsonl"],
)
print(eval_job.get_model_evaluation().metrics)
```

These evaluation results are what the pipeline's `dsl.If` gate (module 09) reads: register and promote only if AUC on the **frozen eval set** clears the threshold, and only if no important slice regresses. That closes the eval loop — measurement in Experiments during development, enforced evaluation in the pipeline before promotion.

## How this fits the whole solution

These four components are the connective tissue of the MLOps loop from module 12. The **Feature Store** takes the streaming features that Dataflow lands in BigQuery (module 16) and serves them two ways — offline for training, online for the Vertex prediction endpoint — killing training-serving skew and, via point-in-time joins, preventing label leakage in the training set. **Experiments** measure every training run so improvement is comparable, and **Model Evaluation** produces the metrics that the Vertex Pipeline (module 09) gates promotion on before writing a new version and alias to the Model Registry. **ML Metadata** records the lineage of all of it automatically, so any deployed model is traceable back to its exact data and code. Module 13's monitoring then watches the served model for drift and feeds a retraining trigger — and because features, experiments, evals, and lineage are all captured, that retrain is reproducible rather than a fresh guess.

## Key takeaways

- The **Feature Store** is BigQuery-backed: **feature groups** register BigQuery tables/views as the **offline store**, **feature views** sync into a **Bigtable online store** for low-latency serving — one feature definition, read offline and online, eliminating **training-serving skew**. Use **point-in-time joins** to avoid **label leakage**.
- **Vertex AI Experiments** track params/metrics/artifacts per run (`log_params`/`log_metrics`/`log_time_series_metrics`), integrate with **TensorBoard** and autologging, and make "the model improved" a measured, comparable fact — keep metric cardinality sane.
- **Vertex ML Metadata** records artifacts/executions/contexts and gives automatic **lineage** from Pipelines — which dataset and execution produced which model — the basis of reproducibility and governance.
- **Model Evaluation** attaches metrics (including **slice-based**) to a registry model version, feeding the pipeline **eval gate**; gotchas are always-on **Bigtable online-store cost**, syncing a feature view **before** serving, and offline BigQuery **source freshness**.

## CLI cheat-sheet

```bash
# --- Feature groups (offline store = BigQuery) ---
gcloud ai feature-groups create fraud_features --region=us-central1 \
  --big-query-source=bq://PROJECT.DATASET.TABLE --entity-id-columns=card
gcloud ai feature-groups features create txn_count_5m \
  --feature-group=fraud_features --region=us-central1
gcloud ai feature-groups list --region=us-central1
gcloud ai feature-groups describe fraud_features --region=us-central1

# --- Online store (Bigtable) + feature views ---
gcloud ai feature-online-stores create fraud_online --region=us-central1 \
  --min-node-count=1 --max-node-count=3 --cpu-utilization-target=60
gcloud ai feature-views create card_features --region=us-central1 \
  --feature-online-store=fraud_online \
  --feature-groups=fraud_features:txn_count_5m,amount_mean_5m \
  --cron="0 * * * *"
gcloud ai feature-views sync card_features \
  --feature-online-store=fraud_online --region=us-central1
gcloud ai feature-view-syncs list --feature-view=card_features \
  --feature-online-store=fraud_online --region=us-central1
gcloud ai feature-online-stores list --region=us-central1

# --- Python: init experiments / log runs / compare ---
# aiplatform.init(experiment="fraud-classifier")
# aiplatform.start_run("run-042"); log_params/log_metrics/log_time_series_metrics; end_run()
# aiplatform.get_experiment_df("fraud-classifier")
# Online serving: FeatureOnlineStoreServiceClient.fetch_feature_values(...)
```

## Try it

Wire feature management and tracking into the fraud pipeline:

1. Point a **feature group** at the `fraud.card_velocity_5m` BigQuery table (from module 16) with `--big-query-source` and `--entity-id-columns=card`, then register two features under it.
2. Create a **Bigtable online store** with autoscaling, create a **feature view** over the group with an hourly `--cron`, run a **sync**, and read a feature vector for one card with `fetch_feature_values`.
3. Instrument a training run with **Vertex AI Experiments** — `log_params`, per-epoch `log_time_series_metrics`, final `log_metrics` — then run it twice with different hyperparameters and compare with `get_experiment_df`.
4. Run the training inside a **Vertex AI Pipeline** and inspect **ML Metadata lineage** to confirm the model version traces back to the BigQuery dataset that produced it.
5. Run a **Model Evaluation** on the registered version against a frozen eval file, add a **slice** by region, and wire the AUC into the pipeline's `dsl.If` promotion gate.
