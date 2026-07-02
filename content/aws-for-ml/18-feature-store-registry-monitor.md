# 18 — SageMaker Feature Store, Model Registry, and Model Monitor

Three SageMaker components appeared briefly across earlier modules — the feature store in the data chapter, the registry and monitor in the pipelines chapter — because they are the connective tissue of MLOps, not standalone tools. This module treats them as the deep subject they deserve. **Feature Store** guarantees the features a model trains on are the features it serves on. **Model Registry** is the versioned, approval-gated catalog that decides what deploys. **Model Monitor** watches production and tells you when reality has drifted from training. Together they turn "a model that worked once" into a system that stays trustworthy over time.

## Feature Store: killing train/serve skew

The classic production failure is **train/serve skew**: a feature is computed one way in the training notebook and a subtly different way in the serving code, so a model that scored 0.94 offline quietly degrades in production. **SageMaker Feature Store** solves this by making a feature's definition the single source of truth, materialized into two synchronized stores. The **online store** is a low-latency key-value store (DynamoDB-class) for millisecond reads at inference time. The **offline store** lives in S3 as time-stamped, append-only Parquet, used to build training sets and backfills. You write a record once; the *same* definition feeds both.

A **feature group** defines the schema — a list of features, a `RecordIdentifierFeatureName` (the entity key, e.g. `user_id`), and an `EventTimeFeatureName` (the timestamp that orders records). You must enable at least one of the two stores.

```bash
# Create a feature group with both online and offline stores
aws sagemaker create-feature-group \
  --feature-group-name user-features \
  --record-identifier-feature-name user_id \
  --event-time-feature-name event_time \
  --feature-definitions '[{"FeatureName":"user_id","FeatureType":"String"},{"FeatureName":"event_time","FeatureType":"String"},{"FeatureName":"purchases_7d","FeatureType":"Integral"}]' \
  --online-store-config '{"EnableOnlineStore":true}' \
  --offline-store-config '{"S3StorageConfig":{"S3Uri":"s3://my-ml-data/feature-store/"}}' \
  --role-arn arn:aws:iam::<acct>:role/ml-sagemaker-exec
```

Ingestion and retrieval use a *separate runtime namespace*, `sagemaker-featurestore-runtime`. `put-record` writes to the online store immediately and buffers into the offline store within about 15 minutes; `get-record` reads the latest values from the online store by identifier — this is the millisecond lookup an inference request makes.

```bash
# Write a record (goes to both stores by default; TargetStores can restrict)
aws sagemaker-featurestore-runtime put-record \
  --feature-group-name user-features \
  --record '[{"FeatureName":"user_id","ValueAsString":"u_123"},{"FeatureName":"event_time","ValueAsString":"2026-07-02T00:00:00Z"},{"FeatureName":"purchases_7d","ValueAsString":"4"}]'

# Read the latest online value at inference time
aws sagemaker-featurestore-runtime get-record \
  --feature-group-name user-features --record-identifier-value-as-string u_123
```

The property that makes the offline store special is **point-in-time-correct** training sets. Because every record is time-stamped, you can reconstruct exactly what a feature's value *was* at the moment of each historical label, avoiding the leakage of letting future information bleed into a training row. You build these with a point-in-time join (the SDK's `create_dataset` / Athena queries over the offline store), not by grabbing the latest value for every entity.

## Model Registry: the deploy gate

A trained artifact in S3 is not yet something you can safely ship — you do not know its metrics, its lineage, or whether a human approved it. The **Model Registry** fixes this. A **model package group** is a named collection of versions of one logical model (e.g. `fraud-scorer`); each training run registers a **model package** (a version) that bundles the artifact location, the inference container, evaluation metrics, and an **approval status**. Deployment reads *from the registry*, so what ships is always a tracked, approved version — the clean hand-off between training (module 09/11) and inference (module 10).

```bash
# One-time: the group that holds all versions of a model
aws sagemaker create-model-package-group --model-package-group-name fraud-scorer

# List versions and their approval status
aws sagemaker list-model-packages --model-package-group-name fraud-scorer

# Promote a candidate: flip PendingManualApproval -> Approved (this is the gate)
aws sagemaker update-model-package \
  --model-package-arn arn:aws:sagemaker:us-east-1:<acct>:model-package/fraud-scorer/3 \
  --model-approval-status Approved
```

In a pipeline (module 11) a **condition step** registers a candidate only if it clears a metric bar, usually as `PendingManualApproval`, and a separate approval — manual, or automated by a CI check — flips it to `Approved`, which an EventBridge rule can pick up to trigger deployment. That status change *is* the promotion event of the whole MLOps loop.

## Model Monitor: detecting drift in production

A model is trained on a snapshot of the world; the world then moves. **Model Monitor** detects that movement by comparing live endpoint traffic against a baseline captured at training time, on a schedule, emitting violation reports and CloudWatch metrics you can alarm on. It requires **data capture** enabled on the endpoint — the endpoint logs a sample of requests and responses to S3, which the monitoring jobs analyze. There are four monitor types, each answering a different "has reality drifted?" question:

- **Data quality** — do the incoming feature distributions and types match the training baseline? (missing values, range shifts, type changes)
- **Model quality** — has predictive accuracy dropped? This needs ground-truth labels ingested after the fact to compare against predictions.
- **Bias drift** — have the fairness metrics (via Clarify) moved since training?
- **Feature attribution drift** — has the *explanation* (which features drive predictions) shifted, an early warning even before accuracy visibly drops?

```bash
# 1) Enable data capture when creating the endpoint config
aws sagemaker create-endpoint-config \
  --endpoint-config-name fraud-scorer-cfg \
  --production-variants '[{"VariantName":"AllTraffic","ModelName":"fraud-scorer","InstanceType":"ml.g5.xlarge","InitialInstanceCount":1}]' \
  --data-capture-config '{"EnableCapture":true,"InitialSamplingPercentage":20,"DestinationS3Uri":"s3://my-ml-data/datacapture/","CaptureOptions":[{"CaptureMode":"Input"},{"CaptureMode":"Output"}]}'

# 2) Define a data-quality monitoring job (runs on a schedule against the baseline)
aws sagemaker create-data-quality-job-definition --cli-input-json file://dq-job.json

# 3) Schedule it; violations surface as CloudWatch metrics you alarm on
aws sagemaker create-monitoring-schedule --cli-input-json file://schedule.json
```

The loop this closes: a drift alarm in CloudWatch fires an **EventBridge** event that restarts the training pipeline, which reads fresh data, retrains, gates on quality, registers a new version, and — on approval — redeploys. Drift is the trigger; the registry is the gate; the feature store keeps the retrain honest. That is MLOps as a self-correcting system rather than a one-time deployment.

## How this fits the whole ML solution

These three are the spine of the reference architecture from module 12. The feature store sits between the data lake and both training and serving, so the same feature definition flows to the offline store (building point-in-time training sets) and the online store (millisecond inference lookups) — no skew. The registry sits between training and inference as the approval gate, so nothing reaches an endpoint unversioned or unapproved. Model Monitor sits on the live endpoint and feeds drift back to EventBridge, closing the retraining loop. Remove any one and the system regresses: without the feature store, skew; without the registry, untracked deploys; without the monitor, silent decay. Together they are what makes the platform operable for years.

## Key takeaways

- Feature Store's synchronized **online** (millisecond) and **offline** (S3 Parquet) stores share one feature definition, eliminating train/serve skew; the offline store's timestamps enable **point-in-time-correct** training sets.
- Ingest/read features through the separate `sagemaker-featurestore-runtime` (`put-record`/`get-record`); online writes are immediate, offline lands within ~15 minutes.
- Model Registry versions models in a **model package group**; the `Approved` status is the deploy gate and the train→serve hand-off.
- Model Monitor compares live traffic to a training **baseline** (requires **data capture** on the endpoint) across data-quality, model-quality, bias, and feature-attribution drift.
- The loop closes drift → CloudWatch alarm → EventBridge → retrain pipeline → gate on quality → register → approve → redeploy.

## CLI cheat-sheet

```bash
# --- Feature Store: control plane ---
aws sagemaker create-feature-group --feature-group-name fg --record-identifier-feature-name id \
  --event-time-feature-name ts --feature-definitions <json> \
  --online-store-config '{"EnableOnlineStore":true}' \
  --offline-store-config '{"S3StorageConfig":{"S3Uri":"s3://.../fs/"}}' --role-arn <arn>
aws sagemaker describe-feature-group --feature-group-name fg
aws sagemaker list-feature-groups
aws sagemaker delete-feature-group --feature-group-name fg

# --- Feature Store: runtime (ingest / read) ---
aws sagemaker-featurestore-runtime put-record --feature-group-name fg --record <json>
aws sagemaker-featurestore-runtime get-record --feature-group-name fg --record-identifier-value-as-string u_123
aws sagemaker-featurestore-runtime batch-get-record --identifiers <json>

# --- Model Registry ---
aws sagemaker create-model-package-group --model-package-group-name fraud-scorer
aws sagemaker create-model-package --model-package-group-name fraud-scorer \
  --inference-specification <json> --model-approval-status PendingManualApproval
aws sagemaker list-model-packages --model-package-group-name fraud-scorer
aws sagemaker describe-model-package --model-package-name <arn>
aws sagemaker update-model-package --model-package-arn <arn> --model-approval-status Approved

# --- Model Monitor ---
# data capture is set in create-endpoint-config --data-capture-config
aws sagemaker create-data-quality-job-definition --cli-input-json file://dq-job.json
aws sagemaker create-model-quality-job-definition --cli-input-json file://mq-job.json
aws sagemaker create-monitoring-schedule --cli-input-json file://schedule.json
aws sagemaker list-monitoring-schedules
aws sagemaker describe-monitoring-schedule --monitoring-schedule-name sched
aws sagemaker list-monitoring-executions --monitoring-schedule-name sched
```

## Try it

Create a feature group with both stores enabled, `put-record` a few entities, and confirm `get-record` returns the latest online values in milliseconds while the offline store populates in S3 within ~15 minutes. Train a model, register it to a model package group as `PendingManualApproval`, then flip it to `Approved` and confirm only the approved version is what your deployment step reads. Finally, deploy an endpoint with data capture enabled, attach a data-quality monitoring schedule with a baseline, send it deliberately drifted inputs, and confirm a CloudWatch metric records the violation — the exact signal that, wired to EventBridge, would kick off a retrain.
