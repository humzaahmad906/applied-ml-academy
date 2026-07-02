# 09 — Vertex AI: Training and Pipelines

Vertex AI is Google Cloud's managed machine learning platform — the place where the raw compute, storage, and data services of earlier modules come together into a coherent ML workflow. (You may see it recently rebranded in some docs as the "Gemini Enterprise Agent Platform"; the product and APIs are the same, and "Vertex AI" remains the working name.) This module covers the training side: running custom training jobs on managed accelerators, scaling to distributed training, orchestrating multi-step workflows with Vertex AI Pipelines, and versioning results in the Model Registry. The Python entry point throughout is the `google-cloud-aiplatform` SDK, imported as `from google.cloud import aiplatform` — this is current and not affected by the generative-AI SDK deprecation that only touches the `vertexai.generative_models` submodules.

## Why managed training

You can train on a raw Compute Engine GPU VM, but you own everything: provisioning, drivers, failure recovery, teardown, and remembering to shut the box off. Vertex AI **custom training** takes a container or Python package, provisions exactly the machine types and accelerators you specify, runs your code, streams logs to Cloud Logging, writes outputs to Cloud Storage, and tears the resources down when done — you pay only for the job's duration. It integrates with the model registry, pipelines, experiments, and monitoring, so a training run is a tracked, reproducible artifact rather than a thing that happened on someone's VM.

## Submitting a custom training job

There are two container models: **prebuilt containers** (Google-provided images with PyTorch/TensorFlow/scikit-learn, into which you inject a training script or Python package) and **custom containers** (your own image from Artifact Registry with your exact environment — the more reproducible choice). You configure **worker pool specs**: the machine type, accelerator type and count, replica count, and the container or package to run.

```python
from google.cloud import aiplatform

aiplatform.init(project="myco-fraud-dev", location="us-central1",
                staging_bucket="gs://myco-fraud-staging")

# A custom-container training job
job = aiplatform.CustomContainerTrainingJob(
    display_name="fraud-train",
    container_uri="us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/train:v1",
    model_serving_container_image_uri=(
        "us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/serve:v1"),
)

model = job.run(
    replica_count=1,
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    args=["--epochs=20", "--data=gs://myco-fraud-data/datasets/v1/"],
    base_output_dir="gs://myco-fraud-models/fraud/run-001",
)
```

Two things to note. First, Vertex AI **mounts your Cloud Storage buckets via Cloud Storage FUSE** into the training container (visible under `/gcs/<bucket>`), so your code can read datasets and write checkpoints with ordinary file paths. Second, if you pass a `model_serving_container_image_uri`, the job can register the trained model directly to the registry, ready to deploy.

The equivalent from the command line, which is what CI/CD often uses. Note that `--worker-pool-spec` takes comma-separated key/value pairs, and you pass runtime flags to your program with `--args` (repeatable) and override the container entrypoint with `--command`. Run as a dedicated least-privilege service account with `--service-account`, and add `--enable-web-access` to open an interactive shell into the training container for live debugging:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=fraud-train \
  --service-account=training-sa@myco-fraud-dev.iam.gserviceaccount.com \
  --enable-web-access \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/train:v1 \
  --args=--epochs=20,--data=gs://myco-fraud-data/datasets/v1/
```

For a **prebuilt container** you instead point the pool at Google's image with `executor-image-uri` and hand it a Python package and module (`python-package-uris`, `python-module`) rather than your own `container-image-uri` — the tradeoff is convenience versus the reproducibility of a fully custom image. Prebuilt image URIs live under `us-docker.pkg.dev/vertex-ai/training/...` (for example `tf-gpu.2-17.py310` or `pytorch-gpu.2-4.py310`); your own custom images live in your Artifact Registry.

Once a job is running you manage it with the rest of the `custom-jobs` verbs — list what is running, inspect one, follow its logs, or kill it:

```bash
gcloud ai custom-jobs list --region=us-central1
gcloud ai custom-jobs describe JOB_ID --region=us-central1
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
gcloud ai custom-jobs cancel JOB_ID --region=us-central1
```

For cost, remember the compute lessons: run interruptible training on **Spot** capacity with frequent checkpointing to Cloud Storage so a preemption resumes rather than restarts. Scheduling strategy is not a top-level flag — you set it in a `--config` YAML on the `CustomJob`'s `scheduling` block (`strategy: SPOT` for Spot, or `strategy: FLEX_START` with a `maxWaitDuration` to queue for on-demand capacity via Dynamic Workload Scheduler):

```yaml
# custom-job.yaml — passed with: gcloud ai custom-jobs create --config=custom-job.yaml ...
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    containerSpec:
      imageUri: us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/train:v1
scheduling:
  strategy: SPOT          # or FLEX_START
  maxWaitDuration: 3600s  # how long to queue for capacity; 0 = wait indefinitely
```

A recurring **gotcha**: the `staging_bucket` (and any data buckets you FUSE-mount) should sit in the same region as the training job. A bucket in the wrong region causes cross-region egress charges and higher latency, and accelerator-backed jobs fail outright if you have not requested **quota** for that accelerator type in that region — check `gcloud compute regions describe us-central1` and raise a Vertex "Custom model training" GPU/TPU quota request before you submit.

## Distributed training

When a model or dataset outgrows a single machine, you scale to **distributed training** by defining multiple **worker pools**. A typical multi-worker GPU job has a primary pool and additional worker pools, all the same machine type, and your framework (PyTorch DDP, JAX, or TensorFlow's distribution strategies) handles the collective communication (NCCL over the internal network) across them. On the command line you express this by repeating `--worker-pool-spec` — the first spec is the primary (chief) pool, subsequent specs are the worker/parameter pools:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=fraud-train-ddp \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/train:v1 \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=3,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/train:v1
```

For synchronous data-parallel training, Vertex AI also offers a **reduction server** — dedicated CPU nodes that aggregate gradients to reduce communication overhead and speed up large multi-GPU jobs. You add it as an *additional* worker pool running Google's reduction-server image (`us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver`) on CPU machines (commonly `n1-highcpu-16`), then set the `NCCL` environment so your training pool routes all-reduce through it:

```bash
  --worker-pool-spec=machine-type=n1-highcpu-16,replica-count=8,container-image-uri=us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver
```

For very large workloads you can target multi-host **TPU** slices (Trillium/v6e, Ironwood) instead of GPUs. The key configuration is still worker pool specs: how many replicas, what accelerators, and which container each pool runs.

## Hyperparameter tuning

Rather than hand-tuning learning rate, batch size, and regularization, you can let Vertex run a **hyperparameter tuning job** that launches many trials, each a full training run with a different point in the search space, and optimizes toward a metric your code reports back. You describe the search in a `studySpec`: the metric to optimize (`metricId` + `goal`), each parameter's range and scale, and (optionally) the search algorithm — Vertex defaults to **Bayesian optimization**, which is far more sample-efficient than grid or random for expensive training runs. The `trialJobSpec` is just a `CustomJob` spec, so every worker-pool, accelerator, and Spot option above applies per trial.

```yaml
# study.yaml
studySpec:
  metrics:
    - metricId: auprc
      goal: MAXIMIZE
  parameters:
    - parameterId: learning_rate
      doubleValueSpec: {minValue: 1.0e-5, maxValue: 1.0e-2}
      scaleType: UNIT_LOG_SCALE
    - parameterId: batch_size
      discreteValueSpec: {values: [128, 256, 512]}
  algorithm: ALGORITHM_UNSPECIFIED   # unspecified = Bayesian optimization
trialJobSpec:
  workerPoolSpecs:
    - machineSpec: {machineType: a2-highgpu-1g, acceleratorType: NVIDIA_TESLA_A100, acceleratorCount: 1}
      replicaCount: 1
      containerSpec:
        imageUri: us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/train:v1
```

```bash
gcloud ai hp-tuning-jobs create \
  --region=us-central1 \
  --display-name=fraud-hpt \
  --config=study.yaml \
  --max-trial-count=40 \
  --parallel-trial-count=4

gcloud ai hp-tuning-jobs list --region=us-central1
gcloud ai hp-tuning-jobs describe JOB_ID --region=us-central1
```

Your training code must **parse a command-line flag named after each `parameterId`** and, crucially, **report the metric back** via the `cloudml-hypertune` package (or `hypertune`), or Vertex has nothing to optimize against. Keep `parallel-trial-count` well below `max-trial-count` so Bayesian optimization can learn from completed trials before launching the next batch — a large parallel count degrades toward random search.

## Vertex AI Pipelines

A real ML workflow is not one script — it is a sequence: validate data, engineer features, train, evaluate, and (if the eval passes) register and deploy. **Vertex AI Pipelines** is the managed orchestrator for this, based on the open **Kubeflow Pipelines (KFP)** SDK. You write each step as a **component** (a containerized function with typed inputs/outputs), wire them into a **pipeline** as a directed graph, compile it, and submit it as a `PipelineJob`. Pipelines give you caching (skip a step whose inputs did not change), lineage tracking, and reproducibility.

```python
from kfp import dsl, compiler

@dsl.component(base_image="python:3.12")
def evaluate(model_dir: str) -> float:
    # ... load model, compute metric on frozen eval set ...
    return 0.94

@dsl.pipeline(name="fraud-pipeline")
def pipeline(data_uri: str):
    train_op = train(data_uri=data_uri)          # a training component
    eval_op = evaluate(model_dir=train_op.output)
    with dsl.If(eval_op.output >= 0.90):
        register(model_dir=train_op.output)      # only register if it passes

compiler.Compiler().compile(pipeline, "fraud_pipeline.yaml")
```

```python
from google.cloud import aiplatform

aiplatform.init(project="myco-fraud-dev", location="us-central1")
run = aiplatform.PipelineJob(
    display_name="fraud-pipeline",
    template_path="fraud_pipeline.yaml",
    parameter_values={"data_uri": "gs://myco-fraud-data/datasets/v1/"},
    pipeline_root="gs://myco-fraud-staging/pipeline-root",
)
run.submit()
```

That `dsl.If` gate — register only if the eval metric clears a threshold — is the single most valuable pattern in the module: it makes "the model improved" a measured, enforced fact rather than a hope. For heavier, cross-service orchestration (mixing BigQuery, Dataflow, and Vertex steps on a schedule), teams also use **Cloud Composer** (managed Apache Airflow); Vertex AI Pipelines is the ML-native choice, Composer the general-purpose one, and they are often used together.

Pipelines are largely SDK-driven — there is no rich `gcloud ai pipelines` surface, so the `aiplatform` SDK is the working interface. Two operational features matter in production. First, **caching**: `PipelineJob` caches each step keyed on its inputs and code, so re-running an unchanged pipeline skips completed steps and only re-executes what actually changed — you can force a fresh run with `enable_caching=False`. Second, **scheduling**: instead of triggering runs by hand, attach a cron schedule with a `PipelineJobSchedule` (or the `job.create_schedule(...)` shortcut) so the train-evaluate-register loop runs on a cadence:

```python
job = aiplatform.PipelineJob(
    display_name="fraud-pipeline",
    template_path="fraud_pipeline.yaml",
    parameter_values={"data_uri": "gs://myco-fraud-data/datasets/v1/"},
    pipeline_root="gs://myco-fraud-staging/pipeline-root",
    enable_caching=True,
)
schedule = job.create_schedule(
    display_name="fraud-nightly-retrain",
    cron="TZ=UTC 0 2 * * *",       # 02:00 UTC daily
    max_concurrent_run_count=1,
    max_run_count=90,
)
```

## The Model Registry

Trained models land in the **Vertex AI Model Registry** — a versioned catalog of your models. Each model has **versions**, and you can assign **aliases** (like `default` or `production`) that point at a specific version, so downstream deployment references a stable alias while you iterate on new versions behind it.

```python
model = aiplatform.Model.upload(
    display_name="fraud-classifier",
    artifact_uri="gs://myco-fraud-models/fraud/run-001/model",
    serving_container_image_uri=(
        "us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/serve:v1"),
    parent_model="projects/.../models/1234567890",  # register as a new VERSION of an existing model
    version_aliases=["candidate"],
)
```

Uploading with a `parent_model` creates a *new version* under the same model resource rather than a brand-new model — this is what keeps a lineage of `v1, v2, v3` behind one name. Aliases are mutable named pointers (`default`, `production`, `candidate`) that you move between versions; the SDK manages them with `model.add_version_aliases([...])` / `model.remove_version_aliases([...])`, which is the reliable path since the gcloud CLI has thin alias support. The common registry operations from the command line:

```bash
gcloud ai models upload \
  --region=us-central1 \
  --display-name=fraud-classifier \
  --artifact-uri=gs://myco-fraud-models/fraud/run-001/model \
  --container-image-uri=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/serve:v1 \
  --version-aliases=candidate

gcloud ai models list --region=us-central1
gcloud ai models describe MODEL_ID --region=us-central1
gcloud ai models list-versions MODEL_ID --region=us-central1
```

Registering to the model registry is the handoff point between training and serving: a registered model version is what gets deployed to a prediction endpoint. Versioning and aliases are what make safe rollouts and rollbacks possible.

## TensorBoard and monitoring

To watch loss curves and metrics *during* training rather than reading logs, attach a **Vertex AI TensorBoard** instance to the job. You create a managed TensorBoard resource once, then point training at it — the SDK's `job.run(..., tensorboard=<resource_name>, service_account=...)` streams your written event files to it live, and you view them in the console. A training job needs the TensorBoard resource name and a service account with `roles/aiplatform.user` to upload.

```bash
gcloud ai tensorboards create --region=us-central1 --display-name=fraud-tb
gcloud ai tensorboards list --region=us-central1
```

Once a model is deployed, **Vertex AI Model Monitoring** watches its live predictions for training-serving skew and feature/prediction drift and alerts when a distribution moves — the trigger that closes the retraining loop. That is a serving-side concern configured against the endpoint; it is covered in depth in module 13 (cost and monitoring). Feature Store, Experiments, and Metadata — the tracking layer that records each run's parameters and metrics — are covered in depth in module 17; reference it there rather than duplicating here.

## How this fits the whole solution

Training is the manufacturing stage of the end-to-end system. It consumes the data plane (a training set assembled in BigQuery, staged in Cloud Storage), runs on the accelerators from the compute module using containers from Artifact Registry, executes as a least-privilege service account per the security module, and is orchestrated by pipelines that gate on a frozen eval and register passing models. The registry then feeds the serving stage. Wrapped in CI/CD (Cloud Build or GitHub Actions authenticating via Workload Identity Federation), this becomes a repeatable train-evaluate-register loop — the heart of MLOps on Google Cloud.

## Key takeaways

- **Vertex custom training** runs your container or Python package on managed accelerators, mounts Cloud Storage via FUSE, and tears down on completion; use `from google.cloud import aiplatform` (current, not affected by the genai SDK deprecation). Run interruptible jobs on **Spot** with checkpointing.
- **Distributed training** is configured with multiple **worker pools**; use a **reduction server** for synchronous data-parallel GPU jobs, or multi-host **TPU** slices for the largest workloads.
- **Vertex AI Pipelines** (Kubeflow-based) orchestrates the ML workflow as compiled, cached, lineage-tracked `PipelineJob`s — gate registration on a **frozen-eval threshold**. Use **Cloud Composer** for general cross-service orchestration.
- The **Model Registry** versions models with aliases (`default`/`production`), forming the reproducible handoff from training to serving.

## CLI cheat-sheet

```bash
# --- Custom training jobs ---
gcloud ai custom-jobs create --region=us-central1 --display-name=fraud-train \
  --service-account=training-sa@myco-fraud-dev.iam.gserviceaccount.com --enable-web-access \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=REGION-docker.pkg.dev/PROJ/ml-images/train:v1 \
  --args=--epochs=20,--data=gs://myco-fraud-data/datasets/v1/
gcloud ai custom-jobs create --region=us-central1 --config=custom-job.yaml   # Spot/FLEX_START via scheduling block
gcloud ai custom-jobs list        --region=us-central1
gcloud ai custom-jobs describe    JOB_ID --region=us-central1
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
gcloud ai custom-jobs cancel      JOB_ID --region=us-central1

# Distributed: repeat --worker-pool-spec (1st=chief, rest=workers); reduction server pool image:
#   us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver  (n1-highcpu-16)

# --- Hyperparameter tuning (studySpec in study.yaml) ---
gcloud ai hp-tuning-jobs create --region=us-central1 --display-name=fraud-hpt \
  --config=study.yaml --max-trial-count=40 --parallel-trial-count=4
gcloud ai hp-tuning-jobs list --region=us-central1
gcloud ai hp-tuning-jobs describe JOB_ID --region=us-central1

# --- Model Registry ---
gcloud ai models upload --region=us-central1 --display-name=fraud-classifier \
  --artifact-uri=gs://myco-fraud-models/fraud/run-001/model \
  --container-image-uri=REGION-docker.pkg.dev/PROJ/ml-images/serve:v1 --version-aliases=candidate
gcloud ai models list          --region=us-central1
gcloud ai models describe      MODEL_ID --region=us-central1
gcloud ai models list-versions MODEL_ID --region=us-central1
# Aliases are managed via the SDK: model.add_version_aliases([...]) / remove_version_aliases([...])

# --- TensorBoard ---
gcloud ai tensorboards create --region=us-central1 --display-name=fraud-tb
gcloud ai tensorboards list   --region=us-central1

# --- Pipelines & scheduling are SDK-driven (aiplatform.PipelineJob / .create_schedule) ---
```

## Try it

Run a gated training pipeline:

1. Containerize a small training script, push it to Artifact Registry, and submit a `CustomContainerTrainingJob` on an A100 (or L4) — confirm it reads data from and writes a model to Cloud Storage.
2. Register the resulting model with `aiplatform.Model.upload` and assign it a `production` alias in the registry.
3. Write a two-component KFP pipeline (train → evaluate) with a `dsl.If` gate that only registers the model when the eval metric clears a threshold; compile and submit it as a `PipelineJob`.
4. Re-run the pipeline unchanged and observe **caching** skip the unchanged steps — then change a parameter and watch the affected steps re-execute.
