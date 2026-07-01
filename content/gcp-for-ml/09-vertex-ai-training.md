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

The equivalent from the command line, which is what CI/CD often uses:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=fraud-train \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/train:v1
```

For cost, remember the compute lessons: run interruptible training on **Spot** capacity (Vertex supports it) with frequent checkpointing to Cloud Storage so a preemption resumes rather than restarts.

## Distributed training

When a model or dataset outgrows a single machine, you scale to **distributed training** by defining multiple **worker pools**. A typical multi-worker GPU job has a primary pool and additional worker pools, all the same machine type, and your framework (PyTorch DDP, JAX, or TensorFlow's distribution strategies) handles the collective communication (NCCL over the internal network) across them. For synchronous data-parallel training, Vertex AI also offers a **reduction server** — dedicated CPU nodes that aggregate gradients to reduce communication overhead and speed up large multi-GPU jobs. For very large workloads you can target multi-host **TPU** slices (Trillium/v6e, Ironwood) instead of GPUs. The key configuration is still worker pool specs: how many replicas, what accelerators, and which container each pool runs.

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

## The Model Registry

Trained models land in the **Vertex AI Model Registry** — a versioned catalog of your models. Each model has **versions**, and you can assign **aliases** (like `default` or `production`) that point at a specific version, so downstream deployment references a stable alias while you iterate on new versions behind it.

```python
model = aiplatform.Model.upload(
    display_name="fraud-classifier",
    artifact_uri="gs://myco-fraud-models/fraud/run-001/model",
    serving_container_image_uri=(
        "us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/serve:v1"),
)
```

Registering to the model registry is the handoff point between training and serving: a registered model version is what gets deployed to a prediction endpoint. Versioning and aliases are what make safe rollouts and rollbacks possible.

## How this fits the whole solution

Training is the manufacturing stage of the end-to-end system. It consumes the data plane (a training set assembled in BigQuery, staged in Cloud Storage), runs on the accelerators from the compute module using containers from Artifact Registry, executes as a least-privilege service account per the security module, and is orchestrated by pipelines that gate on a frozen eval and register passing models. The registry then feeds the serving stage. Wrapped in CI/CD (Cloud Build or GitHub Actions authenticating via Workload Identity Federation), this becomes a repeatable train-evaluate-register loop — the heart of MLOps on Google Cloud.

## Key takeaways

- **Vertex custom training** runs your container or Python package on managed accelerators, mounts Cloud Storage via FUSE, and tears down on completion; use `from google.cloud import aiplatform` (current, not affected by the genai SDK deprecation). Run interruptible jobs on **Spot** with checkpointing.
- **Distributed training** is configured with multiple **worker pools**; use a **reduction server** for synchronous data-parallel GPU jobs, or multi-host **TPU** slices for the largest workloads.
- **Vertex AI Pipelines** (Kubeflow-based) orchestrates the ML workflow as compiled, cached, lineage-tracked `PipelineJob`s — gate registration on a **frozen-eval threshold**. Use **Cloud Composer** for general cross-service orchestration.
- The **Model Registry** versions models with aliases (`default`/`production`), forming the reproducible handoff from training to serving.

## Try it

Run a gated training pipeline:

1. Containerize a small training script, push it to Artifact Registry, and submit a `CustomContainerTrainingJob` on an A100 (or L4) — confirm it reads data from and writes a model to Cloud Storage.
2. Register the resulting model with `aiplatform.Model.upload` and assign it a `production` alias in the registry.
3. Write a two-component KFP pipeline (train → evaluate) with a `dsl.If` gate that only registers the model when the eval metric clears a threshold; compile and submit it as a `PipelineJob`.
4. Re-run the pipeline unchanged and observe **caching** skip the unchanged steps — then change a parameter and watch the affected steps re-execute.
