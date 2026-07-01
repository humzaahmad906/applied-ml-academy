# 09 — SageMaker AI: Training and Experiments

Amazon SageMaker AI is the managed ML platform at the center of most AWS ML systems. (The service was renamed from "Amazon SageMaker" to **Amazon SageMaker AI**, and it now sits inside the broader **Amazon SageMaker Unified Studio**, which brings data, analytics, and AI — including Amazon Bedrock — into one workspace.) This module focuses on the training half: the development environment, managed training jobs, how data reaches the job, distributed training at scale, and how to track experiments so your results are reproducible.

## The development environment: Studio

**SageMaker Studio** is the browser-based IDE for the whole ML lifecycle. The modern Studio experience offers **JupyterLab** for notebooks and a **Code Editor** based on VS Code for writing real code, both running on managed compute you can resize. (The earlier "Studio Classic" is the previous generation.) Studio is where you explore data, prototype models, launch training jobs, and manage experiments — but the key mental shift is that Studio is for *authoring*; the actual training runs on separate, ephemeral infrastructure via training jobs.

## Managed training jobs

A **training job** is SageMaker's core primitive: you specify a container image, an instance type and count, an IAM execution role, input data locations, and hyperparameters, and SageMaker provisions the instances, runs your code, uploads the model artifact to S3, and tears the instances down — you pay only for the seconds the job ran. The Python SDK's `Estimator` is the usual entry point:

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",
    role=role,
    framework_version="2.4",
    py_version="py311",
    instance_type="ml.g5.12xlarge",
    instance_count=1,
    hyperparameters={"epochs": 10, "lr": 3e-4},
    use_spot_instances=True,      # managed spot: up to ~90% cheaper
    max_run=7200,
    max_wait=10800,               # allow time for spot interruptions/retries
    checkpoint_s3_uri="s3://my-ml-data/checkpoints/",
    output_path="s3://my-ml-data/models/",
)
estimator.fit({"train": "s3://my-ml-data/train/",
               "validation": "s3://my-ml-data/val/"})
```

Two cost/efficiency features are worth calling out. **Managed spot training** (`use_spot_instances=True`) runs the job on Spot capacity for large savings; because SageMaker checkpoints to S3, an interrupted job resumes rather than restarts — set `max_wait` to allow retry time. **Warm pools** keep the instances from a finished job alive briefly so your next iteration starts in seconds instead of waiting for fresh provisioning — invaluable during rapid experimentation.

## Data channels and input modes

The dictionary passed to `fit()` defines **input channels** (`train`, `validation`, etc.), each mapping to an S3 location. How the data physically reaches the container is the **input mode**, and it materially affects performance:

- **File mode** (default): SageMaker downloads the whole channel to local disk before training starts. Simple; slow to start for large datasets.
- **FastFile mode**: exposes S3 data as a POSIX filesystem and streams objects on demand as your code reads them — training starts immediately, ideal for large datasets read mostly sequentially.
- **Pipe mode**: streams S3 data to the container as a Unix pipe, for frameworks that consume a stream.

For the largest distributed jobs, mount **FSx for Lustre** as the input, hydrated from S3 — it delivers the hundreds of GB/s of throughput needed to keep thousands of GPU cores fed without S3 request overhead dominating.

## Distributed training

SageMaker ships **distributed training libraries** for scaling beyond one GPU. **Data parallelism** replicates the model across GPUs/nodes and splits the batch, synchronizing gradients — the default for scaling most models. **Model parallelism** (tensor and pipeline) splits a model too large for one GPU across devices, essential for large foundation models. You set `instance_count` above 1 and enable the relevant distribution strategy; SageMaker configures the cluster networking (including EFA on capable instances).

For the largest, longest runs there is **SageMaker HyperPod**, purpose-built infrastructure for distributed training across thousands of accelerators. HyperPod adds resilience for jobs that run for days or weeks: it preconfigures the distributed libraries, supports **Spot instances** with continuous provisioning, and provides features like managed tiered checkpointing and cluster observability so a single node failure does not waste a multi-day run. Regular training jobs suit most work; HyperPod is for foundation-model-scale training where node failures are expected and must be survived automatically.

## Tracking experiments

Reproducibility requires tracking what you ran. SageMaker integrates **managed MLflow** so you can log parameters, metrics, and artifacts using the standard MLflow API against an AWS-managed tracking server — comparing runs, versioning models, and keeping a record without operating MLflow yourself.

```python
import mlflow
mlflow.set_tracking_uri(managed_mlflow_arn)
with mlflow.start_run():
    mlflow.log_params({"lr": 3e-4, "epochs": 10})
    mlflow.log_metric("val_accuracy", 0.94)
    mlflow.log_artifact("confusion_matrix.png")
```

Automatic model tuning (hyperparameter optimization) runs many training jobs across a search space and reports the best configuration, and it records every trial so you can see the whole search, not just the winner.

## How this fits the whole ML solution

Training is one stage in a longer chain, and SageMaker training is built to slot into it. It reads curated data and features from the data layer, writes a versioned artifact to S3, and hands that artifact to the model registry — which gates deployment. The execution role ties it to the security model, VPC mode keeps it on your private network, and managed spot plus warm pools tie it to the cost story. Treated in isolation a training job is just a script that ran; wired into the pipeline it is a reproducible, tracked step that produces a promotable model.

## Key takeaways

- SageMaker AI (inside Unified Studio) offers Studio with JupyterLab and a VS Code-based Code Editor for authoring; training runs on separate ephemeral instances.
- Training jobs provision, run, upload the artifact, and tear down — pay per second; use managed spot for savings and warm pools for fast iteration.
- Input mode matters: File mode (download-first), FastFile (stream on demand), Pipe (stream as pipe), or FSx for Lustre for extreme-throughput distributed jobs.
- Data parallelism scales most models; model parallelism splits models too big for one GPU; HyperPod adds resilience for foundation-model-scale runs.
- Track everything with managed MLflow and use automatic tuning for hyperparameter search.

## Try it

Write a `train.py` for a small model and launch it as a SageMaker training job with `use_spot_instances=True` and a `checkpoint_s3_uri`. Run it once in File mode and once in FastFile mode over a multi-GB dataset and compare job start time. Log parameters and metrics to managed MLflow, then launch a second job with different hyperparameters and compare the two runs in the MLflow UI. Confirm the model artifact landed in your `output_path` in S3 — that artifact is what the next stage of the pipeline will consume.
