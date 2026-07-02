# 09 — SageMaker AI: Training and Experiments

Amazon SageMaker AI is the managed ML platform at the center of most AWS ML systems. (The service was renamed from "Amazon SageMaker" to **Amazon SageMaker AI**, and it now sits inside the broader **Amazon SageMaker Unified Studio**, which brings data, analytics, and AI — including Amazon Bedrock — into one workspace.) This module focuses on the training half: the development environment, managed training jobs, how data reaches the job, distributed training at scale, and how to track experiments so your results are reproducible.

## The development environment: Studio

**SageMaker Studio** is the browser-based IDE for the whole ML lifecycle. The modern Studio experience offers **JupyterLab** for notebooks and a **Code Editor** based on VS Code for writing real code, both running on managed compute you can resize. (The earlier "Studio Classic" is the previous generation.) Studio is where you explore data, prototype models, launch training jobs, and manage experiments — but the key mental shift is that Studio is for *authoring*; the actual training runs on separate, ephemeral infrastructure via training jobs.

Studio is organized as a **domain** (one per account/region for most teams, tied to a VPC and an execution role) containing **user profiles** (one per person, each with its own home directory and role). You will usually create these once via the console or IaC, but the CLI is what a platform team scripts:

```bash
# One domain per team, then a user profile per data scientist
aws sagemaker create-domain \
  --domain-name ml-team \
  --auth-mode IAM \
  --vpc-id vpc-123 --subnet-ids subnet-aaa subnet-bbb \
  --default-user-settings ExecutionRole=arn:aws:iam::123456789012:role/SageMakerStudioRole

aws sagemaker create-user-profile \
  --domain-id d-abc123 --user-profile-name alice \
  --user-settings ExecutionRole=arn:aws:iam::123456789012:role/SageMakerStudioRole
```

The gotcha here is cost leakage: Studio apps (the kernels behind a notebook) keep billing until they are stopped, not when you close the browser tab. Set a lifecycle-config auto-shutdown or check running apps with `aws sagemaker list-apps` before leaving for the day.

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

In the SDK, warm pools are `keep_alive_period_in_seconds` on the `Estimator`; under the hood it maps to `ResourceConfig.KeepAlivePeriodInSeconds` (capped at 3600 s). One gotcha the SDK hides: **warm pools and managed spot are mutually exclusive** — a warm pool holds dedicated capacity, which spot by definition does not give you, so you pick one per job. Warm pools also have their own account resource quota you may have to request an increase for before they take effect.

```python
estimator = PyTorch(
    ..., keep_alive_period_in_seconds=1800,   # reuse instances for 30 min of iteration
    use_spot_instances=False,                 # cannot combine with warm pools
)
```

Everything the SDK does is ultimately a `CreateTrainingJob` call, and it is worth seeing the raw shape once — CI/CD systems and Step Functions pipelines invoke it directly, and error messages reference these field names. The same job launched from the CLI:

```bash
# Managed spot job with a stopping condition and checkpointing (raw API shape)
aws sagemaker create-training-job \
  --training-job-name fraud-2026-07-02 \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole \
  --algorithm-specification TrainingImage=<ecr-image>,TrainingInputMode=File \
  --resource-config InstanceType=ml.g5.12xlarge,InstanceCount=1,VolumeSizeInGB=200 \
  --input-data-config '[{"ChannelName":"train","DataSource":{"S3DataSource":{"S3Uri":"s3://my-ml-data/train/","S3DataType":"S3Prefix"}}}]' \
  --output-data-config S3OutputPath=s3://my-ml-data/models/ \
  --checkpoint-config S3Uri=s3://my-ml-data/checkpoints/ \
  --enable-managed-spot-training \
  --stopping-condition MaxRuntimeInSeconds=7200,MaxWaitTimeInSeconds=10800

aws sagemaker describe-training-job --training-job-name fraud-2026-07-02 \
  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,Saving:BillableTimeInSeconds}'
aws sagemaker stop-training-job --training-job-name fraud-2026-07-02
```

`--enable-managed-spot-training` **requires** a `MaxWaitTimeInSeconds` that is ≥ `MaxRuntimeInSeconds` (the API rejects the job otherwise), and the eventual `BillableTimeInSeconds` reported by `describe-training-job` versus wall-clock is how you confirm the spot discount actually landed.

## Data channels and input modes

The dictionary passed to `fit()` defines **input channels** (`train`, `validation`, etc.), each mapping to an S3 location. How the data physically reaches the container is the **input mode**, and it materially affects performance:

- **File mode** (default): SageMaker downloads the whole channel to local disk before training starts. Simple; slow to start for large datasets.
- **FastFile mode**: exposes S3 data as a POSIX filesystem and streams objects on demand as your code reads them — training starts immediately, ideal for large datasets read mostly sequentially.
- **Pipe mode**: streams S3 data to the container as a Unix pipe, for frameworks that consume a stream.

For the largest distributed jobs, mount **FSx for Lustre** as the input, hydrated from S3 — it delivers the hundreds of GB/s of throughput needed to keep thousands of GPU cores fed without S3 request overhead dominating.

In the SDK the input mode is one argument on the channel, so switching is a one-line experiment (exactly the comparison the *Try it* asks for). The gotcha with FastFile is that it streams lazily: a random-access pattern (shuffling a giant sharded dataset with tiny reads) can be *slower* than File mode's upfront download, so it wins for large, mostly-sequential reads and loses for chatty random I/O.

```python
from sagemaker.inputs import TrainingInput

estimator.fit({
    "train": TrainingInput("s3://my-ml-data/train/", input_mode="FastFile"),
    "validation": TrainingInput("s3://my-ml-data/val/", input_mode="File"),
})
```

Two more channel details matter at scale. `S3DataDistributionType=ShardedByS3Key` splits the objects in a channel *across* instances (each node trains on a disjoint shard) rather than the default `FullyReplicated` (every node gets the whole channel) — you almost always want sharded for data-parallel training so nodes are not all reading the same files. And when mounting FSx, the filesystem must live in the **same subnet/AZ** as the training instances or the job cannot mount it.

## Distributed training

SageMaker ships **distributed training libraries** for scaling beyond one GPU. **Data parallelism** replicates the model across GPUs/nodes and splits the batch, synchronizing gradients — the default for scaling most models. **Model parallelism** (tensor and pipeline) splits a model too large for one GPU across devices, essential for large foundation models. You set `instance_count` above 1 and enable the relevant distribution strategy; SageMaker configures the cluster networking (including EFA on capable instances).

The distribution strategy is a dict on the `Estimator`. For the SageMaker data-parallel library (SMDDP) you pass `distribution={"smdistributed": {"dataparallel": {"enabled": True}}}`; for plain PyTorch DDP or `torchrun`, `distribution={"torch_distributed": {"enabled": True}}` — SageMaker sets the rank/world-size environment variables and launches one process per GPU for you. The practitioner trap is picking multi-node before saturating a single node: prefer one `ml.p5.48xlarge` (8 GPUs, NVLink) over several smaller instances, because intra-node NVLink is far faster than inter-node network, and only cross the node boundary when one box is genuinely full.

```python
estimator = PyTorch(
    ..., instance_type="ml.p4d.24xlarge", instance_count=4,
    distribution={"torch_distributed": {"enabled": True}},   # one process per GPU
)
```

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

Automatic model tuning (hyperparameter optimization) runs many training jobs across a search space and reports the best configuration, and it records every trial so you can see the whole search, not just the winner. In the SDK it is a `HyperparameterTuner` wrapping the estimator; you give it a metric to optimize (parsed from the job's logs via a regex), a search space, and a job budget, and it launches the fleet.

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

tuner = HyperparameterTuner(
    estimator,
    objective_metric_name="val_accuracy",
    metric_definitions=[{"Name": "val_accuracy", "Regex": "val_accuracy=([0-9\\.]+)"}],
    hyperparameter_ranges={"lr": ContinuousParameter(1e-5, 1e-3),
                           "epochs": IntegerParameter(5, 20)},
    strategy="Bayesian",
    max_jobs=20, max_parallel_jobs=4,   # parallelism trades wall-clock for Bayesian signal
)
tuner.fit({"train": "s3://my-ml-data/train/", "validation": "s3://my-ml-data/val/"})
```

The knob that bites people is `max_parallel_jobs`: Bayesian search learns from completed trials, so running everything in parallel throws away that signal and degenerates toward random search — keep parallelism modest relative to `max_jobs`. The raw API is `aws sagemaker create-hyper-parameter-tuning-job`, and `describe-hyper-parameter-tuning-job` reports the best training job when the search finishes.

## How this fits the whole ML solution

Training is one stage in a longer chain, and SageMaker training is built to slot into it. It reads curated data and features from the data layer, writes a versioned artifact to S3, and hands that artifact to the model registry — which gates deployment. The execution role ties it to the security model, VPC mode keeps it on your private network, and managed spot plus warm pools tie it to the cost story. Treated in isolation a training job is just a script that ran; wired into the pipeline it is a reproducible, tracked step that produces a promotable model.

## Key takeaways

- SageMaker AI (inside Unified Studio) offers Studio with JupyterLab and a VS Code-based Code Editor for authoring; training runs on separate ephemeral instances.
- Training jobs provision, run, upload the artifact, and tear down — pay per second; use managed spot for savings and warm pools for fast iteration.
- Input mode matters: File mode (download-first), FastFile (stream on demand), Pipe (stream as pipe), or FSx for Lustre for extreme-throughput distributed jobs.
- Data parallelism scales most models; model parallelism splits models too big for one GPU; HyperPod adds resilience for foundation-model-scale runs.
- Track everything with managed MLflow and use automatic tuning for hyperparameter search.

## CLI cheat-sheet

```bash
# --- Studio setup (platform team, usually once) ---
aws sagemaker create-domain --domain-name ml-team --auth-mode IAM \
  --vpc-id vpc-123 --subnet-ids subnet-aaa subnet-bbb \
  --default-user-settings ExecutionRole=<studio-role>
aws sagemaker create-user-profile --domain-id d-abc123 --user-profile-name alice \
  --user-settings ExecutionRole=<studio-role>
aws sagemaker list-apps                      # find running (billing) kernels

# --- Training jobs ---
aws sagemaker create-training-job \
  --training-job-name fraud-2026-07-02 --role-arn <role> \
  --algorithm-specification TrainingImage=<ecr-image>,TrainingInputMode=File \
  --resource-config InstanceType=ml.g5.12xlarge,InstanceCount=1,VolumeSizeInGB=200 \
  --input-data-config file://channels.json \
  --output-data-config S3OutputPath=s3://my-ml-data/models/ \
  --checkpoint-config S3Uri=s3://my-ml-data/checkpoints/ \
  --stopping-condition MaxRuntimeInSeconds=7200
# managed spot: add --enable-managed-spot-training and set MaxWaitTimeInSeconds >= MaxRuntimeInSeconds
# warm pool:    add KeepAlivePeriodInSeconds=1800 to --resource-config (NOT with spot)

aws sagemaker describe-training-job --training-job-name fraud-2026-07-02 \
  --query '{Status:TrainingJobStatus,Billable:BillableTimeInSeconds}'
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending --max-results 10
aws sagemaker stop-training-job --training-job-name fraud-2026-07-02

# --- Hyperparameter tuning ---
aws sagemaker create-hyper-parameter-tuning-job \
  --hyper-parameter-tuning-job-name fraud-hpo \
  --hyper-parameter-tuning-job-config file://hpo-config.json \
  --training-job-definition file://train-def.json
aws sagemaker describe-hyper-parameter-tuning-job --hyper-parameter-tuning-job-name fraud-hpo \
  --query 'BestTrainingJob.TrainingJobName'
aws sagemaker list-training-jobs-for-hyper-parameter-tuning-job \
  --hyper-parameter-tuning-job-name fraud-hpo

# --- Channel config knobs (in --input-data-config JSON) ---
# TrainingInputMode: File | FastFile | Pipe
# S3DataDistributionType: FullyReplicated (default) | ShardedByS3Key (data-parallel)
```

## Try it

Write a `train.py` for a small model and launch it as a SageMaker training job with `use_spot_instances=True` and a `checkpoint_s3_uri`. Run it once in File mode and once in FastFile mode over a multi-GB dataset and compare job start time. Log parameters and metrics to managed MLflow, then launch a second job with different hyperparameters and compare the two runs in the MLflow UI. Confirm the model artifact landed in your `output_path` in S3 — that artifact is what the next stage of the pipeline will consume.
