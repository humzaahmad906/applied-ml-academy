# 09 — Azure Machine Learning: Training

Azure Machine Learning is the managed service that turns the raw primitives — compute, storage, identity, containers — into an ML platform. It gives you a **workspace** to organize everything, **compute** to run on, **jobs** to encapsulate runs, **environments** for reproducibility, **data assets** for versioned inputs, and a **model registry** for outputs, all with tracked lineage. This section covers the training half: how to define, run, scale, and track training on Azure ML using the `azure-ai-ml` Python SDK (v2). In the end-to-end solution, this is the factory that consumes curated data from the lake and produces registered, versioned models the deployment layer serves.

## The workspace: your ML control plane

The **workspace** is the top-level Azure ML resource. It ties together a default storage account (for datasets and artifacts), a Key Vault (for secrets), a container registry (for environment images), and Application Insights (for telemetry), and it is the scope for everything you author — compute, jobs, data assets, models, endpoints, pipelines. You interact with it through an `MLClient`:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="rg-mlx-dev",
    workspace_name="mlw-mlx-dev",
)
```

The rest of this module shows the `az ml` **CLI v2** (extension `ml`) alongside the SDK, because v2 is YAML-driven: you describe each asset — workspace, compute, environment, data, job — in a YAML spec and apply it with `--file`, which is exactly what lives in your Git repo and runs in CI/CD. Install it once with `az extension add -n ml`. Most create commands accept either inline flags (quick) or a `--file spec.yml` (reproducible); prefer the file for anything that ships.

```bash
# Create a workspace: inline for a quick dev workspace, or from a YAML spec for repeatable infra
az ml workspace create --name mlw-mlx-dev --resource-group rg-mlx-dev --location eastus2
az ml workspace create --file workspace.yml -g rg-mlx-dev            # reproducible, source-controlled
az ml workspace list   -g rg-mlx-dev -o table
az ml workspace show   -n mlw-mlx-dev -g rg-mlx-dev
az ml workspace update -n mlw-mlx-dev -g rg-mlx-dev --description "dev ML platform"
```

For a production workspace you enable the **managed VNet** in the spec, and `provision-network` materializes the private endpoints ahead of the first job (otherwise the first job pays the provisioning latency):

```yaml
# workspace.yml — managed-VNet workspace wired to the shared platform resources
$schema: https://azuremlschemas.azureedge.net/latest/workspace.schema.json
name: mlw-mlx-dev
location: eastus2
managed_network:
  isolation_mode: allow_internet_outbound   # or allow_only_approved_outbound for locked-down prod
tags:
  project: mlx
  env: dev
  owner: platform
```

```bash
az ml workspace provision-network -n mlw-mlx-dev -g rg-mlx-dev   # build the managed VNet + PEs up front
```

The workspace inherits the identity and networking posture from earlier: attach the shared managed identity, and for production enable the **managed VNet** so the workspace and its compute run isolated with private endpoints to storage, vault, and registry. A useful convenience: set the default workspace/group once with `az configure --defaults workspace=mlw-mlx-dev group=rg-mlx-dev` so you can drop the `-w`/`-g` flags on every subsequent `az ml` call.

## Compute: clusters, instances, and serverless

Azure ML gives you three ways to get compute, each an abstraction over the VM layer:

- **Compute cluster (`AmlCompute`)** — an autoscaling pool of VMs (a managed scale set) that runs your training jobs, scales to zero when idle, and supports the low-priority (spot) tier. This is the workhorse for training.
- **Compute instance** — a single, personal managed VM for interactive development and notebooks.
- **Serverless compute** — you specify a size per job and Azure ML provisions and tears down capacity automatically, with no cluster to manage. Convenient when you do not want to own a cluster.

```python
from azure.ai.ml.entities import AmlCompute

gpu_cluster = AmlCompute(
    name="gpu-cluster",
    size="Standard_NC24ads_A100_v4",
    min_instances=0,          # scale to zero when idle — pay nothing between jobs
    max_instances=4,
    tier="LowPriority",       # spot pricing for fault-tolerant training
    idle_time_before_scale_down=1800,
)
ml_client.compute.begin_create_or_update(gpu_cluster).result()
```

`min_instances=0` plus the LowPriority tier is the cost pattern from the compute section, expressed at the ML layer.

The same cluster is created from the CLI with a YAML spec — this is the form that lives in infrastructure-as-code. A **compute instance** (personal dev VM) is a different `--type`:

```yaml
# gpu-cluster.yml — autoscaling, scale-to-zero, spot-priced training cluster
$schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json
name: gpu-cluster
type: amlcompute
size: Standard_NC24ads_A100_v4
min_instances: 0            # scale to zero when idle
max_instances: 4
tier: low_priority          # spot pricing
idle_time_before_scale_down: 1800
```

```bash
az ml compute create --file gpu-cluster.yml                          # the cluster
az ml compute create --name dev-box --type computeinstance --size Standard_DS11_v2   # personal dev VM
az ml compute list        -o table
az ml compute show        --name gpu-cluster
az ml compute update      --name gpu-cluster --min-instances 0 --max-instances 8     # widen bounds
az ml compute stop        --name dev-box                              # instances: stop when not working
az ml compute delete      --name gpu-cluster --yes
```

Stopping (not just idling) a compute instance is the cost step people miss — a compute *cluster* scales itself to zero, but a compute *instance* keeps billing until you `az ml compute stop` it.

## Environments and data assets: reproducibility

A **job's environment** is a container image plus dependencies, defined once and reused so every run is byte-identical. You build from a curated base image and a conda/pip file; Azure ML materializes and caches it in the workspace registry.

```python
from azure.ai.ml.entities import Environment

env = Environment(
    name="train-pytorch",
    version="2",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.4-cuda12.4:latest",
    conda_file="conda.yml",   # your pinned deps
)
ml_client.environments.create_or_update(env)
```

From the CLI the environment is a YAML spec applied with `--file` — the canonical, source-controlled form:

```yaml
# env.yml — curated PyTorch base image + pinned conda deps
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: train-pytorch
version: "2"
image: mcr.microsoft.com/azureml/curated/acpt-pytorch-2.4-cuda12.4:latest
conda_file: conda.yml
```

```bash
az ml environment create --file env.yml
az ml environment list -o table
az ml environment show --name train-pytorch --version 2
```

A **data asset** is a versioned pointer to data in a datastore (from the storage section). Referencing a versioned asset in a job is what makes training reproducible — the run records exactly which data version it consumed. You register, list, inspect, and *archive* data assets from the CLI; archiving hides an old version from `list` without deleting it, so lineage to past runs stays intact:

```yaml
# data.yml — a versioned folder asset pointing at a gold table in the lake
$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: fraud-train
version: "3"
type: uri_folder
path: azureml://datastores/stmlxdata/paths/gold/fraud/train/
```

```bash
az ml data create  --file data.yml
az ml data create  --name fraud-train --version 4 --type uri_folder --path azureml://datastores/stmlxdata/paths/gold/fraud/train/
az ml data list    --name fraud-train -o table       # all versions
az ml data show    --name fraud-train --version 3
az ml data archive --name fraud-train --version 1    # hide old version; still usable, keeps lineage
```

## Jobs: the unit of a training run

A **command job** wraps a command run against a data input, on a compute target, in an environment. It captures code, inputs, outputs, logs, metrics, and the resulting model — the full lineage of one run.

```python
from azure.ai.ml import command, Input, Output
from azure.ai.ml.constants import AssetTypes

job = command(
    code="./src",                                  # snapshotted with the run
    command=("python train.py --data ${{inputs.train}} "
             "--epochs 10 --out ${{outputs.model}}"),
    inputs={"train": Input(type=AssetTypes.URI_FOLDER,
                           path="azureml:fraud-train:3")},   # versioned data asset
    outputs={"model": Output(type=AssetTypes.URI_FOLDER)},
    environment="azureml:train-pytorch:2",
    compute="gpu-cluster",
    display_name="fraud-train-run",
    experiment_name="fraud-detection",
)
returned = ml_client.jobs.create_or_update(job)
print(returned.studio_url)   # live logs and metrics in the studio
```

The same command job expressed as a YAML spec is what you submit from CI/CD with `az ml job create --file`. The `${{inputs.*}}` / `${{outputs.*}}` substitution and `azureml:` asset references are identical to the SDK form:

```yaml
# train-job.yml — a command job pinned to versioned data, env, and compute
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: ./src
command: >-
  python train.py --data ${{inputs.train}} --epochs 10 --out ${{outputs.model}}
inputs:
  train:
    type: uri_folder
    path: azureml:fraud-train:3          # versioned data asset
outputs:
  model:
    type: uri_folder
environment: azureml:train-pytorch:2
compute: azureml:gpu-cluster
display_name: fraud-train-run
experiment_name: fraud-detection
```

```bash
az ml job create --file train-job.yml --stream   # submit and tail logs in one shot
az ml job list   --query "[?status=='Running'].name" -o tsv
az ml job show   --name <job-name>
az ml job stream --name <job-name>                # attach to a running job's logs
az ml job cancel --name <job-name>
az ml job download --name <job-name> --output-name model --download-path ./artifacts  # pull outputs/logs
```

`--stream` blocks and prints logs, which is what you want in a pipeline step; without it the command returns immediately and you attach later with `az ml job stream`. Inside `train.py`, use **MLflow** (Azure ML is MLflow-native) to log parameters, metrics, and artifacts — they surface automatically in the workspace, and MLflow autologging captures most of it with one line:

```python
import mlflow
mlflow.autolog()             # logs params, metrics, and the model automatically
# ... your training loop ...
mlflow.log_metric("val_auc", 0.981)
```

Reproducibility habits carry over from good ML practice: set seeds on the framework, NumPy, and Python; split before augmentation; and let the versioned data asset pin the exact inputs so a run is fully reconstructable.

## Distributed training: scaling across GPUs and nodes

For models too big or slow for one GPU, Azure ML runs **distributed jobs** across a cluster. You declare `instance_count` (nodes) and a **distribution** strategy, and Azure ML sets the environment variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`) your framework needs. For PyTorch DDP, use `PyTorchDistribution`:

```python
from azure.ai.ml import command
from azure.ai.ml.entities import PyTorchDistribution

dist_job = command(
    code="./src",
    command="python -m torch.distributed.run train_ddp.py --data ${{inputs.train}}",
    inputs={"train": Input(type=AssetTypes.URI_FOLDER, path="azureml:fraud-train:3")},
    environment="azureml:train-pytorch:2",
    compute="gpu-cluster",
    instance_count=2,                                   # 2 nodes
    distribution=PyTorchDistribution(process_count_per_instance=4),  # 4 GPUs/node
)
ml_client.jobs.create_or_update(dist_job)
```

As a YAML spec, the distribution and node count are declarative fields — this is the form you scale in production:

```yaml
# ddp-job.yml — 2 nodes x 4 GPUs, PyTorch DDP
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
code: ./src
command: python -m torch.distributed.run train_ddp.py --data ${{inputs.train}}
inputs:
  train: { type: uri_folder, path: azureml:fraud-train:3 }
environment: azureml:train-pytorch:2
compute: azureml:gpu-cluster
resources:
  instance_count: 2                    # nodes
distribution:
  type: pytorch
  process_count_per_instance: 4        # GPUs per node
```

```bash
az ml job create --file ddp-job.yml --stream
```

For genuine multi-node scale you want an **ND-series** cluster with InfiniBand/RDMA so gradient all-reduce is not bottlenecked by the network — recall the SKU letters from the compute section. Azure ML also supports MPI and TensorFlow distributions, and for very large jobs integrates with DeepSpeed and similar libraries through the environment. Use **bf16** mixed precision on H100/H200-class GPUs by default; drop to fp16 only under memory pressure.

## Sweep jobs: hyperparameter tuning

To search hyperparameters, wrap a command job in a **sweep job**: you declare a `search_space`, a `sampling_algorithm` (random / grid / bayesian), an `objective` metric to optimize (the name must match what you `mlflow.log_metric`), and an early-termination policy so unpromising trials are killed to save GPU hours. Azure ML fans the trials across the cluster and tracks them as child runs under one parent.

```yaml
# sweep-job.yml — Bayesian search maximizing val_auc, killing laggards early
$schema: https://azuremlschemas.azureedge.net/latest/sweepJob.schema.json
type: sweep
trial:
  code: ./src
  command: >-
    python train.py --data ${{inputs.train}} --lr ${{search_space.lr}} --batch ${{search_space.batch}}
  environment: azureml:train-pytorch:2
inputs:
  train: { type: uri_folder, path: azureml:fraud-train:3 }
compute: azureml:gpu-cluster
sampling_algorithm: bayesian
search_space:
  lr:    { type: loguniform, min_value: -7, max_value: -2 }
  batch: { type: choice, values: [32, 64, 128] }
objective:
  primary_metric: val_auc
  goal: maximize
early_termination:
  type: bandit
  slack_factor: 0.15
  evaluation_interval: 2
limits:
  max_total_trials: 40
  max_concurrent_trials: 4
```

```bash
az ml job create --file sweep-job.yml --stream   # parent run; each trial is a child run
```

## Pipelines: multi-step training DAGs

Real training is rarely one command. A **pipeline** composes reusable **components** (each a containerized step with typed inputs/outputs) into a DAG — for example: validate data → engineer features → train → evaluate → register. Azure ML caches step outputs, so unchanged upstream steps are skipped on reruns, and the whole DAG can be scheduled or triggered.

```python
from azure.ai.ml import dsl, load_component

prep = load_component(source="components/prep.yml")
train = load_component(source="components/train.yml")
evaluate = load_component(source="components/eval.yml")

@dsl.pipeline(compute="gpu-cluster", experiment_name="fraud-detection")
def fraud_pipeline(raw_data):
    p = prep(data=raw_data)
    t = train(train_data=p.outputs.clean)
    evaluate(model=t.outputs.model, test_data=p.outputs.test)
    return {"model": t.outputs.model}

pl = fraud_pipeline(raw_data=Input(path="azureml:fraud-raw:5"))
ml_client.jobs.create_or_update(pl)
```

A **component** is itself a YAML spec (a reusable, typed, containerized step), registered once and reused across pipelines; the pipeline is another YAML spec that wires components into a DAG:

```yaml
# components/train.yml — a reusable typed step
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: train_fraud
version: "1"
type: command
inputs:
  train_data: { type: uri_folder }
outputs:
  model: { type: uri_folder }
code: ./src
command: python train.py --data ${{inputs.train_data}} --out ${{outputs.model}}
environment: azureml:train-pytorch:2
```

```bash
az ml component create --file components/train.yml
az ml component list -o table
az ml component show --name train_fraud --version 1

# Submit a whole pipeline defined in YAML (jobs: block referencing components) and manage it as a job
az ml job create --file pipeline.yml --stream
az ml job show   --name <pipeline-run-name>
```

The Azure ML pipeline owns the *training* DAG; Data Factory owns the *data-movement* DAG (from the data-services section) — keep that boundary clean.

## The model registry: versioned outputs and lineage

A trained model becomes a first-class, versioned asset in the **model registry**, complete with lineage back to the job, data, and environment that produced it. Registration is the handoff from training to deployment.

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = Model(
    path=f"azureml://jobs/{returned.name}/outputs/model",   # output of the run
    name="fraud-detector",
    type=AssetTypes.MLFLOW_MODEL,
    description="Gradient-boosted fraud model, val_auc=0.981",
)
registered = ml_client.models.create_or_update(model)   # -> fraud-detector:1
```

From the CLI you register a model straight from a finished job's output — the `azureml://jobs/.../outputs/model` path is what preserves lineage back to the run:

```bash
# Register from a job output (keeps lineage) — inline or from a YAML spec
az ml model create --name fraud-detector --version 1 --type mlflow_model \
  --path azureml://jobs/<job-name>/outputs/model
az ml model create --file model.yml
az ml model list  --name fraud-detector -o table       # all versions
az ml model show  --name fraud-detector --version 1
```

A **registry** (distinct from the per-workspace registry) can be shared across workspaces so you promote a model trained in a dev workspace into a prod workspace — the backbone of a promotion-based MLOps flow. You target a registry instead of a workspace by swapping `--workspace-name` for `--registry-name`, which is how a CI/CD step copies the winning version from dev to prod:

```bash
az ml model create --file model.yml --registry-name reg-mlx-shared    # promote into the shared registry
az ml model list --registry-name reg-mlx-shared --name fraud-detector -o table
```

Tags and stages let you mark which version is a candidate versus production.

## How training fits the whole solution

This is the middle of the pipeline. Curated **gold** datasets from the lake become versioned **data assets**. A **pipeline** of components runs on an autoscaling **compute cluster** (LowPriority, scale-to-zero), distributed across ND nodes when the model demands it, tracked with MLflow. The output is a **registered, versioned model** with full lineage. CI/CD (a later topic) triggers this pipeline on data or code changes; a shared **registry** promotes the winning model from dev to prod; and the deployment layer picks it up from there. Because every input is versioned and every run is tracked, you can always answer "what data and code produced this exact model" — the property that makes an ML system auditable and reproducible rather than a pile of one-off notebooks.

## Key takeaways

- The **workspace** is the ML control plane, tying storage, Key Vault, registry, and telemetry together; drive it with `MLClient` + `DefaultAzureCredential`.
- Train on **autoscaling compute clusters** (`min_instances=0`, LowPriority tier) for cost; use **serverless** to skip cluster management and **compute instances** for interactive dev.
- Make runs reproducible with **versioned environments** (container images) and **versioned data assets**; log everything with **MLflow** (autolog).
- Scale with **distributed jobs** (`instance_count` + `PyTorchDistribution`) on **ND** InfiniBand clusters; default to **bf16** on H100/H200.
- Compose multi-step **pipelines** of reusable components (the training DAG), and register outputs to the **model registry** with lineage — the handoff to deployment and the basis of dev→prod promotion.

## CLI cheat-sheet

```bash
# --- Setup ---
az extension add -n ml                                          # the CLI v2 extension
az configure --defaults workspace=mlw-mlx-dev group=rg-mlx-dev  # drop -w/-g on every call

# --- Workspace ---
az ml workspace create --file workspace.yml -g rg-mlx-dev
az ml workspace create --name mlw-mlx-dev -g rg-mlx-dev --location eastus2
az ml workspace list   -g rg-mlx-dev -o table
az ml workspace show   -n mlw-mlx-dev -g rg-mlx-dev
az ml workspace update -n mlw-mlx-dev -g rg-mlx-dev --description "dev ML platform"
az ml workspace provision-network -n mlw-mlx-dev -g rg-mlx-dev   # build managed VNet + PEs up front

# --- Compute (cluster + instance) ---
az ml compute create --file gpu-cluster.yml
az ml compute create --name dev-box --type computeinstance --size Standard_DS11_v2
az ml compute list   -o table
az ml compute show   --name gpu-cluster
az ml compute update --name gpu-cluster --min-instances 0 --max-instances 8
az ml compute stop   --name dev-box                            # instances bill until stopped
az ml compute delete --name gpu-cluster --yes

# --- Environments ---
az ml environment create --file env.yml
az ml environment list -o table
az ml environment show --name train-pytorch --version 2

# --- Data assets ---
az ml data create  --file data.yml
az ml data list    --name fraud-train -o table
az ml data show    --name fraud-train --version 3
az ml data archive --name fraud-train --version 1

# --- Jobs (command / sweep / distributed / pipeline all via YAML) ---
az ml job create   --file train-job.yml --stream
az ml job list     --query "[?status=='Running'].name" -o tsv
az ml job show     --name <job-name>
az ml job stream   --name <job-name>
az ml job cancel   --name <job-name>
az ml job download --name <job-name> --output-name model --download-path ./artifacts

# --- Components (reusable pipeline steps) ---
az ml component create --file components/train.yml
az ml component list -o table
az ml component show --name train_fraud --version 1

# --- Model registry (workspace and shared registry) ---
az ml model create --name fraud-detector --version 1 --type mlflow_model --path azureml://jobs/<job-name>/outputs/model
az ml model create --file model.yml --registry-name reg-mlx-shared     # promote dev -> prod
az ml model list   --name fraud-detector -o table
az ml model show   --name fraud-detector --version 1
```

## Try it

Create a workspace and a scale-to-zero GPU compute cluster on an NC A100 SKU. Register a small versioned data asset pointing at a folder in your lake, define a curated-image environment, and submit a `command` job that trains a tiny model on that data with `mlflow.autolog()` enabled. Open the returned `studio_url`, watch the metrics stream, then register the resulting model to the registry and confirm you can trace it back to the job, environment, and data version that produced it.
