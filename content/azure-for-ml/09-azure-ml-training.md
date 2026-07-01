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

```bash
# Create a workspace from the CLI
az ml workspace create --name mlw-mlx-dev --resource-group rg-mlx-dev --location eastus2
```

The workspace inherits the identity and networking posture from earlier: attach the shared managed identity, and for production enable the **managed VNet** so the workspace and its compute run isolated with private endpoints to storage, vault, and registry.

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

A **data asset** is a versioned pointer to data in a datastore (from the storage section). Referencing a versioned asset in a job is what makes training reproducible — the run records exactly which data version it consumed.

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

Inside `train.py`, use **MLflow** (Azure ML is MLflow-native) to log parameters, metrics, and artifacts — they surface automatically in the workspace, and MLflow autologging captures most of it with one line:

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

For genuine multi-node scale you want an **ND-series** cluster with InfiniBand/RDMA so gradient all-reduce is not bottlenecked by the network — recall the SKU letters from the compute section. Azure ML also supports MPI and TensorFlow distributions, and for very large jobs integrates with DeepSpeed and similar libraries through the environment. Use **bf16** mixed precision on H100/H200-class GPUs by default; drop to fp16 only under memory pressure.

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

A **registry** (distinct from the per-workspace registry) can be shared across workspaces so you promote a model trained in a dev workspace into a prod workspace — the backbone of a promotion-based MLOps flow. Tags and stages let you mark which version is a candidate versus production.

## How training fits the whole solution

This is the middle of the pipeline. Curated **gold** datasets from the lake become versioned **data assets**. A **pipeline** of components runs on an autoscaling **compute cluster** (LowPriority, scale-to-zero), distributed across ND nodes when the model demands it, tracked with MLflow. The output is a **registered, versioned model** with full lineage. CI/CD (a later topic) triggers this pipeline on data or code changes; a shared **registry** promotes the winning model from dev to prod; and the deployment layer picks it up from there. Because every input is versioned and every run is tracked, you can always answer "what data and code produced this exact model" — the property that makes an ML system auditable and reproducible rather than a pile of one-off notebooks.

## Key takeaways

- The **workspace** is the ML control plane, tying storage, Key Vault, registry, and telemetry together; drive it with `MLClient` + `DefaultAzureCredential`.
- Train on **autoscaling compute clusters** (`min_instances=0`, LowPriority tier) for cost; use **serverless** to skip cluster management and **compute instances** for interactive dev.
- Make runs reproducible with **versioned environments** (container images) and **versioned data assets**; log everything with **MLflow** (autolog).
- Scale with **distributed jobs** (`instance_count` + `PyTorchDistribution`) on **ND** InfiniBand clusters; default to **bf16** on H100/H200.
- Compose multi-step **pipelines** of reusable components (the training DAG), and register outputs to the **model registry** with lineage — the handoff to deployment and the basis of dev→prod promotion.

## Try it

Create a workspace and a scale-to-zero GPU compute cluster on an NC A100 SKU. Register a small versioned data asset pointing at a folder in your lake, define a curated-image environment, and submit a `command` job that trains a tiny model on that data with `mlflow.autolog()` enabled. Open the returned `studio_url`, watch the metrics stream, then register the resulting model to the registry and confirm you can trace it back to the job, environment, and data version that produced it.
