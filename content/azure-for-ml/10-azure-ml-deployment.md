# 10 — Azure Machine Learning: Deployment

A registered model earns its keep only when it serves predictions. Azure Machine Learning's deployment layer takes a versioned model from the registry and stands it up as an inference service — either **real-time** (a live HTTPS endpoint returning predictions in milliseconds) or **batch** (asynchronous scoring of large datasets on a schedule). Both are managed: Azure handles the serving infrastructure, scaling, health, and monitoring so you focus on the model. This section covers **managed online endpoints**, **batch endpoints**, safe rollout, and autoscaling. In the end-to-end solution, this is where the trained model meets production traffic.

## Endpoints and deployments: the two-layer model

Azure ML separates the stable **endpoint** (a durable URL and auth boundary) from the **deployment** (a specific model + environment + compute behind it). One endpoint can host multiple deployments and split traffic between them — this indirection is what enables blue-green and canary rollouts without ever changing the URL your callers use. You update the model by adding a new deployment and shifting traffic, not by editing a live one.

## Managed online endpoints: real-time serving

A **managed online endpoint** deploys a model to fully managed CPU or GPU infrastructure with a turnkey HTTPS API — Azure takes care of serving, scaling, securing, and monitoring. You create the endpoint, then a deployment specifying the model, an inference environment, an instance type (a VM SKU), and an instance count.

```python
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# 1. The stable endpoint (URL + auth)
endpoint = ManagedOnlineEndpoint(
    name="fraud-endpoint",
    auth_mode="key",          # or "aml_token" for Entra-based tokens
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 2. A deployment behind it
blue = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="fraud-endpoint",
    model="azureml:fraud-detector:1",           # from the registry
    instance_type="Standard_DS3_v2",            # CPU here; NC-series for GPU models
    instance_count=2,
)
ml_client.online_deployments.begin_create_or_update(blue).result()

# 3. Send all traffic to blue
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

For MLflow models, Azure ML generates the scoring wrapper automatically. For custom logic you provide a scoring script with an `init()` (loads the model once) and a `run()` (scores each request) plus a custom environment. Invocation is a normal HTTPS POST:

```bash
az ml online-endpoint invoke --name fraud-endpoint \
  --request-file sample-request.json \
  --resource-group rg-mlx-dev --workspace-name mlw-mlx-dev
```

Authenticate callers with a key or, better, an **Entra token** (`aml_token`), and grant the endpoint's managed identity `AcrPull` on the registry and read on any storage/features it needs. For network isolation, deploy the endpoint into the workspace's managed VNet with a private endpoint so it is not reachable from the public internet.

## Safe rollout: blue-green and canary

The two-layer model makes safe deploys mechanical. To ship a new model version, add a **green** deployment alongside blue, then move traffic gradually and watch metrics — this is **canary** deployment. If green misbehaves, shift traffic back to blue instantly; no callers ever saw a broken URL.

```python
green = ManagedOnlineDeployment(
    name="green", endpoint_name="fraud-endpoint",
    model="azureml:fraud-detector:2",             # the new version
    instance_type="Standard_DS3_v2", instance_count=2,
)
ml_client.online_deployments.begin_create_or_update(green).result()

# Canary: 10% to green, 90% to blue; ramp up as confidence grows
endpoint.traffic = {"blue": 90, "green": 10}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

Azure ML also supports **mirrored traffic** — copying a fraction of live requests to a new deployment for shadow testing without returning its responses to users, so you validate a candidate on real traffic risk-free before it serves anyone.

## Autoscaling online endpoints

Fixed instance counts waste money off-peak and fall over under spikes. Managed online endpoints integrate with **Azure Monitor autoscale**, so you scale instance count on metrics (CPU/GPU utilization, request latency, requests per second) or on a schedule (scale up for business hours, down overnight).

```bash
# Scale the blue deployment on CPU utilization between 2 and 10 instances
az monitor autoscale create -g rg-mlx-dev \
  --resource "$(az ml online-deployment show -n blue --endpoint-name fraud-endpoint \
      -g rg-mlx-dev -w mlw-mlx-dev --query id -o tsv)" \
  --name autoscale-blue --min-count 2 --max-count 10 --count 2

az monitor autoscale rule create -g rg-mlx-dev --autoscale-name autoscale-blue \
  --condition "CpuUtilizationPercentage > 70 avg 5m" --scale out 2
az monitor autoscale rule create -g rg-mlx-dev --autoscale-name autoscale-blue \
  --condition "CpuUtilizationPercentage < 30 avg 5m" --scale in 1
```

Keep a sensible **minimum** (real-time endpoints do not scale to zero the way training clusters do — you keep warm instances to avoid cold-start latency) and cap the **maximum** to bound cost. For GPU-served models, autoscale on GPU utilization and size instances on NC-series SKUs.

## Batch endpoints: asynchronous scoring at scale

Not all inference is real-time. **Batch endpoints** score large volumes asynchronously: you point them at data (a folder in the lake, a data asset, or a storage path) and they run a job that parallelizes scoring across a **compute cluster**, reading inputs and writing outputs directly to storage. They are ideal for nightly scoring of a whole customer base, backfilling predictions, or any high-throughput job where latency does not matter. Notably, the endpoints and deployments themselves are free — you pay only for the compute the jobs consume.

```python
from azure.ai.ml.entities import BatchEndpoint, ModelBatchDeployment, ModelBatchDeploymentSettings

be = BatchEndpoint(name="fraud-batch")
ml_client.batch_endpoints.begin_create_or_update(be).result()

bd = ModelBatchDeployment(
    name="default", endpoint_name="fraud-batch",
    model="azureml:fraud-detector:1",
    compute="cpu-cluster",                       # runs on an AmlCompute cluster
    settings=ModelBatchDeploymentSettings(
        instance_count=4, max_concurrency_per_instance=2,
        mini_batch_size=64, output_action="append_row",
    ),
)
ml_client.batch_deployments.begin_create_or_update(bd).result()
```

```bash
# Kick off a scoring job over a folder of inputs; outputs land in storage
az ml batch-endpoint invoke --name fraud-batch \
  --input azureml://datastores/stmlxdata/paths/scoring/inputs/ \
  --resource-group rg-mlx-dev --workspace-name mlw-mlx-dev
```

A batch endpoint can also wrap a whole **pipeline** (not just a single model), so a multi-step scoring workflow — preprocess → score → postprocess — becomes one invocable, schedulable endpoint. Trigger batch runs on a schedule (Data Factory or a timer Function) or in reaction to new data landing in the lake (a blob-triggered Function).

## Real-time vs batch: choosing

- **Managed online endpoint** — low-latency, per-request, always-warm, autoscaled. For interactive apps, fraud checks in the transaction path, recommendations at page load.
- **Batch endpoint** — high-throughput, asynchronous, scale-out on clusters, free except for compute. For scheduled mass scoring, backfills, and offline pipelines.

Many systems run both from the same registered model: a batch endpoint precomputes scores nightly into Azure SQL for the common case, and an online endpoint handles the real-time long tail.

## How deployment fits the whole solution

Deployment is where the model becomes a product surface. The **online endpoint** sits behind the thin HTTP Function (or API gateway) from the serverless section, inside the managed VNet, served on autoscaled CPU/GPU, reading online features from Cosmos DB in the request path. The **batch endpoint** runs scheduled scoring over lake data on scale-to-zero clusters, writing results back to Azure SQL for applications. CI/CD promotes a new model version by creating a **green** deployment and canarying traffic, with instant rollback via the traffic split. **Azure Monitor** watches latency and utilization to autoscale, and data drift monitoring compares live inputs to the training baseline. The registry hands off the model; the endpoints turn it into served predictions.

## Key takeaways

- Separate the durable **endpoint** (URL + auth) from the **deployment** (model + env + compute); the traffic split between deployments enables **blue-green and canary** rollout with instant rollback.
- **Managed online endpoints** give turnkey real-time serving on managed CPU/GPU; MLflow models get an auto-generated scorer, custom models use an `init`/`run` scoring script.
- **Autoscale** online endpoints on utilization/latency with a warm **minimum** (no scale-to-zero for real-time) and a cost-bounding **maximum**.
- **Batch endpoints** score large data asynchronously on **compute clusters**, read/write storage directly, are free except for compute, and can wrap whole pipelines.
- Choose **online** for low-latency per-request serving, **batch** for high-throughput scheduled scoring — and often run both from one registered model.

## Try it

Take the model you registered previously and deploy it as a `blue` managed online deployment with 100% traffic; invoke it with a sample request and confirm a prediction comes back. Then register a second model version, create a `green` deployment, and shift the traffic split to `{blue: 90, green: 10}` — a live canary — before rolling forward to 100% green or back to blue. Separately, create a batch endpoint over the same model and invoke it against a folder of inputs in your lake, confirming the outputs land in storage. You have now shipped the same model as both a real-time and a batch service.
