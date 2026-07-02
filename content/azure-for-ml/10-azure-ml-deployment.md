# 10 — Azure Machine Learning: Deployment

A registered model earns its keep only when it serves predictions. Azure Machine Learning's deployment layer takes a versioned model from the registry and stands it up as an inference service — either **real-time** (a live HTTPS endpoint returning predictions in milliseconds) or **batch** (asynchronous scoring of large datasets on a schedule). Both are managed: Azure handles the serving infrastructure, scaling, health, and monitoring so you focus on the model. This section covers **managed online endpoints**, **batch endpoints**, safe rollout, and autoscaling. In the end-to-end solution, this is where the trained model meets production traffic.

## Endpoints and deployments: the two-layer model

Azure ML separates the stable **endpoint** (a durable URL and auth boundary) from the **deployment** (a specific model + environment + compute behind it). One endpoint can host multiple deployments and split traffic between them — this indirection is what enables blue-green and canary rollouts without ever changing the URL your callers use. You update the model by adding a new deployment and shifting traffic, not by editing a live one.

In practice the CLI v2 workflow is entirely **YAML-driven**: you describe the endpoint and each deployment in spec files and apply them with `--file`, so the same definitions live in source control and deploy identically through CI/CD. The endpoint spec is a handful of lines — a name and an auth mode; the deployment spec carries the real weight (model, environment, instance type, request and probe settings). Everything you can do in the studio has a CLI verb, and the common ones you will reach for daily are `create`, `update`, `show`, `list`, `invoke`, and `delete`:

```bash
# The stable endpoint, from a tiny YAML spec (name + auth_mode)
az ml online-endpoint create --file endpoint.yml \
  -g rg-mlx-dev -w mlw-mlx-dev
# Inspect and enumerate what exists
az ml online-endpoint show -n fraud-endpoint -g rg-mlx-dev -w mlw-mlx-dev
az ml online-endpoint list -g rg-mlx-dev -w mlw-mlx-dev -o table
# Tear it down (removes all its deployments)
az ml online-endpoint delete -n fraud-endpoint -g rg-mlx-dev -w mlw-mlx-dev --yes
```

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

The deployment YAML is the artifact worth knowing well, because it is where you tune cost, latency, and resilience. A managed online deployment spec names the model and environment, picks an `instance_type` (VM SKU) and `instance_count`, and — the settings most people skip — configures `request_settings` (concurrency and timeout), a `liveness_probe`/`readiness_probe`, and `scale_settings`:

```yaml
# blue-deployment.yml — a full managed online deployment spec
$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
name: blue
endpoint_name: fraud-endpoint
model: azureml:fraud-detector:1
environment: azureml:fraud-serving-env:3      # curated or custom image
code_configuration:
  code: ./src
  scoring_script: score.py                     # your init()/run()
instance_type: Standard_DS3_v2                 # NC-series for GPU models
instance_count: 2
request_settings:
  max_concurrent_requests_per_instance: 4
  request_timeout_ms: 5000                      # 90s is the gateway ceiling
  max_queue_wait_ms: 2000
liveness_probe:
  initial_delay: 30
  period: 10
  failure_threshold: 3
scale_settings:
  type: default                                 # or target_utilization for built-in autoscale
```

```bash
# Apply it; --all-traffic sends 100% here on success (skip for canary)
az ml online-deployment create --file blue-deployment.yml --all-traffic \
  -g rg-mlx-dev -w mlw-mlx-dev
```

For MLflow models, Azure ML generates the scoring wrapper automatically. For custom logic you provide a scoring script with an `init()` (loads the model once) and a `run()` (scores each request) plus a custom environment. When a deployment misbehaves, the first move is always to pull its container logs — the `inference-server` container holds your scoring-script output and stack traces, the `storage-initializer` container tells you whether the model actually downloaded:

```bash
# Invocation is a normal HTTPS POST
az ml online-endpoint invoke --name fraud-endpoint \
  --request-file sample-request.json \
  --resource-group rg-mlx-dev --workspace-name mlw-mlx-dev

# Debug: tail the container logs; -c switches container
az ml online-deployment get-logs -n blue --endpoint-name fraud-endpoint \
  --lines 200 -g rg-mlx-dev -w mlw-mlx-dev
az ml online-deployment get-logs -n blue --endpoint-name fraud-endpoint \
  -c storage-initializer -g rg-mlx-dev -w mlw-mlx-dev

# Iterate on a spec without recreating: update, list, and inspect deployments
az ml online-deployment update --file blue-deployment.yml -g rg-mlx-dev -w mlw-mlx-dev
az ml online-deployment list --endpoint-name fraud-endpoint -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml online-deployment show -n blue --endpoint-name fraud-endpoint -g rg-mlx-dev -w mlw-mlx-dev
```

A gotcha worth internalizing: **before you ever push to the cloud, deploy locally**. Adding `--local` to `online-deployment create` runs the exact scoring container in Docker on your machine, so you catch a broken `score.py` or a missing dependency in seconds instead of after a five-minute cloud provision.

Authenticate callers with one of two modes set on the endpoint's `auth_mode`: **`key`** (static primary/secondary keys, simplest for machine-to-machine) or **`aml_token`** (short-lived **Entra tokens**, the keyless default you should prefer). Retrieve or rotate credentials with the endpoint's own subcommands — never copy a key into code:

```bash
# Fetch the current key/token for a caller
az ml online-endpoint get-credentials -n fraud-endpoint -g rg-mlx-dev -w mlw-mlx-dev
# Rotate on a schedule or after a suspected leak (primary or secondary)
az ml online-endpoint regenerate-keys -n fraud-endpoint --key-type primary \
  -g rg-mlx-dev -w mlw-mlx-dev
```

Grant the endpoint's managed identity `AcrPull` on the registry and read on any storage/features it needs. For network isolation, deploy the endpoint into the workspace's managed VNet with a private endpoint so it is not reachable from the public internet.

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

All of this is equally scriptable from the CLI, which is what CI/CD actually runs. Traffic and mirroring both live on `az ml online-endpoint update`: `--traffic` takes space-separated `deployment=percent` pairs (they must sum to 100), and `--mirror-traffic` shadows a single deployment:

```bash
# Ship the green deployment with zero live traffic first
az ml online-deployment create --file green-deployment.yml -g rg-mlx-dev -w mlw-mlx-dev

# Canary: split live traffic 90/10 (pairs must total 100)
az ml online-endpoint update -n fraud-endpoint \
  --traffic "blue=90 green=10" -g rg-mlx-dev -w mlw-mlx-dev

# Ramp to 100% green once metrics look clean, then retire blue
az ml online-endpoint update -n fraud-endpoint \
  --traffic "blue=0 green=100" -g rg-mlx-dev -w mlw-mlx-dev
az ml online-deployment delete -n blue --endpoint-name fraud-endpoint \
  -g rg-mlx-dev -w mlw-mlx-dev --yes
```

Azure ML also supports **mirrored traffic** — copying a fraction of live requests to a new deployment for shadow testing without returning its responses to users, so you validate a candidate on real traffic risk-free before it serves anyone. Mirroring is capped at 50% and is independent of the live split:

```bash
# Shadow 10% of live requests onto green without returning its responses
az ml online-endpoint update -n fraud-endpoint \
  --mirror-traffic "green=10" -g rg-mlx-dev -w mlw-mlx-dev
# Turn shadowing off
az ml online-endpoint update -n fraud-endpoint \
  --mirror-traffic "green=0" -g rg-mlx-dev -w mlw-mlx-dev
```

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

Metric rules handle spikes, but predictable daily patterns are better served by **schedule-based** rules — scale the floor up for business hours and back down overnight so you are not paying for warm GPU at 3 a.m. Add profiles and recurrence to the same autoscale setting:

```bash
# Business-hours floor: 4 instances weekday mornings, back to 2 at night
az monitor autoscale profile create -g rg-mlx-dev --autoscale-name autoscale-blue \
  --name business-hours --min-count 4 --max-count 10 --count 4 \
  --recurrence week Mon Tue Wed Thu Fri --timezone "Eastern Standard Time" \
  --start 08:00 --end 18:00

# Inspect and clean up autoscale settings
az monitor autoscale show -g rg-mlx-dev --name autoscale-blue
az monitor autoscale list -g rg-mlx-dev -o table
az monitor autoscale rule list -g rg-mlx-dev --autoscale-name autoscale-blue -o table
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

Batch endpoints follow the same endpoint/deployment two-layer model as online, and the same YAML-driven CLI. The `default` deployment is the one an invocation targets unless you override it, so promoting a new model version is again a matter of adding a deployment and repointing the default. Note the batch deployment settings that govern throughput and correctness: `instance_count` (cluster nodes), `max_concurrency_per_instance` (parallel scoring processes per node), `mini_batch_size` (rows or files per task), `output_action` (`append_row` into one file, or `summary_only`), and `error_threshold` (how many bad records abort the job):

```bash
# Create the endpoint and a default deployment from specs
az ml batch-endpoint create --file batch-endpoint.yml -g rg-mlx-dev -w mlw-mlx-dev
az ml batch-deployment create --file batch-deployment.yml --set-default \
  -g rg-mlx-dev -w mlw-mlx-dev

# Kick off a scoring job over a folder of inputs; outputs land in storage
az ml batch-endpoint invoke --name fraud-batch \
  --input azureml://datastores/stmlxdata/paths/scoring/inputs/ \
  --input-type uri_folder --mini-batch-size 64 \
  --resource-group rg-mlx-dev --workspace-name mlw-mlx-dev

# The invoke returns immediately with a job; track it and enumerate history
az ml batch-deployment list-jobs --name default --endpoint-name fraud-batch \
  -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml batch-endpoint list -g rg-mlx-dev -w mlw-mlx-dev -o table
```

Because `invoke` is asynchronous it hands back a **job** you monitor with `az ml job show` (or `job stream` to follow logs); `batch-deployment list-jobs` gives you the run history for auditing and debugging failed scoring runs. A batch endpoint can also wrap a whole **pipeline** (not just a single model), so a multi-step scoring workflow — preprocess → score → postprocess — becomes one invocable, schedulable endpoint. Trigger batch runs on a schedule (Data Factory or a timer Function) or in reaction to new data landing in the lake (a blob-triggered Function).

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

## CLI cheat-sheet

```bash
# --- Online endpoints (az ml v2; -g rg-mlx-dev -w mlw-mlx-dev assumed) ---
az ml online-endpoint create --file endpoint.yml       # create from YAML spec
az ml online-endpoint show -n fraud-endpoint           # details (scoring_uri, state)
az ml online-endpoint list -o table                    # all endpoints in workspace
az ml online-endpoint update -n fraud-endpoint --traffic "blue=90 green=10"
az ml online-endpoint update -n fraud-endpoint --mirror-traffic "green=10"  # shadow (max 50%)
az ml online-endpoint invoke -n fraud-endpoint --request-file req.json
az ml online-endpoint invoke -n fraud-endpoint -d green --request-file req.json  # hit one deployment
az ml online-endpoint get-credentials -n fraud-endpoint            # keys / aml_token
az ml online-endpoint regenerate-keys -n fraud-endpoint --key-type primary
az ml online-endpoint delete -n fraud-endpoint --yes   # removes all deployments

# --- Online deployments ---
az ml online-deployment create --file blue-deployment.yml --all-traffic   # 100% on success
az ml online-deployment create --file blue-deployment.yml --local         # Docker, test locally
az ml online-deployment update --file blue-deployment.yml                 # apply spec changes
az ml online-deployment show -n blue --endpoint-name fraud-endpoint
az ml online-deployment list --endpoint-name fraud-endpoint -o table
az ml online-deployment get-logs -n blue --endpoint-name fraud-endpoint --lines 200
az ml online-deployment get-logs -n blue --endpoint-name fraud-endpoint -c storage-initializer
az ml online-deployment delete -n blue --endpoint-name fraud-endpoint --yes

# --- Batch endpoints & deployments ---
az ml batch-endpoint create --file batch-endpoint.yml
az ml batch-deployment create --file batch-deployment.yml --set-default
az ml batch-endpoint invoke -n fraud-batch --input azureml://datastores/stmlxdata/paths/scoring/inputs/ \
  --input-type uri_folder --mini-batch-size 64
az ml batch-deployment list-jobs --name default --endpoint-name fraud-batch -o table
az ml batch-endpoint list -o table
az ml job show -n <job-name>                           # track an async batch job
az ml job stream -n <job-name>                         # follow its logs

# --- Autoscale (Azure Monitor) ---
DEP_ID=$(az ml online-deployment show -n blue --endpoint-name fraud-endpoint --query id -o tsv)
az monitor autoscale create -g rg-mlx-dev --resource "$DEP_ID" \
  --name autoscale-blue --min-count 2 --max-count 10 --count 2
az monitor autoscale rule create -g rg-mlx-dev --autoscale-name autoscale-blue \
  --condition "CpuUtilizationPercentage > 70 avg 5m" --scale out 2
az monitor autoscale rule create -g rg-mlx-dev --autoscale-name autoscale-blue \
  --condition "CpuUtilizationPercentage < 30 avg 5m" --scale in 1
az monitor autoscale profile create -g rg-mlx-dev --autoscale-name autoscale-blue \
  --name business-hours --min-count 4 --max-count 10 --count 4 \
  --recurrence week Mon Tue Wed Thu Fri --start 08:00 --end 18:00 --timezone "Eastern Standard Time"
az monitor autoscale show -g rg-mlx-dev --name autoscale-blue
```

## Try it

Take the model you registered previously and deploy it as a `blue` managed online deployment with 100% traffic; invoke it with a sample request and confirm a prediction comes back. Then register a second model version, create a `green` deployment, and shift the traffic split to `{blue: 90, green: 10}` — a live canary — before rolling forward to 100% green or back to blue. Separately, create a batch endpoint over the same model and invoke it against a folder of inputs in your lake, confirming the outputs land in storage. You have now shipped the same model as both a real-time and a batch service.
