# 10 — Vertex AI: Prediction and Endpoints

A trained model in the registry produces no value until it serves predictions. Vertex AI Prediction is the managed serving layer that turns a registered model into a scalable, monitored, secured inference service — either as a live **online** endpoint answering requests in real time, or as a **batch** job scoring a large dataset all at once. This module covers both, plus the autoscaling, networking, and cost decisions that make serving production-grade. As in training, the SDK entry point is `from google.cloud import aiplatform`.

## Online prediction: endpoints and deployed models

The online path has two objects. An **Endpoint** is a stable, addressable serving resource with a URL. A **Model** from the registry is **deployed** onto an endpoint, becoming a **DeployedModel** backed by a pool of serving replicas on machine types you choose. Separating the two is deliberate: an endpoint can host multiple deployed models and split traffic between them, which is how you do canary rollouts and A/B tests.

```python
from google.cloud import aiplatform

aiplatform.init(project="myco-fraud-dev", location="us-central1")

endpoint = aiplatform.Endpoint.create(display_name="fraud-endpoint")

model = aiplatform.Model("projects/.../models/1234567890")  # from the registry
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5,
    accelerator_type="NVIDIA_L4",   # attach a GPU if the model needs one
    accelerator_count=1,
    traffic_percentage=100,
)

# Call it
prediction = endpoint.predict(instances=[{"amount": 42.0, "hour": 3}])
print(prediction.predictions)
```

The gcloud path is what CI/CD uses. It is a three-step dance — the model is uploaded to the registry (module 09), an endpoint is created, then the model is deployed onto it as a `DeployedModel`:

```bash
# 1. Create the endpoint
gcloud ai endpoints create --region=us-central1 --display-name=fraud-endpoint

# 2. Deploy a registry model onto it
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --region=us-central1 \
  --model=MODEL_ID \
  --display-name=fraud-v1 \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --min-replica-count=1 \
  --max-replica-count=5 \
  --service-account=serving-sa@myco-fraud-dev.iam.gserviceaccount.com \
  --traffic-split=0=100

# 3. Call it, then list/undeploy/tear down
gcloud ai endpoints predict ENDPOINT_ID --region=us-central1 --json-request=instances.json
gcloud ai endpoints list --region=us-central1
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1
gcloud ai endpoints undeploy-model ENDPOINT_ID --region=us-central1 --deployed-model-id=DEPLOYED_MODEL_ID
gcloud ai endpoints delete ENDPOINT_ID --region=us-central1
```

The `--traffic-split` flag takes `DEPLOYED_MODEL_ID=PERCENT` pairs (the special key `0` targets the model being deployed in this call); this is the mechanism behind canary and A/B rollouts — deploy a second model with `--traffic-split=old-id=90,0=10` to send it 10% of traffic. A recurring **gotcha**: the model, its serving container image, and the endpoint must all be in the **same region**, and accelerator-backed deployments need serving-side accelerator **quota** in that region — distinct from the training quota you raised in module 09.

## Autoscaling

A deployed model autoscales between `min_replica_count` and `max_replica_count`. This is the central cost/latency knob:

- **`min_replica_count`** — the floor. Set it to at least 1 for a latency-sensitive production endpoint so there is always a warm replica (no cold-start penalty). Setting it higher pre-provisions for baseline traffic.
- **`max_replica_count`** — the ceiling, which caps both your ability to absorb spikes and your maximum spend.

Vertex scales replicas based on utilization and traffic. By default it targets **60%** on CPU and, for accelerator-backed models, on GPU duty cycle — scaling up when *either* metric exceeds its target and down only when *both* are under. You can override the target per metric with `--autoscaling-metric-specs`, whose accepted metric names include `cpu-usage`, `gpu-duty-cycle`, and `request-counts-per-minute` (plus vLLM-specific metrics for open-model serving):

```bash
gcloud ai endpoints deploy-model ENDPOINT_ID --region=us-central1 --model=MODEL_ID \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --min-replica-count=1 --max-replica-count=8 \
  --autoscaling-metric-specs=cpu-usage=70,gpu-duty-cycle=75
```

Right-sizing the machine type and accelerator matters enormously here: serve on an **L4 (G2-class)** for most models rather than an expensive training-grade GPU, and only reach for larger accelerators when the model genuinely requires the memory or throughput. You pay for the replicas that are running, for as long as they run — so an over-provisioned `min_replica_count` (the **warm floor**) on a GPU endpoint is a standing 24/7 cost even when idle. Set it to the smallest count that covers your latency SLO for baseline traffic; a floor of 1 avoids cold starts, a floor above that is pre-provisioning you pay for continuously.

## Dedicated vs shared, and private endpoints

Endpoints come in a couple of flavors that matter for latency and security:

- **Dedicated endpoints** — your deployed model runs on machines reserved for it (the pattern shown above with an explicit machine type), giving predictable performance and a dedicated DNS path. This is the norm for production, and it is now the default endpoint type. **Shared** (public, multi-tenant) endpoints still exist for lightweight/experimental use but offer weaker isolation.
- **Private Service Connect endpoints** — instead of a public endpoint, prediction traffic flows over **Private Service Connect** through your VPC's internal network, never touching the public internet. You opt in at endpoint-creation time by attaching a PSC config, then deploy the model onto it:

```bash
gcloud ai endpoints create --region=us-central1 --display-name=fraud-endpoint-psc \
  --private-service-connect-config=enable-private-service-connect=true,project-allowlist=myco-fraud-prod
```

  This is the standard for security-sensitive or latency-critical serving, and it ties directly back to the networking module: a private endpoint plus no-external-IP callers keeps inference traffic entirely inside your trusted boundary.

For high-throughput or specialized serving (many models, multi-GPU, custom routing, LoRA adapters), teams sometimes serve on **GKE** with the inference-optimized gateway instead of a Vertex endpoint — but for a registered model that needs a managed, autoscaling, monitored endpoint, Vertex online prediction is the direct path.

## Request/response logging

For debugging, auditing, and building the dataset that feeds drift detection, enable **request/response logging** on the deployed model so a sample of live inputs and outputs is written to a **BigQuery** table. You turn it on with `--enable-access-logging` (access logs to Cloud Logging) at deploy time, and configure the BigQuery prediction-log sink and sampling rate on the DeployedModel. Those logged predictions become the raw material for **Vertex AI Model Monitoring** to compare live feature distributions against the training baseline — the mechanism that closes the retraining loop, covered in module 13. You can also `--disable-container-logging` to suppress your container's stdout/stderr if it is noisy or contains sensitive payloads.

## Custom serving containers

Vertex can serve a model with a prebuilt container, or you can bring a **custom serving container** from Artifact Registry that implements the expected HTTP contract (a health route and a predict route). This gives you full control over preprocessing, the framework, and the runtime — the same reproducibility argument as custom training containers. You declare the contract when you `gcloud ai models upload` the model, with the container flags that tell Vertex how to talk to your server:

```bash
gcloud ai models upload --region=us-central1 --display-name=fraud-classifier \
  --container-image-uri=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/serve:v1 \
  --artifact-uri=gs://myco-fraud-models/fraud/run-001/model \
  --container-ports=8080 \
  --container-health-route=/health \
  --container-predict-route=/predict \
  --container-env-vars=MODEL_NAME=fraud,LOG_LEVEL=info
```

The **HTTP contract** is strict and a common source of failed deployments: your container must listen on `--container-ports` (default 8080), return `200` on `--container-health-route` so Vertex knows the replica is live, and accept prediction requests on `--container-predict-route`, replying with a JSON body of the shape `{"predictions": [...]}`. If health checks never pass — wrong port, slow model load exceeding the startup probe, a health route that 404s — the deployment hangs and eventually fails. Vertex substitutes `AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, and `AIP_PREDICT_ROUTE` environment variables into the container so your server can read them rather than hard-coding.

## Batch prediction

Not every use case needs a live endpoint. When you have a large dataset to score and can tolerate minutes-to-hours latency — nightly risk scoring of every account, backfilling predictions, offline evaluation — **batch prediction** is cheaper and simpler. You submit a `BatchPredictionJob` that reads inputs from Cloud Storage or BigQuery, provisions transient workers (which you can size and accelerate), scores everything, writes results back to Cloud Storage or BigQuery, and tears down. No endpoint to keep warm, no autoscaling to tune.

```python
model = aiplatform.Model("projects/.../models/1234567890")

batch_job = model.batch_predict(
    job_display_name="fraud-nightly-score",
    bigquery_source="bq://myco-fraud-dev.fraud.accounts_to_score",
    bigquery_destination_prefix="bq://myco-fraud-dev.fraud",
    machine_type="n1-standard-8",
    starting_replica_count=2,
    max_replica_count=10,
)
```

The source and destination can be **BigQuery** (as above) or **Cloud Storage**, and for Cloud Storage inputs you declare the file layout with an instances format — `jsonl` (one JSON instance per line), `csv`, `tf-record`, or `file-list`:

```python
batch_job = model.batch_predict(
    job_display_name="fraud-nightly-score",
    gcs_source="gs://myco-fraud-data/to-score/*.jsonl",
    instances_format="jsonl",
    gcs_destination_prefix="gs://myco-fraud-models/predictions/",
    predictions_format="jsonl",
    machine_type="n1-standard-8",
    starting_replica_count=2,
    max_replica_count=10,
)
```

Batch prediction is largely SDK-driven; the `gcloud ai batch-prediction-jobs` surface exists but is thin, so `model.batch_predict(...)` (or a `--config` JSON of the `BatchPredictionJob` resource) is the practical interface. Because batch workers are transient and interruption-tolerant, batch prediction is an ideal fit for **Spot** capacity — another cost lever. The BigQuery-in, BigQuery-out flow shown earlier plugs the results straight back into the data plane for downstream queries and dashboards.

## Choosing online vs batch

The decision is about latency and access pattern, not model quality:

- **Online** — you need a prediction *now*, per request, in response to user or system events (fraud check at checkout, a recommendation on page load). Pay for warm capacity; tune autoscaling.
- **Batch** — you need predictions for *many* records and can wait; there is no per-request latency requirement (nightly scoring, bulk enrichment). Pay only for the job; use Spot.

Many systems use both: an online endpoint for real-time decisions and a nightly batch job for bulk scoring and analytics, both serving the same registered model version.

## How this fits the whole solution

Prediction is where the whole pipeline pays off. A model manufactured by the training stage and versioned in the registry is deployed here to an autoscaling, privately-networked endpoint (for real-time decisions) and/or run as a batch job writing back to BigQuery (for bulk scoring). It sits behind IAM and, for GenAI-facing layers, alongside Cloud Run and Gemini. Its metrics and drift feed the monitoring loop, and its autoscaling bounds and Spot batch jobs feed the cost story. Deployed correctly — right-sized accelerators, sane autoscaling floors, private endpoints, batch for bulk — serving is where an ML system becomes a product.

## Key takeaways

- **Online prediction** deploys a registry **Model** onto an **Endpoint** as a **DeployedModel** on chosen machine types/accelerators; one endpoint can host multiple models and **split traffic** for canary/A-B rollouts.
- **Autoscaling** between `min`/`max` replica counts is the core cost/latency knob — keep a warm floor for latency, cap the ceiling for spend, and right-size to an **L4** for most models.
- Use **dedicated resources** for predictable performance and **Private Service Connect endpoints** to keep inference traffic private; bring a **custom serving container** for full runtime control.
- **Batch prediction** (`BatchPredictionJob`) scores large Cloud Storage/BigQuery datasets on transient, Spot-friendly workers and writes results back — use it whenever real-time latency is not required.

## CLI cheat-sheet

```bash
# --- Upload a model with a custom serving container (HTTP contract) ---
gcloud ai models upload --region=us-central1 --display-name=fraud-classifier \
  --container-image-uri=REGION-docker.pkg.dev/PROJ/ml-images/serve:v1 \
  --artifact-uri=gs://myco-fraud-models/fraud/run-001/model \
  --container-ports=8080 --container-health-route=/health --container-predict-route=/predict \
  --container-env-vars=MODEL_NAME=fraud

# --- Endpoints: create, deploy, split traffic, call, tear down ---
gcloud ai endpoints create --region=us-central1 --display-name=fraud-endpoint
gcloud ai endpoints create --region=us-central1 --display-name=fraud-endpoint-psc \
  --private-service-connect-config=enable-private-service-connect=true,project-allowlist=myco-fraud-prod

gcloud ai endpoints deploy-model ENDPOINT_ID --region=us-central1 --model=MODEL_ID \
  --display-name=fraud-v1 --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --min-replica-count=1 --max-replica-count=8 \
  --autoscaling-metric-specs=cpu-usage=70,gpu-duty-cycle=75 \
  --service-account=serving-sa@myco-fraud-dev.iam.gserviceaccount.com \
  --enable-access-logging --traffic-split=0=100

gcloud ai endpoints predict  ENDPOINT_ID --region=us-central1 --json-request=instances.json
gcloud ai endpoints list     --region=us-central1
gcloud ai endpoints describe ENDPOINT_ID --region=us-central1
gcloud ai endpoints undeploy-model ENDPOINT_ID --region=us-central1 --deployed-model-id=DEPLOYED_MODEL_ID
gcloud ai endpoints delete   ENDPOINT_ID --region=us-central1   # stops the warm-replica billing

# --- Batch prediction is SDK-driven: model.batch_predict(gcs_source=..., bigquery_source=...) ---
# autoscaling metric names: cpu-usage | gpu-duty-cycle | request-counts-per-minute
```

## Try it

Serve one registered model two ways:

1. Deploy a model from the registry to an **Endpoint** with an L4 GPU, `min_replica_count=1`, `max_replica_count=3`, and call it with `endpoint.predict(...)`.
2. Deploy a *second* version of the model to the same endpoint with a small traffic percentage (a canary) and observe traffic splitting.
3. Run a **batch prediction** job over a BigQuery table with `model.batch_predict(...)`, writing results back to BigQuery, and query the output table.
4. Compare the two: note the always-on cost of the warm online endpoint versus the pay-per-job cost of batch, and decide which each of your use cases should use. Then delete the endpoint to stop it billing.
