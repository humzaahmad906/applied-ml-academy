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

The gcloud path mirrors this with `gcloud ai endpoints create`, `gcloud ai models upload`, and `gcloud ai endpoints deploy-model`.

## Autoscaling

A deployed model autoscales between `min_replica_count` and `max_replica_count`. This is the central cost/latency knob:

- **`min_replica_count`** — the floor. Set it to at least 1 for a latency-sensitive production endpoint so there is always a warm replica (no cold-start penalty). Setting it higher pre-provisions for baseline traffic.
- **`max_replica_count`** — the ceiling, which caps both your ability to absorb spikes and your maximum spend.

Vertex scales replicas based on utilization (CPU, or GPU duty cycle for accelerator-backed models) and traffic. Right-sizing the machine type and accelerator matters enormously here: serve on an **L4 (G2-class)** for most models rather than an expensive training-grade GPU, and only reach for larger accelerators when the model genuinely requires the memory or throughput. You pay for the replicas that are running, for as long as they run — so an over-provisioned `min_replica_count` on a GPU endpoint is a standing cost.

## Dedicated vs shared, and private endpoints

Endpoints come in a couple of flavors that matter for latency and security:

- **Dedicated resources** — your deployed model runs on machines reserved for it (the pattern shown above with an explicit machine type), giving predictable performance. This is the norm for production.
- **Private Service Connect endpoints** — instead of a public endpoint, prediction traffic flows over **Private Service Connect** through your VPC's internal network, never touching the public internet. This is the standard for security-sensitive or latency-critical serving, and it ties directly back to the networking module: a private endpoint plus no-external-IP callers keeps inference traffic entirely inside your trusted boundary.

For high-throughput or specialized serving (many models, multi-GPU, custom routing, LoRA adapters), teams sometimes serve on **GKE** with the inference-optimized gateway instead of a Vertex endpoint — but for a registered model that needs a managed, autoscaling, monitored endpoint, Vertex online prediction is the direct path.

## Custom serving containers

Vertex can serve a model with a prebuilt container, or you can bring a **custom serving container** from Artifact Registry that implements the expected HTTP contract (a health route and a predict route). This gives you full control over preprocessing, the framework, and the runtime — the same reproducibility argument as custom training containers. You reference the serving image when you upload the model to the registry, and Vertex runs it behind the endpoint.

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

Because batch workers are transient and interruption-tolerant, batch prediction is an ideal fit for **Spot** capacity — another cost lever. The BigQuery-in, BigQuery-out flow shown here plugs the results straight back into the data plane for downstream queries and dashboards.

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

## Try it

Serve one registered model two ways:

1. Deploy a model from the registry to an **Endpoint** with an L4 GPU, `min_replica_count=1`, `max_replica_count=3`, and call it with `endpoint.predict(...)`.
2. Deploy a *second* version of the model to the same endpoint with a small traffic percentage (a canary) and observe traffic splitting.
3. Run a **batch prediction** job over a BigQuery table with `model.batch_predict(...)`, writing results back to BigQuery, and query the output table.
4. Compare the two: note the always-on cost of the warm online endpoint versus the pay-per-job cost of batch, and decide which each of your use cases should use. Then delete the endpoint to stop it billing.
