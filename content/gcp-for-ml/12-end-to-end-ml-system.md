# 12 — Building an End-to-End ML System on GCP

Every module before this taught one instrument. This is where they play together. A production ML system is never "a model on Vertex AI" — it is a dozen Google Cloud services composed into a pipeline that ingests data, turns it into features, trains and evaluates models, deploys them safely, and watches them in production, all defined as code and governed by IAM. This module presents a reference architecture that wires those services together, shows the infrastructure-as-code that provisions it, and explains the design decisions that make it robust.

## The reference architecture

Consider a real-time fraud-scoring system for a payments product. It must score transactions in milliseconds, retrain as fraud patterns shift, and expose an analyst assistant that answers questions grounded in policy documents. Here is how the services compose, stage by stage.

**1. Ingestion and streaming.** Transaction events are published to **Pub/Sub**. Two consumers read the stream: a **Pub/Sub → BigQuery subscription** lands raw events directly for analytics, and a **Dataflow** streaming pipeline (Apache Beam) computes windowed, aggregated **features** (rolling transaction counts, velocity, merchant risk) in real time.

**2. Data lake and warehouse.** Bulk and historical data lives in **Cloud Storage** (the data lake); **BigQuery** is the warehouse where training sets are assembled with SQL, partitioned by date and clustered for cost. Raw, curated, and feature layers are separate datasets.

**3. Feature management.** Feature definitions are registered in the **Vertex AI Feature Store**: **feature groups** point at BigQuery source tables/views (the offline store), and **feature views** sync selected features into a **Bigtable-backed online store** for low-latency retrieval at serving time. This solves the training-serving skew problem structurally — training reads features from BigQuery, serving reads the *same* feature definitions from the online store.

**4. Training and evaluation.** A **Vertex AI Pipeline** (Kubeflow-based) orchestrates the workflow: pull the training set from BigQuery, run **Vertex AI custom training** on GPU (an A100/L4, on Spot with checkpointing), evaluate against a **frozen eval set**, and — gated by a metric threshold — register the model to the **Vertex AI Model Registry** with a new version.

**5. CI/CD.** Code changes trigger **Cloud Build** (or GitHub Actions authenticating via **Workload Identity Federation**, no keys): build the training and serving containers, push to **Artifact Registry**, run tests, and kick off the pipeline. Model promotion (moving the `production` alias to a new version) is gated on the eval.

**6. Orchestration.** The pipeline runs on a schedule and on new-data triggers via **Vertex AI Pipelines**; for cross-service, time-based orchestration that also touches BigQuery and Dataflow, **Cloud Composer** (managed Airflow) coordinates the broader workflow.

**7. Serving.** Real-time scoring runs on a **Vertex AI online prediction endpoint** (autoscaling, behind a **Private Service Connect** endpoint so traffic stays in the VPC), reading fresh features from the online store. Nightly bulk scoring runs as a **Vertex batch prediction** job (Spot workers, BigQuery in and out). The analyst assistant is a **Cloud Run** service calling **Gemini** on Vertex AI, grounded via the **RAG Engine** over policy documents in Cloud Storage.

**8. Monitoring and observability.** **Cloud Monitoring** dashboards track endpoint latency, error rate, and GPU duty cycle; **Vertex AI Model Monitoring** watches for training-serving skew and prediction drift and alerts when a feature distribution moves — feeding back into a retraining trigger.

**9. Security and governance.** Every workload runs as a **dedicated least-privilege service account**; secrets (third-party API keys, DB credentials) live in **Secret Manager**; sensitive data and services sit inside a **VPC Service Controls** perimeter with **private networking** and no external IPs.

**10. Cost.** Every resource carries **labels** (`team`, `env`, `component`); **billing export to BigQuery** enables per-component cost queries; **budgets with alerts** and Spot/committed-use discounts keep spend bounded.

## The data flow, end to end

The lifecycle of a single transaction and the model that scores it:

```
Transaction event
   → Pub/Sub topic
      → (a) BigQuery subscription  → raw events table (analytics)
      → (b) Dataflow streaming     → computed features → BigQuery feature tables
                                                       → Feature Store online sync (Bigtable)
Training loop (scheduled / triggered):
   BigQuery training set → Vertex Pipeline → custom training (GPU, Spot)
      → evaluate on frozen set → [gate] → Model Registry (new version)
      → CI/CD promotes `production` alias → redeploy endpoint
Serving:
   Live request → Vertex endpoint (PSC) → reads online features → prediction
   Nightly → Vertex batch prediction (Spot) → scores → BigQuery
   Analyst → Cloud Run → Gemini + RAG (policy docs) → grounded answer
Observability:
   Endpoint metrics → Cloud Monitoring; feature drift → Model Monitoring → retrain trigger
```

Notice the loop closes: monitoring detects drift, which triggers retraining, which produces a new registered version, which CI/CD promotes and redeploys. That feedback loop is what makes the system *live* rather than a one-time deployment.

## Infrastructure as code

None of this should be clicked together by hand. The whole system is provisioned with **Terraform** (the de facto IaC standard on Google Cloud, via the `google` provider; Google's own Config Controller and the older Deployment Manager also exist, but Terraform is the mainstream choice). IaC makes the environment reproducible, reviewable, and identical across `dev` and `prod`. A representative slice:

```hcl
# Enable the APIs the system needs
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com", "bigquery.googleapis.com",
    "pubsub.googleapis.com", "dataflow.googleapis.com",
    "run.googleapis.com", "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
  ])
  service = each.value
}

# Data lake bucket, region-co-located, lifecycle-managed
resource "google_storage_bucket" "data_lake" {
  name                        = "myco-fraud-data"
  location                    = "US-CENTRAL1"
  uniform_bucket_level_access = true
  autoclass { enabled = true }
  labels = { team = "fraud", env = "prod", component = "data" }
}

# Warehouse dataset
resource "google_bigquery_dataset" "fraud" {
  dataset_id = "fraud"
  location   = "us-central1"
  labels     = { team = "fraud", env = "prod", component = "data" }
}

# Ingestion topic and direct-to-BigQuery subscription
resource "google_pubsub_topic" "transactions" {
  name   = "transactions"
  labels = { team = "fraud", env = "prod", component = "ingest" }
}

# Least-privilege service account for training
resource "google_service_account" "training" {
  account_id   = "training-sa"
  display_name = "Vertex AI training jobs"
}

resource "google_project_iam_member" "training_bq" {
  project = "myco-fraud-prod"
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.training.email}"
}

# Artifact Registry for containers
resource "google_artifact_registry_repository" "images" {
  repository_id = "ml-images"
  location      = "us-central1"
  format        = "DOCKER"
}
```

The training pipeline, serving container, and endpoint deployment are then driven by CI/CD against this provisioned infrastructure — code builds the images and submits the pipeline, Terraform owns the durable resources.

## Design principles that hold it together

- **Separation of concerns.** Ingestion, features, training, serving, and monitoring are independent stages connected by well-defined interfaces (a Pub/Sub topic, a BigQuery table, a registry alias). Any stage can be changed without rewriting the others.
- **The registry as the contract** between training and serving. Serving references a stable alias; training pushes new versions behind it.
- **Gate on a frozen eval.** No model reaches production without clearing a measured threshold on an unchanging test set.
- **Least privilege and private by default.** Dedicated service accounts, keyless auth, Secret Manager, private endpoints, VPC-SC.
- **Everything as code.** Terraform for infrastructure, KFP for pipelines, containers for environments — the whole system is reconstructable from a repo.
- **Close the loop.** Monitoring feeds retraining; the system maintains itself rather than decaying silently.

## How this fits the whole solution

This *is* the whole solution — the point where projects, IAM, compute, storage, networking, containers, functions, the data plane, Vertex training and prediction, and Gemini stop being separate topics and become one system. The skill this course builds toward is not operating any single service but composing them: knowing that a fraud score needs a streaming feature pipeline behind it, that a Gemini assistant needs RAG and a private container, that a model needs a gated pipeline and drift monitoring, and that all of it needs to be codified, secured, and costed. That composition is what "GCP for ML engineers" means.

## Key takeaways

- A production ML system composes ~a dozen services across **ingestion (Pub/Sub, Dataflow), lake+warehouse (Cloud Storage, BigQuery), feature management (Feature Store: BigQuery offline + Bigtable online), training+registry (Vertex Pipelines, custom training, Model Registry), CI/CD (Cloud Build / GitHub Actions + Workload Identity Federation), serving (Vertex endpoints/batch, Cloud Run + Gemini/RAG), monitoring (Cloud Monitoring, Model Monitoring), security (IAM, Secret Manager, VPC-SC), and cost (labels, budgets)**.
- The **Model Registry alias** is the contract between training and serving; **gate promotion on a frozen eval**; **close the loop** so drift monitoring triggers retraining.
- Provision durable infrastructure with **Terraform** (IaC) and drive pipelines/deployments with CI/CD — the whole system is reconstructable from a repository.
- Design for **separation of concerns, least privilege, private-by-default networking, and cost attribution** from the start.

## Try it

Assemble a minimal end-to-end slice:

1. In Terraform, provision the durable pieces: a data-lake bucket, a BigQuery dataset, a Pub/Sub topic with a BigQuery subscription, an Artifact Registry repo, and a least-privilege training service account. `terraform apply`.
2. Publish a few events to the topic and confirm they land in BigQuery; assemble a small training table with SQL.
3. Run a Vertex AI Pipeline that trains on that table, evaluates against a frozen set, and registers the model only if it passes — then deploy the registered version to an endpoint.
4. Add a Cloud Run service that calls Gemini with RAG over a couple of documents, and a Cloud Monitoring dashboard plus a Model Monitoring config on the endpoint. Step back and trace one request through every service — you have built the whole loop.
