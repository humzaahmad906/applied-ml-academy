# 06 — Containers: Artifact Registry, Cloud Run, GKE

Containers are how modern ML gets from a laptop to production reproducibly. Your model, its dependencies, the exact CUDA and framework versions, and your serving code all get baked into an image, and that image runs identically on your machine, in a training job, and behind a live endpoint. This module covers the three services that make containerized ML work on Google Cloud: **Artifact Registry** (where images live), **Cloud Run** (serverless containers), and **Google Kubernetes Engine / GKE** (Kubernetes for when you need full control).

## Artifact Registry: where your images live

**Artifact Registry** is Google Cloud's package and image store. It fully replaces the old Container Registry, which was shut down in early 2025 — for anything new, Artifact Registry is the only choice. Beyond Docker/OCI container images it also hosts Python, npm, Apt, Maven, and other artifact types, so it can be the single home for both your serving images and your internal Python wheels.

You create a repository (typically Docker-format), authenticate Docker to it, then push and pull:

```bash
# Create a Docker repository in your compute region
gcloud artifacts repositories create ml-images \
  --repository-format=docker \
  --location=us-central1

# Configure Docker to authenticate to Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build, tag, and push a serving image
docker build -t us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/fraud-serve:v1 .
docker push us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/fraud-serve:v1
```

The image path grammar is `LOCATION-docker.pkg.dev/PROJECT/REPO/IMAGE:TAG`. Every downstream service — Cloud Run, GKE, Vertex AI custom training and prediction — pulls from this path. Artifact Registry also scans images for vulnerabilities and integrates with IAM so you grant `roles/artifactregistry.writer` to CI and `roles/artifactregistry.reader` to the service accounts that run your workloads.

## Cloud Run: serverless containers

**Cloud Run** runs your container without any cluster to manage. You hand it an image; it scales the number of instances up with traffic and **down to zero** when idle, and you pay per request-time. It is the fastest path to a production HTTP service and, for a large fraction of ML serving, the right default.

Two flavors:

- **Cloud Run services** — long-lived, request-driven HTTP endpoints (your inference API).
- **Cloud Run jobs** — run-to-completion tasks (batch scoring, a preprocessing step, a one-off backfill), no HTTP endpoint.

What makes Cloud Run genuinely good for ML now is **GPU support**, which is generally available: you can attach an **NVIDIA L4** (and larger Blackwell-class GPUs) to a Cloud Run service, with fast instance startup and scale-to-zero, and **no quota request required** for L4. That combination — serverless, pay-per-use, scale-to-zero, *with a GPU* — is uniquely suited to bursty or intermittent model serving where a dedicated always-on GPU endpoint would sit idle and expensive.

```bash
# Deploy a GPU-backed inference service that scales to zero
gcloud run deploy fraud-serve \
  --image=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/fraud-serve:v1 \
  --region=us-central1 \
  --gpu=1 --gpu-type=nvidia-l4 \
  --cpu=4 --memory=16Gi \
  --min-instances=0 --max-instances=10 \
  --concurrency=8 \
  --no-allow-unauthenticated \
  --service-account=serving-sa@myco-fraud-dev.iam.gserviceaccount.com

# You can also deploy straight from source (Cloud Build builds the image for you)
gcloud run deploy prep-service --source . --region=us-central1
```

Key knobs: **concurrency** (how many simultaneous requests one instance handles — for a GPU model you often set this low so each request gets the accelerator), **min instances** (keep some warm to avoid cold starts on a latency-sensitive endpoint), **max instances** (a spend ceiling), and **scale-to-zero** (`--min-instances=0`) for cost when idle. Cloud Run also handles secrets, VPC connectivity, and authentication cleanly. Note: the current serverless-functions offering, Cloud Run functions, is built on this same infrastructure.

## GKE: Kubernetes when you need control

**Google Kubernetes Engine** is managed Kubernetes. You reach for it when Cloud Run's model is too constraining — you need custom networking, sidecars, complex multi-container topologies, fine-grained scheduling across many GPUs/TPUs, or a serving stack (like a high-throughput LLM inference server) that expects Kubernetes primitives.

Two operating modes:

- **Autopilot** — Google manages nodes; you declare workloads and their resource requests and pay for what pods use. Less operational burden; the recommended default unless you need node-level control.
- **Standard** — you manage node pools directly, including creating dedicated **GPU and TPU node pools** with specific accelerators. Maximum control, more responsibility.

```bash
# Autopilot cluster (Google manages the nodes)
gcloud container clusters create-auto ml-cluster --region=us-central1

# Standard cluster with a dedicated L4 GPU node pool for serving
gcloud container clusters create ml-std --zone=us-central1-a
gcloud container node-pools create gpu-pool \
  --cluster=ml-std --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --num-nodes=1 --enable-autoscaling --min-nodes=0 --max-nodes=4
```

GKE is where large-scale, high-throughput inference platforms and custom distributed training often live. It has ML-focused capabilities — GPU/TPU scheduling, autoscaling that understands accelerators, and inference-optimized routing/gateway features — that make it the substrate for teams running many models at scale. Authentication to Google APIs from pods uses **Workload Identity Federation for GKE** (from the security module), so pods act as Google service accounts with no key files.

## Choosing between them — and Vertex AI

A practical decision guide for ML serving and jobs:

- **Cloud Run** — default for HTTP model serving and containerized batch jobs; scale-to-zero, GPU support, minimal ops. Best when you want to ship a container and not think about infrastructure.
- **GKE** — when you need Kubernetes-level control: complex topologies, many models, custom inference servers, dense GPU/TPU packing, or an existing K8s platform.
- **Vertex AI prediction** — when you want a fully managed ML-serving control plane with model registry, versioning, traffic splitting, and built-in monitoring, and you are happy to deploy models rather than raw containers.

These are not mutually exclusive — a real system often uses Cloud Run for lightweight and GenAI-proxy services, Vertex endpoints for registered models, and GKE for a specialized high-throughput serving tier, all pulling images from the same Artifact Registry.

## How this fits the whole solution

Artifact Registry is the shared image store the whole pipeline builds against: CI (Cloud Build or GitHub Actions) pushes training and serving images there, Vertex AI custom training and prediction pull from it, and Cloud Run and GKE deploy from it. Cloud Run and GKE are two of the three serving surfaces in the reference architecture — Cloud Run for cost-efficient, scale-to-zero and GenAI-facing services, GKE for control-hungry high-throughput serving, and Vertex endpoints for managed registered models. Standardizing on containers in Artifact Registry is what makes every stage reproducible and portable across these surfaces.

## Key takeaways

- **Artifact Registry** replaces the retired Container Registry and is the single home for container images (and Python/other artifacts); push with `gcloud auth configure-docker` + `docker push` to `LOCATION-docker.pkg.dev/...`.
- **Cloud Run** runs containers serverlessly with **scale-to-zero and GA GPU support (L4, no quota request)** — the default for HTTP model serving and containerized batch jobs; tune concurrency, min/max instances.
- **GKE** (Autopilot vs Standard) is for Kubernetes-level control: custom topologies, dense GPU/TPU scheduling, and high-throughput inference platforms; pods authenticate via Workload Identity Federation.
- Pick **Cloud Run** for simplicity, **GKE** for control, **Vertex AI prediction** for a managed ML-serving control plane — often combined, all sharing Artifact Registry.

## Try it

Ship a containerized model three ways and compare:

1. Write a tiny FastAPI/Flask app that returns a prediction, containerize it, and push the image to an Artifact Registry Docker repo.
2. Deploy it to **Cloud Run** with `--min-instances=0`, hit the URL, then watch it scale to zero when idle (observe in the Console that no instances bill at rest).
3. Redeploy the same image to Cloud Run with a GPU (`--gpu=1 --gpu-type=nvidia-l4`) and note what changed in startup and cost.
4. Create a small **Autopilot** GKE cluster and deploy the same image as a Kubernetes `Deployment` + `Service`. Reflect on the operational difference versus Cloud Run — and when the extra control would be worth it.
