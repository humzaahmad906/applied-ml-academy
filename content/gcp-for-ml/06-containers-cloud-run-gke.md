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

Once images are flowing you spend most of your time listing, inspecting, and pruning. `repositories list/describe` shows what exists and how big it is, `docker images list` and `docker tags list` enumerate what you have pushed, and reading the **vulnerability scanning** results tells you whether a base image has a known CVE before you ship it.

```bash
gcloud artifacts repositories list --location=us-central1
gcloud artifacts repositories describe ml-images --location=us-central1

# What images/tags are in the repo?
gcloud artifacts docker images list us-central1-docker.pkg.dev/myco-fraud-dev/ml-images
gcloud artifacts docker tags list \
  us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/fraud-serve

# Read vulnerability scan results for a specific image digest
gcloud artifacts docker images describe \
  us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/fraud-serve:v1 \
  --show-package-vulnerability
```

Image storage is a silent cost sink — every CI run pushes a new tag and old ones pile up. **Cleanup policies** delete images by age or keep only the most recent N, and you can run them in `--dry-run` first to see what would go before you delete anything.

```bash
gcloud artifacts repositories set-cleanup-policies ml-images \
  --location=us-central1 --policy=cleanup.json --dry-run
```

Two repository modes save you from external dependencies. A **remote repository** proxies and caches an upstream like Docker Hub or PyPI, so a base-image pull that would hit a rate limit or an outage is served from your cache instead. A **virtual repository** presents several standard and remote repos behind one URL, so workloads pull from a single path while you reorganize the backing repos.

```bash
gcloud artifacts repositories create dockerhub-remote \
  --repository-format=docker --location=us-central1 \
  --mode=remote-repository --remote-docker-repo=DOCKER-HUB
```

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

# You can also deploy straight from source (Cloud Build builds the image via buildpacks)
gcloud run deploy prep-service --source . --region=us-central1
```

The `--source .` path is worth knowing: Cloud Run hands your directory to Cloud Build, which uses **buildpacks** to produce an image with no Dockerfile required — handy for a simple Python service, though for ML you usually want an explicit Dockerfile to pin CUDA and framework versions.

Key knobs: **concurrency** (how many simultaneous requests one instance handles — for a GPU model you often set this low so each request gets the accelerator), **min instances** (keep some warm to avoid cold starts on a latency-sensitive endpoint), **max instances** (a spend ceiling), and **scale-to-zero** (`--min-instances=0`) for cost when idle. By default Cloud Run only allocates CPU during a request; for a service that does background work or needs to stay warm you set **`--cpu-boost`** for faster cold starts or **`--no-cpu-throttling`** (CPU always allocated) at a higher cost.

You inspect and adjust running services with `list`, `describe`, and `update` rather than a full redeploy:

```bash
gcloud run services list --region=us-central1
gcloud run services describe fraud-serve --region=us-central1
gcloud run services update fraud-serve --region=us-central1 --max-instances=20
```

**Environment variables and secrets.** Pass plain config with `--set-env-vars`, but never put credentials there — mount them from Secret Manager with `--set-secrets`, which injects the secret as an env var or file and rotates cleanly. **Private networking.** To let the service reach private resources (a database on your VPC, a private Vertex endpoint) use **Direct VPC egress** (`--network`/`--subnet`, no connector to manage) or the older `--vpc-connector`.

```bash
gcloud run deploy fraud-serve \
  --image=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/fraud-serve:v1 \
  --region=us-central1 \
  --set-env-vars=MODEL_VERSION=v1 \
  --set-secrets=API_KEY=fraud-api-key:latest \
  --network=ml-vpc --subnet=ml-train-us --vpc-egress=private-ranges-only
```

**Traffic splitting for safe rollouts.** Each deploy creates a new immutable revision. By default all traffic shifts to the latest, but you can pin percentages across revisions for a canary — send 10% to the new model, watch metrics, then ramp to 100% (or roll back instantly by shifting traffic back).

```bash
# Deploy without taking traffic, then canary 10% to it
gcloud run deploy fraud-serve --image=...:v2 --region=us-central1 --no-traffic
gcloud run services update-traffic fraud-serve --region=us-central1 \
  --to-revisions=fraud-serve-v2=10
# Promote to 100% once it looks good
gcloud run services update-traffic fraud-serve --region=us-central1 --to-latest
```

**Jobs, not just services.** For run-to-completion work — batch scoring, a nightly backfill, a preprocessing step — use a Cloud Run **job**, which has no HTTP endpoint and exits when done. You can fan out with `--tasks` for embarrassingly parallel batches.

```bash
gcloud run jobs create score-batch \
  --image=us-central1-docker.pkg.dev/myco-fraud-dev/ml-images/scorer:v1 \
  --region=us-central1 --tasks=10 --max-retries=3 \
  --set-secrets=API_KEY=fraud-api-key:latest
gcloud run jobs execute score-batch --region=us-central1
```

Note: the current serverless-functions offering, Cloud Run functions, is built on this same infrastructure.

A few limits shape what fits on Cloud Run. Requests have a **maximum timeout** (configurable up to 60 minutes for services with `--timeout`), so genuinely long inference or training belongs in a job or on GKE. **Cold starts** add latency when scaling from zero — mitigate with `--min-instances=1` or `--cpu-boost` on latency-sensitive endpoints. And Cloud Run **GPUs are region-limited**: L4 is available in a subset of regions, so confirm your serving region supports the accelerator before you commit.

## GKE: Kubernetes when you need control

**Google Kubernetes Engine** is managed Kubernetes. You reach for it when Cloud Run's model is too constraining — you need custom networking, sidecars, complex multi-container topologies, fine-grained scheduling across many GPUs/TPUs, or a serving stack (like a high-throughput LLM inference server) that expects Kubernetes primitives.

Two operating modes:

- **Autopilot** — Google manages nodes; you declare workloads and their resource requests and pay for what pods use. Less operational burden; the recommended default unless you need node-level control.
- **Standard** — you manage node pools directly, including creating dedicated **GPU and TPU node pools** with specific accelerators. Maximum control, more responsibility.

Enable **Workload Identity** at cluster creation with `--workload-pool` so pods authenticate to Google APIs as service accounts with no key files — it is far easier to set at creation than to retrofit. After creating any cluster, fetch credentials so `kubectl` can talk to it; that single command writes your kubeconfig context.

```bash
# Autopilot cluster with Workload Identity (Google manages the nodes)
gcloud container clusters create-auto ml-cluster --region=us-central1 \
  --workload-pool=myco-fraud-dev.svc.id.goog

# Wire up kubectl to the cluster
gcloud container clusters get-credentials ml-cluster --region=us-central1
kubectl get nodes   # now targets the GKE cluster

# Standard cluster with a dedicated L4 GPU node pool for serving
gcloud container clusters create ml-std --zone=us-central1-a \
  --workload-pool=myco-fraud-dev.svc.id.goog
gcloud container node-pools create gpu-pool \
  --cluster=ml-std --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --num-nodes=1 --enable-autoscaling --min-nodes=0 --max-nodes=4
```

On **Standard** clusters you manage node pools directly — list them, resize, add specialized pools, and delete when done. Beyond GPUs you can attach **TPU** slices for large training/inference, and use **Spot** nodes (`--spot`) to run preemptible, cost-cheap inference or batch work at a steep discount:

```bash
gcloud container node-pools list --cluster=ml-std --zone=us-central1-a
# Spot GPU pool for cost-tolerant batch inference
gcloud container node-pools create spot-gpu --cluster=ml-std --zone=us-central1-a \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --spot --enable-autoscaling --min-nodes=0 --max-nodes=8
gcloud container node-pools delete spot-gpu --cluster=ml-std --zone=us-central1-a
```

The `--enable-autoscaling --min-nodes=0` combination is the cluster-autoscaler contract that lets a GPU pool scale to zero when idle and back up under load — the GKE analog of Cloud Run's scale-to-zero, and how you keep expensive accelerators from idling.

GKE is where large-scale, high-throughput inference platforms and custom distributed training often live. It has ML-focused capabilities — GPU/TPU scheduling, autoscaling that understands accelerators, and inference-optimized routing/gateway features — that make it the substrate for teams running many models at scale. Authentication to Google APIs from pods uses **Workload Identity Federation for GKE** (from the security module, set with the `--workload-pool` flag above), so pods act as Google service accounts with no key files.

Two GKE gotchas to plan for. **Node auto-upgrade** is on by default and will recreate nodes to keep them on a supported version — good for security, but it drains and reschedules pods, so configure a **maintenance window** and use PodDisruptionBudgets for serving tiers you cannot let blink. And like Cloud Run, **GPU and TPU availability is per-region/zone**; pick your cluster location around where the accelerator you need actually exists, and request quota ahead of time.

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

## CLI cheat-sheet

```bash
# --- Artifact Registry ---
gcloud artifacts repositories create ml-images --repository-format=docker --location=us-central1
gcloud artifacts repositories create dockerhub-remote --repository-format=docker \
  --location=us-central1 --mode=remote-repository --remote-docker-repo=DOCKER-HUB
gcloud artifacts repositories list --location=us-central1
gcloud artifacts repositories describe ml-images --location=us-central1
gcloud auth configure-docker us-central1-docker.pkg.dev
gcloud artifacts docker images list us-central1-docker.pkg.dev/PROJECT/ml-images
gcloud artifacts docker tags list us-central1-docker.pkg.dev/PROJECT/ml-images/fraud-serve
gcloud artifacts docker images describe IMAGE:TAG --show-package-vulnerability
gcloud artifacts repositories set-cleanup-policies ml-images --location=us-central1 \
  --policy=cleanup.json --dry-run

# --- Cloud Run services ---
gcloud run deploy fraud-serve --image=IMAGE --region=us-central1 \
  --gpu=1 --gpu-type=nvidia-l4 --cpu=4 --memory=16Gi \
  --min-instances=0 --max-instances=10 --concurrency=8 --no-allow-unauthenticated \
  --set-env-vars=MODEL_VERSION=v1 --set-secrets=API_KEY=fraud-api-key:latest \
  --network=ml-vpc --subnet=ml-train-us --vpc-egress=private-ranges-only
gcloud run deploy prep --source . --region=us-central1        # buildpacks, no Dockerfile
gcloud run services list --region=us-central1
gcloud run services describe fraud-serve --region=us-central1
gcloud run services update fraud-serve --region=us-central1 --max-instances=20
gcloud run services update-traffic fraud-serve --region=us-central1 --to-revisions=fraud-serve-v2=10
gcloud run services update-traffic fraud-serve --region=us-central1 --to-latest

# --- Cloud Run jobs ---
gcloud run jobs create score-batch --image=IMAGE --region=us-central1 --tasks=10 --max-retries=3
gcloud run jobs execute score-batch --region=us-central1

# --- GKE ---
gcloud container clusters create-auto ml-cluster --region=us-central1 \
  --workload-pool=PROJECT.svc.id.goog
gcloud container clusters get-credentials ml-cluster --region=us-central1
gcloud container node-pools create gpu-pool --cluster=ml-std --zone=us-central1-a \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --enable-autoscaling --min-nodes=0 --max-nodes=4
gcloud container node-pools create spot-gpu --cluster=ml-std --zone=us-central1-a \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 --spot \
  --enable-autoscaling --min-nodes=0 --max-nodes=8
gcloud container node-pools list --cluster=ml-std --zone=us-central1-a
```

## Try it

Ship a containerized model three ways and compare:

1. Write a tiny FastAPI/Flask app that returns a prediction, containerize it, and push the image to an Artifact Registry Docker repo.
2. Deploy it to **Cloud Run** with `--min-instances=0`, hit the URL, then watch it scale to zero when idle (observe in the Console that no instances bill at rest).
3. Redeploy the same image to Cloud Run with a GPU (`--gpu=1 --gpu-type=nvidia-l4`) and note what changed in startup and cost.
4. Create a small **Autopilot** GKE cluster and deploy the same image as a Kubernetes `Deployment` + `Service`. Reflect on the operational difference versus Cloud Run — and when the extra control would be worth it.
