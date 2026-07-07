# Module 12 — DevOps and Deployment

## Why this module matters

Every earlier module in this course taught you to design and train. This one teaches you to ship. A model that works on a laptop but never reaches users is a research artifact, not a product — and the gap between the two is exactly what this module covers.

This chapter is the consolidated reference for containers, Kubernetes, and deployment CI/CD. The training chapter points here for the full container treatment (multi-stage CUDA builds, local testing). The serving chapter points here for the Kubernetes mechanics (GPU resource requests, rolling updates, secrets). You do not need to repeat those cross-references when answering interview questions; you need to be able to reproduce any manifest or workflow step from memory.

Junior engineers know how to write a model. Senior engineers know how to containerize it, gate its release against an eval harness, deploy it without downtime, scale it under load, and roll it back in under five minutes when something goes wrong. Interviewers probe all five. This module builds that operational picture end to end.

---

## 1. Containerization

### A working Dockerfile for FastAPI + PyTorch

The packaging contract is simple: the image must run identically on a developer laptop, a staging GPU node, and a production GPU node, with no dev toolchain or secrets embedded. Multi-stage builds deliver this. The builder stage holds the CUDA devel image — which carries compilers and headers needed to build or verify compiled extensions — and the runtime stage holds only the CUDA libraries the running process actually needs. The image size drop is typically 60–70%.

```dockerfile
# syntax=docker/dockerfile:1
# Representative base images as of 2026 — verify at
# https://hub.docker.com/r/nvidia/cuda/tags before pinning.

# ── Stage 1: builder ────────────────────────────────────────────────
# Full CUDA 12.4 devel image: compilers, headers, toolkit.
# Use this if any package builds a native extension against CUDA.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch against the matching CUDA wheel index first;
# then install the rest of the service deps.
# --prefix=/install isolates the packages so the runtime stage
# can copy them without carrying along any build toolchain.
RUN python3.11 -m pip install --upgrade pip --no-cache-dir \
 && python3.11 -m pip install --no-cache-dir --prefix=/install \
        "torch>=2.4" \
        --index-url https://download.pytorch.org/whl/cu124

WORKDIR /build
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ────────────────────────────────────────────────
# CUDA 12.4 runtime + cuDNN: only shared libraries the process needs.
# No compilers, no headers, no Python dev tools.
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Pull the installed packages from the builder — nothing else.
COPY --from=builder /install /usr/local

# Copy application code last: deps layer is large and rarely changes;
# code layer changes frequently. This order maximises cache hits.
COPY app/ /app

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Exec form (no shell wrapper) ensures SIGTERM reaches uvicorn directly.
# Workers=1: each process loads the model; multiple workers on one GPU
# serialize all GPU calls anyway and multiply memory usage.
CMD ["python3.11", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

`requirements.txt` contains only what the running service needs — `fastapi`, `uvicorn[standard]`, `pydantic`, and any inference utilities. Dev tooling (`pytest`, `black`, `mypy`, type stubs) must not appear here; they belong in a separate `requirements-dev.txt` that never touches the Docker build.

A minimal FastAPI service with correct startup and health probes:

```python
# app/main.py
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    inputs: list[float]


class PredictResponse(BaseModel):
    outputs: list[float]
    model_version: str


_model: torch.nn.Module | None = None
_model_version: str = "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version
    model_path = os.environ["MODEL_PATH"]
    checkpoint = torch.load(
        model_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        weights_only=True,
    )
    # Replace with your actual model class and load logic.
    # _model = MyModel(**checkpoint["config"]).eval()
    # _model.load_state_dict(checkpoint["state_dict"])
    _model_version = checkpoint.get("version", "unknown")
    logger.info("model loaded from %s version=%s", model_path, _model_version)
    yield
    _model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    with torch.inference_mode():
        x = torch.tensor(req.inputs, dtype=torch.float32).cuda()
        out = _model(x)
    return PredictResponse(outputs=out.cpu().tolist(), model_version=_model_version)
```

The separate `/health` and `/ready` endpoints matter: Kubernetes uses liveness to decide whether to restart the container, and readiness to decide whether to send traffic. Model loading can take 30–120 seconds; using a single endpoint for both will cause Kubernetes to kill the pod before the model finishes loading.

### Testing locally

```bash
# Build the image (runs both stages)
docker build -t inference-service:dev .

# Run with GPU access; bind-mount a local model checkpoint for testing.
# Requires NVIDIA Container Toolkit installed on the host.
docker run --gpus all --rm \
  -p 8000:8000 \
  -e MODEL_PATH=/model/checkpoint.pt \
  -v /path/to/local/model:/model \
  inference-service:dev

# Confirm the GPU is visible inside the container:
docker run --gpus all --rm nvidia/cuda:12.4.1-devel-ubuntu22.04 nvidia-smi

# Smoke test the service:
curl -s http://localhost:8000/ready
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [1.0, 2.0, 3.0]}'
```

Test the multi-stage build locally before pushing. The builder stage can mask missing runtime-stage deps that only surface when the container is actually running.

### The CUDA/driver compatibility gotcha

This is the single most common deployment failure for ML services and it is entirely avoidable. The NVIDIA Container Toolkit exposes the host node's driver libraries inside the container at runtime — the driver is not in the image. The rule: the CUDA toolkit version in your container image must be **less than or equal to** the maximum CUDA version your host driver supports.

Check the host node's CUDA ceiling with `nvidia-smi`: the `CUDA Version: X.Y` in its output is the maximum, not the installed toolkit version. If your image uses `cuda:12.6-runtime` (requires driver >= 560) but your cluster nodes run driver 520 (supports up to CUDA 11.8), the container will start and then fail at the first CUDA call with a cryptic `no kernel image available` error — not at container start, which makes it hard to catch without a GPU-specific health check.

Before locking a base image version: query all your K8s node pools with `kubectl get nodes -o json | jq '.items[].status.nodeInfo.kernelVersion'` and confirm the driver version with the node's cloud console or `kubectl describe node <name>`. Pin the container CUDA version to the lowest driver ceiling across your node pools.

---

## 2. Kubernetes

### A minimal Deployment and Service

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
  namespace: ml-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: inference-service
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0   # never reduce capacity below the desired count
      maxSurge: 1         # allow one extra pod during the rollout
  template:
    metadata:
      labels:
        app: inference-service
    spec:
      containers:
        - name: inference-service
          image: registry.example.com/inference-service:v1.2.0
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: /model/checkpoint.pt
            - name: MODEL_REGISTRY_TOKEN
              valueFrom:
                secretKeyRef:
                  name: model-registry-creds
                  key: token
          resources:
            requests:
              memory: "16Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
            limits:
              memory: "24Gi"
              cpu: "8"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: model-cache
              mountPath: /model
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 30   # model loading window
            periodSeconds: 5
            failureThreshold: 24      # 24 × 5s = 2 min before failing
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc
      # Tolerate the taint GPU nodes typically carry so the pod is
      # eligible for scheduling onto GPU nodes without a custom scheduler.
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: inference-service
  namespace: ml-serving
spec:
  selector:
    app: inference-service
  ports:
    - name: http
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### GPU resource requests and limits

The `nvidia.com/gpu` resource is provided by the NVIDIA device plugin for Kubernetes, which runs as a DaemonSet on every GPU node and advertises each physical GPU as a schedulable resource. A few non-obvious rules:

- **Requests must equal limits** for `nvidia.com/gpu`. Unlike CPU, fractional GPU allocation via the standard device plugin is not supported (time-slicing and MIG are separate mechanisms with their own tradeoffs).
- Setting `nvidia.com/gpu: 1` in both `requests` and `limits` is the correct, standard form. Omitting the resource entirely causes the pod to land on a GPU node by chance (if GPU nodes tolerate other workloads) without exclusive GPU access — a race condition that surfaces as intermittent OOM failures.
- Set memory `requests` conservatively (what the model needs at baseline load) and `limits` aggressively enough to not OOM under a spike. CPU requests affect scheduling; CPU limits trigger throttling — keep them proportionate.

In interviews, volunteering that you need the NVIDIA device plugin and what happens if you omit the resource request is a senior signal. Most candidates know `nvidia.com/gpu: 1` exists; fewer know why and what breaks without it.

### Rolling updates and canary

`maxUnavailable: 0` with `maxSurge: 1` means Kubernetes brings up one new pod, waits for its readiness probe to pass, then terminates one old pod — this is the zero-downtime rolling update. It only works if the readiness probe correctly reflects the model load state (which the `/ready` endpoint above does). With a liveness-only probe or a trivially fast readiness check, you will route traffic to pods that haven't finished loading the model.

**Canary pattern.** A full canary (route 5% of traffic to a new version, 95% to the stable version) requires either a service mesh (Istio/Linkerd with traffic-split VirtualServices) or a separate Deployment with a shared Service selector using a weight-aware ingress. The minimal approach — a second Deployment named `inference-service-canary` with `replicas: 1` alongside a stable Deployment with `replicas: 19` — gives roughly 5% traffic naturally through a shared Service selector. It is crude (traffic split depends on replica counts, not weights) but deployable without a service mesh. Use Argo Rollouts if you need a proper canary lifecycle with automated analysis.

### Secrets — model-registry credentials and API keys

Never bake credentials into an image. The failure mode is obvious — anyone with the image has the secret — but it still happens because it is convenient at prototype time and never cleaned up.

```yaml
# k8s/secret.yaml
# In production, populate this via the External Secrets Operator
# (syncing from AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault)
# rather than storing literal values in a manifest committed to git.
apiVersion: v1
kind: Secret
metadata:
  name: model-registry-creds
  namespace: ml-serving
type: Opaque
stringData:
  token: "REPLACE_WITH_EXTERNAL_SECRETS_OPERATOR_OR_CI_INJECTION"
```

The Deployment manifest above references this Secret via `secretKeyRef`, injecting the token as an environment variable. The alternative — `envFrom: secretRef` for the whole Secret object — is acceptable but injects every key in the Secret as an env var, which can silently shadow other env vars if key names collide.

For production, the External Secrets Operator (ESO) pattern is the right answer: ESO syncs secrets from your cloud provider's secret store into K8s `Secret` objects on a configurable refresh interval, so there are no literal secret values in any manifest or git repo.

---

## 3. Autoscaling

### HPA vs KEDA

**HPA (Horizontal Pod Autoscaler)** is built into Kubernetes. It polls the Metrics API on a configurable interval (default 15s) and scales the target Deployment up or down based on observed metric values. Out of the box it handles CPU and memory utilization; custom metrics (queue depth, GPU utilization, request latency) require a custom metrics adapter that bridges your observability stack (usually Prometheus) to the K8s Custom Metrics API.

```yaml
# HPA scaling on CPU utilization — simple, works without any extra operator.
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-service-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

HPA's limitation: it cannot scale from zero. `minReplicas: 0` is only supported for certain metrics types and requires explicit configuration. For workloads with bursty traffic from a message queue, HPA is also reactive in the wrong direction — it sees high CPU only after the service is already overloaded, not when messages are piling up upstream.

**KEDA (Kubernetes Event-Driven Autoscaler)** fills the gap. It is a CNCF project that runs as a controller alongside the standard HPA and lets you scale on event-source metrics directly — queue depth in RabbitMQ/SQS/Kafka/Redis, Prometheus queries, Azure/GCP/AWS-native metrics — and it supports `minReplicaCount: 0` natively.

```yaml
# KEDA ScaledObject: scale the Deployment based on a RabbitMQ queue length.
# Requires the KEDA operator installed in the cluster.
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: inference-service-keda
  namespace: ml-serving
spec:
  scaleTargetRef:
    name: inference-service
  minReplicaCount: 1       # keep at least one replica to avoid cold starts
  maxReplicaCount: 20
  pollingInterval: 10      # seconds between checks
  cooldownPeriod: 60       # seconds before scaling back down after a burst
  triggers:
    - type: rabbitmq
      metadata:
        protocol: amqp
        queueName: inference-requests
        host: amqp://rabbitmq.ml-serving.svc:5672
        mode: QueueLength
        value: "5"          # target: ≤5 messages per replica
```

### Scaling on GPU-utilization as a custom metric

To expose GPU utilization to the HPA: deploy **DCGM Exporter** as a DaemonSet on GPU nodes → it emits per-GPU Prometheus metrics (e.g., `DCGM_FI_DEV_GPU_UTIL`) → **Prometheus Adapter** translates those metrics into the K8s Custom Metrics API → HPA references the custom metric by name. The setup has several moving parts, but in production it is the correct approach for GPU-bound services where CPU utilization is a poor proxy for actual load.

The KEDA Prometheus trigger is a simpler path for the same goal: configure a KEDA ScaledObject with `type: prometheus` and a PromQL query such as `avg(DCGM_FI_DEV_GPU_UTIL{namespace="ml-serving"})`, with a target value of 70 (representing 70% GPU utilization as the scale-out threshold).

### The cold-start problem

Scale-to-zero is attractive on paper — zero cost when idle — but it fails silently for large model inference services. A 7B model checkpoint is roughly 14 GB in BF16; loading it from a network-attached PVC takes 20–60 seconds on a typical cluster storage backend, plus 5–15 seconds for CUDA initialization and model warm-up. The first request after the pod reaches zero replicas stalls for the entire cold-start duration — which for LLM-class models (dozens to hundreds of GBs) can be 2–5 minutes. Users experience this as a timeout, not a loading indicator.

Practical mitigations, in order of increasing infrastructure cost:

1. **Set `minReplicaCount: 1`.** Keep at least one warm replica at all times. Costs idle GPU hours but eliminates cold starts entirely. Right answer for latency-sensitive products.
2. **Use local NVMe storage for model weights.** Network-attached PVCs (NFS, GCS Fuse, EFS) are the primary bottleneck. Node-local SSDs mounted as `hostPath` or via a local PV can cut load time by 3–5×. Tradeoff: the model must be pre-staged on every GPU node, or the pod must be pinned to a specific node (losing scheduling flexibility).
3. **Async startup with response queuing.** Accept requests immediately, queue them in memory, return results when the model finishes loading. Requires client-side retry or a message-queue frontend — adds complexity, appropriate only when cold starts are genuinely infrequent and tolerable.
4. **Separate the warmup path.** A lightweight "keepwarm" CronJob that pings the service every N minutes prevents the KEDA cooldown from scaling to zero during low-traffic periods without running continuously.

In interviews: stating "I'd set minReplicas to 1 and not scale to zero for a latency-sensitive LLM service" is the correct and senior answer. "I'd scale to zero to save money" sounds cost-conscious but shows unfamiliarity with the operational reality.

---

## 4. CI/CD

### The full GitHub Actions pipeline

The goal is a pipeline that runs the eval harness before anything is deployed, builds a deterministically tagged image, runs integration tests in staging, and requires a human approval gate before touching production.

```yaml
# .github/workflows/deploy.yaml
name: ML Service Deploy

on:
  push:
    branches: [main]

env:
  REGISTRY: registry.example.com
  IMAGE_NAME: inference-service

jobs:

  # ── 1. Run the eval harness before building anything ──────────────
  eval:
    name: Eval harness
    runs-on: [self-hosted, gpu]   # self-hosted runner with GPU access
    steps:
      - uses: actions/checkout@v4

      - name: Pull model artifact from registry
        env:
          MODEL_REGISTRY_TOKEN: ${{ secrets.MODEL_REGISTRY_TOKEN }}
          MODEL_URI: ${{ secrets.MODEL_REGISTRY_URI }}
        run: |
          python scripts/pull_artifact.py \
            --model-uri "$MODEL_URI" \
            --output-dir ./model_artifact

      - name: Run offline eval harness
        run: |
          python evaluate.py \
            --config eval/config.yaml \
            --model-dir ./model_artifact \
            --output eval_results.json

      - name: Enforce eval gates
        # Fails the job (and blocks the pipeline) if metrics fall below thresholds.
        run: |
          python scripts/check_eval_gate.py \
            --results eval_results.json \
            --gate eval/gates.yaml

      - uses: actions/upload-artifact@v4
        with:
          name: eval-results-${{ github.sha }}
          path: eval_results.json

  # ── 2. Build and push the container image ─────────────────────────
  build:
    name: Build and push
    needs: eval
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.version }}
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3

      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ secrets.REGISTRY_USER }}
          password: ${{ secrets.REGISTRY_PASSWORD }}

      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=,format=short

      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          # BuildKit layer cache stored in the registry — no external cache service needed.
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache
          cache-to: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:buildcache,mode=max

  # ── 3. Deploy to staging ──────────────────────────────────────────
  deploy-staging:
    name: Deploy to staging
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4

      - name: Patch staging manifest with new image tag
        run: |
          sed -i \
            "s|image: .*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build.outputs.image-tag }}|g" \
            k8s/staging/deployment.yaml

      - name: Apply to staging cluster
        env:
          KUBECONFIG_DATA: ${{ secrets.STAGING_KUBECONFIG }}
        run: |
          echo "$KUBECONFIG_DATA" | base64 -d > /tmp/kubeconfig
          KUBECONFIG=/tmp/kubeconfig kubectl apply -f k8s/staging/ -n ml-serving-staging
          KUBECONFIG=/tmp/kubeconfig kubectl rollout status \
            deployment/inference-service -n ml-serving-staging --timeout=5m

  # ── 4. Integration tests against staging ──────────────────────────
  integration-test:
    name: Integration tests
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run integration test suite
        env:
          SERVICE_URL: ${{ secrets.STAGING_SERVICE_URL }}
        run: |
          python -m pytest tests/integration/ \
            -v \
            --timeout=60 \
            --tb=short

  # ── 5. Deploy to production (requires manual approval) ────────────
  deploy-prod:
    name: Deploy to production
    needs: integration-test
    runs-on: ubuntu-latest
    # The "production" environment in GitHub settings must have a
    # required reviewer configured — this is the approval gate.
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Patch prod manifest with new image tag
        run: |
          sed -i \
            "s|image: .*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build.outputs.image-tag }}|g" \
            k8s/prod/deployment.yaml

      - name: Apply to production cluster
        env:
          KUBECONFIG_DATA: ${{ secrets.PROD_KUBECONFIG }}
        run: |
          echo "$KUBECONFIG_DATA" | base64 -d > /tmp/kubeconfig
          KUBECONFIG=/tmp/kubeconfig kubectl apply -f k8s/prod/ -n ml-serving
          KUBECONFIG=/tmp/kubeconfig kubectl rollout status \
            deployment/inference-service -n ml-serving --timeout=10m
```

A few design decisions worth defending in an interview:

- The `eval` job runs on a `self-hosted` GPU runner. The eval harness runs against the current champion model checkpoint (not a newly trained one); it gates the code change, ensuring that application changes don't regress the deployed model's behavior. Model promotion is a separate event in the model registry (described in the eval and observability chapter).
- The `build` job only runs if `eval` passes. The image tag is the git short SHA — immutable, traceable, never "latest". Using `latest` as a production image tag makes rollback impossible without additional state.
- Staging and production use separate Kubernetes clusters (separate `KUBECONFIG` secrets). Using namespaces in a single cluster is cheaper but couples the blast radius.
- The production deployment is gated by a GitHub Environment with required reviewers. This is a free, auditable approval mechanism that most teams already have access to — no external deployment tool required for a first implementation.

### Model artifacts live in a registry, not the image

The Dockerfile above does not `COPY` any model weights. This is deliberate. A 7B model checkpoint is roughly 14 GB; a frontier model is hundreds of GB. Embedding weights in the image means:

- Registry push/pull times are measured in hours, not seconds.
- Every code change triggers a rebuild and re-upload of the weights, even if the weights haven't changed.
- Image versioning and model versioning become conflated, making it impossible to deploy a new model version without rebuilding the image (and vice versa).

The correct architecture: model weights live in a model registry (MLflow Model Registry, Weights & Biases Artifacts, or a cloud object store with versioning enabled). At container startup, the init sequence pulls the weights from the registry into a PVC or ephemeral volume. The `MODEL_REGISTRY_TOKEN` secret and `MODEL_URI` env var enable this without embedding credentials or weights in the image. The eval and observability chapter covers the full model registry and lineage workflow.

### Versioning and rollback

Every production deployment should answer two questions instantly: "what version is running?" and "how do I go back?"

**What is running:** `kubectl describe deployment inference-service -n ml-serving` shows the current image tag under `Images:`. The git SHA tag makes this unambiguous.

**Rollback procedure:**

```bash
# View rollout history (shows previous image tags if annotated):
kubectl rollout history deployment/inference-service -n ml-serving

# Roll back to the immediately previous version (seconds, no rebuild):
kubectl rollout undo deployment/inference-service -n ml-serving

# Roll back to a specific revision:
kubectl rollout undo deployment/inference-service -n ml-serving --to-revision=5

# Confirm the rollback completed:
kubectl rollout status deployment/inference-service -n ml-serving
```

`kubectl rollout undo` reuses the previous Deployment spec — no image rebuild, no registry push. The rollback time is bounded by the pod startup time (model load + readiness probe), not by the CI/CD pipeline. For a service with a 60-second model load time, rollback-to-serving is roughly 90 seconds. Know this number for your service and state it in interviews.

For model-level rollbacks (new model weights introduced a regression): update the `MODEL_URI` env var in the Deployment manifest to point to the previous model version in the registry and re-apply. This is a config change, not an image rebuild, and it completes at the same pod-startup speed.

---

## 5. Day-2 Operations

Getting to first deploy is the easy part. What you do the week after is what separates teams that improve from teams that plateau.

**Monitoring hookup.** Wire the service to your infrastructure observability stack (Prometheus + Grafana or your cloud provider's equivalent) for pod-level metrics: request rate, error rate, latency (p50/p95/p99), GPU utilization (via DCGM Exporter), and memory pressure. These catch infrastructure failures — OOM kills, node evictions, CUDA errors logged as container restarts. For model-level monitoring — input drift, prediction-distribution drift, online quality sampling, the production flywheel — follow the full framework in the evaluation and observability chapter. Infrastructure monitoring and model monitoring are complementary layers; teams that only instrument one of them miss half the incident classes.

**Model updates.** When a new model checkpoint is promoted in the model registry, update the `MODEL_URI` env var in the Deployment manifest and apply it. This triggers a rolling restart (pods pull the new weights from the registry on startup). Run the eval harness against the new checkpoint before promoting it — the model registry's promotion gate is the same CI mechanism described in the eval and observability chapter's champion/challenger workflow.

**Rollback drill.** Every team should execute a rollback in staging at least once per quarter — deliberately deploy a bad version and time the full cycle: detect → decide → `kubectl rollout undo` → confirm healthy. Teams that have never drilled rollback discover their rollback procedure is broken at the worst possible time. Common failure modes: the previous image was evicted from the registry (use an immutable registry with retention policies); the old PVC model weights were deleted (keep at least two versions in the model registry); the `KUBECONFIG` secret for the prod cluster expired (rotate secrets on a schedule). Identify and fix these in staging, not under a live incident.

---

## References

- The NVIDIA container runtime documentation and the CUDA compatibility matrix are the authoritative sources for the base-image version decision — consult them before picking a CUDA version for a new project.
- The KEDA project documentation covers all available scalers and their configuration; the RabbitMQ and Prometheus scalers are the two most commonly used in ML serving pipelines.
- The NVIDIA device plugin for Kubernetes repository documents GPU resource requests, time-slicing configuration, and MIG (Multi-Instance GPU) partitioning for workloads that don't need a full GPU.
- Argo Rollouts (a CNCF project) provides production-grade canary and blue/green release controllers for Kubernetes with automated metric-based analysis — the natural upgrade from the manual canary pattern described here.
- The evaluation and observability chapter covers the model registry and lineage workflow, the champion/challenger promotion gate, and the production monitoring flywheel that this chapter's Day-2 section points to.
- The LLM serving chapter covers the Kubernetes-native orchestration layer above individual pods (disaggregated serving, KV-aware routing, autoscaling specific to LLM workloads) — this chapter covers the pod-level infrastructure that layer sits on.

---

## Project 12 — Containerize and deploy a pipeline from an earlier module

Take the RAG service from the retrieval chapter or the inference service from the post-training pipeline in the training chapter.

1. **Containerize it.** Write the multi-stage Dockerfile as specified above. Verify: the final image has no compiler or dev-tool packages (inspect with `docker run --rm <image> dpkg -l | grep -E 'gcc|clang|python.*dev'`); model weights are not in the image; the `/health` and `/ready` endpoints behave correctly under a deliberate startup delay.
2. **Write the K8s manifests.** Deployment (2 replicas, rolling update, GPU resource requests, Secret reference), Service, and a Secret manifest. Apply to a local Kind cluster with a GPU simulated or to a cloud cluster if available. Confirm `kubectl rollout status` completes cleanly.
3. **Add a KEDA ScaledObject** that scales the Deployment based on a Redis list length (use a local Redis deployment in the cluster). Push 50 items to the queue and watch the replica count scale up; let it drain and watch it return to `minReplicaCount`.
4. **Wire a GitHub Actions workflow** that runs `pytest tests/` (use the integration test stubs), builds and pushes the image to a free registry (GitHub Container Registry, GHCR), and on `main` applies the staging manifest. Skip the prod approval gate for this project; document where it would go and why.
5. **Measure cold-start time.** Scale the Deployment to 0 replicas manually, then send a request and measure time-to-first-response. Document what you would change to reduce it.

Stretch: add a Prometheus ServiceMonitor to scrape request latency from the FastAPI service (use the `prometheus-fastapi-instrumentator` package) and build a Grafana dashboard panel showing p99 latency over a 1-hour window.

---

## Interview Q&A

**Q1. Walk me through your Dockerfile for a GPU inference service. Why multi-stage?**

**A.** I use two stages: a `devel` stage (full CUDA devel image, Python dev headers, build tools) and a `runtime` stage (CUDA runtime + cuDNN only). The devel stage installs all Python packages into a prefixed directory with `--prefix=/install`; the runtime stage copies that directory and the application code only. The final image has no compilers, no CUDA toolkit headers, and no Python dev tooling — roughly 60–70% smaller than a single-stage build. I pin the PyTorch wheel install to the matching CUDA index URL for the CUDA version in the base image, so the GPU-accelerated kernels are actually present. I separate `/health` (liveness) from `/ready` (readiness) because model loading can take up to two minutes for large checkpoints — using liveness for both causes Kubernetes to restart a perfectly healthy pod that just hasn't finished loading yet.

The base-image version gotcha I always flag: the CUDA version in the container must be ≤ the maximum CUDA version the host node's driver supports, visible from `nvidia-smi`. Mismatching this gives a cryptic CUDA error at runtime that looks like a model bug. I check the driver ceiling across all node pools before selecting a base image version.

**Q2. We want to scale our inference service automatically based on request queue depth. How would you design it?**

**A.** HPA alone won't work here — it scales on CPU/memory or custom metric values observed from running pods, not on upstream queue depth. I'd use KEDA with a queue-source trigger (RabbitMQ, SQS, or Kafka depending on the stack). A KEDA `ScaledObject` targets the Deployment and defines a trigger with `queueLength: "5"` — scale so each pod handles at most 5 queued messages. KEDA polls the queue on a configurable interval (I'd use 10s) and drives the replica count up or down.

The critical design decision is `minReplicaCount`. For a latency-sensitive service, I set it to 1 — scale-to-zero is attractive but impractical for large models because cold-start time (model load + CUDA init) can exceed a minute for 7B+ models and several minutes for larger ones. Users experience this as a timeout. If cost matters more than cold-start latency — batch/async workloads — I'd set minReplicas to 0 and accept the cold-start, but I'd pre-stage the model weights on a fast local NVMe PVC to minimize load time. I'd measure the cold-start end-to-end in staging and set client timeout accordingly.

**Q3. A bad model update went to production and caused a quality regression. How do you roll back, and how do you prevent this next time?**

**A.** Immediate rollback: `kubectl rollout undo deployment/inference-service -n ml-serving`. This reverts the Deployment to its previous spec (previous image tag and environment variables) using already-pulled layers, so the time to recovery is bounded by pod startup time — typically under two minutes — not by a CI pipeline rebuild. I'd confirm with `kubectl rollout status` and verify the service is healthy via the monitoring dashboard and a manual spot-check.

If the regression is in the model weights (not the application code), rollback is a config change: update `MODEL_URI` in the Deployment to point to the previous version in the model registry and re-apply. Same speed, same mechanism.

Prevention: the eval harness gate in the CI pipeline is the primary defense. Every candidate model version must pass offline eval gates (accuracy, latency on the golden set) before the Deployment manifest is updated. The CI pipeline I described in this chapter runs the eval step before building or pushing anything — a regression fails the pipeline at the cheapest possible point. For model-specific promotion, the eval and observability chapter's champion/challenger gate is the complementary mechanism: the new model must beat the champion on the eval suite before its `MODEL_URI` is updated in the Deployment. Deploying a model that failed either gate requires an explicit override, which creates an audit trail.

## You can now

- write a multi-stage CUDA Dockerfile that ships a runtime image 60–70% smaller than a single-stage build, with no compilers, dev tooling, or embedded model weights.
- separate `/health` (liveness) from `/ready` (readiness) so Kubernetes doesn't kill a pod that is still loading a large checkpoint.
- diagnose the CUDA-toolkit-vs-host-driver ceiling mismatch before it surfaces as a cryptic `no kernel image available` error at the first CUDA call.
- choose between HPA and KEDA for a given workload, and justify `minReplicaCount: 1` for a latency-sensitive LLM service against the cold-start reality.
- build an eval-gated GitHub Actions pipeline with a human production-approval gate, and execute a sub-two-minute `kubectl rollout undo` for both code and model-weight regressions.
