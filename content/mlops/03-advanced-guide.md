# 03 — Advanced Guide: Scaling, Distributed Training, and Production Serving

**Topics:** Distributed training (PyTorch DDP / FSDP, accelerator-aware setups), GPU operations, Kubernetes for ML workloads, model serving at scale (KServe, BentoML, Triton), real-time inference patterns, streaming feature ingestion, capstone project.

**Time:** 6–8 weeks at 10 hrs/week.
**Goal:** Handle ML systems that don't fit on one machine. Operate them on Kubernetes the way real F50 platforms do. Then prove it with a portfolio-grade capstone.

## What You'll Be Able to Do After This Tier

- Reason about GPU memory and where it goes during training (parameters, gradients, optimizer state, activations)
- Set up and operate distributed training jobs on multiple GPUs and multiple nodes
- Use Kubernetes natively for training and serving — Jobs, Pods, StatefulSets, GPU scheduling, autoscaling
- Deploy production-grade inference services using KServe / BentoML / Triton, with canary, A/B, and shadow traffic
- Design online + streaming feature pipelines that update features in real time
- Reason about latency and throughput trade-offs (batching, quantization, KV cache, speculative decoding)
- Build an end-to-end ML system that combines training + serving + monitoring + feedback

Most senior MLOps engineers are stronger in *either* training infrastructure (DL-heavy, distributed training, GPU operations) *or* serving infrastructure (real-time, low-latency, high-throughput). Cover both at competence level; pick one to specialize.

---

## Week 1 — GPU Operations and Memory

### Why You Need to Know This Cold

GPU costs dominate ML budgets. An H100 is roughly $40K to buy, $2–4/hr to rent. The engineer who can squeeze 2x more out of the same hardware is the engineer who gets promoted.

### What's in GPU Memory During Training

For a model with **P** parameters in FP32:

| Component | Memory |
|---|---|
| Parameters | 4P bytes |
| Gradients | 4P bytes |
| Optimizer state (Adam: m and v) | 8P bytes |
| Activations | depends on batch size, seq length, architecture |
| Workspace (cuDNN, NCCL buffers) | typically GB-scale |

For a 1B-parameter model with Adam in FP32: ~16GB just for parameters/grads/optimizer, before activations. Mixed precision (FP16/BF16 with FP32 master weights) roughly halves this.

For a 70B-parameter LLM: ~1.1TB at FP32, ~560GB at mixed precision. No single GPU has that much memory — *which is why distributed training exists*.

### `nvidia-smi` Fluency

```bash
nvidia-smi                          # snapshot
nvidia-smi -l 1                     # refresh every second
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv -l 1
nvidia-smi dmon                     # device monitor with rolling stats
```

What you watch:

- **GPU-Util %.** Anything under 80% in a training run is wasted money. Under 50% suggests data-loading bottlenecks.
- **Memory.Used.** Should be 80%+ utilized. Lower means you can probably increase batch size.
- **Power.Draw.** If your GPU is drawing 50% of its TDP, you're not bound on compute.

### Profiling Tools

- **PyTorch Profiler** with TensorBoard plugin — first stop for "where is training spending time."
- **NVIDIA Nsight Systems** — system-level view including CPU/GPU sync, NCCL communication.
- **NVIDIA Nsight Compute** — kernel-level GPU profiling. Use when you suspect a custom kernel is the bottleneck.

```python
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./tb_logs"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(10):
        train_step()
        prof.step()
```

Then `tensorboard --logdir ./tb_logs`. The PyTorch Profiler tab shows GPU/CPU timeline, top kernel times, suggestions. Spend an afternoon driving it on a real model — it pays back forever.

### Common Performance Pitfalls

1. **CPU-bottlenecked data loading.** GPU sits idle waiting for batches. Fix: more `DataLoader` workers, prefetching, pinned memory, faster decoding.
2. **Per-step Python overhead.** Small models on big GPUs are often Python-bound. `torch.compile()` helps.
3. **Synchronization in metric logging.** `loss.item()` syncs CPU↔GPU. Don't do it every step.
4. **NumPy/CPU operations sprinkled in the training loop.** Keep tensors on GPU.
5. **Activations exceeding budget.** Gradient checkpointing trades compute for memory.

### Mixed Precision Training

Standard since ~2020. Use BF16 on Ampere+ (A100/H100) — wider exponent range, no loss scaling needed:

```python
# PyTorch 2.x: import from torch.amp, pass device type explicitly.
# `torch.cuda.amp` still works but is deprecated.
from torch.amp import autocast, GradScaler

# GradScaler is only needed for FP16 (to avoid underflow). BF16 has FP32's
# exponent range, so no scaling required.
use_fp16 = False
scaler = GradScaler("cuda", enabled=use_fp16)
amp_dtype = torch.float16 if use_fp16 else torch.bfloat16

for batch in loader:
    optimizer.zero_grad()
    with autocast("cuda", dtype=amp_dtype):
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Or use `torch.compile` and a modern training loop with no manual AMP — increasingly the default.

### Exercises

1. Profile a model training step. Identify the top 3 time sinks. Reduce one by 20%.
2. Increase batch size until you OOM. Note the limit. Try gradient checkpointing — what's the new limit?
3. Switch from FP32 to BF16. Compare throughput and final accuracy.

---

## Week 2 — Distributed Training

### The Distribution Strategies

| Strategy | What it splits | When to use |
|---|---|---|
| **Data Parallel (DP)** | Mini-batches across replicas | Default. Each replica has full model; gradients all-reduced |
| **Distributed Data Parallel (DDP)** | Same as DP, but cross-node | The standard for multi-GPU and multi-node training |
| **Tensor Parallel (TP)** | Model layers split across GPUs (split a linear layer's weight matrix) | When a single layer doesn't fit on one GPU |
| **Pipeline Parallel (PP)** | Different layers on different GPUs | Very deep models; introduces "bubble" inefficiency unless mitigated |
| **Fully Sharded Data Parallel (FSDP / ZeRO-3)** | Shards parameters, gradients, optimizer state across DP replicas | The standard for large-model training in 2026 |
| **Expert Parallel (EP)** | MoE experts across GPUs | Mixture-of-experts models only |

The 2026 frontier combines them: **TP within a node** (NVLink-fast), **PP across nodes** (slower but tolerable), **DP / FSDP across pipeline replicas**. This is **3D parallelism**.

### PyTorch DDP — The Floor

```python
# train.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup():
    """torchrun sets RANK (global), LOCAL_RANK (within node), WORLD_SIZE.
    On multi-node setups, `device_ids` must be the LOCAL rank — the GPU
    index on *this* host — not the global rank.
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank()

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank, global_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    model = MyModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(train_dataset)
    loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch.to(device))
            loss.backward()
            optimizer.step()

    cleanup()

if __name__ == "__main__":
    main()
```

Launch:

```bash
torchrun --nproc_per_node=8 train.py            # one node, 8 GPUs
torchrun --nnodes=2 --nproc_per_node=8 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=master:29500 train.py  # two nodes, 8 GPUs each = 16 GPUs
```

Things to internalize:

1. **`nccl` backend** — NVIDIA's collective communication library. Use it on NVIDIA hardware.
2. **`DistributedSampler` + `set_epoch`** — without `set_epoch`, every epoch shuffles the same way across replicas. Easy bug, painful to find.
3. **`pin_memory=True`** — enables faster CPU→GPU transfers via pinned memory.
4. **Effective batch size** is `per_device_batch * world_size`. Scale your learning rate accordingly (linear scaling rule, or use a tuned schedule).

### PyTorch FSDP (ZeRO-3 Equivalent)

For large models where parameters don't fit on a single GPU:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = MyTransformer()
my_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={MyTransformerLayer},
)
model = FSDP(
    model,
    auto_wrap_policy=my_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    ),
    device_id=torch.cuda.current_device(),
)
```

FSDP shards parameters, gradients, and optimizer state across the data-parallel group. Each GPU only holds 1/N of each. Trades extra communication for dramatically reduced memory. Standard for any LLM-class training in 2026.

### DeepSpeed and Accelerate

- **DeepSpeed** (Microsoft) — gives you ZeRO-1/2/3, pipeline parallelism, MoE support, CPU/NVMe offloading. The standard for "I want to train a big model without writing low-level distributed code."
- **Accelerate** (Hugging Face) — a thin wrapper that handles DDP/FSDP/DeepSpeed launch and per-device dispatch with minimal code change. The standard for "I want my training script to work on 1 GPU or 64 GPUs without rewriting."
- **Megatron-LM** (NVIDIA) and **DeepSpeed** (Microsoft) are separate frameworks; **Megatron-DeepSpeed** is a community fork that merges them. You'll see all three at frontier labs and any F50 with serious LLM ambitions.

For a portfolio project, demonstrate competence with Accelerate + FSDP. Mention DeepSpeed and Megatron in your README to signal awareness.

### Multi-Node Networking

For multi-node training, you need:

- **High-speed interconnect** — InfiniBand or RoCE; on AWS, EFA. Plain TCP across regions kills training.
- **NCCL_DEBUG=INFO** when something is wrong. NCCL is one of the most opaque sources of training failures.
- **Rendezvous protocol** — `c10d` for vanilla torchrun; etcd or a job manager (Kubeflow, Ray, Slurm) for production.

### Cost-Conscious Training

A 70B-parameter model trained on 8 H100s for a week is roughly $5K. Sloppy training (poor utilization, premature stops, bad HPO) easily doubles that. Discipline:

- **Profile before scaling.** A model running at 30% GPU utilization on 64 GPUs needs to be at 80% on 24 GPUs.
- **Use spot/preemptible instances** for training. Checkpoint every N steps; lose at most N steps on preemption.
- **Reserved capacity** for known-recurring jobs. F50 negotiates yearly GPU reservations.
- **Cheaper hardware where viable.** A100s are roughly half the cost of H100s. T4s/L4s for inference. Inferentia and Trainium on AWS for specific workloads.

### Exercises

1. Run a single-node multi-GPU DDP training job on the [Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset with a small Transformer (~50M params). 4 GPUs if you can get them; 2 is fine.
2. Convert it to FSDP. Compare memory and throughput.
3. Use HuggingFace Accelerate to run the same script unchanged on 1 GPU and on 4 GPUs.
4. Profile a 1000-step run. Report the breakdown of time across compute / communication / dataloading.
5. (Stretch) Rent two cheap nodes on Lambda or RunPod for an hour. Run a 2-node training job.

---

## Week 3 — Kubernetes for ML Workloads

### Why Kubernetes Won for ML Infra

- Declarative resource management (specify what you want, not how)
- GPU scheduling primitives (the NVIDIA device plugin lets you request `nvidia.com/gpu: 1`)
- Autoscaling at multiple layers (HPA for pods, Cluster Autoscaler / Karpenter for nodes)
- Operators model lets ML systems package their lifecycle as CRDs (Kubeflow, KServe, Spark Operator, Ray Operator)
- Multi-tenant isolation via namespaces, RBAC, network policies
- Cloud-portable — same YAML works on EKS, GKE, AKS, on-prem

Every F50 ML platform runs on Kubernetes in 2026. Knowing it is no longer optional.

### The Pieces You Care About

| Resource | What it is | When you use it for ML |
|---|---|---|
| **Pod** | Smallest unit; one or more containers sharing network | The thing your training script or model server runs in |
| **Job** | Run-to-completion pod | Training, batch inference, data prep |
| **CronJob** | Scheduled Job | Daily retraining trigger |
| **Deployment** | Long-running stateless replicas | Serving service |
| **StatefulSet** | Replicas with stable identity | Less common — sharded model serving |
| **Service** | Stable DNS + load balancing for pods | Putting an endpoint in front of your model service |
| **ConfigMap / Secret** | Config / secrets injected as env or files | Hyperparameters, credentials |
| **PersistentVolume / PVC** | Durable storage attached to pods | Training checkpoints, model artifacts |
| **HPA / VPA** | Autoscalers | Scale a serving deployment by QPS or latency |
| **Custom Resource (CRD)** | Operator-defined types | `PyTorchJob`, `InferenceService`, `RayCluster`, `SparkApplication` |

### Local K8s for Dev

For a portfolio project, you don't need a real cluster. Use one of:

- **kind** — Kubernetes in Docker. The most portable.
- **minikube** — Single-node cluster as a VM.
- **k3d** — k3s (lightweight Kubernetes) in Docker.

```bash
kind create cluster --name mlops --config kind-config.yaml
kubectl cluster-info
kubectl get nodes
```

### A Training Job Manifest

```yaml
# k8s/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-income-classifier
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: ghcr.io/me/mlops-project/train:latest
          command: ["python", "-m", "pipelines.train"]
          env:
            - name: MLFLOW_TRACKING_URI
              valueFrom:
                secretKeyRef:
                  name: mlflow-secret
                  key: tracking_uri
            - name: APP_N_ESTIMATORS
              value: "500"
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "4"
              memory: "8Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: training-data-pvc
```

Apply with `kubectl apply -f training-job.yaml`. Watch with `kubectl logs -f job/train-income-classifier`.

### The Kubeflow Training Operator

For real distributed training jobs:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-train
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
            - name: pytorch
              image: ghcr.io/me/train:latest
              resources: { limits: { nvidia.com/gpu: 1 } }
    Worker:
      replicas: 3
      template:
        spec:
          containers:
            - name: pytorch
              image: ghcr.io/me/train:latest
              resources: { limits: { nvidia.com/gpu: 1 } }
```

The operator handles rendezvous (sets `MASTER_ADDR`, `WORLD_SIZE`, `RANK`) — you just write a normal `torchrun`-style script. Standard pattern at F50.

### Spark Operator and Ray Operator

- **Spark Operator** for batch processing of large training data, feature engineering, big-data ETL.
- **Ray Operator (KubeRay)** for distributed Python — Ray Train for distributed training, Ray Serve for serving, Ray Tune for HPO, Ray Data for distributed datasets. Ray is increasingly the unified ML compute platform; learn it.

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: tune-hyperparams
spec:
  entrypoint: python /home/ray/tune_job.py
  rayClusterSpec:
    headGroupSpec:
      template: { spec: { containers: [{name: head, image: rayproject/ray:2.40.0-py311-gpu, resources: {limits: {nvidia.com/gpu: 1}}}] } }
    workerGroupSpecs:
      - replicas: 3
        groupName: workers
        template: { spec: { containers: [{name: worker, image: rayproject/ray:2.40.0-py311-gpu, resources: {limits: {nvidia.com/gpu: 1}}}] } }
```

### GitOps with Argo CD or Flux

Don't `kubectl apply` from CI directly. The 2026 pattern:

1. Manifests live in a Git repo
2. Argo CD (or Flux) watches the repo
3. When you merge, Argo CD reconciles the cluster to match Git

This makes "what's in the cluster?" answerable by reading `main`, and rollback is `git revert`. Standard at any F50.

### GPU Orchestration Beyond Device Plugins

The NVIDIA device plugin (`nvidia.com/gpu: 1`) is where everyone starts. It's a blunt instrument: you get a GPU count, no more. Production ML fleets need something more expressive. Three layers of the 2026 answer:

#### Kubernetes DRA (Dynamic Resource Allocation)

DRA replaces the device-plugin model with **structured claims**: a pod declares what it needs from a hardware class, and the driver satisfies that claim based on actual device capabilities (memory, topology, peer bandwidth). Shipped alpha in Kubernetes 1.26, reached beta through 1.32–1.34. The claim API (`ResourceClaim`, `DeviceClass`) is the direction the ecosystem is moving — device plugins will eventually be legacy.

Why it matters: instead of asking for "1 GPU," a training job can ask for "2 GPUs that share NVLink, with at least 40GB HBM each." The scheduler can satisfy that claim — or fail fast with a clear reason, instead of silently running on suboptimal hardware.

#### Kueue: Quota-Based Batch Queueing

[Kueue](https://kueue.sigs.k8s.io/) is a Kubernetes-native batch queueing system that sits on top of the scheduler. It manages **quotas across multiple teams** (ClusterQueues with resource flavors), does **gang admission** (a distributed training job only starts when *all* pods can be scheduled simultaneously, preventing partial starts that block resources), and handles **preemption** and **borrowing** across teams.

Before Kueue, a common F50 problem: team A submits 100 training jobs, team B's jobs queue indefinitely. Kueue adds fairness without building a separate cluster per team. The right tool when you need quota + fairness across ML teams sharing a fleet.

#### NVIDIA KAI Scheduler

[KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler) (Apache 2.0, open-sourced from Run:ai in 2025) is a Kubernetes batch scheduler purpose-built for AI/ML workloads. Key capabilities:

- **Gang scheduling** — all-or-nothing admission for distributed training jobs (similar to Kueue's gang admission but at the scheduler level)
- **Topology-aware placement** — places pods to maximize NVLink bandwidth utilization; avoids putting co-communicating workers on different racks
- **Fractional GPU sharing** — allocate a memory fraction of a GPU per pod (e.g., a 24GB GPU split into 3 × 8GB allocations for inference deployments)

When each makes sense:

| Tool | Use it when |
|---|---|
| **Kueue** | Quota, fairness, and priority across multiple teams sharing a cluster |
| **KAI Scheduler** | GPU packing, topology-aware placement, fractional GPU sharing for inference fleets |
| **DRA** | New clusters where you want the future-proof claim API from day one |

The utilization angle: a fleet running at 30% average GPU utilization is the norm without intentional scheduling. Getting to 80%+ with topology-aware packing and fractional sharing translates to millions of dollars per year at F50 scale (an H100 node runs ~$30K/mo reserved; 50% utilization lift on 100 nodes = $1.5M/mo recovered). This is the number that gets VP attention.

### SkyPilot and Modal: The No-K8s Path

Not every training workload belongs on a Kubernetes cluster you own. Two tools that cover the burst-and-spiky case:

**SkyPilot** (open-source, BSD) abstracts over clouds: you write a task YAML declaring compute requirements, and SkyPilot finds the cheapest available GPU — across AWS, GCP, Azure, Lambda, CoreWeave, and others — launches it, runs your job, and tears it down. It handles spot instance preemption with auto-recovery from the last checkpoint. You get cross-cloud GPU spot access without managing a cluster. The workflow:

```yaml
# sky.yaml
resources:
  accelerators: A100:8
  cloud: cheapest          # SkyPilot bids across providers

run: |
  torchrun --nproc_per_node=8 train.py \
    --checkpoint-dir $SKYPILOT_TASK_ID/ckpt
```

```bash
sky launch sky.yaml --spot --retry-until-up
```

**Modal** (serverless GPU, ~$4/hr H100, per-second billing) is the right tool for spiky workloads — a fine-tuning run, an eval job, a batch embedding pipeline — where you'd otherwise sit on idle reserved capacity. You write a Python function decorated with `@app.function(gpu="H100")` and invoke it; Modal handles the container launch, GPU attachment, and teardown. No YAML, no cluster management.

Rule of thumb:

- **Steady-state, high-utilization** training → reserved capacity + Kubernetes
- **Bursty training, evaluation, one-off fine-tunes** → SkyPilot for spot-cross-cloud or Modal for serverless
- **The no-ops startup that wants GPUs today** → Modal first, K8s later when the team hires an infra engineer

### Exercises

1. Set up a local Kubernetes cluster (kind / k3d). Deploy your model serving image as a Deployment + Service. Hit it via port-forward.
2. Convert your training job to a Kubernetes Job. Run it.
3. Install the Kubeflow Training Operator. Run a 2-worker PyTorchJob.
4. Set up Argo CD in your local cluster. Point it at a Git repo with your manifests. Merge a change to the deployment; watch Argo CD apply it.
5. (Stretch) Install Kueue on a local cluster. Define two ClusterQueues with a shared resource pool. Submit competing training jobs and observe fairness enforcement.
6. (Stretch) Run a training job on SkyPilot against a cloud provider. Observe checkpoint recovery after a simulated preemption.

---

## Week 4 — Production Inference Serving

The other half of the MLOps job. Most teams underinvest here and pay for it forever.

### The Serving Stack Choices

| Tool | Strength | When to use |
|---|---|---|
| **FastAPI + uvicorn** | Simple, fast, batteries-not-included | Single-model services, low-medium scale |
| **BentoML** | Python-native packaging, multi-framework, batching | Mid-scale; great DX |
| **KServe** | Kubernetes-native, multi-framework, autoscaling-from-zero, multi-model | Production K8s standard |
| **NVIDIA Triton** | GPU-optimized, multi-framework (TensorRT, PyTorch, ONNX), dynamic batching, ensembles | High-throughput GPU serving |
| **vLLM / TGI / SGLang** | LLM-specific (continuous batching, PagedAttention) | LLM serving — covered in the Next Steps chapter |
| **Ray Serve** | Python-first, multi-replica composition | When you've adopted Ray |
| **TorchServe / TensorFlow Serving** | Framework-native | Less popular; you'll see them at older deployments |

For a portfolio project: build with BentoML *and* KServe. Two artifacts. Two stories.

### BentoML — The Pragmatic Choice

BentoML packages a model + its preprocessing + its serving code into a versioned **Bento** (a directory or container). Then you deploy that.

```python
# service.py
import bentoml
from bentoml.io import JSON
import numpy as np
from pydantic import BaseModel

class PredictInput(BaseModel):
    features: list[float]

income_runner = bentoml.sklearn.get("income_classifier:latest").to_runner()
svc = bentoml.Service("income_service", runners=[income_runner])

@svc.api(input=JSON(pydantic_model=PredictInput), output=JSON())
async def predict(data: PredictInput) -> dict:
    x = np.array([data.features])
    proba = await income_runner.predict_proba.async_run(x)
    return {"probability": float(proba[0, 1])}
```

```bash
bentoml build      # produces a Bento (versioned by hash)
bentoml containerize income_service:latest   # produces a Docker image
bentoml serve income_service:latest          # local serving
```

Built-in features that matter:

- **Adaptive batching** — multiple requests within X ms collected into one inference call. Critical for GPU throughput.
- **Multi-framework runners** — mix PyTorch, scikit-learn, ONNX in one service.
- **Yatai** (BentoML's K8s deployment service) for cluster ops.

### KServe — Kubernetes-Native Serving

KServe (formerly KFServing) is a Kubernetes CRD that wraps your model in a production-grade serving pod:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: income-classifier
spec:
  predictor:
    sklearn:
      storageUri: s3://my-bucket/models/income-classifier/v17/
      resources:
        requests: { cpu: "100m", memory: "256Mi" }
        limits: { cpu: "2", memory: "4Gi" }
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 80
    scaleMetric: concurrency
```

What you get out of the box:

- **Multi-framework predictors** (sklearn, tensorflow, pytorch, xgboost, lightgbm, huggingface, custom)
- **Autoscaling from zero** (using Knative under the hood) — pay nothing when no traffic
- **Canary deployments** with explicit traffic percentages
- **Transformers and Explainers** as separate pods sharing the same inference service
- **gRPC and REST** simultaneously
- **Open Inference Protocol** — standardized API across frameworks

```yaml
# Canary 90/10 split
spec:
  predictor:
    canaryTrafficPercent: 10
    canary:
      sklearn:
        storageUri: s3://my-bucket/models/income-classifier/v18/
    sklearn:
      storageUri: s3://my-bucket/models/income-classifier/v17/
```

Watch metrics on both. When v18 is confident, bump to 100. When it regresses, drop to 0.

### NVIDIA Triton — Maximum GPU Throughput

For GPU-bound serving at scale, Triton is king. Key features:

- **Dynamic batching** — server-side batching across requests, configurable max delay
- **Multiple backends** — TensorRT, ONNX Runtime, PyTorch, TensorFlow, custom Python, vLLM
- **Model ensembles** — preprocess → predict → postprocess as one Triton inference (single round trip)
- **Concurrent model execution** — multiple model instances on the same GPU
- **Sequence batching** — for stateful models like RNNs or LLMs

`config.pbtxt` per model:

```
name: "income_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 64
dynamic_batching {
  preferred_batch_size: [32, 64]
  max_queue_delay_microseconds: 5000
}
input  [{ name: "input"  data_type: TYPE_FP32 dims: [10] }]
output [{ name: "output" data_type: TYPE_FP32 dims: [2] }]
```

Triton on KServe is a thing — `predictor.triton` in the InferenceService spec.

### The Serving Patterns

#### 1. Shadow Traffic

Route real prediction requests to the new model *in parallel* with the current model. Return the current model's output to the user. Log both. Compare offline.

```
[Request] ──► [Current model] ──► [Response]
       │
       └──► [Shadow model] ──► [Log only]
```

Lets you validate a new model against real production traffic with zero user impact. Standard pre-promotion check.

#### 2. Canary

Route X% of traffic to the new model. Monitor metrics. Increase X gradually if healthy; rollback if degraded.

#### 3. A/B Test

Split traffic between two models long enough to measure business outcomes. Statistical significance matters — bake in a power analysis up front.

#### 4. Multi-Armed Bandit

A smarter A/B test where the routing percentage adapts based on observed performance. Standard pattern at recommendation-heavy companies.

#### 5. Blue/Green

Two identical environments. Switch all traffic from blue to green at once. Fast rollback by switching back. Riskier than canary; rare for ML serving where you want gradual rollouts.

### Latency Engineering for ML Serving

The latency budget. Typical:

| Use case | P95 latency budget |
|---|---|
| Real-time ads | 50ms total, model under 20ms |
| Real-time fraud | 100ms |
| Live recommendations | 200ms |
| Conversational AI (LLM, time to first token) | 500ms |
| Batch / async | seconds to minutes |

Where latency goes in a typical online prediction:

```
Network (client → LB)          5ms
TLS termination, LB routing   10ms
Service request handling      10ms
Feature lookup (Redis)         5–15ms
Model inference               20–100ms
Post-processing                5ms
Network back                   5ms
─────────────────────────────────
Total                         ~80–150ms
```

The big knobs:

1. **Batch.** Even at low QPS, micro-batching with a short window (1–10ms) often improves both throughput *and* per-request latency under load.
2. **Quantize.** FP16/BF16, INT8 (quantization-aware training), INT4 (LLM-specific). Often 2–4x speedup, often <1% accuracy delta.
3. **Compile / optimize.** `torch.compile`, TensorRT compilation, ONNX with execution providers. 1.5–3x typical wins.
4. **Right-size hardware.** A 1B-parameter model probably doesn't need an H100. T4 / L4 / A10 may halve cost at similar latency.
5. **Cache.** If the same input recurs, cache predictions. Useful in recommendation systems, document classification, embeddings.

### Exercises

1. Package your tier-2 model as a Bento. Containerize. Run locally with adaptive batching enabled.
2. Deploy the same model as a KServe InferenceService on your local cluster.
3. Convert the model to ONNX. Serve it via Triton. Compare P95 latency to your FastAPI baseline.
4. Implement a canary 90/10 with KServe and a second model version.
5. Load-test with `vegeta` or `k6`. Find your service's saturation point.

---

## Week 5 — Streaming Features and Real-Time Inference

### Why You Need Streaming Features

Many real ML problems need *fresh* features. Examples:

- **Fraud detection:** "amount spent in the last 10 minutes" is meaningful at second-level granularity; nightly batch is too late.
- **Recommendations:** "items the user has clicked in this session" must update as the session continues.
- **Demand pricing:** real-time inventory and event signals drive pricing.

The pattern:

```
[Events]  ──►  [Kafka topic]  ──►  [Flink job]  ──►  [Online feature store]
                                                              │
                                                              ▼
                                                      [Inference service]
```

You'll touch Kafka and Flink at competence level. Deep mastery is covered in the Advanced Topics chapter.

### Kafka in 60 Seconds

- **Topic:** a named stream of events
- **Partition:** a topic is split into partitions; ordering is per-partition
- **Producer:** writes events with a key (hash of key → partition)
- **Consumer Group:** multiple consumers cooperate; Kafka divides partitions among them
- **Offsets:** consumers track where they are

For ML: produce events from your application, consume them in a feature computation job, write derived features to the online store.

### A Simple Streaming Feature Job

```python
# stream_features.py — using Kafka client + Redis
from confluent_kafka import Consumer
import redis
import json

consumer = Consumer({
    "bootstrap.servers": "kafka:9092",
    "group.id": "feature-counter",
    "auto.offset.reset": "earliest",
})
consumer.subscribe(["purchases"])

r = redis.Redis(host="redis", port=6379)

while True:
    msg = consumer.poll(1.0)
    if msg is None or msg.error():
        continue
    event = json.loads(msg.value())
    user_id = event["user_id"]
    amount = event["amount"]
    # Sliding count over last 24h via Redis sorted set
    now = int(event["timestamp"])
    r.zadd(f"purchases:{user_id}", {f"{now}:{amount}": now})
    r.zremrangebyscore(f"purchases:{user_id}", 0, now - 86400)
    count = r.zcard(f"purchases:{user_id}")
    total = sum(float(x.decode().split(":")[1]) for x in r.zrange(f"purchases:{user_id}", 0, -1))
    r.hset(f"features:{user_id}", "purchases_24h_count", count)
    r.hset(f"features:{user_id}", "purchases_24h_total", total)
    consumer.commit(msg)
```

That's a real (if naive) streaming feature pipeline. In production you'd use Flink (real watermarks, exactly-once semantics, stateful sessions); the principles are the same.

### Flink in 60 Seconds

- **Stateful stream processing engine** — maintains TB-scale state with checkpoints
- **Event-time-first** — first-class watermarks, late data handling, windowing (tumbling, sliding, session)
- **Exactly-once via two-phase-commit sinks** — strongest guarantee across systems
- **SQL, Table API, DataStream API**

```sql
-- Flink SQL: windowed user purchase counts
CREATE TABLE purchases (
  user_id STRING,
  amount DOUBLE,
  ts TIMESTAMP(3),
  WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
) WITH ('connector'='kafka', 'topic'='purchases', 'format'='avro');

CREATE TABLE user_features
WITH ('connector'='redis', ...)
AS
SELECT
  user_id,
  TUMBLE_START(ts, INTERVAL '1' MINUTE) AS window_start,
  COUNT(*) AS purchase_count_1m,
  SUM(amount) AS purchase_total_1m
FROM purchases
GROUP BY TUMBLE(ts, INTERVAL '1' MINUTE), user_id;
```

The output flows into Redis (or any online store) and is read by your inference service.

### The Online + Offline Consistency Problem

The trick: features computed on the stream must match features computed in batch. Otherwise training-serving skew creeps back in.

Standard solution: **the same transformations, expressed once, run in both modes.** Apache Beam, Flink (with the same job in batch mode), or a feature store with a unified definition (Feast with streaming sources, Tecton, Feathr).

A weaker but pragmatic solution: define batch features in dbt/SQL, derive streaming features by porting the SQL to Flink SQL, run a daily reconciliation that compares the two and alerts on divergence.

### Exercises

1. Set up a single-node Kafka + Redis stack in Docker Compose. Produce synthetic purchase events.
2. Write a streaming feature job (Python or Flink SQL) that maintains "purchases in the last hour per user."
3. Modify your serving service to read these features from Redis at inference time.
4. Inject a malformed event. Confirm the job handles it gracefully (logs, doesn't crash, doesn't poison downstream).
5. Run a reconciliation: compute the same feature in batch from a CSV of the same events. Verify it matches.

---

## Week 6 — Capstone Project Planning

The capstone is the big project that anchors interviews. We sketch the spec here; you build it in the projects file.

### The Capstone Spec (Preview)

A real-time anomaly detection system with a closed feedback loop:

- **Source events** from a synthetic generator (controllable fraud injection rate)
- **Streaming features** via Flink → Redis
- **A trained model** with proper experiment tracking, registry, promotion
- **A serving service** with sub-100ms P95 latency at 1000 RPS, deployed via KServe
- **Predictions logged** to a lake (Iceberg or plain Parquet)
- **A labeling UI** (Streamlit) for analysts to confirm/reject flagged events
- **Labels feeding back** to a retraining pipeline
- **Daily monitoring** of drift + precision/recall on labels
- **CI/CD/CT** in GitHub Actions
- **Everything on Kubernetes**, deployable with one `make up`

Detailed acceptance criteria, architecture diagrams, and decision rationale are in the Fortune 50 Projects chapter.

### What You Build in Week 6

Just enough to validate the architecture:

1. Skeleton repo with the project structure
2. Docker Compose stack with Kafka, Redis, Postgres, MLflow, MinIO, your services
3. A *trivial* version of every component — event generator emits one event/sec, feature job computes one feature, model is a stub, service returns random scores
4. The whole thing wired up. End to end. Trivial.

This is the "skeleton walking" milestone. You've proven the architecture flows. Now you spend the next 8–12 weeks making each component good.

---

## Confidence Checks Before Moving On

1. You can describe the memory breakdown in GPU training and why FSDP/ZeRO exists.
2. You can sketch DDP, FSDP, TP, PP and say when each is appropriate.
3. You can read a Kubernetes manifest and find the GPU request, the resource limit, and the environment injection.
4. You can describe shadow / canary / A/B / multi-armed-bandit patterns and when each is right.
5. You can explain why micro-batching often improves *both* throughput and latency.
6. You understand the online/offline consistency problem and the typical solutions.
7. You've built a streaming feature pipeline end-to-end, even at toy scale.

When all seven feel solid, move on to the Next Steps chapter to specialize.
