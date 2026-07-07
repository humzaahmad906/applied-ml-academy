# 03 — Advanced Guide: Scaling, Distributed Training, and Production Serving — Part 1 of 2: GPU Operations, Distributed Training, and Kubernetes

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

## Try it

Train a small Transformer (~50M parameters) on the [Tiny Stories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) using DDP across two GPUs with BF16 mixed precision. Profile one training epoch with the PyTorch Profiler and identify your single biggest time sink (compute, data loading, or NCCL communication). Then containerize the training script and run it as a Kubernetes Job manifest on a local kind cluster, requesting one GPU per worker and injecting the MLflow tracking URI via a Kubernetes Secret.

## You can now

- Account for where GPU memory goes during training (parameters, gradients, optimizer state, activations) and choose among DDP, FSDP/ZeRO, TP, and PP for a given model and hardware topology.
- Launch single-node and multi-node distributed training with `torchrun`, wrap a model in FSDP with BF16 mixed precision, and apply gradient checkpointing when memory is the bottleneck.
- Profile a training run with the PyTorch Profiler, isolate compute, data-loading, and NCCL communication costs, and apply the right fix for each.
- Read and write Kubernetes manifests that request GPUs, run distributed training via the Kubeflow Training Operator, and reason about DRA, Kueue, and KAI Scheduler for GPU fleet utilization.
- Choose among dedicated Kubernetes, SkyPilot spot-across-clouds, and Modal serverless based on workload shape and cost profile.
