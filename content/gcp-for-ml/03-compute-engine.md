# 03 — Compute: Compute Engine

Compute Engine is Google Cloud's virtual machine service — raw, configurable compute you rent by the second. Even though most of your ML training and serving will eventually run through the higher-level Vertex AI (which provisions Compute Engine under the hood), you must understand the machine families, accelerators, and pricing models here, because those are exactly the knobs Vertex exposes and exactly what determines your bill. This module is your map of "what hardware can I run, and how do I not overpay for it."

## Machine families

A **machine type** defines vCPU and memory. They come in families tuned for different workloads:

- **General-purpose** (E2, N2/N2D, N4, and the C3/C4 series) — balanced CPU/memory for data preprocessing, orchestration, and light inference. Note Google's taxonomy quirk: despite the "C" prefix, **C3 and C4 are classified as general-purpose**, not compute-optimized.
- **Compute-optimized** (C2, C2D, H3, H4D) — high per-core performance for CPU-bound feature engineering and tightly-coupled HPC.
- **Memory-optimized** (M-series, X4) — very large RAM (multi-terabyte) for in-memory datasets and analytics.
- **Accelerator-optimized** (A-series, G2, G4) — the GPU/TPU machines that matter most for ML, covered below.

You can also define **custom machine types** with a specific vCPU/memory ratio when a predefined shape wastes money — useful for data jobs that need lots of RAM but few cores.

```bash
gcloud compute instances create prep-node \
  --zone=us-central1-a \
  --machine-type=n4-standard-16 \
  --image-family=debian-12 --image-project=debian-cloud
```

## GPU machines

GPUs attach to specific accelerator-optimized families. As of 2026 the lineup, from most to least powerful, is:

- **A4X** — NVIDIA **GB200** Grace Blackwell Superchips (an exascale, NVLink-connected rack-scale platform); frontier foundation-model training.
- **A4** — eight NVIDIA **B200** Blackwell GPUs connected by fifth-gen NVLink; large-scale training and low-latency serving of big models.
- **A3 Ultra** — NVIDIA **H200** GPUs with the highest networking in the A3 line; **A3 Mega/High** carry **H100** GPUs.
- **A2** (Standard/Ultra) — NVIDIA **A100** GPUs (40 GB or 80 GB); still capable and widely available for mid-size training.
- **G4** — NVIDIA **RTX PRO 6000 Blackwell** GPUs (GA since late 2025); a newer cost-efficient serving and fine-tuning tier above the L4.
- **G2** — NVIDIA **L4** GPUs (24 GB); the workhorse for cost-efficient **inference** and lighter fine-tuning. The L4 is the sweet spot for serving most models.

(There is also an **A4X Max** with GB300-class superchips at the very top of the range for rack-scale frontier training.)

Two practical rules. First, **the big Blackwell and Hopper VMs are region-constrained** — verify availability in your target region before designing around them. Second, **match the chip to the job**: L4 (G2) for serving and small fine-tunes, A100 (A2) or H100/H200 (A3) for serious training, and B200/GB200 (A4/A4X) only when you are genuinely training or serving frontier-scale models. Attaching an A4 to an inference service that an L4 would handle is how budgets evaporate.

```bash
# A serving-class GPU VM (L4)
gcloud compute instances create infer-node \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE
```

## TPUs

**Cloud TPUs** are Google's custom ML accelerators, ideal for large transformer and JAX/PyTorch-XLA workloads. The current generations:

- **Trillium (v6e)** — the sixth-generation TPU, strong for training, fine-tuning, and serving of transformers, text-to-image, and CNNs.
- **Ironwood (tpu7x)** — the seventh generation, purpose-built for the age of large-model inference and training, with a large jump in per-chip compute and memory (192 GB per chip) over Trillium. Ironwood is generally available.
- Earlier **v5e** (cost-efficient) and **v5p** (high-performance training) remain in use.

TPUs are referenced by their generation on all API surfaces (for example `v6e`). They shine when your framework supports XLA well and your model maps cleanly to their systolic-array design; for arbitrary PyTorch code with custom CUDA kernels, GPUs are usually the pragmatic choice. Most ML engineers reach TPUs through Vertex AI training rather than raw Compute Engine.

## Images and environments

A VM boots from an **image**. For ML you rarely start from a bare OS — instead use the **Deep Learning VM images** (the `deeplearning-platform-release` project), which come with CUDA, cuDNN, and a preinstalled framework matched to driver versions (image families like `pytorch-latest-gpu`, or pinned ones such as `pytorch-2-9-cu129-ubuntu-2204`). This saves the hours of driver-versioning pain that plague fresh GPU boxes. Note the active families are now **PyTorch** and **base CUDA**; the standalone TensorFlow and CPU-only Deep Learning VM images have been deprecated, so bake TensorFlow into your own container if you need it. For JAX, use the dedicated JAX AI container images. You can snapshot a configured VM into a **custom image** to standardize your team's environment, though the more reproducible path is to bake your environment into a **container** (covered later) and run it on Vertex AI or Cloud Run.

## Spot VMs — the biggest cost lever

**Spot VMs** are spare-capacity instances at a steep discount (often 60–90% off on-demand), with the catch that Google can **preempt** (reclaim) them at any time with a short warning. (Spot is the current model; the older "preemptible VMs" are the legacy 24-hour-capped version — use Spot.)

For ML, spot is transformational when used correctly:

- **Great for training** *if* your job **checkpoints frequently** to Cloud Storage, so a preemption costs you minutes, not the whole run. Distributed training frameworks and Vertex AI both support resuming from checkpoints.
- **Great for batch prediction and preprocessing** — stateless, retryable, throughput-oriented work.
- **Bad for online serving** — you cannot have your production endpoint vanish mid-request; use on-demand (or committed-use) capacity there.

```bash
gcloud compute instances create train-spot \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release
```

Beyond spot, the other cost levers are **committed-use discounts** (commit to steady baseline capacity for 1–3 years at a large discount — sensible for always-on serving) and simply **turning things off**: idle GPU VMs bill by the second whether or not they compute, so stop them, and prefer autoscaling/scale-to-zero serving where possible.

## When to use raw Compute Engine vs managed services

Reach for raw Compute Engine when you need full control of the environment, custom drivers or kernels, interactive multi-day experimentation on a fixed box, or an architecture the managed services do not fit. Reach for **Vertex AI** for training and prediction when you want managed job lifecycle, autoscaling, and MLOps integration; **Cloud Run** or **GKE** for containerized serving. The default for most ML work is a managed service — but that service is provisioning the exact machine types and accelerators you just learned, so this knowledge transfers directly.

## How this fits the whole solution

Compute is the engine room of the end-to-end system. Preprocessing runs on general-purpose or compute-optimized VMs (or Dataflow); training runs on GPU/TPU accelerators (usually via Vertex AI, which selects these same machine types); batch scoring runs cheaply on spot capacity; and serving runs on L4-class GPUs or CPUs behind an autoscaler. The choices you make here — right-sizing the accelerator to the task, using spot for interruptible work, checkpointing to survive preemption, and shutting off idle capacity — are the difference between a system that costs hundreds and one that costs thousands per month for the same output.

## Key takeaways

- Machine **families** target workloads: general-purpose and compute-optimized for data prep, accelerator-optimized (A-series, G2) for ML.
- **GPU lineup (2026):** G2/L4 for cost-efficient serving; A2/A100 and A3/H100–H200 for training; A4/B200 and A4X/GB200 for frontier scale. Match the chip to the job and verify regional availability.
- **TPUs:** Trillium (v6e) and Ironwood (tpu7x, GA) for large XLA/transformer workloads, usually accessed through Vertex AI.
- Use **Deep Learning VM images/containers** to avoid driver pain, **Spot VMs** with frequent checkpointing to slash training cost, committed-use discounts for steady serving, and always **stop idle GPU VMs**.

## Try it

Practice right-sizing and cost-aware provisioning:

1. Launch an on-demand `g2-standard-8` (L4) VM from a Deep Learning image and confirm the GPU is visible with `nvidia-smi`.
2. Launch a second training-class VM (e.g. `a2-highgpu-1g`) with `--provisioning-model=SPOT` and `--instance-termination-action=STOP`, and note the price difference in the create output versus on-demand.
3. Write a tiny script on the spot VM that checkpoints a counter to a Cloud Storage bucket every few seconds, so you can see how a preemption-resilient job is structured.
4. Stop both VMs and confirm with `gcloud compute instances list` that they no longer bill for compute — then delete them. Reflect on which of these you would run on Vertex AI instead, and why.
