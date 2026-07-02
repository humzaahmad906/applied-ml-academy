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

Before you hard-code a machine type, discover what a zone actually offers and filter to the shape you want — this is how you find, say, the smallest N-series that clears a memory floor:

```bash
gcloud compute machine-types list \
  --filter="zone:us-central1-a AND guestCpus>=16 AND memoryMb>=65536" \
  --format="table(name, guestCpus, memoryMb)"

# Which accelerators exist in a zone (and their per-VM count limits)
gcloud compute accelerator-types list --filter="zone:us-central1-a"
```

Instances have a lifecycle you drive from the CLI, not just a create-and-forget call. These are the operations you run constantly:

```bash
gcloud compute instances list
gcloud compute instances describe prep-node --zone=us-central1-a \
  --format="value(status, machineType.basename())"
gcloud compute instances stop prep-node --zone=us-central1-a    # deallocate CPU/GPU
gcloud compute instances start prep-node --zone=us-central1-a
gcloud compute instances reset prep-node --zone=us-central1-a   # hard reboot
gcloud compute instances delete prep-node --zone=us-central1-a

# Resize a stopped VM (right-size after profiling instead of recreating)
gcloud compute instances set-machine-type prep-node \
  --zone=us-central1-a --machine-type=n4-standard-32
```

**Gotcha — per-second billing, and stopped VMs still cost.** Compute bills per second (after a one-minute minimum). Stopping a VM stops the *compute* charge, but any attached **persistent disks keep billing** while the VM is stopped, and so do reserved static IPs. "Stopped" is not "free" — delete the disk (or the whole VM) when you are truly done.

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

Two GPU gotchas bite everyone at least once:

- **`--maintenance-policy=TERMINATE` is mandatory for GPU VMs.** GPUs cannot live-migrate, so the host-maintenance policy must be `TERMINATE` (the VM is stopped and restarted around host events). The G2/A-series create commands set this for you or require it; if you script a raw instance with an accelerator and leave the default `MIGRATE`, the create fails.
- **GPU quota defaults to zero and must be requested.** A fresh project has a `0` limit on GPUs of a given type per region (see the quota discussion in module 01). Your first `create` with a GPU fails with `QUOTA_EXCEEDED` until you request an increase in **IAM & Admin → Quotas**. Request it *before* the run, and remember the quota is per-region and per-GPU-type.

For guaranteed capacity — so the frontier GPUs you need are actually available when a run starts — use **reservations** (block a specific count of a machine type in a zone; the VMs bill whether or not they run, but the capacity is yours) or **sole-tenancy** node groups (dedicated physical hosts, for licensing or isolation needs). Reservations are the pragmatic answer to "the A3/A4 create keeps failing with `ZONE_RESOURCE_POOL_EXHAUSTED`."

## TPUs

**Cloud TPUs** are Google's custom ML accelerators, ideal for large transformer and JAX/PyTorch-XLA workloads. The current generations:

- **Trillium (v6e)** — the sixth-generation TPU, strong for training, fine-tuning, and serving of transformers, text-to-image, and CNNs.
- **Ironwood (tpu7x)** — the seventh generation, purpose-built for the age of large-model inference and training, with a large jump in per-chip compute and memory (192 GB per chip) over Trillium. Ironwood is generally available.
- Earlier **v5e** (cost-efficient) and **v5p** (high-performance training) remain in use.

TPUs are referenced by their generation on all API surfaces (for example `v6e`). They shine when your framework supports XLA well and your model maps cleanly to their systolic-array design; for arbitrary PyTorch code with custom CUDA kernels, GPUs are usually the pragmatic choice. Most ML engineers reach TPUs through Vertex AI training rather than raw Compute Engine.

## Images and environments

A VM boots from an **image**. For ML you rarely start from a bare OS — instead use the **Deep Learning VM images** (the `deeplearning-platform-release` project), which come with CUDA, cuDNN, and a preinstalled framework matched to driver versions (image families like `pytorch-latest-gpu`, or pinned ones such as `pytorch-2-9-cu129-ubuntu-2204`). This saves the hours of driver-versioning pain that plague fresh GPU boxes. Note the active families are now **PyTorch** and **base CUDA**; the standalone TensorFlow and CPU-only Deep Learning VM images have been deprecated, so bake TensorFlow into your own container if you need it. For JAX, use the dedicated JAX AI container images. You can snapshot a configured VM into a **custom image** to standardize your team's environment, though the more reproducible path is to bake your environment into a **container** (covered later) and run it on Vertex AI or Cloud Run.

You can also boot a VM that runs a container image directly with `gcloud compute instances create-with-container`, which starts the VM on Container-Optimized OS and launches your container on boot — a lightweight way to run a serving container without a full orchestrator:

```bash
gcloud compute instances create-with-container infer-box --zone=us-central1-a \
  --machine-type=g2-standard-8 --maintenance-policy=TERMINATE \
  --container-image=us-central1-docker.pkg.dev/myco-fraud-prod/serving/infer:latest
```

## Disks, snapshots, and images

Every VM has a **boot disk**, and you attach additional **persistent disks** for data. Disk type is a performance and cost lever that matters for training throughput: `pd-balanced` is the sensible default, `pd-ssd` for IOPS-hungry workloads, and `hyperdisk-ml` is the read-optimized option built for streaming large training datasets and model weights to many VMs at once. For the highest-throughput scratch space — shuffling a dataset during training — attach **Local SSD**, physically-attached NVMe that is far faster than any network disk but **ephemeral** (its data is lost when the VM stops or is preempted), so use it only for regenerable scratch, never the sole copy of anything.

```bash
# Create and attach a data disk; detach when done
gcloud compute disks create fraud-scratch --zone=us-central1-a \
  --size=1000GB --type=pd-ssd
gcloud compute instances attach-disk train-node --zone=us-central1-a --disk=fraud-scratch
gcloud compute instances detach-disk train-node --zone=us-central1-a --disk=fraud-scratch
gcloud compute disks list

# Attach Local SSD at create time (ephemeral, highest throughput)
gcloud compute instances create train-node --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --local-ssd=interface=NVME \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE
```

**Snapshots** are point-in-time, incremental backups of a persistent disk (good for durability and moving data across zones); a custom **image** is what you boot new VMs from and is how you standardize a team environment:

```bash
gcloud compute snapshots create fraud-scratch-snap \
  --source-disk=fraud-scratch --source-disk-zone=us-central1-a
gcloud compute snapshots list

# Bake a configured VM's boot disk into a reusable image
gcloud compute images create fraud-train-base \
  --source-disk=train-node --source-disk-zone=us-central1-a \
  --family=fraud-train
gcloud compute images list --filter="family=fraud-train"
```

## Connecting: SSH, IAP tunneling, and OS Login

You reach a VM with `gcloud compute ssh`, which manages keys for you. In a locked-down network your GPU boxes should have **no external IP**; reach them instead through **Identity-Aware Proxy (IAP) TCP forwarding**, which tunnels SSH over an IAP-authorized connection so the VM stays off the public internet. **OS Login** ties SSH access to IAM (grant `roles/compute.osLogin`) instead of manually managed key files — the same least-privilege story as module 02, applied to shell access.

```bash
# SSH via IAP tunnel to a VM with no external IP
gcloud compute ssh train-node --zone=us-central1-a --tunnel-through-iap

# Enable OS Login project-wide, then run a command remotely over the tunnel
gcloud compute project-info add-metadata --metadata=enable-oslogin=TRUE
gcloud compute ssh train-node --zone=us-central1-a --tunnel-through-iap \
  --command="nvidia-smi"
```

## Serving fleets: instance templates and managed instance groups

A single VM is fine for experimentation, but a production serving tier needs to scale with traffic and self-heal. That is what a **managed instance group (MIG)** does: you define an **instance template** (the immutable spec — machine type, image, disks, GPU) once, then a MIG maintains N identical VMs from it, recreates any that fail, and **autoscales** on CPU, load-balancer utilization, or a custom Cloud Monitoring metric. This is the raw-Compute pattern behind an autoscaling GPU inference fleet (though for most model serving you will reach for Vertex endpoints or Cloud Run, which do this for you).

```bash
# 1. Template: an L4 serving box
gcloud compute instance-templates create infer-tmpl \
  --machine-type=g2-standard-8 --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release

# 2. MIG from the template
gcloud compute instance-groups managed create infer-mig \
  --template=infer-tmpl --size=2 --zone=us-central1-a

# 3. Autoscale between 2 and 10 replicas on a CPU target
gcloud compute instance-groups managed set-autoscaling infer-mig \
  --zone=us-central1-a \
  --min-num-replicas=2 --max-num-replicas=10 \
  --target-cpu-utilization=0.6 --cool-down-period=90
```

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

`--instance-termination-action` decides what happens on preemption: `STOP` (deallocate but keep the disk so you can restart and resume from a checkpoint — the right choice for training) or `DELETE` (throw the VM away entirely — for fully stateless batch work). You can also convert an existing standard VM to spot, or back, by stopping it and running `set-scheduling`:

```bash
gcloud compute instances set-scheduling train-node --zone=us-central1-a \
  --provisioning-model=SPOT --instance-termination-action=STOP
```

Beyond spot, two discounts apply *automatically or by commitment* on standard VMs. **Sustained-use discounts** kick in automatically as a VM (of eligible families) runs a large fraction of the month — no action needed. **Committed-use discounts (CUDs)** are an explicit 1- or 3-year commitment to a baseline of vCPU/memory (or specific GPUs) for a large discount — the right lever for always-on serving whose floor you can predict:

```bash
gcloud compute commitments create serving-cud --region=us-central1 \
  --plan=twelve-month --resources=vcpu=32,memory=128GB
```

The simplest lever of all is **turning things off**: idle GPU VMs bill by the second whether or not they compute, so stop them, and prefer autoscaling/scale-to-zero serving where possible.

## When to use raw Compute Engine vs managed services

Reach for raw Compute Engine when you need full control of the environment, custom drivers or kernels, interactive multi-day experimentation on a fixed box, or an architecture the managed services do not fit. Reach for **Vertex AI** for training and prediction when you want managed job lifecycle, autoscaling, and MLOps integration; **Cloud Run** or **GKE** for containerized serving. The default for most ML work is a managed service — but that service is provisioning the exact machine types and accelerators you just learned, so this knowledge transfers directly.

## How this fits the whole solution

Compute is the engine room of the end-to-end system. Preprocessing runs on general-purpose or compute-optimized VMs (or Dataflow); training runs on GPU/TPU accelerators (usually via Vertex AI, which selects these same machine types); batch scoring runs cheaply on spot capacity; and serving runs on L4-class GPUs or CPUs behind an autoscaler. The choices you make here — right-sizing the accelerator to the task, using spot for interruptible work, checkpointing to survive preemption, and shutting off idle capacity — are the difference between a system that costs hundreds and one that costs thousands per month for the same output.

## Key takeaways

- Machine **families** target workloads: general-purpose and compute-optimized for data prep, accelerator-optimized (A-series, G2) for ML.
- **GPU lineup (2026):** G2/L4 for cost-efficient serving; A2/A100 and A3/H100–H200 for training; A4/B200 and A4X/GB200 for frontier scale. Match the chip to the job and verify regional availability.
- **TPUs:** Trillium (v6e) and Ironwood (tpu7x, GA) for large XLA/transformer workloads, usually accessed through Vertex AI.
- Use **Deep Learning VM images/containers** to avoid driver pain, **Spot VMs** with frequent checkpointing to slash training cost, committed-use discounts for steady serving, and always **stop idle GPU VMs**.

## CLI cheat-sheet

```bash
# --- Discover what's available in a zone ---
gcloud compute machine-types list --filter="zone:us-central1-a AND guestCpus>=16"
gcloud compute accelerator-types list --filter="zone:us-central1-a"
gcloud compute images list --project=deeplearning-platform-release --filter="family~pytorch"
gcloud compute regions describe us-central1     # regional quota / capacity view

# --- Create VMs ---
# CPU prep node
gcloud compute instances create prep-node --zone=us-central1-a \
  --machine-type=n4-standard-16 \
  --image-family=debian-12 --image-project=debian-cloud
# L4 serving GPU (TERMINATE maintenance policy is required for GPUs)
gcloud compute instances create infer-node --zone=us-central1-a \
  --machine-type=g2-standard-8 --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release
# Spot training GPU that stops (not deletes) on preemption
gcloud compute instances create train-spot --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release

# --- Lifecycle (stopped != free: disks and static IPs still bill) ---
gcloud compute instances list
gcloud compute instances describe VM --zone=ZONE
gcloud compute instances stop  VM --zone=ZONE     # release CPU/GPU charge
gcloud compute instances start VM --zone=ZONE
gcloud compute instances reset VM --zone=ZONE     # hard reboot
gcloud compute instances set-machine-type VM --zone=ZONE --machine-type=TYPE  # resize (stopped)
gcloud compute instances delete VM --zone=ZONE

# --- Disks, snapshots, images ---
gcloud compute disks create DISK --zone=ZONE --size=1000GB --type=pd-ssd
gcloud compute instances attach-disk VM --zone=ZONE --disk=DISK
gcloud compute instances detach-disk VM --zone=ZONE --disk=DISK
gcloud compute snapshots create SNAP --source-disk=DISK --source-disk-zone=ZONE
gcloud compute images create IMG --source-disk=VM --source-disk-zone=ZONE --family=FAM
# Local SSD (ephemeral scratch) is added at create time:
#   gcloud compute instances create VM ... --local-ssd=interface=NVME

# --- Connect (via IAP, no external IP needed) ---
gcloud compute ssh VM --zone=ZONE --tunnel-through-iap --command="nvidia-smi"

# --- Serving fleet: template + MIG + autoscaling ---
gcloud compute instance-templates create TMPL --machine-type=g2-standard-8 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release
gcloud compute instance-groups managed create MIG --template=TMPL --size=2 --zone=ZONE
gcloud compute instance-groups managed set-autoscaling MIG --zone=ZONE \
  --min-num-replicas=2 --max-num-replicas=10 --target-cpu-utilization=0.6

# --- Cost levers ---
gcloud compute commitments create my-cud --region=us-central1 \
  --plan=twelve-month --resources=vcpu=32,memory=128GB   # committed-use discount
gcloud compute instances set-scheduling VM --zone=ZONE \
  --provisioning-model=SPOT --instance-termination-action=STOP  # convert to spot (stopped VM)
```

## Try it

Practice right-sizing and cost-aware provisioning:

1. Launch an on-demand `g2-standard-8` (L4) VM from a Deep Learning image and confirm the GPU is visible with `nvidia-smi`.
2. Launch a second training-class VM (e.g. `a2-highgpu-1g`) with `--provisioning-model=SPOT` and `--instance-termination-action=STOP`, and note the price difference in the create output versus on-demand.
3. Write a tiny script on the spot VM that checkpoints a counter to a Cloud Storage bucket every few seconds, so you can see how a preemption-resilient job is structured.
4. Stop both VMs and confirm with `gcloud compute instances list` that they no longer bill for compute — then delete them. Reflect on which of these you would run on Vertex AI instead, and why.
