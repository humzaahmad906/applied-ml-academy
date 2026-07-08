# 17b — GPU Capacity and Resilience

Module 09 trained the fraud model and module 17 wired up features, experiments, and lineage — all of which quietly assume the accelerators you asked for actually exist when the job runs. In 2026 that assumption is the first thing to break. High-end GPUs and TPUs are chronically capacity-constrained: you submit a training job requesting eight H200s and Vertex returns a `STOCKOUT` error, or your endpoint autoscaler tries to add a node during a regional crunch and can't. This module is about the two failure modes that follow from scarcity — **getting the hardware at all**, and **staying up when a region or a preemptible node goes away**. It uses `gcloud` and the `aiplatform` SDK, and everything here changes fast, so treat specific machine names and quotas as *check-current-docs* facts.

## The accelerator families you're actually choosing between

Before you can request capacity you need to know what to request. As of mid-2026 the accelerator-optimized families are, roughly newest to oldest:

- **A4X Max** — NVIDIA GB300 (Grace Blackwell Ultra) superchips, the exascale tier for the largest foundation-model training.
- **A4X** — NVIDIA GB200 NVL72 (Grace + Blackwell), purpose-built for very large-scale training and long-context / reasoning models.
- **A4** — 8× NVIDIA B200 (Blackwell) per machine; the mainstream large-scale training tier, ~3× the throughput of the prior generation.
- **A3 Ultra / A3 Mega** — NVIDIA H200 and H100; still the workhorse for most fine-tuning and mid-scale training.
- **G4 / L4** — smaller GPUs for inference and light fine-tuning.

On the TPU side: **v5e** (cheap entry point, cost-efficient inference and small training), **v5p** (large-scale training), **v6e "Trillium"** (best perf/$ for transformers and CNNs), and **v7 "Ironwood" (TPU7x)** for the largest dense and MoE workloads. Google has also announced NVIDIA Vera Rubin instances for later in 2026. Which of these you can even *see* depends on region and quota — the newest families land in a handful of regions first. **Verify the exact machine type, region availability, and Vertex support before you architect around it; this list is the fastest-moving fact in the whole course.**

For the fraud model, none of this exascale hardware is warranted — a single H100 or L4 fine-tunes an XGBoost or small neural net fine. The capacity *techniques* below matter regardless of tier, and they matter more the scarcer your chosen accelerator is.

## Step one: quota, not code

The first wall is almost never "no capacity" — it's **quota**. A new project has near-zero GPU/TPU quota. Requesting hardware you have no quota for fails immediately, before scheduling ever tries. Check and raise quota first:

```bash
# What GPU quota do I actually have in this region?
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric, quotas.limit, quotas.usage)" \
  | grep -i gpu

# Preemptible/Spot quotas are tracked separately from on-demand
gcloud compute project-info describe \
  --format="value(quotas)" | tr ',' '\n' | grep -i nvidia
```

Quota increases are requested from the **IAM & Admin → Quotas** console page or the Cloud Quotas API. High-end accelerator quota (A4, H200, TPU v6e+) often requires a **sales / capacity conversation**, not just a self-serve bump — budget days, not minutes. The lesson: quota is a lead-time item you handle *before* the sprint that needs it, not the morning of.

## Getting scarce accelerators: the four paths

Once quota exists, you have four ways to actually land accelerators, trading off commitment, urgency, and price.

**1. On-demand.** Ask and hope. Fine for L4/small GPUs; unreliable for A4/H200/TPU v6e, where on-demand frequently returns stockouts.

**2. Dynamic Workload Scheduler (DWS).** DWS is the Vertex-native answer to "there are none right now." It has two modes:

- **Flex Start** — you submit the job and DWS runs it *as soon as* all requested accelerators are simultaneously available, up to a 7-day run. You trade "start now" for "start soon, cheaper, and it actually runs." This is the default recommendation for L4/A100/H100/H200/B200 training jobs where you can tolerate a queue. In a Vertex custom job you set the scheduling strategy to `FLEX_START`:

```python
from google.cloud import aiplatform

aiplatform.init(project="myco-fraud-dev", location="us-central1")

job = aiplatform.CustomJob(
    display_name="fraud-train-flex",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "a3-highgpu-8g",
            "accelerator_type": "NVIDIA_H100_80GB",
            "accelerator_count": 8,
        },
        "replica_count": 1,
        "container_spec": {"image_uri": "us-docker.pkg.dev/myco-fraud-dev/train/fraud:latest"},
    }],
)
job.run(scheduling_strategy=aiplatform.compat.types.Scheduling.Strategy.FLEX_START)
```

- **Calendar mode** (preview in 2026) — reserves *co-located* GPU/TPU capacity for a fixed future window, up to **90 days**, without a long-term commitment. It extends Compute Engine future reservations. This is the right tool when you know you need N accelerators for a two-week training run starting the 14th and you want them guaranteed, not queued. Flex Start and Calendar mode are being extended to the newest families (A4X, A4X Max, G4) through 2026 — **check which modes your target family supports.**

**3. Reservations + Committed Use Discounts (CUDs).** For steady, predictable load — an always-on serving fleet or a team that trains continuously — buy a **reservation** (guarantees the capacity exists in a zone) and attach a **1- or 3-year committed use discount** for the price break. Reservations remove the stockout risk entirely at the cost of paying whether or not you use them. This is the *opposite* end from DWS: maximum certainty, maximum commitment.

**4. Spot VMs.** Spot capacity is up to ~80% cheaper than on-demand, has no 24-hour cap (unlike the deprecated preemptible VMs it replaced), and can be **reclaimed at any time with ~30 seconds' notice**. Spot is the cheapest way to get scarce GPUs — *if* your job survives interruption. Which brings us to checkpointing.

## Spot + checkpointing: cheap training that survives preemption

A Spot training job that doesn't checkpoint is a bet you will eventually lose: preemption throws away all progress. Vertex retries a preempted job up to six times, but each retry restarts from wherever your code resumes — so the whole strategy hinges on **frequent checkpoints to durable storage plus resume-on-start**.

Enable Spot on a custom job and make the code robust:

```python
job.run(scheduling_strategy=aiplatform.compat.types.Scheduling.Strategy.SPOT)
```

```python
import os, signal, torch

CKPT = "gs://myco-fraud-ckpts/fraud/latest.pt"  # durable GCS, not local disk

def save_checkpoint(model, opt, step):
    torch.save({"step": step, "model": model.state_dict(),
                "opt": opt.state_dict()}, "/tmp/ckpt.pt")
    # then upload /tmp/ckpt.pt -> CKPT with google-cloud-storage / gcsfs

# Handle the ~30s preemption warning: flush a checkpoint on SIGTERM
signal.signal(signal.SIGTERM, lambda *_: save_checkpoint(model, opt, step))

# On startup, resume if a checkpoint exists (name the missing keys, no silent strict=False)
start_step = maybe_load_checkpoint(model, opt, CKPT)
```

Three rules make this work. **Checkpoint to Cloud Storage, never local disk** — the node vanishes on preemption. **Checkpoint frequently** — the guidance is at least every four hours even on-demand, and more often on Spot; a good heuristic is "cheaper to re-run than to save more often." **Handle SIGTERM** for a graceful final flush inside the ~30-second window. Cost note: Spot turns a $30/hr 8×H100 job into roughly $6/hr, so even with occasional restart waste it's dramatically cheaper — the tradeoff is wall-clock unpredictability, which is fine for research runs and wrong for a deadline-bound production retrain.

**Rule of thumb for capacity:** deadline-bound and predictable → reservation + CUD; needs to run soon but flexible → DWS Flex Start; scheduled future block → Calendar mode; cost-sensitive and interruption-tolerant → Spot + checkpointing.

## Multi-region and DR for ML

Getting the model trained is half the resilience story; the other half is surviving a regional outage in production. Frame it with two numbers: **RPO** (recovery point objective — how much data/state you can afford to lose) and **RTO** (recovery time objective — how long you can be down).

**Model artifacts.** Store trained models and checkpoints in a **dual-region or multi-region GCS bucket**, not a single-region one. Dual-region buckets use an **active-active** architecture: if one region is down, objects are transparently served from the other with an effective **RTO of zero** — no manual failover, no failback. Default replication targets 99.9% of new objects within an hour; if your RPO is tighter, enable **turbo replication** for a 15-minute RPO backed by SLA (at extra cost).

```bash
# Dual-region bucket for model artifacts, with turbo replication (15-min RPO)
gcloud storage buckets create gs://myco-fraud-models \
  --location=nam4 \
  --enable-turbo-replication

# Or a broad multi-region for read-anywhere artifacts
gcloud storage buckets create gs://myco-fraud-models-mr --location=us
```

**Serving.** A Vertex endpoint is **regional** and, within a region, gets multi-zone HA (99.9% SLA) — that survives a zone failure but *not* a region failure. For cross-region resilience you choose an architecture:

- **Active-passive.** Deploy the model to endpoints in two regions; route all traffic to the primary and fail over to the secondary on outage. Lower cost (the passive fleet can be minimal), higher RTO (failover time), and it depends on health-checking and DNS/router changes.
- **Active-active.** Serve from endpoints in multiple regions simultaneously behind a **Global External Load Balancer** (with serverless NEGs) or a Cloud Run "smart router" using Private Service Connect. Traffic is balanced continuously, so a region loss is near-transparent — lowest RTO, highest cost (full fleets in both regions). Google's managed **multi-region endpoints** (GA for Claude models as of May 2026) do this pooling-and-failover automatically; for your *own* custom-container model you build the load-balancer pattern yourself.

Because model artifacts already live in a dual-region bucket, standing up the passive/second endpoint is just "deploy the same artifact in region B" — the DR story for ML is mostly *artifact portability plus a routing decision*. The reproducibility machinery from module 17 (Model Registry, lineage) is what lets region B serve a byte-identical model.

## Key takeaways

- **Quota is the first wall, not capacity.** New projects have near-zero GPU/TPU quota; high-end accelerator quota may need a sales conversation. Raise it days ahead, before the sprint needs it.
- **Know your family.** 2026 tiers run A4X Max (GB300) → A4X (GB200) → A4 (B200) → A3 Ultra (H200/H100) → G4/L4, and TPU v5e/v5p/v6e Trillium/v7 Ironwood. Names, regions, and Vertex support move monthly — **check current docs.**
- **Four ways to get scarce accelerators:** on-demand (unreliable at the top end), **DWS Flex Start** (`FLEX_START`, runs soon, ≤7 days), **DWS Calendar mode** (reserve a future block ≤90 days, preview), **reservations + CUDs** (certainty for steady load), and **Spot** (~80% off, reclaimable in ~30s).
- **Spot only works with checkpointing:** durable **GCS** checkpoints, frequent saves (≥ every 4h), **SIGTERM** handler for a final flush, resume-on-start. Vertex retries a preempted job up to six times.
- **DR for ML = artifact portability + routing.** Dual-region GCS (active-active, RTO≈0; turbo replication for 15-min RPO) for artifacts; endpoints are regional (multi-zone HA only), so choose **active-passive** (cheaper, higher RTO) or **active-active** (LB/smart-router, lowest RTO, higher cost) for cross-region serving.

## Try it

Harden the fraud training and serving path against scarcity and outage:

1. Run `gcloud compute regions describe us-central1` and confirm your GPU and **Spot** quotas; identify one accelerator family from the 2026 list you'd actually have quota for.
2. Submit the fraud training `CustomJob` with `scheduling_strategy=FLEX_START` and observe it queue-then-run; compare against an on-demand submit in a busy region.
3. Add **SIGTERM-driven GCS checkpointing** and resume-on-start to the training code, switch the job to `SPOT`, and force-verify it resumes from a checkpoint after an interruption.
4. Recreate the model-artifact bucket as **dual-region** (`--location=nam4 --enable-turbo-replication`) and re-point the training output; note the RPO change.
5. Deploy the registered model to endpoints in **two regions** and sketch an **active-passive** vs **active-active** router in front of them — write down the RTO/RPO and monthly cost you're buying with each.
