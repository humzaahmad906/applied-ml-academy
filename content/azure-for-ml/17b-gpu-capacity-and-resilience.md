# 17b — GPU Capacity and Resilience

The previous modules assumed the compute you asked for was there when you asked. In production that assumption breaks twice: first when you *cannot get a GPU at all* because the region is out of H100s or your subscription's GPU quota is literally zero, and again when a region-wide outage takes your training runs and scoring endpoints down together. Both are capacity problems — one is "there is no supply," the other is "the supply I had disappeared" — and both have concrete Azure mechanisms. This module covers getting scarce GPUs (quota increases, capacity reservations, Spot) and surviving the loss of a region (geo-redundant storage, multi-region workspaces and endpoints, RTO/RPO framing). GPU SKUs and regional availability change fast, so treat every SKU name below as *check current docs* before you commit budget.

## Getting a GPU when there are none

The first surprise for most teams: **specialized GPU VM families default to zero cores.** A fresh subscription can spin up general-purpose D-series VMs, but the N-series families that carry GPUs — `NCADSH100v5`, `NCAST4v3`, the `ND` families — start at a hard limit of zero. You do not get "capacity denied" from a scheduler; you get a quota error at submit time, because you were never allocated any cores to begin with. Check what you actually have before you plan a run:

```bash
# What N-series (GPU) core quota do I have in this region, and how much is used?
az vm list-usage --location eastus2 -o table | grep -Ei "name|NC|ND|standard N"
```

Quota in Azure is **per VM family, per region.** "GPU quota" is not one number — `Standard NCADSH100v5 Family vCPUs` and `Standard NDSH100v5 Family vCPUs` are separate limits, and East US 2 is separate from Sweden Central. Raising one does nothing for the others. You request an increase from the **Usage + quotas** blade in the portal (or the Quotas view inside Azure ML studio), selecting the exact family and region and entering the new *vCPU* ceiling — quotas are counted in vCPUs, not GPUs, so an 8-GPU `ND96isr_H100_v5` node consumes 96 vCPUs of the ND-H100-v5 family quota. Modest bumps on common families clear in minutes; **specialized GPU increases routinely take 1–5 business days** and large asks may route to a capacity team, so file the request well before the run you need it for. There is no supported CLI verb to *approve* a quota bump (it is a support-backed operation), but you can script the submission via the Quotas REST API / `az rest` if you file many.

**Quota is permission, not a promise.** Having 96 vCPUs of ND-H100-v5 quota means Azure *lets* you ask for that node; it does not guarantee the node exists in the region at the moment you deploy. Popular GPU SKUs are frequently sold out region-by-region. This is where the two mechanisms below come in.

### On-Demand Capacity Reservations

An **On-Demand Capacity Reservation (CRG)** reserves actual hardware capacity for a specific VM size in a specific region (optionally pinned to an availability zone), for as long as you keep it — with **no one- or three-year term commitment** (that is what distinguishes it from a Reserved Instance, which is a *billing* discount, not a capacity guarantee). Once the reservation succeeds, that capacity is held for you and cannot be handed to another tenant, so your training node is there when you deploy against it. The cost model is the catch: **you pay the pay-as-you-go rate for the reserved size whether or not a VM is running on it** — an idle reservation for eight H100s bills like eight running H100s. Reserve only for the window you genuinely need the guarantee.

```bash
# 1) Create the reservation group (the container), then 2) the reservation itself.
az capacity reservation group create \
  --resource-group rg-mlx-dev --name crg-mlx-h100 --location eastus2

az capacity reservation create \
  --resource-group rg-mlx-dev --capacity-reservation-group crg-mlx-h100 \
  --name res-h100 --sku Standard_ND96isr_H100_v5 --capacity 2 --location eastus2
```

If Azure lacks the capacity at creation time, **the reservation deployment fails** — a reservation cannot conjure hardware that is not there; it can only fence off hardware that is. You also need enough family quota to cover the reserved quantity *before* you reserve. Azure ML managed compute clusters cannot always be pointed at a CRG directly through the `az ml` surface today, so a common pattern is to reserve capacity and consume it from IaaS/AKS attached compute; **check current docs** for whether managed cluster CRG association has shipped for your SKU.

### Spot / low-priority for interruptible training

At the opposite end of the cost curve: **Spot VMs** run on Azure's spare capacity at up to ~90% off pay-as-you-go, in exchange for the right to be **evicted with about 30 seconds' notice** when Azure needs the hardware back. Note the 2026 change: **Azure ML low-priority compute is being retired (around March 31, 2026); Spot is the supported replacement.** If you still have `tier: low_priority` clusters, migrate them.

Spot is ideal for training *only if the job is checkpointable* — a 30-second warning is not enough to gracefully finish, but it is enough to have already been writing checkpoints. Create a Spot-priority cluster and design the job to resume:

```bash
az ml compute create --name gpu-spot --type amlcompute \
  --size Standard_NCADSH100v5 --tier low_priority \
  --min-instances 0 --max-instances 4 \
  -g rg-mlx-dev -w mlw-mlx-dev
# NOTE: --tier low_priority selects Spot-class capacity for AML clusters (check current CLI).
```

The engineering discipline that makes Spot pay off is **checkpoint-and-resume**: write model + optimizer + step/epoch state to the mounted output store every N steps, and on job start, look for the latest checkpoint and continue from it rather than from scratch. On eviction the node is reclaimed; the cluster (with `min-instances 0`) re-provisions when capacity returns, the job restarts, finds the last checkpoint, and loses only the work since it. Without checkpointing, Spot is a trap — a 12-hour run evicted at hour 11 has cost you 11 hours and produced nothing. Set seeds and persist RNG state in the checkpoint too, or a resumed run diverges from the interrupted one.

A pragmatic mix: **Spot for exploration and sweeps** (cheap, many short jobs, interruption is tolerable), **on-demand dedicated for the final training run** (you want it to finish), and **a capacity reservation only when even dedicated is selling out** for the SKU you need.

## Multi-region and disaster recovery for ML

The blunt fact to internalize first: **Azure Machine Learning does not provide automatic cross-region failover.** A workspace lives in one region. Azure ML **cannot sync or recover artifacts or metadata between workspaces**, and **run history is not backed up or restorable.** So "DR for ML" is not a switch you flip — it is an architecture you build, and you decide how much of it is worth the cost.

Frame it with two numbers. **RPO (Recovery Point Objective)** — how much data/work you can afford to lose, measured in time. **RTO (Recovery Time Objective)** — how long you can afford to be down before service is restored. A batch-scoring pipeline might tolerate RTO of hours and RPO of a day; a fraud endpoint in the request path needs RTO in minutes and RPO near zero. These two numbers drive every choice below.

### Geo-redundant storage for model artifacts

The workspace's backing storage account (models, data assets, job outputs) should use **GRS** (geo-redundant: async replication to a paired secondary region) or **GZRS** (zone-redundant in the primary *and* geo-replicated to the secondary) rather than the default LRS. Two properties matter for your RPO/RTO math: replication is **asynchronous**, so a regional loss can cost you **up to ~15 minutes** of the most recent writes (that is your storage-layer RPO floor), and **failover is not automatic** — you (or an automation) must initiate the account failover to the secondary. A registered model artifact written just before a regional outage may not have replicated; a promotion-based MLOps flow that keeps the *golden* artifacts in a **shared registry** (module 17) with its own redundancy is more robust than relying on a single workspace's storage.

```bash
# Provision the workspace storage account as GZRS (do this at creation; changing SKU later is constrained)
az storage account create --name stmlxgzrs --resource-group rg-mlx-dev \
  --location eastus2 --sku Standard_GZRS --kind StorageV2
```

### Workspace and endpoint resilience

Because nothing syncs automatically, the workable pattern is **infrastructure-as-code deployed to two regions.** Your CI/CD pipeline provisions a primary workspace (e.g. East US 2) and a secondary (e.g. West US 3), and — critically — **deploys to both**, so environments, components, compute definitions, and endpoint deployments stay identical and do not drift. Models and data assets are pushed to both workspaces (or pulled from a shared registry) as part of the release. Recreating a workspace by hand during an outage is how RTOs blow past their target.

For **online endpoints** in the request path, put the two regional managed online endpoints behind **Azure Traffic Manager** (DNS-based global load balancer with health probes and automatic failover):

- **Active-passive:** Traffic Manager routes all traffic to the primary endpoint and fails over to the secondary only when health probes mark the primary down. Simpler and cheaper (the secondary can run minimal instances), but failover incurs DNS-TTL delay and cold-scale time — an RTO of minutes, not seconds.
- **Active-active:** both regional endpoints serve live traffic (weighted or performance routing). A region loss is absorbed with no manual step and near-zero RTO, at the cost of running full capacity in both regions continuously — roughly double the serving spend.

Choose per workload against its RTO/RPO: most teams run the fraud/real-time endpoints active-active and everything else active-passive or single-region with a documented manual failover runbook. **Check current docs** for managed-online-endpoint regional availability and whether native multi-region endpoint features have shipped beyond the Traffic Manager pattern.

## Key takeaways

- **GPU families default to zero quota.** Quota is per-family, per-region, counted in **vCPUs** (an 8×H100 `ND96isr_H100_v5` = 96 vCPUs); request increases early — specialized GPU bumps take **1–5 business days**.
- **Quota is permission, not supply.** An **On-Demand Capacity Reservation** fences off real hardware with no term commitment, but bills at PAYG **whether or not a VM runs on it**, and fails to create if the region lacks capacity.
- **Spot** (the replacement for retiring Azure ML low-priority, ~March 2026) is up to ~90% cheaper but evicts with **~30s notice** — only viable with **checkpoint-and-resume** (persist model+optimizer+RNG state; `min-instances 0` so the cluster re-provisions).
- Current GPU families to verify: **ND-H100-v5** (8× H100 SXM5, NVLink, InfiniBand), **ND-H200-v5** (141 GB HBM3e), **ND-GB200-v6** (Blackwell), **NCads H100 v5** (single H100 NVL 94 GB) — all *check current docs* for regional availability.
- **Azure ML has no automatic DR:** no cross-workspace sync, run history not restorable. Use **GRS/GZRS** storage (async, ~15-min RPO floor, **manual** failover), **IaC deployed to two regions** to prevent drift, and **Traffic Manager** (active-passive vs active-active) for endpoint failover — sized to each workload's **RTO/RPO**.

## Try it

Run `az vm list-usage` in two regions and find the current limit and usage for one ND or NC family — note whether it is zero. File a quota-increase request for a GPU family (even a small one) and time how long approval takes. Create a capacity reservation group and a small reservation for a GPU size you have quota for, confirm it succeeds (or read the failure if capacity is unavailable), then **delete it immediately** so it stops billing. Stand up a Spot-tier compute cluster with `min-instances 0`, and sketch the checkpoint-and-resume logic your training script would need: where it writes checkpoints, how it finds the latest on restart, and what state (weights, optimizer, step, RNG) must be in the checkpoint for a resumed run to match an uninterrupted one. Finally, for one real endpoint, write down its target RTO and RPO, then decide whether it justifies active-passive or active-active behind Traffic Manager — and price the difference.
