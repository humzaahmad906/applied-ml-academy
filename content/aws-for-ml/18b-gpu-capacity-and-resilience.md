# 18b — GPU Capacity and Resilience

Two problems bite teams the moment they move ML from a demo to a business: you cannot get the GPUs, and when a Region has a bad day your model stops serving. Neither is a modeling problem — both are capacity-and-operations problems, and both have concrete AWS answers. This module covers the first half, **getting and keeping accelerated capacity** (quotas, reservations, Capacity Blocks, Spot), and the second half, **surviving a Region failure** for a serving stack (artifact replication, endpoint failover, and an RTO/RPO-driven DR plan). GPU instance families and Capacity Block details change fast; the specific instance types and prices below are July 2026 — always **check current docs** before you commit spend.

## The capacity problem: there are no GPUs

You launch a `p5.48xlarge` training job and get `InsufficientInstanceCapacity`. This is not a bug and not a quota error — it means AWS has no H100 capacity in that Availability Zone right now. Modern accelerators (H100, H200, and the Blackwell B200/B300 generation) are genuinely scarce, so "just launch it" is not a reliable plan for anything large. There are two distinct walls you can hit, and they need different fixes:

- **A quota wall** — your account is *administratively* capped below what you asked for. Fixed by requesting a limit increase. Fast-ish, free.
- **A physical-capacity wall** — the hardware simply is not free in that AZ. Fixed by *reserving* capacity ahead of time, or by using interruptible Spot. Costs money or tolerance for interruption.

Diagnose which wall you hit before reacting: a quota error names the quota; a capacity error says `InsufficientInstanceCapacity`.

## Service Quotas: clearing the administrative cap

GPU launches are gated by the **"Running On-Demand P instances"** and **"Running On-Demand G instances"** quotas — and critically, these are measured in **vCPUs**, not instance count. A single `p5.48xlarge` is 192 vCPUs, so a default account quota can be smaller than one instance. The workflow is always check → request → wait.

```bash
# 1) Check your current P-instance quota (L-417A185B = Running On-Demand P instances)
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-417A185B --region us-east-1

# 2) Request an increase (value is in vCPUs — size it to your largest concurrent job)
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-417A185B \
  --desired-value 384 --region us-east-1

# 3) Track the request
aws service-quotas list-requested-service-quota-change-history \
  --service-code ec2 --region us-east-1
```

Quotas are **per-Region and per-family**, so a G-instance increase in `us-east-1` does nothing for P instances in `us-west-2`. Approvals for large GPU asks are not instant and sometimes route to a human, so request **well ahead** of when you need the capacity. And clearing the quota only removes the administrative wall — it does not conjure hardware. If the physical capacity is not there, you still get `InsufficientInstanceCapacity`, which is what reservations solve.

## Reserving capacity: ODCR, Capacity Blocks, and Spot

Three mechanisms trade off differently across *when* you need capacity and *how much interruption* you tolerate.

**On-Demand Capacity Reservations (ODCR)** reserve capacity in a specific AZ starting *now*, held until you cancel. You pay the On-Demand rate for the reserved capacity whether or not instances are running — the reservation *is* the charge. Use it when you need assured, open-ended capacity (a long-running training cluster, a always-on GPU endpoint).

```bash
# Reserve 2 p5.48xlarge in a specific AZ, held until cancelled
aws ec2 create-capacity-reservation \
  --instance-type p5.48xlarge \
  --instance-platform Linux/UNIX \
  --availability-zone us-east-1a \
  --instance-count 2 \
  --instance-match-criteria open   # any matching instance auto-slots into the reservation
```

With `--instance-match-criteria open`, any instance you launch with matching attributes automatically consumes the reservation; `targeted` requires instances to reference the reservation explicitly, which is safer when you want to guarantee *only* a specific job uses it.

**EC2 Capacity Blocks for ML** are the answer when you need a large GPU cluster for a *future, bounded window* — a two-week fine-tuning run three weeks from now. You reserve a block for a future start date and pay upfront for that window only. As of July 2026, Capacity Blocks support **P6e-GB200, P6-B300, P6-B200, P5en, P5e, P5, and P4d** (Blackwell, H200, H100, and A100 respectively), instances are colocated in **EC2 UltraClusters** for low-latency multi-node training, you can reserve a start time up to **8 weeks out**, and a single block can hold up to **64 instances** (up to 256 across blocks). Pricing is a moving target: AWS raised Capacity Block rates ~15% on Jan 4 2026 and another ~20% on Jul 1 2026 — **check current pricing** before you plan a budget.

```bash
# Find purchasable Capacity Block offerings for a future window, then buy one
aws ec2 describe-capacity-block-offerings \
  --instance-type p5.48xlarge --instance-count 16 \
  --start-date-range 2026-07-20T00:00:00Z \
  --end-date-range 2026-08-03T00:00:00Z \
  --capacity-duration-hours 336

aws ec2 purchase-capacity-block \
  --capacity-block-offering-id cbo-0abcd1234 \
  --instance-platform Linux/UNIX
```

(SageMaker also exposes **training plans**, which wrap Capacity Blocks so a SageMaker training or HyperPod job reserves its future GPU window without you managing raw EC2 — check current docs, as this surface is evolving.)

**Spot** is the opposite trade: 50–90% cheaper than On-Demand, but AWS can reclaim the instance with two minutes' notice. That is fine for training *if and only if you checkpoint*, so an interruption costs minutes, not the whole run. SageMaker **managed spot training** makes this turnkey — it handles the interruption, saves your checkpoints to S3, and resumes from the latest one when capacity returns.

```python
from sagemaker.estimator import Estimator

est = Estimator(
    image_uri=training_image, role=role,
    instance_type="ml.p5.48xlarge", instance_count=1,
    use_spot_instances=True,          # opt into Spot
    max_run=48 * 3600,                # wall-clock cap for the job
    max_wait=72 * 3600,               # must be >= max_run when using Spot (includes wait-for-capacity)
    checkpoint_s3_uri="s3://my-ml-data/checkpoints/run-42/",  # resume point on interruption
)
est.fit({"train": "s3://my-ml-data/train/"})
```

**Cost framing:** the mental model is *certainty vs. price*. Spot is cheapest but interruptible — use it for fault-tolerant training. ODCR/Capacity Blocks cost full rate (or a premium) but guarantee the hardware is there — use them for deadlines and always-on serving. The expensive mistake is reserving On-Demand capacity you leave idle: the reservation bills whether or not you run anything, so cancel ODCRs the moment a run finishes.

## The resilience problem: surviving a Region

A model that only exists in `us-east-1` disappears when `us-east-1` does. Disaster recovery for a serving stack is planned against two numbers:

- **RPO (Recovery Point Objective)** — how much *data/state* you can afford to lose, measured in time. For ML this is usually "how stale can the model artifact in the DR Region be?"
- **RTO (Recovery Time Objective)** — how long you can afford to be *down* while failing over.

Tighter objectives cost more. AWS frames the spectrum as four strategies, from cheapest/slowest to priciest/fastest: **Backup & Restore** (artifacts copied, nothing running in DR — hours of RTO), **Pilot Light** (data replicated, core infra dormant — must be scaled up), **Warm Standby** (a scaled-down endpoint already running, ready to scale — minutes), and **Active-Active / Multi-site** (full endpoints live in both Regions — near-zero RTO). Most ML serving stacks land on **active-passive Warm Standby** as the sensible default: cheap enough to run, fast enough to fail over.

### Replicating the model artifact (RPO)

Your model artifact and container live in S3. **S3 Cross-Region Replication (CRR)** copies objects to a bucket in the DR Region automatically and asynchronously. It requires **versioning enabled on both buckets**, and with **S3 Replication Time Control (RTC)** 99.99% of objects replicate within 15 minutes — that 15 minutes is effectively your artifact RPO.

```bash
# CRR requires versioning on source AND destination buckets first
aws s3api put-bucket-versioning --bucket my-ml-artifacts-primary \
  --versioning-configuration Status=Enabled

# Then attach a replication config (role + rules JSON) pointing at the DR bucket
aws s3api put-bucket-replication --bucket my-ml-artifacts-primary \
  --replication-configuration file://crr-config.json
```

For a relaxed RPO, scheduled AWS Backup cross-Region copies are cheaper than continuous CRR — you trade a longer recovery point for lower storage and transfer cost. Mirror your **Model Registry** approvals too: the DR Region should deploy the same approved version, so replicate the package group or re-register in the DR Region as part of the pipeline.

### Failing over the endpoint (RTO)

Deploy a SageMaker endpoint in the DR Region (dormant for Pilot Light, running small for Warm Standby), front both Regions with a custom domain, and let **Route 53 failover routing** with **health checks** send traffic to the primary while it is healthy and cut to the secondary when it fails. Clients keep calling one DNS name; the failover is invisible to them.

```bash
# Health check on the primary endpoint's inference URL
aws route53 create-health-check --caller-reference ml-primary-$(date +%s) \
  --health-check-config '{"Type":"HTTPS","FullyQualifiedDomainName":"api.mymodel.example.com","Port":443,"ResourcePath":"/ping","RequestInterval":30,"FailureThreshold":3}'
# Then create PRIMARY + SECONDARY failover records (Failover=PRIMARY references this health check)
```

**Cost framing:** active-active doubles your serving bill for near-zero RTO — justified only for revenue-critical inference. Warm Standby runs a minimal DR endpoint (e.g. a single small instance) that autoscales on failover — a fraction of the cost for minutes of RTO. Pilot Light pays only for replicated storage until disaster strikes, then eats the cold-start time to launch. Pick the cheapest tier whose RTO/RPO your business actually requires, and **test the failover on a schedule** — an untested DR plan is a hope, not a plan.

## Key takeaways

- Distinguish the two capacity walls: a **quota** error (administrative, fix with a Service Quotas increase) vs. `InsufficientInstanceCapacity` (physical, fix with a reservation or Spot).
- P/G quotas are counted in **vCPUs per Region per family** — a single large GPU instance can exceed a default quota; request increases well ahead of need.
- **ODCR** assures open-ended capacity now (billed whether used or not); **Capacity Blocks** reserve a future bounded window (up to 8 weeks out, P4d→P6e-GB200, in UltraClusters); **Spot** is 50–90% cheaper but needs checkpointing (`use_spot_instances=True` + `checkpoint_s3_uri`).
- DR is planned against **RPO** (tolerable data loss) and **RTO** (tolerable downtime); tighter = pricier, across Backup&Restore → Pilot Light → Warm Standby → Active-Active.
- Replicate artifacts with **S3 CRR** (needs versioning; RTC ~15 min RPO) and fail endpoints over with **Route 53 failover + health checks**; active-passive Warm Standby is the sensible default. GPU families and Capacity Block prices shift often — **check current docs**.

## Try it

Check your "Running On-Demand P instances" quota with `get-service-quota` and note it is in vCPUs — compute how many `p5.48xlarge` (192 vCPUs each) that allows, then file a `request-service-quota-increase` for a target you would actually train at. Run `describe-capacity-block-offerings` for a 16-instance window two weeks out and read the price back (do not purchase) to feel the real cost of reserved GPU time. Launch a small training job with `use_spot_instances=True` and a `checkpoint_s3_uri`, and confirm checkpoints land in S3 — the mechanism that makes an interruption survivable. Finally, sketch a one-page DR plan for a serving endpoint: state a target RTO and RPO, enable S3 CRR on a versioned artifact bucket, and describe which of the four DR strategies meets those numbers and why.
