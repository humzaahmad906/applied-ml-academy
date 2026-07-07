# 12 — Cloud from Basics to ML Expert (DL-Focused) — Part 1b: Cloud Foundations II (A7–A11)

*This is part 1b of the Cloud Foundations material — continuing Part A from compute onward. It assumes A1–A6 (mental model, Linux, networking, CIDR) from part 1a.*

## Part A — Universal Cloud Foundations (continued)

### A7. Compute models

Five flavors:

1. **VMs (IaaS)** — EC2, GCE, Azure VM. You manage the OS and everything above. Most control, most ops.
2. **Containers on managed clusters (CaaS)** — EKS, GKE, AKS. You manage workloads; the cloud manages the cluster control plane.
3. **Serverless containers** — Fargate (AWS), Cloud Run (GCP), Container Apps (Azure). You give a container image; the cloud runs it.
4. **Functions (FaaS)** — Lambda, Cloud Functions, Azure Functions. You give a function; the cloud runs it. Short-lived; cold starts.
5. **Managed services** — RDS, SageMaker, Vertex, etc. You give configuration; the cloud runs and operates.

For DL specifically:

- **Training** typically runs on managed K8s (EKS / GKE) for distributed jobs, or on SageMaker / Vertex Training Jobs for managed simplicity.
- **Serving** runs on managed K8s with KServe / Triton, or on Bedrock / SageMaker / Vertex Endpoints when managed-ness is preferred.
- **Embeddings / RAG** often run on Cloud Run / Fargate for the API, with EKS for self-hosted models.

<details>
<summary><strong>F500 Q:</strong> When would you reach for Cloud Run over GKE for a CV inference service? When would the answer flip?</summary>

**In-depth answer**

**Reach for Cloud Run when**:

- **Request-based, bursty traffic** with idle periods. Cloud Run
  scales to zero and you pay per request-second. A GKE cluster with
  even one running node bills 24/7.
- **CPU-only or single-GPU inference** at modest QPS (Cloud Run GPU
  supports L4 as of 2024).
- **Stateless** workload — every request independent. Cloud Run
  containers are ephemeral; no persistent disk.
- **Small team, no K8s expertise** — Cloud Run = `gcloud run deploy`.
  Zero YAML.
- **Latency budget tolerates cold start** (~1-3 seconds for a fresh
  container; less with min-instances=1, but then you're paying).

**Flip to GKE when**:

- **Multi-GPU per pod** or **multi-node training** — Cloud Run is one
  container per request, no NCCL across pods.
- **Persistent state** — model warm-up that takes 30s+ to load (large
  CV models), shared volumes, sidecars for monitoring/logging.
- **Custom networking** — VPC peering, Private Service Connect to
  on-prem, Istio service mesh.
- **High sustained QPS** (> a few hundred RPS) — at that scale GKE's
  fixed-node cost amortizes and you have more control over autoscaling
  policy (HPA on custom metrics, KEDA, etc.).
- **Tight integration with existing K8s tooling** — KServe, Kubeflow,
  Argo CD, Helm.
- **Multi-tenancy** — namespaces + RBAC give isolation; Cloud Run's
  per-project model is coarser.

**The break-even rule of thumb**: at ~$200/month of Cloud Run spend
you start looking at GKE Autopilot; at ~$1000/month GKE Standard
starts winning on cost; at multi-team you're definitely on GKE.

**SA-level twist**: Cloud Run also has Cloud Run Functions (2nd gen
Cloud Functions) — same runtime, less container. For pure stateless
JSON-in/JSON-out inference of small models you may not even need
Cloud Run; Cloud Run Functions can be cheaper.

</details>

### A8. Storage primitives

Three classes:

- **Object storage** (S3, GCS, Azure Blob) — flat namespace, HTTP API, infinite-scale, cheap. Best for datasets, model artifacts, logs, backups, anything sequential and append-only.
- **Block storage** (EBS, Persistent Disk, Managed Disk) — looks like a disk; attached to one VM (mostly); fastest random I/O. Best for databases, OS volumes.
- **File storage** (EFS, Filestore, Azure Files; also FSx Lustre / Filestore for HPC) — POSIX shared filesystem; attached to many VMs. Best for shared workspaces, training data accessible from many GPU nodes.

For DL training:

- **Object** is where training data lives at rest.
- **Block** is for the OS volume and any local scratch.
- **File / parallel file system** (FSx Lustre, GCS Filestore-HPC) is where high-throughput training accesses data — local cache + striped reads — when your dataset is too big for in-memory or local SSD.

<details>
<summary><strong>F500 Q:</strong> You're training a 50B-parameter model on 64 H100 GPUs. The dataset is 8 TB of tokenized data. Where does the data live during training (object vs file vs local), and what's the read pattern?</summary>

**In-depth answer**

**At-rest source of truth**: tokenized parquet/arrow on **S3** (or GCS).
S3 is the durable, cheap, replicable home. You never train directly
off S3 for a multi-epoch run — too much egress / latency.

**Hot path for training**: shard data into **WebDataset .tar shards**
(say, 1024 shards of ~8 GB each). At job start, each of the 64 worker
processes is assigned a disjoint subset of shard URLs and streams
them sequentially.

Two read-pattern options:

1. **Stream from S3 directly via WebDataset** — workers fetch shards
   on demand. Works if S3-to-EC2 bandwidth (same region, instance has
   ≥ 25 Gbps NIC) is enough. Each H100 needs ~3-5 GB/s of token
   throughput for compute-bound training; 8 workers per node × 8 nodes
   = 64 streams; if you spread shards across many S3 prefixes you can
   saturate inbound NIC bandwidth.

2. **Stage to FSx for Lustre** — at job start, hydrate FSx from S3
   (FSx has native S3 linkage). 8 TB at FSx's ~25 GB/s aggregate
   throughput hydrates in ~5 minutes. Workers then read from the
   shared POSIX file system. Most predictable read pattern; best
   if you re-run many epochs.

**Local NVMe role**: each node typically has 8x NVMe SSDs (on `p5.48xlarge`,
~30 TB local). Use as a **read-through cache**: first epoch streams
from FSx or S3; subsequent epochs read from local NVMe at line rate.

**The read pattern within training**:

- Sequential reads of WebDataset shards (great for object storage
  bandwidth utilization; terrible for random key/value lookups).
- Each shard fully consumed before moving on (good prefetching).
- Workers shuffle shard order per epoch (epoch-level randomness);
  inside a shard the samples are pre-shuffled at write time.
- `num_workers` ≥ 4 per GPU, `prefetch_factor=4`, `pin_memory=True`.

**SA-level twist**: don't store small files in S3. 1024 large shards
beats 5 million tiny JPEGs / .pt files in every dimension — fewer S3
GETs, better TCP window utilization, less metadata overhead. For
data-parallel training the rule is "1 shard ≥ 1 GB; total shard count
≥ 10x your number of workers."

</details>

### A9. Encryption

Two axes:

- **In transit** — TLS for everything network-facing; mTLS for service-to-service in zero-trust environments.
- **At rest** — disk encryption (default on most cloud volumes), bucket encryption (default on S3, GCS, Azure Blob in modern accounts).

Key management:

- **CMK (Customer Master Key) / KMS keys** — the cloud holds the key material; you control access via IAM.
- **Customer-managed keys** — you bring or generate the key material; cloud uses it.
- **External KMS (HSM, BYOK)** — your key never leaves your hardware.

Envelope encryption: data is encrypted with a data key, which is itself encrypted with the KMS key. Lets you rotate KMS keys without re-encrypting all data.

<details>
<summary><strong>F500 Q:</strong> A regulator asks "how is training data encrypted at rest, and who can decrypt it." Walk through the answer for S3 with KMS, naming exactly which IAM principals can read.</summary>

**In-depth answer**

**The answer template**:

> "Training data in `s3://ml-training-prod` is encrypted at rest
> using server-side encryption with a customer-managed KMS key
> (`arn:aws:kms:us-east-1:123:key/abc-123`). The key's policy grants
> `kms:Decrypt` to exactly three IAM principals: (1) the
> `EksTrainingRole` assumed by training pods via IRSA, (2) the
> `DataEngineerRole` for break-glass debugging (audited via
> CloudTrail with mandatory ticket reference), and (3) the
> `KMSAdminRole` for key rotation operations. No other identity in
> the account, including the root user, can decrypt."

**Mechanism walkthrough**:

1. S3 receives an object PUT. With SSE-KMS, S3 calls
   `kms:GenerateDataKey` against the configured CMK to get a data
   key, encrypts the object with the data key, stores the encrypted
   data key alongside the object, and forgets the plaintext.
2. On GET, S3 calls `kms:Decrypt` against the encrypted data key to
   recover the plaintext data key, decrypts the object, streams
   plaintext to the caller. S3 itself is identified as `s3.amazonaws.com`
   on the call.
3. **KMS evaluates two policies on every decrypt**:
   - The **key policy** — who's allowed to use this key.
   - The **caller's IAM policy** — does the caller have `kms:Decrypt`
     on this specific key ARN.

   Both must allow. Either denies → no decryption.

4. CloudTrail logs every `kms:Decrypt` and `s3:GetObject` with the
   identity, source IP, principal ARN, the object key, the request
   ID — 7-year retention via S3 → Glacier per your retention policy.

**The audit-grade answer adds**:

- **`aws:ViaAWSService` condition** on the key policy can scope use
  to via-S3 only.
- **Encryption context** can pin the decrypt to specific S3 paths
  (`s3:ResourceArn` condition).
- **MFA delete** on the bucket prevents accidental purging.
- **VPC endpoint policy** prevents data egress outside the VPC.
- **Block Public Access** flipped on at account level.

**SA-level twist**: customer-managed KMS keys cost $1/month + per-API
charges (~$0.03 per 10,000 ops). For an org with billions of S3
operations / month against ML data, KMS API costs become noticeable.
Mitigate with S3 Bucket Keys — S3 cache a derived key per bucket per
day, reducing KMS API calls by ~99%.

</details>

### A10. Observability — logs, metrics, traces

- **Logs** — discrete events (structured JSON ideal). Stored in CloudWatch Logs (AWS), Cloud Logging (GCP), Log Analytics (Azure), or external (Loki, Splunk, Datadog).
- **Metrics** — time-series numbers. CloudWatch Metrics, Cloud Monitoring, Azure Monitor, Prometheus.
- **Traces** — request lifecycle across services. X-Ray, Cloud Trace, App Insights, OpenTelemetry → Tempo / Jaeger.

For ML:

- Training: log per-step loss + GPU utilization + step time → metrics. Stack traces on failure → logs. Per-epoch runs → traces.
- Inference: standard request/latency/error → metrics. Per-prediction features + outputs → logs. Per-request end-to-end timing across preprocessing / inference / postprocessing → traces.

<details>
<summary><strong>F500 Q:</strong> What's the difference between OpenTelemetry and Prometheus? Why do you typically need both for an ML platform?</summary>

**In-depth answer**

**OpenTelemetry (OTel)** is a **specification and instrumentation SDK**
for emitting telemetry — logs, metrics, traces — in a vendor-neutral
format (OTLP protocol). It does *not* store data; you ship via an
OTel Collector to a backend.

**Prometheus** is a **time-series database + pull-based scraper +
PromQL query language**. It stores metrics, has its own scraping model
(`/metrics` endpoint), and is the dominant OSS metrics backend.

The relationship: OpenTelemetry can *emit* Prometheus-format metrics;
the OTel Collector can scrape Prometheus targets and forward to other
backends. They overlap on metrics but are complementary in scope.

**Why ML platforms need both**:

| Concern | Prometheus | OpenTelemetry |
|---|---|---|
| GPU utilization, mem, temp | ✓ (DCGM exporter → scrape) | (overlap) |
| Per-request latency, errors | ✓ | ✓ |
| End-to-end trace (preprocess → inference → postprocess) | ✗ | ✓ |
| Cross-service correlation (e.g., model → feature store → cache) | ✗ | ✓ |
| Logs with trace IDs | ✗ | ✓ |
| Long retention metrics | ✓ (via Mimir / Thanos) | (via export to Mimir / Loki / Tempo) |

**The 2026 standard ML observability stack**:

- **Instrumentation**: OpenTelemetry SDK in the application (Python,
  Go) emits metrics + traces + logs over OTLP.
- **Collector**: OTel Collector in DaemonSet on every node. Receives
  OTLP, scrapes Prometheus `/metrics` endpoints (vLLM, NVIDIA DCGM,
  KServe), batches, forwards.
- **Metrics backend**: AWS Managed Prometheus (AMP) or self-hosted
  Mimir.
- **Traces backend**: AWS X-Ray, Tempo, or Jaeger.
- **Logs backend**: CloudWatch Logs, Loki, or OpenSearch.
- **Visualization**: AWS Managed Grafana or self-hosted Grafana
  unifying all three.

**SA-level twist**: vendor lock-in is real. Datadog, New Relic, and
Splunk Observability accept OTLP natively, so OTel instrumentation
gives you portability. Vendor agents tied to a single product create
re-instrumentation pain on migration. F500 architects pick OTel for
optionality.

**The one-liner**: Prometheus stores numbers; OpenTelemetry produces
and ships everything. ML platforms need numbers (Prometheus) *and*
end-to-end request stories across services (OTel traces).

</details>

### A11. Cost in the cloud

Three principles:

- **Tag everything.** Without tags, you can't attribute spend.
- **Show back, charge back.** Even visibility changes behavior. Chargeback is stronger but politically harder.
- **Reserve / commit aggressively for known load.** Reserved Instances (AWS), CUDs (GCP), Savings Plans give 30–60% off list.

The GPU-specific cost moves:

- Use spot / preemptible for training (with checkpointing).
- Right-size GPU class for inference. A 7B-INT4 LLM serves fine on L4; an A100 is wasted.
- Scale-to-zero where you can tolerate cold starts.
- Quantize, batch, cache.

<details>
<summary><strong>F500 Q:</strong> Your ML org spends $400K/month on GPUs. Walk through the FinOps program you'd run in the first 90 days to cut 25% with no quality regression.</summary>

**In-depth answer**

**Days 1-14 — Diagnose**:

1. **Tag audit**. List every GPU instance / SageMaker endpoint /
   Bedrock provisioned throughput. Verify ownership tags: team,
   project, environment, cost_center. Untagged = 0% accountability;
   tag first.
2. **Utilization scrape**. DCGM metrics → Prometheus → Grafana panel
   showing top 50 GPUs by hours-billed × (1 - sm_util). The bottom
   of the leaderboard is your $100K of waste.
3. **Endpoint dashboard**. SageMaker endpoints with `min_instances ≥
   1` and < 100 requests/day. Each is paying $700+/month for nothing.
4. **Cross-region transfer audit**. CUR query for `DataTransfer-Regional`
   line items > $1K/month.
5. **Idle-notebook audit**. SageMaker Studio Notebooks; CloudWatch
   metric `NotebookInstancesAvailable` × runtime. Common surprise:
   $30K/month of idle.

By day 14 you have a leaderboard of waste with names attached.

**Days 15-45 — Quick wins (no quality risk)**:

1. **Auto-stop idle resources**. Lambda + EventBridge: SageMaker
   notebooks with no kernel activity > 4 hours → stopped. 24-hour
   cool-down before delete. Saves $15-30K/month.
2. **Scale-to-zero on dev/staging endpoints**. SageMaker Serverless
   Inference or KServe scale-to-zero. Saves $20-40K/month.
3. **Right-size existing endpoints**. Inference Recommender or manual
   benchmark on `g6e` (L40S) vs current `p4d` (A100). Most CV
   inference workloads run fine on L40S at 60% lower cost.
4. **Lifecycle S3 to Intelligent-Tiering**. ML logs + intermediate
   training artifacts. Saves storage 30-60% with no app change.
5. **VPC endpoints** for S3, ECR, DynamoDB. Eliminate NAT egress.
   Saves $5-15K/month in NAT data charges.
6. **Compress CloudWatch Logs** + lifecycle to S3 + delete. ML logs
   are voluminous; CloudWatch storage is expensive.

Cumulative by day 45: ~10-15% reduction with zero quality impact.

**Days 46-75 — Structural wins (quality validation required)**:

1. **Quantize served LLMs to INT4** (AWQ / GPTQ). Bench on gold set;
   require < 2% quality regression vs FP16. Cost reduction 50-70%
   for those workloads.
2. **Multi-LoRA serving** consolidation. If you serve 10 fine-tuned
   variants on 10 endpoints, consolidate to 1 base model with 10
   adapters via vLLM. Saves 80%+ on those endpoints.
3. **Move training to Spot** + checkpoint every 10 min. Saves 60-70%
   on training compute. Risk: pre-emption. Mitigation: checkpoint
   discipline.
4. **Reserved capacity / Savings Plans** for steady state. Audit
   actual GPU hours over 90 days; commit to 70% of baseline.
   30-40% discount on covered hours.

**Days 76-90 — Lock in + measure**:

- Per-team budgets in AWS Budgets with 50/80/100% alerts.
- Weekly cost report: top movers, anomaly detection.
- Chargeback rolled out (showback first, then real charges).
- Cost dashboard reviewed at each ML platform leadership weekly.
- Required: every new model launch has a cost estimate in its design
  doc; over-budget requires VP approval.

**Outcome**: 25-35% reduction is consistently achievable in the first
90 days at any F500 that has never done FinOps for ML. The hardest
part isn't technical — it's the political work of getting team leads
to care.

**SA-level twist**: the AWS Cost Anomaly Detection service and AWS
Compute Optimizer give you ML-driven recommendations free. Most teams
ignore them; the savings sit on the table. Don't.

</details>

---


## You can now

- Explain the control-plane/data-plane split in managed Kubernetes and reason about where ML platform cost and failure isolation actually live.
- Do CIDR math and size a VPC/subnet layout for an EKS cluster running thousands of GPU pods, including prefix delegation as the pod-density fix.
- Diagnose GPU starvation vs. GPU under-spec using `nvidia-smi dmon`, `htop`, and `iostat` before opening an SRE ticket blaming the cluster.
- Set up OIDC federation for CI/CD instead of long-lived IAM keys, and write a least-privilege IAM policy for a training job that reads S3 and writes DynamoDB.
- Run a 90-day GPU FinOps program from diagnosis through quick wins to structural savings, and justify cross-region data-placement decisions in dollars, not vibes.

## Try it

Pick a training workload you've actually run (or invent one: a 50B-parameter model on 64 GPUs). Write out, in a page: the VPC CIDR and subnet layout, the IAM role with its trust policy and its separate permission policy, where the training data lives at rest vs. in the hot read path, and the encryption and observability stack you'd put around it. Then pick the one "F500 Q" from this part you'd be least comfortable answering out loud in an interview, and go find the answer before moving to Part 2.
