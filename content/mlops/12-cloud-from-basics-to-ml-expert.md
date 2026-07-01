# 12 — Cloud from Basics to ML Expert (DL-Focused)

You're a CV / NLP engineer who knows DL but feels shaky on cloud. This chapter is the long path from "I can spin up a VM" to "I can architect AWS / GCP / Azure for production-scale DL training and inference at Fortune 500 quality."

Structure:

- **Part A** — universal cloud foundations: Linux, networking, IAM, storage, compute, observability, cost. Read once carefully.
- **Part B** — AWS in depth (primary F500 cloud). Most of the chapter.
- **Part C** — GCP and Azure focused on what differs from AWS.
- **Part D** — DL-specific cloud patterns (GPU networking, high-throughput storage, multi-node training, LLM serving).
- **Part E** — capstone scenarios and an integrated F500 question set.

After each concept block you'll see an "**F500 Q:**" line. Try to answer before reading on. Then read the next concept. By the end you should be able to walk into any F500 cloud-ML round.

This chapter assumes you've read the course overview and the DL track chapter. It cross-references the rest of the curriculum throughout.

---

## Part A — Universal Cloud Foundations

### A1. The cloud mental model

Cloud is a rental marketplace for compute, storage, network, and managed services, accessed via APIs. Three core promises:

1. **Elastic** — you can rent more or less, by the minute, without provisioning hardware.
2. **API-driven** — every resource has a JSON / Terraform shape. Idempotent reconciliation is possible.
3. **Operationally outsourced** — the cloud handles hardware, base OS, sometimes the runtime; you handle code, config, and policy.

The cloud isn't "someone else's computer" — that's a slogan. It's a control plane (the API + IAM + billing layer) over a globally-distributed data plane (the actual servers, disks, switches).

<details>
<summary><strong>F500 Q:</strong> What's the difference between the control plane and the data plane in a managed Kubernetes service?</summary>

**In-depth answer**

**Control plane** = the brain: API server, scheduler, controller manager,
etcd. It accepts your `kubectl apply`, decides where pods go, stores
desired state. In EKS / GKE / AKS, the cloud runs and patches this for
you — you pay a small fixed fee (~$72/month on EKS) and never touch a
control plane node.

**Data plane** = the muscles: the worker nodes that actually run your
pods. You provision EC2 instances (or GKE node pools, or AKS node pools)
and the kubelet on each node pulls pod specs from the control plane and
runs them. GPU nodes, autoscaling, disk space, OS patches — your problem.

**Why this matters for ML**:
1. **Cost** — control plane is cheap; data plane (GPUs) is 99% of your
   K8s bill. Optimization focus belongs in the data plane.
2. **Failure isolation** — a control plane outage (rare) means no new
   deployments, but running pods keep serving. A data plane node failure
   kills the pods on it but the control plane reschedules.
3. **Upgrades** — control plane upgrades are routine. Data plane (node
   group) upgrades require draining workloads — for GPU training jobs
   that means losing in-progress training unless you checkpoint first.

**F500 interview red flag**: candidates who say "Kubernetes" without
distinguishing these layers usually haven't operated a real cluster.

</details>

### A2. Regions, availability zones, edge

- **Region** — a geographic cluster of data centers (e.g., `us-east-1`, `eu-west-1`). Independent from other regions; data does not flow between regions unless you make it.
- **Availability Zone (AZ)** — a fault-isolated data center inside a region (e.g., `us-east-1a`, `us-east-1b`). Several per region. Network latency between AZs is typically 1–2 ms.
- **Edge / Point of Presence (PoP)** — CDN nodes; not full data centers.

Multi-AZ = HA within a region. Multi-region = disaster recovery + data-residency compliance.

<details>
<summary><strong>F500 Q:</strong> Your training cluster is in `us-east-1`. Your training data sits in `us-west-2`. What's wrong with this and what specifically does it cost you?</summary>

**In-depth answer**

Three problems, in order of pain:

1. **Latency**. Every batch your DataLoader fetches has to traverse
   ~60 ms of cross-region round-trip. For a Transformer reading 1024
   tokens/batch from sharded TFRecords, that's a 5–10× slowdown vs
   same-region read. Your GPUs idle waiting.

2. **Bandwidth tax**. S3 → EC2 in the same region is free. Cross-region
   data transfer is **$0.02/GB** outbound (us-west-2 → us-east-1). At
   training scale (terabytes per epoch, multiple epochs) you're paying
   thousands of dollars per training run *just* in egress.

3. **Throughput cap**. S3's effective throughput from a remote region is
   bounded by the inter-region backbone, not by S3's local 100+ GB/s
   capacity. Even with parallel workers you hit a ceiling.

**Fix in order of effort**:

1. **Copy the data**. `aws s3 sync` once to `us-east-1`. Pay the egress
   bill once; then training runs are free.
2. **Replicate the bucket**. S3 Cross-Region Replication. Continuous,
   automatic. Adds storage cost but no per-run egress.
3. **Move the cluster**. If the data is the source of truth and lives
   for compliance reasons in us-west-2, move your GPU cluster there.

**The interview move**: bring up cost specifically. "At 3 TB/epoch ×
10 epochs that's $600 per run, plus 5× slowdown — moving the cluster
or replicating the bucket pays for itself in week one." Numbers
distinguish senior from mid-level.

</details>

### A3. Linux essentials for cloud

You'll SSH into instances less than you used to (managed services replace many SSH workflows), but Linux fluency still gates cloud productivity.

- File system layout (`/`, `/home`, `/var/log`, `/etc`, `/usr/local`), permissions, ownership.
- Process basics (`ps`, `top`, `htop`, `kill`, `&`, `nohup`, `tmux`).
- Disk and memory (`df -h`, `du -sh`, `free -h`, `iostat`, `dmesg | tail`).
- Networking (`ip addr`, `ss -tlnp`, `netstat`, `dig`, `nslookup`, `curl -v`).
- GPU (`nvidia-smi`, `nvtop`, `nvidia-smi dmon`, `lsmod | grep nvidia`).
- Systemd (services, journals — `systemctl`, `journalctl -u <unit>`).
- SSH (key auth, agent forwarding, jump hosts, `~/.ssh/config`).

<details>
<summary><strong>F500 Q:</strong> Your training job's GPU utilization sits at 20%. List three Linux-level diagnostics that help isolate "GPU starvation" from "GPU under-spec" before you blame the cluster.</summary>

**In-depth answer**

1. **`nvidia-smi dmon -s puct -d 1`** — rolling per-second view of
   utilization (`sm`), memory (`mem`), power (`pwr`), temperature
   (`tmp`). If `sm` is 20% but `mem` is 95%, you're memory-bound, not
   starved. If `pwr` is 50% of TDP and `sm` is 20%, you're compute-
   underloaded — likely starvation.

2. **`htop` / `top` filtered to your training process** — are your
   DataLoader worker processes pegged at 100% CPU? Then they can't
   feed the GPU fast enough. Symptom: GPU goes 100% → 0% → 100% in
   sawtooth pattern. Fix: more `num_workers`, `pin_memory=True`,
   `prefetch_factor=4`, faster decoding (e.g., NVIDIA DALI for vision).

3. **`iostat -xz 1` or `iotop`** — is your disk read I/O saturated?
   Training data on slow EBS gp2 with random reads will starve any
   GPU. Symptom: high `%util` on the volume, low `wMB/s` but high
   `r_await`. Fix: gp3 with provisioned IOPS, local NVMe scratch,
   FSx Lustre, or WebDataset shards that sequentialize reads.

**Bonus**: `nsys profile` (NVIDIA Nsight Systems) overlay shows a
visual timeline of CPU/GPU/CUDA-sync; a single glance distinguishes
starvation (gaps in GPU activity) from genuine compute saturation
(GPU pinned 100% with no gaps).

**The "blame the cluster" warning**: 9 times out of 10 the cluster
is fine and the bottleneck is the DataLoader. Senior engineers
diagnose *before* opening an SRE ticket.

</details>

### A4. Networking 101 — what every cloud engineer needs

- **OSI model in spirit** — L2 (Ethernet), L3 (IP), L4 (TCP/UDP), L7 (HTTP).
- **IP addressing** — IPv4 vs IPv6, CIDR notation (`10.0.0.0/16` = 65,536 addresses).
- **DNS** — A / AAAA / CNAME / TXT records, propagation, TTLs.
- **TLS** — handshake, certs, SNI; mTLS for service-to-service.
- **Ports** — 22 SSH, 80 HTTP, 443 HTTPS, 8000 / 8080 typical app, 9090 Prometheus, 3306 MySQL, 5432 Postgres, 6379 Redis, 9092 Kafka.
- **HTTP** — methods, status codes, headers, content negotiation.
- **gRPC** — HTTP/2-based RPC; protobuf payloads; bidirectional streams.
- **Load balancing** — L4 (TCP, no inspection) vs L7 (HTTP, header-aware). Algorithms: round robin, least connections, IP hash.
- **CDN basics** — caching at the edge, cache-control headers, invalidation.

<details>
<summary><strong>F500 Q:</strong> Your CV inference service returns 502 errors intermittently under load. Walk through the stack from client → CDN → ALB → service and name the most common cause at each layer.</summary>

**In-depth answer**

A 502 is "the upstream returned something I can't parse" — typically
an upstream that died or timed out.

Layer-by-layer:

1. **Client → CDN (CloudFront)** — rarely the source of 502s. CDN
   passes through. If you see 502s only at the edge: check origin
   timeout (default 30s; CV preprocessing + inference may exceed).

2. **CDN → ALB** — TLS handshake failures or stale origin certificates
   produce 502. Less likely under load.

3. **ALB → Target (your pod)** — **this is the most common source**:
   - **Target unhealthy**: ALB pulled the pod from rotation; remaining
     pods overload, more get marked unhealthy, cascade. Check
     `HealthyHostCount` metric.
   - **Connection reset**: pod's worker (uvicorn / gunicorn) crashed
     or restarted under load (OOM-killed for memory; CUDA OOM for
     GPU memory). Check pod logs + container restart count.
   - **Target timeout**: ALB's idle timeout (default 60s) hit because
     inference + preprocessing exceeded budget. Symptom: 504 vs 502.
   - **Keep-alive mismatch**: ALB keep-alive > target keep-alive. ALB
     reuses a connection the target already closed. Fix: target
     keep-alive timeout must exceed ALB's.

4. **Pod application code** — uncaught exception, segfault in C++
   extension (libtorch, OpenCV), CUDA error, model load failure.
   Logs are the source of truth.

**Diagnostic order**:
1. Pod restart count + `kubectl describe pod`
2. ALB `TargetResponseTime` + `HTTPCode_Target_5XX` metrics
3. Pod logs for stack traces around the 502 timestamps
4. `nvidia-smi` inside pod for CUDA OOM

**War story**: most common F500 incident: 7B LLM endpoint runs at 75%
GPU memory; a single request with 8K context spikes to 95% and OOMs;
pod restarts; pods cycling cause cascading 502s. Fix: KV cache
quota, request length limits, queue depth circuit breaker.

</details>

<details>
<summary><strong>F500 Q:</strong> What's the difference between an L4 NLB and an L7 ALB in AWS? When does each fit ML serving?</summary>

**In-depth answer**

**NLB (Network Load Balancer)** — L4. Looks at TCP/UDP source/dest;
doesn't parse the payload. Forwards bytes. Properties:
- **Lowest latency** (~100 μs added)
- **Highest throughput** (millions of connections)
- **Preserves source IP** by default
- **Long-lived connections** (great for gRPC streaming, WebSocket)
- **TLS termination** optional (NLB or pass through to target)
- No HTTP-level features (no path routing, no header injection)

**ALB (Application Load Balancer)** — L7. Parses HTTP. Properties:
- **Path-based routing** (`/predict` → service A; `/embed` → service B)
- **Header / host-based routing**
- **HTTPS / TLS termination** with ACM certs
- **WebSocket and HTTP/2** native
- **WAF integration** for security rules
- **Higher latency** (~ms vs NLB)
- Per-request cost slightly higher

**For ML serving**:

| Workload | Pick |
|---|---|
| Single model HTTP API at low scale | ALB — features outweigh latency cost |
| Many models, path-routed (`/v1/classify`, `/v1/segment`) | ALB |
| gRPC inference (Triton, TF Serving) | NLB — gRPC needs HTTP/2 long connections, ALB had historical quirks |
| High-RPS, low-latency CV serving | NLB if every ms matters; ALB if path routing matters |
| LLM with streaming (SSE / WebSocket) | ALB — WS native; or NLB if you want lowest TTFT |
| Internal-only, K8s in-cluster traffic | Service of type ClusterIP — no LB at all |

**Real production**: most F500 ML stacks use ALB for the public API
gateway, then internal NLB or ClusterIP service-to-service. ALB at the
edge gives you WAF, path routing, certs. Internal NLB skips the L7
overhead where you don't need it.

</details>

### A5. CIDR math for cloud networking

You'll size subnets and design VPCs constantly. The math:

- `/24` = 256 IPs (≈ 251 usable in cloud — providers reserve 5).
- `/16` = 65,536 IPs.
- `/20` = 4,096 IPs.
- `/26` = 64 IPs (small subnet for a NAT or jump host).

For DL training pods on K8s, you frequently exhaust IPs because each pod gets one from the VPC CIDR. Size accordingly — `/16` per cluster is generous; `/24` is dangerous if you'll run > 250 pods.

<details>
<summary><strong>F500 Q:</strong> You're designing a VPC for an EKS cluster that will run up to 5,000 pods at peak. What CIDR do you choose for the cluster's VPC and why?</summary>

**In-depth answer**

**The math first**. EKS by default uses the AWS VPC CNI, which assigns
each pod a real ENI IP from the VPC. So:

- 5,000 pods → minimum 5,000 IPs in your private subnets.
- Add headroom for ~30% growth + warm pool: ~7,000 IPs.
- Reserve some IPs for nodes themselves, NAT, load balancers, future
  services: pad to 10,000.
- AWS reserves 5 IPs per subnet; account for it.
- Distribute across 3 AZs for HA, so each AZ subnet needs ~3,500 IPs.

**The pick**: VPC `10.0.0.0/16` (65,536 IPs). Carve up:

- Three private subnets, `/19` each (8,192 IPs / AZ) for pod ENIs.
- Three public subnets, `/24` each (256 IPs / AZ) for NAT + ALBs.
- Three database subnets, `/24` each, reserved.
- Spare `/20` blocks for future expansion.

```
VPC          10.0.0.0/16   (65,536)
Public  AZa  10.0.0.0/24
Public  AZb  10.0.1.0/24
Public  AZc  10.0.2.0/24
DB      AZa  10.0.3.0/24
DB      AZb  10.0.4.0/24
DB      AZc  10.0.5.0/24
Private AZa  10.0.32.0/19  (8,192)
Private AZb  10.0.64.0/19  (8,192)
Private AZc  10.0.96.0/19  (8,192)
```

**Alternatives**:
- **Custom networking with secondary CIDR** — add `100.64.0.0/16` to
  the VPC just for pods. Lets the VPC keep small "primary" subnets
  while pods use the huge secondary range. Standard pattern when
  you ran out of IPs in an existing VPC and can't expand.
- **VPC CNI prefix delegation** — `/28` blocks per ENI instead of
  individual IPs. Multiplies effective pod density 16× without a
  bigger CIDR. Use this; it's nearly free.
- **Cilium ENI mode or Calico VXLAN** — pods get cluster-internal IPs
  not from VPC. Sidesteps the problem at cost of operational
  complexity. Rare at F500.

**The interview move**: lead with the math, then the design, then
prefix delegation as the optimization. Mention RFC 1918 vs RFC 6598
(carrier-grade NAT space, 100.64.0.0/10) for the secondary range —
this signals senior AWS networking.

</details>

### A6. Authentication, authorization, identity

Three core ideas:

- **Authentication** — proving who you are (password, key, token).
- **Authorization** — what you're allowed to do (policies, RBAC).
- **Federation** — identities established in one system (Okta, Google Workspace, GitHub) recognized by another (AWS, GCP).

In cloud:

- **Users** — human identities.
- **Roles / Service Accounts** — non-human identities that workloads assume.
- **Policies** — rules attached to users / roles defining permitted actions.
- **STS (Security Token Service)** — issues short-lived credentials.
- **MFA** — required for high-privilege identities at any sane F500.

Key principle: **least privilege**. Every identity should have exactly the permissions it needs, no more. The 2026 standard: short-lived credentials (STS, OIDC), no long-lived access keys checked into code or CI.

<details>
<summary><strong>F500 Q:</strong> Explain in 60 seconds why OIDC for GitHub Actions → AWS is better than IAM access keys for CI/CD.</summary>

**In-depth answer**

**The old way (IAM access keys)**:
1. Create IAM user. Generate access key + secret.
2. Paste into GitHub Actions secret.
3. Workflow uses keys to assume permissions.

Problems:
- Keys live in GitHub forever; rotation is manual and almost never
  done.
- If a key leaks (paste-bin accident, exposed log), an attacker has
  long-lived AWS access.
- Hard to audit: CloudTrail shows the IAM user, not the workflow.
- One key = one identity; can't have per-environment scoping easily.

**The OIDC way**:
1. Configure AWS to trust GitHub's OIDC provider:
   `arn:aws:iam::123:oidc-provider/token.actions.githubusercontent.com`
2. Create IAM role with trust policy restricted to specific
   `repo:my-org/my-repo:ref:refs/heads/main`.
3. Workflow gets a JWT from GitHub, exchanges for short-lived AWS
   credentials via `sts:AssumeRoleWithWebIdentity`.

Benefits:
- **Zero long-lived secrets** in GitHub. Nothing to rotate, nothing to
  leak.
- **Short-lived credentials** (default 1 hour) — minimal blast radius.
- **Workflow-scoped trust** — the role can only be assumed from the
  specific repo + branch + workflow. A compromised forked PR can't
  assume your prod role.
- **Audit-grade trail** — CloudTrail records the GitHub workflow ID
  in `sts:AssumeRoleWithWebIdentity` events.

**The 60-second answer**: OIDC trades long-lived keys for short-lived,
context-scoped tokens. Nothing leaks because nothing persists. Trust
policies enforce "only this repo on this branch can assume this role."
The audit trail is real. This is the 2026 default at any well-run
F500.

</details>

<details>
<summary><strong>F500 Q:</strong> Your data scientist needs to run training jobs on EC2 + write models to S3 + read features from DynamoDB. Sketch the IAM role policy at a high level. What are the failure modes if you're too permissive vs too restrictive?</summary>

**In-depth answer**

**Minimum-viable IAM role policy** (least privilege):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "WriteModelsToSpecificBucket",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:AbortMultipartUpload"],
      "Resource": "arn:aws:s3:::ml-models-prod/models/${aws:PrincipalTag/Team}/*"
    },
    {
      "Sid": "ListBucketRestricted",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::ml-models-prod",
      "Condition": {"StringLike": {"s3:prefix": ["models/${aws:PrincipalTag/Team}/*"]}}
    },
    {
      "Sid": "ReadFeaturesFromSpecificTable",
      "Effect": "Allow",
      "Action": ["dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:Query"],
      "Resource": "arn:aws:dynamodb:us-east-1:123456789012:table/features-online"
    }
  ]
}
```

The role is **assumed by the EC2 instance profile** (or IRSA on EKS),
not by the human. The human's user identity has permission to *launch*
training jobs that assume this role.

**Too permissive — failure modes**:
1. `s3:*` on `*` — a compromised training pod can list and exfiltrate
   every bucket in the account. Compliance violation; potential breach
   on day one.
2. Wildcard `dynamodb:*` — bad code can drop tables. Production data
   loss.
3. No condition restrictions — a Team-A engineer's pod can write to
   Team-B's S3 prefix; cross-tenant contamination.
4. **Long-term consequence**: when an auditor asks "what can this role
   do?" you can't enumerate it. You fail SR 11-7 / SOC 2.

**Too restrictive — failure modes**:
1. Missing `s3:AbortMultipartUpload` — orphaned multipart uploads that
   you can't clean up. S3 keeps charging you for them.
2. Missing `dynamodb:DescribeTable` — boto3 calls fail with cryptic
   errors at job start.
3. Missing KMS decrypt on the bucket's encryption key — `AccessDenied`
   that looks like an S3 problem but isn't.
4. Missing `s3:GetObject` on a config bucket — your training job
   crashes on startup; engineers waste a day debugging.

**The senior move**: use a permissions boundary on the role so it
*can't* be made more permissive even if someone tries. Pair with
CloudTrail to detect unusual API calls. Test the policy with the
IAM Policy Simulator before deploying.

</details>

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

## Part B — AWS in Depth

AWS is the most common F500 cloud. This section is intentionally long.

### B1. AWS account topology

Real F500 setups have many accounts, not one big account.

- **Organization** — the root container.
- **Organizational Units (OUs)** — logical groups (Production, Non-Prod, Sandbox, Security, ...).
- **Accounts** — isolation boundary. Separate billing, IAM, blast radius.
- **AWS Control Tower** — managed multi-account setup with guardrails.
- **Service Control Policies (SCPs)** — restrict what *any* identity in an OU/account can do (e.g., "deny launching GPU instances outside this region").

A typical ML org structure:

```
Organization
├── Security OU
│   ├── log-archive account
│   └── audit account
├── Shared Services OU
│   └── shared-tooling account (artifact registry, CI runners)
├── ML Platform OU
│   ├── ml-prod account
│   ├── ml-staging account
│   └── ml-sandbox account
└── Workload OUs (one per business unit)
```

<details>
<summary><strong>F500 Q:</strong> Why have a separate "ml-sandbox" account instead of giving data scientists a folder in the prod account?</summary>

**In-depth answer**

**Five reasons, in order of importance**:

1. **Blast radius**. A misconfigured policy, a leaked credential, an
   accidentally-public S3 bucket — contained to the sandbox account.
   Prod data, prod IAM, prod KMS keys, prod logs never co-mingle.
   This is the SR 11-7 + SOC 2 + ISO 27001 baseline expectation.

2. **Cost attribution and limits**. Each account has its own bill,
   Budget alerts, Service Quotas. A data scientist can't accidentally
   spin up 32 H100 instances in prod. Service Control Policies at
   the OU level can hard-cap instance types per account.

3. **IAM clarity**. In a single-account model, you fight policy
   wildcards forever. With separate accounts, the *account ID* is
   the boundary. Cross-account access is explicit (AssumeRole, S3
   bucket policies); intra-account access is just "you're in here."

4. **Compliance evidence**. Auditors want to see "production data
   never crossed into experimental workloads." With one account,
   you prove this with tag filters and prayers. With two accounts,
   you prove it with the absence of cross-account roles.

5. **Operational simplicity**. Sandbox can be aggressive: SCPs
   forcing instance stop after 12 hours, mandatory tagging, auto-
   deletion of unused buckets. Prod stays stable. Experimentation
   doesn't fight the safety rails meant for prod.

**The well-run topology**:

```
Organization Root
├── Security OU (log-archive, audit accounts)
├── Production OU (workload accounts, ml-prod)
├── Non-Prod OU (ml-staging, ml-dev)
└── Sandbox OU (ml-sandbox, individual-dev-accounts)
```

SCPs on Sandbox OU: deny KMS access to prod keys, deny IAM creation
of admins, mandate session tags, restrict to non-prod regions for
test, max instance types capped at $10/hour.

**SA-level twist**: AWS Control Tower automates this. Account Factory
(via Service Catalog) lets you provision a new sandbox account in
~30 minutes via API call, fully baseline-configured (log forwarding,
GuardDuty enabled, Config rules, tag policies). At F500 scale, the
"one account per engineer for prototyping" pattern works because
Control Tower makes it operationally cheap.

</details>

### B2. IAM deep — the most important AWS concept

Five core constructs:

- **Users** — long-lived human identities. Increasingly rare in well-run orgs; federation replaces them.
- **Groups** — sets of users sharing policies.
- **Roles** — non-human identities; assumed temporarily by users, services, or workloads.
- **Policies** — JSON documents attached to identities (identity-based) or resources (resource-based).
- **STS** — issues short-lived credentials (`AssumeRole`, `GetSessionToken`).

A policy looks like:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::training-data-bucket/*",
      "Condition": {
        "StringEquals": {
          "aws:PrincipalTag/Team": "ml-platform"
        }
      }
    }
  ]
}
```

Key patterns:

- **IAM Identity Center (formerly SSO)** — federate from Okta / Google / AD; users get SSO + role assumption.
- **IAM Roles for EC2 / EKS pods (IRSA) / Lambda / SageMaker** — workloads assume roles via instance metadata; no key in code.
- **OIDC federation for GitHub Actions** — GitHub provides a short-lived token; AWS trusts it via an OIDC provider; assume a role; do the thing.
- **Resource-based policies on S3 buckets / KMS keys** — useful for cross-account access.

The most-failed F500 IAM topics:

- **Trust policy vs permission policy** — trust policy says who can assume me; permission policy says what I can do.
- **Permissions boundaries** — a ceiling on what a role can do regardless of policies attached.
- **Service-linked roles** — roles owned by AWS services that you can't fully customize.

<details>
<summary><strong>F500 Q:</strong> Explain the difference between an IAM role's trust policy and its permission policy. Walk through what happens when GitHub Actions assumes that role via OIDC.</summary>

**In-depth answer**

**Trust policy** — answers "who can become me?" Attached to the role.
Defines the set of principals (users, roles, services, federated
identities) that can call `sts:AssumeRole*` to get temporary credentials
for this role.

**Permission policy** — answers "what can I do once I'm this role?"
A standard IAM policy attached to the role defining allowed/denied
actions on resources.

These are two separate JSON documents. Confusing them is the most
common F500 IAM mistake.

**The GitHub Actions → AWS via OIDC flow**:

1. **One-time setup**: in AWS, create an Identity Provider of type
   "OpenID Connect" pointing at
   `https://token.actions.githubusercontent.com`. AWS now trusts
   tokens signed by GitHub's OIDC issuer.

2. **Create the IAM role** with a trust policy like:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Principal": {
         "Federated": "arn:aws:iam::123:oidc-provider/token.actions.githubusercontent.com"
       },
       "Action": "sts:AssumeRoleWithWebIdentity",
       "Condition": {
         "StringEquals": {
           "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
         },
         "StringLike": {
           "token.actions.githubusercontent.com:sub": "repo:my-org/my-repo:ref:refs/heads/main"
         }
       }
     }]
   }
   ```
   Note: the `sub` claim is the tight binding. Without it any GitHub
   workflow in any repo could assume your role.

3. **Attach a permission policy** (separately) like:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["s3:PutObject"],
       "Resource": "arn:aws:s3:::ml-artifacts/*"
     }]
   }
   ```

4. **In the GitHub Actions workflow**:
   ```yaml
   permissions:
     id-token: write
     contents: read
   steps:
     - uses: aws-actions/configure-aws-credentials@v4
       with:
         role-to-assume: arn:aws:iam::123:role/gha-ml-publish
         aws-region: us-east-1
   ```

5. **At runtime**:
   - GitHub Actions generates an OIDC JWT containing claims:
     `iss=https://token.actions.githubusercontent.com`, `aud=sts.amazonaws.com`,
     `sub=repo:my-org/my-repo:ref:refs/heads/main`, plus repo, run_id,
     actor.
   - `configure-aws-credentials` calls `sts:AssumeRoleWithWebIdentity`
     passing the JWT.
   - STS validates: signature (against GitHub's JWKS), `aud`, `iss`,
     `exp`. Then evaluates the role's trust policy — does `sub` match
     the StringLike pattern?
   - If yes: STS issues short-lived AWS credentials (access key,
     secret key, session token; default 1-hour TTL).
   - The workflow uses these credentials. CloudTrail logs
     `AssumeRoleWithWebIdentity` with the GitHub workflow info as
     `requestParameters`.

**The win**: zero long-lived AWS secrets in GitHub. Trust policy
restricts to specific repo + branch + (optionally) environment +
workflow file. Credentials auto-expire.

**SA-level twist**: pin the `aud` claim to `sts.amazonaws.com` and
the `sub` to *exact* values, not loose globs. Common mistakes: using
`StringEquals` instead of `StringLike` and missing a pull-request
event scope (PRs have `sub=repo:org/repo:pull_request`, not `ref:`).
Add explicit conditions for `repository_owner` and `environment` for
defense in depth.

</details>

<details>
<summary><strong>F500 Q:</strong> Your ML pipeline pod in EKS needs to read from S3 in account A and write to DynamoDB in account B. Sketch the IRSA + cross-account assume-role setup.</summary>

**In-depth answer**

**Setup overview**:

```
EKS cluster (account C)        Account A (data)         Account B (online store)
─────────────────────          ────────────────         ────────────────────────
Pod
└── SA: ml-pipeline ──IRSA──► EksPipelineRole (in C)
                                      │
                                      ├─AssumeRole──► CrossAccountReadRole (A)
                                      │                    └── policy: s3:Get on bucket
                                      │
                                      └─AssumeRole──► CrossAccountWriteRole (B)
                                                          └── policy: dynamodb:Put on table
```

**Step by step**:

1. **In account C (cluster account)** — create `EksPipelineRole` with
   IRSA trust to the K8s ServiceAccount `ml-pipeline` in namespace
   `ml-prod`. Its permission policy grants `sts:AssumeRole` on the
   two cross-account roles:
   ```json
   {
     "Effect": "Allow",
     "Action": "sts:AssumeRole",
     "Resource": [
       "arn:aws:iam::AAAA:role/CrossAccountReadRole",
       "arn:aws:iam::BBBB:role/CrossAccountWriteRole"
     ]
   }
   ```

2. **In account A** — create `CrossAccountReadRole` with:
   - Trust policy: trusts `arn:aws:iam::CCCC:role/EksPipelineRole`
     as principal.
   - Permission policy: `s3:GetObject`, `s3:ListBucket` on the
     specific bucket only.

3. **In account B** — create `CrossAccountWriteRole` with:
   - Trust policy: trusts `arn:aws:iam::CCCC:role/EksPipelineRole`.
   - Permission policy: `dynamodb:PutItem`, `dynamodb:BatchWriteItem`
     on the specific table.

4. **Annotate the K8s ServiceAccount**:
   ```yaml
   metadata:
     annotations:
       eks.amazonaws.com/role-arn: arn:aws:iam::CCCC:role/EksPipelineRole
   ```

5. **Pod code** (boto3 example):
   ```python
   import boto3
   # Default session uses IRSA-injected credentials → EksPipelineRole
   sts = boto3.client("sts")
   # Read from account A
   creds_a = sts.assume_role(
       RoleArn="arn:aws:iam::AAAA:role/CrossAccountReadRole",
       RoleSessionName="read-data",
       DurationSeconds=3600,
   )["Credentials"]
   s3 = boto3.client(
       "s3",
       aws_access_key_id=creds_a["AccessKeyId"],
       aws_secret_access_key=creds_a["SecretAccessKey"],
       aws_session_token=creds_a["SessionToken"],
   )
   data = s3.get_object(Bucket="data-bucket", Key="...")["Body"].read()

   # Write to account B (separate assume)
   creds_b = sts.assume_role(...)
   ddb = boto3.client("dynamodb", ...)
   ddb.put_item(...)
   ```

**Key details**:

- **External ID** in cross-account trust policies (for third-party
  cases) — not strictly required for internal cross-account, but
  good hygiene.
- **MFA condition** on the assume role can be enforced for sensitive
  cross-account access.
- **Session tags** can be passed through; useful for column-level
  filtering in Lake Formation.
- **`DurationSeconds`** caps at 12 hours (or whatever
  `MaxSessionDuration` is set to on the target role).

**SA-level twist**: at F500 scale, this gets tedious. Consider AWS
Resource Access Manager (RAM) for sharing specific resources directly
(e.g., Glue catalog tables, Lake Formation grants) without the
explicit assume-role dance. For data-plane sharing (S3 reads at
scale) S3 Access Points + IAM cross-account is more performant than
serial assume-role calls per request.

</details>

<details>
<summary><strong>F500 Q:</strong> A junior engineer pasted an AWS access key into a public repo. Walk through the incident response, including how a well-designed account topology limits blast radius.</summary>

**In-depth answer**

**Minute 0-5 — Contain**:

1. **Deactivate the key immediately**. `aws iam update-access-key
   --status Inactive --access-key-id <ID>`. Don't delete yet — you
   need the audit trail.
2. **Revoke active sessions** for the user: `aws iam attach-user-policy
   --policy-arn arn:aws:iam::aws:policy/AWSDenyAll --user-name X`,
   then `aws iam delete-access-key`.
3. **Force GitHub secret push protection** — verify the leak is
   actually public; check repo's commit history for the key string.

**Minute 5-30 — Assess**:

4. **CloudTrail query** — search for `accessKeyId = <ID>` over the
   last 7-30 days:
   ```sql
   SELECT eventTime, eventName, sourceIPAddress, awsRegion, userAgent
   FROM cloudtrail_logs
   WHERE accessKeyId = 'AKIA...'
   ORDER BY eventTime DESC
   ```
   Look for: unusual IPs (esp. Tor exit nodes), unusual regions
   (your team uses us-east-1, calls from ap-southeast-2 are
   suspicious), unusual API calls (`ec2:RunInstances` for crypto
   mining is canonical).
5. **GuardDuty findings** — should already be alerting on credential
   anomalies. Check the console.

**Minute 30-2hr — Eradicate**:

6. **Rotate any secondary credentials** in the same account (S3
   bucket policies, KMS key access, downstream system credentials)
   if the leaked key had access.
7. **Scrub Git history** — `git filter-repo` or BFG. The key is
   already known by GitHub's secret scanning anyway, but reduce
   future discoverability.
8. **AWS Support case** if you suspect actual compromise — AWS will
   help review.

**Minute 2hr+ — Recover and prevent**:

9. **Restore the user** with a fresh key (or, better, migrate them
   to AWS IAM Identity Center / SSO so they never have a long-lived
   key again).
10. **Update GitHub secret scanning** — `repository_security_policy`
    + push protection.
11. **Run the postmortem**. What controls failed? Why didn't push
    protection block? Why did the user have a long-lived key at all?

**How account topology limits blast radius**:

If the leaked key belonged to a user in the **prod account**, blast
radius = prod. Catastrophic.

In the well-designed topology:

- **The user has no long-lived key**. They authenticate via SSO →
  AssumeRole → 1-hour creds. There's no key to leak.
- **If they're in a sandbox account**: blast radius = sandbox.
  SCPs cap instance types ($/hour limit), block cross-account
  access to prod, restrict to non-prod regions. Worst case is a
  $5K crypto-mining bill, contained.
- **If they're in shared services**: trust policies on prod roles
  require MFA via condition (`aws:MultiFactorAuthPresent = true`).
  The leaked key without MFA can't AssumeRole into prod.
- **GuardDuty + Security Hub** aggregate findings across accounts;
  one alert pipeline catches anomalies wherever they originate.

**The 2026 prevention stack**:

- **IAM Identity Center** — no long-lived keys for humans, ever.
- **OIDC for CI** — no long-lived keys for automation.
- **GitHub push protection + secret scanning** — blocks at the push.
- **AWS Access Analyzer** — finds unused permissions to prune.
- **AWS IAM Roles Anywhere** — for on-prem / non-EKS workloads that
  used to need long-lived keys.

**SA-level twist**: the architect's question after the incident is
"why did a long-lived access key exist at all?" The right answer
isn't better key rotation — it's eliminating the failure mode by
moving everyone to short-lived federated credentials. That's a
multi-quarter program at F500 scale.

</details>

### B3. VPC deep — networking on AWS

VPC = Virtual Private Cloud, your isolated network.

Components:

- **VPC** — a CIDR (e.g., `10.0.0.0/16`).
- **Subnets** — sub-CIDRs, each tied to one AZ.
- **Internet Gateway (IGW)** — the door for VPC ↔ public internet (egress + ingress).
- **NAT Gateway** — lets private subnets *initiate* outbound traffic without being publicly addressable. Costs real money per GB.
- **Route tables** — per subnet; determine where traffic goes.
- **Security groups** — stateful firewall at instance/ENI level. Default deny inbound.
- **NACLs (Network ACLs)** — stateless firewall at subnet level. Less commonly used.
- **VPC endpoints** — private network paths to AWS services (S3, DynamoDB, ECR, etc.) that bypass the public internet.
- **VPC peering / Transit Gateway** — connect VPCs.

A canonical ML VPC layout per region:

```
VPC 10.0.0.0/16
├── Public subnets   (10.0.0.0/22 across 3 AZs)
│       └── ALBs, NAT gateways, bastion (if any)
├── Private subnets  (10.0.16.0/20 across 3 AZs)
│       └── EKS nodes, EC2 training instances, SageMaker, etc.
└── Database subnets (10.0.32.0/24 across 3 AZs)
        └── RDS, ElastiCache
```

Three NICE-to-knows:

- **Egress-only Internet Gateway** for IPv6.
- **VPC Flow Logs** — capture all traffic to/from ENIs.
- **PrivateLink** — expose your service privately to other VPCs / accounts.

<details>
<summary><strong>F500 Q:</strong> Why is a VPC endpoint for S3 not just a security improvement but a cost optimization?</summary>

**In-depth answer**

**Security side**:

- Without a VPC endpoint, S3 traffic from a private subnet has to
  hairpin through a NAT gateway → IGW → public internet → S3 → and
  back. The traffic uses public IPs even if it never leaves AWS's
  backbone.
- With a Gateway VPC endpoint, the traffic stays inside AWS, routed
  via the endpoint's route entry, never touching the public internet.
- VPC endpoint policies + bucket policies + `aws:SourceVpce`
  conditions let you assert: "this bucket is only readable from
  this VPC's endpoint." Defense in depth.

**Cost side — where the real argument lives**:

- **NAT gateway pricing is brutal**: $0.045/hour ($32/month) per NAT
  per AZ — that's the fixed cost. The killer is **$0.045 per GB of
  data processed**.
- For ML workloads, that S3 ↔ EC2 traffic is *massive*. A training
  job reading 5 TB of images via NAT: 5,000 GB × $0.045 = **$225 per
  run**. Multiply across many experiments and the NAT bill exceeds
  the GPU bill.
- An S3 Gateway VPC endpoint costs **$0**. No hourly charge, no per-
  GB charge. It's literally free.

**The economics**:

| Workload | Without endpoint | With endpoint |
|---|---|---|
| Pull 5 TB train data once | $225 NAT egress | $0 |
| 100 training runs/month at 5 TB each | $22,500 NAT | $0 |
| Plus NAT fixed cost (3 AZ × $32) | $96/month | $96/month (still need NAT for other traffic) |

A typical F500 ML team saves $10-30K/month going from "everything
through NAT" to "VPC endpoint for S3" — and that's *just* the data
transfer charge.

**Other endpoints worth doing** (also Gateway → free):

- **DynamoDB Gateway endpoint** — free, same pattern. Online feature
  store reads stop charging.

**Interface endpoints** (paid — different story):

- **ECR, CloudWatch Logs, STS, SSM, Bedrock, SageMaker, etc.** —
  Interface endpoints are AWS PrivateLink and cost ~$0.01/hour per
  endpoint per AZ + ~$0.01/GB. Still cheaper than NAT for high
  traffic, but verify the math per service.

**SA-level twist**: at F500 scale, NAT data-transfer charges are the
single most common "hidden cost" line item that surprises VPs at the
quarterly review. The architect who proactively audits and adds VPC
endpoints saves the company $100K+/year in a multi-VPC environment,
often pays for their salary.

</details>

<details>
<summary><strong>F500 Q:</strong> Your GPU training instances in a private subnet need to pull container images from ECR. Walk through the connectivity options (NAT vs VPC endpoint vs PrivateLink) and trade-offs.</summary>

**In-depth answer**

**The three options**:

1. **NAT Gateway path** — pod requests image → kube-proxy routes to
   ECR's public endpoint → traffic egresses through NAT → IGW → public
   internet → ECR. Works out of the box. Costly: NAT $0.045/GB; a
   2 GB container image pulled 100x = $9 just for that one image's
   pulls. Across an org, easily $5-15K/month wasted.

2. **VPC Interface endpoint for ECR** — AWS PrivateLink. You enable
   two endpoints: `com.amazonaws.<region>.ecr.api` (control plane)
   and `com.amazonaws.<region>.ecr.dkr` (data plane). Plus the **S3
   gateway endpoint** (ECR stores blob layers in S3 under the hood).
   Then containerd / Docker daemon resolves ECR through these
   endpoints, no NAT involved.
   - Cost: ~$0.01/hour per endpoint per AZ × 3 AZs × 2 ECR endpoints
     = ~$43/month fixed + $0.01/GB data processed.
   - Worth it once you push more than ~1 TB/month through ECR.

3. **VPC Peering / Transit Gateway to a shared services VPC that
   owns ECR endpoints** — when many VPCs need ECR, centralize the
   endpoints in a shared VPC and peer/TGW. Saves replicating per
   VPC. F500 standard.

**Three gotchas**:

- **Forgetting the S3 gateway endpoint**. ECR's image layers come
  from S3. Without the S3 endpoint, layers still hairpin through
  NAT even with ECR endpoints.
- **Endpoint private DNS** — must enable. Without it, the ECR
  hostnames resolve to public IPs and your traffic exits the VPC
  even though the endpoint exists.
- **Cross-region pulls** — VPC endpoints are region-local. Pulling
  from `us-east-1` ECR in `us-west-2` requires either replication
  (ECR replication is one-way, async, low-cost) or paying egress.

**SA-level twist**: ECR Pull-Through Cache lets your ECR pull from
Docker Hub / Quay / GHCR through your AWS-resident cache. Combined
with VPC endpoints, even your third-party images don't hairpin
through the public internet. Audit-friendly + cost-friendly.

**The recommendation order for a new platform**:

1. S3 gateway endpoint (free, always)
2. DynamoDB gateway endpoint (free, if you use DDB)
3. ECR interface endpoints + S3 (paid but cheap; pays for itself
   at moderate traffic)
4. Other interface endpoints (STS, CloudWatch Logs, Bedrock,
   SageMaker) as workload demands
5. Pull-Through Cache when third-party registries enter the picture

</details>

<details>
<summary><strong>F500 Q:</strong> A pod in EKS can't reach an external API. Walk the diagnostic from inside out: security group, NACL, route table, NAT, IGW, DNS.</summary>

**In-depth answer**

**The diagnostic order — inside out**:

1. **Inside the pod**:
   ```sh
   kubectl exec -it pod -- curl -v https://external.api/v1
   ```
   Read the error. `Connection timed out` = network blocked. `Could
   not resolve host` = DNS broken. `Connection refused` = reachable
   but service not listening. Different errors = different problems.

2. **DNS resolution**:
   ```sh
   kubectl exec -it pod -- nslookup external.api
   kubectl exec -it pod -- cat /etc/resolv.conf
   ```
   If nslookup fails, suspect CoreDNS pod problems (`kubectl get pods
   -n kube-system | grep coredns`), VPC DHCP option set, or a
   security group blocking UDP 53 to the VPC resolver
   (`169.254.169.253`).

3. **Pod-to-service security group**. Pod's ENI security group must
   allow egress on the target port (typically 443). EKS by default
   uses the cluster security group + node group SGs; verify with:
   ```sh
   aws ec2 describe-security-groups --group-ids <sg-of-eni>
   ```
   The egress rules. Look for `allow tcp 443 to 0.0.0.0/0` (or to
   the API's CIDR if more restrictive).

4. **Subnet's NACL**. NACLs are stateless — must allow both inbound
   *and* outbound for the response port range. Typical mistake:
   blocking ephemeral ports (1024-65535) inbound, so SYN-ACK from
   the API can't return.

5. **Route table**. The pod's subnet must have a route to the
   destination. For external API → `0.0.0.0/0` should point to a
   NAT (if private subnet) or IGW (if public subnet). For a peered
   VPC API → the peering connection. Common bug: route table missing
   `0.0.0.0/0` entirely.

6. **NAT gateway / IGW health**.
   - NAT in the route table, but NAT exists in the right AZ?
   - NAT has an EIP?
   - NAT not in failed state (CloudWatch metric
     `IdleTimeoutCount` spiking suggests overload)?

7. **Source IP check**. CloudWatch / VPC Flow Logs:
   ```
   SELECT * FROM flow_logs WHERE srcaddr = '<pod-ip>'
   AND dstaddr = '<api-ip>'
   ```
   Action = `REJECT` tells you which SG / NACL dropped it. Most
   useful diagnostic.

8. **External — DNS** of the API itself. From outside AWS, does
   `external.api` resolve and respond? If no, the problem isn't you.

**The mnemonic order** (memorize for interviews):

> Pod → SG out → NACL out → Route table → NAT/IGW → External →
> NACL in (ephemeral) → SG in (stateful, auto-allowed by SG, must
> open by NACL)

**SA-level twist**: at F500 scale, the most common diagnosis is
"egress went through NAT but the destination is on AWS's public
endpoint — should be on VPC endpoint." Or: cluster pod is using a
different ENI than expected (with multi-ENI VPC CNI configurations,
ENIs have different SGs). Don't forget about `IRSA` token endpoint
failures masquerading as network problems.

The diagnostic that finds 80% of cases in 30 seconds: VPC Flow Logs
with REJECT filter on the pod IP.

</details>

### B4. EC2 instance types for ML

The ones you care about:

**General compute (CPU-only training / inference for tabular and small NN):**

- `c7i.*` — Intel Xeon, compute-optimized
- `m7i.*` — balanced
- `r7i.*` — memory-optimized (big-vocab tokenization workloads)
- Graviton (`c7g`, `m7g`, `r7g`) — ARM, cheaper, fewer ML frameworks fully supported

**GPU for training:**

- `p5.*` — H100 80 GB (8 GPUs per node, 640 GB total)
- `p4d.*` / `p4de.*` — A100 40/80 GB
- `p3.*` — V100 (legacy)
- `g5.*` — A10G (great for inference, OK for small fine-tuning)
- `g6.*` — L4 (newer, very efficient for inference)
- `g6e.*` — L40S (great middle-ground for inference + small training)
- `dl1.*` — Habana Gaudi (lower-cost training alternative; smaller ecosystem)

**AWS custom silicon:**

- `trn1.*` / `trn2.*` — Trainium for training
- `inf1.*` / `inf2.*` — Inferentia for inference
- Lower cost than NVIDIA equivalents; software (Neuron SDK) has fewer model types supported

**High-performance interconnect:**

- `p5.48xlarge` + EFA (Elastic Fabric Adapter) gives you NCCL over RDMA. Essential for multi-node training.
- Cluster placement groups put instances physically close for low-latency between them.

Pricing models:

- **On-demand** — list price.
- **Spot** — up to 90% off; can be reclaimed with 2-minute warning. Great for training with checkpointing.
- **Reserved Instances** — 1 or 3 year commit; up to 72% off.
- **Savings Plans** — flexible compute commitment; usually 30–66% off.
- **Capacity Reservations** — guarantee availability (separate from price).

<details>
<summary><strong>F500 Q:</strong> You need to train a 13B-parameter model. Pick an instance type + count, justify, and explain what you'd put in your spot/RI/SP mix.</summary>

**In-depth answer**

**The memory math first**:

13B params × 16 bytes/param (mixed-precision Adam: 2P + 4P + 2P + 8P
for params, master, grads, optimizer) = ~208 GB before activations.
Add ~50 GB for activations at reasonable batch size.

So per replica we need ~260 GB GPU memory. One H100 = 80 GB. We can't
fit one full replica on one GPU; FSDP shards it. With FSDP/ZeRO-3
across 4 H100s (320 GB total), we fit comfortably.

**Recommended setup**: **`p5.48xlarge`** (8× H100-80GB on one node).
One node = 640 GB GPU memory aggregate. With FSDP `FULL_SHARD` plus
activation checkpointing, a 13B fine-tune fits comfortably with room
for batch size 4-8 per GPU at 4K-8K context.

**For full from-scratch training** (not just fine-tune):

- 4 × `p5.48xlarge` nodes (= 32 × H100 = 2.5 TB GPU memory)
- 3D parallelism: TP=8 within node (NVLink), PP=4 across nodes, no
  DP. Or pure FSDP across all 32 if your framework prefers.
- EFA-enabled placement group, NCCL over RDMA.
- Estimated cost: $98/hour × 4 nodes × ~2 weeks training time =
  ~$110K total. Real number.

**The spot / RI / Savings Plan mix**:

For training specifically (interruption-tolerant with checkpointing):

- **70-80% Spot for training**. Discount up to ~70%. Checkpoint every
  10-30 min so a Spot interruption costs ≤ 30 min of progress. The
  AWS H100 spot market is occasionally thin; have us-east-1 + us-west-2
  + us-east-2 as fallback regions.
- **10-20% Savings Plan (Compute SP, 1-year, no upfront)** — covers
  baseline always-on capacity (development clusters, debugging
  instances, persistent experiment runners). 30% discount.
- **0-10% On-Demand** — for time-critical runs you can't pre-empt.

For *inference* (different math):

- **70% Savings Plan / Reserved Instances** — endpoints run 24/7;
  predictable, commit to capacity. 1-year SP = 30%, 3-year RI no-
  upfront = 50%+.
- **30% On-Demand** — overflow, burst, regional failover capacity.
- **Spot is rarely good for inference** — interruption mid-request
  is unacceptable for user-facing serving.

**Capacity guarantees**:

- **EC2 Capacity Reservations** — guarantee instance availability
  (NOT a discount; pay on-demand). Use when you have a deadline and
  the spot market for H100 is shaky.
- **ML Capacity Blocks for Reservations** — newer AWS offering, lets
  you reserve GPU capacity in 1-day to 14-day blocks, weeks in
  advance. Built for "I need 64 H100s for 5 days starting next
  Tuesday."

**SA-level twist**: at F500 scale negotiate Enterprise Discount
Program (EDP) or Private Pricing Agreement (PPA). Bulk commits over
$1M/year unlock 5-20% additional discounts on top of SP/RI. The
architect's job is to forecast credibly enough to commit at the
right tier.

**Senior signal**: mention the right *anti-pattern* — using on-demand
H100s 24/7 for "always-available" notebooks. That's $70K/month per
node sitting idle 90% of the time. Move to a SageMaker Studio
ephemeral instance or per-request Lambda dispatcher pattern.

</details>

<details>
<summary><strong>F500 Q:</strong> You're serving an INT4-quantized 7B LLM. Pick an instance type for 200 RPS at P95 < 500 ms TTFT. Why not p5? Why not g5?</summary>

**In-depth answer**

**The math**:

INT4-quantized 7B: ~4 GB model weights. KV cache at 4096 context per
request in FP16: ~2 GB. At, say, 50 concurrent in-flight requests
(typical with continuous batching at 200 RPS, 4s avg generation),
that's 100 GB of KV cache.

So you need ~100+ GB GPU memory, throughput for ~200 RPS at sub-500ms
TTFT.

**Why not `p5.48xlarge`** (8× H100, 640 GB total):
- Massively overspecced. 7B INT4 doesn't need 80GB cards.
- $98/hr on-demand. 200 RPS doesn't justify this.
- You'd be using ~5% of GPU memory and a fraction of compute.

**Why not `g5.xlarge`** (1× A10G, 24 GB):
- 24 GB is borderline for 7B INT4 + KV cache at 50 concurrent. You
  hit OOM with longer contexts or higher concurrency.
- A10G is older silicon; FP8 not supported.
- Throughput lower than newer L4 / L40S at similar cost.

**Pick**: **`g6e.xlarge`** or **`g6e.2xlarge`** (1× L40S, 48 GB) — or
**`g6.12xlarge`** (4× L4, 96 GB) for higher concurrency at lower
$/RPS. Why:
- L40S is built for inference; FP8 supported; ~$1.86/hr on-demand.
- 48 GB fits 7B INT4 plus generous KV cache headroom.
- 200 RPS @ vLLM continuous batching = single-instance achievable
  for many 7B INT4 workloads; you might even need only one replica
  plus a hot standby for HA.

**Multi-replica**: 2× `g6e.xlarge` behind an NLB gives HA and lets
you do rolling deploys. ~$2,700/month total. Cheap for 200 RPS LLM
serving.

**Bonus consideration**: if your latency is dominated by **TTFT**
(prompt processing) more than throughput, an H100's higher SM count
helps even at lower memory. But at 200 RPS, L40S is the cost-
optimal pick.

**Senior signal**: mention `g6` vs `g6e` distinction (L4 vs L40S),
quantization quality vs INT8 (AWQ/GPTQ specific), and `inf2.xlarge`
(Inferentia 2) as an even cheaper option if your model has a Neuron-
supported architecture.

</details>

### B5. S3 deep — object storage you need to master

S3 is the gravity well of AWS. Everything ends up here.

Concepts:

- **Bucket** — namespace; globally unique name.
- **Object** — file plus metadata; up to 5 TB.
- **Key** — the object's "path" inside the bucket. There are no real folders; `/` is a convention.
- **Prefix** — substring at the start of keys; used for sharded access.
- **Versioning** — every PUT creates a new version; deletes are tombstones.
- **MFA Delete** — requires MFA token to permanently delete a version.

Storage classes:

- **S3 Standard** — hot tier; default.
- **S3 Intelligent-Tiering** — auto-migrates objects between tiers based on access. Default for unknown access patterns.
- **S3 Standard-IA / One Zone-IA** — infrequent access; cheaper storage, retrieval cost.
- **S3 Glacier Instant Retrieval** — archive, milliseconds to retrieve.
- **S3 Glacier Flexible Retrieval / Deep Archive** — minutes to hours retrieval; cheapest.

Performance:

- 3,500 PUT/COPY/POST/DELETE per second per prefix. 5,500 GET/HEAD per prefix. Scales with prefix count.
- For training data, **shard your training data across prefixes** so concurrent workers don't hammer one prefix.

Lifecycle policies:

- Transition objects to cheaper tiers after N days.
- Expire objects (delete) after M days.
- Expire incomplete multipart uploads (sneaky cost driver).

Replication:

- **Cross-Region Replication (CRR)** — DR or multi-region read.
- **Same-Region Replication (SRR)** — log aggregation, cross-account workflows.

Access:

- **Bucket policy** — resource-based; great for cross-account.
- **Block Public Access** — flip it on at account level. The most common F500 audit finding is "S3 public exposure." Don't be that team.
- **S3 Access Points** — named per-application access policies; clean for multi-tenant.
- **VPC Endpoint for S3** — private network path; cheaper egress.

<details>
<summary><strong>F500 Q:</strong> Your training data sits in `s3://datasets/imagenet/` as 1.3 M JPEG objects with random keys. Training is slow — workers are I/O bound. Walk through the diagnostic and the fix (think prefix design, WebDataset / shards, FSx for Lustre).</summary>

**In-depth answer**

**Diagnostic**:

1. **Confirm I/O bound, not GPU bound**. `nvidia-smi` shows GPU
   utilization sawtoothing 0→100%; `iostat -xz 1` shows low CPU
   utilization; many DataLoader workers stuck in `D` state. Classic
   small-file I/O starvation.

2. **Count S3 GETs**. CloudWatch metric `NumberOfObjects`. 1.3M
   files × N epochs = ten of millions of GETs. Each takes ~5-50ms
   round-trip; even at 32 worker processes you're capped at maybe
   1000 GETs/second total, which can't feed 8 H100s.

3. **Check S3 prefix distribution**. If keys are
   `imagenet/n01440764/n01440764_10026.JPEG` etc., 1000 class
   prefixes — okay. If keys are `imagenet/00001.jpg`, `imagenet/
   00002.jpg`, ... — single prefix, throttled to ~5500 GETs/sec
   total *for everyone in your account*.

4. **Check listing patterns**. If your DataLoader does
   `aws s3 ls` to enumerate before each epoch, that's a ListObjects
   storm at job start; minutes of wasted time.

**The fix — in increasing order of work**:

1. **Reshard into WebDataset .tar files**. Group 1000 images into
   one ~250 MB tar. Total: ~1300 shards instead of 1.3M objects.
   Each shard is one S3 GET. WebDataset streams sequentially —
   throughput scales linearly with parallelism.
   ```python
   # Quick reshard
   import webdataset as wds
   with wds.TarWriter("imagenet-train-000000.tar") as w:
       for img_path, label in dataset:
           w.write({"__key__": str(idx), "jpg": open(img_path, "rb").read(), "cls": label})
   ```

2. **Shard across S3 prefixes**. Put shards under
   `s3://datasets/imagenet/shards/00/`, `.../01/`, ..., `.../63/` —
   each prefix gets its own throughput allocation. 64 prefixes × 5500
   GETs/sec = 350K GETs/sec aggregate. Plenty.

3. **Each training worker reads disjoint shards** — no two workers
   GET the same shard. The shard URL list is shuffled per epoch.

4. **For multi-epoch heavy-IO training, stage to FSx for Lustre**.
   FSx auto-imports from S3 (data repository association); aggregate
   throughput 25+ GB/s; POSIX file access. Cost: ~$0.145/GB/month
   for persistent SSD storage. For 1 TB of ImageNet, ~$145/month.
   Worth it for hundreds of GPU-hours.

5. **For very long runs, cache hot data on instance-local NVMe**.
   `p5.48xlarge` has 30+ TB of local NVMe; copy the shards there
   once at job start. Subsequent epochs from local NVMe = 100+ GB/s
   per node.

**The decision tree**:

| Scenario | Solution |
|---|---|
| One-shot training, ~100 GB data | WebDataset shards on S3 directly |
| Heavy multi-epoch, ~1 TB | FSx Lustre with S3 import |
| Massive scale, many concurrent jobs | FSx Lustre + local NVMe cache |
| Streaming (training data continues to arrive) | WebDataset + WebDataset's resampling pipeline |

**SA-level twist**: there's a meta-pattern here. **The problem isn't
S3** — it's an impedance mismatch between training's small-batch read
pattern and S3's per-GET overhead. The right fix is always "make the
unit of I/O bigger." Tar shards, parquet files, mosaic format —
all variants of the same idea.

**Senior signal**: mention NVIDIA DALI or torchdata's DataPipe for
the read side, and that for purely sequential reading (typical of
LLM pretrain on tokenized data), MosaicML's StreamingDataset format
is the 2026 best-of-breed (built-in resumability, deterministic
shuffling, S3-friendly).

</details>

<details>
<summary><strong>F500 Q:</strong> Walk through the S3 storage class lifecycle for ML logs that you query frequently for 30 days, occasionally for 1 year, and almost never after that. What's the cost difference vs all-Standard?</summary>

**In-depth answer**

**The lifecycle policy**:

```json
{
  "Rules": [{
    "ID": "ml-logs-tiering",
    "Filter": {"Prefix": "logs/"},
    "Status": "Enabled",
    "Transitions": [
      {"Days": 30, "StorageClass": "STANDARD_IA"},
      {"Days": 90, "StorageClass": "GLACIER_IR"},
      {"Days": 395, "StorageClass": "DEEP_ARCHIVE"}
    ],
    "Expiration": {"Days": 2555}
  }]
}
```

7-year retention (2555 days) is typical for SR 11-7 / financial /
healthcare compliance.

**Tier behaviour and cost**:

| Tier | Storage cost/GB | Retrieval cost | Min duration | Min size |
|---|---|---|---|---|
| Standard | $0.023 | free | none | none |
| Standard-IA | $0.0125 | $0.01/GB | 30 days | 128 KB |
| Glacier IR | $0.004 | $0.03/GB | 90 days | 128 KB |
| Deep Archive | $0.00099 | $0.02/GB + 12h | 180 days | 128 KB |

**Cost calculation** for, say, **100 TB of logs accumulating over
1 year, then 6 more years of retention**:

| Approach | Year 1 | Years 2-7 (each) | 7-year total |
|---|---|---|---|
| All Standard | ~$28K | ~$28K | ~$197K |
| Tiered (above policy) | ~$15K | ~$1.2K | ~$22K |

That's ~$175K saved over 7 years on a 100 TB log dataset. Real
money.

**Important caveats**:

1. **Intelligent-Tiering** can be a better default if you don't
   know access patterns. It auto-moves objects between Frequent,
   Infrequent, and Archive tiers, $0.0025/1000 objects monitoring
   fee. For logs with known patterns, explicit lifecycle is cheaper
   per GB but more bespoke.

2. **Glacier retrieval latency**:
   - Glacier Instant Retrieval: milliseconds (same as S3 Standard).
   - Glacier Flexible Retrieval: 1-5 minutes (expedited), 3-5 hours
     (standard), 5-12 hours (bulk).
   - Glacier Deep Archive: 12-48 hours.
   For ML logs you query "occasionally for 1 year" — Glacier IR
   matches that pattern. Deep Archive only for truly cold storage.

3. **Per-object minimum duration**. Objects in IA / Glacier IR /
   Deep Archive are billed for at least 30 / 90 / 180 days
   respectively, even if you delete them earlier. For high-churn
   logs this matters.

4. **Per-object minimum size**. Objects < 128 KB are billed as
   128 KB in IA / Glacier tiers. For ML small-log files, batch
   into larger objects first (Athena / Glue compaction or just
   batched writes).

**SA-level twist**: don't only think about storage cost — think about
*retrieval* cost. If a compliance audit asks for 6-year-old logs,
retrieving 100 GB from Deep Archive is $2 storage + $20 retrieval +
12 hours. Plan the retrieval SLA into your runbook.

**The right ML-logs architecture**: write to S3 Standard via Kinesis
Firehose / FluentBit, partition by `year=/month=/day=` for Athena
queries, lifecycle on date prefix, query via Athena (which works
across tiers transparently for Glacier IR; needs `RESTORE` for
Deep Archive).

</details>

### B6. EKS — Kubernetes on AWS

EKS is AWS's managed Kubernetes. Control plane managed by AWS (~$72/month). You bring data plane (nodes).

Three data plane options:

- **EC2 managed node groups** — AWS provisions EC2 nodes for you per a config.
- **Self-managed nodes** — you launch the EC2 yourself.
- **AWS Fargate for EKS** — serverless pods (no node management). Limited GPU support; rare for ML.

The pieces that matter for ML on EKS:

- **AWS Load Balancer Controller** — provisions ALBs/NLBs for K8s services and ingresses.
- **AWS EBS / EFS / FSx CSI drivers** — durable storage.
- **NVIDIA Device Plugin** — exposes `nvidia.com/gpu` as schedulable.
- **NVIDIA GPU Operator** — installs drivers + plugin + DCGM monitoring as a bundle.
- **Karpenter** — modern node provisioner; faster + smarter than Cluster Autoscaler. Standard for new clusters.
- **IRSA (IAM Roles for Service Accounts)** — pods assume IAM roles via OIDC; no AWS keys in pods.
- **AWS Secrets Manager / External Secrets Operator** — sync secrets into K8s secrets.
- **Argo CD / Flux** — GitOps reconciliation.

Capacity providers:

- **Karpenter with EC2 GPU node pools** — instances scale up/down with demand.
- **Cluster Autoscaler** (older) — works but slower; replace with Karpenter when you can.

<details>
<summary><strong>F500 Q:</strong> Walk through the IRSA setup: K8s ServiceAccount, IAM role trust policy, pod usage. What does this give you vs storing AWS keys in a Secret?</summary>

**In-depth answer**

**IRSA = IAM Roles for Service Accounts**. The mechanism by which an
EKS pod assumes an IAM role without any AWS credentials in its
environment.

**The flow**:

1. **OIDC provider per cluster**. EKS exposes an OIDC issuer URL. Add
   it as an IAM Identity Provider in your AWS account (one-time).

2. **IAM role with a trust policy** scoped to the cluster's OIDC issuer
   AND a specific K8s ServiceAccount:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Principal": {"Federated": "arn:aws:iam::123:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/EXAMPLED539D4633"},
       "Action": "sts:AssumeRoleWithWebIdentity",
       "Condition": {
         "StringEquals": {
           "oidc.eks.us-east-1.amazonaws.com/id/EXAMPLED539D4633:sub": "system:serviceaccount:ml-prod:trainer",
           "oidc.eks.us-east-1.amazonaws.com/id/EXAMPLED539D4633:aud": "sts.amazonaws.com"
         }
       }
     }]
   }
   ```

3. **K8s ServiceAccount** annotated with the role ARN:
   ```yaml
   apiVersion: v1
   kind: ServiceAccount
   metadata:
     name: trainer
     namespace: ml-prod
     annotations:
       eks.amazonaws.com/role-arn: arn:aws:iam::123:role/eks-trainer-role
   ```

4. **Pod uses the ServiceAccount**:
   ```yaml
   spec:
     serviceAccountName: trainer
     containers: [...]
   ```

5. **EKS Pod Identity webhook** mutates the pod at admission, injecting:
   - `AWS_ROLE_ARN` env var
   - `AWS_WEB_IDENTITY_TOKEN_FILE` env var pointing at a projected
     volume containing a short-lived JWT.

6. **AWS SDK in the pod** detects these vars, calls
   `sts:AssumeRoleWithWebIdentity` automatically. No code change.

**What IRSA gives you vs AWS keys in a Secret**:

| Concern | Keys in Secret | IRSA |
|---|---|---|
| Credential lifetime | Permanent | ~1 hour, auto-rotated |
| Leak blast radius | Forever, account-wide | < 1 hour, scoped |
| Pod-level isolation | All pods using the secret share the identity | Each ServiceAccount → distinct role |
| Audit trail | IAM user level | Pod-level via OIDC sub |
| Rotation | Manual, error-prone | Automatic |
| Secret sprawl | Real | None |
| Compliance | Audit findings | Passes |

**The 2026 alternative**: EKS Pod Identity (no OIDC, no token volume).
Same outcome with simpler setup. Both are supported; new clusters can
pick either. IRSA is the established pattern; Pod Identity is the
future. Know both.

**Senior signal**: mention that the OIDC subject claim
(`system:serviceaccount:ns:sa-name`) is the auth boundary. Mistake-
proof your trust policy by always pinning both namespace and
service-account name; otherwise any SA in the cluster can assume.

</details>

<details>
<summary><strong>F500 Q:</strong> Your training job needs 8 H100 GPUs on one node, plus 1 Gbps node-to-node throughput. Walk through the EKS configuration: instance type, node group, placement group, EFA, NCCL setup.</summary>

**In-depth answer**

**Two scopes to address**:

1. **8 H100 GPUs on one node** — choose the right instance and node
   group configuration.
2. **1 Gbps internode throughput** — this is *low* for ML training
   (modern setups run 100-400 Gbps). I'll cover both the literal ask
   and what's expected at F500 scale.

**Instance type**: `p5.48xlarge` (8× H100-80GB on a single node,
NVLink for intra-node bandwidth ~900 GB/s aggregate, 3.2 Tbps EFA
network ports).

**EKS configuration**:

1. **Node group** dedicated to GPU training:
   ```yaml
   nodeGroups:
     - name: gpu-train-h100
       instanceType: p5.48xlarge
       minSize: 0
       maxSize: 8
       desiredCapacity: 0  # scale up on demand
       availabilityZones: ["us-east-1a"]  # all in one AZ for placement
       privateNetworking: true
       efaEnabled: true   # 32 EFA NICs per p5.48xlarge
       placement:
         groupName: ml-train-cluster   # cluster placement group
       taints:
         - key: nvidia.com/gpu
           value: "true"
           effect: NoSchedule
   ```

2. **Cluster placement group** — created beforehand:
   ```bash
   aws ec2 create-placement-group --group-name ml-train-cluster \
     --strategy cluster
   ```
   This places all instances in the group physically close (same
   rack / same spine) for lowest latency.

3. **Install NVIDIA GPU Operator** in the cluster:
   ```bash
   helm install gpu-operator nvidia/gpu-operator -n gpu-operator
   ```
   This handles drivers, NVIDIA container toolkit, DCGM monitoring,
   MIG (if needed).

4. **EFA device plugin** for K8s — exposes EFA NICs as schedulable
   resources:
   ```yaml
   resources:
     limits:
       nvidia.com/gpu: 8
       vpc.amazonaws.com/efa: 32
   ```

5. **NCCL setup** inside the container:
   ```dockerfile
   FROM nvcr.io/nvidia/pytorch:24.10-py3
   # Install EFA software stack
   RUN curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz \
       | tar zx && cd aws-efa-installer && ./efa_installer.sh -y --skip-kmod
   ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$LD_LIBRARY_PATH
   ENV PATH=/opt/amazon/openmpi/bin:$PATH
   # NCCL plugin for AWS EFA
   RUN git clone https://github.com/aws/aws-ofi-nccl.git ...
   ```

   Required env vars for NCCL to use EFA:
   ```bash
   FI_PROVIDER=efa
   FI_EFA_USE_DEVICE_RDMA=1
   NCCL_PROTO=simple
   NCCL_DEBUG=INFO  # for first run; verify EFA in the log
   ```

6. **Pod spec**:
   ```yaml
   spec:
     hostNetwork: true            # required for EFA performance
     dnsPolicy: ClusterFirstWithHostNet
     containers:
       - name: trainer
         image: my-trainer:latest
         resources:
           limits:
             nvidia.com/gpu: 8
             vpc.amazonaws.com/efa: 32
             hugepages-2Mi: 5120Mi
             memory: 1900Gi
   ```

7. **Launch with PyTorchJob** (Kubeflow Training Operator) or `mpijob`
   for MPI-based launchers. Single-node training: vanilla Job.

**About the "1 Gbps" requirement**:

If literally 1 Gbps suffices (e.g., infrequent gradient sync for
a small model), `p5.48xlarge` overkills the network. Single-AZ
TCP between regular VPC-routed EC2 instances delivers 25-50 Gbps
trivially; no EFA needed. Skip the EFA complexity, use cheaper
`g6.48xlarge` (8× L40S, 48 GB ea).

**At F500 scale you'd usually want 100+ Gbps**:

- `p5.48xlarge` delivers 3.2 Tbps via 32 EFA NICs.
- Verify with `NCCL_DEBUG=INFO` showing
  `NET/AWS Libfabric/0/EFA-`.
- All-reduce benchmark (`nccl-tests`) should show 90%+ of theoretical
  bandwidth.

**SA-level twist**: for multi-node H100 training, the AWS UltraCluster
configuration (P5 + EFA + Cluster Placement Group + EKS Hyperpod or
ParallelCluster) is the path. SageMaker HyperPod handles all this
automatically — good when you want managed Slurm-like resilience.

**Senior signal**: mention `NCCL_TOPO_FILE`, `NCCL_IB_DISABLE=1` (force
EFA), `NCCL_SOCKET_IFNAME` for diagnosis, and the `aws-ofi-nccl`
plugin version compatibility matrix with PyTorch.

</details>

### B7. SageMaker — AWS's managed ML platform

SageMaker is broad. The pieces that matter for DL:

- **SageMaker Studio** — managed JupyterLab. Convenient; can rack up costs (idle notebooks).
- **Training Jobs** — managed distributed training. Script mode (your script + base image) or BYO container.
- **Hyperparameter Tuning Jobs** — managed Bayesian / Hyperband / Grid.
- **Pipelines** — managed DAGs with the SageMaker SDK.
- **Model Registry** — versioned models with approval status.
- **Endpoints** — managed serving:
  - Real-time — always-on
  - Async — long-running inference, requests queued
  - Batch transform — large batch inference
  - Serverless — scale-to-zero (cold-start cost)
  - Multi-model — many models on one endpoint
- **JumpStart** — pre-built fine-tuning templates for popular models (Llama, Mistral, BERT, ViT).
- **Model Monitor** — built-in drift and bias monitoring.
- **Feature Store** — online (DynamoDB-backed) + offline (S3 / Iceberg) feature store.
- **Inference Recommender** — automated benchmarking across instance types.
- **Clarify** — explainability + bias.
- **Neuron** — for Trainium / Inferentia.

The honest assessment of SageMaker for DL:

- **Training Jobs** — fine for typical fine-tunes. For frontier-scale (multi-node Megatron), people often skip SageMaker and run on raw EKS / Slurm for control. SageMaker HyperPod (2024+) addresses this for large clusters.
- **Endpoints** — fine for stable workloads. For LLM-style serving (continuous batching, PagedAttention), prefer Bedrock or roll your own vLLM on EKS — SageMaker's LLM hosting has caught up but is not the default in 2026.
- **Feature Store** — usable; less feature-rich than Tecton or Feast.

<details>
<summary><strong>F500 Q:</strong> When would you reach for a SageMaker Training Job vs a vanilla `kubectl apply` PyTorchJob on your own EKS cluster? Where's the break-even?</summary>

**In-depth answer**

**SageMaker Training Job wins when**:

1. **You don't run a K8s cluster** or don't want to maintain one
   for ML training. SageMaker = managed; you `boto3.start_training_job(...)`.
2. **Small team** (1-5 ML engineers). EKS GPU operations need a
   dedicated platform engineer; SageMaker abstracts that away.
3. **Bursty workloads** — train once a week, want zero infra cost
   between runs. SageMaker spins instances up/down per job.
4. **You need built-in features**: managed Spot training (auto-
   checkpoint + resume on interruption), warm pools (pre-warmed
   instances cut start time from ~5 min to ~30 sec), automatic
   model registry integration, automatic hyperparameter tuning.
5. **Compliance value** — SageMaker Training Jobs run in AWS-managed
   accounts; the audit boundary is cleaner than DIY EKS for some
   regulators.
6. **Distributed Training Library** — SageMaker's SMDDP provides
   AWS-optimized NCCL collectives that often outperform vanilla
   NCCL by 10-30% on AWS hardware.

**EKS PyTorchJob wins when**:

1. **You already run EKS** for other workloads. Adding ML training
   reuses the platform. Marginal cost = the GPU nodes themselves.
2. **Large team / many concurrent jobs** — cluster scheduler (Volcano,
   Kueue, Yunikorn) gives gang scheduling, fairness, quotas that
   SageMaker's per-job model doesn't.
3. **Specialized infrastructure** — custom kernel modules, advanced
   networking (custom CNI for high-speed RDMA), persistent shared
   filesystems (Lustre), bespoke schedulers (Slurm + EKS hybrid).
4. **Cost at scale**. SageMaker Training Jobs have a per-second
   premium of ~15-30% over equivalent EC2 + your operating cost.
   Above ~10K GPU-hours/month, owning the infrastructure pays back.
5. **Multi-cloud or hybrid** — if you also train on-prem or in GCP,
   K8s gives you a portable abstraction; SageMaker locks you to AWS.
6. **Experiment tooling** — your team uses W&B / MLflow / Argo
   Workflows / Kubeflow Pipelines that integrate cleanly with K8s.

**The break-even**:

| Indicator | SageMaker | EKS |
|---|---|---|
| < 5 ML engineers | ✓ | |
| < ~3 GPU-hours / day average | ✓ | |
| Team has K8s platform engineer | | ✓ |
| > 10 concurrent training jobs typical | | ✓ |
| > $100K/month GPU spend | | ✓ (with break-even ~$200-500K) |
| Custom networking needed | | ✓ |
| Single AWS account, simple use case | ✓ | |

**The 2026 middle ground**: **SageMaker HyperPod**. Persistent multi-
node clusters with deep K8s/Slurm integration, but managed by AWS.
You don't manage the cluster control plane; you do get the long-
running persistent compute model. Good for orgs that have outgrown
Training Jobs but don't want to staff a K8s platform team.

**SA-level twist**: at F500 hiring, "I architected the SageMaker →
EKS migration when our spend hit X" is a high-signal story. The
opposite ("we built EKS too early and had ops overhead with one
training job a week") is the more common mistake.

**Senior signal**: bring up CUR data — actual GPU-hour spend vs
expected — as the deciding input. Don't decide on intuition.

</details>

<details>
<summary><strong>F500 Q:</strong> You're serving a fine-tuned Llama-3-8B at 50 RPS to internal employees. Pick a SageMaker endpoint type vs Bedrock vs self-hosted vLLM on EKS. Justify in cost + latency + ops terms.</summary>

**In-depth answer**

**The three options**:

1. **SageMaker Real-Time Endpoint** with Llama-3-8B JumpStart template.
2. **Bedrock** with on-demand Llama 3 (managed, no infra at all).
3. **Self-hosted vLLM on EKS** on `g6e.xlarge` (L40S).

**Workload context**: 50 RPS internal. Assume avg 200 tokens in, 300
tokens out. So ~5K input + 7.5K output tokens/sec aggregate.
Sub-second TTFT requirement.

**Bedrock (managed)**:
- **Cost**: Llama 3 70B Bedrock is roughly $0.00265 input + $0.0035
  output / 1K tokens. 8B is cheaper, say ~$0.0003 input + $0.0006
  output / 1K tokens.
  Monthly = (5K × 30d × 86400s × $0.0003 / 1K + 7.5K × 30d ×
  86400 × $0.0006 / 1K)/1K ≈ $15K/month.
- **Latency**: ~500-1200ms TTFT typical; meets sub-2s easily.
- **Ops**: zero. No infrastructure.
- **Fine-tune support**: Bedrock supports Llama fine-tunes via Bedrock
  Custom Models; requires extra "provisioned throughput" purchase
  for serving (separate cost line), often the deal-breaker.

**SageMaker Real-Time Endpoint**:
- **Instance**: `ml.g6e.xlarge` (L40S) — sufficient for 8B INT8 or
  FP16 at this RPS with continuous batching (SageMaker Large Model
  Inference DJL container provides it).
- **Cost**: ~$1.86/hr × 730 = $1,360/month per replica. For 50 RPS
  with bursts, 2 replicas + auto-scaling = ~$2,720/month base. Add
  ~30% for SageMaker premium = ~$3,500/month.
- **Latency**: ~400-800ms TTFT.
- **Ops**: light. Endpoint config, model registration, auto-scaling
  policies. No K8s expertise needed.
- **Custom model support**: trivial — upload your fine-tuned weights
  to S3, register, deploy.

**Self-hosted vLLM on EKS**:
- **Instance**: `g6e.xlarge` on-demand $1.86/hr × 730 = $1,360/month
  per replica. 2 replicas = $2,720/month. 1-year Savings Plan brings
  this to ~$1,900/month.
- **Latency**: ~300-700ms TTFT (vLLM continuous batching is
  excellent; matches or beats SageMaker LMI container).
- **Ops**: real. You operate the K8s deployment, the autoscaler,
  the metrics, the rolling upgrades, GPU node lifecycle. Probably
  0.2-0.5 FTE of platform engineering time/month.
- **Custom model**: trivial — vLLM loads HF format directly.

**The pick at 50 RPS internal**:

- **For a small team (no platform engineer)**: **Bedrock or SageMaker**.
  Bedrock if pricing fits and your fine-tune is supported; SageMaker
  if Bedrock's fine-tune path is painful or your routing logic is
  complex.
- **For a team with K8s operations capability and bigger plans**:
  **self-hosted vLLM on EKS**. Saves ~$1.5K/month vs SageMaker, the
  bigger win is *future flexibility* — multi-LoRA serving, prefix
  caching, quantization-aware deployment.
- **For variable / spiky workload**: SageMaker Serverless or Async
  inference; cold start tolerable for internal users.

**The decision matrix**:

| Concern | Bedrock | SageMaker RT | vLLM/EKS |
|---|---|---|---|
| Time to ship | Days | 1-2 weeks | 2-4 weeks |
| Monthly cost | $15K | $3.5K | $2K |
| Custom model | Painful | Easy | Easy |
| Multi-LoRA | No | Limited | Native |
| Ops burden | None | Low | Medium |
| Latency control | Black box | Some | Full |
| Fallback option | Auto | None | DIY |

**SA-level twist**: at 50 RPS the *cost* answer doesn't really matter
($2K-15K is rounding error at most F500s). The right question is
**what will this become at 500 RPS or 5000 RPS in 18 months?** If
the answer is "way bigger," start with vLLM/EKS now because the
migration later is painful. If "this will stay small forever,"
Bedrock's zero-ops wins.

</details>

### B8. AWS Bedrock — managed LLMs

Bedrock = AWS's managed gateway to multiple LLM providers (Claude, Llama, Mistral, Amazon Titan, Cohere, AI21, Stability) with one API.

Key features:

- **Provisioned Throughput** — buy dedicated capacity for a model. Predictable cost, higher throughput.
- **On-demand** — pay per token.
- **Knowledge Bases** — managed RAG (you give docs; it builds an index + answers).
- **Agents** — multi-step planning with tool use.
- **Guardrails** — input/output filtering for safety.
- **Model Customization** — fine-tune via SFT or continued pretraining.
- **Cross-region inference** — Bedrock routes to the nearest region with capacity.

For F500 LLM use cases:

- **Pros** — no model ops; FSI-compliant (data not used for training under enterprise terms); fast time-to-value.
- **Cons** — cost premium; vendor lock-in; less control over latency tails.

<details>
<summary><strong>F500 Q:</strong> A FSI customer asks "Is our prompt + response data used to train Bedrock's models?" Walk through the answer.</summary>

**In-depth answer**

**The short answer**:

> No. Under AWS's data handling terms for Bedrock, customer prompts
> and responses are NOT used to train AWS or any third-party model
> provider's foundation models. This applies whether you use
> on-demand inference, provisioned throughput, or fine-tuning.

**The longer answer (what regulators and risk officers want to hear)**:

1. **Contractual basis**:
   - AWS Service Terms ([Bedrock section](https://aws.amazon.com/service-terms/))
     state: *AWS will not use your inputs or outputs to train models*.
   - The AWS DPA (Data Processing Addendum) covers GDPR / data
     subject rights.
   - For BAA / HIPAA, Bedrock is HIPAA-eligible — AWS will sign a
     BAA covering it.

2. **Network and data flow**:
   - Prompts go from your VPC → AWS Bedrock VPC endpoint → Bedrock
     model invocation service → the underlying model (e.g., Claude
     hosted in AWS infrastructure).
   - Data stays in your chosen region.
   - You can enforce in-VPC access only via VPC endpoint policies
     and `aws:SourceVpce` conditions on IAM policies — preventing
     any Bedrock call from outside your network.

3. **Encryption**:
   - At rest: SSE with AWS-managed or customer-managed KMS keys.
   - In transit: TLS 1.2+.
   - In use: AWS Nitro Enclaves on the inference path (model
     provider–dependent).

4. **Model provider terms**:
   - Anthropic, Meta, Mistral, AI21, Cohere, Stability — each model
     provider's terms via Bedrock are governed by AWS's terms.
     Customer data does NOT flow back to model providers.
   - The model provider's *own* hosted APIs (e.g., anthropic.com)
     have separate terms — *not* the same as Bedrock.

5. **Audit and logging**:
   - **Bedrock Model Invocation Logging** — opt-in feature; writes
     prompt + response logs to your S3 bucket. *You* control retention,
     access, encryption.
   - CloudTrail logs the *fact* of each invocation (who called, when)
     but not the content unless you've enabled the invocation log.

6. **Compliance posture**:
   - Bedrock is included in AWS's SOC 1/2/3, ISO 27001/27017/27018,
     PCI DSS, HIPAA (BAA-eligible), FedRAMP Moderate/High (in
     GovCloud), IRAP, C5, ENS-High, K-ISMS, and other regional
     attestations. Verify the specific list for your region in the
     AWS Compliance documentation.

**The crucial distinction**:

- **Bedrock** (this answer): customer data NOT used for training.
- **Claude on anthropic.com** (different service): governed by
  Anthropic's commercial terms — they similarly do not train on
  business-tier customer data, but the contract is different.
- **OpenAI consumer (chat.openai.com)**: opt-out for training; not
  the same as the enterprise/API tier.

**SA-level twist**: when a FSI customer asks this question, they're
usually really asking "can I trust this to be compliant with
[OCC bulletin / NYDFS Part 500 / FFIEC guidance / GDPR Article
22]?" Don't just answer the literal question — proactively map
their regulatory frameworks to specific AWS attestations and
contractual provisions. Bring the AWS FSI Compliance Center
documentation; bring the BAA template; bring the auditor-pack.

That's the difference between an engineer's answer and an
architect's answer.

</details>

<details>
<summary><strong>F500 Q:</strong> Compare Bedrock Knowledge Bases against rolling your own pgvector + OpenSearch RAG on EKS. When does each win?</summary>

**In-depth answer**

**Bedrock Knowledge Bases (managed RAG)**:

What it gives you out of the box:
- Document ingestion from S3 / Confluence / Salesforce / SharePoint /
  web pages.
- Automatic chunking (semantic, hierarchical, fixed-size, or
  custom Lambda).
- Embedding via your choice (Titan, Cohere, or your own).
- Vector store: OpenSearch Serverless (default), Aurora pgvector,
  Pinecone, MongoDB Atlas, Redis (your choice).
- Hybrid search (semantic + keyword).
- Retrieval + augmented generation via a single API call.
- Re-ranking (Cohere Rerank built-in).
- Citation / source attribution.

**Roll-your-own (pgvector + OpenSearch on EKS)**:

What you build:
- Document ingestion pipeline (S3 events → Lambda / SQS / step
  functions for chunking + embedding).
- Chunking strategy (you implement, you tune).
- Embedding service (Bedrock / OpenAI / self-hosted via vLLM).
- Vector store: pgvector for small (< 10M vectors); OpenSearch with
  k-NN plugin for larger.
- BM25 (OpenSearch native).
- Hybrid scoring: RRF or weighted sum, implemented in your retrieval
  service.
- Reranker (Cohere Rerank API or self-hosted).
- Retrieval API service (FastAPI / NestJS / whatever).
- Generation: call Bedrock / OpenAI / vLLM with retrieved context.

**When Bedrock KB wins**:

1. **Time to first version**: 2-4 weeks. DIY: 2-4 months.
2. **Small team without distributed-systems engineers**.
3. **Stable corpus, modest scale** (< 100M chunks, < 50K queries/day).
4. **Connector-heavy use case** (your data lives in SharePoint /
   Confluence / Salesforce and you don't want to build connectors).
5. **Compliance-sensitive** — Bedrock's audit trail is built-in.
6. **AWS-only stack** — every component natively integrated.

**When DIY wins**:

1. **Bespoke retrieval logic** — query rewriting, multi-step retrieval,
   query routing across multiple indexes, graph augmentation.
2. **Custom chunking** — domain-specific (code repos, tabular data,
   medical records, legal documents) where generic chunking fails.
3. **Custom embedding** — domain-fine-tuned embeddings, asymmetric
   query/doc encoders, ColBERT-style late interaction.
4. **Performance at high scale** — > 1B chunks, > 1000 QPS, P99 <
   100ms. Bedrock KB tops out earlier.
5. **Reranker innovation** — you want your own cross-encoder or
   ColBERT-style late interaction; Bedrock's reranker is fixed.
6. **Multi-cloud / data residency outside AWS**.
7. **Cost at scale** — Bedrock KB's per-query and storage cost adds
   up; at 100K+ queries/day, DIY on EKS is often 50-70% cheaper.

**The hybrid (what real F500 architectures do)**:

Use Bedrock KB for v1 (ship fast, learn the corpus + user patterns),
then incrementally replace components when you outgrow the managed
constraints:

1. v1: Bedrock KB end-to-end.
2. v2: Bedrock KB retrieval but custom reranker.
3. v3: Custom retrieval (OpenSearch + pgvector + reranker), Bedrock
   for generation only.
4. v4: Fully self-hosted including LLM (vLLM) when cost justifies.

Most F500s in 2026 are at v1 or v2.

**SA-level twist**: Bedrock Knowledge Bases is a *lock-in vector*.
Once your prompts assume Bedrock's chunking shape and citation
format, migrating away requires re-evaluating quality. Architects
who recommend Bedrock KB without a thin abstraction shim (Portkey,
LiteLLM, or a custom interface layer) leave a future maintenance
problem behind.

**Senior signal**: mention RAGAS / ARES for evaluation — both work
agnostic of which RAG implementation you've picked, so you can run
the same eval suite as you migrate from Bedrock KB to DIY.

</details>

### B9. AWS data services for ML

- **Glue** — managed Spark + serverless ETL + Data Catalog. Catalog is the metastore for Athena, EMR, Redshift, Lake Formation.
- **Athena** — SQL over S3 + Iceberg + Hudi + Delta.
- **EMR / EMR Serverless** — Spark, Hive, Trino on demand.
- **Redshift / Redshift Serverless** — analytical data warehouse.
- **Lake Formation** — governance over the data lake (row-level security, column masking, tag-based access).
- **DataZone** — newer; data governance + discovery across the org.
- **OpenSearch** — managed Elasticsearch; vector search via k-NN plugin.
- **ElastiCache (Redis)** — managed Redis; online feature store.
- **DynamoDB** — key-value NoSQL; another common online feature store.
- **Kinesis / MSK** — streaming. MSK = managed Kafka. Kinesis is AWS-native.
- **Iceberg on S3** — native first-class support across Glue, Athena, EMR, Redshift Spectrum.

<details>
<summary><strong>F500 Q:</strong> Design the data layer for an ML org with 50 TB of training data, real-time event ingestion at 100K events/sec, and a need to feature-engineer in both batch and streaming. Pick the AWS services and lay out their roles.</summary>

**In-depth answer**

**The architecture**:

```
        Events @ 100K/s
              │
              ▼
   [Kinesis Data Streams OR MSK Kafka]
       │              │
       ▼              ▼
[Kinesis Firehose]  [MSK Connect / Flink on EKS]
       │              │
       │              ├─► Streaming features → ElastiCache Redis
       │              │   (online store)
       │              │
       │              └─► Aggregated tables → Iceberg on S3
       ▼
[S3 raw landing zone — Iceberg tables, hour-partitioned]
       │
       ▼
[Glue Catalog + Lake Formation governance]
       │
       ├────────────────────────────┐
       ▼                            ▼
[EMR / EMR Serverless Spark]   [Athena ad-hoc SQL]
       │
       ▼
[Iceberg curated tables — batch features]
       │
       ▼
[SageMaker Feature Store OR custom] → Training datasets
       │
       ▼
[SageMaker Training / EKS PyTorchJob]
```

**Each service's role**:

| Service | Role | Why |
|---|---|---|
| **Kinesis Data Streams** | Event ingestion at 100K/s | Native AWS, 7-day retention, ordered per-shard, autoscale via `OnDemand` mode |
| Or **MSK** (Kafka) | Same | If you need broader Kafka ecosystem, exactly-once via transactional producer/consumer, longer retention |
| **Kinesis Firehose** | Buffered S3 write of raw events | Managed, no code, partitions to Iceberg/Parquet |
| **MSK Connect / Flink on EKS** | Streaming feature compute | Stateful windowed aggregations, watermarks, exactly-once |
| **ElastiCache Redis** | Online feature store | Sub-millisecond reads for serving |
| **S3 + Iceberg** | Lakehouse storage | Open format, time travel, schema evolution, partitioned at hour level |
| **Glue Data Catalog** | Metastore | Shared metadata across Athena, EMR, Redshift, Spark |
| **Lake Formation** | Governance + RLS/CLM | Row + column security, tag-based access for compliance |
| **EMR Serverless** | Batch feature compute | Spark for joins, aggregations, complex transforms over 50 TB |
| **Athena** | Ad-hoc SQL | Analyst access without spinning EMR |
| **SageMaker Feature Store** | Feature definition + serving abstraction | Optional; some orgs hand-roll instead |
| **OpenLineage + Marquez** | Lineage tracking | Compliance |

**Capacity / sizing math**:

- 100K events/sec × 1 KB/event = 100 MB/sec = ~8.6 TB/day raw.
- Kinesis: shards at 1 MB/s write each; need ~100 shards (or
  OnDemand mode with auto-scaling).
- 50 TB existing + 8.6 TB/day = ~315 TB after a month. Lifecycle
  Iceberg snapshots aggressively (expire old snapshots, compact
  small files).
- Streaming feature state in Redis: a few GBs typically.

**Compliance and governance layer**:

- **Lake Formation tag policies** for PII fields.
- **Column-level masking** for sensitive fields when accessed by
  non-privileged roles.
- **OpenLineage events** flowing into a graph DB for "which dataset
  was used in which model" queries.
- **Glue Data Quality** rules on bronze→silver pipeline.

**Cost ballpark** (educational estimate, real numbers depend on
workload):

- Kinesis OnDemand: ~$36/day base + per-GB charges → ~$3K/month.
- Firehose: ~$0.029/GB ingested → ~$7K/month.
- S3 Standard for hot tier: ~$25/TB/month × 100 TB = $2.5K/month.
- S3 Glacier IR for cold: ~$4/TB/month.
- EMR Serverless on-demand: $0.052/vCPU-hour, varies.
- ElastiCache Redis: depending on cluster size, $1-5K/month.
- Total ML data layer: ~$20-40K/month at this scale.

**SA-level twist**: the *partitioning strategy* is the make-or-break
decision. Iceberg's hidden partitioning + bucketing for high-
cardinality join keys (user_id) is critical for both feature
backfill scans (point-in-time-correct joins) and concurrent reader
performance. Pick partitions before going to scale, because
re-partitioning 50 TB is painful.

**Senior signal**: discuss **Lambda-vs-Kappa** explicitly. The
default in 2026 is Kappa-ish: stream is the source of truth, batch
queries are time-traveled reads of the same lakehouse tables.
Avoids the double-codebase problem.

</details>

### B10. AWS observability and operations

- **CloudWatch Logs** — log destination; agent on EC2/EKS via Fluent Bit or CloudWatch Logs Agent.
- **CloudWatch Metrics** — TS metrics; alarms.
- **CloudWatch Logs Insights** — query language for logs.
- **CloudWatch Container Insights** — K8s + Fargate metrics.
- **X-Ray / AWS Distro for OpenTelemetry (ADOT)** — traces.
- **CloudTrail** — API audit log. *Every* API call. Required for compliance.
- **Config** — resource configuration history + compliance rules.
- **GuardDuty** — security threat detection.
- **Security Hub** — aggregates findings across services.

For ML serving specifically, a typical stack: ADOT collector in the pod → Prometheus-format metrics scraped by AMP (Managed Prometheus) → Amazon Managed Grafana → CloudWatch Alarms / SNS for paging.

<details>
<summary><strong>F500 Q:</strong> Walk through a request path from API Gateway → ALB → EKS pod, showing where you'd capture metrics, where logs, where traces, and how you'd correlate them.</summary>

**In-depth answer**

**The full path**:

```
Client
  │  HTTP request with X-Request-ID (or we generate one)
  ▼
[API Gateway]
  │  Metric: count, latency, 4xx/5xx, throttle.
  │  Log: full request/response (configurable; expensive).
  │  Trace: start root span; propagate trace-id in W3C TraceContext header.
  ▼
[ALB]
  │  Metric: TargetResponseTime, HTTPCode_Target_5XX, HealthyHostCount.
  │  Log: access logs to S3 (optional; cheap).
  │  Trace: pass-through. ALB doesn't emit trace spans natively;
  │         use X-Ray ALB integration or just pass headers.
  ▼
[EKS Pod — Service]
  │  Instrument: OpenTelemetry SDK in app code.
  │  Metric: per-endpoint request count, latency histogram,
  │          per-model inference time, GPU utilization (DCGM),
  │          cache hit rate, request queue depth.
  │          Emitted to Prometheus /metrics scraped by OTel Collector.
  │  Log: structured JSON via stdout; FluentBit ships to CloudWatch /
  │       Loki. Every log line includes trace-id + request-id.
  │  Trace: spans for preprocessing → inference → postprocessing.
  │         Spans tagged with model_version, tenant_id, request_size.
  └──► Downstream: Bedrock / feature store / model server.
        Each call is a child span.
```

**Where to capture each pillar**:

- **Metrics** (Prometheus + Mimir / AMP):
  - API Gateway → CloudWatch metrics auto-emitted; mirror to
    Prometheus via cloudwatch-exporter or alternatives.
  - ALB → CloudWatch metrics auto-emitted.
  - EKS pod → app code emits via OTel; OTel Collector scrapes
    `/metrics` and forwards.
  - Node GPU → NVIDIA DCGM exporter as a DaemonSet.

- **Logs** (Loki / CloudWatch Logs / OpenSearch):
  - API Gateway → optional access logs to S3 or CloudWatch.
  - ALB → access logs to S3.
  - EKS pod → FluentBit DaemonSet → CloudWatch / Loki.
  - All logs include `trace_id`, `span_id`, `request_id`, `tenant_id`.

- **Traces** (Tempo / X-Ray / Jaeger):
  - OTel Collector receives OTLP from pods; ships to Tempo / X-Ray.
  - W3C TraceContext header (`traceparent`) propagated end-to-end.
  - For LLM apps, add custom attributes per span:
    `gen_ai.system="bedrock"`, `gen_ai.usage.input_tokens`,
    `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`.

**How to correlate**:

1. **Trace ID is the master key**. Every log line carries it. Every
   metric (for slow-paths or errors) is annotated with it via
   exemplars.
2. **In Grafana**: click on a slow latency point → exemplar links
   directly to the trace in Tempo → trace shows spans across pod,
   feature store call, Bedrock call → click a span → linked log
   query in Loki shows the structured log lines.
3. **CloudWatch alternative**: ServiceLens + X-Ray Service Map gives
   you the topology view; CloudWatch Container Insights provides
   pod/container metrics; Logs Insights queries with `parse @message
   '*"trace_id": "*"*'` to filter by trace.

**The full stack diagram** (in 2026 best-practice form):

```
EKS Pod
  ├── App (Python) with otel-sdk
  │     emits OTLP over gRPC to localhost:4317
  ├── OTel Collector DaemonSet
  │     receives OTLP
  │     scrapes /metrics
  │     batches, processes, forwards
  └── FluentBit DaemonSet
        ships stdout to log backend

OTel Collector forwards:
  ├── Metrics → AWS Managed Prometheus
  ├── Traces → AWS X-Ray (or Tempo)
  └── Logs → CloudWatch Logs (or Loki)

Grafana / AWS Managed Grafana unifies all three.
```

**SA-level twist**: at F500 scale the trace sampling strategy matters.
100% sampling is expensive (storage, processing). Use **tail-based
sampling** in OTel Collector — keep all error traces, slow traces,
random 1% baseline. Saves 95%+ of trace storage with no loss of
diagnostic value.

**Senior signal**: mention **OpenTelemetry GenAI semantic conventions**
(stable as of 2024) — the namespaced attributes for LLM observability
(`gen_ai.*`). Using these makes traces queryable across LLM
observability vendors (Langfuse, Braintrust, Helicone, Datadog) and
future-proof against tool migration.

</details>

### B11. AWS cost management

- **AWS Cost Explorer** — interactive UI for spend.
- **AWS Budgets** — alerts on spend / forecast.
- **AWS Cost and Usage Report (CUR)** — detailed billing data into S3; query with Athena.
- **Compute Optimizer** — recommendations for right-sizing.
- **Trusted Advisor** — best practices including cost.
- **Savings Plans** vs **Reserved Instances** vs **Spot** — pick mix.
- **Resource Tagging** + **Cost Allocation Tags** — required for attribution.

The DL-specific cost killers:

- Idle GPU instances (notebooks, endpoints with `min_instances > 0`).
- Cross-AZ / cross-region data transfer for training.
- CloudWatch Logs ingestion at high volume.
- NAT Gateway egress for pod-to-pod traffic that should have been on a VPC endpoint.
- Storage in Standard that should be in Intelligent-Tiering.
- Misconfigured S3 multipart uploads accumulating.

<details>
<summary><strong>F500 Q:</strong> Your ML org's bill jumped 40% month over month with no apparent workload change. Walk through the diagnostic protocol.</summary>

**In-depth answer**

**Phase 1 — Quick categorization (1 hour)**:

1. **Cost Explorer view: month-over-month delta by service**.
   ```
   Filter: Linked Account = ml-prod
   Group by: Service
   Compare: This month vs last month
   ```
   Sort by absolute delta. The top 3 services usually explain 80%
   of the surprise.

2. **Cost Explorer view: delta by usage type**.
   Group by `UsageType`. Look for spikes in `BoxUsage:*` (EC2),
   `DataTransfer-*`, `Requests-*` (S3), `KMS-Requests`, etc.

3. **Tag audit**. Group by your `Project` or `Team` tag. Identify
   which team's spend jumped.

By the end of Phase 1 you should know: which service, which team,
which usage type.

**Phase 2 — Drill into the suspect (2-4 hours)**:

If the jump is in **EC2** (most common):
- Compute Optimizer recommendations for over-provisioned instances.
- Cost Explorer with daily granularity to find the day the jump
  started. Often correlates with a deploy or experiment kickoff.
- CloudTrail `RunInstances` events filtered by usage type.

If the jump is in **S3**:
- S3 Storage Lens or Cost Explorer by `UsageType=TimedStorage-*`.
- Did you forget to lifecycle to Glacier? Did versioning explode
  storage with old versions?
- Look for incomplete multipart uploads in old buckets.

If the jump is in **Data Transfer**:
- VPC Flow Logs (or Athena query over CUR data) showing top inter-
  region or out-to-internet traffic.
- Common culprit: a new training pipeline pulling data from a
  bucket in a different region.

If the jump is in **CloudWatch Logs**:
- Logs Insights query: by log group, summed size of incoming logs
  in the last month vs prior. Often a verbose new logging line
  in an inner loop explodes ingestion.

If the jump is in **NAT Gateway**:
- VPC Flow Logs showing top talker destinations through the NAT.
- Common: a new ECR pull pattern or S3 access without a VPC
  endpoint.

If the jump is in **SageMaker / Bedrock / Endpoints**:
- Endpoint hours billed = number of instances × hours.
- Did anyone disable scale-to-zero? Did `min_instances` get bumped?
- Bedrock provisioned throughput purchases (auto-renew default
  enabled — watch this).

**Phase 3 — Investigate the proximate cause (4-8 hours)**:

Once you've narrowed to a service + team + usage type:
- Talk to the team. Did they ship a new pipeline? Run an
  experiment? Forget to teardown?
- CloudTrail: list the IAM identities (roles, users) that
  initiated the spend.
- Check the CI/CD logs for the period — what deployed when?

**Phase 4 — Fix and prevent**:

- Right-size, scale to zero, lifecycle, add VPC endpoints — the
  standard playbook.
- **Set per-team budgets with 50/80/100% alerts**. The root cause
  of "no one noticed" is "no one was paying attention." Budgets
  fix this.
- **AWS Cost Anomaly Detection** — ML-driven anomaly detection on
  spend. Free. Turn it on if it isn't already.

**SA-level twist**: the diagnostic protocol works because you have
**CUR (Cost and Usage Report) in S3 queryable via Athena**. Without
it, you're stuck with Cost Explorer's UI which has aggregation
limits. The architect's first move at any new F500 engagement:
verify CUR + tags + Athena. If not in place, that's project zero.

**Senior signal**: mention that 40% MoM with no workload change is
almost never one thing — it's usually 2-3 stacked things (a new
endpoint stayed on, plus more storage, plus a new dev cluster).
The fix is per-team budgets to prevent the *next* one, not just
hunt-and-kill the current one.

</details>

---

## Part C — GCP and Azure (DL Focus)

### C1. GCP for DL — what differs

- **GKE** — GCP's K8s. Standard mode (you manage nodes) and Autopilot (Google manages nodes). For DL, prefer Standard with GPU node pools.
- **Vertex AI** — GCP's managed ML platform. Vertex Training (custom jobs), Vertex Pipelines (KFP), Vertex Model Registry, Vertex Prediction (endpoints), Vertex Feature Store.
- **TPUs** — Google's custom chips. v4 / v5 widely available. JAX is the natural framework; PyTorch works via PyTorch/XLA but with rough edges.
- **GCS** — object storage. Generally faster small-object reads than S3 in benchmarks; same mental model otherwise.
- **BigQuery** — column-store data warehouse. Exceptional for analytical SQL; integrates tightly with Vertex (BQ table → Vertex training in two clicks).
- **Vertex AI Vector Search** (formerly Matching Engine) — managed ANN search.
- **Vertex Model Garden** — pre-built fine-tuning / serving for Llama, Gemma, etc.
- **Gemini via Vertex** — GCP's managed LLM.
- **IAM** — fundamentally similar but with predefined roles; service accounts are first-class identities.

For DL specifically, GCP shines if your data already lives in BigQuery. The BigQuery → Vertex pipeline is the smoothest of the major clouds.

<details>
<summary><strong>F500 Q:</strong> Compare GCS + GKE + Vertex against S3 + EKS + SageMaker for a CV training pipeline at 50 GPU-hours / week. What's meaningfully different operationally?</summary>

**In-depth answer**

**At 50 GPU-hours/week the absolute cost difference is small** (under
$1000/week either way for L4 or A10G workloads). The differences
that actually matter operationally:

**Tooling integration**:

- **GCS + GKE + Vertex**: BigQuery → Vertex training is the smoothest
  data → model loop in the industry. If your data is in BigQuery,
  GCP wins by a mile. Vertex Pipelines (KFP-based) is more verbose
  than SageMaker Pipelines but more portable.
- **S3 + EKS + SageMaker**: SageMaker Studio is the integrated
  notebook + training + endpoint experience. SageMaker Pipelines
  uses a Python SDK that's easier than KFP. Trainium / Inferentia
  for custom silicon if you go down that path.

**Networking**:

- **GCP**: Cloud Interconnect, VPC peering, Private Service Connect.
  Networking is generally simpler than AWS — fewer concepts (no
  separate NAT GW, no separate IGW), but less granular control.
- **AWS**: more concepts, more verbose, more control. Steeper
  learning curve.

**GPU availability**:

- **AWS**: broadest GPU instance catalog (p5, p4d, g6e, g6, inf2,
  trn1, trn2). H100 generally available on-demand though spot
  varies.
- **GCP**: A100 / H100 / L4 / TPU. TPU is a real lever if your model
  is JAX/XLA-friendly. H100 availability varies by region.

**Container orchestration**:

- **GKE**: standard mode is similar to EKS. Autopilot mode (Google
  manages nodes) is unique — billed per pod-resource, not per node.
  For bursty ML, Autopilot can be cost-effective; for steady GPU
  workloads, Standard with GPU node pools.
- **EKS**: Karpenter is the modern node autoscaler; matches Autopilot
  for fast provisioning, more control. EKS Fargate has limited GPU
  support.

**IAM**:

- **GCP IAM**: principal-centric, simpler model — roles attached to
  principals; service accounts are first-class. Workload Identity
  Federation is GCP's OIDC equivalent.
- **AWS IAM**: more granular, more verbose. Trust + permission policies
  are powerful but error-prone.

**ML platform integration**:

- **Vertex Model Registry** ↔ SageMaker Model Registry — feature parity.
- **Vertex AI Platform Pipelines** uses KFP DSL — portable across
  K8s. SageMaker Pipelines uses bespoke Python SDK — AWS-only.
- **Vertex AI Vector Search** (formerly Matching Engine) vs OpenSearch
  k-NN. Matching Engine is faster and cheaper at scale but less
  flexible.
- **Vertex Model Garden** vs **Bedrock + SageMaker JumpStart** —
  similar premise (managed access to Llama/Gemma/etc.).

**Costs at this scale**:

- 50 GPU-hours/week ≈ 200 hours/month. L4 on-demand ~$0.85/hr =
  $170/month. T4 / A10G similar.
- Storage: 1-5 TB training data at ~$0.020-0.023/GB/month = $20-100.
- Egress: trivial at this scale.

**Where the operational difference is biggest** (the answer that
distinguishes seniors):

**For a small team / tabular-heavy or notebook-driven workflow**:
**Vertex AI**. The Vertex Workbench → BigQuery → Vertex Training →
Vertex Endpoints flow is genuinely smoother than the SageMaker
equivalent. Less YAML, less IAM ceremony.

**For a team already on AWS**:
**SageMaker + EKS hybrid**. SageMaker for training jobs;
EKS+vLLM/Triton for serving when you need it. Migration cost is
zero.

**For broadest ecosystem / Llama-class LLM serving**:
**AWS** wins. Bedrock + vLLM-on-EKS gives you both managed and
self-hosted on the same platform.

**For ML+data engineering**:
**GCP** wins. BigQuery's column store + serverless query model is
genuinely better than Redshift / Athena for ad-hoc analytical SQL.

**SA-level twist**: at 50 GPU-hours/week the choice isn't a cost
optimization — it's a *people optimization*. Pick the platform your
team already knows. Migration cost > any per-platform savings at
this scale.

**Senior signal**: mention multi-cloud reality. Most F500s end up with
both (Azure for Entra ID + OpenAI; AWS or GCP for ML workload). The
architect's question is "what's your second-cloud strategy?" — most
orgs don't have one.

</details>

### C2. Azure for DL — what differs

- **AKS** — Azure's K8s.
- **Azure ML** — managed ML platform with MLflow native (Azure ML hosts MLflow Tracking).
- **Azure OpenAI Service** — exclusive enterprise access to GPT / o-series models with compliance terms.
- **Azure AI Search** — managed search + vector retrieval.
- **Azure AI Studio** — generative AI umbrella.
- **Blob Storage** — object storage. Hot / Cool / Archive tiers.
- **AKS + ND-series VMs** — GPU instances. ND H100 v5 = 8x H100 / node.
- **Microsoft Entra ID (formerly Azure AD)** — identity, federation to AWS / GCP common at multi-cloud F500s.

For F500s with deep Microsoft investments (most banks, pharmas, governments), Azure + Azure OpenAI is often the path of least friction politically.

<details>
<summary><strong>F500 Q:</strong> A bank's GenAI strategy mandates "OpenAI but with FSI compliance." Walk through Azure OpenAI vs Bedrock-Anthropic. Which fits better and why?</summary>

**In-depth answer**

**The literal ask**: "OpenAI but FSI-compliant" means **Azure OpenAI
Service**. AWS Bedrock doesn't offer OpenAI models.

But the senior answer reframes the question.

**The real question**: "We want frontier-quality LLM with FSI
compliance posture and a vendor relationship we trust." That's a
broader space.

**Azure OpenAI Service**:

- **Models**: GPT-4o, GPT-4o-mini, GPT-4.1, o1, o3-mini family, DALL-E,
  Whisper, embeddings. Frontier OpenAI models.
- **FSI compliance**: SOC 1/2/3, ISO 27001, HIPAA (BAA-eligible),
  PCI DSS, FedRAMP High (Azure Gov), CSA STAR, plus regional (FFIEC
  guidance compatible, NYDFS Part 500 compatible).
- **Data handling**: customer prompts not used to train models.
  Abuse-monitoring data retained for 30 days (or zero if you apply
  for "data residency for abuse monitoring waived" — required for
  some regulated industries).
- **Network**: deployable to VNet via Private Link.
- **Identity**: Entra ID native; trivially integrates with banks'
  existing AD/Entra.
- **Pricing**: per-token + provisioned throughput model. PTUs
  (Provisioned Throughput Units) for committed capacity.
- **Models lifecycle**: OpenAI's frontier models hit Azure with a
  delay (sometimes weeks). For most enterprises that's acceptable.

**AWS Bedrock + Anthropic (Claude)**:

- **Models**: Claude family (Sonnet, Opus, Haiku 4.x), plus Llama,
  Mistral, Cohere, AI21, Stability, Amazon Titan, Amazon Nova.
- **FSI compliance**: SOC 1/2/3, ISO 27001/27017/27018, HIPAA (BAA-
  eligible), PCI DSS, FedRAMP High in AWS GovCloud, plus regional
  attestations.
- **Data handling**: same posture as Azure OpenAI — no training on
  customer data, opt-in invocation logging to your own S3.
- **Network**: VPC endpoint via PrivateLink; in-region.
- **Identity**: AWS IAM; integrates via SAML / OIDC federation with
  Entra ID if needed.
- **Pricing**: per-token + Provisioned Throughput.
- **Models lifecycle**: Anthropic's frontier ships to Bedrock close
  to same-day.

**Which fits better — depends on the bank**:

**Choose Azure OpenAI when**:
- The bank is Microsoft-heavy (Outlook, Teams, Entra ID, SharePoint
  data sources) — integration is free.
- The use case is GPT-specific (GPT-4o for general; o-series for
  reasoning).
- Microsoft 365 Copilot is also in the mix — same Azure tenancy.

**Choose AWS Bedrock + Claude when**:
- The bank's data and ML stack live in AWS — your training data,
  embeddings, feature store, serving cluster all in AWS already.
- The use cases involve long-context (Claude excels at 200K+ context
  windows for document review).
- You want a multi-model gateway (Claude *and* Llama *and* Titan all
  via one Bedrock API).

**The senior answer**:

> "If by 'OpenAI' you mean the OpenAI brand and GPT-class quality,
> Azure OpenAI is the path. If you mean 'frontier-quality LLM with
> FSI compliance,' both work — choose by your existing stack.
> Critically: **don't lock in to one provider**. Use a thin
> abstraction layer (LiteLLM, Portkey, or your own gateway) so the
> bank can mix providers and avoid pricing/quality risk.
> Most successful F500 banks in 2026 run **multi-provider**:
> Azure OpenAI for general workloads, Bedrock-Claude for long-
> context document analysis, self-hosted Llama for low-cost bulk
> tasks."

**SA-level twist**: bank regulators have specific concerns:
- **Model risk management (SR 11-7)**: every model needs validation,
  monitoring, change controls. LLMs are models. You need a
  validation framework, not just a vendor compliance attestation.
- **Third-party risk (FFIEC TPRM)**: vendor due diligence, exit
  rights, audit access. Both Azure and AWS provide these.
- **Data residency**: validate that the model's invocation stays
  in the bank's region; check the provider's regional availability.

The compliance posture is necessary but not sufficient. The bank's
MRM committee still needs to approve each use case, regardless of
provider.

</details>

### C3. Multi-cloud reality

Most F500s are multi-cloud in practice (primary cloud + Azure for AD + a "second cloud strategy"). Genuine multi-cloud applications are rare; multi-cloud bills are everywhere.

The honest take:

- **Pick a primary cloud.** Be deeply expert in it.
- **Know one other well enough** to be credible. (AWS engineers should know GCP; GCP engineers should know AWS; everyone should know Azure superficially because of OpenAI / Entra.)
- **Avoid building "cloud-agnostic" abstractions** at the platform layer unless you have a real reason. They cost real ergonomics and rarely pay off.

<details>
<summary><strong>F500 Q:</strong> A new CTO says "I want our ML platform to be cloud-agnostic." Argue back. What do you propose instead?</summary>

**In-depth answer**

**The argument-back**:

> "Cloud-agnostic sounds smart but rarely is. It almost always means
> we pay the cost of three abstraction layers and get the benefit
> of zero. Here's the alternative: pick a primary cloud, build for
> it deeply, but preserve optionality at the boundaries that matter."

**Why "cloud-agnostic" usually fails**:

1. **Lowest-common-denominator features**. You can't use SageMaker
   Pipelines if you also want it to work on Vertex. You strip out
   the best parts of every cloud to find the intersection.
2. **Three sets of skills**. Your team has to be expert in AWS *and*
   GCP *and* Azure for ML primitives. F500 hiring at that depth
   for three clouds is unrealistic.
3. **3x infrastructure cost**. Cloud-agnostic deployments often run
   in multiple clouds simultaneously. Triple the bill.
4. **Slower velocity**. Every change tested across three platforms.
5. **The abstractions break**. You eventually hit a feature that
   one cloud has and others don't (provisioned throughput on
   Bedrock, TPUs on GCP, ND H100 on Azure). The "agnostic" facade
   cracks.

**The 80% of cases where it's wrong**:

- Migration insurance: "what if AWS prices spike?" — they won't,
  market forces are strong. Multi-cloud insurance is more expensive
  than the hypothetical risk.
- Vendor lock-in fear: real, but addressed at the data layer, not
  the platform layer.
- Regulatory: "we need to keep data in-region" — that's a region
  choice, not a multi-cloud requirement.

**The 20% of cases where multi-cloud is right**:

- **Geographic coverage** — your customers are in a region one cloud
  doesn't serve.
- **Specific provider capability**: TPUs for one workload, Bedrock
  for another.
- **Specific compliance**: FedRAMP High variants differ across clouds.
- **Cost arbitrage at extreme scale** — hundreds of millions/year ML
  budget, where 10% savings via spot capacity across clouds matters.

**What to propose instead — "thoughtful single-cloud"**:

1. **Pick a primary cloud**. AWS is the F500 default; do it unless
   you have a strong reason otherwise.
2. **Be expert in it**. Use its full feature set.
3. **Preserve optionality at the data layer**:
   - **Open formats** (Parquet, Iceberg, Arrow) — no Snowflake-
     proprietary.
   - **Open ML formats** (ONNX, GGUF, MLflow's open registry).
   - **Thin abstraction at provider boundaries** — LiteLLM /
     Portkey at the LLM provider boundary lets you swap OpenAI for
     Anthropic for Llama without code change.
4. **Document a migration playbook annually**. Not as a plan, but
   as a discipline. The exercise of "what would moving look like"
   forces you to notice creeping lock-in.
5. **Limited multi-cloud where strategically warranted**:
   - Azure Entra ID + AWS for ML (most F500 banks)
   - GCP BigQuery + AWS for serving (some data-heavy orgs)
   - But the *applications* stay primarily in one cloud.

**The CTO conversation**:

> "I hear you. The risk you're trying to manage is real — vendor
> lock-in, pricing risk, capability risk. Cloud-agnostic isn't the
> answer; it's a costly hedge. Let me propose: AWS as primary,
> Azure for Entra and Office integration where unavoidable.
> Data layer in open formats so the underlying storage is portable.
> LLM gateway with provider abstraction so we can swap providers.
> Annual migration tabletop exercise. That gets us 80% of the
> optionality benefit at 20% of the cost."

**SA-level twist**: the architect's job in this conversation is to
*name the underlying risk* the CTO is trying to manage and propose
a cheaper, sharper way to address it. Multi-cloud is often a
non-technical demand from a CIO who got burned at a prior company
or read a Gartner report. Acknowledge the concern; reframe the
solution.

**Senior signal**: cite Dan McKinley's "Choose Boring Technology"
and the cost of each new technology in an ML platform context. New
clouds are a few of your "innovation tokens" — spend them where
they create competitive advantage, not where they create symmetric
abstraction overhead.

</details>

---

## Part D — DL-Specific Cloud Patterns

### D1. GPU choice across clouds in 2026

| Workload | NVIDIA | AWS custom | GCP custom |
|---|---|---|---|
| Training large LLM | H100, H200, B200 | Trainium 2 | TPU v5p |
| Training mid LLM / CV | A100, H100, L40S | Trainium | TPU v5e |
| Inference LLM heavy | H100, H200 | Inferentia 2 | TPU v5e |
| Inference LLM light / CV | L4, L40S, A10G | Inferentia 2 | TPU v5e |
| Edge / device | Jetson | — | Coral |

Pricing rough order: spot < on-demand < reserved. NVIDIA on-demand > Trainium / TPU on-demand > NVIDIA spot in many cases.

The decision: NVIDIA gives you the broadest software stack and best ergonomics. AWS custom and TPUs give you the lowest cost per FLOP if your model fits the supported software path. For LLM training the choice is increasingly about whether you can tolerate the silicon-specific software (Neuron SDK, JAX/XLA).

<details>
<summary><strong>F500 Q:</strong> You're planning a 12-month training program for a series of 13B-class models. Pick NVIDIA H100 vs Trainium 2. Walk through the trade-offs.</summary>

**In-depth answer**

**NVIDIA H100**:

- **Software**: PyTorch + CUDA + cuBLAS + cuDNN + NCCL — the most
  mature stack. Every paper's reference implementation runs on it.
- **Memory**: 80 GB HBM3 per GPU. 8 GPUs/node (`p5.48xlarge`) = 640 GB.
- **Compute**: ~989 TFLOPS FP16, ~1979 TFLOPS FP8 (sparsity-disabled).
- **Network**: NVLink 4 within node (900 GB/s), EFA-RDMA between
  nodes (3.2 Tbps on `p5.48xlarge`).
- **Cost**: `p5.48xlarge` on-demand ~$98/hr. Spot fluctuates; can be
  70% off but availability variable.
- **Ecosystem**: vLLM, TensorRT-LLM, FlashAttention-3 (Hopper-tuned).
  Every optimization shows up here first.

**AWS Trainium 2** (Trn2):

- **Software**: Neuron SDK. PyTorch via `torch_neuronx`; not full
  PyTorch ecosystem. JAX, TF, Megatron-LM via Optimum-Neuron.
- **Memory**: Trn2 instance (`trn2.48xlarge`) has 16 chips × 96 GB =
  1.5 TB of accelerator memory. Massive.
- **Compute**: Each Trainium 2 chip ~650 TFLOPS at FP16; aggregate
  ~10 PFLOPS per node — competitive with H100 node.
- **Network**: NeuronLink within node; EFA between nodes.
- **Cost**: roughly 30-40% lower $/effective-FLOP than equivalent
  H100 on-demand. The headline AWS pitch.
- **Ecosystem**: Hugging Face Optimum-Neuron supports common
  architectures (Llama, Mistral, T5). For custom architectures or
  bleeding-edge papers (Mamba, new attention variants), you wait
  for support or implement Neuron kernels yourself.

**For a 12-month 13B-class training program**:

**H100 wins on**:
- Time to first training run. Day 1, full PyTorch works.
- Iteration speed when you're experimenting with architecture
  variations (DPO, attention variants, MoE).
- Recovery from bugs / kernel issues — community support, public
  Stack Overflow.
- Compatibility with third-party tools (Hugging Face, vLLM, LangChain
  ecosystem).

**Trainium 2 wins on**:
- $/effective-FLOP for stable, well-defined training workloads.
- Memory per chip — 96 GB per chip is excellent for big-context
  training.
- Vendor lock-in flip side: AWS's roadmap commitment is to
  Trainium; pricing and availability incentives align.
- Combined with the AWS architecture (FSx, SageMaker HyperPod), the
  total managed-training experience is smoother than DIY H100.

**The recommendation depends on**:

| Factor | Pick H100 if | Pick Trainium 2 if |
|---|---|---|
| Custom architecture / research-heavy | ✓ | |
| Reference architectures (Llama, Mistral) | | ✓ |
| Team familiar with PyTorch/CUDA, not Neuron | ✓ | |
| Team willing to learn Neuron SDK | | ✓ |
| Budget cap is tight | | ✓ |
| Time-to-market is tight | ✓ | |
| Multi-cloud / portability concern | ✓ | |
| Already deep AWS commit | | ✓ |
| Need vLLM serving downstream | (training H100; serving could be either) | |

**A pragmatic 12-month plan**:

1. **Months 1-2**: Start on H100 for fast iteration. Validate
   architecture, data pipeline, training procedure end-to-end.
2. **Months 3-4**: Port training script to Neuron via Optimum-Neuron.
   Benchmark cost + throughput on Trn2.
3. **Months 5-12**: Bulk training runs on Trn2 for capacity-cost
   reasons; H100 reserved for experimentation and final fine-tuning
   passes.

**SA-level twist**: this hybrid pattern is what many F500 LLM teams
in 2026 actually do. The pure H100 path is "fastest but most
expensive"; the pure Trn2 path is "cheapest but slow to learn." The
hybrid splits the difference and de-risks vendor concentration.

**Senior signal**: mention that Trainium is also competitive with TPUs
on $/effective-FLOP for some workloads. If your team is comfortable
with non-NVIDIA accelerators (JAX/XLA experience), then GCP TPU v5
is another credible option in the same architectural conversation —
not for AWS's strategic reasons, but for capability.

</details>

### D2. Multi-node training networking

The non-obvious part: between-node bandwidth dominates training scaling.

- **NVLink** within a node (8 GPUs in `p5.48xlarge`): ~900 GB/s aggregate.
- **EFA + RDMA** between `p5.48xlarge` nodes: 400 Gbps per node.
- **InfiniBand** in HPC clouds (Lambda, CoreWeave): 800 Gbps per node common.
- **Standard ethernet** between random instances: 10–25 Gbps.

NCCL (NVIDIA Collective Communications Library) is the protocol; it automatically picks the fastest available transport. `NCCL_DEBUG=INFO` shows what it picked.

For multi-node training to scale, you need:

1. Instances in the same cluster placement group.
2. EFA enabled (`elastic-fabric-adapter-installer`).
3. Container with EFA + libfabric + NCCL plugin.
4. Pod spec requesting `vpc.amazonaws.com/efa` resource (on EKS).

If any layer is wrong, your training run will silently fall back to TCP and run at 1/10th throughput.

<details>
<summary><strong>F500 Q:</strong> Your two-node H100 training run gets 50% throughput of a one-node H100 run. List four diagnostic checks in order.</summary>

**In-depth answer**

This is a classic "scaling efficiency" problem. 50% on 2 nodes means
your distributed setup is poorly optimized — could be many things.

**The four checks in priority order**:

**Check 1 — NCCL is using EFA, not TCP**:

`NCCL_DEBUG=INFO` in environment. In the logs at job start, look for:

```
NCCL INFO NET/AWS Libfabric/0/EFA-rdma-write
```

If instead you see:

```
NCCL INFO NET/Socket : Using [0]eth0:...
```

You've silently fallen back to TCP. Possible causes:
- EFA not enabled on instance (verify with `fi_info -p efa`).
- EFA software stack not installed in container (`aws-ofi-nccl`
  plugin missing).
- Pod doesn't have `hostNetwork: true` (required for EFA in K8s).
- Wrong protocol — `NCCL_PROTO=simple` if you have issues.

**Fix**: install EFA + aws-ofi-nccl in container; enable
`hostNetwork`; set `FI_PROVIDER=efa`.

**Check 2 — Instances in the same cluster placement group**:

```bash
aws ec2 describe-instances --instance-ids ... \
  --query 'Reservations[].Instances[].Placement.GroupName'
```

If they're in different placement groups (or no placement group),
they may be on physically distant racks. Multi-rack TCP latency
~100µs vs intra-rack ~5µs; for all-reduce of GBs of gradients, this
adds up.

**Fix**: launch in a single cluster placement group. If `Insufficient
Capacity` errors, retry; AWS reserves cluster placement capacity
on a best-effort basis. ML Capacity Blocks help here.

**Check 3 — Batch size and gradient sync pattern**:

Run with `NCCL_DEBUG_SUBSYS=COLL`. Look at the all-reduce time per
step.

- If communication time is comparable to compute time, you're
  bandwidth-bound — you scale poorly.
- Solutions:
  - **Gradient accumulation**: do N micro-batches, then all-reduce
    once. Reduces sync frequency.
  - **Per-step batch size**: increase per-device batch to push the
    compute/communication ratio toward compute.
  - **FSDP `BACKWARD_PRE` vs `BACKWARD_POST` reshard policy** —
    overlapping comm with compute.
  - **`gradient_as_bucket_view=True`** in DDP — reduces memory
    fragmentation that slows comm.

**Check 4 — DataLoader and CPU/IO bottleneck**:

`nvidia-smi dmon -s u -d 1` showing GPU utilization sawtoothing
across all GPUs simultaneously? That's DataLoader starvation, not
distributed. Both nodes are stalling waiting for data.

- Check `iostat -xz 1` on each node — if disk is saturated, you've
  exceeded your local disk's throughput.
- Increase `num_workers`, `prefetch_factor`, use FSx Lustre or local
  NVMe cache.
- For multi-node, ensure data is *equally* fast on both nodes
  (one node reading from FSx, the other from S3 directly = the slower
  one bottlenecks).

**Other checks (5-N)**:

- **CUDA driver version mismatch** between nodes.
- **NCCL_TOPO_FILE** — explicitly set if NCCL is mis-detecting topology.
- **NCCL_SOCKET_IFNAME** — wrong NIC selected.
- **MTU mismatch** — non-9001 MTU on EFA links cuts performance.
- **Memory pressure** causing host-to-device transfers — verify with
  `free -h` no swap activity.
- **CPU governor** — set to `performance` not `powersave`.

**The diagnostic order matters**:

1. Verify EFA (90% of "bad scaling" cases — silent TCP fallback).
2. Verify placement (5%).
3. Verify communication/compute ratio (3%).
4. Verify DataLoader (2%).

**SA-level twist**: F500 ML platform teams build this as a *startup
diagnostic*. Every training job at startup logs a "scaling health"
report: detected NCCL transport, all-reduce bandwidth from a
benchmark, dataloader throughput. Catches misconfigurations before
the engineer wastes a week debugging.

**Senior signal**: mention `nccl-tests` (the canonical benchmark) and
that any new cluster should pass it (≥ 90% of theoretical bandwidth)
before being declared training-ready. If your nccl-tests numbers are
bad, no amount of framework tuning fixes it.

</details>

### D3. High-throughput storage for training

When your dataset can't fit in node-local SSD and you need many GPUs reading it simultaneously, you need a shared file system with high aggregate throughput.

Options:

- **FSx for Lustre** (AWS) — parallel file system; up to ~25 GB/s per file system; backed by S3 for durability. Mount on EKS via CSI.
- **GCS Fuse** — mount GCS as filesystem; OK for read-heavy, not POSIX-strict.
- **Filestore HPC** (GCP) — Lustre-equivalent on GCP.
- **WekaFS** — third-party; cross-cloud; very fast.
- **Lakehouse direct** (Iceberg over S3) — works fine for streaming reads with proper sharding.

Patterns that work for DL training:

- **WebDataset** sharded tar files in S3 + multi-worker DataLoader = scales well without a parallel FS.
- **Local SSD cache** — read once from object store at job start; subsequent epochs from local NVMe.
- **FSx Lustre** for genuinely random-access reads on multi-TB datasets.

<details>
<summary><strong>F500 Q:</strong> You're fine-tuning a 70B model on 8 nodes × 8 H100. Dataset is 3 TB of tokenized parquet. Where does the data live, in what format, and how do you ensure all 64 workers read at sufficient throughput?</summary>

**In-depth answer**

**The setup**: 70B fine-tune at, say, BF16 with FSDP. 64 H100s
collectively need ~2-5 GB/sec aggregate token throughput at decent
batch sizes. 3 TB of parquet must be readable in this pattern.

**Where the data lives**:

1. **Source of truth**: S3 (e.g., `s3://ml-data-prod/llm-finetune/v3/`).
   Tokenized parquet partitioned by shard (e.g., 1024 shards of
   3 GB each).
2. **Hot path**: FSx for Lustre, S3-imported, mounted to all 8 nodes.
3. **Local cache**: instance-local NVMe on each `p5.48xlarge` (30+ TB
   per node). Optional but worthwhile for multi-epoch training.

**In what format**:

For LLM token data specifically, the modern best practice is:

- **Tokenized arrow / parquet shards** (~1-3 GB each), pre-tokenized
  with the *exact* tokenizer (and revision hash) the model uses.
- **MosaicML Streaming format** (`.mds` shards) — better than vanilla
  parquet for LLM training. Built-in deterministic shuffling,
  resumability, sharding by global step. Becoming the 2026 standard.
- Alternative: **WebDataset .tar** — also good; less LLM-specific.

**The 64-worker read pattern**:

```
                S3 (source of truth, 3 TB)
                       │
                       ▼  (one-time hydration; FSx auto-imports)
            FSx for Lustre (3 TB persistent)
              │
              ├──► Node 1 (8 workers)
              ├──► Node 2 (8 workers)
              ...
              └──► Node 8 (8 workers)

Each worker:
  - Reads disjoint shard URLs (computed deterministically from rank +
    epoch + seed)
  - Streams shards sequentially from FSx mount
  - Optionally caches to local NVMe after first epoch
```

**Throughput math**:

- FSx for Lustre persistent SSD: ~250 MB/s per TB provisioned baseline
  + burst credits, scalable up to multiple GB/s aggregate. With 3 TB
  of data on a 9.6 TB FSx file system, aggregate throughput is ~2-4
  GB/s sustained — sufficient for 64 workers.
- Each worker reads ~30-60 MB/s. Total ~3 GB/s. Within FSx capacity.

**Ensuring even read throughput**:

1. **Disjoint sharding per rank**:
   ```python
   shards = list_of_all_shard_urls
   my_shards = shards[global_rank :: world_size]  # interleaved
   ```
2. **Pre-shuffle within shards at write time** — shards are
   sequentially read; no in-memory shuffle needed. The shuffle is
   in shard *order* (per-epoch) and in shard *contents* (precomputed).
3. **Use `prefetch_factor`** so workers prefetch next shard while
   current one is being consumed.
4. **MosaicML Streaming or HuggingFace `datasets` with `IterableDataset`**
   — both stream from object storage with proper buffering.

**Why not stream from S3 directly**:

You can. Pros: free, simplest. Cons: harder to predict throughput
across 64 workers; S3 ListBucket / GetObject latency adds tail
latency to data loading. For a 3 TB dataset and many epochs, FSx
amortizes hydration cost.

**For genuinely random access** (rare in LLM training, but if you
need it):

- DuckDB or in-memory parquet on each node — load shard, query,
  discard. Only viable if shards fit in memory.
- Skip FSx; use S3 Express One Zone for low-latency S3 random
  access.

**SA-level twist**: the resumability pattern matters. With Spot
training, a node interruption needs to:
1. Detect interruption (Spot termination notice).
2. Save model + optimizer + dataloader state (which shards consumed,
   which were in-flight).
3. New node spins up; FSx persists across so data path is unchanged.
4. Resume from same step number; same data order.

MosaicML Streaming and DDP `DistributedSampler` with `set_epoch` give
you this. Roll-your-own dataloaders rarely handle resumability
correctly.

**Senior signal**: mention `aws s3 cp --recursive` is the wrong way to
hydrate 3 TB — use `s5cmd` (10x faster, parallel) or FSx's native
S3 import.

</details>

### D4. LLM serving deployment on a cloud

For a self-hosted LLM (vLLM) on EKS, the typical setup:

- Container image with vLLM + the model weights (or weights mounted via FSx / S3 sync).
- Pod requests `nvidia.com/gpu: 1` (or more, with tensor parallelism).
- Service exposes the vLLM OpenAI-compatible API.
- Horizontal Pod Autoscaler on a custom metric (queue depth, tokens per second).
- Karpenter provisions GPU nodes as needed.
- VPC endpoint for ECR pull to avoid NAT cost on image pulls.
- ALB or NLB in front; ACM cert; WAF if internet-facing.
- CloudWatch metrics for tokens/sec, TTFT, P95, GPU utilization.

The cost trap: leaving min replicas > 0 on a GPU-class endpoint. A `g5.2xlarge` (1 × A10G) is roughly $1.20/hr — $864/month per replica. For internal-only workloads tolerating cold starts, scale-to-zero with KServe or Knative.

<details>
<summary><strong>F500 Q:</strong> Walk through everything you'd configure to deploy a vLLM-served Llama-3.1-70B on EKS for 200 internal users at sub-2-second TTFT, with cost as a first-class concern.</summary>

**In-depth answer**

**The architecture**:

```
                 Users (internal, ~200)
                       │
                       ▼
              [Internal ALB]
                       │
                       ▼
              [vLLM service in EKS]
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Pod replica 1     Pod replica 2
          1× `p4d.24xlarge`  (HA spare)
          (8× A100-80GB)
                       │
                       ▼
              [S3: model weights, INT8 quant]
```

**Capacity sizing**:

- Llama-3.1-70B at INT8 (AWQ or GPTQ): ~70 GB weights. Fits on a
  single 80 GB GPU with FP16 KV cache — barely. Better: tensor-
  parallel across 4 GPUs (one node has 8 × A100; use 4 for TP, leave
  4 for the second replica).
- 200 users, typical 10 messages/day each = 2000 messages/day = ~25
  RPS peak. With ~500 token avg output and continuous batching,
  one TP=4 deployment handles this comfortably.
- TTFT target < 2s: A100 + vLLM achieves ~500-1000ms TTFT for 70B
  INT8 with batched prompts up to a few K tokens. Meets target.

**Instance choice**:

- **`p4d.24xlarge`** (8× A100-80GB) — sweet spot for 70B INT8 with TP.
  ~$32/hr on-demand. 1-year Savings Plan brings to ~$22/hr.
- Alternative: **`p5.48xlarge`** (8× H100) — faster TTFT, ~3x cost.
  Overkill for 200 internal users.
- Two replicas (one per node) for HA. = $64/hr on-demand, $44/hr
  with SP. ~$32K/month with SP. Real money — make sure the use case
  justifies it.

**EKS deployment YAML** (skeleton):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama70b
  namespace: ml-prod
spec:
  replicas: 2
  selector: { matchLabels: { app: vllm-llama70b } }
  template:
    metadata:
      labels: { app: vllm-llama70b }
    spec:
      serviceAccountName: vllm-sa  # IRSA for S3 access to weights
      nodeSelector:
        node.kubernetes.io/instance-type: p4d.24xlarge
      tolerations:
        - key: nvidia.com/gpu
          value: "true"
          effect: NoSchedule
      containers:
        - name: vllm
          image: 123.dkr.ecr.us-east-1.amazonaws.com/vllm:v0.6.3-llama70b-int8
          args:
            - "--model=/models/llama-3.1-70b-int8"
            - "--tensor-parallel-size=4"
            - "--quantization=awq_marlin"
            - "--max-model-len=8192"
            - "--gpu-memory-utilization=0.92"
            - "--enable-prefix-caching"
            - "--max-num-batched-tokens=8192"
            - "--max-num-seqs=128"
          resources:
            limits:
              nvidia.com/gpu: 4
          volumeMounts:
            - name: model-cache
              mountPath: /models
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180  # 70B model load takes minutes
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 240
      volumes:
        - name: model-cache
          emptyDir:
            medium: Memory  # tmpfs for fast load (or PVC if persistent)
```

**Network**:

- Internal ALB (Kubernetes Service of type LoadBalancer with
  `service.beta.kubernetes.io/aws-load-balancer-internal: "true"`).
- VPC endpoint for S3 (free) so model weight pulls don't hit NAT.
- ACM cert; TLS terminates at ALB.

**Observability**:

- Prometheus scrape `/metrics` endpoint vLLM exposes (TTFT, tokens/s,
  KV cache utilization, queued requests, request errors).
- OpenTelemetry traces tagged with model_version, tenant.
- CloudWatch + AMP via OTel collector DaemonSet.
- Grafana dashboard: TTFT P50/P95/P99, tokens/s, queue depth,
  GPU memory, errors.

**Cost optimization moves**:

1. **Quantize to INT4 if quality is acceptable** — would let you serve
   on a single `p4d.24xlarge` (TP=2 across 2 GPUs); halve cost.
   Validate quality on gold set first.
2. **Use Savings Plans** — 1-year covers baseline; ~30% off.
3. **Skip HA for internal-only** — single replica saves another 50%
   if users tolerate brief outages during maintenance.
4. **Schedule scale-down off-hours**. 200 internal users probably
   don't use it weekends. Cron-based scale to zero Sat/Sun.
5. **Consider `g6e.48xlarge`** (8× L40S, 48GB ea) at ~$16/hr — for
   INT4 70B with TP=8, possibly viable. Half the cost of p4d.
6. **Multi-LoRA strategy** — if you have multiple Llama 70B fine-
   tunes, serve as adapters on one base model. One $32/hr cluster
   serves dozens of variants.

**Failure modes to instrument**:

- KV cache exhaustion → reject new requests until cache frees.
- Single GPU node failure → traffic auto-routes to second replica;
  EKS reschedules.
- Model load failure on startup → pod fails readiness probe; ALB
  doesn't route.
- Slow client connections monopolize compute → set client request
  timeout, request concurrency limit per IP.

**SA-level twist**: at 200 users / 25 RPS the *Bedrock Llama 3.1 70B*
option is genuinely competitive: ~$0.00265 input + $0.0035 output
per 1K tokens, no infra. At average 700 tokens per request and 25
RPS, monthly Bedrock cost ≈ 25 × 60 × 60 × 24 × 30 × 700/1000 ×
$0.003 ≈ $36K/month. Comparable to your self-hosted setup, with
zero ops. The break-even on self-hosting is somewhere around 100
RPS sustained. Under that, Bedrock often wins on total cost of
ownership.

**Senior signal**: explicitly compare against Bedrock; don't just
assume "self-host wins." Run the math, ship a decision document
with the trade-offs.

</details>

### D5. Cost optimization for GPU workloads — the FinOps playbook

In order of impact:

1. **Quantize.** INT4 (AWQ/GPTQ for LLM) cuts memory ~4x and often increases throughput; INT8 for CV is standard via TensorRT.
2. **Right-size.** L4 instead of A100 for small inference. T4 instead of A100 for tabular DL.
3. **Spot for training.** With checkpointing every N minutes, you tolerate preemption.
4. **Scale-to-zero** where cold starts are tolerable.
5. **Multi-LoRA serving.** One base model, dozens of adapters, one set of GPU instances.
6. **Continuous batching.** Throughput multiplier for LLM.
7. **Prefix caching.** Free wins for shared system prompts and RAG contexts.
8. **VPC endpoint for ECR + S3.** Stop paying NAT egress for image pulls and dataset reads.
9. **Reserved capacity** for steady-state workloads. 30–60% off list.
10. **Compress logs + lifecycle to cold storage.** ML logs are voluminous.

<details>
<summary><strong>F500 Q:</strong> Your cluster spends 60% of GPU-hours below 30% utilization. Walk through your investigation: is this a workload-shape problem, a scheduling problem, or both?</summary>

**In-depth answer**

**The metric to start with**: DCGM `DCGM_FI_DEV_GPU_UTIL` aggregated
by pod and node, plotted as a histogram per day.

You're told 60% of GPU-hours are at < 30% util. Three categories of
cause:

**Category 1: Workload-shape problem**

The job is *inherently* low-utilization. Common cases:

- **CPU/data-loading bound** — small models, slow data pipeline.
  GPU sits idle waiting.
- **Communication-bound multi-node training** — gradient sync time
  exceeds compute time; GPU stalls.
- **Misconfigured batch size** — too small to saturate the GPU.
- **Notebook + Streamlit "demos"** — a GPU instance serving a
  Jupyter kernel with one user; 99% idle.
- **Inference at QPS << capacity** — endpoint at 1 RPS on an A100
  built for 50.

**Diagnostic**:
```promql
avg by (pod) (
  rate(DCGM_FI_DEV_GPU_UTIL[5m]) +
  rate(DCGM_FI_DEV_MEM_COPY_UTIL[5m])
) < 0.3
```
For each low-util pod: is it training (look at process), inference
(traffic patterns), or a notebook (long-lived idle kernel)?

**Fix**:
- Right-size: move from p4d/p5 to g6/g6e for inference; from H100
  to A100 if H100 utilization < 50%.
- Quantize models for inference.
- Co-locate small inference workloads via MIG (Multi-Instance GPU)
  or NVIDIA MPS (Multi-Process Service).
- Kill or auto-stop idle notebooks (Lambda + EventBridge).

**Category 2: Scheduling problem**

The workload could be high-util but the cluster scheduler isn't
packing it well:

- **GPU fragmentation** — a node has 4 free GPUs but the next pod
  needs 8 on one node; the pod sits Pending while the GPUs are
  unused.
- **Resource reservations too coarse** — pods reserve 1 GPU but use
  10%; can't co-schedule.
- **No gang scheduling for multi-node** — partial allocations bind
  GPUs that wait for their peers.
- **Spot interruptions** — pod evicted, restarted, but Karpenter
  hasn't provisioned the next node yet; the running pod's peers
  wait.

**Diagnostic**:
- `kubectl describe nodes` showing GPU allocation per node — find
  nodes with idle GPUs but Pending pods elsewhere.
- Look for fragmentation: nodes with 1-2 free GPUs each, but a
  pending pod wants 4 on one node.
- Check scheduler: is it Kubernetes default, Volcano, Kueue,
  Yunikorn? Default scheduler is bad at gang scheduling.

**Fix**:
- **Volcano** or **Kueue** for gang scheduling.
- **Karpenter** for fast node provisioning matched to pending pods.
- **Bin-packing scheduler hints** (resource priorities, taints).
- **Bigger nodes for multi-GPU jobs** so 8-GPU pods schedule on
  single nodes without fragmentation.

**Category 3: Often both**

In practice, low utilization is usually a combination:

- Many small workloads + no co-location = workload-shape *and*
  scheduling.
- Mix of training and inference on same cluster = scheduling
  imbalance.
- "We have GPUs reserved for X team" = social problem manifesting
  as scheduling problem.

**Investigation order**:

1. **Pareto by pod**: top 20% of low-util pods explain 80% of
   wasted hours. Start there.
2. **For each top-waster**: is it workload-shape or scheduling?
3. **Aggregate**: how many fall into each bucket? Tells you which
   intervention has the bigger ROI.

**Common findings at F500**:

- 30-40% are notebook / dev workloads that should auto-stop.
- 20-30% are inference endpoints on overspec hardware (right-size).
- 10-20% are training jobs with bad data pipelines (fix DataLoader).
- 10-15% are scheduler fragmentation.
- 5-10% are legitimate "must reserve capacity" workloads.

**SA-level twist**: the *cultural* problem matters as much as the
technical. Engineers hate having their notebooks stopped or their
"reserved" GPUs reclaimed. Phase the program: first showback
(visibility), then quotas (limits), then chargeback (financial
accountability). Skip a phase and you fail politically.

**The 90-day plan**:

- Days 1-14: instrument (DCGM + Prometheus + Grafana panel).
- Days 15-30: auto-stop idle notebooks. Quick win.
- Days 30-60: right-size inference endpoints. Real money.
- Days 60-90: deploy Volcano / Kueue for multi-node training jobs.

**Senior signal**: mention NVIDIA MIG (Multi-Instance GPU) — A100/H100
can be partitioned into smaller schedulable instances. For mixed
inference + training clusters, MIG lets one A100 serve 7 distinct
inference services concurrently with isolation. Underused in most
F500 clusters.

</details>

---

## Part E — Capstone Scenarios

### Scenario E1 — Greenfield DL platform on AWS

You're hired into a F500 that's done some MLOps but never DL at production scale. Mandate: stand up a DL platform that can support 5 model teams. Budget: $1.5 M / year. Timeline: 12 months.

Sketch the architecture, the AWS service choices, the team org, the cost model, the migration sequence, and the operational concerns. Use the question prompts above as the surface area you cover.

### Scenario E2 — Migrate from SageMaker to self-hosted EKS

Your team is currently fully on SageMaker. Cost is exploding (model serving + notebook hours). You suspect 40% can be saved by moving to EKS + KServe + vLLM. Design the migration: ADRs, parallel-run validation, decommission criteria, training the team on EKS, the new on-call rotation, the cost model proof.

### Scenario E3 — Multi-region LLM serving for global enterprise

You serve an internal LLM platform to a global F500 (users in US, EU, APAC). Requirements: P95 TTFT < 1 second from any region, data residency for EU (no traffic leaves EU), failover within 60 seconds of regional outage, per-tenant cost attribution. Walk through architecture across regions, model artifact distribution, gateway routing, observability, and the operational model.

### Scenario E4 — Train and serve a CV model fleet on a budget

You're at a smaller F500. You need to train and serve 12 CV models (image classification, object detection, semantic segmentation) across product teams. Total budget: $200K / year for cloud. Pick the cloud, the architecture, the training rhythm, the serving stack. Show the math.

---

## How to Use This Chapter for Interview Prep

1. **Cover Part A in one session.** Master each "F500 Q" before moving on. These are the universal foundations.
2. **Do Part B over 4–5 sessions.** AWS depth is a long game. After each subsection, answer the F500 Q's aloud.
3. **Skim Part C** unless you target a GCP-heavy or Azure-heavy F500.
4. **Hit Part D twice.** This is the DL-cloud intersection where most F500 senior interviews score you.
5. **Take Part E like a system-design round.** Time-box. Write notes. Sketch architecture. Then rehearse the verbal answer for 5 minutes.

In total, this chapter is about 8–12 hours of careful reading + practice. Done well, it closes the cloud gap for a DL engineer who already has the modeling background.

The remaining DL specializations — RAG architecture, fine-tuning factory, multi-tenant LLM platform, real-time CV serving — are covered in the practitioner and specialization chapters of this course. With the cloud vocabulary from this chapter, you can consume all of them at depth.


---

## Capstone Project — F500-Grade Greenfield AWS Account Topology + EKS DL Cluster

_Anchored on the section: **Part B — AWS in Depth**. The headline build that turns this chapter's knowledge into a Fortune 500 portfolio artifact._

### What you'll build

A complete F500-style AWS bring-up: (1) AWS Control Tower with 4 OUs + 4 accounts (Security, Shared Services, ML-Platform, Workload), (2) SCPs enforcing 'no GPU instances outside approved regions' and 'no public S3,' (3) EKS cluster in ML-Platform account with Karpenter + GPU Operator + IRSA, (4) cross-account S3 read from EKS pod to training-data bucket in Workload account, (5) vLLM serving Qwen2.5-7B-AWQ at 50 RPS with VPC endpoint for ECR, (6) CloudWatch + AMP + Managed Grafana, (7) CUR → Athena query showing per-tag cost, (8) one-page architecture diagram. Tear down nightly.

### Skills demonstrated

- AWS Organizations + Control Tower
- SCPs
- EKS + Karpenter + GPU Operator + IRSA
- cross-account IAM trust
- VPC endpoints for cost
- Managed Prometheus + Grafana
- CUR + Athena
- Bedrock vs self-hosted LLM trade-offs

### Tech stack

`AWS Control Tower + Organizations` · `Terraform 1.9+ for everything` · `EKS 1.30+ with Karpenter 0.37+` · `NVIDIA GPU Operator + EFA driver` · `vLLM 0.6+ on g5.2xlarge` · `Qwen2.5-7B-Instruct-AWQ` · `AMP (Managed Prometheus) + Managed Grafana` · `CUR → Athena → QuickSight (optional)`

### Acceptance criteria

- [ ] 4 AWS accounts under Control Tower with SCPs enforced
- [ ] EKS cluster up; vLLM serving at 50 RPS with measured P95 TTFT
- [ ] Cross-account S3 read via IRSA proven
- [ ] CUR shows per-tag cost in Athena
- [ ] Tear-down script saves >80% nightly cost (kill GPU node pool)
- [ ] Architecture diagram committed (drawio or Excalidraw)

### Fortune 500 talking point

> I have run a Control-Tower-managed multi-account AWS with EKS + GPU Operator + IRSA + vLLM + CUR for a real workload. Most candidates have touched maybe half of this. AWS-heavy F500 (most of them) interview score senior on this depth.

**Estimated time:** 30

**Stretch goal:** Add SageMaker HyperPod as a comparison training environment; document the operator-experience delta vs raw EKS.



---

## 📝 Your Notes

> Take 10 minutes after this chapter. Answer below and revisit weekly.

**What surprised me in this chapter?**

_(write here)_

**Three open questions I still need to answer:**

1.
2.
3.

**Code snippets, links, or portfolio references to remember:**

-
-
-

**Concepts I should re-explain aloud to verify understanding:**

-
-

---
