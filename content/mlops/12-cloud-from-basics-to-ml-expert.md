# 12 — Cloud from Basics to ML Expert (DL-Focused) — Part 1a: Cloud Foundations I (A1–A6)


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


## You can now

- Explain the cloud mental model, regions/AZs/edge, and pick placement for latency and resilience.
- Use Linux fundamentals in a cloud VM confidently (processes, permissions, networking tools).
- Reason about networking 101 (VPC, subnets, routes, security groups) and do CIDR math to carve a VPC into subnets.
- Trace the full network path of a request to a private service and spot where it would break.

## Try it

Take a `/20` VPC and split it into four balanced subnets — write out the CIDR ranges, usable host counts, and which are public vs private. Then draw the packet path for a browser request reaching a database that sits in a private subnet behind a load balancer, naming every hop (DNS → ALB → target → NAT/route → DB) and the one security-group rule that, if wrong, drops the connection.
