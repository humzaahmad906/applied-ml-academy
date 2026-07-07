# 12 — Cloud from Basics to ML Expert (DL-Focused) — Part 3 of 8: EC2, S3 & EKS for ML (Part B, B4–B6)

This is part 3 of 8, continuing the AWS deep dive (Part B) from part 2, which covered account topology, IAM, and VPC. Here we cover compute and storage for ML workloads: EC2 instance selection for training and serving (B4), S3 patterns that keep training from going I/O-bound (B5), and EKS as the Kubernetes layer for DL workloads (B6).

---

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

---

## You can now

- Pick the right EC2 instance family and count for a large training job (e.g., a 13B-parameter model) and justify the spot / reserved / savings-plan mix.
- Pick an inference instance type for a quantized LLM under a P95 TTFT budget, and explain why the biggest available GPU isn't always the right serving choice.
- Diagnose I/O-bound training on S3 (millions of small JPEGs at random keys) and fix it with prefix design, WebDataset shards, or FSx for Lustre.
- Design an S3 lifecycle policy for ML logs and artifacts that balances frequent-access, occasional-access, and archive tiers against cost.
- Configure IRSA for EKS pods and stand up a GPU node group with placement groups and EFA for multi-node training, reasoning about what it does to NCCL.

## Try it

Take a training job you're familiar with, or invent one — a 13B-parameter model on 8 nodes of 8×H100. Write out the exact EC2 instance type and count, the spot/RI/savings-plan mix and why, the S3 prefix/shard layout for the training data, and the EKS node group + placement group + EFA configuration you'd request. Then identify the single most expensive line item in your design and describe the one change that would cut it the most without hurting throughput.
