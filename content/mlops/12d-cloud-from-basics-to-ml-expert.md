# 12 — Cloud from Basics to ML Expert (DL-Focused) — Part 4 of 8: SageMaker & Bedrock (Part B, B7–B8)

This is part 4 of 8, continuing Part B (AWS in depth). Parts 2 and 3 covered account topology, IAM, VPC, EC2, S3, and EKS. Here we cover AWS's managed ML services: SageMaker (B7) for managed training and serving, and Bedrock (B8) for managed LLMs — including when each beats a self-hosted alternative.

---

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

---

## You can now

- Decide between a SageMaker Training Job and a raw `PyTorchJob` on your own EKS cluster, and name the break-even point in team size and control needed.
- Choose between a SageMaker endpoint, Bedrock, and self-hosted vLLM on EKS for serving a fine-tuned LLM, and justify the choice in cost, latency, and ops terms.
- Answer an FSI customer's "does Bedrock train on our data" question accurately, including the enterprise-terms nuance that changes the answer.
- Compare Bedrock Knowledge Bases against a DIY pgvector/OpenSearch RAG stack, and know when the managed option stops being worth the loss of control.
- Explain to a stakeholder why "just use SageMaker for everything" and "just self-host everything" are both wrong defaults — the right answer is workload-dependent.

## Try it

Take one use case — serving a fine-tuned Llama-3-8B to 50 internal employees — and write three one-paragraph pitches: one for a SageMaker endpoint, one for Bedrock, one for self-hosted vLLM on EKS. Each pitch must include a real cost estimate and a real latency number, not just a qualitative trade-off. Pick a winner, then write the two conditions under which you'd switch to a different option six months later.
