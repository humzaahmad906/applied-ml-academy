# 12 — Cloud from Basics to ML Expert (DL-Focused) — Part 6 of 8: GCP, Azure & Multi-Cloud Reality (Part C)

This is part 6 of 8 of the "Cloud from Basics to ML Expert" lesson. Parts 1–5 covered Part A (universal foundations) and all of Part B (AWS in depth). Here we cover Part C: what's meaningfully different on GCP (C1) and Azure (C2) for DL workloads, and the honest reality of multi-cloud strategy (C3).

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

## You can now

- Compare GCS + GKE + Vertex against S3 + EKS + SageMaker for a CV training pipeline, and name what's operationally different rather than just differently named.
- Advise a bank on Azure OpenAI vs. Bedrock-Anthropic for an FSI-compliant GenAI strategy, and defend the recommendation on compliance and ecosystem grounds, not brand preference.
- Push back credibly on a CTO's "make our ML platform cloud-agnostic" mandate, and propose what to build instead of an abstraction layer nobody asked for.
- Recognize which AWS-specific patterns from Part B (IAM, VPC, SageMaker/Bedrock) have direct GCP/Azure analogs and which don't transfer cleanly.

## Try it

Take one AWS-specific decision from Part B — the SageMaker vs. Bedrock vs. self-hosted vLLM choice from part 4 — and re-derive it for GCP (Vertex vs. self-hosted on GKE) and for Azure (Azure ML vs. Azure OpenAI vs. self-hosted on AKS). Write down which parts of your AWS reasoning ported over directly and which had to change because of a real platform difference, not just a naming difference.
