# 05 — Fortune 50 Portfolio Projects

Seven projects engineered to demonstrate the technical depth, architectural judgment, and production thinking that Fortune 50 MLOps / ML Platform teams hire for.

## Why "Seven" Doesn't Mean "Build Seven"

Build **two or three**, deeply. Not seven shallowly. A portfolio of half-finished projects telegraphs "I quit when things get hard."

The right strategy:

1. Pick **one project from a list of three "must do"** (below) — universally valued
2. Pick **one project from the remaining four** that aligns with your career direction
3. Optionally pick a third for adjacent breadth

## The Three "Must Do" Projects (Pick One)

Each alone anchors a senior MLOps interview at any F50.

1. **Project 1: Real-Time Anomaly Detection with Feedback Loop** — the closed-loop classical ML system
2. **Project 2: Production LLM Platform with RAG, Evals, and Routing** — the LLMOps system
3. **Project 4: ML Cost Crime Scene** — the senior-signaling cost audit

Pick one based on the direction you're heading. Projects 3, 5, 6, 7 are excellent but more specialized.

---

## Project 1 — Real-Time Anomaly Detection with Feedback Loop

### The Business Framing

Pick a domain with a stream of events where anomalies matter: financial transactions (fraud), logistics events (lost packages, wrong-warehouse routings), web traffic (DDoS, bot activity), IoT sensors (equipment failure). The pipeline doesn't just *detect* anomalies — it sustains a closed loop where analysts review flagged events, label them, and labels feed back to improve detection.

Most "fraud detection" projects stop at "send to a queue when something looks weird." That's the easy half. The interesting half: how do you keep the detector calibrated, measure false positive rate over time, handle concept drift?

### Tech Stack

- **Ingestion:** Synthetic event generator → Kafka (you build the generator with `faker` plus a configurable "fraud injection" rate)
- **Streaming features:** Flink (or Kafka Streams for simpler version) for windowed feature computation
- **Online store:** Redis or DynamoDB for feature serving
- **Offline store:** Iceberg or plain Parquet on S3/MinIO for training data
- **Model training:** XGBoost or LightGBM (start) → small Transformer (stretch)
- **Experiment tracking + registry:** MLflow
- **Serving:** KServe InferenceService with sub-50ms latency, autoscaling
- **Labeling UI:** Streamlit or a small NextJS app where analysts review flagged events
- **Feedback:** Labels → Kafka topic → consumed by retraining pipeline → new model registered + canary deployed
- **Orchestration:** Prefect or Airflow for retraining
- **Observability:** Prometheus + Grafana for system; Evidently for drift; custom dashboards for precision/recall over time
- **Everything on Kubernetes**, brought up with `make up`

### Architecture Sketch

```
[Event Generator] ──► [Kafka: raw_events]
                              │
                              ├──► [Flink: feature engineer]
                              │           │
                              │           ▼
                              │     [Online store: Redis]
                              │           │
                              │           ▼
                              │     [KServe: detector]   ──► [Kafka: alerts]
                              │                                    │
                              │                                    ▼
                              │                          [Streamlit labeling UI]
                              │                                    │
                              │                                    ▼
                              │                          [Kafka: labels]
                              │                                    │
                              ▼                                    ▼
                       [Iceberg: training data]            [Airflow: retraining]
                              │                                    │
                              ▼                                    ▼
                       [dbt marts]                          [MLflow Registry]
                              │                                    │
                              ▼                                    ▼
                       [Drift dashboard]                  [KServe canary deploy]
```

### The Interesting Technical Decisions

1. **Feature engineering: real-time vs batch.** Some features (transaction amount) are point-in-time. Others (transactions in last 24 hours for this user) require state. Flink's keyed state is the right home — but you have to design the state schema carefully because schemas evolve and state is hard to migrate. Document your approach.

2. **The cold start problem.** When you deploy a new detector, where do labels come from? You need a synthetic labeling phase and rules-based detection as a baseline. Implement both.

3. **Concept drift.** Fraud patterns change. Precision will drift even if the detector is "the same." You need monitoring that distinguishes detector degradation from environment change. Implement this and document it.

4. **Exactly-once vs at-least-once.** Be honest about which guarantee you're providing and where. False alerts from at-least-once delivery are usually acceptable; missed fraud is not.

5. **Online/offline consistency.** Show that the same feature computed online in Flink and offline in batch agrees. Build a reconciliation job.

### Acceptance Criteria

- 1000+ events/second sustained throughput, documented load test
- End-to-end latency (event → alert) under 100ms at P95
- A working labeling UI with at least 50 labeled events stored
- A retraining pipeline that runs nightly and deploys a versioned model with canary rollout
- Dashboards tracking: events/sec, alerts/sec, precision over time, recall over time, latency P50/P95/P99, GPU/CPU utilization
- A README that includes the trade-offs you made and what you'd build next
- Total cost: under $50/month at the demo throughput

### Stretch Goals

- A/B test two detectors in parallel — route a small % of traffic to the new one, compare metrics
- Add an "explain this alert" feature using SHAP
- Implement automatic rollback when precision drops below a threshold
- Online learning: gradient updates to the model as new labels arrive (much more advanced — read about it carefully before attempting)

### Interview Talking Points

- "Walk me through how you'd design a fraud detection system." — 30-minute answer.
- "How do you handle concept drift?" — You've dealt with it.
- "How would you operationalize an ML model in a streaming pipeline?" — Done this exactly.
- "How do you handle training-serving skew with streaming features?" — Concrete reconciliation story.

### Realistic Time Estimate

**10–14 weeks** at 10 hrs/week. Substantial. Don't underestimate the labeling UI — it's the part most engineers skip and it's the part that demonstrates production thinking.

---

## Project 2 — Production LLM Platform with RAG, Evals, and Routing

### The Business Framing

Every F50 has spun up dozens of LLM-powered features in the last two years. The bottleneck isn't the model — it's the platform around it: prompt management, retrieval infrastructure, evaluation harness, cost controls, observability, guardrails. Build that platform.

Your "tenant" use cases (you'll simulate three):

1. A customer-support chatbot grounded in product documentation
2. An internal Q&A bot over engineering docs
3. A structured-output extraction service (e.g., resume parser, invoice parser)

All three share the same platform; the platform shows how to scale to dozens.

### Tech Stack

- **API gateway:** LiteLLM or a thin FastAPI shim. Unified API across model providers.
- **Model backends:** OpenAI + Anthropic + Bedrock + a self-hosted vLLM serving an open-weights model (Llama-3.1-8B or Qwen-class)
- **Vector DB:** pgvector for one tenant, Qdrant for another (justify the choice in your README)
- **Reranker:** BGE-Reranker or Cohere Rerank
- **Embedding model:** OpenAI text-embedding-3-small for one path, BGE-large for another
- **Prompt registry:** simple — versioned prompts in a Postgres table with stage promotion (dev / staging / prod), or a tool like Pezzo
- **Evaluation harness:** Braintrust or Langfuse (or your own Python harness)
- **Observability:** Langfuse for traces; Prometheus + Grafana for system metrics
- **Cost tracking:** per-tenant, per-request, per-model token counts and dollar costs
- **Guardrails:** input filtering (prompt injection detection, PII), output filtering (toxicity, schema validation)
- **Caching:** semantic cache via embeddings + exact-match cache via Redis

### Architecture Sketch

```
[Client]
   │
   ▼
[FastAPI / LiteLLM gateway]
   │
   ├──► [Input filters (PII, prompt injection)]
   │
   ├──► [Router: pick model based on tenant + complexity]
   │
   ├──► [Retrieval: tenant-scoped vector + BM25 → rerank]
   │
   ├──► [Cache check (exact + semantic)]
   │
   ├──► [Prompt assembly from registry]
   │
   ├──► [LLM call (OpenAI / Anthropic / Bedrock / self-hosted vLLM)]
   │
   ├──► [Output filters (schema validation, toxicity)]
   │
   ├──► [Log to Langfuse + emit metrics + cost attribution]
   │
   ▼
[Response]
```

### The Interesting Technical Decisions

1. **Routing strategy.** When to use a $30/M-token model vs a $0.50/M-token model vs your self-hosted free model. Build it as a function of (tenant tier, task type, content length, recent failure rate). Document the policy.

2. **Hybrid retrieval architecture.** Vector + lexical + reranker. Tune k1 (vector top-K), k2 (BM25 top-K), and reranker top-K for each tenant. Show a recall@10 curve.

3. **The evaluation harness.** Build it for at least one tenant: a gold set of 100+ examples, automated LLM-as-judge eval, pairwise comparison between prompts/models, regression tests in CI. The harness is often the most impressive part.

4. **Cost as a first-class metric.** Every response logs tokens in, tokens out, cost. Daily dashboard shows cost per tenant, cost per request type, cost trends. Per-tenant rate limits configurable.

5. **Self-hosted model lifecycle.** Show vLLM serving with continuous batching, prefix caching, and the right quantization. Document P50/P95 latency and tokens/second under load.

6. **Prompt versioning and rollback.** Promote a new prompt; canary 10% of traffic; roll back if eval scores or business metrics regress.

### Acceptance Criteria

- Three simulated tenants, each with its own corpus, prompts, and use case
- A self-hosted vLLM endpoint serving an open-weights 7B-class model with at least 300 tokens/sec/replica
- An evaluation harness with at least 100 labeled examples for one tenant; automated comparison of 3+ prompt variants
- Cost dashboard with daily attribution per tenant
- Sub-2-second end-to-end latency (P95) for a typical RAG request including retrieval + LLM call
- A working semantic cache (demonstrate the cost saving)
- Guardrails enforced — show input that gets blocked
- A README documenting the routing policy, the eval methodology, and the cost trade-offs

### Stretch Goals

- Fine-tune the self-hosted model on synthetic data from the strong model (distillation). Compare quality and cost.
- Add an agent layer: tool use, multi-turn planning, tool-call safety.
- Multi-modal: add an image input path.
- Multi-region: route requests to nearest model region; failover when one is down.

### Interview Talking Points

- "How would you build an LLM platform for an enterprise?" — Detailed answer.
- "How do you evaluate an LLM application?" — Concrete harness with methodology.
- "Walk me through your RAG architecture." — Specific, opinionated.
- "How do you control LLM cost at scale?" — Cost dashboard, routing, caching, quantization.

### Realistic Time Estimate

**10–14 weeks** at 10 hrs/week. Worth every hour. As of 2026 this is the most-asked-for skill set in F50 ML hiring.

---

## Project 3 — Multi-Tenant Feature Store

### The Business Framing

You're building the feature platform for a 50-team ML org. Each team has its own features, but the platform centralizes definitions, governance, freshness SLAs, and lineage. This is the architecture pattern that large-scale ML orgs converge on. Building one convincingly is rare in a portfolio and instantly differentiating.

### Tech Stack

- **Definitions:** Python (Feast-like) — declarative entities, features, sources
- **Storage — offline:** Iceberg tables on S3, partitioned by entity and time
- **Storage — online:** Redis or DynamoDB, with a clear write path from offline materialization + a streaming write path from Flink
- **Catalog:** a simple Postgres metastore + a web UI (Streamlit or NextJS)
- **Quality:** Great Expectations or Soda on every source table; row-level freshness SLAs
- **Lineage:** OpenLineage events flowing into a small graph DB or Postgres
- **Streaming:** Flink for real-time features; Kafka as the source
- **Compute (offline):** Spark on Kubernetes for materialization
- **API:** gRPC + REST for online reads (sub-10ms P99)

### The Interesting Technical Decisions

1. **Point-in-time correctness.** This is the project's headline feature. Implement temporal as-of joins that scale (Iceberg's metadata + time-bucketed partitioning makes this practical).

2. **Online/offline consistency.** Implement a parity job — for a sample of features, recompute online values from offline storage and assert equality. Alert on drift.

3. **Multi-tenant isolation.** Three angles:
   - Logical (namespace per team)
   - Storage (separate Iceberg databases / Redis logical DBs)
   - Compute quotas (limit how much one team can blow up materialization)
4. **Discovery.** The UI lets engineers browse features across teams. A team should be able to find "is there a `customer_lifetime_value` feature already?" before defining their own.

5. **Schema evolution.** When a feature's schema changes, the offline store needs migration; the online store needs careful handling. Implement and document.

6. **Cost attribution.** Per-team materialization cost. Per-team online read cost. Per-team storage cost.

### Acceptance Criteria

- At least 3 entities, 15 features, 4 feature views
- Both batch and streaming features
- Sub-10ms P99 online read latency
- A web UI for browsing
- A parity job that runs nightly and reports drift
- Multi-tenant test: prove that team A can't read team B's features without permission

### Stretch Goals

- A "feature retirement" workflow — deprecate a feature, alert downstream consumers, eventually remove
- Time travel: query feature values as of any historical timestamp
- Embedding feature support (variable-length tensors)
- Automatic feature transformation discovery (suggest features from raw data)

### Interview Talking Points

- "How would you build a feature store?" — Walk through your decisions.
- "How do you handle training-serving skew?" — Concrete architecture answer.
- "How do you handle online/offline consistency?" — You built the parity job.

### Realistic Time Estimate

**10–14 weeks** at 10 hrs/week.

---

## Project 4 — ML Cost Crime Scene

### The Business Framing

The most senior thing an MLOps engineer does isn't building new platforms. It's keeping the existing one from bankrupting the company. This project frames you as a "ML cost archaeologist" — you've inherited a poorly-designed ML platform and your job is to make it 60% cheaper without breaking anything.

You build both halves yourself: first the deliberately-bad platform, then the audit, then the refactor. Document everything as if it were a real consulting engagement.

### Why This Project Wins Interviews

Most candidates talk about projects they built. Almost no one talks about projects they fixed. The latter signals seniority. F50 hiring managers are *desperate* for engineers who think about cost — every company is panicking about GPU bills.

### Tech Stack

- A cloud account (AWS, GCP, or Azure) with a real ML stack: SageMaker / Vertex / Databricks
- A few synthetic training pipelines that run on a schedule
- A serving stack with several models
- A workload generator that calls the models on a schedule

### The "Crime Scene" — What You Build First

A stack with deliberate but realistic anti-patterns:

1. **A 70B-parameter LLM endpoint on an A100 left running 24/7 at <10% utilization**
2. **Training jobs that train from scratch every night** instead of incrementally
3. **SageMaker notebooks left running** for "monitoring" — really billing hours
4. **HPO with no early stopping** — 200 trials when 50 would suffice
5. **Models running on GPU when they could run on CPU** at 5x cheaper for same latency
6. **Per-request feature pulls from a remote feature store** instead of caching
7. **A daily full retraining** when an incremental update would suffice
8. **Logs and prediction records stored uncompressed** at hot tier
9. **Cross-region data transfer** because the training cluster is in `us-east-1` and the data is in `us-west-2`
10. **Endpoint autoscaling minimums set to 4 replicas** — pays for 4 instances at 3am

Run this for a week. Document the baseline: total cost, top 10 most expensive line items, GPU utilization distribution.

### The Refactor

For each anti-pattern, write a section:

1. **The pattern:** What's wrong
2. **The cost:** Quantify it (instance hours, $)
3. **The fix:** Specific change (quantize the model, switch to T4, enable scale-to-zero, ...)
4. **The risk:** What could break
5. **The validation:** How you proved the fix worked

Re-run for another week. Document the new baseline. Target: 60% cost reduction.

### Specific Quantitative Wins You Should Hit

| Anti-pattern | Fix | Typical saving |
|---|---|---|
| LLM on A100 underutilized | Move to L4 / A10, INT8 quantize, batch | 70–90% |
| Full daily retraining | Incremental, weekly full | 60–80% |
| HPO no early stop | Successive halving / Hyperband | 40–60% |
| Cross-region transfer | Co-locate | 90%+ on transfer cost |
| Endpoint min 4 replicas | Scale-to-zero with cold start mitigation | 50–80% |
| Uncompressed logs | Compress + tier to cold storage | 80–95% |

### The Final Artifact

A 20–30 page report (markdown + charts):

- Executive summary (one page)
- Audit methodology
- Findings — the 10 anti-patterns with before/after
- Refactor strategy
- Implementation plan with risk mitigation
- Final cost analysis with a 12-month forecast

This document is your portfolio piece. Worth more than most resumes.

### Acceptance Criteria

- The "crime scene" is realistic — a senior reviewer would believe it could be a real company's stack
- The audit identifies at least 8 distinct anti-patterns
- The refactor achieves at least 50% cost reduction (60% target)
- No regression in correctness; validation tests in place
- The report is publishable on a blog without changes

### Stretch Goals

- Build a "ML cost CI" — automated checks in pull requests that flag expensive patterns
- Per-team cost dashboard with budget alerts
- Follow-up blog post: "The 10 most expensive MLOps mistakes I see in interviews"

### Interview Talking Points

- "Tell me about a time you reduced cost." — 40-minute answer.
- "How do you think about FinOps for ML?" — Specific, opinionated.
- "What's the most expensive ML pattern you've seen?" — Top-10 list.

### Realistic Time Estimate

**6–8 weeks** at 10 hrs/week.

---

## Project 5 — Distributed Fine-Tuning Platform for LLMs

### The Business Framing

Many F50s run dozens of fine-tuning jobs per week — for customer-support assistants, code completion, document understanding, marketing copy. Each team has its own data; the platform standardizes infrastructure, distillation, eval, and deployment.

This project bridges DE/MLOps and serious deep learning infrastructure.

### Tech Stack

- **Training:** PyTorch + FSDP, or DeepSpeed ZeRO-3
- **Framework:** Hugging Face Transformers + PEFT (LoRA / QLoRA) for parameter-efficient FT
- **Orchestration:** Ray Train or Kubeflow Training Operator on Kubernetes
- **GPUs:** rented (RunPod, Lambda, Modal) or simulated with smaller models
- **Dataset prep:** dlt + Spark for cleaning + tokenization at scale
- **Experiment tracking:** Weights & Biases (better for DL than MLflow)
- **Model registry:** Hugging Face Hub (private) or MLflow
- **Eval harness:** lm-eval-harness + custom task-specific evals
- **Serving:** vLLM with multi-LoRA serving (one base model, many adapters)
- **Distillation pipeline:** big model generates labels; small model fine-tunes on labels

### The Interesting Technical Decisions

1. **LoRA vs full fine-tuning vs QLoRA.** When each wins. Build all three for one task; compare quality, cost, latency.

2. **DPO vs SFT vs ORPO.** Build a small preference dataset; train via SFT only, SFT+DPO, ORPO; compare.

3. **Multi-LoRA serving.** vLLM serves one base model with many LoRA adapters loaded as needed. Show 5 adapters being swapped per request based on tenant routing. Document throughput impact.

4. **Cost per fine-tuning job.** Auto-estimated before launch; alerts when over budget.

5. **Data quality.** Bad data > smart algorithm. Build dedup, perplexity filtering, contamination detection (training data leaking into eval set).

### Acceptance Criteria

- At least 3 fine-tuning jobs end-to-end on 7B-class open-weights base
- LoRA + SFT pipeline, with a DPO pipeline as second pass
- Multi-LoRA serving with at least 3 adapters
- Eval harness with at least 5 metrics: helpfulness, format compliance, safety refusal rate, calibration, task-specific F1 or BLEU
- Cost per fine-tune documented; pre-launch estimator
- Contamination check on at least one eval set

### Stretch Goals

- Federated fine-tuning: simulate two "tenants" with private data; show that each model improves with their data only (privacy by data segregation)
- Knowledge distillation pipeline: GPT-4o → your-7B-LoRA on synthetic data
- Reward model training for full RLHF (PPO) — be honest about how finicky it is
- Continued pre-training on a domain corpus before SFT

### Interview Talking Points

- "How would you fine-tune an LLM for an enterprise use case?" — End-to-end answer.
- "When would you use SFT vs DPO?" — Concrete trade-offs.
- "How do you efficiently serve many fine-tuned variants?" — Multi-LoRA serving story.

### Realistic Time Estimate

**12–16 weeks** at 10 hrs/week. Heaviest of the seven. Worth it if your target is a frontier lab or LLM-heavy F50.

---

## Project 6 — Online Inference Platform with SLOs

### The Business Framing

You build the serving platform for a high-traffic ML org. Tens of models, tens of millions of requests per day, P99 latency SLOs in the 50–100ms range, autoscaling, multi-region, observability, gradual rollouts. This is the work of the inference team at Netflix, Uber, Pinterest, Stripe, DoorDash.

### Tech Stack

- **Serving:** KServe (sklearn / pytorch / triton predictors) on Kubernetes
- **GPU layer:** NVIDIA Triton with TensorRT-optimized models where applicable
- **Routing:** Envoy or Istio for traffic splitting, canary, header-based routing
- **Autoscaling:** HPA + KEDA (custom metric-driven scaling); Karpenter / Cluster Autoscaler for nodes
- **Multi-region:** at least two regions with primary/secondary failover
- **Observability:** OpenTelemetry, Prometheus, Grafana, Tempo, Loki
- **SLO tooling:** Sloth or OpenSLO for SLO definition; Grafana for SLO burn rate alerts
- **Load testing:** k6 or Locust at 10K+ RPS

### The Interesting Technical Decisions

1. **The latency budget.** For one specific model, decompose target P99 latency across components. Show that your system meets it.

2. **Cold start.** Scale-to-zero saves money but cold starts cost latency. Pre-warm pools, image pre-pull, model pre-load — implement.

3. **Multi-model serving.** Many small models on one GPU via Triton. Quantify the win vs one model per GPU.

4. **Batching.** Server-side micro-batching. Quantify the throughput vs latency curve.

5. **Multi-region.** Active/active or active/passive? Stickiness? Cross-region replication of model artifacts? Document the choice.

6. **SLOs and error budgets.** Define availability + latency SLOs. Implement burn-rate alerts. Tie deployment freezes to error budget exhaustion.

### Acceptance Criteria

- Serving stack handles 10K+ RPS sustained with P99 under 100ms for a representative model
- Canary deployment workflow: one CLI command shifts 10% traffic; monitoring decides promotion
- Multi-region failover: kill one region, traffic continues from the other within 30s
- Per-model SLO dashboard
- Load test report documenting the saturation point

### Stretch Goals

- A/B testing infrastructure with proper statistical analysis
- Shadow traffic infrastructure (mirror requests to a candidate model, log only)
- Adaptive timeout: if backend is slow, return cached / simpler fallback
- Per-tenant quota and rate limiting via Envoy

### Interview Talking Points

- "How do you serve ML models at scale?" — Long answer with specifics.
- "How do you handle cold start?" — You've benchmarked your own.
- "What's your approach to error budgets in ML serving?" — SRE-fluent answer.

### Realistic Time Estimate

**10–12 weeks** at 10 hrs/week.

---

## Project 7 — Federated ML Platform with Model Contracts

### The Business Framing

The trend in F50 ML architecture: individual domain teams own their models; the central team provides the platform, governance, standards. You can't build a "real" federated ML platform alone, but you can simulate one convincingly. Build three "domain teams" — each with its own training repo, model serving, and CI — publishing models that other domains consume via contracts.

### Tech Stack

- **Three repos** — `team-fraud`, `team-recs`, `team-pricing`. Each has its own training, registry, serving, CI.
- **A "platform" repo** — `ml-platform` with shared infrastructure (Terraform, monitoring config, contract validators)
- **Model registry** — central MLflow with team namespacing
- **Lineage** — OpenLineage events flowing into a central catalog
- **Discovery** — a UI where teams browse other teams' models, their contracts, their performance over time
- **Contracts** — input schema, output schema, latency SLO, refresh frequency, fairness metrics commitment

### Contracts as Code

```yaml
# team-fraud/contracts/fraud-classifier-v3.yaml
model: fraud-classifier
version: 3
owner: team-fraud@example.com
sla:
  latency_p99_ms: 50
  availability_pct: 99.9
  freshness_hours: 24
input_schema:
  user_id:
    type: string
    required: true
  amount:
    type: float
    required: true
    constraints: { min: 0.0, max: 1000000.0 }
  ...
output_schema:
  fraud_probability:
    type: float
    constraints: { min: 0.0, max: 1.0 }
  model_version:
    type: string
fairness:
  monitored_slices: [country, age_bucket, account_age_days]
  performance_disparity_tolerance: 0.05
```

A PR to `team-fraud` that breaks this contract is rejected by CI. Downstream consumers (team-recs uses fraud predictions as a feature) get notified of versioned changes.

### The Interesting Technical Decisions

1. **What's a model product?** Define concretely: a named, versioned, contract-enforced model with documented inputs, outputs, SLAs, fairness commitments, owner, deprecation policy.

2. **Cross-team consumption.** Team-recs consumes team-fraud's predictions. Their CI checks team-fraud's contract. If team-fraud breaks it, team-recs' CI catches it.

3. **Governance without bureaucracy.** Document workflows for: deprecating a model, versioning a contract, security/access reviews.

4. **Central platform team's role.** They build the platform, define standards, enforce contracts. They don't train models. Embody this in your three-repo structure.

### Acceptance Criteria

- Three working domain repos with their own CI
- At least 5 model products across the domains
- At least 2 cross-domain dependencies
- A contract validator that catches breaking changes in CI
- A central catalog UI showing all model products, contracts, lineage
- A simulated incident: break a contract, watch CI block the PR, fix, see CI pass

### Stretch Goals

- Contract evolution: producer releases v2 of a contract; consumers migrate at their pace
- Model quality scoring (freshness × accuracy × test pass rate × consumer adoption)
- Consumer SLOs: what guarantees does the producer promise the consumer

### Interview Talking Points

- "What do you think about data/ML mesh?" — You built one; you have opinions.
- "How do you handle dependencies between ML teams?" — Concrete answer with code.
- "How would you approach ML governance at scale?" — Real-world answer.

### Realistic Time Estimate

**10–14 weeks** at 10 hrs/week.

---

## How to Present These Projects

### The Repo Structure

For each project:

```
project-name/
├── README.md                 # The most important file
├── ARCHITECTURE.md           # Detailed architecture
├── DECISIONS.md              # ADRs you wrote during the build
├── docker-compose.yml        # One-command local startup
├── Makefile                  # `make up`, `make test`, `make seed`, `make load-test`
├── terraform/                # Cloud infrastructure
├── kubernetes/               # K8s manifests, ArgoCD config
├── pipelines/                # Training pipelines
├── services/                 # Serving services
├── notebooks/                # Exploration; never the source of truth
├── monitoring/               # Grafana dashboards, Evidently configs
├── tests/                    # Unit, integration, model, load
├── .github/workflows/        # CI/CD/CT
└── docs/                     # Additional docs, diagrams
```

### The README

The most-read part. Treat it as a tech blog post.

**Sections (in order):**

1. **One-paragraph summary.** What and why it's interesting.
2. **Architecture diagram.** Mermaid is fine.
3. **Tech stack.** One sentence on *why* each choice.
4. **Quickstart.** 5 commands to get running.
5. **Key design decisions.** 3–5 things you thought hard about, with trade-offs.
6. **Operational notes.** How it's monitored, how it handles failure, cost.
7. **Evaluation results.** Tables of numbers. Latency, throughput, model quality.
8. **What I'd build next.** Shows you have an extension roadmap.
9. **What didn't work.** Self-awareness. Optional but powerful.

Length: 1500–3000 words. Long enough to be substantial, short enough to read in one sitting.

### Where to Host

- Public GitHub repo
- A blog post (Medium / your own site / dev.to). Cross-link with the repo.
- A short demo video (3–5 minutes, screen recording, narrated). Drop it in the README.

### How to Talk About It in Interviews

A formula that works:

1. **One sentence framing.** "I built a real-time anomaly detection platform with a closed feedback loop because most fraud projects skip the operational reality of keeping the detector calibrated."
2. **The interesting decision.** Pick the *one* non-obvious technical decision and explain it for 3 minutes. Trade-offs, alternatives, why you picked what you picked.
3. **What you'd do differently.** Self-awareness is high-signal.

If the interviewer wants depth, you have hours of material. If not, you've shown your best work in 5 minutes.

---

## A Realistic Timeline to F50-Ready

Assuming you finish the foundations through specialization chapters (≈4–5 months at 10 hrs/week):

- **Month 5–9:** Project #1 — pick one of the "must do" three. Deep work. Treat like a real job.
- **Month 10–12:** Project #2 — second one, in a different specialization to show breadth.
- **Month 13:** Polish. Blog posts. LinkedIn presence. Apply.

Total: 12–14 months of focused part-time work from a serious start, with 2 portfolio projects deep enough to anchor any F50 interview.

If you only have time for one, pick **Project 1, 2, or 4**. Any alone is enough.

---

## A Note on Talking About These Projects

These are *learning projects*. They're not "I built this for my employer." Be honest in interviews — F50 hiring managers respect candidates who built something serious in their own time more than they respect candidates who exaggerate work experience.

The right framing: "I built this on my own to deeply learn X. Here's what I'd do differently with a team and a year of runway."

That's a senior signal.
