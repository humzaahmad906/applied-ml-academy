# 06 — Advanced Topics: Everything Else Worth Knowing — Part 4 of 5: Security, Governance, and Operational Concerns

This is part 4 of the Advanced Topics reference catalog. Here we cover security for ML, governance and compliance, modeling techniques beyond the default toolbox, disaster recovery, FinOps, and architectural patterns for ML.

## Phase 11 — Security for ML

### The ML Attack Surface

- **Model extraction** via API querying
- **Adversarial examples** — perturbations that flip decisions
- **Membership inference** — figure out whether a record was in the training set
- **Model inversion** — reconstruct training data from outputs
- **Data poisoning** — bad training data corrupting the model
- **Prompt injection** — LLM-specific; user input bypasses the system prompt
- **Indirect prompt injection** — malicious instructions in retrieved content
- **Tool abuse** in agents — getting the model to take harmful actions
- **Supply chain** — malicious pretrained weights, pip packages, container images

### Mitigations

- **Authentication and rate limiting** on every inference endpoint
- **Differential privacy** in training, where threat model warrants — adds calibrated noise; bounds membership inference. Implementation: Opacus (PyTorch), TF-Privacy.
- **Federated learning** when data can't leave its source (healthcare, banking).
- **Watermarking** model outputs (e.g., text watermarking for LLMs; image watermarking for diffusion).
- **Input sanitization** for LLMs — prompt injection detection, PII redaction.
- **Output filtering** — toxicity, hallucination detection.
- **Sigstore for ML artifacts** — sign model weights; verify on load.
- **SBOMs** for ML pipelines.

### Secrets Management

- Never in Git, never in container images, never in DAG code
- AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault
- External Secrets Operator on Kubernetes
- OIDC for short-lived cloud credentials in CI

### Regulatory Specifics for ML

- **EU AI Act** — high-risk systems require risk management, technical docs, data governance, transparency, human oversight, robustness, post-market monitoring.
- **NYC Local Law 144** — annual bias audit for employment AI.
- **NIST AI RMF** — voluntary risk framework; increasingly a baseline.
- **GDPR Article 22** — restrictions on solely automated decisions.
- **HIPAA** for health — PHI protection, audit logging, BAA.
- **Colorado AI Act, California regulations** — emerging US state-level.

### Exercises

1. Add authentication (API key) to your serving service. Add per-key rate limiting.
2. Train a small model with Opacus. Compare quality and DP-noise trade-off.
3. Try a prompt-injection attack on your LLM project. Then build a detector.

---

## Phase 12 — Governance and Compliance

### Model Inventory

Every production model registered with:

- Owner (team, individual)
- Purpose (the problem it solves)
- Training data lineage (which sources, versions)
- Performance per slice (overall and per protected attribute where applicable)
- Known limitations (out-of-distribution behavior, failure modes)
- Deployment scope (which services, which traffic)
- Approval status and reviewers

### Model Cards

For each model, a public-ish document:

- Intended use, intended users
- Out-of-scope uses
- Training data summary (sources, period, sample size, demographic breakdown)
- Performance overall and per slice
- Known biases and limitations
- Ethical considerations
- Last updated, version, changelog

Format: see [HuggingFace model cards](https://huggingface.co/docs/hub/model-cards) for a public-facing example. For internal, your registry can host them.

### Audit Trails

For regulated domains, log:

- Every promotion (who, when, which version, evidence pack)
- Every prediction (sampled for high-volume; full for low-volume regulated decisions like credit)
- Every label feedback
- Every model card change
- Every contract change

Retain 7+ years for finance/healthcare. Make queryable.

### Approval Workflows

Bake into the registry. Promotion to Production for high-risk models requires:

- Bias audit passed
- Security review
- Risk committee sign-off
- Tested rollback plan

### Explanation and Recourse

Where applicable (credit, insurance, employment):

- SHAP / LIME / counterfactual explanations
- Human-readable reason codes
- Appeal / manual-review path

### Exercises

1. Add a model card template to your project. Generate one for your model.
2. Add an audit log table; log promotions and a sampled fraction of predictions.
3. Add a SHAP-based explanation endpoint. Show how a specific prediction is explained.

---

## Phase 13 — Modeling Beyond the Defaults

### Causal Inference

Most ML predicts; some problems need to *intervene*. "If I show this ad, will the user buy?" is causal, not predictive.

- **Randomized experiments** (A/B tests) — the gold standard
- **Propensity score matching** — for observational data
- **Difference-in-differences** — natural experiments
- **Synthetic control** — comparing affected unit to a weighted average of unaffected units
- **Uplift modeling / treatment-effect estimation** — predict treatment effects, not outcomes
- **Doubly Machine Learning (DML)** — combines ML with causal estimation; semi-parametric

Tools: EconML (Microsoft), CausalML (Uber), DoWhy.

### Bandits and Online Learning

- **Multi-armed bandits** — Thompson sampling, UCB, ε-greedy
- **Contextual bandits** — when context matters
- **Reinforcement learning** — when actions affect future states

Used heavily in recommendations, ads, dynamic pricing.

### Time-Series Forecasting

- **Classical:** ARIMA, ETS, Prophet
- **DL:** N-BEATS, NHITS, Temporal Fusion Transformers, TimesNet
- **Foundation models for time series:** TimeGPT, Chronos, Lag-Llama — the emerging frontier

For most business forecasting, simple beats fancy. Always try a strong baseline (last-week-this-day, sliding mean, Prophet) before reaching for transformers.

### Graph ML

- **GNNs** — message-passing networks for graph data
- **Use cases:** fraud rings, recommendation, drug discovery, supply-chain
- **Frameworks:** PyTorch Geometric, DGL

For most companies, graph problems are solved with graph databases (Neo4j) + heuristics. GNNs come in when you have lots of labeled graph data and the patterns are subtle.

### Tabular Deep Learning

Despite the deep learning revolution, **gradient boosting (XGBoost, LightGBM, CatBoost) still wins on most tabular problems** in 2026. Tabular DL has matured (TabNet, FT-Transformer, SAINT) but hasn't displaced GBT for most use cases.

Senior engineers know when to *not* reach for deep learning.

---

## Phase 14 — Backup, Disaster Recovery, Business Continuity for ML

### RPO and RTO for ML Artifacts

- **RPO (Recovery Point Objective):** how much can you lose? For training data, often hours; for model artifacts, ideally zero.
- **RTO (Recovery Time Objective):** how fast back online? For real-time serving, minutes.

### What Needs Backup

- Model artifacts (registry contents)
- Feature definitions and historical features
- Training data
- Configuration (DAGs, prompts, feature definitions, deployment manifests in Git)
- Online feature store (with reconstruction-from-offline as fallback)

### Multi-Region for ML Serving

- Active/active for stateless serving (with cross-region model replication)
- Active/passive for training (cheaper; failover for DR only)
- Online feature stores: replicated or rebuildable from offline
- Avoid cross-region inference at request time (latency, cost)

### What Can Go Wrong

- A single AZ outage
- A region outage (rare but real)
- A bad model deploy (most common; mitigated by canary + automated rollback)
- A feature pipeline regression (silently wrong predictions; the worst kind of outage)
- A vendor outage (OpenAI / Anthropic down → your LLM app down). Multi-provider routing is the mitigation.

### Runbooks

For each system, a one-pager runbook:

- How to detect a failure
- Immediate mitigations (rollback model, switch traffic, route to backup provider)
- Investigation steps
- Communication plan
- Post-incident review template

### Multi-Region LLM Serving

Multi-region serving for LLMs is harder than multi-region serving for classical ML. The model is stateless; the compliance constraints are not.

**Control-plane / data-plane split:**

The control plane (routing decisions, quota enforcement, configuration) can be centralized. The data plane (actual inference) must be regional. A request from Frankfurt should never cross to us-east-1 for inference — latency and residency both prohibit it. The AI gateway layer (covered in the LLMOps section) handles the routing; the inference fleet is deployed per-region.

**The six data residency leak surfaces:**

Data residency is not just about where inference runs. A compliant system must trace all six paths:

1. **Inference** — prompt and response processed in the correct region (the obvious one)
2. **Telemetry** — OTel spans, logs, and metrics must export to a regional endpoint, not a global collector
3. **Eval pipelines** — if eval jobs use production prompts as test cases, those prompts cannot leave the region for evaluation compute
4. **Prompt caches** — semantic caches must be regional; a cached response from a EU user must not be retrievable by a US instance
5. **Fine-tune feedback loops** — if production interactions feed back into fine-tuning datasets, that data movement must respect residency
6. **Observability** — dashboards and alerting platforms that aggregate cross-region data must not expose individual prompts to operators in the wrong jurisdiction

Most teams get #1 right and miss #2–6. An architecture review for EU AI Act or GDPR compliance should walk all six explicitly.

**Region-specific model availability:**

Model availability is not uniform across regions. AWS Bedrock, Azure OpenAI, and Vertex AI all have different model catalogs per region. Claude 3.5 Sonnet may be available in us-east-1 but not eu-central-1 at a given point in time. Your gateway's model routing table must be region-aware, with fallback logic that respects both residency constraints and availability.

**Failover automation and brownout testing:**

Standard health checks detect hard failures (503, timeout). LLM serving also degrades gracefully: a provider may be available but returning degraded quality or elevated latency. Brownout testing — deliberately injecting latency or error rates on one region's provider path — validates that your failover logic triggers on soft failures, not just hard ones. Run brownout tests quarterly; they catch routing bugs that health checks miss.

**Research grounding:**

The cross-region load balancing problem for LLM serving has been formalized in recent work. SkyWalker (arXiv:2505.24095) and GORGO (arXiv:2602.11688) both address the optimization of cross-region request routing under latency, cost, and capacity constraints — useful reading if you're designing the routing policy for a large fleet.

---

## Phase 15 — Cost and FinOps for ML

### The Cost Model

| Layer | Drivers | Typical % of bill |
|---|---|---|
| Training | GPU-hours, dataset prep compute | 20–40% |
| Inference | GPU-hours, request volume, model size | 30–60% |
| Storage | TB-months across hot/warm/cold | 5–15% |
| Data egress | Cross-region, cross-cloud | 5–15% |
| Vendors | LLM API spend, observability, registries | 5–25% |
| Engineering tooling | W&B, MLflow Cloud, etc. | 1–5% |

### The Levers

- **Inference:** quantize, distill, batch, route, cache, scale-to-zero, right-size hardware
- **Training:** spot instances, early stopping, smarter HPO, smaller search spaces, transfer learning
- **Storage:** tier old data; compress; lifecycle to cheap storage
- **Egress:** co-locate data and compute; VPC endpoints
- **Vendors:** negotiate annual commits, route to cheaper provider when quality allows
- **People-process:** chargeback (not showback), per-team budgets, weekly cost review

### LLM Cost Specifically

- Token cost dominates. Compress prompts (LLMLingua), shorter outputs, JSON not prose.
- Self-host the most-used models. The break-even at 2026 prices is roughly: if you spend >$10K/month on a hosted LLM, evaluate self-hosting.
- Distill expensive models into cheap ones for specific tasks.
- Cache aggressively (semantic cache catches more than exact-match).

### Exercises

1. Pick one of your projects. Build a per-day cost breakdown across training / inference / storage / egress / vendors.
2. Identify the top 3 cost drivers. Reduce one by 50%. Document.

---

## Phase 16 — Architectural Patterns for ML

### The Outbox Pattern

When the app makes a prediction *and* publishes an event ("we predicted X for user Y"), do both in the same DB transaction (write to an outbox table), then a separate process reads outbox and publishes to Kafka. Prevents inconsistency between app state and event stream. CDC reads outbox directly.

### Event Sourcing

Store the log of events as the source of truth. Current state is a fold of events. Kafka + Iceberg is event-sourcing-shaped. Multiple read projections (warehouse, online feature store, OLAP) is CQRS.

### Reverse ML

The pattern of pushing model outputs back into operational tools. Predictions to Salesforce, scores to marketing automation, segments to ad platforms. Tools: Hightouch, Census, Polytomic.

### Lambda vs Kappa for ML

- **Lambda:** two paths (batch + speed), merge for queries. Complex; two codebases.
- **Kappa:** one streaming path; reprocess history by replaying. Simpler; needs stream retention.

Most modern ML platforms are Kappa-ish over Iceberg/Delta. Batch is just "view the streaming-written table."

### Model as a Product

The mindset shift: a model isn't a side effect of training; it's a deliverable with consumers, SLAs, contracts, documentation, versioning, deprecation policy. This is what makes a "model platform" work at scale (Project 7).

### Federated Learning

Train a global model on data that stays distributed. Each node trains locally; gradients (or models) aggregated centrally. Used in healthcare, finance, mobile (Google's keyboard). Frameworks: Flower, FedML, NVIDIA FLARE.

---

## You can now

- Enumerate the ML-specific attack surface (model extraction, membership inference, prompt injection, data poisoning) and pick the matching mitigation (DP training, watermarking, input/output filtering, artifact signing).
- Stand up a governance program — model inventory, model cards, audit trails, and approval workflows — that satisfies EU AI Act / NIST AI RMF expectations.
- Know when to reach for causal inference, bandits, time-series foundation models, or graph ML instead of a default supervised model — and when gradient boosting still wins on tabular data.
- Reason about the six data-residency leak surfaces for multi-region LLM serving and design a control-plane/data-plane split that survives a GDPR or EU AI Act review, backed by an RPO/RTO-driven DR plan.
- Apply the standard ML cost levers (quantize/distill/cache/route) and architectural patterns (outbox, event sourcing, reverse ETL, Lambda vs Kappa) to a platform's FinOps and system design.
