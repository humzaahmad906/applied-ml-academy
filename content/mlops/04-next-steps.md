# 04 — Next Steps: Specialization Beyond the Foundations

You've built foundations, productionized a workflow, and gone deep on scale. The frontier for F50 MLOps in 2026 is wider than what's in any single course. This file closes the gap between "I finished a strong MLOps curriculum" and "I can credibly interview at OpenAI, Anthropic, Google DeepMind, Meta AI, Apple ML, Amazon Search, Netflix, Stripe, Uber, Snowflake-AI, Databricks."

**Time:** 6–8 weeks at 10 hrs/week. Treat it as a second course.

## The Honest Map: Foundations vs F50 Reality

| Layer | Foundations | F50 Reality |
|---|---|---|
| Cloud | Any one | AWS dominates, Azure second (esp. with OpenAI), GCP third |
| Orchestration | Prefect / Airflow | Airflow (mostly), increasingly Dagster; Temporal for ML-adjacent workflows |
| Experiment tracking | MLflow | MLflow / W&B (often W&B at frontier labs — note W&B is now owned by CoreWeave, Mar 2025; weigh the GPU-vendor lock-in), custom internals at the biggest |
| Feature store | Feast | Tecton / Databricks FS / SageMaker FS / Feathr / internal builds |
| Training | PyTorch on K8s | PyTorch + DeepSpeed/Megatron + Ray/Slurm; massive in-house tooling |
| Serving | KServe / BentoML | KServe / Triton / Ray Serve / **vLLM / TGI / SGLang for LLMs** |
| Monitoring | Evidently | Arize / Fiddler / WhyLabs / custom; LLM-specific evals (Braintrust, Langfuse) |
| Governance | Light | Heavy — model cards, AI risk frameworks, regulator-facing audit trails |
| LLM ops | Touched | A distinct sub-discipline (LLMOps) — RAG, evals, prompt management, agent ops |
| Vector DBs | Touched | First-class infra component; Pinecone / Weaviate / Qdrant / pgvector / Vespa |
| Quality | dbt tests | dbt + Great Expectations + Soda + custom drift detection |
| CI/CD | GitHub Actions | GitHub Actions / GitLab / Jenkins; Argo CD / Flux for K8s GitOps |
| GPU ops | Basics | NCCL tuning, fabric topology, GPU fleet management |

Strong on the columns where foundations and F50 agree. Gaps to close: managed ML platforms, LLMOps, governance, advanced monitoring, vector / retrieval infrastructure, GPU operations at scale.

---

## Phase 1 — Cloud ML Platforms (2 weeks; pick one to start)

You should know one cloud's ML platform fluently. AWS is the most-asked in F50 interviews. GCP is the most pedagogically pleasant (BigQuery + Vertex are cohesive). Azure is rising via OpenAI integration.

### AWS SageMaker (Most Common at F50)

What to know:

1. **SageMaker Studio** — the IDE. Notebooks, terminals, Git, JupyterLab. Connects to all SageMaker services.
2. **Training Jobs** — managed training. You provide a Docker image + entry script; SageMaker handles instances, retries, checkpointing to S3. Supports script mode (provide a Python script, use a prebuilt container) and BYO container.
3. **Hyperparameter Tuning Jobs** — managed Bayesian / Hyperband / Grid HPO.
4. **Pipelines** — managed DAGs for training/inference workflows; SDK is Python. Integrates with Model Registry.
5. **Model Registry** — SageMaker's equivalent to MLflow registry. Includes model approval workflow.
6. **Endpoints** — managed serving. Real-time, batch, async, serverless options. Multi-model endpoints (host many models on one instance).
7. **Feature Store** — SageMaker's feature store. Online (DynamoDB-backed) + offline (S3). Less mature than Feast/Tecton but tightly integrated with the rest of SageMaker.
8. **Model Monitor** — built-in drift detection, bias detection, explainability.
9. **Clarify** — bias and explainability reports.
10. **Inferentia and Trainium** — AWS's custom ML chips. Often dramatically cheaper than equivalent NVIDIA, with worse software support. Know they exist; pick them up if you target an AWS-heavy F50.

For a portfolio project: port your tier-2 project to SageMaker. Same dbt code (if any), same model code, different training launch and serving stack. Document the cost and DX differences.

### GCP Vertex AI

What to know:

1. **Vertex Workbench** — notebook environment. Connects to BigQuery / GCS / Vertex services.
2. **Vertex Training** — custom training jobs on managed infrastructure. Tightly integrated with BigQuery as data source.
3. **Vertex Pipelines** — managed Kubeflow Pipelines (KFP). KFP DSL is more cumbersome than Airflow/Prefect; you'll need to learn it.
4. **Vertex Model Registry**.
5. **Vertex Prediction** — online / batch / private endpoints.
6. **Vertex Feature Store** — fully managed feature store. Online (Bigtable-backed) + offline (BigQuery).
7. **Vertex Model Monitoring** — drift and skew detection out of the box.
8. **Matching Engine** — managed vector search. Increasingly central to GCP's GenAI story.
9. **Vertex Agent Builder** — managed RAG + agent infrastructure.
10. **TPUs** — Google's custom ML chips. Mature, occasionally faster than equivalent GPUs, software fully through JAX/XLA. Worth at least having seen.

Strong fit: companies with heavy BigQuery investment. The BigQuery → Vertex pipeline is exceptionally smooth.

### Azure ML

What to know:

1. **Azure ML Studio** — workspace UI, designer, notebooks.
2. **Compute targets** — instances and clusters; can be CPU, GPU, or AKS-backed.
3. **AML Pipelines** — DAG framework, similar to Vertex Pipelines.
4. **MLflow as the default tracker** — Azure ML uses MLflow natively, with managed hosting. This is often the cleanest cloud-MLflow experience.
5. **Model Registry, Endpoints, Managed Online Endpoints** — like the others.
6. **Azure OpenAI Service** — *the* differentiator. Hosted GPT/Claude-adjacent models with enterprise compliance.
7. **Azure AI Search** — managed search + vector retrieval, deep RAG integration.
8. **Azure AI Studio** — the new umbrella for generative AI on Azure.

Strong fit: any organization already in the Microsoft / OpenAI orbit. Pharma, finance, government.

### Databricks ML

What to know:

1. **Notebook-first dev** — Python, SQL, Scala, R coexist.
2. **MLflow native** — Databricks built MLflow; the Databricks MLflow experience is the best version.
3. **Unity Catalog** — governance layer for data, features, models. Increasingly the centerpiece.
4. **Feature Engineering in Unity Catalog** — Databricks' feature store, integrated with Delta tables.
5. **Mosaic AI** — model training (especially LLM fine-tuning) tooling, includes the integrated MPT/Mosaic stack.
6. **Vector Search** — managed vector index on Delta tables.
7. **Model Serving** — managed real-time serving, autoscales to zero.
8. **AI Gateway** — proxies LLM calls with rate limiting, observability, and cost control across providers.

Strong fit: companies with heavy lakehouse / Spark investment. Any large enterprise migrating off Hadoop / EMR.

### Cost Discipline Across All Cloud ML Platforms

The hidden money pits to watch for:

- **Endpoints autoscaling minimums set to 1+ replicas** — pays even at zero traffic
- **Cross-region storage / training** — data egress kills you
- **Idle SageMaker Studio notebooks** — they bill while running, hours after you closed the tab
- **NAT Gateway for outbound S3** — use VPC endpoints instead
- **GPU instances launched manually for "quick experiments"** — left running over weekends
- **HPO jobs with no early stopping** — running 200 trials when 50 would have sufficed
- **Logs piling up** — CloudWatch / Stackdriver / Log Analytics can cost more than compute

Tag every resource with team/project/environment. Build a weekly cost report. Read it.

### What to Build

Port your medium-tier project to one cloud's ML platform end to end. Same model code, different infrastructure. Document everything that surprised you — that's the interview story.

---

## Phase 2 — LLMOps and Foundation Model Operations (2 weeks)

The biggest single shift in MLOps since the field existed. Different enough from classical ML ops that many companies have separate "LLMOps" or "AI Engineering" tracks.

### Why LLMOps Is Different

Classical ML | LLMOps
---|---
Train a model on your data | Use someone else's model; adapt it
Metric: AUC / accuracy / F1 | Metric: helpfulness, harmlessness, faithfulness, ... harder
Single output per prediction | Sequence of tokens, variable length
Inference cost: pennies | Inference cost: dollars per million tokens
Deterministic given seed | Stochastic, often by design (sampling)
Ground truth labels | Often no ground truth — human or LLM-as-judge
Drift = retrain | Drift = re-prompt, re-RAG, re-fine-tune
Auditability via training data | Auditability via prompts + retrieval + tool calls

### The LLMOps Stack

```
[App] ─► [Gateway / Router (Portkey, LiteLLM, Helicone)] ─► [Model providers]
                                                                  │
                                                                  │  (OpenAI, Anthropic,
                                                                  │   self-hosted vLLM,
                                                                  │   Bedrock, Vertex,
                                                                  │   together.ai, ...)
                                                                  ▼
                                            [Prompts: versioned in a prompt registry]
                                            [Retrieval: vector DB + reranker]
                                            [Evals: offline + online, LLM-as-judge]
                                            [Observability: Langfuse / Braintrust / W&B Weave]
                                            [Guardrails: NeMo Guardrails / Lakera / built-in]
                                            [Fine-tuning pipeline: LoRA / DPO / SFT]
```

### What You Need to Master

#### 1. Model Serving for LLMs

The default LLM-serving stack:

- **vLLM** — open-source, dominant. The **V1 engine** is the default architecture since 2025: disaggregated prefill/decode scheduling, rewritten async execution loop, prefix caching on by default. Stripe reported a 73% inference cost reduction after migrating to vLLM — the canonical public case for what a modern serving stack does to unit economics. Continuous batching, PagedAttention, speculative decoding, multi-LoRA serving all included.
- **TGI (Text Generation Inference)** — Hugging Face's stack; similar capabilities.
- **SGLang** — production-ready peer of vLLM (no longer "the newer one"): ~29% higher throughput on H100 for 7–8B models and up to 6.4× on prefix-heavy workloads (RAG, multi-turn agents) thanks to RadixAttention prefix caching; strong structured output. Serving trillions of tokens daily across major deployments.
- **Triton + TensorRT-LLM** — NVIDIA's serving stack; lowest latency / highest throughput on NVIDIA hardware.

Key concepts:

- **Continuous batching** — instead of waiting for a batch to be ready, the server interleaves new requests into the running batch as token-by-token generation progresses. Drastically improves throughput.
- **PagedAttention / KV cache management** — the KV cache is the main memory cost of LLM inference. PagedAttention pages it like virtual memory; lets you fit way more concurrent requests in the same memory.
- **Prefix caching** — when many requests share a common prefix (system prompt, RAG context), cache the KV state. Reuses compute.
- **Speculative decoding** — a small draft model proposes the next N tokens; the big model verifies in parallel. 2–4x speedup typical.
- **Speculative + tree decoding (Medusa, EAGLE)** — generalizations.
- **Quantization** — INT8 (AWQ, GPTQ), INT4 (Marlin kernels), FP8 (Hopper / Blackwell). Often 2–4x throughput with <1% quality delta.

#### 2. RAG (Retrieval-Augmented Generation)

The pattern almost every enterprise AI app uses:

```
[User query] → [Embed] → [Vector DB search + keyword search] →
[Rerank] → [Top K passages] → [LLM with passages as context] → [Response]
```

What to learn:

- **Chunking strategies** — fixed-size, semantic, hierarchical. Bad chunking is the #1 RAG failure.
- **Embedding models** — text-embedding-3-large vs Cohere embed-v3 vs open-source BGE / E5. Trade off cost, quality, and dimensionality.
- **Vector DBs** — pgvector, Pinecone, Weaviate, Qdrant, Vespa, LanceDB. Algorithms: HNSW vs IVF + PQ; you balance recall, latency, memory.
- **Hybrid search** — combine vector (semantic) and BM25 (lexical). Almost always beats either alone.
- **Reranking** — a cross-encoder (Cohere Rerank, BGE-Reranker) on the top 50–200 candidates to find the best 5–10. Big quality win for cheap compute.
- **Evaluation** — RAGAS, ARES, LLM-as-judge for faithfulness, context relevance, answer relevance.

A solid RAG project is a high-leverage portfolio piece. Many F50s have "build a RAG over our enterprise docs" as an interview problem.

#### 3. Fine-Tuning

When prompting + RAG aren't enough:

- **SFT (Supervised Fine-Tuning)** — gather instruction → response pairs, train on next-token prediction over the response only. The baseline.
- **LoRA / QLoRA** — train small adapter weights on top of a frozen base. Massively cheaper; 1–10% of compute of full fine-tuning, often within 1–2% of quality.
- **DPO (Direct Preference Optimization)** — given pairs of (chosen, rejected) responses, train the model to prefer chosen. Cheaper than RLHF, often equivalent results.
- **RLHF (Reinforcement Learning from Human Feedback)** — the original alignment method. PPO on a reward model. Expensive and finicky; mostly replaced by DPO/ORPO/KTO for new work.
- **Continued pretraining** — for major domain shifts; not common but worth knowing.

The 2026 default: **start with a strong open-weights base (Llama-class, Qwen, Mistral, Gemma), apply LoRA SFT for capability, follow with DPO for alignment**. The underlying math is covered in the advanced topics chapter.

#### 4. Evaluation

The hardest part of LLMOps. Approaches in increasing reliability:

1. **Automated metrics** — BLEU, ROUGE, perplexity. Cheap, weak.
2. **LLM-as-judge** — a strong model grades your model's output. Works for many tasks but has known biases (verbosity, self-preference).
3. **Programmatic checks** — does the output validate against the requested schema? Match required substrings? Pass unit tests if it's code?
4. **Pairwise comparison + ELO** — two outputs, which is better; aggregate ELO scores.
5. **Human eval with rubrics** — gold standard, expensive, slow.

Tools:

- **Braintrust, Langfuse, W&B Weave, LangSmith** — eval + observability platforms
- **RAGAS** — RAG-specific
- **Promptfoo** — local-friendly prompt evals
- **EleutherAI lm-eval-harness** — standard academic benchmarks

For a portfolio project, build a real evaluation harness for whatever LLM app you build. The eval harness is often more impressive than the app.

#### Eval as a CI/CD Gate

Offline eval is the new "we test in prod." If your eval suite only runs before a quarterly release, you're testing in production — you're just doing it slowly.

The mature pattern: **eval suites block deploys.** Every pull request and every deployment triggers the eval suite; if scores regress beyond a threshold, the deploy is blocked. This is the LLM analog of the test suite.

- **Braintrust** has native CI integration: define a baseline eval score, run the suite in CI, fail the check on regression. Pull requests show eval diffs the way code review shows code diffs.
- **Galileo** evaluates 100% of production traffic at <200ms latency — not just a sample, not just offline. Quality regressions surface within hours of a deploy, not days.

What "regression" means in practice: you define a quality metric (task-specific accuracy, LLM-as-judge score, schema validity rate) and a threshold (e.g., "new version must score within 2% of champion on the held-out eval set"). Any model or prompt change that crosses the threshold requires human review before shipping.

This is what separates a mature LLMOps org from one that pushes prompts to production and hopes. The eval gate is the guardrail; everything else is cleanup.

#### 5. Guardrails and Safety

- **Input filtering** — prompt injection detection, PII redaction
- **Output filtering** — toxicity, hallucination detection, format enforcement
- **Tool use guardrails** — explicit allow-lists of tools, parameter validation, dry-run mode for sensitive actions
- **Rate limits and budgets** — per-user, per-key cost ceilings
- **Audit logs** — every prompt, response, retrieval; required for any enterprise deployment

Tools: **NeMo Guardrails** (NVIDIA), **Lakera Guard**, **OpenAI's moderation endpoint**, **Anthropic's safety features**, plus a lot of bespoke code.

#### 6. The AI Gateway Layer

LLM cost can blow a budget overnight — but cost management is now the smallest reason to run a gateway. Gateways are infrastructure, not optional. More than 90% of production AI teams run 5 or more models in 2026. Without a gateway, you're hardcoding provider SDKs into application code, which means a provider outage is an application outage.

**Why the gateway is now mandatory:**

The November 2025 Cloudflare outage cascaded into ChatGPT downtime. OpenAI had multiple multi-hour incidents in late 2025. Any production system with a single-provider dependency has a reliability ceiling set by that provider. Multi-provider failover is table stakes.

**What a gateway does:**

- **Provider failover** — primary call fails; route to fallback provider within the same request latency budget
- **Per-tenant quotas and token-level cost attribution** — internal chargebacks, per-team rate limiting, budget alerts
- **Model cascading** — route to a cheap model first; only escalate to an expensive one when the cheap model signals low confidence or the task type requires it
- **Semantic caching** — cache responses by semantic similarity, not just exact match; hit rates of 30–60% on repetitive workloads
- **A/B routing** — split traffic between models, measure quality metrics, promote winners
- **Unified observability** — all provider calls through one trace path, regardless of which provider answers

**Decision tree for gateway choice:**

| Situation | Tool |
|---|---|
| Self-hosted default, <$50K/month LLM spend | **LiteLLM** — OSS, OpenAI-compatible proxy, handles 100+ models, Python-native config |
| Want managed features without lock-in | **Portkey** — went fully Apache 2.0 in March 2026; managed tier available |
| API-gateway team already runs Kong | **Kong AI Gateway** — AI plugin on top of existing Kong infrastructure; one platform for all API traffic |
| Already deep in MLflow | **MLflow AI Gateway** — native integration with MLflow tracking and model registry |

**Uber's GenAI Gateway** is the canonical internal case: a centralized gateway handling all LLM traffic across thousands of internal services, with cost attribution per business unit, automatic fallback across providers, and a semantic cache layer. The pattern is increasingly the standard at F50 scale.

```python
# LiteLLM: the same code, any provider
from litellm import completion

# Failover: tries gpt-4o first, claude-3-5-sonnet second
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    fallbacks=["claude-3-5-sonnet-20241022"],
    metadata={"user_id": user_id, "team": "search"},  # for cost attribution
)
```

Cost tactics that live inside the gateway:

- **Prompt compression** — LLMLingua and similar tools reduce token count before the request leaves the gateway
- **Model cascading** — embed a confidence classifier at the gateway; route simple queries to a cheap model, complex ones to the expensive one
- **Cache warming** — pre-populate semantic cache with known-frequent queries

#### 7. Managed Fine-Tuning (FTaaS)

Fine-tuning infrastructure is a significant operational burden if you build it yourself: GPU reservation, distributed training setup, checkpoint management, experiment tracking, LoRA adapter storage, serving the adapted model. Before committing to self-hosted fine-tuning infrastructure, price the alternative.

FTaaS providers in 2026:

- **Together AI** — wide model selection, transparent ~$3–4/hr H100-equivalent pricing, LoRA and full fine-tune
- **Fireworks AI** — fast serving of fine-tuned models, competitive pricing
- **Modal** — serverless; ~$4/hr H100, per-second billing; bring your own training code
- **Tinker** (Mira Murati, 2025) — purpose-built fine-tuning platform with a strong ops-first design philosophy
- **Nebius Token Factory** — European provider, useful for data-residency cases

The economics: at ~$3–4/hr for H100-equivalent compute, a LoRA fine-tuning run on a 7B model costs $10–50 depending on dataset size and epochs. The break-even for self-hosting is when you have **sustained GPU utilization** — idle-GPU economics destroy the self-hosted case. A team running one fine-tuning job per week pays 2 hours of GPU time; owning the H100 node pays whether it runs or sits idle.

Decision rule: **self-host only when you have sustained high utilization or data residency constraints that prohibit sending training data to a third party.** Otherwise FTaaS wins on unit economics.

### What to Build

Pick one:

1. A RAG app over a real corpus (your own notes, a public document set) with proper chunking, embedding, hybrid search, reranking, and an eval harness.
2. A small fine-tuning project: take a 7B open-weights base, LoRA-SFT it on a focused task (e.g., natural language to SQL on a specific schema), DPO it on a small preference set, eval it against the base. Run the fine-tuning on a FTaaS provider; compare to what self-hosting would cost.
3. An agentic system that uses tools (calculator, search, API calls) with proper guardrails, logging, and replay.

---

## Phase 3 — Vector Databases and Retrieval Infrastructure (1 week)

### What a Vector DB Is

A specialized DB for **approximate nearest neighbor (ANN)** search over high-dimensional vectors. Algorithms:

- **HNSW (Hierarchical Navigable Small World)** — graph-based, very fast, more memory. The dominant choice.
- **IVF (Inverted File)** — partition vectors into clusters, search nearest clusters. Cheaper, slightly worse recall.
- **PQ (Product Quantization)** — lossy compression of vectors. Often combined with IVF (IVF-PQ).
- **DiskANN** — disk-backed, good for billion-scale on cheap hardware.

You don't implement these. You configure them and tune recall/latency/memory.

### The Players

| Tool | Strengths | Trade-offs |
|---|---|---|
| **pgvector** | Postgres extension; SQL-native; mature; great for <50M vectors | Slower at scale than purpose-built |
| **Pinecone** | Managed, fully hosted, scales transparently | Expensive; no open-source path |
| **Weaviate** | Open-source, GraphQL + REST, modules for embedding, hybrid built-in | Heavier ops than pgvector |
| **Qdrant** | Rust-based, fast, OSS, payload filtering | Newer; smaller ecosystem |
| **Milvus** | Mature, scales to billions, GPU support | Heavier ops |
| **LanceDB** | Embedded, like DuckDB for vectors | Small scale; great DX |
| **Vespa** | Yahoo's engine; hybrid + relevance ranking; production-proven | Complex; high ceiling |
| **OpenSearch with k-NN** | If you already have OpenSearch | Decent enough; common at AWS shops |

### The Hybrid Search Pattern

```
query → [embedding] ──► [vector search top 100]
              │                                  ├──► [merge / rerank with cross-encoder] ──► top 10
              └─► [BM25 keyword search top 100] ─┘
```

Lexical and semantic agree more often than not; reranking sharpens the top. Quality typically beats either path alone by 10–30%.

### What You Build

1. A pgvector setup with 1M embedded documents. Implement hybrid search. Measure recall@10 against a labeled query set.
2. Compare pgvector vs Qdrant vs Weaviate on the same workload. Note latency, throughput, ops complexity.
3. Add a reranker. Measure quality lift.

---

## Phase 4 — Advanced Monitoring and ML Observability (1 week)

The medium-tier guide covered the basics. Here's what F50 production looks like.

### The Five Dimensions of Production ML Observability

1. **System health** — request latency, error rate, throughput, resource utilization. Standard SRE stuff.
2. **Data drift** — input feature distributions over time.
3. **Concept / performance drift** — output quality, when labels exist.
4. **Prediction drift** — output distributions.
5. **Business metrics** — the actual KPI the model exists to move (conversion, click-through, fraud rate, NPS).

The strongest signal is often #5 — when conversion drops, you'd better know.

### Tools

| Tool | What |
|---|---|
| **Evidently / WhyLogs** | OSS, drift reports, profile-based monitoring |
| **Arize, Fiddler, Aporia, Truera** | Commercial ML observability — most F50 standardize on one of these |
| **Datadog ML** | If you're a Datadog shop |
| **Langfuse / Braintrust / W&B Weave** | LLM-specific observability |
| **Grafana / Mimir / Tempo / Loki** | The OSS observability stack — metrics, traces, logs |
| **OpenLineage + Marquez** | Lineage; *which dataset, which pipeline, which model*? |

### Specific Patterns to Master

#### 1. The Reference Dataset Problem

What do you compare current production data to? Three options:

- **The training set** — the most common; cheap; but loses sensitivity over time as the training set ages.
- **A rolling window of recent production data** — sensitive to gradual shifts; can mask large persistent drift.
- **A held-out "golden" set with known properties** — best for behavioral tests; not useful for distribution drift.

Most teams use *both* training set drift and rolling-window drift.

#### 2. Slicing

Aggregate metrics hide subgroup failures. Always slice:

- By geography
- By device / platform
- By customer cohort / segment
- By new vs returning user
- By protected attribute (where applicable)
- By model feature buckets (e.g., low / medium / high price tier)

The slice that lights up first when things go wrong is usually a specific cohort, not the aggregate.

#### 3. Lineage and Root Cause

When monitoring fires: "the model is getting worse." Useful follow-on questions:

- Which feature drifted? (data drift)
- Did an upstream pipeline change? (lineage check)
- Did a feature's source schema change? (schema diff)
- Is a specific cohort being affected? (slicing)
- Did model training pull from a different dataset than before? (training data audit)

Modern observability platforms integrate with lineage (OpenLineage) and the model registry so a single click jumps from "alert" to "the training data was different last Tuesday."

### OpenTelemetry GenAI Semantic Conventions

The observability stack for LLMs has been a Wild West of incompatible formats. OTel shipped the **GenAI semantic conventions** (experimental, 2026) to standardize this: a defined vocabulary of span attributes for LLM calls, agent steps, MCP tool calls, and optional content capture.

Key attributes from the `gen_ai.*` namespace:

- `gen_ai.system` — the model provider (`openai`, `anthropic`, `aws_bedrock`, ...)
- `gen_ai.request.model` — which model was requested
- `gen_ai.response.model` — which model actually responded (may differ after failover)
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` — for cost attribution
- `gen_ai.operation.name` — `chat`, `text_completion`, `embeddings`
- Content capture attributes (optional, gated by a flag for PII safety)

The ecosystem is converging: **Datadog, Honeycomb, and New Relic** all support ingestion of GenAI convention spans. **LangChain, CrewAI, and AutoGen** emit OTel-compliant spans natively. **MLflow 3 tracing is OTel-compatible** — if you're already in MLflow, you get standardized trace export for free.

Why this matters operationally: **vendor-neutral tracing means no observability lock-in.** Instrument once, point the exporter anywhere. Switch from Datadog to Honeycomb without rewriting your instrumentation.

```python
import mlflow

# Option A: decorator — MLflow 3 wraps the function in an OTel-compatible span
@mlflow.trace
def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Option B: manual span with gen_ai.* attributes
from opentelemetry import trace
from opentelemetry.semconv.ai import SpanAttributes  # opentelemetry-semantic-conventions-ai

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("llm.chat") as span:
    span.set_attribute(SpanAttributes.GEN_AI_SYSTEM, "openai")
    span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MODEL, "gpt-4o")
    response = call_model(prompt)
    span.set_attribute(SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS, response.usage.prompt_tokens)
    span.set_attribute(SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, response.usage.completion_tokens)
```

For agent workloads, the GenAI conventions extend to tool calls (each tool invocation is a child span with `gen_ai.tool.name` and `gen_ai.tool.call.id`), enabling end-to-end trace trees for multi-step agent executions. This is the instrumentation layer that makes agent observability tractable at production scale — covered in more detail in the agentic systems section of this course.

### What to Build

Take your medium-tier project's monitoring layer and elevate it:

- Add slice-aware monitoring on at least 3 dimensions
- Connect monitoring → lineage → root-cause runbook
- Set up alert routing: severity 1 (paging), severity 2 (Slack), severity 3 (dashboard only)
- Add a one-pager runbook per alert type
- If you have an LLM component: instrument with OTel GenAI semconvs; export to any OTLP-compatible backend

---

## Phase 5 — Governance, Security, and the AI Act (1 week)

The single area separating "I can build ML systems" from "I can run an ML platform at a regulated F50."

### Governance: What F50 Companies Actually Need

- **Model inventory** — every production model registered with owner, purpose, data sources, training data lineage
- **Model cards** — for each model: intended use, training data, performance per slice, known limitations, fairness considerations. Published internally and (sometimes) externally.
- **Approval workflow** — high-risk model deployments require sign-off from a risk committee. Bake the workflow into your registry.
- **Audit trail** — every promotion, every prediction (sampled in high-volume cases), every label feedback. Retain per regulatory requirements (often 7 years for finance/healthcare).
- **Right to explanation** — for credit, insurance, employment models, regulators require explanations of decisions. SHAP / LIME / counterfactual explanations, with a path to human review.

### Regulatory Landscape (As of 2026)

- **EU AI Act** — risk-based; "high-risk" systems (employment, education, credit, law enforcement, medical) require conformity assessment, technical documentation, post-market monitoring. **Timeline matters: prohibited-practices + AI-literacy rules already apply; GPAI obligations since Aug 2025; general enforcement 2 Aug 2026; high-risk systems get until 2 Dec 2027; AI embedded in regulated products until Aug 2028.** Fines up to €35M or 7% global turnover. Know the dates — "is the Act enforced?" is now a trick interview question.

#### EU AI Act: The Technical Controls (Annex IV Checklist)

The regulatory timeline is in the file. Here's the engineering translation: what Annex IV requires from high-risk systems, and what MLOps artifact satisfies each requirement.

| Annex IV Requirement | What it demands | MLOps artifact that satisfies it |
|---|---|---|
| **Risk-management system** | Documented, continuously updated process for identifying and mitigating risks | Risk register in your model registry; reviewed at each promotion gate |
| **Data governance documentation** | Provenance, representativeness, bias examination for training data | Data lineage (OpenLineage), dataset cards, demographic slice reports from your eval harness |
| **Automatic record-keeping / logging** | Traceability through the full model lifecycle; logs must be tamper-evident | Prediction logs + audit trail in immutable storage (S3 Object Lock or equivalent); lineage metadata |
| **Transparency documentation** | Capabilities and limitations disclosed to deployers (not end users — deployers) | Model cards; API documentation with known failure modes; release notes per version |
| **Human oversight mechanisms** | Defined intervention points; humans can override, disable, or correct the system | Approval workflow baked into model registry promotion; kill-switch endpoint in serving layer |
| **Accuracy, robustness, cybersecurity specs** | Declared metrics + measured evidence that the system meets them | Eval report as a formal artifact; adversarial test results; penetration test summary |

The operational insight: every item on this list maps to something a mature MLOps team should build anyway. The EU AI Act is not a compliance tax on top of good engineering — it is a forcing function to productionize what you should have built regardless. The teams that will find compliance easy are the ones that already have model cards, eval reports, audit logs, and a promotion gating workflow.

For high-risk system preparation: build an **evidence pack** — a versioned folder per model that contains model card, eval report, data provenance summary, risk assessment, and the approval sign-off. Every production promotion produces one. The Annex IV conformity assessment is then an audit of your evidence packs, not a scramble to reconstruct history.

- **NYC Local Law 144** — automated employment decision tools must have annual bias audits.
- **Colorado AI Act** — broader, also covers high-risk AI in employment, education, finance.
- **California, Illinois, Texas** — various sector-specific AI regulations.
- **HIPAA** (US healthcare) — PHI protection, audit logging, BAA agreements.
- **GDPR Article 22** — restrictions on solely automated decisions with significant effects on individuals.
- **NIST AI Risk Management Framework** — voluntary US framework increasingly used as a baseline.

You don't need to be a lawyer. You need to know which frameworks apply, what technical controls they require, and how to bake them in.

### Security: The Real ML Attack Surface

- **Model theft** via API extraction (querying enough to clone the model)
- **Adversarial examples** that flip model decisions
- **Data poisoning** in training data
- **Prompt injection** in LLM apps
- **Model inversion / membership inference** — leaking training data through outputs
- **Supply-chain attacks** via pretrained model weights or pip packages

Standard mitigations:

- Rate-limit prediction APIs and require authentication
- Watermark or fingerprint model outputs
- Input sanitization (especially for LLMs)
- Differential privacy in training, where the threat model warrants it
- Model weight signing and verification (Sigstore for ML artifacts is emerging)
- SBOMs for ML pipelines

### What to Build

Add a governance section to one of your existing projects:

- Model card for each registered model
- Audit log table: every promotion and every prediction sampled
- Slice-aware fairness metrics on a protected attribute
- A short "AI risk assessment" doc that a regulator could read

Mention this in your README. It's a senior signal.

---

## Phase 6 — System Design Interviews for MLOps (1 week)

F50 senior MLOps interviews almost always include a system design round. The format: "Design a system that does X." You're judged on clarity, awareness of trade-offs, and the right vocabulary.

### Common Prompts

1. **"Design a real-time fraud detection system for a payments network."**
2. **"Design a recommendation system for a streaming service with 200M users."**
3. **"Design a feature store for an organization with 50 ML teams."**
4. **"Design the serving infrastructure for an LLM API supporting 1000 internal teams."**
5. **"Design a system that retrains a model when it detects performance drift."**
6. **"Design an embedding pipeline for a corpus of 500M documents that needs to support semantic search."**
7. **"Design end-to-end MLOps infrastructure for a brand-new ML org joining a 50-year-old bank."**
8. **"How would you migrate from SageMaker to a self-hosted Kubernetes ML platform without downtime?"**

### LLMOps Design Prompts

These four are the emerging class of LLMOps-specific system design questions. F50 interviewers are moving fast in this direction; practice them explicitly.

1. **"Design an internal LLM gateway for 1,000 teams with cost attribution and failover."**
   *What they're probing:* whether you understand gateway responsibilities beyond "proxy" — per-tenant quota enforcement, token-level chargeback, semantic caching, multi-provider failover with health-check-based routing, and the observability model that makes all of this legible to platform operators.

2. **"Design an eval pipeline that gates deploys for a customer-support agent."**
   *What they're probing:* whether you treat evals as first-class infrastructure rather than a notebook someone runs once. The answer needs: eval dataset versioning, LLM-as-judge metric definition, CI integration (the eval blocks the deploy), regression threshold policy, and how you handle the feedback loop from production failures back into the eval set.

3. **"Design multi-region serving for an EU+US assistant under data residency requirements."**
   *What they're probing:* the six leak surfaces — inference, telemetry, eval pipelines, prompt caches, fine-tune feedback loops, and observability pipelines — must all be residency-aware, not just the inference path. Region-specific model availability (Bedrock model catalogs differ per region), failover automation, and brownout testing are expected.

4. **"Design a fine-tuning platform: multi-tenant isolation, quotas, and leak prevention between tenants' adapters."**
   *What they're probing:* multi-tenancy is the hard part of FTaaS. Expect to discuss: separate LoRA adapter storage per tenant with access controls, training job isolation (no shared GPU memory between tenants), quota enforcement at the job-submission layer, model serving isolation (one tenant's adapter must not influence another's inference), and audit logging that satisfies enterprise security review.

### The Approach That Works

1. **Clarify (5–10 min).** Scale (QPS, data volume, model size), latency budget, freshness, accuracy target, team size, deadline. Pick aggressive numbers — interviewers want to see you handle scale.
2. **High-level architecture (10 min).** Sketch the boxes and arrows. C4 Container-level. Name every component.
3. **Drill into 2–3 components (15–20 min).** Pick the most interesting ones. Discuss trade-offs explicitly.
4. **Failure modes (5 min).** What breaks first under load? What if a component goes down? How do you detect and recover?
5. **Cost (5 min).** Rough unit economics. Knowing the numbers is senior signal.
6. **Operational and org concerns (5 min).** Who owns what? How does the on-call rotation work? Governance?

The mistake to avoid: going deep on a single component for 40 minutes while leaving the rest hand-wavy. Cover the breadth first; let the interviewer pull you into depth.

### Specific Tradeoffs Interviewers Love

- **Batch vs online vs streaming** for feature computation
- **Push vs pull** for online features
- **Centralized vs federated** ownership (model platform vs per-team)
- **Build vs buy** for feature store, monitoring, vector DB
- **Open source vs managed** for serving
- **GPU vs CPU vs custom silicon** for inference
- **Caching layers** at every level
- **Synchronous vs asynchronous** prediction
- **Single multi-task model vs specialized models** per task
- **Real-time training vs scheduled batch retraining**

For each, internalize one paragraph on when each side wins. Practice answering out loud.

### Reading List for MLOps System Design

- **Designing Machine Learning Systems** by Chip Huyen — the canonical book; cover-to-cover, twice.
- **Designing Data-Intensive Applications** by Martin Kleppmann — distributed systems foundations.
- **Machine Learning Systems Design** lecture notes (Chip Huyen, free) — also great.
- **Reliable Machine Learning** by Cathy Chen et al. (O'Reilly).
- **Building Recommendation Systems in Python and JAX** (Bryant, Hawkins) — concrete recommendation case study.
- **The Hugging Face transformers and PEFT documentation** — for LLM patterns.
- **Engineering blog posts** from Netflix, Uber, Lyft, Airbnb, Pinterest, Spotify, DoorDash, Meta — all publish frequent, deep ML platform writeups.

---

## A Word on Specialization

After this phase, you have an honest choice:

1. **Training infrastructure specialist** — distributed training, large-scale GPU operations, training framework internals. Small market, high comp, lives at frontier labs and the biggest F50s.
2. **Inference / serving specialist** — low-latency serving, GPU optimization, KV cache and quantization mastery. Big market, growing with LLMs.
3. **Feature platform specialist** — feature stores, streaming features, point-in-time correctness, data plumbing for ML. High leverage at any large org.
4. **LLM / AI engineering specialist** — RAG, fine-tuning, agents, evals. The hottest sub-discipline as of 2026.
5. **ML platform generalist** — the architect track. Build the platform, design the standards, lead the team. Path is covered in the ML architect track.
6. **MLOps for regulated industries** — finance, healthcare, government. Governance and compliance specialization on top of the technical stack.

Don't pick yet. Build one project from the Fortune 50 portfolio chapter in each of two of these specializations to see what energizes you. Then go deep on that.

---

## When You're Done with This File

You should have:

- A working stack on one major cloud's ML platform (SageMaker / Vertex / Azure ML / Databricks)
- An LLM-flavored project (RAG, fine-tune, or agent) with a real eval harness
- A vector DB project with hybrid search and reranking
- Advanced monitoring on at least one of your projects: slicing, lineage, alert routing
- A governance section in at least one project
- A bookshelf with Designing Machine Learning Systems, DDIA, and one LLM-specific book
- A draft answer to at least 4 of the F50 system design prompts above

Now you're ready to build something that lands you the F50 role. Move on to the Fortune 50 portfolio projects chapter.
