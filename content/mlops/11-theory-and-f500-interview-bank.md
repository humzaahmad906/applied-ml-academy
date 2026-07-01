# 11 — Theory Primers and F500 Interview Bank

The practitioner-focused chapters are about building: build the thing, then study what you built. This chapter is for the opposite mode — you have an interview in two weeks, you need to compress the curriculum into 60 self-contained theory primers, each ending with the questions a Fortune 500 panel actually asks.

Use this chapter two ways:

1. **Per-section review.** After finishing a topic in the practitioner chapters, jump to the matching section here. Read the theory primer. Try the questions out loud. The gap between "I built this" and "I can explain this" is what interviews test.
2. **Final-week sprint.** Block 2 hours / day for 10 days. Go section by section. By day 10 you can verbalize answers to every prompt.

The questions are calibrated to senior IC / staff / principal F500 MLOps interviews — the kind of question that gets a one-paragraph answer if you've worked through the curriculum, and a stammer if you haven't.

---

## How to Read Each Section

Each section follows this shape:

- **The Theory** — a 1–2 page primer covering the underlying concepts. Read this first.
- **The Mental Model** — the one or two pictures you should carry into the interview.
- **Why F500 Asks This** — the production reality the question is probing for.
- **Interview Questions** — graded:
  - 🟢 Phone-screen / mid-level (1–2 min answer)
  - 🟡 Senior / staff (3–5 min answer)
  - 🔴 Principal / architect (5–10 min answer, often a system-design wedge)

Treat 🔴 questions as paragraph-length essays, not bullet-point answers.

---

## Section 1 — Reproducibility and Determinism

### The Theory

In production ML, reproducibility means: same code + same data + same hyperparameters + same seed → same numbers, every time, anywhere. Lacking it, you can't audit, you can't roll back, you can't migrate hardware, you can't debug.

Three layers of reproducibility:

1. **Code reproducibility** — pinned dependencies via lock files (`uv.lock`, `poetry.lock`), code in Git with a specific commit hash.
2. **Data reproducibility** — versioned data, retrievable via DVC pointer or lakehouse time travel.
3. **Runtime reproducibility** — seeded RNGs, deterministic GPU ops (where possible), pinned container images.

Two non-obvious truths:

- **GPU determinism is partial.** Some CUDA ops are nondeterministic by design (atomic adds, certain reductions). Even with `torch.use_deterministic_algorithms(True)`, you get bit-identical results only within a fixed hardware generation, kernel version, and PyTorch build.
- **Reproducibility ≠ replication.** "Reproducible" means you can re-run and verify. "Replicable" means an independent team can reach the same conclusion. The second is stronger; the first is a prerequisite.

### The Mental Model

```
Code (commit hash) ──┐
Data (DVC hash)      ├──► Pipeline ──► Artifact (hash) ──► Numbers
Hyperparams + seed ──┤
Container image (sha)┘
```

Every input above must be pinned. The output artifact's hash is the "did we reproduce" check.

### Why F500 Asks This

Regulators (SR 11-7 for banks, FDA SaMD for healthcare) require auditable model lineage. Audit means "show us this model was trained on exactly that data with exactly that code." If you can't, the model can't ship.

### Interview Questions

🟢 What does a lock file give you that `requirements.txt` doesn't?

🟢 Why isn't `random_state=42` alone enough for reproducible deep learning?

🟢 You return to a six-month-old model. How do you reproduce its metrics?

🟡 Walk me through everything you'd pin in a training pipeline to make it auditable. What can't you pin?

🟡 Where does GPU non-determinism come from, and which uses can tolerate it?

🟡 You promote a model. Six months later the auditor asks for the training data. How does your pipeline answer this in one minute, not one week?

🔴 Design an audit-grade ML training pipeline for a US bank under SR 11-7. What gets logged, what gets stored where, with what retention, and what's the query path when an examiner asks "what's the data lineage for the loan decisioning model in production on April 14"?

---

## Section 2 — Data Versioning, DVC, and Lakehouse Patterns

### The Theory

You version data because:

- Reproducing a model requires its training data.
- "The model got worse" often reduces to "the data changed."
- Compliance asks "which version of the data was used."

Four approaches:

| Approach | When it fits |
|---|---|
| **DVC** | Project-scoped, small-medium teams, Git-native workflow |
| **Lakehouse time travel** (Iceberg, Delta) | Data already in lakehouse; time travel is free |
| **LakeFS / Nessie / Pachyderm** | Git-like branches over object storage; shared-lake orgs |
| **S3 date-stamped prefixes** | Smallest teams; lowest ceremony |

DVC writes a small `.dvc` pointer to Git; the actual data lives in S3/GCS/Azure. `dvc pull` retrieves data on a fresh checkout. `dvc.yaml` declares pipelines (stage dependencies + outputs); `dvc repro` re-runs only stages with changed inputs.

Iceberg / Delta give you SQL-level time travel: `SELECT * FROM events FOR VERSION AS OF 17` or `... FOR TIMESTAMP AS OF '2026-04-01'`. This is the cleanest data-versioning UX once your data already lives in a lakehouse.

### The Mental Model

```
   Git (code)            DVC / Lakehouse (data)
   ────────              ──────────────────────
   commit abc123  ◄──►   data version 17
                            │
                            ▼
                    [Pipeline] ──► metrics, model, predictions
```

Both axes versioned independently and tied at training time.

### Why F500 Asks This

Data drift, retraining audits, and compliance reviews all hinge on data versioning. A team without it is mid-level forever.

### Interview Questions

🟢 What does `dvc add data/raw/train.csv` actually do?

🟢 You change a feature definition; which `dvc.yaml` stages re-run?

🟢 Compare DVC, LakeFS, and Iceberg time travel for ML data versioning.

🟡 Explain why "the model got worse" often reduces to a data versioning question.

🟡 Walk through `dvc exp run` end to end and what artifacts it produces.

🔴 Design a data versioning strategy for a 200-engineer ML org with 50 ML pipelines, regulated finance workloads, and a 7-year retention requirement. Pick one approach and justify against the alternatives.

---

## Section 3 — Experiment Tracking and Model Registry

### The Theory

Experiment tracking captures *every input and output of a training run* — code version, data version, hyperparameters, environment, metrics, artifacts. Model registry is the next layer: which versioned artifacts are eligible for which environment.

MLflow's components:

- **Tracking** — log params, metrics, artifacts, environment per run.
- **Models** — a packaging format that includes signature + environment.
- **Registry** — versioned, aliased model store.
- **Projects** — runnable-package format (rarely used; Docker beats it).

In 2026, MLflow uses **aliases** (e.g., `@champion`, `@challenger`), not stages. Aliases let you have multiple labeled versions simultaneously and swap them atomically.

The pattern for production:

1. Every training run logs into Tracking.
2. The best run gets registered as a Model.
3. The registered Model gets an alias: `@challenger`.
4. Tests pass → alias `@champion` moves to it (atomic).
5. Serving code loads `models:/income_classifier@champion` — no version pinned.

### The Mental Model

```
[Run]  ──register──►  [Model v17]  ──alias──►  [@champion]
                                                  │
                                                  ▼
                                            [Serving] (loads @champion)
```

Promotion is an alias swap. Rollback is the same.

### Why F500 Asks This

Every F500 model deployment must have an audit trail. The registry is that audit trail. Engineers who can't talk about model lifecycle credibly don't get senior MLOps offers.

### Interview Questions

🟢 What's the difference between MLflow tracking, models, and registry?

🟢 Why are MLflow aliases preferred over stages in 2026?

🟡 Walk me through promoting a model to production atomically with rollback.

🟡 What gets logged per training run for it to be truly auditable later?

🟡 You discover at week 4 of a deployed model that its training data was wrong. Walk through the rollback + retrain.

🔴 Design a multi-tenant ML registry for a 50-team org. Cover namespacing, RBAC, approval workflow, model card enforcement, alias governance, and how new models get evaluated before promotion.

---

## Section 4 — Feature Pipelines and the Feature Store

### The Theory

A feature is a function from raw data to a model input. A feature *pipeline* runs that function consistently. A feature *store* coordinates pipelines so that the same definition produces the same value online and offline.

The most insidious bug in production ML: **training-serving skew** — the feature computed at training time differs from the feature computed at serving time. The model trained on training-time data sees something different at inference.

Two flavors:

- **Offline materialization** — features computed in batch, written to Parquet / Iceberg, used to build training sets.
- **Online materialization** — features written to Redis / DynamoDB / Bigtable, looked up at sub-10ms by the inference service.

The "feature store" abstraction (Feast, Tecton, internal builds) is a coordination layer — it doesn't *compute* features; it *defines* them and lets both worlds materialize the same definition.

**Point-in-time correctness** is the headline feature operation. For a training row at time `t`, look up the feature *as of t* (not the latest value). Implemented via an "as-of join" — SQL pattern at the bottom. Without it, you train on data from the future, and your model looks brilliant in eval and fails in prod.

### The Mental Model

```
Raw events
    │
    ▼
Feature definition (one set of functions)
    │
   ┌┴──────────────┐
   ▼               ▼
Offline (Iceberg)  Online (Redis)
   │               │
   ▼               ▼
Training set    Inference lookup
```

Same definition, two materializations, two consumers.

### Why F500 Asks This

Feature stores are the most-misunderstood part of MLOps. Senior interviews dig here because the wrong answer costs production accuracy.

### Interview Questions

🟢 Define training-serving skew with one example.

🟢 What's an as-of join and why do you need it for training data?

🟢 Online store vs offline store — what's in each and why.

🟡 You're given a feature `user_avg_purchase_last_30d`. Walk through how to compute it correctly point-in-time.

🟡 When would you reach for Feast vs Tecton vs hand-rolled?

🟡 How do you reconcile online and offline features to detect drift between them?

🔴 Design a feature store for a 50-team ML org with batch and streaming features, sub-10ms online reads, point-in-time-correct training data generation, multi-tenant quota isolation, and OpenLineage-traceable lineage end to end.

---

## Section 5 — Orchestration (Prefect / Airflow / Dagster)

### The Theory

Orchestration tools schedule, retry, monitor, and visualize multi-step pipelines. Three contenders in 2026:

- **Airflow** — mature, ubiquitous, KubernetesExecutor for isolation. The F500 default.
- **Prefect** — modern, Pythonic, hybrid (local + control plane).
- **Dagster** — Software-Defined Assets (data + ML as first-class entities), strong typing, asset lineage.

Patterns that work:

- Each task does *one thing*; passes paths/URIs, not large objects.
- Retries with exponential backoff and jitter; bounded retries to avoid infinite loops.
- Idempotency: tasks must be safe to re-run for any date.
- Sensors in `reschedule` mode, not `poke` mode, to free worker slots while waiting.

ML pipelines specifically:

```
Pull data → Validate → Feature engineer → Train → Eval → Register →
Promote (canary alias) → Test → Promote champion → Notify
```

### The Mental Model

```
                   ┌─────────┐
                   │Scheduler│
                   └────┬────┘
                        ▼
                  [DAG / Flow]
                    │  │  │  │
                    ▼  ▼  ▼  ▼
                   Task graph w/ retries + lineage
```

The orchestrator is the brain; tasks are the muscles.

### Why F500 Asks This

Every F500 ML pipeline runs on one of these tools. Engineers should know one fluently and have opinions about the others.

### Interview Questions

🟢 What is `XCom` in Airflow and why is it usually wrong to pass large objects through it?

🟢 Why use `mode="reschedule"` for an Airflow sensor?

🟢 Idempotency — what's an idempotent task and why does it matter?

🟡 Walk me through a daily training DAG: tasks, retries, alerts, lineage.

🟡 Compare Prefect, Airflow, Dagster. When does each win?

🟡 You have a 7-day backfill that must be idempotent. How do you ensure no duplicate writes?

🔴 Design an orchestration layer for a 200-team ML org. Cover: tool choice, multi-tenancy, isolation, secret management, observability, lineage, on-call ergonomics, and the migration path from a hodgepodge of cron jobs.

---

## Section 6 — CI/CD/CT for ML

### The Theory

Three Cs:

- **CI** — continuous integration. Every PR: lint, type check, unit tests, data tests, model tests on fixtures.
- **CD** — continuous delivery. Every merge: build versioned container image, auto-deploy to staging.
- **CT** — continuous training. On schedule (or trigger): run real training, register candidate, run model tests, promote.

CT is the MLOps-specific addition. Plain software has CI/CD; ML adds CT because *the artifact (the model) drifts even when the code doesn't*.

Tests for ML, four categories:

1. **Unit tests on transforms** — pure-function feature code.
2. **Data tests** — schema, ranges, nulls, distributions.
3. **Model tests** — behavioral (known-good inputs), invariance (irrelevant perturbations don't change output), directional (relevant perturbations change in the right direction), fairness (per-slice metrics).
4. **Integration tests** — end-to-end pipeline against fixtures.

### The Mental Model

```
PR              Merge to main      Schedule
│                 │                  │
▼                 ▼                  ▼
CI                CD                 CT
(lint, test)      (build, deploy)    (train, test, register)
                                       │
                                       ▼
                          Gated promotion + monitoring
```

The third arrow is unique to ML.

### Why F500 Asks This

Without CT, you ship a model once and pray. Senior MLOps interviews probe whether you understand the model-drift / retraining loop.

### Interview Questions

🟢 What's the third C in CI/CD/CT?

🟢 Why test model behavior, not just metrics?

🟢 What's an invariance test? Give a concrete example.

🟡 Walk me through a GitHub Actions workflow that covers all three Cs.

🟡 Your CT pipeline runs nightly. It produces a bad model on Tuesday. Walk through what catches it before it serves real traffic.

🟡 OIDC for cloud auth in CI — what does it solve and what would you have without it?

🔴 Design CI/CD/CT for an organization that ships 30 production models with regulatory model risk requirements. Cover gating, approvals, evidence packs, audit logs, rollback playbooks, and the human review steps your pipeline preserves.

---

## Section 7 — Docker and Containerization for ML

### The Theory

Containers solve the "works on my laptop" problem with multipliers in ML:

- CUDA + cuDNN + driver versions must align with framework versions.
- System libraries (libgomp, MKL, OpenBLAS) affect performance and correctness.
- Hardware-specific builds (Apple Silicon vs x86-64 vs ARM64).

Production-quality Dockerfile principles:

1. **Multi-stage builds** — final image contains only runtime, not toolchain.
2. **Layer order** — slow-changing files first (deps, lock file), fast-changing last (source).
3. **`--frozen` installs** — never recompute lock file during build.
4. **Non-root user** — required by Kubernetes Pod Security Standards.
5. **`PYTHONUNBUFFERED=1`** — logs appear in real time, not in lumpy bursts.

For GPU workloads: base off `nvidia/cuda:X.Y.Z-cudnn-runtime-ubuntuW.W`. Run with `docker run --gpus all`. On Kubernetes, request `nvidia.com/gpu: 1`.

### The Mental Model

```
   Build stage              Runtime stage
   ───────────              ─────────────
   Compiler                 .venv (from build)
   Build tools     copy ──► src
   Lock + sync              minimal libs
   Source                   non-root user
                            CMD
```

### Why F500 Asks This

Every production ML service runs in a container. Bad Dockerfiles are an immediate signal of weak production experience.

### Interview Questions

🟢 What does a multi-stage Docker build buy you?

🟢 Why copy `pyproject.toml` before `src/` in a Dockerfile?

🟢 What does `PYTHONUNBUFFERED=1` do and when does it matter?

🟡 Walk me through a production-quality Python Dockerfile for an ML service.

🟡 Your GPU Dockerfile builds locally but fails to use the GPU on the cluster. What are the top three diagnostics?

🔴 Design the container build pipeline for a 50-model ML org: caching, signing, SBOMs, vulnerability scanning, multi-arch builds, multi-cloud registry strategy.

---

## Section 8 — Serving Architectures (FastAPI, KServe, BentoML, Triton, vLLM)

### The Theory

Pick the serving framework by workload shape:

| Tool | Strength | When |
|---|---|---|
| FastAPI + uvicorn | Simple, batteries-out | Single-model, low-medium scale |
| BentoML | Python-native packaging, multi-framework | Mid-scale, great DX |
| KServe | K8s-native, multi-framework, scale-to-zero | Production K8s standard |
| Triton | GPU-optimized, dynamic batching, ensembles | High-throughput GPU |
| vLLM / TGI / SGLang | LLM-specific (continuous batching, PagedAttention) | LLM serving |
| Ray Serve | Python-first, composition | Ray-shop standard |

The patterns to know:

- **Shadow traffic** — route real prediction requests to both current and candidate; return current's result; log candidate's; compare offline.
- **Canary** — route X% of traffic to candidate; scale up if metrics hold.
- **A/B test** — split traffic, measure business outcomes with power analysis.
- **Multi-armed bandit** — adaptive A/B; reweights based on observed performance.
- **Blue/green** — twin environments, switch all at once. Less popular for ML than canary.

Latency engineering:

| Component | Typical time |
|---|---|
| Network (client → LB) | 5 ms |
| TLS, LB routing | 10 ms |
| Service handling | 10 ms |
| Feature lookup (Redis) | 5–15 ms |
| Model inference | 20–100 ms |
| Post-processing | 5 ms |
| Return | 5 ms |

The big knobs: batch, quantize, compile, right-size hardware, cache.

### The Mental Model

```
Client ──► LB ──► Service ──► [Feature lookup] ──► [Model] ──► Response
                                        │            │
                                        ▼            ▼
                                  [Online store] [Compiled engine]
```

### Why F500 Asks This

Serving is half the MLOps job. Knowing the framework landscape and the latency math is non-negotiable for senior interviews.

### Interview Questions

🟢 FastAPI vs BentoML vs KServe — pick a framework for sub-100ms ML serving.

🟢 What's shadow traffic? Why is it useful?

🟢 What's a canary deployment vs a blue/green deployment?

🟡 Walk me through micro-batching at the service layer. Why does it often improve both throughput and latency?

🟡 KServe scale-to-zero — what's the cold-start problem and how do you mitigate?

🟡 You need P95 < 50ms at 5000 RPS for a CV model. Walk through your stack.

🔴 Design serving infrastructure for tens of millions of requests/day across tens of models, with SLOs, multi-region failover, gradual rollouts, observability, and per-tenant quotas.

---

## Section 9 — Monitoring and Drift Detection

### The Theory

Five dimensions of ML monitoring:

1. **System health** — latency, error rate, throughput, resource utilization.
2. **Data drift** — input feature distributions over time.
3. **Concept / performance drift** — output quality where labels exist.
4. **Prediction drift** — output distribution.
5. **Business metrics** — the actual KPI.

Drift detection metrics:

- **PSI (Population Stability Index)** — most common in finance. PSI > 0.25 = significant.
- **KS statistic** — continuous features.
- **Chi-squared** — categorical features.
- **JS / KL divergence** — information-theoretic.

For DL: PSI on embedding distance distributions; KS on per-class confidence distributions; per-slice accuracy regression.

The reference dataset problem: what do you compare current data to?

- The training set (loses sensitivity over time).
- A rolling window of recent prod (can mask gradual shifts).
- A held-out golden set (good for behavioral tests; not distribution drift).

### The Mental Model

```
Production
   │ logs predictions + features
   ▼
[Monitoring job] ──► [Drift metrics in TSDB]
   │                       │
   │                       ▼
   │                  [Dashboard]
   ▼                       │
[Per-day reports]           ▼
                       [Alerts → Slack / pager]
                            │
                            ▼
                       [Trigger retraining]
```

Monitoring closes the loop on CT.

### Why F500 Asks This

Without monitoring, you don't know when the model has degraded. Sensors-asleep ML is the most common F500 production failure.

### Interview Questions

🟢 What are the three (or five) kinds of drift?

🟢 How is PSI computed?

🟢 Why is concept drift harder to detect than data drift?

🟡 Walk me through a drift monitoring stack: logging, computation, dashboarding, alerting, retraining trigger.

🟡 Your offline metrics are stable but online business metrics drop. Walk the diagnostic.

🟡 What's a "golden set" and where does it fit in monitoring?

🔴 Design ML observability for a 200-team org with 1000+ models, multi-region serving, regulated workloads, slice-aware analysis, and integration with the model registry for one-click root-cause investigation.

---

## Section 10 — Distributed Training (DDP, FSDP, ZeRO, TP, PP)

### The Theory

The five parallelism strategies:

| Strategy | Splits | Use |
|---|---|---|
| DDP | Batches across replicas | Default multi-GPU |
| FSDP / ZeRO-3 | Params + grads + optimizer state | Model > one GPU |
| Tensor Parallel | Layer weight matrices | Single layer > one GPU |
| Pipeline Parallel | Layers across GPUs | Very deep models |
| Expert Parallel | MoE experts | MoE only |

**FSDP / ZeRO-3** is the 2026 standard for any single-model training that doesn't fit on one GPU. Shards everything; pays in communication.

**3D parallelism** = TP within a node (NVLink-fast) + PP across nodes + DP/FSDP across pipeline replicas. Frontier models use this.

Memory math (mixed precision Adam):

| Component | Bytes per param |
|---|---|
| FP16 working + FP32 master | 6 |
| Gradients (FP16) | 2 |
| Optimizer state (Adam m, v in FP32) | 8 |
| Total | **16P** before activations |

For Llama-2-7B: ~112 GB before activations. FSDP/ZeRO-3 divides by N GPUs.

LoRA flips this: train ~0.5% of params, optimizer and gradients shrink ~200x. A 7B model fine-tunes on a single 24 GB GPU.

### The Mental Model

```
DDP:   each GPU has full model, full optimizer; gradients all-reduce
FSDP:  each GPU has 1/N of params + 1/N optimizer; all-gather then compute
TP:    one layer's weight split across GPUs; all-reduce per layer
PP:    each GPU holds a stage; activations flow forward, grads back
```

### Why F500 Asks This

Training cost dominates ML budgets at frontier labs and LLM-heavy F500s. Senior interviews dig hard here.

### Interview Questions

🟢 DDP vs FSDP — when do you reach for FSDP?

🟢 What's the relationship between FSDP and DeepSpeed ZeRO-3?

🟢 Why does mixed-precision training save memory and compute?

🟡 Walk me through GPU memory during Adam-based training. Where does each chunk go?

🟡 You have 8×A100-80GB and a 13B parameter model. How do you fine-tune?

🟡 Tensor parallel vs pipeline parallel — when do you reach for each?

🟡 Activation recomputation — what does it trade?

🔴 Design training infrastructure for an org that trains 5 models per week ranging from 7B to 70B parameters with mixed budget priority. Cover scheduling, preemption, checkpointing, multi-tenancy, observability, and the failure-recovery story.

---

## Section 11 — GPU Operations and Inference Optimization

### The Theory

GPU memory during inference is dominated by:

1. Model parameters (in chosen precision).
2. KV cache for autoregressive models (Transformers / LLMs).
3. Activations for the forward pass.

For LLM inference, the KV cache often beats parameters as the bottleneck:

```
KV cache size = 2 × N_layers × seq_len × hidden_dim × bytes_per_element
```

For Llama-2-7B at 4096 context in FP16: ~2 GB per request. At 100 concurrent requests: 200 GB.

The 2026 inference toolkit:

- **Continuous batching** — new requests join the batch at every token. Throughput 5–10x of static batching.
- **PagedAttention** — pages the KV cache like virtual memory; eliminates fragmentation.
- **Prefix caching** — reuse KV state for shared prefixes (system prompt, RAG context).
- **Speculative decoding** — small draft model proposes, big model verifies in parallel. 2–4x speedup.
- **Quantization** — INT8 (AWQ, GPTQ), INT4 (Marlin), FP8 (Hopper+). 2–4x throughput, < 1% quality loss.

For CV: the path is PyTorch → ONNX → TensorRT. 1.5–3x speedup typical.

### The Mental Model

```
Single GPU:
  Params (quantized)
  KV cache (paged, possibly quantized)
  Activations (forward only)
  Workspace (small)
                      ↑
                      └── all fit because we paged + quantized + batched
```

### Why F500 Asks This

Inference cost is roughly half of total ML spend. The engineer who can cut it 60% is the engineer who gets promoted.

### Interview Questions

🟢 Why is the KV cache often the bottleneck for LLM inference?

🟢 What does PagedAttention solve?

🟢 Continuous batching — why does throughput improve dramatically?

🟡 Walk me through quantization options for a 7B LLM. What do you pick and why?

🟡 Speculative decoding — describe end to end. When does it help, when does it hurt?

🟡 You move a CV model from FP32 PyTorch to INT8 TensorRT. What's the workflow and what can break?

🔴 Design an LLM inference cluster serving 50 fine-tuned variants of a base model, at 10K concurrent users, sub-second TTFT, with cost attribution per variant and quality regression alerts. Cover: hardware choice, serving framework, batching, caching, autoscaling, fallback, observability.

---

## Section 12 — Kubernetes for ML

### The Theory

K8s primitives that matter for ML:

| Resource | What |
|---|---|
| Pod | Smallest unit; one or more containers |
| Job | Run-to-completion pod |
| CronJob | Scheduled Job |
| Deployment | Long-running stateless replicas |
| Service | Stable DNS + load-balance over pods |
| ConfigMap / Secret | Config / secrets injected |
| PV / PVC | Durable storage |
| HPA / VPA | Pod autoscalers |
| Custom Resource (CRD) | Operator-defined (PyTorchJob, RayCluster, InferenceService) |

ML-specific:

- **NVIDIA device plugin** exposes `nvidia.com/gpu` as schedulable.
- **MIG** partitions A100/H100 into smaller schedulable instances.
- **Karpenter** (AWS) / **Cluster Autoscaler** for node count.
- **KEDA** for event-driven autoscaling on Kafka lag, queue depth, etc.

Operators to know: Kubeflow Training Operator, KubeRay, KServe, Spark Operator, Flink Operator, Argo Workflows.

GitOps with Argo CD / Flux: manifests in Git; controller reconciles cluster to match. `kubectl apply` from CI is anti-pattern.

### The Mental Model

```
Git (manifests)
    │
    ▼
Argo CD / Flux watches and reconciles
    │
    ▼
Kubernetes API
    │
    ▼
Pods / Jobs / Services / GPU / Operators
```

### Why F500 Asks This

Every F500 ML platform runs on K8s. Period.

### Interview Questions

🟢 Liveness vs readiness probes — when does each fire?

🟢 What's a CRD?

🟢 How do you request a GPU in a pod spec?

🟡 Walk me through a Job manifest for a training run with checkpointing.

🟡 MIG — when do you use it?

🟡 Multi-tenancy on a shared GPU cluster — what do you set up?

🔴 Design a multi-cluster K8s strategy for an org with 200 ML engineers, 50 training jobs running simultaneously, 200+ inference services, multi-region, multi-cloud. Cover: cluster topology, GPU pooling, fairness, isolation, networking, GitOps, secrets, observability.

---

## Section 13 — Streaming Features (Kafka, Flink)

### The Theory

You need streaming features when freshness matters: fraud (seconds), recommendations (this session), demand pricing (real time).

Architecture:

```
Events ──► Kafka ──► Flink (windowed feature compute) ──► Online store ──► Service
```

Flink concepts:

- **Watermarks** — "I've seen all events up to time T"; closes windows.
- **State backends** — HashMap (heap), RocksDB (disk-spillable for TB-scale state).
- **Checkpoints** — periodic durable snapshots; recovery point.
- **Savepoints** — user-triggered snapshots; lets you upgrade jobs.
- **Exactly-once via two-phase commit** — requires sink support.

Kafka KRaft mode removed Zookeeper; standard for new deployments.

Online/offline consistency: same transformations expressed once, run in both modes. Pure Beam, Flink in batch mode, or a feature store with a unified definition.

### The Mental Model

```
Event time vs processing time:
   Event time = when it happened
   Processing time = when Flink saw it
Watermark progresses event time; windows close on watermarks.
```

### Why F500 Asks This

Streaming features separate mid-level from senior MLOps engineers. Most candidates have not done this.

### Interview Questions

🟢 What's a watermark?

🟢 Difference between event time and processing time?

🟢 Why does exactly-once require two-phase commit at the sink?

🟡 Walk through a windowed feature computation: tumbling window of count + sum per user per minute.

🟡 Online/offline consistency — how do you guarantee a streaming feature matches its batch counterpart?

🟡 Flink savepoint workflow — what does it enable that checkpoints don't?

🔴 Design real-time fraud detection at 10K transactions/sec with sub-100ms decision latency, exactly-once labeling, model retraining triggered by drift, and a labeling UI feedback loop.

---

## Section 14 — LLMOps Foundations

### The Theory

LLMOps differs from classical MLOps:

| Classical | LLMOps |
|---|---|
| Train on your data | Adapt someone else's model |
| AUC / F1 metrics | Faithfulness, helpfulness — hard to score |
| Single output | Sequence; variable length |
| Pennies per inference | Dollars per million tokens |
| Deterministic given seed | Stochastic by design |
| Ground truth labels | LLM-as-judge or human |
| Retrain when drift | Re-prompt, re-retrieve, re-fine-tune |

The LLM serving stack: **gateway** → input filters → router → retrieval → cache → prompt registry → LLM (hosted or self-hosted) → output filters → logging.

Fine-tuning hierarchy:

- **SFT** — supervised, baseline.
- **LoRA / QLoRA** — adapter weights; 1–10% of full FT compute.
- **DPO** — preference pairs; cheaper, often equivalent to RLHF.
- **RLHF (PPO)** — heavy and finicky; mostly replaced by DPO/ORPO/KTO/GRPO.

Evaluation hierarchy:

- Automated metrics (BLEU/ROUGE/perplexity) — weak.
- LLM-as-judge — biased but cheap.
- Programmatic (schema/regex/test pass) — strong where applicable.
- Pairwise + ELO — strong for comparison.
- Human with rubrics — gold standard, expensive.

### The Mental Model

```
              [Cost]              [Quality]
                │                     │
Hosted frontier (GPT-4o, Claude)      │  ←── max quality, max cost
                │                     │
Hosted small (GPT-4o-mini)            │
                │                     │
Self-hosted 70B                       │
                │                     │
Self-hosted 7B-LoRA distilled         │  ←── low cost, often "good enough"
```

The architect's job: routing requests onto the right tier of this stack.

### Why F500 Asks This

Every F500 spun up dozens of LLM-powered features in the last two years. The bottleneck is the platform around the model.

### Interview Questions

🟢 SFT vs DPO — when do you reach for each?

🟢 What's continuous batching?

🟢 Why does the KV cache page like virtual memory?

🟡 Walk me through a RAG pipeline end to end. Where does most quality come from?

🟡 Build an eval harness for an LLM-powered customer-support assistant. What does success look like?

🟡 Compare vLLM, TGI, SGLang, TensorRT-LLM for serving.

🟡 Prompt injection — three defenses, where each fails.

🟡 Multi-LoRA serving — what does it solve?

🔴 Design an internal LLM platform for a 200-team enterprise. Cover: model layer (hosted + self-hosted), gateway, routing, prompt registry, eval, observability, cost attribution, governance, fallback, multi-region.

---

## Section 15 — Vector Databases and Retrieval

### The Theory

Vector DBs do approximate nearest-neighbor search. Main algorithms:

- **HNSW** — graph; fastest; more memory.
- **IVF + PQ** — clusters + product quantization; cheaper.
- **DiskANN** — disk-backed; billion-scale on cheap hardware.

Players: pgvector, Pinecone, Weaviate, Qdrant, Milvus, LanceDB, Vespa, OpenSearch k-NN.

Hybrid search:

```
query → embed ──► vector top-100 ──┐
        └──────► BM25 top-100 ─────┴──► merge or rerank ──► top-10
```

Reranking with a cross-encoder (BGE-Reranker, Cohere Rerank) on the top 50–200 candidates is a big quality lift for cheap compute.

Embedding model choice matters: text-embedding-3-large vs BGE-large-v1.5 vs domain-fine-tuned. Dimensionality: 1536-dim has capacity; 256/384-dim is often enough and 4–6x cheaper.

### The Mental Model

```
Recall vs Latency vs Memory — pick two.
HNSW    → recall + latency, costly memory
IVF-PQ  → recall + memory, slower
DiskANN → memory + scale, slower
```

### Why F500 Asks This

Every RAG / search / personalization system has a vector DB. Knowing trade-offs is table stakes.

### Interview Questions

🟢 HNSW vs IVF — when do you pick each?

🟢 What does product quantization buy you?

🟢 Reciprocal Rank Fusion vs weighted score normalization for hybrid search.

🟡 Walk me through tuning HNSW parameters (M, ef) for recall@10 vs latency.

🟡 You need to upgrade your embedding model from v1 to v2. What's the migration path for an existing 100M-vector index?

🟡 Pinecone vs Qdrant vs pgvector — pick for a 50M-document RAG system.

🔴 Design an embedding pipeline + retrieval layer for a 500M-document corpus with weekly refresh, hybrid search, reranking, multi-tenant isolation, sub-200ms P99, and embedding-model rolling upgrade.

---

## Section 16 — Governance, Compliance, AI Act

### The Theory

Governance = "who's accountable for what this model does." Compliance = "we've documented that we did the right thing."

What every regulated production model needs:

- **Model inventory** — owner, purpose, lineage, slice metrics, limitations.
- **Model card** — public-facing description of intended use, performance, ethics.
- **Audit trail** — every promotion, every prediction (sampled), every label feedback. 7-year retention common.
- **Approval workflow** — high-risk models gated by risk committee.
- **Explanation and recourse** — SHAP / LIME / counterfactuals for credit, insurance, employment.

The 2026 regulatory landscape:

- **EU AI Act** — risk-based; high-risk systems require conformity assessment, technical documentation, post-market monitoring.
- **NYC Local Law 144** — bias audits for employment AI.
- **Colorado AI Act, California regulations** — emerging US state-level.
- **GDPR Article 22** — restrictions on solely automated decisions.
- **HIPAA** — PHI protection.
- **SR 11-7** — US banks; model risk management.
- **NIST AI RMF** — voluntary; increasingly a baseline.

### The Mental Model

```
                  [Risk classification]
                          │
            ┌─────────────┼──────────────┐
            ▼             ▼              ▼
       [Minimal]    [Limited]       [High-risk]
                                          │
                          ┌───────────────┴───────────────┐
                          ▼               ▼               ▼
                  [Risk mgmt]      [Data gov]      [Tech docs +
                                                   monitoring]
```

The higher the risk tier, the heavier the process.

### Why F500 Asks This

The architect who can't talk fluently about governance can't run an ML platform at a regulated F500.

### Interview Questions

🟢 What's a model card?

🟢 What does GDPR Article 22 restrict?

🟢 SR 11-7 — one sentence.

🟡 Walk me through what an EU AI Act high-risk model's documentation looks like.

🟡 You ship an automated credit decisioning model. What governance must exist before launch?

🟡 Bias audit for an employment AI system — what's in scope, what's out, who signs off?

🔴 Design the governance layer of an ML platform for a US bank that wants to ship 30 generative-AI use cases in 2 years. Cover: risk classification, intake, MRM workflow, evaluation evidence, monitoring, escalation, audit retention, regulator reporting.

---

## Section 17 — Security for ML

### The Theory

The ML attack surface:

- **Model extraction** — clone via API queries.
- **Adversarial examples** — perturbations that flip decisions.
- **Membership inference** — was X in the training set?
- **Model inversion** — reconstruct training data from outputs.
- **Data poisoning** — corrupt training data.
- **Prompt injection** — bypass system prompt in LLMs.
- **Indirect prompt injection** — malicious instructions in retrieved content.
- **Tool abuse** in agents.
- **Supply-chain** — malicious weights, packages, images.

Mitigations:

- AuthN/AuthZ on every inference endpoint; rate limit.
- Differential privacy when threat model warrants (Opacus).
- Watermarking outputs.
- Input sanitization + output filtering.
- Sigstore for model weights; SBOMs for ML pipelines.
- OIDC for short-lived CI credentials; secret manager for runtime.

### The Mental Model

```
[Threat] ──► [Attack vector] ──► [Mitigation]

Model theft  ──► API extraction        ──► Rate limit + auth
Adversarial  ──► Crafted inputs        ──► Robust training; input filter
Poisoning    ──► Training data corrupt ──► Provenance + audit
Inversion    ──► Output queries        ──► DP training
Prompt inj   ──► User input            ──► Input filter + jail confinement
```

### Why F500 Asks This

Senior interviews probe whether you've thought about attackers, not just users.

### Interview Questions

🟢 What's prompt injection? Direct vs indirect?

🟢 Differential privacy — one-sentence definition.

🟢 Why is rate limiting an anti-model-extraction defense?

🟡 Walk me through three prompt-injection defenses and where each fails.

🟡 Adversarial example in CV — how do you generate one, how do you defend?

🟡 Your LLM-powered agent has tool access to a database. Walk me through your safety controls.

🔴 Design the security architecture for an internal LLM platform with hosted + self-hosted models. Cover: input filtering, output filtering, tool sandboxing, audit, secret hygiene, supply chain, attack monitoring, incident response.

---

## Section 18 — Cost and FinOps for ML

### The Theory

Where the bill goes:

| Layer | % of total (typical) |
|---|---|
| Training | 20–40% |
| Inference | 30–60% |
| Storage | 5–15% |
| Egress | 5–15% |
| Vendors (LLM APIs, observability) | 5–25% |
| Tooling | 1–5% |

Inference levers: quantize, distill, batch, route, cache, scale-to-zero, right-size hardware.

Training levers: spot instances, early stopping, smarter HPO, transfer learning.

LLM-specific: prompt compression (LLMLingua), self-hosting once spend > ~$10K/month, distillation, semantic caching.

The classic anti-patterns: idle GPUs, endpoints with min-replicas > 0, cross-region transfer, HPO without early stopping, hot-tier log storage forever.

### The Mental Model

```
                    Cost per inference
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
        Hardware       Software       Process
        - right-size   - batch        - cache
        - quantize     - compile      - route
        - serverless   - share GPU    - scale-zero
```

### Why F500 Asks This

CFOs see the ML bill. Engineers who can't talk cost can't get to staff.

### Interview Questions

🟢 What's the biggest single cost driver for LLM inference?

🟢 Why is scale-to-zero risky for production serving?

🟡 Walk me through five cost-cutting moves on an inference cluster.

🟡 Self-host vs hosted LLM — when does the breakeven flip and how do you model it?

🟡 Egress cost — what's the most common architecture mistake that creates it?

🔴 Take an organization spending $5M/year on ML, broken roughly 50/30/20 inference/training/storage. Design a year-long cost reduction program to cut 35% without regressing quality. Cover diagnostics, sequencing, risk, validation, organizational change.

---

## Section 19 — System Design for ML Specifically

### The Theory

The 45-minute interview rhythm:

1. **Clarify (5–10 min).** Scale, latency, freshness, accuracy, team, deadline. Always.
2. **High-level architecture (10 min).** Boxes and arrows. Container-level C4.
3. **Drill into 2–3 components (15–20 min).** Trade-offs explicit.
4. **Failure modes (5 min).** What breaks first.
5. **Cost (5 min).** Rough unit economics.
6. **Operational and org (5 min).** Who owns what.

Trade-offs interviewers love:

- Batch vs online vs streaming for features.
- Push vs pull for online features.
- Centralized vs federated org structure.
- Build vs buy.
- OSS vs managed serving.
- GPU vs CPU vs custom silicon.
- Caching at every layer.
- Sync vs async prediction.
- Single multi-task vs specialized models.
- Real-time vs scheduled retraining.

### The Mental Model

```
Question → Clarify → Architecture → Components → Failures → Cost → Ops

Each transition is explicit. Never skip clarify.
```

### Why F500 Asks This

System design is the senior signal. It tests whether you can think at altitude without losing detail.

### Interview Questions

🟢 What's the first thing you do when given a system design prompt?

🟡 Walk me through your structure for a 45-minute ML system design round.

🟡 What's the most common failure mode in system-design interviews?

🔴 Pick one of these and answer for 30 minutes:

- "Design real-time fraud detection for a payments network."
- "Design a recommendation system for a streaming service with 200M users."
- "Design a feature store for an organization with 50 ML teams."
- "Design the serving infrastructure for an internal LLM API supporting 1000 teams."
- "Design a system to retrain when it detects performance drift."
- "Design an embedding pipeline for 500M documents."
- "Design MLOps for a brand-new ML org joining a 50-year-old bank."
- "Migrate from SageMaker to self-hosted K8s ML platform with zero downtime."

For each: clarify scale, propose architecture, drill into 2–3 components, list failure modes, sketch cost, mention org.

---

## Section 20 — Architect-Level Decisions (Build vs Buy, ADRs, Migration)

### The Theory

**Type 1 vs Type 2 decisions (Bezos's frame):**

- Type 1 = irreversible. Heavy process.
- Type 2 = reversible. Fast.

For ML: primary cloud, primary feature store, build-vs-buy for serving, LLM provider, data residency are Type 1. Experiment tracker, HPO library, specific RAG framework are Type 2.

**ADRs** (Architecture Decision Records) — short doc per decision: context, decision, alternatives, consequences. Lives in version control. Numbered.

**Build vs buy frame:** 3-year TCO including hidden costs (engineering hours, operations, switching), time to value, strategic fit, vendor risk, talent availability, switching cost, compliance.

The bias to correct: junior engineers see "self-hosted is free." Architects see "self-hosted is ~$90K/year fully loaded for a small piece of software."

**Migration patterns:** strangler fig (incremental cutover, never big bang), parallel run with validation, explicit decommission criteria. Plan for 1.5–2x your initial estimate.

### The Mental Model

```
                     [Reversibility]
                    │             │
            Reversible           Irreversible
                    │             │
   ┌────────┬──────┴───┐    ┌────┴───┬─────────┐
   │ Low    │ High     │    │ Low    │ High    │
   │ cost   │ cost     │    │ cost   │ cost    │
   ▼        ▼          ▼    ▼        ▼         ▼
  Try    Prototype   Decide  Months, ADR, multiple reviews
```

Calibrate process to the cell.

### Why F500 Asks This

Architect interviews probe judgment under ambiguity. The right answer is rarely a tool; it's a framing.

### Interview Questions

🟢 Type 1 vs Type 2 decisions — give an ML example of each.

🟢 What's an ADR?

🟡 Walk me through your build-vs-buy framework for a feature store.

🟡 A vendor wants 5 years. What do you negotiate for?

🟡 The strangler-fig pattern — walk through it for a migration off SageMaker.

🔴 An F500 has accumulated 15 years of ML platforms: a legacy SAS install, an internal Spark/Airflow stack, a SageMaker tenant, a recent Databricks adoption. The new CIO wants "one platform." Design the migration strategy with explicit ADRs for the Type 1 decisions, sequencing, success criteria per phase, organizational implications.

---

## Closing — How to Use This Chapter the Week Before an Interview

1. **Day 1–2.** Read every theory primer. Skim only.
2. **Day 3–5.** Pick 8 sections that match the JD. For each: 30 minutes practicing the 🟡 questions out loud. Time yourself.
3. **Day 6.** Pick 3 sections most relevant. Spend an hour on the 🔴 question of each. Write out the answer in bullet points, then rehearse the verbal version.
4. **Day 7.** Whiteboard or text-editor practice for the system design round using the Section 19 prompts.

The ROI on talking out loud is 10x reading. Most candidates fail interviews not because they don't know the material but because they've never said it aloud.

Good luck. The work compounds.
