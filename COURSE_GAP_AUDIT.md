# Course Gap Audit — Full Catalog

Audit of all 21 courses in `content/`. Every gap identified, critical and nice-to-have.
Method: heading inventory per course, minimal full-lesson reads. Judged against a strong
mid-2026 ML-engineering curriculum.

---

## Part 1 — Per-course gaps

### Math & ML foundations

#### linear-algebra (7 lessons)
Covers: vectors/geometry → matrices/transpose → matmul → norms/distances → dot product/cosine/projection → eigen+SVD → NumPy capstone.

Critical:
- Stops at 2-D matrices. No tensors, broadcasting, batched matmul, einsum — every downstream DL lesson batches inputs as 3-D+ tensors.
- Matrix-as-linear-map framing arrives only in lesson 06; should land by lesson 03. Matmul currently taught as mechanical rule first.

Nice-to-have:
- Rank / linear independence / span (needed for LoRA, low-rank approximation).
- Orthogonality / orthonormal matrices, identity/inverse.
- Outer product (attention scores, gradient of a linear layer).

#### calculus-gradients (6 lessons)
Covers: functions/limits → derivatives → chain rule → partials/gradients (with numeric gradient checking) → gradient descent → backprop intuition.

Critical:
- No derivatives of exp, log, sigmoid — no bridge to log-likelihood / cross-entropy. Rules lesson is polynomial-only.
- Vector/matrix chain-rule shape intuition missing. Backprop lesson scalar-only; no "gradient w.r.t. a weight matrix" shape-matching.

Nice-to-have:
- Convexity vs non-convexity, local minima / saddle points.
- Sum rule for gradients over a batch (why loss is a mean).

#### probability-stats (7 lessons)
Covers: probability rules/conditional/independence → RVs/distributions → expectation/variance → Bayes → sampling/CLT → MLE + log-likelihood → hypothesis testing/p-values.

Critical:
- **No information theory — entropy, cross-entropy, KL divergence. Single biggest gap in the whole curriculum.** Cross-entropy loss appears from DL lesson 05 onward with no probabilistic grounding.
- Softmax / categorical-as-model-output framing absent (calibration, "model outputs a distribution").

Nice-to-have:
- Covariance / correlation (feature relationships; also missing from ml-foundations).
- Confidence intervals (more useful than p-values for comparing model runs).
- Note: hypothesis-testing lesson is least ML-relevant of the 7; entropy should outrank it if slots are scarce.

#### ml-foundations (8 lessons)
Covers: what ML is + first sklearn model → splits/leakage (thorough) → linear/logistic regression → trees/RF/gradient boosting → metrics → overfitting/bias-variance/regularization → cross-validation → end-to-end project. Leakage and baseline emphasis are above-level.

Critical:
- No feature engineering / preprocessing as a topic — scaling, one-hot encoding, missing values. Lesson 08 uses a preprocessing pipeline but nothing teaches what goes in it.
- No kNN or any unsupervised content. Lesson 01 promises supervised vs unsupervised, then 100% supervised. Add k-means at minimum or drop the promise.

Nice-to-have:
- Regression metrics (MAE/RMSE/R²) — lesson 05 is classification-only yet lesson 03 teaches linear regression.
- Class imbalance handling (class weights, resampling, threshold moving).
- Model calibration / predict_proba trustworthiness.

#### deep-learning-foundations (8 lessons)
Covers: linear→neurons via XOR → activations (sigmoid/tanh/ReLU, dying gradients) → forward pass/batching → backprop → MSE/cross-entropy + SGD/momentum/Adam → training loops/LR decay → regularization → NumPy XOR capstone (He init mentioned).

Critical:
- **No normalization layers.** BatchNorm/LayerNorm absent entirely. Cannot be "DL foundations" in 2026 — LayerNorm is in every transformer.
- Weight initialization as a topic. He init appears only as a code comment in lesson 08; deserves its own section.
- Softmax + multi-class cross-entropy underweight. Lesson 05 headings suggest binary framing; multi-class is the real-world default.

Nice-to-have:
- Gradient clipping, vanishing/exploding gradients in deep stacks (named explicitly).
- Modern activations (GELU/SiLU) — one paragraph in lesson 02.
- Forward-looking signpost lesson toward CNN/RNN/transformer families; course currently ends with no signpost.

#### pytorch-essentials (7 lessons)
Covers: tensors → autograd → nn.Module/Sequential → Dataset/DataLoader → full training loop + validation → state_dict save/load → devices/GPU/.to().

Critical:
- No broadcasting / shape-debugging. Same hole as linear-algebra; reading shape-mismatch errors is the top real-world beginner pain.
- No torch.compile — its absence dates the course.
- No mixed precision (autocast + GradScaler, bf16). Every real GPU training run uses it.

Nice-to-have:
- Distributed basics — "what DDP/device_mesh are, don't use DataParallel" orientation.
- A real dataset (course runs on synthetic tensors); one MNIST-shaped example.
- `weights_only=True` / safetensors note in lesson 06 (torch 2.6 security default change).
- LR schedulers as objects (`torch.optim.lr_scheduler`).
- Verify device lesson names `mps`, not just `cuda`, given the Mac audience.

### Programming & tooling

#### python-foundations (8 lessons)
Covers: setup/REPL → variables/types/control flow → lists/dicts/sets/tuples → functions/modules → file I/O + try/except → OOP basics → comprehensions/iterators/generators → NumPy.

Critical:
- No debugging — no pdb/breakpoint(), no reading tracebacks as a skill. Only error content is try/except (handling, not diagnosing).
- Strings / f-string formatting get no dedicated treatment.

Nice-to-have:
- `*args/**kwargs`, mutable-default-argument trap, `if __name__ == "__main__"`.
- dataclasses (bridges OOP → Pydantic), pathlib over bare `open`, `Counter`.
- Depth note: ~1,000 words/lesson — thinner than the 15-25 min target; beginners need more worked examples.

#### pandas-analysis (7 lessons)
Covers: Series/DataFrame → read_csv/json + inspect → loc/iloc/masks → dtypes/NaN/duplicates/string cleaning → groupby/agg/transform → merge/concat/pivot/melt → plotting + EDA.

Critical:
- **No datetime / time series.** No `to_datetime`, `.dt` accessor, no resample. Biggest hole in the data track.
- No method chaining / `assign` and the copy-vs-view (SettingWithCopy) story.

Nice-to-have:
- `apply` vs vectorized (performance framing), categorical dtype/memory.
- Parquet (CSV-only is dated in 2026), one honest paragraph on Polars/DuckDB, `pd.cut`/binning.
- Depth note: shortest per-lesson course (~870 words); lesson 07 does two lessons' work.

#### sql-databases (7 lessons)
Covers: relational model/keys → SELECT/WHERE/ORDER BY/LIMIT → all four joins → aggregates/GROUP BY/HAVING → subqueries + CTEs → indexing/EXPLAIN → normalization/schema design.

Critical:
- **No window functions** — `ROW_NUMBER`, `RANK`, `LAG`, running aggregates. Loudest gap in the programming track; non-negotiable for point-in-time features and every SQL interview. Course builds the on-ramp (CTEs, GROUP BY) and stops one lesson short.
- No writing data — no INSERT/UPDATE/DELETE, no transactions/ACID. Read-only SQL is not enough for ML engineers who write feature tables.

Nice-to-have:
- CASE expressions, date functions, string functions, dedup patterns (`DISTINCT ON`).
- One paragraph on analytical vs transactional engines (SQLite/Postgres/BigQuery/DuckDB).
- Indexing + EXPLAIN at beginner level is a genuine strength — keep.

#### cli-git (6 lessons)
Covers: shell navigation → cat/less/grep/pipes/redirection/wildcards → bash scripting → git init/add/commit/log/diff/.gitignore → branching/merging/conflicts → remotes/clone/push/pull/PR.

Critical:
- No undo story — no stash, restore, reset, revert, reflog. "How do I undo this?" is the first beginner panic.
- No SSH-key / GitHub auth setup, yet lesson 06 pushes to GitHub. The actual blocker for a first push.

Nice-to-have:
- rebase and merge-vs-rebase (one section), `.gitconfig` aliases.
- `ssh` to a remote box (ML engineers live on GPU servers — arguably critical for this audience), `find`/`xargs`, `curl`+`jq`, tmux.
- env vars / `export` / PATH in the shell lessons (introduced cold later in swe-practices).

#### software-engineering-practices (11 lessons)
Covers: pyproject/src layout → uv/lockfiles → pytest → fixtures/parametrize/coverage → type hints + mypy/pyright + Pydantic → ruff + pre-commit → logging + structlog → exceptions/context managers → 12-factor config/.env/secrets → PR discipline → GitHub Actions CI. Strongest of the six programming courses; modern stack.

Critical:
- No debugging and profiling — no pdb/debugger workflow, no cProfile/py-spy, no "my code is slow, now what." Logging covered; interactive diagnosis not.

Nice-to-have:
- Docstring conventions (Google/NumPy), README hygiene, mkdocs.
- Mocking / `monkeypatch` / `responses` in the testing lessons.
- Semantic versioning / changelogs.

#### apis-web-services (10 lessons)
Covers: HTTP methods/status/headers/TLS → JSON + REST/idempotency → requests/httpx/timeouts/retries/pagination → FastAPI app/params/docs → Pydantic models/validators → HTTPException/DI/lifespan/middleware → serving sklearn (load-once/predict-many, health checks) → API keys/JWT/OAuth2/rate limiting/CORS → async/gather → TestClient/versioning + gRPC/GraphQL note.

Critical:
- **No streaming responses / SSE.** In 2026 "call an LLM API and stream tokens" is the canonical beginner ML-API task. No StreamingResponse, no SSE, no chunked responses. Add a streaming section + a worked LLM-API-consumption example in lesson 03.

Nice-to-have:
- WebSockets (recognize-level), file uploads (`UploadFile` — image-in/prediction-out shape).
- Background tasks / queue mention (inference too slow for request-response).
- Batch-endpoint pattern.
- Verify the "from endpoint to deployed service" section names uvicorn workers + Docker + a platform, else it's hand-waving.

### Cloud & ops

#### aws-for-ml (18 lessons)
Covers: CLI/IAM/EC2/S3/VPC → ECR/ECS/EKS/Fargate → Lambda/API GW → data services → SageMaker training/inference → pipelines (SageMaker Pipelines, Step Functions, MWAA, EventBridge, CI/CD) → end-to-end with CDK/CloudFormation → cost → Bedrock → secrets/KMS → CloudWatch/X-Ray → SQS/SNS/Kinesis/Firehose/MSK → Feature Store/Registry/Monitor. Most complete of the three clouds; GPU depth strong (EFA, placement groups, Capacity Blocks, DLAMIs, managed spot training).

Critical:
- Dedicated vector search reduced to a one-line parenthetical inside Bedrock Knowledge Bases. No lesson on provisioning/querying OpenSearch Serverless / Aurora pgvector / S3 Vectors — the roll-your-own-RAG path is missing.
- Terraform only name-dropped twice; IaC taught as CDK/CloudFormation. Terraform is the industry default; add at least an equivalent `aws` provider section.

Nice-to-have:
- EKS GPU stack depth: Karpenter, Kueue, NVIDIA device plugin, KubeRay.
- Multi-region / DR strategy lesson (endpoint failover, cross-region artifacts, RTO/RPO).
- GPU quota as an explicit workflow (Service Quotas: check → request → wait).
- Currency good: Bedrock covers Converse API, cross-region inference profiles, Guardrails, Knowledge Bases, Agents + AgentCore.

#### azure-for-ml (17 lessons)
Covers: overview/Bicep → Entra ID/RBAC/managed identities → VMs/Spot/scale sets → Blob/ADLS → VNet/private endpoints → ACR/ACI/AKS (incl KAITO) → Functions → data (SQL/Cosmos/Event Hubs/Fabric/ADF) → Azure ML training → deployment (blue-green, canary, shadow/mirror) → Foundry/OpenAI/AI Search → end-to-end Bicep + OIDC CI/CD → cost/governance/Policy/Defender → Key Vault → Monitor/KQL → messaging → feature store/registry. Best of the three; strongest safe-rollout treatment.

Critical: none unique. Vector search covered properly (AI Search vector/hybrid with SDK code, lesson 11) — better than AWS and GCP.

Nice-to-have:
- Runnable Terraform (Bicep-first with honest tradeoff paragraph, but no Terraform code).
- Capacity-reservation story (no ND-series quota-escalation workflow).
- Multi-region / DR lesson.
- AKS ML depth beyond KAITO (no Kueue/Volcano).
- Currency good: Microsoft Foundry rebrand, GPT-5.x, model-retirement caution, Foundry IQ, Agent Service, Flex Consumption Functions.

#### gcp-for-ml (17 lessons)
Covers: gcloud/hierarchy/billing → IAM/WIF → Compute Engine (GPUs, TPUs, MIGs, Spot) → GCS → VPC/PSC/VPC-SC → Artifact Registry/Cloud Run (GPU, scale-to-zero)/GKE → Cloud Run functions → BigQuery (strong: partitioning/clustering, BQML) → Vertex training (Spot + FLEX_START/DWS) → prediction → Gemini/Model Garden → end-to-end with Terraform → cost (CUDs) → Secret Manager/KMS/CMEK → observability → Pub/Sub/Dataflow → Feature Store/Experiments/Metadata. Best IaC positioning (Terraform is the taught default).

Critical:
- **Vertex AI Vector Search entirely absent.** Lesson 11 covers embeddings and mentions "Grounding and RAG" but no Vector Search / Matching Engine / AlloyDB pgvector anywhere. Biggest single-course cloud gap — GCP's headline vector product is unmentioned.

Nice-to-have:
- GKE ML depth: no Kueue, JobSet, TPU-on-GKE, Ray-on-GKE, despite GKE being GCP's flagship ML-infra story.
- Multi-region / DR (incidental dual-region buckets only).
- Currency good: google-genai SDK, gemini-2.5-flash, context caching, Model Garden HF deploys, DWS/FLEX_START.

#### cloud-linux (7 lessons)
Covers: filesystem/shell → users/permissions/processes → VM anatomy/pricing/stop-vs-terminate → block/object/file storage → IP/ports/DNS/firewalls → IAM/secrets → billing/commitment tradeoffs. Provider-agnostic, sound sequencing.

Nice-to-have (judged against beginner remit):
- ssh key mechanics / tmux / scp remote-work module — most-used skill on a cloud GPU box.
- systemd/services, cron, disk mounting (mkfs/mount/fstab) — week-one ops trio on a training VM.
- `nvidia-smi` / GPU health basics.

#### docker-containers (7 lessons)
Covers: why containers → images vs containers → Dockerfile basics → run/ports/volumes/env → Compose → packaging an ML inference service → registries/tag/push/pull. Clean arc landing on a real ML deliverable; `.dockerignore` covered.

Critical:
- **Zero GPU containers.** No `--gpus all`, no nvidia-container-toolkit, no CUDA base images (`nvidia/cuda`, framework images). The #1 reason ML people use Docker.
- No multi-stage builds + image-size for ML. "Use slim" is the only size advice; nothing on build/runtime dep separation or the multi-GB PyTorch image problem.

Nice-to-have:
- Security: image scanning (trivy), non-root `USER`, `HEALTHCHECK`.
- BuildKit cache mounts for pip.

### Advanced Nanodegrees

#### language-modeling (22 files) — LLM Foundation-Model Engineer
Covers: byte-BPE tokenization → resource accounting (FLOPs/2N-6N/memory/precision) → modern transformer block (RMSNorm, SwiGLU, RoPE, GQA/MQA/MLA) → MoE (routing, load balancing, shared experts, sparse upcycling) → GPU execution/roofline → Triton kernels + FlashAttention → full parallelism stack (DDP/ZeRO/FSDP/TP/PP/SP/CP, 3D-4D) → scaling laws (Chinchilla vs Kaplan) → inference (KV cache, speculative decoding, quantization) → evaluation (perplexity, contamination, LLM-judge) → data (dedup/filter/PII/decontam/synthetic) → alignment (SFT→DPO→PPO→expert-iteration→GRPO) → 2026 stack chapter → 5 capstones + 4 interview banks. Very current; MLA, FP8, muP, RoPE scaling, GRPO/RLVR all present and deep.

Critical:
- **Distillation entirely absent** (zero mentions). No logit/sequence-level KD, no on-policy distillation, no distill-from-a-bigger-teacher recipe (now standard: Gemma, Qwen small, DeepSeek-R1 distilled). Sibling vlm-guide has it — port over.
- Linear/hybrid attention & SSMs shunted to the margins. Mamba/SSM/linear-attention appear only in the inference chapter + interview banks, not in architecture ch03 or MoE ch04 where model shape is taught. 2025-26 saw hybrids go mainstream (Jamba, Qwen-hybrid, MiniMax, Mamba-2) — needs first-class treatment.

Nice-to-have:
- Constitutional AI / RLAIF absent (ch12 covers RLHF/DPO/GRPO but not AI-feedback/self-critique).
- Multimodal training not covered as training (defensible — sibling VLM course — but cross-reference so learners don't think text-only is the whole story).
- Verify MXFP4/NVFP4 microscaling 4-bit formats get a mention (FP8 is heavy in ch14).
- Staleness: none material.

#### vlm-guide (24 files incl 7 labs) — GenAI Engineer
Covers: math/tokenization/transformer foundations → LLM architecture (attention variants, MoE, SSMs/hybrids as first-class, MTP) → pretraining/scaling/data → post-training (SFT→preference opt→RLVR/GRPO, process reward models, test-time compute) → inference/efficiency (KV-cache quant, FlashAttention, PagedAttention/vLLM, RadixAttention/prefix caching, disaggregated prefill/decode + Dynamo, quantization, spec decoding, edge) → VLMs (ViT, projectors, fusion taxonomy, LLaVA) → RAG (chunking, hybrid + query-transform, reranking, agentic RAG, eval) → agents (ReAct, tool use, planning, memory, multi-agent, context engineering, agentic RL, MCP) → 2017→2026 timeline → interview banks + 7 labs. Remarkably current.

Critical:
- **Guardrails / safety / prompt-injection dramatically underweight.** Only ch09b (interview) + one passing line in ch06. No dedicated treatment of prompt injection, jailbreaks, output moderation, or tool-use sandboxing — despite the agents chapter teaching browser/DB/API/email tool calls. Single biggest hole; injection defense + irreversible-action confirmation deserve their own section.
- Computer-use / GUI agents only historical (timeline files), never taught as a pattern in ch06 or the agents lab. A defining 2025-26 capability — teach screenshot→action, accessibility-tree grounding.

Nice-to-have:
- Structured outputs coverage diffuse across 9 files; no consolidated "structured outputs / tool-schema / constrained-decoding" treatment.
- GraphRAG listed under advanced RAG but likely shallow — verify it's more than a name-drop (entity-graph construction is where learners fail).
- **Maintenance flag:** course states speculative future models as fact ("DeepSeek V4-Pro Apr 2026", "Qwen3.5", "Gemini 3.0"). Forward-dated named models are a liability — reads as fabrication to an advanced paying audience if they didn't ship as described. Fact-check pass needed.

#### mlops (33 files, ~107k words) — practitioner→architect
Covers: repro/MLflow/DVC/Docker/FastAPI → Feast/orchestration/registry/CI-CD-CT/Evidently drift → GPU ops/DDP/K8s/canary serving/Kafka-Redis streaming features → cloud ML platforms/LLMOps/LiteLLM/vector DBs/MLflow-3/OTel gen_ai observability/AI Act → 7 F50 portfolio projects → 5-part advanced topics (distributed systems, inference optimization, RLHF/DPO, agentic, Flink/Kafka, K8s depth, security/governance/FinOps/DR) → 2 architect career tracks + 8 case scenarios + DL/CV/NLP track + 20-section interview bank + 8-part cloud chapter. In good 2026 shape.

Critical:
- Prompt management name-checked twice, never taught. No prompt versioning/registries, prompt CI, or regression-testing prompt changes — table-stakes 2026 LLMOps.

Nice-to-have:
- Shadow deploys thinner than canary (canary implemented in code; shadow mostly descriptive). Add a worked shadow/interleaving comparison.
- Rest of checklist genuinely present and current (Feast, registries, Evidently, Kubeflow, Ray, vLLM+SGLang, wandb/MLflow-3, GPU ops, FinOps).

#### data-engineering (21 files, ~58k words) — beginner→Fortune-100
Covers: Docker/Postgres, DuckDB+Polars, Terraform/GCP, Kestra, dlt → BigQuery, dbt (deep), dimensional modeling, ODCS v3.1 data contracts, Bruin/SQLMesh → Spark, Kafka, Debezium CDC + outbox → lakehouse capstone → Airflow (asset-aware), Snowflake/Databricks, Iceberg, data quality/observability, CI/CD → advanced topics (Flink, ClickHouse/Druid/Pinot, Arrow, Trino, semantic layer, vector DBs, K8s, agentic DE) → 4-part Data Architect track + 8 cases.

Critical:
- Real-time ML feature pipelines are the weakest item. Feature store appears only as Project 5 + mentions — no taught lesson on streaming feature freshness, online/offline sync, or serving joins. Missing bridge to the mlops course.

Nice-to-have:
- Great Expectations only in 2 files (quality at survey level). Add a hands-on GX/Soda/dbt-tests contract-enforcement lab.
- Dagster present but Kestra→Airflow is the taught path; its asset model deserves more given the course teaches asset-aware Airflow.
- Streaming (Kafka/Flink/CDC/Debezium), lakehouse (Iceberg/Delta), dbt, contracts all strong.

#### ml-system-design (20 files, ~80k words) — staff-level design + interview
Covers: interview framework + cost math → feature platforms/synthetic data → 4D parallelism + post-training + MFU → quantization/spec decoding/on-device → two-part LLM serving (continuous batching, PagedAttention, prefill/decode disaggregation, GenAI gateway, cascades/routing, capacity planning, vLLM benchmarking lab) → RAG + eval harness → agentic systems w/ MCP, context engineering, HITL, agent attacks → classic recsys/search/fraud → eval/observability/MLOps → economics/TCO module → DevOps/K8s/KEDA → 5 end-to-end case studies → domain variations → cumulative capstone → career. Most current course in the catalog.

Critical: none.

Nice-to-have:
- MCP in only 3 files — could carry more weight as agent interop hardens.
- Multi-agent orchestration patterns thin relative to single-agent loops.

#### principal-ml-engineer (20 files, ~79k words) — org-scale
Covers: level ladder/operating model → Rumelt-style technical strategy → platform-as-product, five planes, multi-tenancy/chargeback → GPU fleet economics + capacity planning → serving portfolio/consolidation/cascade routing → data flywheels/LLM-as-labeler → eval as org discipline (statistics, launch gates, eval-set governance) → migration playbook + parity metrics → build-vs-buy TCO + CFO memo → ML reliability/SLOs/degradation ladders → 11-pattern failure library → governance/MRM/EU AI Act/fairness → influence/org design → technology bets 70/20/10 → interview bank + capstone. Real 2026 content.

Critical: none.

Nice-to-have:
- No dedicated "operating agent fleets" section (agent economics appears in exercises/scenarios, not as a named topic). Add: per-task cost budgets, autonomy tiers, agent-specific incident patterns.

---

## Part 2 — Cross-cutting structural findings

1. **Cross-entropy chasm.** No entropy/KL in prob-stats, no log/exp derivatives in calculus, softmax underweight in DL — yet cross-entropy is the loss used from DL lesson 05 onward. Fix in prob-stats (new lesson) + a section each in calculus-02 and DL-05.
2. **Broadcasting/tensor shapes ownerless.** linear-algebra ends at 2-D, pytorch-01 skips broadcasting. Assign it explicitly to one course.
3. **Beginner courses are the thin ones.** python-foundations and pandas-analysis average ~900-1,000 words/lesson vs ~1,200 for advanced courses. Backwards — beginners need more worked examples.
4. **Architect tracks are structural clones.** `mlops/07*-ml-architect-track` and `data-engineering/07*-data-architect-track` share an identical 17-phase skeleton — same phase names, even the same ADR-0042 example number. ~10 of 17 phases are domain-agnostic and near-certainly duplicated prose. Extract a shared "Architect Core"; keep only domain-specific reference architectures per course.
5. **principal-ml-engineer re-covers architect material a third time** at higher altitude (build-vs-buy, migrations, capacity/cost, org design). Defensible as a level-up, but the three should cross-reference instead of restating.
6. **Cost modeling appears 4×** (mlops FinOps + architect Phase 8; ml-system-design module 11; principal module 09). ml-system-design 11 and principal 09 are the strong versions; trim mlops's to a pointer.
7. **Interview banks in all four advanced courses.** Expected for positioning, but mlops's ML-system-design questions (Section 19) should defer to the ml-system-design course rather than duplicate.
8. **Cloud parity.** Vector search: Azure ≫ AWS > GCP (fix GCP first). GPU capacity: each cloud covers only its own half (AWS Capacity Blocks, GCP DWS, Azure quota) — add a symmetric "getting GPUs when there are none" section to all three. Safe rollout: shadow/mirror only in Azure. K8s-for-ML uniformly shallow. Multi-region/DR a shared gap across all three.
9. **Consistent lesson template** (Why-this-matters / Key takeaways / Try it) across all 42 foundation lessons — genuinely good scaffolding, keep it.

---

## Part 3 — Missing courses (ranked)

1. **Computer Vision / CNNs** — biggest catalog hole. deep-learning-foundations ends at MLPs; vlm-guide assumes ViT knowledge nobody taught. No convolutions, no detection/segmentation anywhere. Breaks the catalog's own prereq chain (vlm-guide lists deep-learning-foundations, but the MLP→ViT jump is unsupported).
2. **AI Security & Guardrails** — prompt injection, jailbreaks, output moderation, tool sandboxing, agent permissions. Gap flagged in both flagship GenAI courses. Standalone sells better in 2026 than a buried module.
3. **Kubernetes for ML** — shared shallow spot across all 3 cloud courses + mlops. Kueue, KubeRay, Kubeflow, GPU scheduling, autoscaling inference. Natural specialization next to the cloud trio.
4. **Experimentation & Causal Inference** — A/B testing, power analysis, uplift, causal basics. prob-stats stops at p-values; ml-system-design recsys chapter assumes it. Product-ML staple.
5. **Reinforcement Learning Foundations** — RLHF/GRPO taught in two courses with no RL grounding (no MDPs, policy gradients, value functions). Short bridge course serving the language-modeling track.
6. **Time Series & Forecasting** — zero coverage anywhere; common industry workload.
7. **Fine-tuning in Practice** — applied LoRA/QLoRA/dataset prep/eval loop as its own short course; currently one lab in vlm-guide. High commercial demand.
8. **DSA / Coding Interview Prep** — catalog sells interview readiness (4 interview banks) but has no coding-round prep. Optional, possibly off-mission.

---

## Part 4 — Highest-impact fixes (ordered)

1. Entropy/KL/cross-entropy lesson in probability-stats (unblocks everything downstream).
2. SQL window functions lesson.
3. GPU containers + multi-stage builds in docker-containers.
4. Vertex Vector Search in gcp-for-ml (+ a vector-store section in aws-for-ml).
5. Guardrails/injection module in vlm-guide.
6. Distillation chapter in language-modeling (port from vlm-guide).
7. Normalization layers + weight init in deep-learning-foundations.
8. pandas datetime/time-series lesson.
9. Streaming/SSE in apis-web-services.
10. De-duplicate the two architect tracks (mlops + data-engineering).
11. Prompt-management lesson in mlops.
12. Real-time feature-pipeline lesson in data-engineering (bridge to mlops Feast content).
13. Fact-check vlm-guide's speculative model names against what actually shipped.
