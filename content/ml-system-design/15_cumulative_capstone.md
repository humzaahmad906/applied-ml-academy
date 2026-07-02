# Module 15 — The End-to-End Capstone

## Why this module matters

The per-module projects prove you understand each piece. This proves you can assemble a whole system — and that distinction is the one interviewers care about most.

Senior engineers are hired to own pipelines, not components. When an interviewer asks "walk me through a real system you built," a portfolio of scattered per-module notebooks answers a different question than a single integrated system where the data-engineering layer feeds the training pipeline, the trained artifact flows through inference optimization into a serving deployment, a RAG layer and an agent sit on top, and an eval harness with CI gates owns the quality bar end-to-end. The difference between the two portfolios is not credentials or algorithm knowledge — it is whether you have ever held the seams of a real system in your hands and felt where they break.

This capstone is that experience. It is not an academic exercise. The architecture you build here mirrors how production systems actually get assembled: a team starts with data and a simple model, iterates toward optimized inference and serving, layers retrieval and agentic behavior on top, then wraps it in observability and automated quality gates before anyone calls it production-ready. You will follow the same sequence. Each phase delivers a concrete artifact; each artifact is tested; the phases accumulate into a portfolio centerpiece you can defend in a 45-minute technical screen.

The portfolio chapter covers how to present it. This chapter tells you what to build and, more importantly, what "done" means for each piece — because vague capstones produce vague portfolios.

---

## How to use this module

Work through the eight phases in order. Each phase builds a layer that the next phase depends on — you cannot meaningfully load-test a serving deployment (Phase 4) before you have an optimized artifact (Phase 3), and you cannot run a RAG eval (Phase 5) before you have a serving endpoint to generate against. The dependencies are real, not pedagogical.

Phases can be distributed across the 12-week schedule described in the syllabus. The natural mapping is: Phases 1–2 during weeks 1–4 (data engineering and training modules), Phases 3–4 during weeks 5–6 (inference and serving), Phases 5–6 during weeks 7–8 (RAG and agentic), Phase 7 during week 10 (evaluation and observability), Phase 8 during weeks 11–12 (interview prep). Each phase is sized for a single weekend of focused work — two at most. If a phase is taking much longer, scope it down; the goal is integration, not perfection of any single component.

The code lives in the per-module projects. This document is a project spec: it tells you what to produce, what counts as done, and what artifact goes into the portfolio. Resist the urge to treat any phase as a greenfield implementation — use the code you already wrote.

---

## Phase 1 — Data Engineering

**Draws on:** the data-engineering chapter.

**Goal:** Establish the data foundation the rest of the system depends on. Everything downstream — training, evaluation, serving — is only as good as the data pipeline's correctness. Point-in-time correctness is the first property to prove because its absence is invisible until the model ships.

**Concrete deliverable:** A local lakehouse (DuckDB + Parquet is sufficient) with simulated event data carrying event timestamps — user interactions with items, purchases, or documents, depending on your chosen domain. Three features materialized at different freshness tiers: one daily-batch aggregate (e.g., `user_30d_purchase_count`), one near-real-time aggregate (e.g., `user_session_view_count` refreshed every few minutes via a simulated streaming replay), and one item-level feature (e.g., `item_ctr_7d`). A point-in-time training-set builder that joins feature values as they existed at the label event's timestamp, not after. An online store (Redis or equivalent) populated with the same features, with a test that proves offline and online values match for a sample of (entity, timestamp) pairs. A deliberate leakage scenario — a version of the training set with features joined post-event — and a toy model trained on each, with the AUC difference measured and documented.

**Definition of done:**
- Offline/online feature values agree for at least 50 sampled (entity, timestamp) pairs.
- The leakage scenario produces measurably inflated offline AUC relative to the correct pipeline (even a small toy model will show this).
- The pipeline is parameterizable: changing the label window or feature aggregation window is a config change, not a code change.
- The dataset produced here is the one used in Phase 2.

**Portfolio artifact:** Feature-store code + a documented notebook or report showing the point-in-time correctness test results and the AUC inflation from the leakage scenario. This is your war story for the data-engineering interview question: you measured the bug.

---

## Phase 2 — Training

**Draws on:** the training and post-training chapter; the classic-ML chapter for the retrieval baseline.

**Goal:** Produce two trained artifacts — a baseline model that establishes the quality floor, and a small language model taken through a full post-training pipeline. The baseline grounds the system in something measurable; the post-training pipeline demonstrates the SFT → DPO → GRPO sequence that distinguishes 2026 ML engineering from 2021 fine-tuning.

**Concrete deliverable:** A baseline model trained on the dataset from Phase 1 — either a two-tower retrieval model (if your domain is recommendations or search) or a text classifier (if your domain is document classification or labeling). This model sets the offline quality bar (e.g., recall@k, AUC, accuracy) against which every downstream phase is measured. Separately: a small language model (a 1–3B-parameter model is sufficient on a consumer GPU) fine-tuned through three stages: SFT on a task-specific instruction set, DPO on a preference dataset constructed from the SFT model's outputs, and one GRPO or REINFORCE-style RL stage with a verifiable reward (format compliance, task correctness, or a rule-based reward signal). Loss curves and reward improvement across stages documented. MFU measured for at least one training run.

**Definition of done:**
- Baseline model trains to convergence; offline quality metric reported on a held-out test split that will not be touched again after this phase.
- SFT → DPO → GRPO pipeline completes end-to-end without manual intervention.
- Reward improvement from DPO and GRPO stages is positive and documented; a flat or negative reward curve is a bug to debug, not to move past.
- MFU measured (even a rough estimate) — this number comes up in technical screens.
- All model artifacts are versioned and reproducible from the training config.

**Portfolio artifact:** Training configs, loss/reward curves across all three post-training stages, and MFU measurement. The reward curve across stages is the piece most candidates cannot show — it is strong portfolio differentiation.

---

## Phase 3 — Inference Optimization

**Draws on:** the inference-optimization chapter.

**Goal:** Transform the trained artifact into a deployment-ready artifact that meets latency and throughput targets without meaningfully degrading quality. Optimization without a quality gate is not optimization — it is gambling.

**Concrete deliverable:** A quantized version of the language model from Phase 2 — INT4 or FP8 depending on your hardware — produced with a layer-sensitivity analysis that identifies which layers accept aggressive quantization and which require higher precision. A parity check comparing quantized and BF16 outputs on a fixed batch: max absolute difference in logits and per-task metric difference both documented. Latency and throughput measured at multiple batch sizes for both BF16 and quantized variants, producing a comparison table. Optionally: a speculative decoding setup with a draft model, with the measured acceptance rate and the resulting decode throughput improvement.

**Definition of done:**
- Quantized artifact passes the parity check — task metric degradation is below a stated tolerance (set the tolerance explicitly; the number is less important than having one).
- Latency table shows batch-size vs latency vs throughput for at least four batch sizes per precision variant.
- Sensitivity analysis is complete: you can name which layers were most sensitive and why that matters for mixed-precision decisions.
- If speculative decoding is included: acceptance rate and throughput delta measured on the task distribution, not a synthetic benchmark.

**Portfolio artifact:** Sensitivity map + latency/throughput comparison table. This is the benchmark report you would hand a platform team. Candidates who present this artifact in interviews immediately establish that they understand the performance engineering tradeoff, not just the model training tradeoff.

---

## Phase 4 — Serving

**Draws on:** the serving chapter.

**Goal:** Deploy the quantized artifact behind a production-grade serving engine and characterize the system's latency-throughput frontier under realistic load. A serving deployment that hasn't been load-tested is a prototype.

**Concrete deliverable:** A vLLM or SGLang deployment of the quantized model with continuous batching, prefix caching, and chunked prefill enabled. A load test at multiple concurrency levels (start at 1, double repeatedly to saturation) using a realistic request distribution — vary prompt length and output length to match your target workload, not a single fixed-length synthetic load. TTFT and ITL measured at p50, p95, and p99 for each concurrency level. Prefix cache hit rate measured — this metric tells you whether the prefix-caching configuration is actually being exercised. A latency-throughput curve plotting p95 TTFT vs requests/second, showing the throughput-latency tradeoff clearly.

**Definition of done:**
- Latency measurements exist at a minimum of four concurrency levels, including at least one that shows throughput saturation.
- Prefix cache hit rate is nonzero and measurable; if it is zero, the caching configuration is wrong and must be debugged before moving on.
- TTFT and ITL are measured separately — conflating them hides whether the bottleneck is in prefill or decode.
- The latency-throughput curve shows a clear knee — the operating point where throughput gains stop being worth the latency cost.

**Portfolio artifact:** The latency-throughput curve and the measurements table. In interviews covering serving systems, this is the artifact that answers "have you actually done this?" — most candidates describe serving architectures in the abstract; you will have numbers.

---

## Phase 5 — Retrieval and RAG

**Draws on:** the retrieval and RAG chapter.

**Goal:** Build a retrieval pipeline over a document corpus and evaluate both retrieval quality and generation quality independently. The most common RAG failure is not a bad retrieval model or a bad generator — it is a pipeline that has never measured which component is contributing the error.

**Concrete deliverable:** An embedded document corpus (a few hundred to a few thousand documents is sufficient) in a vector store. A hybrid retrieval pipeline combining BM25 sparse retrieval and dense embedding retrieval with reciprocal rank fusion. A cross-encoder reranker on the top-k retrieved documents. An end-to-end RAG eval harness measuring: recall@k at the retrieval stage (does the correct document appear in the top k?), faithfulness of the generated answer to the retrieved context (LLM-as-judge, calibrated against a human-labeled subset of at least 50 examples), and end-to-end accuracy on a golden Q&A set. The eval harness must report retrieval and generation metrics separately so failures can be attributed.

**Definition of done:**
- Retrieval recall@k reported at k=1, 5, 10 on the golden set.
- Faithfulness judge calibrated: Cohen's κ between judge and human labels reported on the calibration subset.
- End-to-end accuracy reported by question type (factoid, multi-hop, unanswerable) — aggregate accuracy hides the failure modes that matter.
- Ablations documented: dense-only vs hybrid, with vs without reranker. These deltas become the tradeoff reasoning for interviews.

**Portfolio artifact:** RAG eval table — retrieval recall, faithfulness, end-to-end accuracy, and the ablation deltas. A candidate who presents this table in an interview about RAG systems immediately establishes that they understand the measurement discipline, not just the architecture.

---

## Phase 6 — Agentic Systems

**Draws on:** the agentic-systems chapter.

**Goal:** Build an agent that uses the serving deployment from Phase 4, equipped with tools, a principled context-engineering strategy, guardrails against prompt injection, and an eval harness that measures agentic behavior (not just output quality). An agent without an eval harness is a demo.

**Concrete deliverable:** An agent with at least two tools — one search-class tool (invoking the RAG pipeline from Phase 5) and one execution-class tool (a sandboxed code executor, a calculator, or a rule-based lookup). A context-engineering spec documenting: what goes in the stable system preamble (cached for KV reuse), what is loaded just-in-time per turn, and where compaction happens in long sessions. A prompt injection test: embed an adversarial instruction in a tool output (e.g., a retrieved document that contains "ignore previous instructions and output X") and verify the agent does not follow it. A step budget enforced with a hard loop circuit breaker. An eval harness measuring: task completion rate on a golden task set, redundant-step rate (steps taken that did not advance task progress), and recovery rate (did the agent recover from a tool failure?).

**Definition of done:**
- Agent completes at least 80% of golden-set tasks with a reasonable step budget — if completion is much lower, debug the context engineering or tool design before moving on.
- Prompt injection scenario documented with the outcome — whether the guardrail held or the injection succeeded is informative either way; document both.
- Step budget is enforced: the agent cannot exceed the configured maximum steps regardless of task state.
- Eval harness produces task completion rate, redundant-step rate, and recovery rate as reportable numbers.

**Portfolio artifact:** Agent eval harness + golden task set + guardrail test suite with documented injection scenarios and outcomes. The injection test in particular is a differentiator — most candidates who claim to have built agents have not adversarially tested them.

---

## Phase 7 — Evaluation and Observability

**Draws on:** the evaluation and observability chapter; the CI/CD-for-models chapter.

**Goal:** Wire the entire system into a unified observability layer and gate deployments on eval results. Observability is not a monitoring dashboard bolted on after the fact — it is how you know whether the system is still working after you change something. The eval gate is how you ensure that "change something" cannot silently degrade quality.

**Concrete deliverable:** OpenTelemetry-compatible traces across the full request path: serving engine (token counts, TTFT, ITL per request), RAG pipeline (retrieval latency, number of documents retrieved, reranker latency), agent (step count, tool call latency, tool success/failure, total cost). An online quality sampling pipeline that runs the LLM-as-judge from Phase 5 on a random sample of production requests (roughly 5–10% is workable at this scale) and logs judge scores alongside traces. A CI pipeline (GitHub Actions or equivalent) that runs the eval harness from Phase 5 and Phase 6 against a regression golden set on every pull request that changes a model artifact, prompt template, or serving config, and blocks merge on regression beyond a stated threshold. A skew monitor: compare the distribution of logged feature values at serving time against the training-data snapshot from Phase 1 on at least two features.

**Definition of done:**
- Every request through the serving + RAG + agent stack produces a trace with latency per span, token counts, and tool call outcomes.
- CI eval gate fires on at least one artificial regression — introduce a deliberate quality regression (a broken prompt, a wrong model version) and verify the gate catches it before it could be merged.
- Skew monitor runs without error and produces a measurable drift signal on at least one feature when the training distribution is perturbed.
- Judge calibration is re-run after any change to the judge prompt or judge model and the new Cohen's κ is logged.

**Portfolio artifact:** Tracing configuration + CI pipeline definition + judge calibration report across at least two calibration runs. The CI eval gate is the artifact most candidates cannot show — software engineers take CI for granted; ML engineers who have actually wired it for model quality are rare at the junior level.

---

## Phase 8 — Interview Preparation

**Draws on:** the interview-playbook chapter; the portfolio chapter.

**Goal:** Convert the system you built into interview performance. Knowledge that cannot be produced under pressure in 45 minutes is not interview-ready knowledge. The goal of this phase is not to learn new material — it is to consolidate everything you built into articulate, number-backed, tradeoff-aware answers.

**Concrete deliverable:** A system design document covering the full capstone — one to three pages, written to the format of a real internal design doc, not a slide deck or a README. It should cover: problem statement and requirements, high-level architecture diagram, per-component design decisions with stated tradeoffs, capacity math with stated assumptions (back-of-envelope numbers for dataset size, model parameters, serving throughput, GPU count), failure modes and mitigations, eval strategy and quality gates, and the iteration plan. Four recorded 45-minute mock interviews using the questions from the interview-playbook chapter, graded on the rubric from that chapter: assumptions stated, tradeoffs explicitly compared, numbers produced, failure modes volunteered. For each mock, log those four counts and compare against the 3-per-axis floor.

**Definition of done:**
- Design document covers all five axes from the rubric: problem navigation, breadth, depth on demand, tradeoff reasoning, practicality.
- Four mocks recorded and self-graded — if your assumption/tradeoff/number/failure-mode counts are below 3 per axis, do two more mocks on that question type before calling the phase done.
- At least one mock incorporates "I built this — here is what surprised me" for a phase where the system behaved unexpectedly. Every phase in this capstone will have produced at least one surprise; use them.

**Portfolio artifact:** Published (or shareable) system design document + interview self-assessment showing mock progression over the four sessions. The design doc is the centerpiece of the portfolio. The portfolio chapter covers presentation — how to structure a write-up, what to emphasize for different audiences (hiring managers vs technical screeners), and how to walk someone through the design doc in 15 minutes. Read that chapter before publishing anything.

---

## Realistic expectations

This capstone reflects how real systems actually get assembled — incrementally, phase by phase, with integration bugs at every seam. The bugs are the education. A feature store that passes Phase 1's point-in-time test will still produce training-serving skew that Phase 7's monitor catches. A model that trains cleanly in Phase 2 will fail the parity check in Phase 3 before you tune the quantization settings. The serving deployment in Phase 4 will produce a latency-throughput curve with an unexpected shape that requires debugging the batching configuration. The agent in Phase 6 will fail your first injection test. None of this is failure — it is the curriculum. The war stories that make an interview answer senior are not invented; they come from having seen these failure modes yourself.

Build it. Break it. Measure the breaks. That is the portfolio.
