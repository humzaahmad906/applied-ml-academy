# Module 17 — Course Delivery & Roadmap

## Why this module matters

This is the meta-chapter. It does not teach a new system design pattern; it teaches you how to learn the ones already covered and how to stay current as the field moves under you. It also documents the course-design decisions so that self-directed learners know which modules to prioritize, and so that instructors running cohorts have an honest account of what the course is — and what it is not yet.

One clarification up front: several recommendations from the course review — a Discord community, scheduled office hours, a gamified progress tracker, video lectures — are **product and operational decisions**, not content decisions. They require infrastructure, maintainer time, and sustained community management. This chapter does not fabricate those as existing features. Where they appear, they appear as a **roadmap**: what a well-run cohort program would look like, framed as guidance for anyone who chooses to build it.

The genuine content in this chapter is: a key-formulas cheat sheet (the most time-efficient thing in the entire course to memorize), spaced-repetition guidance, a forward-looking section on what is likely to shift by 2027, and an honest account of how the course is sequenced and which parts are elective.

---

## 1. Learning design

### 1.1 Prerequisite self-check

Before starting module 01, verify these three things. If any of them are missing, fix the gap first — the course assumes them and will not explain them:

1. **Python fluency.** You can write a dataloader, a training loop with gradient accumulation, and a simple HTTP endpoint without looking things up. A rough bar: you can explain why `optimizer.zero_grad()` comes before `loss.backward()`.
2. **PyTorch at a sketch level.** You know what a `nn.Module` is, what `forward()` returns, what `.detach()` does and why it matters. You do not need to know `torch.compile` internals.
3. **Core ML vocabulary.** Loss functions, bias-variance tradeoff, precision vs recall (and the confusion matrix behind them), what a transformer block does at the residual-stream level. You do not need to have implemented multi-head attention from scratch — but you need to be able to sketch the QKV projection and explain why attention scales as O(n²) in sequence length.

If you are missing item 1 or 2: Andrej Karpathy's *Neural Networks: Zero to Hero* video series covers both. If item 3 is the gap: Coursera's *Machine Learning Specialization* covers the vocabulary fast.

### 1.2 Three learning paths

Different learners enter this course with different gaps. The README defines three entry paths; this section restates them with concrete advice.

**Path A — SWE→ML (software engineer transitioning into ML roles).**
You have strong systems intuitions — distributed systems, service reliability, cost analysis — but weak model-side vocabulary. The training and post-training chapter will feel slow; the serving and observability chapters will feel fast. Suggestion: move quickly through modules 03 and 04, pausing only on the FLOPs arithmetic and quantization correctness (the ML-specific traps are there); go deep on modules 05, 07, 08, and 09 where your systems background translates directly into interviewer credibility. The single biggest risk for SWE→ML candidates in interviews: answering serving questions well but hand-waving the metric and evaluation sections. Read module 09 twice.

**Path B — DS→MLE (data scientist moving toward ML engineering).**
You know models and metrics but underestimate infrastructure. The foundation and inference chapters may feel like new vocabulary; the evaluation chapter will feel familiar. Go deep on modules 03, 04, 05, and 08 — these are where DS-background candidates most often have gaps that get probed. Pay specific attention to the capacity-math sections; DS-background candidates frequently skip them and lose points on the practicality axis of the interview rubric. The cheat sheet in section 1.4 of this chapter is your highest-leverage study artifact.

**Path C — MLE interview prep (already an ML engineer, studying for FAANG loops).**
You probably know the content of most chapters. Use the course as a structured vocabulary audit, not a learning sequence. Work through the interview Q&A at the end of each chapter first; any question you answer weakly identifies a gap. Spend your time on the mock questions in the capstones chapter and on measuring your actual answers against the rubric there. The highest-value chapters for interview prep specifically are the foundations chapter (the framework), the LLM serving chapter (disaggregation and KV math are heavily probed), the agentic systems chapter (security + eval framing separates senior from junior answers), and the evaluation chapter (LLM-as-judge calibration and the CI gate pattern).

### 1.3 Spaced-repetition practice

The research on learning retention is unambiguous: distributed practice beats massed practice. If you read one module per sitting and never return, you will retain roughly 20% of the technical vocabulary 30 days later. Three low-friction practices close this gap:

**After each module:** write a one-page design doc that applies the module to a system you invent. Not a summary of the chapter — an actual design. This takes 20–30 minutes and is the highest-leverage single practice per chapter.

**Weekly review:** pick one module from any earlier week and answer its interview Q&A out loud, without notes. Grade yourself on the rubric in the capstones chapter (assumptions stated, tradeoffs compared, numbers produced, failure modes volunteered). The goal is not to recite — it is to reconstruct the reasoning from first principles under mild pressure.

**Cumulative exercises:** at the end of weeks 4, 8, and 12, design a system that requires three or more modules together. Good prompts: "Design a fine-tuning pipeline for a domain-specific coding assistant and the serving infrastructure to deploy it" (modules 03 + 04 + 05); "Design a document-processing system with a retrieval layer and production monitoring" (modules 02 + 06 + 09). The integration is where real interview depth lives.

**Review questions by module group (use as flashcard seeds):**

- *Foundations:* What are the five rubric axes interviewers grade? Walk the offline/online/feedback-loop anatomy for a fraud system.
- *Data engineering:* What is point-in-time correctness and what breaks if you violate it? Name three feature-freshness tiers and give a use case for each.
- *Training:* What is tensor parallelism and when is it preferable to pipeline parallelism? What does 6ND represent and where does the 6 come from?
- *Inference:* What determines whether a decode step is compute-bound or memory-bound? Derive the decode throughput limit from HBM bandwidth.
- *LLM serving:* What does PagedAttention solve? What is the KV-hit-rate metric and what system behavior does it measure? When do you disaggregate prefill and decode?
- *RAG:* What is the difference between a bi-encoder and a cross-encoder and why does it matter for latency? What is calibration and why does it matter for cascade routing?
- *Agentic:* What is the prompt-injection attack surface in a code agent? Name three bounded-agent failure modes.
- *Classic ML:* What is logQ correction and why does in-batch negatives training need it? What is position debiasing and what experiment reveals that it is necessary?
- *Evaluation:* What does LLM-as-judge calibration mean and how is it measured? What is the CI eval gate pattern?

### 1.4 Key-formulas cheat sheet

These constants recur across the training, inference, and serving chapters. Interviewers test them. Memorize them — not as trivia, but as the building blocks of live capacity math.

---

**FLOPs per token — dense transformer**

```
Forward pass:   FLOPs ≈ 2N
Training step:  FLOPs ≈ 6N
```

*N* = number of non-embedding parameters. The forward-pass formula (2N) follows from the dominant term being matrix multiplications: for each token, every parameter is multiplied once (weight × activation → ~2 ops per multiply-add). The training-step formula (6N) adds the backward pass (~4N: 2N for the gradient with respect to weights + 2N for the gradient with respect to activations). This is the formula underlying the Chinchilla compute budget: *C ≈ 6ND* where *D* is the number of training tokens. In an interview, volunteering "the forward pass costs 2N FLOPs per token and a full training step costs 6N" is a senior signal — most candidates treat FLOPs as a black box.

**Derivation sanity check:** a 7B-parameter model, 1 T training tokens, bf16 training on H100s at 40% MFU. *C = 6 × 7×10⁹ × 10¹² = 4.2×10²² FLOPs.* H100 peak = 1 PFLOP/s bf16 = 10¹⁵ FLOP/s; at 40% MFU = 4×10¹⁴ FLOP/s. Time = 4.2×10²² / 4×10¹⁴ ≈ 1.05×10⁸ s. Divide by seconds/day (86 400) → ~1 200 GPU-days → ~5 days on 256 H100s. Order-of-magnitude consistent with published runs.

---

**KV-cache size**

```
KV-cache bytes = 2 × L × n_kv × d_head × S × B_elem
```

- *L* = number of layers
- *n_kv* = number of KV heads (equals n_heads for MHA; reduced for GQA/MQA)
- *d_head* = head dimension = *d_model / n_heads*
- *S* = sequence length (tokens currently in context)
- *B_elem* = bytes per element (2 for fp16/bf16, 1 for fp8)
- The leading *2* accounts for K and V (each stored separately)

Example: LLaMA-3.1-8B: 32 layers, GQA with 8 KV heads, d_head = 128, fp16.

```
Per-token KV = 2 × 32 × 8 × 128 × 2 = 131 072 bytes ≈ 128 KB per token
128K context window → 128 KB × 128 000 ≈ 16 GB  — close to the full model weight size
```

This explains why long-context serving is memory-constrained and why fp8 KV caching halves this number. In an interview, being able to derive that a 128K-context request consumes ~16 GB of KV cache for an 8B model — and then naming fp8 KV as the mitigation — is the exact depth-on-demand that separates mid-level from senior-level answers.

---

**HBM bandwidth-bound decode rule**

At batch size 1, a single decode step (one token generated) must load every model weight from HBM at least once. This is the memory-bandwidth bottleneck:

```
time_per_token ≈ model_bytes / HBM_bandwidth
```

Example: a 7B fp16 model = 14 GB; H100 SXM HBM3 bandwidth = ~3.35 TB/s.

```
14 × 10⁹ bytes / 3.35 × 10¹² B/s ≈ 4.2 ms per token
```

Upper-bound throughput at batch=1: ~240 tok/s. Real systems achieve 150–200 tok/s at small batch due to attention and non-matmul overhead. Increasing batch size amortizes the weight reads across multiple concurrent tokens — the system transitions from memory-bound to compute-bound somewhere around batch 32–128 depending on model size. Batch size at the memory/compute crossover = *2 × N / (d_model)* in simplified analysis — the practical takeaway is that continuous batching exists precisely to keep the GPU on the right side of this transition.

The corollary: on a smaller HBM-bandwidth chip, latency degrades proportionally. This is the central argument for disaggregating prefill and decode: prefill is compute-bound (many tokens processed in parallel); decode is memory-bound (one token at a time). Mixing them on the same hardware forces a compromise. The prefill/decode disaggregation chapter covers this in detail.

---

**MFU ~40% planning budget**

Model FLOP Utilization (MFU) measures the fraction of the hardware's theoretical peak FLOP/s that the training job actually achieves:

```
MFU = (measured throughput in tok/s × FLOPs/tok) / (n_GPUs × peak_FLOP/s)
```

Well-optimized runs on modern hardware (H100, well-tuned FSDP or Megatron, high-BF16 utilization, minimal communication overhead) land in the **35–50% MFU** range. Budget **40%** as a planning number; use it to estimate training time from compute budget. Anything below 30% warrants investigation (communication bottleneck, memory contention, kernel inefficiency). Anything above 55% on a dense transformer at scale is either a small model, a benchmark artifact, or a very careful implementation — mention it if you achieved it, name the techniques that got you there (torch.compile, flash-attention, custom kernels, carefully tuned TP degree).

In an interview context: volunteering MFU as a metric you'd track for a training infrastructure question signals hands-on training experience. Connecting it to the 6N formula ("to estimate GPU-hours, I'd take the Chinchilla compute budget, divide by peak FLOP/s × MFU, and divide by n_GPUs") is the senior-level answer.

---

## 2. Community & support (roadmap)

This section is framed as guidance for anyone running a cohort with this material, not a description of infrastructure that currently exists.

### What a well-run learner cohort provides

The biggest gap between solo self-study and a structured program is not content — it is **accountability and calibrated feedback**. Reading a chapter and recognizing the material is easy. Reconstructing a design under time pressure in front of another person reveals the actual gap between recognition and production.

**Study groups (2–4 people, weekly).** The high-value format is not reading together but *designing together*. Each person picks a mock question, answers it out loud for 35 minutes while the others listen, then receives structured feedback on the rubric axes from the capstones chapter: assumptions stated, tradeoffs compared, numbers produced, failure modes volunteered. Calibrating to "what does a senior answer actually sound like" is much faster when you have a reference human answer to compare against than when you self-grade in isolation.

**Design-review circles (4–8 people, biweekly).** Each participant presents their current project's design — not the code, the design — and receives adversarial questions. The project review format mirrors the interview format almost exactly. The goal is to get comfortable defending design decisions under pushback. "Why not just use X?" "What breaks at 10× scale?" "You said you'd monitor Y — how?" These are the questions interviewers ask; practicing with humans who will actually push back is the fastest calibration.

**Mock-interview pairing.** Pairs within a cohort swap interviewer/interviewee roles on a rotating schedule. The interviewer reads the rubric, manages time, probes depth by drilling ("go deeper on how PagedAttention handles fragmentation"), and grades on the five axes. This is uncomfortable. It is also the only practice format that approximates the actual test. If you are running a cohort, building in at least two mock-interview pairs per participant in the final two weeks is the highest-leverage investment you can make in outcomes.

**Asynchronous Q&A (forum or message channel).** The most common use case is not "explain this concept" but "I got this question in a practice interview and I answered Y — was that wrong?" A searchable async channel where past answers are visible reduces repeated questions and builds a shared knowledge artifact over time. Moderating for quality (not quantity) is what distinguishes a useful async community from noise.

If you are building a community infrastructure for this course, the minimal viable version is: a shared async channel, a biweekly mock-interview pairing schedule, and one or two experienced practitioners willing to drop into design-review circles occasionally. The gamification and progress-tracker features are motivational but secondary to the feedback structures above.

---

## 3. Staying current

### 3.1 What is likely to shift by 2027

The course covers the durable core of ML system design as of 2026. Some of it will still be durable in 2027. Some of it will be the old way. Here is a honest forward look at the volatile areas — not predictions, but directions the field is clearly moving:

**Test-time scaling becomes a first-class design variable.** In 2026, inference-time compute scaling (chain-of-thought, majority voting, best-of-N, tree search, RLVR-trained reasoning models) is already a significant architectural force, primarily in reasoning-heavy domains. By 2027 it is likely to be a standard system design consideration for any non-trivial task: the serving chapter's latency-throughput tradeoff becomes a three-way tradeoff when you add "how much thinking compute per request." Expect interview questions about when to scale test-time compute versus training compute, how to budget it per request class, and how to serve it efficiently (speculative reasoning, early-exit under confidence thresholds, KV-cache reuse across rollout branches). The capacity math in this chapter still applies; the design space expands.

**More efficient MoE with finer routing.** Sparse mixture-of-experts (MoE) models at scale are already mainstream in 2026 — the FLOPs-per-token formula in this chapter's cheat sheet applies, but the parameter-to-active-parameter ratio is now a first-class design axis. By 2027 expect: finer-grained expert granularity (more, smaller experts), smarter routing (shared experts for common knowledge, specialized experts for domains), and MoE at the post-training level (different expert subsets fine-tuned per downstream task). The serving implications are non-trivial: expert parallelism across nodes means network topology matters, and the KV-cache math changes because active-parameter count per token is lower than total-parameter count.

**Next-generation accelerator architectures.** The GPU-centric world is widening. Groq-style LPUs (streaming matrix engines with on-chip SRAM, no HBM), Tenstorrent's RISC-V mesh, wafer-scale engines (Cerebras), and dedicated inference chips from hyperscalers are all in production or late-stage deployment. By 2027 the self-hosted serving decision is not just "which GPU configuration" but "which compute architecture for which workload class." The HBM bandwidth-bound decode rule in this chapter generalizes: any architecture with on-chip SRAM that dwarfs its external bandwidth changes the arithmetic in favor of latency-sensitive workloads. The durable skill is being able to re-derive the bandwidth-vs-compute bound for an unfamiliar chip from its spec sheet.

**Multimodal as the default.** In 2026, text-primary models with bolt-on vision are still common. By 2027 the expectation is that a capable model handles text, images, and audio natively — not as adapters but as first-class modalities in the same token stream. For system design, this shifts the data-engineering chapter's feature-platform discussions (how do you store and retrieve image embeddings alongside text?), the KV-cache sizing (multi-modal tokens are longer; video frames especially), and the evaluation chapter (judge models need to evaluate multi-modal outputs). The modality-routing patterns from the agentic systems chapter — "does this request need vision?" — become more important as models become capable of handling it natively at higher cost.

**What is not going to change.** The offline/online/feedback-loop anatomy from the foundations chapter is as durable as system design gets. The cascade-funnel pattern, training-serving skew, feedback-loop integrity, eval-driven development, point-in-time correctness, the multi-stage ranking funnel — none of these are going away. Interviewers testing for these things in 2027 interviews are testing the same underlying engineering judgments they tested in 2024. The specific tools (vLLM → whatever succeeds it, DPO → RLVR → whatever post-training recipe is current) will rotate; the ability to reason about latency-throughput tradeoffs, capacity math, and failure modes will not.

### 3.2 How to keep the durable core while re-checking the volatile tool layer

The practical discipline for a working ML engineer: **separate your knowledge into tiers by half-life**.

**Half-life > 3 years (durable core):** the capacity math formulas, the system anatomy (offline/online/feedback), the multi-stage funnel, cascade economics, the evaluation discipline (offline metrics → online metrics → guardrails → CI gate), training-serving skew and its mitigations, data flywheel mechanics. These evolve slowly enough that you can internalize them once and they will pay off for years.

**Half-life ~1 year (stable tools):** specific model architecture choices (GQA vs MQA, which attention kernel), specific serving frameworks (vLLM, SGLang), specific post-training recipes (which RLVR variant is dominant). Read the release notes and model cards of whatever the leading systems are when you enter a role. Update your serving-chapter knowledge by reading the current serving-framework documentation and one or two engineering blog posts per year.

**Half-life < 6 months (volatile):** specific benchmark scores, pricing from any particular provider, exact version pins for any package, claims about what the "best" model is. Never memorize these as facts; always re-check before citing them in any context that matters.

### 3.3 Quarterly maintainer review checklist

For anyone maintaining or teaching this course, a concrete quarterly review practice:

**Content review (2–3 hours per quarter):**
- [ ] Check the serving chapter: has a new version of the leading serving framework changed the default tuning guidance? Has disaggregated prefill/decode moved from advanced to table-stakes?
- [ ] Check the post-training chapter: is RLVR/GRPO still the dominant post-training paradigm, or has a successor stabilized?
- [ ] Check the inference chapter: are the quantization recommendations (FP8 default, INT4 for memory pressure) still accurate? Have new calibration methods changed the sensitivity-mapping practice?
- [ ] Check the evaluation chapter: has the LLM-as-judge calibration guidance been superseded by a stronger approach?
- [ ] Check the agentic systems chapter: have MCP or tool-call specifications evolved in ways that change the security-posture guidance?

**Mock questions audit (1 hour per quarter):**
- [ ] Are the six mock questions in the capstones chapter still representative of what FAANG-level interviews are actually testing? Add one new question per year if the distribution has shifted.
- [ ] Are the model answers still defensible? The most common drift: a specific architectural claim (e.g., "use vLLM for serving") is superseded by a better-supported alternative.

**Deprecation check:**
- [ ] Any explicit tool recommendations (library installs, exact commands) — verify they still work or add a "representative as of [year]" qualifier.
- [ ] Pricing figures — these have a six-month half-life; replace with "current provider pricing sheets" references rather than embedding specific numbers.

The goal is not to keep the course perfectly current — that is impossible at this field's pace. The goal is to prevent *wrong* content from accumulating. Outdated-but-labeled-as-outdated is fine. Wrong-and-unlabeled is what misleads learners.

---

## 4. Course structure rationale

### 4.1 The current sequence (modules 01–10)

The ten-module sequence is not alphabetical or encyclopedic — it is causal. Each module introduces vocabulary and mental models that later modules assume.

**Core group (modules 01–09):** the foundations chapter establishes the interview framework and the offline/online/feedback anatomy that every subsequent chapter uses without re-explaining. The data-engineering chapter introduces point-in-time correctness and the feature-store pattern that the training chapter (which trains on features) and the serving chapter (which reads features at inference time) both assume. The training chapter introduces the FLOPs and parallelism vocabulary that the inference chapter relies on for capacity math. The inference chapter introduces quantization and latency arithmetic that the serving chapter uses. The retrieval chapter introduces ANN indexes and hybrid search that the agentic chapter uses for tools. The classic ML chapter closes the loop: it applies the full stack (features, training, serving, evaluation) to the highest-stakes domain (recsys, fraud) where the systems design tradeoffs are most sharply economic. The evaluation chapter is last in the core because it requires experience with all the prior chapters' failure modes to make sense — "you can't define a good eval until you know what can go wrong."

**Capstones chapter (module 10):** the integration point. The six mock questions are not new content — they are synthesis exercises that require all nine prior modules to answer well. If you read modules 01–09 and attempt the mock questions before reading the model answers, you will find the gaps that your self-study of individual chapters missed.

### 4.2 The extension modules (11–17)

The core (01–10) is the design-interview spine. Modules 11–17 extend it outward — from "can you design the system" to "can you cost it, ship it, situate it in a real domain, build it end to end, and get hired for it." They are sequential but the core does not depend on them, so a learner short on time can stop at 10 and still be interview-ready for a generalist loop.

**Practical modules (11–12):** the economics chapter (11) is the consolidated home for cost math — cloud/API pricing tables, the build-vs-buy crossover derivation, and a TCO worksheet; individual chapters point here rather than each carrying its own pricing. The DevOps & deployment chapter (12) is the "how you actually ship it" reference — Dockerfiles, Kubernetes manifests, autoscaling, and a full CI/CD pipeline with an eval gate; the training and serving chapters point here for containers and orchestration.

**Applied modules (13–15):** the case-studies chapter (13) traces five end-to-end systems (a ranking funnel, an LLM code assistant, an enterprise RAG assistant, an agentic executor, a mid-size post-training pipeline) so the abstract patterns become narratives you can retell in an interview. The domain-variations chapter (14) shows how the invariant core bends for healthcare, finance, autonomous systems, and manufacturing. The cumulative capstone (15) is the single end-to-end build that spans every core module — the portfolio centerpiece.

**Career & meta (16–17):** the career chapter (16) covers the full job-search arc for ML engineers — portfolio structure, reading a JD for its real technical expectations, adapting to interviewer type, the MLE-vs-research fork, career progression, and a durable reading list. This chapter (17) is the course-design rationale, the capacity-math cheat sheet, and the staying-current policy you are reading now.

### 4.3 Which modules a time-constrained learner can treat as electives

If you have four weeks, not twelve, and you are targeting a specific role, here is the minimum viable path:

**For LLM serving / GenAI platform roles:** 01 (framework) → 04 (inference arithmetic) → 05 (LLM serving) → 09 (evaluation) → 10 (mock questions 1 and 6). Skip or skim 02, 03, 06, 07, 08.

**For MLOps / platform engineer roles:** 01 → 02 (feature platform) → 03 (training infra, skim the post-training sections) → 09 (eval + observability) → 10. Skip 06, 07.

**For applied ML / recsys / search roles:** 01 → 02 → 06 (retrieval) → 08 (classic ML, read in full) → 09 → 10 (mock questions 3 and 4). Skip 07.

**For agentic / AI product roles:** 01 → 05 (serving, focus on KV-cache and routing) → 06 (retrieval, it's the tool stack) → 07 (agentic, read in full) → 09 → 10 (mock question 6).

The risk of the elective path: you will have gaps that surface as interview questions. "You said you'd use a feature store — what is point-in-time correctness?" will catch a candidate who skipped module 02. Know your gaps before you enter the loop so you can manage the scope of the design question rather than being caught flat-footed on a concept you have never seen.

---

## Interview Q&A

**Q: "Walk me through how you'd structure a 12-week ML systems design curriculum for a junior engineer who is also working full-time."**

A: The constraint is cognitive load per week, not total content. I'd organize it as: (a) two weeks on foundations and data — the vocabulary and the feature-platform fundamentals everything else builds on; (b) one heavy week on training infrastructure, which is the conceptually densest module; (c) two weeks on inference and serving, where the capacity math becomes practical through the inference arithmetic and LLM-serving chapters; (d) two weeks on retrieval and agentic systems, which are the fastest-moving areas and require hands-on projects to internalize; (e) one week on classic ML systems, which ties the whole funnel together; (f) one week on evaluation and observability, which crystallizes what monitoring actually means at each layer; (g) two weeks of integration — the capstone and timed mock interviews. The single most important structural decision: the project after each module, done immediately, before the next module starts. Knowledge that hasn't been stress-tested by an implementation is only recognition, not production ability.

**Q: "What is MFU and why would you track it during training?"**

A: MFU is Model FLOP Utilization — the ratio of observed compute throughput to theoretical peak FLOP/s of the hardware. You compute it as (tokens per second × FLOPs per token) / (number of GPUs × peak FLOP/s per GPU). A well-tuned run on H100s lands at 35–50%; budgeting 40% is standard for planning. Tracking it during training gives you early warning of regression: if MFU drops from 42% to 28% after a code change, something changed in the communication pattern, kernel selection, or memory layout — identify it before you waste GPU-hours at reduced efficiency. It is also the number you'd cite when estimating training cost from first principles: take the Chinchilla compute budget (6ND), divide by (n_GPUs × peak_FLOP/s × MFU), get GPU-hours.

**Q: "If you had to describe what makes this course different from 'Design X' articles online, what would you say?"**

A: Three things. First, it covers both genres — classic ML systems (recsys, search, fraud) and GenAI systems (LLM serving, RAG, agents) — and the interview landscape in 2026 tests both; most written material covers one or the other. Second, it runs the capacity math throughout: FLOPs per token, KV-cache sizing, bandwidth-bound decode, MFU — these are not appendices, they are practiced in each module until they are automatic. Third, it is explicit about the feedback loop and the eval story, which are the components juniors most consistently hand-wave and which interviewers most consistently probe to separate senior from junior candidates.

---

*This course is a living document. The formulas in the cheat sheet are durable; the tool recommendations have a shelf life. If you are reading this more than a year after the current date, re-check the volatile layer before citing specific tool guidance in an interview.*
