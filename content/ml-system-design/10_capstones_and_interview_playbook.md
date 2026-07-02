# Module 10 — Capstones & the Interview Playbook

## Why this module matters

Everything before this was knowledge. This chapter is performance: converting all of it into 45-minute answers under pressure, and into portfolio projects that prove you can actually build these systems. Work through the mock questions with a timer and a whiteboard before reading the model answers.

## 1. The rubric — what interviewers actually grade

Across companies, the rubric converges on five axes:

1. **Problem navigation.** Did you scope before designing? State assumptions? Drive the conversation rather than waiting to be led?
2. **Breadth of the design.** Did you cover the full lifecycle — data, training, serving, evaluation, monitoring, iteration — or just the model box?
3. **Depth on demand.** When the interviewer drilled into one component, could you go two levels deeper (the math, the failure modes, the alternatives)?
4. **Tradeoff reasoning.** Every choice framed as "X over Y because Z, and I'd revisit if W." Junior answers assert; senior answers compare.
5. **Practicality.** Capacity math with stated assumptions, cost awareness, simplest-thing-first with upgrade triggers, operational concerns (rollback, on-call, abuse).

What a *junior* answer sounds like: jumps to a model name, single path, no metrics discussion, no numbers, silent on failure modes. What a *senior* answer sounds like: ten minutes of requirements and metrics, a drawn funnel/pipeline, deliberate simplifications stated out loud, one or two volunteered war stories ("the classic bug here is point-in-time leakage..."), numbers on demand.

A useful self-test: record yourself answering, then count (a) assumptions stated, (b) tradeoffs explicitly compared, (c) numbers produced, (d) failure modes volunteered. Fewer than 3 of each = keep drilling.

## 2. Mock interview questions with model answers

Below are six full questions spanning both genres. Model answers are condensed outlines — in a real interview each expands to 35–45 minutes. Attempt each yourself first.

---

### Mock 1 — "Design the serving infrastructure for a ChatGPT-style assistant"

**Requirements (state aloud):** 5M DAU assumed, peak 5% concurrent → 250k active; mixed chat (short) + document (long-prompt) traffic; SLOs: TTFT < 1.5 s p99, ITL < 60 ms p99; multi-turn; cost matters.

**Architecture outline:**

- **Model tier:** an 8–70B served fleet (or MoE) behind an OpenAI-compatible gateway; FP8 weights + FP8 KV as the default precision; quantization validated on task evals, not just perplexity.
- **Engine layer:** vLLM/SGLang replicas — continuous batching, PagedAttention, automatic prefix caching, chunked prefill on. Structured decoding for tool calls.
- **Routing layer:** session-affine, KV-cache-aware routing so multi-turn requests land on the replica holding their prefix; KV hit rate tracked as a first-class metric.
- **Phase split:** start co-located with chunked prefill; **disaggregate prefill/decode** when long-document traffic makes ITL p99 unholdable — separate pools, a dedicated KV-transfer engine over RDMA-class paths, independent scaling (this is the 2026-standard evolution; be ready to name the goodput-optimal disaggregation and KV-cache-centric architectures behind it).
- **Capacity math (do it live):** ~250k active users, message every 40 s → ~6k req/s; ~300 output tokens → ~1.9M decode tok/s peak. At ~5–10k tok/s per replica-equivalent → hundreds of GPUs; then show the levers: prefix caching kills most multi-turn prefill, FP8 doubles batch, cascade routes easy traffic to a small model.
- **Reliability/ops:** goodput-based autoscaling with headroom, canary per engine upgrade, per-tenant rate limits, abuse filtering before the GPU.
- **Failure modes to volunteer:** cache-hit collapse from a random load balancer; one mega-prompt stalling decode (chunked prefill exists for this); silent quality drift after engine/quant upgrades → regression evals on the deployment artifact.

---

### Mock 2 — "Design a document-extraction system processing 10M docs/day"

**Requirements:** invoices/forms/IDs arriving as scans and photos; structured-field output; per-doc latency seconds-to-minutes (batch-tolerant) but some synchronous traffic; accuracy is contractual; cost per doc matters at 10M/day.

**Architecture outline:**

- **Ingest & triage:** queue (Kafka/SQS); a tiny classifier routes by doc type and quality (blur/skew detection); rectification (corner detection → perspective warp) before any recognition — geometric distortion is the dominant real-world accuracy killer.
- **Cascade as the core economic design:** Tier 1 — small specialized models (classical CV + compact OCR + a fine-tuned 1–4B VLM) handle the 90%+ standard cases at ~zero marginal cost; Tier 2 — frontier VLM API for low-confidence/exotic docs; Tier 3 — human review queue sized to budget. Confidence calibration is the routing currency — per-field confidence must be calibrated (see the evaluation chapter) or the cascade routes garbage.
- **Training story:** synthetic data factory (template-rendered docs with pixel-perfect labels) + measured physical-degradation augmentation + a small real fine-tuning set; report transfer accuracy on real held-out captures, never synthetic-on-synthetic.
- **Serving:** Tier 1 on a small GPU pool with continuous batching (throughput-optimized, relaxed latency); sync traffic on a separate latency-tuned pool.
- **Eval & monitoring:** per-field precision/recall on a golden set sliced by doc type and capture quality; production flywheel — human-review corrections flow back as labels; drift monitor on input image stats (new doc templates appear constantly).
- **Numbers:** 10M docs/day ≈ 115/s sustained; at ~1 s GPU-time per doc on Tier 1 → ~120 GPUs naive, cut by batching and routing 30% of simple docs to CPU-only classical pipeline; Tier 2 at 2% escalation = 200k API calls/day — show the cost line and why the cascade exists.

---

### Mock 3 — "Design video recommendations for a short-video app"

**Outline:** business metric = retention proxied by time-well-spent (state the proxy trap: raw watch-time rewards rabbit holes; add explicit-feedback and report-rate guardrails). **Funnel:** candidate gen — two-tower (user sequence tower, in-batch negatives + logQ), plus fresh/trending, followed-creator, and exploration sources; ANN over item embeddings refreshed near-real-time for new uploads. **Ranking:** multi-task (P(complete), P(like), P(share), P(report)) with a value formula encoding product strategy; sequence features from the last N interactions; position-debiasing via position-as-feature + randomization slice. **Re-rank:** diversity (MMR), creator fairness/exposure floors, integrity filters. **Cold start:** content embeddings from a VLM on frames+audio+caption (new items recommendable at upload), semantic-ID generative retrieval as the frontier upgrade (point to HSTU-style generative recommenders and semantic IDs as where this stack is heading). **Feedback integrity:** logged propensities, exploration budget, popularity-bias monitoring. **Eval:** offline replay with care (exposure bias), interleaving for ranker iterations, A/B with retention holdbacks measured over weeks, not days.

---

### Mock 4 — "Design semantic search for an e-commerce catalog"

Use the semantic-search answer from the retrieval chapter as the skeleton: hybrid BM25+dense with RRF, fine-tuned bi-encoder on click/purchase pairs, IVF-PQ/sharded-HNSW at 200M items, cross-encoder rerank of top 100, streaming freshness tier, filters pre/post depending on selectivity. Add the e-commerce-specific layers: query understanding (spell, attribute extraction → filters, LLM rewriting of conversational queries), business re-rank (margin, availability, sponsored slots with calibrated relevance floors), and the eval story — offline recall@k against purchase logs, online interleaving, guardrail = zero-result rate and add-to-cart rate. Volunteer the classic failure: dense-only retrieval missing exact model numbers ("SM-G998B") → hybrid is non-negotiable in product search.

---

### Mock 5 — "Design a real-time payment-fraud system"

**Outline:** decision in < 200 ms inside the authorization flow; actions = approve / step-up / block / review. **Features:** three freshness tiers (the worked answer in the data-engineering chapter) — the streaming velocity counters are the heart. **Models:** calibrated GBDT baseline (interpretable, fast) + GNN over the entity graph (devices/cards/addresses) for ring detection, scores fused; threshold set by asymmetric cost analysis with step-up as the cheap middle action. **Labels:** chargebacks at 60-day delay → maturity-aware training, fast proxies (review verdicts) for monitoring (as covered in the evaluation chapter). **Adversarial posture:** champion/challenger weekly retrains, novelty-cluster alerts, rules-engine fallback kill-switch. **Eval:** recall@fixed-FPR and $-weighted metrics, never accuracy; review-queue precision as the live canary; shadow-then-canary deployment because blocking is state-mutating (simulate, don't execute, in shadow).

---

### Mock 6 — "Design a coding agent for enterprise codebases"

**Outline:** workflow-vs-agent fork — code Q&A and review-comment generation are workflows; "fix this failing test" is a bounded agent. **Harness:** tools = code search (hybrid + symbol index), file read, test runner, patch apply behind gates; sandboxed execution (Firecracker-class) with no prod credentials and egress allowlists; step/cost budgets and loop circuit breakers. **Context engineering:** stable preamble for KV-cache hits, just-in-time file reading (never dump the repo), compaction at breakpoints, repo-specific notes file. **Serving:** session-affine routing, prefix caching as the dominant cost lever (agent loops resend growing transcripts — see the serving chapter's treatment of agent-loop caching). **Security:** repo content is untrusted input (injection via README/comments is a demonstrated attack); least-privilege tokens; human approval on merge. **Eval:** SWE-bench-style internal harness from the company's own resolved tickets, pass@1 and pass^k gates in CI, trajectory metrics (redundant-step rate, recovery rate), online = PR acceptance rate and human-edit distance. **Cost:** tokens per resolved task as the unit economic, cascade easy tasks to a small distilled model.

---

---

### Mock 7 — "Design a voice assistant for hospital inpatient units"

**Requirements (state aloud):** nurses and physicians dictating clinical notes, placing orders, and asking protocol questions hands-free; real-time (patient room, mid-procedure); latency budget <500 ms end-to-end turn; HIPAA compliance; medical domain; escalate-to-human for safety-critical actions; English + regional accent variation; peak usage during shift changes.

**Architecture fork — cascaded vs end-to-end:**

The first design decision the interviewer is probing: **cascaded pipeline** (ASR → LLM → TTS, separate models per modality) vs **end-to-end speech-to-speech** (a single model that accepts audio and produces audio).

- **Cascaded pipeline**: established, separately optimizable, modular, each stage can be specialized (medical ASR fine-tuned on clinical dictation; LLM fine-tuned on protocols; TTS with clinical cadence). The dominant production choice today. Downside: each stage adds latency and error propagation — ASR errors compound into LLM errors.
- **End-to-end speech-to-speech**: **Qwen2.5-Omni** (Thinker/Talker architecture) is the current reference — a unified model with a reasoning "Thinker" component and a streaming "Talker" component that generates audio tokens directly, enabling native real-time interaction without a separate TTS stage. Advantages: no ASR error propagation, native prosody modeling, barge-in handled at the model level. Tradeoffs: less modularity for medical domain fine-tuning (fine-tuning the full audio stack is harder than fine-tuning text-only LLM), larger serving footprint, less production deployment history. State the fork explicitly and defend your choice — the senior answer is cascaded for initial deployment with a migration path to end-to-end as the technology matures.

**Latency budget — 500 ms total, broken down:**

| Stage | Budget | Notes |
|---|---|---|
| VAD + audio capture | ~20 ms | Voice activity detection to detect end-of-utterance |
| ASR (streaming) | ~80–120 ms | Streaming partial transcripts during speech; final at end-of-utterance |
| LLM (TTFT) | ~150–200 ms | First token after ASR final; prefix-cached system prompt critical |
| TTS (first audio chunk) | ~80–100 ms | Streaming synthesis; first chunk before full text is ready |
| Network + audio playback start | ~30–50 ms | Edge deployment reduces this dramatically |

**Total: ~360–490 ms** — achievable with co-located edge compute in the facility. Cloud-only deployment adds ~50–100 ms round-trip and likely breaks the SLO.

**Real-time mechanics:**

- **VAD (voice activity detection)**: a lightweight model (Silero VAD, WebRTC VAD) running on the device detects speech start/end without sending audio to the server until an utterance completes. Critical for latency and privacy — audio is not streamed continuously.
- **Streaming partial transcripts**: ASR emits partial hypotheses during dictation (useful for showing the user what's being heard), with a final hypothesis on end-of-utterance. The LLM processes the final hypothesis, not partials.
- **Barge-in / interruption handling**: if the user speaks while the assistant is responding, the system must detect the new speech, abort the current TTS stream, and re-enter the ASR pipeline. This is an edge-case coordination problem — requires a session state machine tracking "speaking" vs "listening" vs "processing" states, with abort signals to the TTS synthesis service.
- **Streaming TTS**: synthesis begins on the first sentence of the LLM output, not after the full response is generated. The LLM emits text tokens; the TTS service streams audio tokens back as text accumulates. First audio chunk is playable before the LLM has finished — this is the primary technique for hitting <500 ms perceived latency.

**Safety and guardrails (medical domain):**

- **Medical-domain guardrails**: output rails that check clinical plausibility — drug-drug interaction flags (rule-based, not model-generated), dose-range validation, protocol-deviation detection. These are deterministic rule checks, not model judgment — model-generated safety checks in a clinical setting are a liability without human-in-the-loop verification.
- **Escalate-to-human triggers**: any order above a threshold criticality level (e.g., high-alert medications, code-status changes, DNR orders) is confirmed with a "please confirm: you are ordering X for patient Y" read-back + explicit verbal confirmation before action is taken, and simultaneously queued for nursing station review. The model does not act unilaterally on high-blast-radius clinical decisions.
- **HIPAA pointer**: patient name, MRN, and clinical content are PHI. Audio data must not leave the facility boundary (edge inference is the privacy control); transcripts are PHI and subject to access logging, retention limits, and breach notification requirements. Any cloud component must be a HIPAA Business Associate.

**Eval:** ASR word error rate on clinical vocabulary (drug names, anatomical terms — the long tail where consumer ASR fails); end-to-end task success rate on a golden set of clinical scenarios; latency p50/p99 under peak concurrency (shift-change load test); safety-critical scenario eval: does the system correctly escalate and refuse to act unilaterally on high-alert orders?

---

### Reading the room — adapting to interviewer type and handling gaps

**Who is in the room** shapes the vocabulary you reach for, not the depth of analysis. A product-side interviewer rewards user-facing framing: "this reduces zero-result rate by X points" lands better than "this improves NDCG@10." A research interviewer rewards mathematical precision — name the objective function, state the loss formulation, acknowledge convergence properties; the product business case is secondary. An infra/platform interviewer cares about operational contracts: SLOs, rollback mechanism, resource efficiency, on-call load. Read the room by asking a calibrating question early ("are you more interested in the modeling approach or the system reliability story?") and adjusting as the interviewer signals which threads to pull. Most interviewers telegraph what they care about in the first follow-up question — treat it as a routing signal, not a test you passed or failed.

**Handling gaps honestly.** "I don't know the exact paper, but here's how I'd reason about it" scores higher than a confident wrong answer, almost universally. Signal competence three ways: (a) name what you do know that is adjacent; (b) derive what you can from first principles — the interviewer is watching the reasoning, not the recall; (c) flag the gap cleanly ("my instinct says X because of Y — this is where I'd go deeper before committing"). What damages a candidate is not the gap itself but papering over it: a senior interviewer will follow up on any claim you made, and if that follow-up exposes a fabrication the trust damage is larger than the gap ever was. A clean "I'm not solid here" preserves your credibility for the components you do own. Audience-adaptation tactics for portfolio materials and written narratives — distinct from in-interview adjustment — are covered in the career and portfolio strategy chapter (module 16).

---

## 3. Capstone projects (pick one, 3–4 weekends)

**Capstone A — Document-AI platform end-to-end** (data engineering, inference optimization, retrieval, evaluation). Synthetic doc generator → fine-tune + QAT a small VLM for field extraction → quantized deployment with a serving endpoint → ColPali-style page-image retrieval over a doc corpus → full eval harness (per-field metrics, real-transfer measurement, judge calibration) → tracing + CI eval gate. *Deliverable: repo + a write-up of synthetic-to-real transfer numbers.*

**Capstone B — Mini serving platform** (training, inference optimization, serving, evaluation). Post-train a small model (SFT→DPO→GRPO) → produce FP8/INT4 artifacts with sensitivity-mapped mixed precision → serve via a modern engine with prefix caching + structured output → load-test to a latency-throughput frontier with $/1M-token analysis → wire tracing and a regression-eval gate. *Deliverable: the benchmark report you'd hand a platform team.*

**Capstone C — Full-funnel recommender** (data engineering, classic ML systems, evaluation). Feature platform with point-in-time correctness → two-tower + ANN retrieval (logQ ablation) → multi-task ranker → diversity re-rank → position-bias study → offline eval suite + simulated A/B with power analysis. *Deliverable: funnel metrics table + the bias-study write-up.*

Each capstone is interview ammunition: every mock question above can be answered with "I actually built this — here's what surprised me."

### The cumulative build — module 15

Capstones A, B, and C are scoped for focused, isolated practice — each exercises three or four modules and is tight enough to finish in a long weekend. **Module 15 is the single end-to-end system** that spans all modules: real data ingestion, a feature platform with point-in-time correctness, a training and post-training loop, a quantized serving stack, a full evaluation harness, and an operations story — all wired together. Where the per-module capstones build vertical depth, the cumulative build tests whether the seams hold: does your feature platform feed your training pipeline without leakage? Does your eval CI gate catch a regression in the serving artifact after a quantization change? The per-module capstones make you dangerous in a design conversation; the cumulative build makes you credible when an interviewer asks for a GitHub link. Components you build in the earlier capstones port directly into module 15 — plan the integration starting in week 10 so the cumulative build inherits rather than duplicates.

## 4. The 12-week plan, restated as outcomes

- **Wk 1–2:** modules 01–02 done; two design docs + feature-platform project in your repo.
- **Wk 3–4:** module 03; one full SFT→DPO→GRPO run with an MFU number you measured.
- **Wk 5–6:** modules 04–05; a quantization sensitivity map and a latency-throughput curve you produced.
- **Wk 7–8:** modules 06–07; a RAG eval table and an agent that survives a prompt-injection you authored.
- **Wk 9:** module 08; a two-stage recommender with a logQ ablation.
- **Wk 10:** module 09; tracing + a CI eval gate on an earlier project.
- **Wk 11–12:** capstone + four recorded 45-min mocks (use the six questions above; grade yourself on the rubric).

## 5. The ten most common mistakes (and their fixes)

1. **Designing the model before the metric.** Fix: metrics section before any architecture, every time.
2. **No numbers.** Fix: memorize the constants (6ND, 2N/token, KV formula, ~10¹⁵ FLOP/s per modern GPU) and practice the arithmetic out loud.
3. **Single-path answers.** Fix: for every component, name one alternative and the condition that would flip your choice.
4. **Ignoring the feedback loop.** Fix: end every design with "and here's how production data improves the system."
5. **Treating evals as an afterthought.** Fix: golden set + judge calibration + CI gate is a 3-sentence pattern — deploy it in every GenAI answer.
6. **Over-engineering on day one.** Fix: "simplest thing first, with named upgrade triggers" — disaggregation, multi-agent, GraphRAG, generative recommenders are all *upgrades with conditions*, not starting points.
7. **Forgetting training-serving skew.** Fix: it's the most common real-world failure; volunteer it.
8. **Hand-waving cost.** Fix: one $/1M-tokens or GPU-count estimate per answer.
9. **No failure modes.** Fix: keep a personal list of three war stories per module (the projects give them to you).
10. **Memorizing architectures instead of tradeoffs.** Fix: interviewers change one requirement mid-answer specifically to test this — practice re-deriving the design when latency drops 10× or scale grows 100×.

## Interview-prep checklist

- Drill the mock questions above out loud, on a timer, with a whiteboard — the rubric at the top of this chapter is your grading sheet.
- Read a few real ML system design case studies each week from the companies you are targeting; large public collections of these exist and are the best breadth-builders.
- In the month before the loop, read the public engineering write-ups of your target companies — interviewers ask about their own stack's problems.
- Record yourself answering, then count assumptions stated, tradeoffs compared, numbers produced, and failure modes volunteered. That count is the fastest feedback signal you have.

## For job search, portfolio structure, and audience strategy — see module 16

This chapter is deliberately scoped to in-interview mechanics: the rubric, the mock questions, the 12-week schedule, and the ten common mistakes. What it does not cover is everything that happens before and after you are in the room — how to structure a portfolio so the right audiences surface the right signals, how to narrate a project for a research role versus an ML platform role, how to sequence a job search, and how to pitch the same system design work differently in written form for different company types. All of that is in the career and portfolio strategy chapter (module 16). In particular, module 16 goes deeper on audience-specific framing; the "reading the room" note above covers only real-time in-interview adjustment — the upstream portfolio and written-narrative strategy is in module 16.

---

*End of course. Build the projects, do the mocks out loud, and remember: the interview is a conversation about tradeoffs, not a recitation of architectures.*
