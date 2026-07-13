# 26 — Capstone: Production Support Intelligence System

Every module in this course taught one skill in isolation. This capstone makes you assemble them
into a system a company would actually pay for: a **support intelligence layer** that reads an
incoming customer ticket, figures out what it is about, pulls the facts it needs from your product
docs, and either answers it, extracts structured data from it, or hands it to a human — with a
latency budget, a cost ceiling, and an eval harness that blocks a regression before it ships.

This is the project that gets interviews, and the reason is narrow: almost every applied-NLP job
opening is some version of this system. Ticket triage, doc-grounded assistants, extraction
pipelines, and "route to the right team or human" logic are the bread and butter of applied NLP in
2026. A candidate who has built the whole loop — not a notebook that fine-tunes one classifier, but
a routed, evaluated, budgeted service — is demonstrating exactly the judgment the role tests for:
knowing when a 140M encoder on a CPU beats a frontier API, how to measure faithfulness instead of
vibes, and where the failure modes hide. You are not building a demo. You are building the artifact
you will spend your on-site talking about.

Everything here runs on free Colab (T4, 16 GB) plus your laptop. No frontier training, no paid GPU.
The only paid call is an optional API fallback you can stub out entirely.

## What you are building

Four models behind one router, one eval harness, and one budget.

```
                       ┌─────────────────────────────────────────────┐
   incoming ticket ───▶│                  ROUTER                      │
   "order #A-4471 on   │  intent conf?  entity coverage?  retrieval   │
    v2.3 keeps         │  hit?          risk flags?                   │
    crashing on        └───┬───────────────┬───────────────┬─────────┘
    export"                │               │               │
                           ▼               ▼               ▼
                 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                 │ CLASSIFY-    │  │ RETRIEVE-AND │  │ ESCALATE-TO  │
                 │ ONLY         │  │ -ANSWER      │  │ -HUMAN       │
                 └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                        │                 │                 │
   ┌────────────────────┼─────────────────┼─────────┐       │
   ▼                    ▼                 ▼          ▼       ▼
┌────────────┐   ┌────────────┐   ┌──────────────────────┐ ┌──────────┐
│ INTENT CLF │   │ NER        │   │ RAG                   │ │ ticket + │
│ DeBERTa-   │   │ products,  │   │ embed→FAISS→rerank    │ │ enriched │
│ v3-small   │   │ versions,  │   │ →small LLM answer     │ │ metadata │
│ (CPU)      │   │ order-ids  │   │ (GPU) / API fallback  │ │ to queue │
└────────────┘   └────────────┘   └──────────────────────┘ └──────────┘
        │                │                    │
        └────────────────┴────────────────────┴──────────┐
                                                          ▼
                                              ┌───────────────────────┐
                                              │ EVAL HARNESS + BUDGET │
                                              │ golden set · F1 ·      │
                                              │ hit@k · faithfulness · │
                                              │ CI regression gate     │
                                              └───────────────────────┘
```

The intent classifier and NER share a frozen encoder backbone and run on CPU for cents. The RAG path
is the only one that touches a generative model, so the router's job is largely to *avoid* invoking
it when a cheaper path suffices. That single design decision — cheap deterministic models in front,
expensive generation behind a gate — is the cost story you will tell in the interview.

## Recommended models (all free-tier friendly)

| Component | Model | Where it runs | Why |
|---|---|---|---|
| Intent classifier | `microsoft/deberta-v3-small` (~140M) | CPU | Strong small encoder; fine-tunes on T4 in minutes |
| NER | same backbone, token-classification head | CPU | One backbone, two heads — cheap to serve |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) | CPU | Fast, good enough for a small doc corpus |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU/GPU | Cross-encoder relevance on the top-k |
| Answer generation | `Qwen/Qwen2.5-0.5B-Instruct` or `SmolLM2-360M-Instruct` | GPU (T4) | Fits free Colab; grounded answers are short |
| API fallback | GPT-4o-mini / Claude Haiku / Gemini Flash | API | For the hard tail; stubbable |

If your dataset is small you can start the intent classifier from `distilbert-base-uncased` and swap
up later. Do not reach for a 7B model — the point of the project is proving you know when *not* to.

## The five milestones

Build them in order. Each one has a deliverable you can point at and an acceptance criterion that is
a number, not a feeling.

### Milestone 1 — Intent classifier

Fine-tune the encoder to route tickets into a fixed label set (e.g. `bug_report`, `how_to`,
`billing`, `feature_request`, `account`, `other`). Use `banking77` or a synthetic support set if you
have no real data. This is [Lab 3](19-lab-finetune-encoder.md) skills applied end to end: HF
`datasets` → tokenizer → `Trainer` → confusion matrix → calibration check. The transfer-learning
mechanics behind this live in [Transfer Learning: The Applied-NLP Workhorse](06-transfer-learning-tasks.md).

- **Deliverable:** a trained checkpoint, a confusion matrix, and a reliability diagram.
- **Acceptance:** macro-F1 ≥ 0.85 on a held-out split, and **calibrated** — you report expected
  calibration error and use the softmax confidence as a real signal, because the router will threshold
  on it. An uncalibrated classifier makes the router lie.

### Milestone 2 — Entity extraction

Add token-classification NER that pulls `PRODUCT`, `VERSION`, and `ORDER_ID` spans out of the ticket
text. Reuse the Milestone-1 backbone with a second head. Handle the two things that break NER in
practice: **subword label alignment** (labels attach to first subword, `-100` on the rest) and
**BIO decoding** into clean spans. Score with `seqeval` at the entity level, not the token level —
this is the exact distinction [Lab 3](19-lab-finetune-encoder.md) drills.

- **Deliverable:** an NER model plus a `extract_entities(text) -> dict` function returning typed spans.
- **Acceptance:** entity-level micro-F1 ≥ 0.80 across the three types, reported per type (order-IDs
  are easy and will inflate a micro average — show the breakdown so `VERSION` recall can't hide).

### Milestone 3 — RAG answerer with reranking

Build the doc-grounded answer path over a small product-docs corpus (20–200 chunks is plenty):
chunk → embed with MiniLM → FAISS → **rerank the top-k with the cross-encoder** → assemble context →
answer with the small instruct model. This is [Lab 5](21-lab-rag-eval.md) end to end, and the
mechanisms (hybrid search, lost-in-the-middle, why the reranker earns its latency) are in
[RAG and Agents: Grounding Models in the World](09-rag-agents.md).

- **Deliverable:** a `answer(query, entities) -> {answer, citations}` function that returns the answer
  *with the chunk IDs it used*. Citations are non-negotiable — an answer with no traceable source is
  a hallucination waiting to be reported.
- **Acceptance:** retrieval **hit@5 ≥ 0.90** on your golden queries, and every generated answer cites
  at least one retrieved chunk. Demonstrate one failure mode (inject a distractor doc) and show the
  reranker recovering it — the same before/after you built in Lab 5.

### Milestone 4 — The router

Wire the three components into a decision function. The router reads intent confidence, entity
coverage, retrieval quality, and risk flags, then picks a path:

- **classify-only** — high-confidence intent that needs no answer (e.g. a clear `feature_request`):
  tag it, extract entities, drop it in the right queue. No LLM call.
- **retrieve-and-answer** — a `how_to` or `bug_report` where retrieval hit@k clears a threshold:
  run RAG and return a cited answer.
- **escalate-to-human** — low intent confidence, retrieval miss, detected PII/anger/legal risk, or a
  low RAG faithfulness score: enrich the ticket with everything extracted and hand it off.

Keep the policy as explicit thresholds, not a learned meta-model — you want to defend every decision
in the interview, and a rules-first router with logged reasons is far more debuggable. The escalation
path is what makes this production-grade rather than a demo: knowing when the system should refuse is
the safety point from [Risks and Safety: What Can Go Wrong and Who Owns It](15-risks-and-safety.md).

- **Deliverable:** `route(ticket) -> {path, reason, payload}` with the deciding signal logged.
- **Acceptance:** on a labeled routing golden set, **escalation recall ≥ 0.95** for tickets marked
  "needs human" (missing one that should escalate is the expensive error), and no path decision is
  taken without a logged reason.

### Milestone 5 — Eval harness, budget, and model card

Turn the whole thing into something you can defend and re-run. This is where [Evaluation: The Skill That Gets You Hired](10-evaluation.md)
becomes the product.

Build a golden set (50–200 tickets with gold intent, entities, routing decision, and reference
answers) and an eval script that reports, in one table: intent macro-F1, entity-level F1 per type,
retrieval hit@k, **LLM-as-judge faithfulness** on RAG answers (rubric-scored, with the position/length
bias caveats from module 10), and routing accuracy. Wire it as a **CI regression gate**: a GitHub
Action runs the harness on every push and fails the build if any metric drops more than a set delta
below the committed baseline. That gate is the single most senior thing in the project.

Then the **cost/latency budget** with real numbers. Measure, don't guess:

| Path | Model | Hardware | Latency (p50) | Marginal cost / 1k tickets |
|---|---|---|---|---|
| classify-only | DeBERTa-v3-small ×2 heads | CPU | ~20–50 ms | ~$0 (self-hosted) |
| retrieve-and-answer | MiniLM + reranker + Qwen2.5-0.5B | T4 GPU | ~0.8–2.5 s | GPU-hours only |
| escalate | (routing only) | CPU | ~30 ms | human time |
| API fallback (hard tail) | GPT-4o-mini / Haiku | API | ~1–3 s | ~$0.15–0.60 / 1M in-tokens |

The headline you compute: if the router keeps, say, 70% of traffic on the CPU path and only 20% hits
the GPU, your blended cost is a fraction of sending every ticket to an API — with numbers you can
show. Finally, write the **model card**: intended use, training data, the metrics table, known
failure modes, and the escalation policy as a documented safety boundary.

- **Deliverable:** `eval_report.md` with the metrics table, a passing CI workflow, the budget table
  above filled with *your* measurements, and `MODEL_CARD.md`.
- **Acceptance:** the CI gate is green, the report regenerates from one command, and every number in
  the README traces to the harness.

## Grading rubric

Score yourself honestly against this before you call it done. "Strong" is a hireable project;
"exceptional" is a talk-worthy one.

| Dimension | Junior | Strong | Exceptional |
|---|---|---|---|
| **Modeling** | Classifier + RAG work in a notebook | All four components hit their acceptance metrics, calibrated | + ablations (backbone size, rerank on/off, k sweep) with numbers |
| **Routing** | If/else with hardcoded paths | Thresholded policy with logged reasons, escalation recall ≥ 0.95 | + cost-aware routing that shifts the CPU/GPU/API mix under a budget |
| **Evaluation** | Accuracy printed once | Golden set + entity F1 + hit@k + judged faithfulness | + CI regression gate, contamination check, judge-bias controls |
| **Cost/latency** | "It's fast" | Measured budget table, blended cost computed | + a documented tradeoff (e.g. quantize the LLM, batch the encoder) with the before/after |
| **Engineering** | One long notebook | Modular code, seeds set, README reproduces results | + tests, a service endpoint, containerized, one-command eval |
| **Communication** | README exists | Metrics table + architecture diagram + honest failure modes | + model card, demo video, and the tradeoffs written up as decisions |

## Deliverables

1. **A public repo** — modular code (not one notebook), pinned deps, seeds set to 42 across `random`,
   `numpy`, and `torch`, and a README that reproduces every number with one command.
2. **A README with a metrics table** — the eval harness output front and center: intent F1, entity F1
   per type, hit@k, judged faithfulness, routing accuracy, and the cost/latency budget table.
3. **A 3-minute demo video** — walk one ticket through each of the three paths, then show the eval
   report and the CI gate failing on a deliberately-broken commit. Screen recording is fine; the point
   is proving it runs.
4. **An eval report** (`eval_report.md`) — the full metrics table, the distractor-injection failure
   demo, and a short "what I'd fix with more time" section. Honesty about limits reads as senior.

## How to talk about it in interviews

You will be asked to defend design decisions, not recite features. Prepare these.

**"Why did you fine-tune a 140M encoder instead of just prompting an LLM for intent?"**
Cost, latency, and calibration. On the classify-only path I serve two heads off one CPU-hosted
backbone at ~20–50 ms and near-zero marginal cost, versus ~1–3 s and per-token billing for an API
call. And a fine-tuned classifier gives me a *calibrated* confidence I can threshold the router on —
an LLM's stated confidence is not reliable in the same way. When the input distribution is stable and
the label set is fixed, the small model wins on every axis that matters. (See
[Transfer Learning: The Applied-NLP Workhorse](06-transfer-learning-tasks.md).)

**"How do you know the RAG answers aren't hallucinated?"**
I separate the two failure modes. Retrieval quality is measured with hit@k on a golden set — if hit@k
is low, no prompt tweak helps and I fix retrieval first. Given good retrieval, I score generation with
an LLM-as-judge rubric for faithfulness, controlling for the position and length biases that skew
those judges. Every answer also carries citations to the chunks it used, so a wrong answer is
traceable. And if faithfulness drops below threshold at request time, the router escalates instead of
answering.

**"Walk me through your routing logic. Why rules and not a learned router?"**
Four signals: intent confidence, entity coverage, retrieval hit@k, and risk flags. I keep it as
explicit thresholds with a logged reason per decision because I need to defend and debug every route,
and because a learned meta-model adds a training/eval surface without buying much on this traffic. The
escalation path is tuned for recall — missing a ticket that needed a human is the expensive error, so
I accept some over-escalation to keep escalation recall ≥ 0.95.

**"What's your cost story at scale?"**
The design keeps cheap deterministic models in front and gates the only expensive path. In my
measured mix, ~70% of tickets resolve on the CPU classify-only path, ~20% hit the GPU RAG path, and
the hard tail can fall back to an API. Blended cost is a fraction of routing everything to a frontier
model, and I can show the budget table. If cost spiked I'd quantize the generator and batch the
encoder — I have the before/after.

**"What would break this in production, and how would you catch it?"**
Distribution shift — new product lines the classifier never saw — shows up as a drop in max softmax
confidence, which the router already thresholds, so those tickets escalate rather than get
misrouted. Stale docs break RAG faithfulness, caught by the judged eval on the golden set. And the CI
regression gate stops a well-meaning change to chunking or prompts from silently degrading hit@k. The
honest gap is my golden set is small; the first production move is to grow it from real escalations.

## Three alternative capstone tracks

Same architecture, different domain — pick one of these if the support system doesn't fit your target
role.

**Multilingual variant.** Take the same ticket system but serve a multilingual user base. Swap the
encoder for a multilingual backbone (mDeBERTa or XLM-R small), and confront the tokenizer fertility
and low-resource penalty from [Tokenization: Turning Text into Model Inputs](03-tokenization.md) head-on: measure per-language
F1 and show the gap. Add language ID as a routing signal and a cross-lingual retrieval eval (query in
one language, docs in another). The interview story is fairness across languages and where the cheap
model's coverage runs out — a hot topic for any global product.

**Document-AI variant.** Replace free-text tickets with scanned documents — invoices, forms,
contracts. The classifier becomes a document-type router, NER becomes key-value extraction over OCR'd
text (with layout as a feature), and RAG grounds answers in the document itself. This leans on the
document-AI material in [Multimodality: When the Model Also Sees and Hears](14-multimodality.md); the killer differentiator is
OCR-free layout understanding and reporting extraction F1 against a field-level gold set. This is the
single most in-demand applied variant in 2026 — enterprise document processing is where the budgets
are.

**Reasoning-agent variant.** Turn the router into a multi-step agent for tickets that need tool use
(look up an order status, check a system, compute a refund). Build a ReAct loop with the small
instruct model, add tools behind a schema, and confront agent failure compounding — per-step error
rates multiplying over a chain — from [RAG and Agents: Grounding Models in the World](09-rag-agents.md). Layer in self-consistency
or a verifier prompt for the decision steps using the test-time-compute techniques from
[Reasoning Models: CoT, Verifiers, and RL with Verifiable Rewards](11-reasoning.md) and [Lab 6](22-lab-reasoning-decoding.md). The story here is
reliability engineering: how you cap the loop, checkpoint, and decide when a small reasoning model
earns its extra tokens.

Whichever you build, the through-line is the same: assemble the pieces, put a number on every claim,
gate regressions, and be ready to defend each tradeoff. That is the job. For the design-round framing
that mirrors these systems, revisit [Interview Bank: Applied NLP System Design](25-interview-applied-design.md);
for the production stack that ships them, [The Modern NLP Stack: Tools, Models, and a 90-Day Plan](16-modern-stack.md).
