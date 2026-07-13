# 25 — Interview Bank: Applied NLP System Design

The applied design round is where NLP hiring decisions actually get made. The breadth round
([concepts](23-interview-concepts.md)) checks that you know things; the coding round
([implementation](24-interview-implementation.md)) checks that you can build things; the design
round checks whether you can be *trusted with a problem* — a vague business ask, real constraints,
no correct answer in the back of the book. It's also where the level decision happens: the gap
between a mid-level and a senior offer is usually one interviewer's note reading "drove the
problem" versus "needed steering." This module gives you the answer framework, six fully worked
cases with the follow-ups interviewers actually ask, and a junior-vs-senior contrast for each. The
numbers in the walkthroughs are the point — memorize their shape, re-derive them on the day.

## The framework: six moves, in order

Every applied NLP design answer is the same six moves. Interviewers grade against this structure
whether they name it or not.

1. **Clarify.** Before proposing anything: volume, latency budget, cost budget, failure cost
   (what happens when the system is wrong?), languages, privacy constraints, and what "good" means
   to the business. Two minutes of questions. Skipping this is the most common down-level signal.
2. **Data.** What exists today (logs, labels, historical decisions)? What can you get labeled, at
   what cost and quality? What's the label taxonomy and who owns it? Data reality determines model
   choice, not the other way around.
3. **Model choice.** Present a ladder, not a single pick: prompted API LLM → RAG → fine-tuned
   small model → hybrid. Name concrete models and why each rung fits the constraints from step 1 —
   the decision framework from [prompting and PEFT](08-prompting-peft.md) is exactly what's tested.
4. **Eval.** Define the metric before the architecture, including the offline golden set, the
   regression gate, and the online metric that tells you the business impact. This is the step that
   separates levels most reliably — see [evaluation](10-evaluation.md).
5. **Serving and cost.** Latency anatomy (TTFT vs throughput), batching, caching, per-request cost,
   what breaks at 10× traffic. Put a dollar figure and a millisecond figure on your design.
6. **Iteration loop.** How errors get discovered, triaged, and fed back: monitoring, drift
   detection, human-in-the-loop escalation, retraining cadence. A design without a loop is a demo.

**How you're scored.** Interviewers rate four axes: *problem shaping* (did you find the real
constraints?), *technical depth* (mechanistic choices or vibes?), *judgment* (did you name what
you're giving up?), and *ownership* (eval, cost, and failure handling without prompting?). Juniors
score on depth alone; seniors score on all four — and the fastest route to ownership points is
raising eval and cost *before* being asked. Timing in a 45-minute round: ~5 minutes clarifying,
~25 on the six moves, ~15 for follow-ups — interviewers save their best questions for that phase,
and a 40-minute monologue robs them of the evidence they need to advocate for you.

---

## Case 1 — Support-ticket triage at 1M tickets/month

> **Prompt:** "Our SaaS company gets one million support tickets a month across email, chat, and
> web forms. Design a system that routes each ticket to the right team and flags urgent ones.
> Today it's done by a human triage team with a 4-hour median delay."

**Clarify.** 1M/month ≈ 23 tickets/minute average, but support traffic is bursty — assume 5× peaks,
~120/minute. How many routing destinations? (Typical: 30–80 teams/queues.) What's the cost of a
misroute — a bounce and re-queue (cheap) or an SLA breach (expensive)? Is "urgent" defined by
policy or learned from past escalations? Languages? Assume English-dominant with a 15% long tail.

**Data.** This is the happy case: the human triage team has been generating labels for years. Pull
12–24 months of (ticket text → final resolving team) pairs — millions of examples, free. Caveat to
raise unprompted: the *final* team isn't always the *correct first* route (tickets bounce), so
filter to tickets resolved by their first assignment, and expect ~5–10% label noise. For urgency,
use historical escalation flags but audit a sample — urgency labels are usually inconsistent.

**Model choice.** Closed-set classification, high volume, tight latency, abundant labels — the
textbook case where a fine-tuned encoder beats an API LLM (the argument from
[transfer learning](06-transfer-learning-tasks.md)). Fine-tune **DeBERTa-v3-base (~184M)** on
100k–500k labeled tickets: one head for team (softmax over ~50 classes), one for urgency (binary).
Expect macro-F1 in the high 80s on head classes; p95 latency under 30 ms on 1–2 GPU replicas or
even CPU, well under $500/month all-in. The LLM comparison: a cheap API model at ~$0.15/1M input
tokens runs $100–300/month here — affordable, but 500–1500 ms latency, no calibrated confidence,
a per-call dependency. Where the LLM earns its keep is the low-confidence tail: route classifier
outputs below a tuned threshold (typically 10–15% of traffic) to an LLM with the team directory
in context, or to the human queue. That hybrid is the senior-shaped answer.

**Eval.** Offline: stratified test set frozen from the last 3 months, macro-F1 (not accuracy —
class imbalance is severe; the top 5 intents are often 60% of volume), per-class confusion matrix,
and **calibration** (ECE), because the confidence threshold is a product feature. For urgency,
recall matters more than precision — set the operating point at ~95% recall and eat the false
positives. Online: misroute rate (measured by ticket bounces), time-to-first-correct-team, and
SLA-breach count, A/B'd against the human baseline.

**Serving/cost.** Batch-friendly: micro-batch requests at 32, single A10-class GPU handles
thousands/minute. The whole system is <$1k/month; quote that against the triage team's cost to
frame the ROI, because interviewers at product companies notice when you do.

**Iteration.** Weekly drift check on the input distribution (new product launches create new
intent mass); monthly retrain on fresh labels; the human-escalation queue *is* the labeling
pipeline for the tail. A new team on the org chart is a new class — keep the taxonomy owned by a person.

**Follow-ups interviewers ask:**

- "A new product launches and 20% of tickets are about it. What breaks and how do you catch it?"
  (Confidence distribution shifts down; drift alarm on softmax entropy; the LLM/human fallback
  absorbs it while you collect labels.)
- "Why not just prompt GPT-4o-class with the list of 50 teams?" (Right *bootstrap*, wrong end
  state — latency, calibration, cost-at-scale, and drift control favor the encoder once you have
  labels. "Start with the LLM, distill to the encoder" is the best answer.)
- "How do you handle a ticket in Turkish?" (Language ID → multilingual encoder variant (mDeBERTa /
  XLM-R) or translate-then-classify; check per-language F1 slices before shipping.)

**Junior vs senior.** The junior answer jumps to "fine-tune BERT" or "call an LLM" in the first
minute and describes the training loop. The senior answer establishes misroute cost and label
source first, proposes the encoder-plus-LLM-fallback hybrid with the threshold as an explicit
product knob, and quotes macro-F1, p95 latency, and monthly cost unprompted. The tell: seniors
treat the triage team as a *component* (labeling + fallback), not as the thing being deleted.

---

## Case 2 — RAG assistant over enterprise legal documents

> **Prompt:** "A 2,000-lawyer firm wants an internal assistant that answers questions over their
> contract repository — about 3 million documents. Design it."

**Clarify.** The question that changes everything: **what is the cost of a wrong answer?** In
legal, a confident hallucination is a professional-liability event — so citations are mandatory,
abstention beats guessing, and the product is a *research accelerant*, not an oracle. Second:
access control — contracts are matter-privileged; users must never retrieve documents they can't
open. Third: query types — clause lookup ("termination notice in the Acme MSA?") vs. cross-document
synthesis ("what indemnity caps do we typically agree to?"). Latency is generous: 5–10 s is fine.

**Data.** 3M documents × ~20 pages ≈ tens of GB of text. Contracts are highly structured — exploit
it. Chunk **clause-aware**: split on section headings, 300–800 tokens per chunk, and prepend the
document title + section path to every chunk (the single highest-ROI trick in enterprise RAG,
because "Section 8.2" retrieved without its contract name is useless). Expect ~100–200M chunks
worth of index.

**Model choice / pipeline.** The standard pipeline from [RAG and agents](09-rag-agents.md), every
stage justified. **Hybrid retrieval** (BM25 + a strong bge/gte-class or API embedding model) —
legal queries mix exact terms ("Section 409A", defined terms) where sparse wins with paraphrases
where dense wins; fuse with RRF, retrieve 50, **cross-encoder rerank** to 8–10. ACL filtering
happens *at retrieval time* as a hard metadata filter — never as a post-hoc check on generated
text. Generator: a frontier API model (Claude / GPT-4o class) if the firm's cloud agreement allows,
else a self-hosted 70B-class open model (Llama/Qwen) in their VPC — in legal, the deployment
boundary is often the real model-selection criterion, and saying so is a strong signal. Prompt
contract: answer only from provided context, cite chunk IDs inline, say "not found in the provided
documents" when support is absent. Render citations as one-click links to the source clause —
verifiability is what makes hallucination survivable as a product.

**Eval.** Build the golden set *with the lawyers*: 300–500 real questions with attorney-verified
answers and supporting clause IDs. Three layers: retrieval **hit@10** against gold clauses (target
>90% — if retrieval misses, nothing downstream saves you); **faithfulness** via LLM-as-judge (is
every claim supported by a cited chunk?) with a 50-sample human audit to validate the judge; and
**abstention correctness** on held-out unanswerable questions (the system must say "not found,"
not improvise). Online: citation click-through, lawyer thumbs, escalation-to-manual-search rate.

**Serving/cost.** Embedding 3M docs once: on the order of a few hundred dollars of GPU time or API
spend. Per-query: retrieval ~100 ms, rerank ~200 ms, generation 3–6 s; cost dominated by generation
at ~$0.01–0.05/query with a frontier model — trivial against a lawyer's billing rate, so don't
over-optimize cost here; optimize *trust*.

**Iteration.** Every thumbs-down gets triaged into retrieval-miss vs. generation-error vs.
bad-question — the fix differs completely per bucket. Index refresh pipeline for new contracts
(daily). Quarterly re-run of the golden set as a regression gate before any model or prompt change.

**Follow-ups:**

- "The model answers correctly but cites the wrong clause. Which layer failed and how do you
  detect it?" (Attribution failure — faithfulness judge with citation-level checking; often a
  reranker problem where a near-duplicate clause from another contract outranked the right one.)
- "How do you handle a 200-page contract that exceeds retrieval granularity — 'summarize the whole
  MSA'?" (Different query class: route to a long-context path — modern 128k+ context fits it —
  or hierarchical map-reduce summarization; detect the intent up front with a cheap classifier.)
- "Lost-in-the-middle: does chunk order matter?" (Yes — put the highest-reranked chunks first and
  last; better, keep k small: reranked top-8 usually beats stuffed top-50.)

**Junior vs senior.** The junior answer is the pipeline diagram — embed, vector DB, retrieve,
generate — interchangeable with a blog post. The senior answer leads with the cost of a wrong
answer and lets it drive everything (citations, abstention, ACLs at retrieval, expert-built golden
set, per-stage error triage), and says what it *won't* build: no agentic multi-hop in v1 — you
can't debug a compound system before the simple one has a measured baseline.

---

## Case 3 — Multilingual content moderation

> **Prompt:** "A social platform with 50M DAU needs to moderate user posts in 30+ languages for
> hate speech, harassment, and violent threats. Posts must be actioned within seconds. Design it."

**Clarify.** 50M DAU might produce 100M+ posts/day ≈ 1–2k/second average, higher at peak — a
*throughput* problem before it's a modeling problem. Policy: who owns the taxonomy, and does a
flag block, downrank, or queue for review? (Almost always: high confidence → auto-action, mid →
human queue, low → allow; design for that triage from the start.) Failure costs are asymmetric
*per class*: missed violent threats are catastrophic (optimize recall); over-removal of borderline
speech is a press cycle and a fairness problem (watch precision, per language).

**Data.** The hard part. Labeled moderation data is scarce outside English, labels are noisy
(annotator disagreement on harassment runs 20–30%), and the adversary adapts. Sources: existing
human-moderation decisions (biased toward what old filters caught — say this out loud), targeted
labeling in the top 10 languages, **translate-train** (machine-translate English labeled data),
and **zero-shot cross-lingual transfer** for the tail — multilingual encoders transfer surprisingly
well, but expect a 5–15 point F1 drop on unseen languages, which you measure, not assume.

**Model choice.** A cascade, because unit economics at 1–2k QPS rule out an LLM on every post.
**Stage 1:** a fine-tuned multilingual encoder — **XLM-R-large or mDeBERTa-v3 (~300–550M)** —
multi-label heads over the policy taxonomy, quantized, running at <20 ms/post; it clears the ~95%
of posts that are obviously fine. **Stage 2:** posts in the uncertain band (say 3–5% of traffic)
go to a stronger model — an 8B-class multilingual instruct model (Qwen3 / Llama family) prompted
with the policy text, which handles context, sarcasm, and code-switching far better. **Stage 3:**
human review for auto-action candidates in sensitive classes and appeals. Also raise tokenizer
fertility ([tokenization](03-tokenization.md)): the same post costs 2–3× the tokens in Burmese as
in English, which hits both cost and effective context.

**Eval.** Per-language, per-class slices are the whole game — a global F1 of 0.88 can hide 0.60
in Swahili, a safety hole and a fairness finding ([risks and safety](15-risks-and-safety.md)) at
once. Fixed multilingual golden set with adversarial examples (leetspeak, homoglyphs, code-switched
abuse, quoted-speech false positives). Report per-class precision/recall at the *deployed
thresholds*, not AUC. Online: violating-content prevalence via random-sample human audit (the
unbiased estimator), appeal overturn rate (precision proxy), review-queue volume (cost proxy).

**Serving/cost.** Stage 1 at 2k QPS: a handful of GPU replicas with dynamic batching —
low-single-digit thousands per month. Stage 2 at ~60–100 QPS with a self-hosted 8B on vLLM: a few
more GPUs. The cascade is what makes the economics work; quote the blended per-post cost (fractions
of a hundredth of a cent) and contrast with all-LLM (~10–100× more).

**Iteration.** Adversarial drift is constant: monitor per-language flag-rate shifts, feed appeal
overturns and audit misses back as training data, retrain stage 1 monthly. Red-team new evasion
patterns proactively.

**Follow-ups:**

- "Recall or precision for violent threats — pick and defend." (Recall, operating point ~95%+,
  with the human queue absorbing the precision cost; but say that the queue has finite capacity,
  so the threshold is really set by review headcount — that's the systems answer.)
- "A language community complains of over-removal. Diagnose." (Pull the per-language slice; check
  training-data provenance — translate-train artifacts and cultural false positives (reclaimed
  slurs, quoted speech) are the usual suspects; fix with in-language data, not a threshold hack.)
- "Why not one big LLM with the policy in the prompt?" (Cost and latency at 2k QPS; but also
  consistency — policy-prompted LLMs drift with paraphrase; a trained classifier is auditable at
  a fixed operating point. Use the LLM where its judgment pays: the uncertain band.)

**Junior vs senior.** The junior answer is a single model and a threshold. The senior answer is a
cascade with per-class asymmetric operating points, per-language eval slices, human review as a
designed capacity-constrained component, and the observation that the labeling pipeline and appeal
loop — not the model — are where moderation systems are won.

---

## Case 4 — Information extraction from scanned invoices

> **Prompt:** "An accounts-payable company processes 200k scanned invoices a month from thousands
> of vendors. Extract vendor, dates, line items, and totals into a database with high accuracy.
> Design the pipeline."

**Clarify.** "High accuracy" must become a number per field: totals and bank details need ~99%+
(money moves on them); line-item descriptions tolerate 95%. What's the human fallback — a review
UI? (There must be; design *for* it.) Input quality: phone photos or clean PDFs? Layout variance
across thousands of vendors is the real difficulty — a layout problem, not a text problem.

**Data.** Gold: historical invoices paired with the AP-entered database records — hundreds of
thousands of (image → structured record) pairs for free, with ~1–3% entry-error noise. Align them
(fuzzy match on totals/dates) to build training and eval sets. Stratify by vendor: the eval set
must include *unseen vendors*, because per-vendor memorization inflates naive metrics.

**Model choice.** Two viable 2026 architectures — present both ([multimodality](14-multimodality.md)).
**(A) OCR + layout-aware encoder** (LayoutLM lineage): OCR → token classification with 2D position
features. Mature, cheap, fast (<1 s/page), but brittle to OCR errors and needs BIO labels.
**(B) OCR-free document VLM** (Donut lineage; today a Qwen2.5-VL-class 3–7B, optionally
LoRA-fine-tuned): image in, JSON out with **constrained decoding against the target schema**
([inference](12-inference-decoding.md)) so output is parseable by construction. B handles layout
variance and unseen vendors better and is where the field has moved; A still wins on cost at
extreme volume. Recommend **B fine-tuned on 50–100k aligned pairs** plus **deterministic
validators**: line items sum to subtotal, subtotal + tax = total, dates parse, vendor in the
master list. Validator failures and low-confidence fields route to the review UI. The validators
carry the 99% requirement — say so explicitly; that's the gap between model and *system* accuracy.

**Eval.** Field-level metrics, not document-level vibes: exact-match for amounts/dates (normalized),
per-field precision/recall, line-item F1 with fuzzy matching on descriptions. The headline product
metric is **touchless rate** (fraction of invoices needing zero human edits — typically 60–85% is
a strong result) and **error escape rate** (wrong data that reached the database — drive toward
~0 via validators + review). Report on the unseen-vendor slice separately.

**Serving/cost.** 200k/month ≈ 7k/day — batch-friendly; latency budget is minutes. A 7B VLM on
one A100-class GPU with vLLM does 1–3 pages/second batched — a fraction of one GPU. Cost: under
$0.005/invoice self-hosted vs. $0.01–0.05 via frontier API; both viable at this volume, so pick on
privacy (bank details) and fine-tunability. Human review at 20% touch × 200k = 40k reviews/month —
quote that: it dwarfs compute cost, and shrinking it is where the model-quality ROI lives.

**Iteration.** Every human correction is a labeled example — feed it back, retrain monthly. Monitor
per-vendor touchless rate; a new large vendor with a weird layout shows up as a cluster of
failures, and few-shot examples or targeted fine-tuning fix it within a week.

**Follow-ups:**

- "The model reads €1.234,56 as $1,234.56. How do you catch and fix it?" (Validators catch the
  arithmetic inconsistency; fix via normalization rules keyed on vendor locale + training data
  coverage; never trust the model alone on currency.)
- "Handwritten invoices?" (Slice the eval; VLMs degrade gracefully but confidence drops — route
  low-confidence to review; if volume justifies, add handwritten samples to fine-tuning.)
- "Why constrained decoding instead of asking for JSON in the prompt?" (Asking gets you ~95–99%
  parseable; constraining gets 100% schema-valid by masking invalid tokens at decode time — at 200k
  docs/month, a 1% parse-failure rate is 2,000 broken records.)

**Junior vs senior.** The junior answer is "run OCR, then an LLM to extract JSON." The senior
answer is a *system*: aligned historical data as free labels, unseen-vendor eval slice, schema-
constrained decoding, deterministic validators carrying the accuracy SLA, a review UI whose
corrections close the loop, and touchless rate — a business metric — as the north star.

---

## Case 5 — Eval strategy for a chatbot launch

> **Prompt:** "We're launching a customer-facing support chatbot next quarter, replacing part of a
> human support tier. You own quality. Design the evaluation strategy, offline and online."

**Clarify.** Does the bot answer from a knowledge base (RAG), take actions (refunds, account
changes), or both? Actions raise the stakes enormously. Business goal — deflection (tickets that
never reach a human) or CSAT? Risk appetite, rollback path? This case is a pure test of
[evaluation](10-evaluation.md) thinking, and the framing that wins: **offline evals gate the
launch; online evals measure the truth; the two must be connected.**

**Offline.** Four layers before any user sees it:

1. **Golden set:** 300–1,000 real historical conversations, stratified by intent frequency and
   difficulty, with reference answers written or verified by senior support agents. Frozen —
   additions fine, edits invalidate history.
2. **Automated scoring:** LLM-as-judge with a per-dimension *rubric* — correctness vs. reference,
   groundedness in the KB, tone, resolution — pairwise against the production baseline. Control
   the known biases (randomize position, cap length reward, judge from a different model family
   than the bot); validate against ~100 human-labeled examples, target ≥85% agreement first.
3. **Safety/adversarial suite:** prompt-injection attempts, requests for policy exceptions
   ("promise me a refund"), competitor questions, PII probes, jailbreaks — pass/fail, and the gate
   is ~100% on the hard-fail subset.
4. **Regression gate in CI:** every prompt, model, or KB change re-runs the golden set; a
   statistically meaningful drop (pre-register the threshold, e.g. −2 points on judged win rate)
   blocks the deploy. This turns eval from a report into a brake.

**Online.** Launch as a canary: 1% → 5% → 25% → 100%, gated on metrics at each step. Primary:
**containment/deflection rate** (conversations resolved without human handoff) and **CSAT on
bot-handled conversations**. Guardrails: escalation rate, user rephrase/repeat rate (a cheap
frustration proxy), thumbs-down rate, and — the one people forget — **downstream ticket reopen
rate**, because a bot that "resolves" tickets that come back in 48 hours is manufacturing
deflection. A/B against the human tier where routing allows, pre/post with a holdout otherwise.
Sample 1–2% of live conversations for weekly human review on the judge's rubric — that keeps the
offline judge honest against real traffic drift.

**The connecting loop.** Every online failure (thumbs-down, escalation, reopen) gets mined weekly;
representative failures are added to the golden set (grow it, never rebalance it); fixes are
verified offline before redeploying. Quote a cadence: weekly triage, biweekly eval-set additions,
monthly judge re-validation.

**Follow-ups:**

- "Your judge says the new prompt is +5 points but thumbs-down went up. Which do you trust?"
  (Online — but first check for confounds: traffic mix shift, novelty effects, judge blind spots.
  Then close the gap: mine the thumbs-down conversations, check whether the judge scores them
  correctly; usually the judge is missing a failure mode — add it to the rubric.)
- "How big does the golden set need to be?" (Big enough that your decision threshold clears noise:
  for a ±2-point gate on a pairwise win rate you need roughly 500–1,000 comparisons; do the
  binomial back-of-envelope in the room.)
- "The bot can issue refunds. What changes?" (Everything: action-level evals with exact-match on
  tool calls, a dry-run/sandbox eval mode, hard policy checks *outside* the model, tighter canary,
  and per-action human approval until precision is proven.)

**Junior vs senior.** The junior answer lists metrics: "BLEU… I mean, LLM-as-judge, and thumbs
up/down." The senior answer is an *apparatus*: frozen golden set with a CI gate, a validated judge
with named bias controls, a staged canary with pre-registered guardrails, and a weekly loop turning
production failures into offline test cases — plus the political reality that deflection can be
gamed by making escalation hard, so it's paired with reopen rate to keep it honest.

---

## Case 6 — "Cut our LLM inference bill 10× without hurting quality"

> **Prompt:** "We spend $400k/month on LLM API calls across our product. Leadership wants it
> under $50k without a measurable quality drop. Go."

**Clarify.** First move: **you cannot optimize an unmeasured bill.** Ask for (or build, week one)
a cost dashboard by feature, model, and token type — input vs. output vs. cached. Typical finding:
3–5 features drive 80% of spend, input tokens outnumber output 5–20× (bloated prompts), and many
calls are near-duplicates. Then make "without hurting quality" operational: a golden set per
feature and a pre-registered non-inferiority margin ([evaluation](10-evaluation.md)) that every
optimization below ships through. Saying this before touching the model is the senior signal.

**The ladder, cheapest lever first.** Multiplicative, so 10× is realistic as a product of modest
wins:

1. **Prompt and context hygiene (1.5–3×, days).** Audit token histograms. Trim boilerplate system
   prompts, deduplicate retrieved context, cap few-shot examples, shorten outputs with format
   instructions (output tokens cost 3–5× input on most APIs). Teams routinely find 60k-token
   prompts doing 5k tokens of work.
2. **Prompt caching (1.3–2× on top, days).** Providers discount cached prefix tokens by ~90%.
   Restructure prompts so the static part (system prompt, tool schemas, common context) is a
   stable prefix and per-request content comes last. For high-QPS features with shared prefixes
   this alone can halve the bill.
3. **Model right-sizing and cascades (2–4×, weeks).** Most calls don't need a frontier model.
   Per-feature, eval a mid-tier model (GPT-4o-mini/Haiku-class, or a hosted Qwen3/Llama endpoint) —
   frontier-to-mini pricing gaps are 10–30×. Where quality drops, cascade: small model answers,
   a cheap confidence check (self-consistency, logprobs, or a verifier prompt) routes the hard
   10–20% to the frontier model. Blended cost approaches the small model's.
4. **Distillation / fine-tuning for the head workloads (2–5× on the biggest features, weeks).**
   For the top spend feature, generate training data from the frontier model's own historical
   outputs and LoRA-fine-tune an 8B-class open model ([prompting and PEFT](08-prompting-peft.md)).
   A tuned 8B matches a prompted frontier model on a *narrow* task far more often than not — that's
   the entire economic point of fine-tuning.
5. **Self-hosting the distilled models (further 2–3× at sustained volume).** vLLM/SGLang with
   continuous batching; an 8B on one H100-class card serves thousands of tokens/second
   ([inference](12-inference-decoding.md)). Quantize (AWQ/INT8) with a parity check on the golden
   set. Only worth it above roughly $10–20k/month per workload — below that, engineer time costs
   more than the API, and knowing when *not* to self-host is judgment.
6. **Semantic caching and batch tiers (opportunistic).** Cache near-duplicate requests
   (embedding-keyed, aggressive TTL, correctness check for anything user-specific); move
   non-interactive workloads to provider batch APIs at ~50% discount or off-peak queues.

A realistic composition: 2× (hygiene) × 1.5× (caching) × 2× (cascades) × 1.7× (distill top
feature) ≈ **10×**, with each step gated on the per-feature golden set and a canary. Timeline:
steps 1–2 in the first two weeks (visible wins buy trust), 3–4 over a quarter.

**Follow-ups:**

- "Quality dropped on one feature after the cascade. Debug it." (Pull routed traffic: router
  miscalibrated, or small model failing on a slice? Fix the threshold or add the slice to
  fine-tuning data; and if the golden set didn't catch it, the set has a coverage gap — fix that too.)
- "Why not just renegotiate the API contract?" (Do that too — committed-use discounts are real
  20–40% levers and cost zero engineering. Naming the non-technical lever is a senior move.)
- "What breaks at 10× traffic growth?" (Self-hosted capacity planning and the review/eval loop;
  per-token API costs scale linearly but self-hosted has step functions — model the crossover.)

**Junior vs senior.** The junior answer is a single technique — "quantize it" — applied globally.
The senior answer starts with measurement, orders levers by effort-to-savings, gates every change
on a per-feature golden set, and composes modest multiplicative wins — including the unglamorous
levers (prompt bloat, caching, contract negotiation) that deliver most of the money.

---

## Presenting the capstone in behavioral and project rounds

The design round tests hypotheticals; the project round ("tell me about something you built")
tests whether you've *actually lived* the loop. The [capstone](26-capstone.md) is built to be that
story. Present it STAR-shaped and metrics-first:

- **Situation** — one sentence, business framing: "Support teams answer repetitive product
  questions; I built the system that triages tickets and drafts grounded answers." Never open
  with the architecture.
- **Task** — the success criteria you set *before building*: "targets: ≥85% intent macro-F1,
  ≥90% retrieval hit@5, ≥90% judged faithfulness, under $0.01 and 2 s p95 per query." Stating
  pre-committed numbers is itself the signal — it shows eval-driven development, the hiring bar.
- **Action** — decisions and their *rejected alternatives*, not the component list: "I chose a
  fine-tuned DeBERTa for intent over prompting an LLM — ~40× cheaper at 25 ms p95 at projected
  volume; here's the comparison table." One genuine failure beats five successes: "my first
  chunking broke tables apart, hit@5 was 71%; heading-aware chunking took it to 93%." Interviewers
  trust candidates who volunteer what went wrong.
- **Result** — final metrics against the pre-stated targets, the regression gate that keeps them
  honest, and what you'd do next with real users ("my judge is unvalidated against real user
  satisfaction — that's the first thing production data would fix").

Practice three cuts: 90 seconds (screen/behavioral), 5 minutes (deep-dive opening), 20 minutes
with a whiteboard. In the deep-dive, expect the same follow-ups as the cases above — "why this
model," "how do you know the judge is right," "what breaks at 10× scale" — and answer them the
same way: mechanism, tradeoff, number. The strongest position in any interview is describing a
system where you already *measured* the thing being asked about. Build the capstone so that's
true, and the design round becomes a conversation about your results rather than a test.
