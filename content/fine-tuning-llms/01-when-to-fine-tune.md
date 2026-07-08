# 01 — When to Fine-Tune (and When Not To)

Fine-tuning is the most over-reached-for tool in the applied LLM toolbox. The instinct — "our
model isn't good enough, let's train it on our data" — is right about the symptom and usually wrong
about the cure. Before you spend a week and a GPU budget adapting weights, you need a decision
framework that tells you whether fine-tuning even addresses the failure you're seeing. This lesson
is that framework. Every later lesson assumes you got here first and concluded, correctly, that
fine-tuning is the right move.

## The three levers, in the order you pull them

You have three ways to change what an LLM does, and they escalate in cost:

1. **Prompt engineering** — change the input. System prompt, few-shot examples, output-format
   instructions, decomposition into steps. Cost: hours. Iteration loop: seconds. No training, no
   serving change.
2. **Retrieval-augmented generation (RAG)** — change the *context* the model sees at inference by
   fetching relevant documents. Cost: days to weeks to build a real pipeline (chunking, embedding,
   a vector store, reranking). Recurring cost: embedding + storage + retrieval latency.
3. **Fine-tuning** — change the *weights*. SFT, then optionally preference tuning. Cost: weeks
   including data curation and evaluation, plus a recurring serving cost (you now host a model or an
   adapter) and a maintenance cost (every base-model upgrade means retraining).

The 2026 consensus among practitioners is blunt: **most teams asking about fine-tuning should not
fine-tune yet.** They should fix their prompts, build a real RAG pipeline, and write evals — in that
order — and only then ask whether a residual failure is weight-shaped. The canonical escalation is
**Prompt → RAG → Fine-tune → Distill.**

## The diagnostic question: is your gap knowledge or behavior?

The single most useful move is to name what's actually failing.

- **The model lacks facts** — company-specific data, recent events, proprietary documents, anything
  that changes weekly. This is a *knowledge* gap. Fine-tuning is the **wrong** tool. Weights are a
  lossy, expensive, un-updatable place to store facts; the moment a document changes you'd have to
  retrain. Use RAG. A retrieved, cited passage beats a memorized, hallucinated one, and you can swap
  the document tomorrow with no training.
- **The model knows enough but behaves wrong** — inconsistent format, wrong tone, ignores your
  structured-output schema, rambles when it should be terse, doesn't follow a fixed policy, or is
  slow because you're burning thousands of few-shot tokens on every call. This is a *behavior* gap.
  This is what fine-tuning is *for*.

The slogan worth memorizing: **fine-tune for form, not for facts.** You use it to shape behavior,
style, structure, and refusal patterns — not to inject knowledge that changes.

## What fine-tuning is genuinely good at

- **Locking in an output format.** If you need every response to be valid JSON matching a schema, or
  to always follow a five-section structure, SFT on a few hundred correct examples makes it reliable
  in a way that prompt instructions never fully do. The model stops "usually" complying and starts
  reliably complying.
- **Domain style and vocabulary.** Legal, medical, a specific brand voice, a terse internal-tool
  register — behavior that's tedious to specify in a prompt but easy to demonstrate.
- **Amortizing a long prompt.** If your prompt carries a 2,000-token instruction block or a dozen
  few-shot examples on every request, fine-tuning bakes that behavior into the weights. You pay once
  in training and save those tokens (latency *and* cost) on every inference forever. For a
  high-volume endpoint this alone can justify the project.
- **A narrow, high-accuracy task.** Structured extraction, classification, routing, a single
  well-defined transformation. A small fine-tuned model often beats a much larger prompted one on
  the narrow task, at a fraction of the inference cost.
- **Teaching a smaller model to imitate a bigger one (distillation).** Generate high-quality outputs
  from a frontier model, fine-tune a small open model on them, deploy the small one cheaply.

## What fine-tuning is bad at

- **Adding knowledge.** Covered above. It "works" in demos and fails in production because the facts
  go stale and the model hallucinates around the edges of what it memorized.
- **Fixing reasoning you don't have data for.** If your model can't do multi-step reasoning on your
  task, SFT on a few hundred examples won't install a capability the base model lacks. You need a
  stronger base model, better prompting, or (if the task has checkable answers) RLVR — not more SFT.
- **Keeping up with a moving target.** If requirements change monthly, the retrain-and-re-eval loop
  will outrun you. Prompts and retrieval indices change in minutes.
- **Small data with high variance.** Fine-tuning on 50 noisy, inconsistent examples will teach the
  inconsistency. Below a few hundred *clean* examples you're usually better off with few-shot
  prompting, which uses the same examples without the training risk.

## The cost reality

Fine-tuning's true cost is dominated not by GPU hours but by the parts nobody budgets for:

- **Data curation** is the real work — typically 60–80% of the project. A few hundred to a few
  thousand *clean, consistent* demonstrations. This is manual, iterative, and unglamorous.
- **Evaluation** — you cannot know if the fine-tune helped without a held-out eval set built *before*
  training (Lesson 06). Building that eval is itself a mini-project.
- **Serving** — you now host a model or a LoRA adapter, monitor it, and pay per-token inference on
  your own infrastructure rather than an API.
- **Maintenance** — when the base model gets a better version (and in the current cadence, it will
  within months), your fine-tune is now on a stale base and you retrain.

The good news for 2026: the *compute* cost has collapsed. A thin **LoRA/QLoRA adapter** on a strong
open base — a 7–8B model fine-tuned with 4-bit quantization — fits on a single consumer 24 GB GPU
and trains in hours, not the rented-A100-cluster of a few years ago. The expensive part was never
the GPU; it's the data and the evaluation discipline. That's exactly why the highest-ROI fine-tune
is a small adapter paired *with* retrieval, not a from-scratch full fine-tune replacing it.

## A worked decision

You're building a support assistant that answers from your product docs in your brand voice, as
JSON with `answer` and `sources` fields, and it's currently unreliable. Decompose the failures:

- Wrong or missing facts about the product → **RAG.** Index the docs; retrieve and cite.
- Wrong JSON shape, inconsistent voice → **fine-tune.** SFT a few hundred examples of correct-shape,
  correct-voice responses.
- One-off phrasing tweaks, adding a disclaimer → **prompt.** Edit the system prompt; ship in minutes.

The mature architecture is all three: a fine-tuned adapter for *form* and voice, retrieval for
*facts*, and a prompt for the last-mile instructions. Fine-tuning does not replace RAG here; it
complements it. Teams that fine-tune *instead* of building retrieval end up with a confidently-wrong
model that's expensive to correct.

## Rough cost-and-time comparison

Concrete numbers help calibrate the escalation. These are order-of-magnitude, 2026-typical figures
for a moderate-volume feature — treat them as ratios, not quotes:

| Lever | Time to first result | Iteration loop | Recurring cost driver | Best at |
|---|---|---|---|---|
| Prompt engineering | Hours | Seconds | Per-token API calls | Phrasing, format nudges, few-shot behavior |
| RAG | Days–weeks | Minutes (reindex) | Embedding + vector store + retrieval latency | Fresh/proprietary facts |
| Fine-tune (LoRA/QLoRA) | Weeks (mostly data) | Hours (retrain) | Self-hosted inference + retraining on base upgrades | Consistent form, style, narrow high-accuracy tasks |

The subtle economics: fine-tuning has high *upfront* cost but can *lower* steady-state cost. If a
long prompt or a big model is on the critical path of millions of calls, a small fine-tuned model
that drops the prompt overhead and runs on cheaper hardware pays back the training investment — this
is often the real business case, not accuracy. Conversely, if volume is low, the upfront cost never
amortizes and you should stay on prompting + RAG.

## The "can prompting get there?" test

Before committing, run the cheap experiment: spend a day *seriously* prompting — a strong system
prompt, 5–10 well-chosen few-shot examples, explicit output-format instructions, maybe a two-step
decomposition. Measure it on your eval. Two outcomes, both useful: either prompting closes the gap
(you just saved a fine-tuning project), or it plateaus at a ceiling with a clear residual — and that
residual, characterized precisely, is the exact behavior your fine-tune needs to install. Few-shot
prompting is also a preview of fine-tuning: if 10 examples in the prompt help, hundreds in the
weights will help more and cost nothing at inference. If 10 examples in the prompt *don't* help, more
of them as training data probably won't either — a strong signal the gap isn't behavior-shaped.

A quick gate before you commit: can you write down (a) a held-out eval that would prove the
fine-tune helped, (b) at least ~200 clean demonstrations you can produce, and (c) a reason prompting
and RAG can't close the gap? If any of the three is "no," you're not ready to fine-tune — and that's
a finding, not a failure.

## Key takeaways

- Escalate in order: **Prompt → RAG → Fine-tune → Distill.** Each step costs 10× more effort than
  the last; don't skip to the expensive one.
- Diagnose the gap first. **Knowledge gap → RAG. Behavior gap → fine-tune.** Fine-tune for *form*,
  not for *facts*.
- Fine-tuning is strong at locking output format, domain style, amortizing long prompts into weights
  (real latency/cost savings), narrow high-accuracy tasks, and distillation.
- Fine-tuning is weak at adding knowledge, installing capabilities the base model lacks, and keeping
  up with fast-changing requirements.
- The dominant cost is **data curation and evaluation**, not GPU time. LoRA/QLoRA made the compute
  cheap; the discipline is still expensive.
- The best real systems combine all three levers. A LoRA adapter for behavior + RAG for facts beats
  either alone.

## Try it

Take one LLM feature you actually work on (or a realistic hypothetical) and write a one-page
fine-tune decision memo. List every failure mode you've seen, and for each, label it *knowledge*,
*behavior*, or *phrasing*, then assign the right lever (RAG / fine-tune / prompt). At the bottom,
answer the three gate questions: (1) What held-out eval would prove a fine-tune helped, and what's
the current baseline number on it? (2) How many clean demonstrations can you realistically produce,
and where do they come from? (3) Why can't prompting + RAG close this specific gap? If you can't
answer all three concretely, your conclusion is "not yet — fix prompts and retrieval first," and you
just saved yourself a wasted week. If you can, you've written the spec for the rest of this course.
