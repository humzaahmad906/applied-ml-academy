# 06 — Evaluation: Did It Actually Improve?

A falling loss curve tells you the model fit your training data. It does not tell you the model got
better at your task, and it certainly doesn't tell you it didn't get *worse* at everything else.
"Model improved" without a measurement is folklore. This lesson is the discipline that turns
fine-tuning from vibes into engineering: building a frozen eval before you train, picking metrics
that match the task, comparing honestly against the base model, using LLM-as-judge without fooling
yourself, and — the mistake that silently invalidates most reported gains — avoiding eval
contamination.

## Eval-driven development: build the eval first

The rule that matters most: **build and freeze your eval set before you train.** This is the ML
analog of test-driven development. If you build the eval after seeing results, you'll
(unconsciously) build one your model passes. A frozen eval, pinned with a baseline number from the
*base* model, is the only thing that makes "+8 points" a fact rather than a hope.

Your eval set should:

- Come from the **same distribution as production**, including the hard and edge cases.
- Be **held out** — never seen in training, and deduplicated against the training set (Lesson 02).
- Be **frozen**: adding samples later is fine; removing or rebalancing invalidates every prior
  number, so you can no longer compare across runs.
- Be **big enough to be stable** — a 10-example eval is noise; aim for at least a few hundred so a
  few flips don't swing your metric.

Pin the baseline first: run the *unmodified base model* on the eval and record every metric. Every
fine-tune is measured as a delta against that line.

## Pick the metric the task actually cares about

The right metric depends entirely on what you fine-tuned for:

- **Structured extraction / classification / routing** — the easy, happy case: you have ground truth.
  Use exact match, F1, field-level accuracy, or JSON-schema-valid rate. These are deterministic,
  cheap, and unarguable. If your task has checkable answers, *this is your eval* — no LLM judge
  needed.
- **Format compliance** — the fraction of outputs that parse and validate. If you fine-tuned to lock
  a JSON schema, "% valid JSON" is a first-class metric and often the whole point.
- **Code** — does it run, does it pass unit tests (pass@k). Verifiable, like extraction.
- **Open-ended generation** (summaries, chat, style) — no single ground truth. Reference-overlap
  metrics (ROUGE/BLEU) are weak proxies for quality; here you reach for LLM-as-judge or human eval.
- **Perplexity / eval loss** — a sanity signal, not a task metric. It measures fit to held-out text,
  not whether the model does the job. Report it, don't trust it alone.

Whenever a task metric is computable from ground truth, prefer it over any judge — it's deterministic
and un-gameable.

## LLM-as-judge, done carefully

For open-ended outputs, a strong LLM scoring responses against a rubric is the pragmatic standard.
Done naively it's misleading; a few disciplines make it trustworthy:

- **Prefer pairwise over absolute.** Asking "is this a 7 or 8 out of 10?" is noisy; asking "which of
  these two responses is better, A or B?" is far more reliable. Judge base-vs-fine-tune head to head
  and report a **win rate**.
- **A concrete rubric, not "which is better."** Spell out the axes (correct, concise, valid format,
  on-tone) so the judge is consistent across examples.
- **Randomize and swap positions.** LLM judges have a **position bias** — they favor the first (or
  second) option. Run each pair both orders and average, or randomize, so position cancels out.
- **Use a different, strong model as judge**, ideally not the same family you fine-tuned, to avoid a
  model preferring its own style.
- **Spot-check the judge against humans** on a sample. If the judge disagrees with your own reading,
  fix the rubric before trusting the numbers.
- **Report cost and variance.** Judge scores wander run to run; fix the judge, its prompt, and
  temperature (0) so the eval is reproducible.

A simple, robust setup: 200 held-out prompts, generate from base and fine-tune, judge each pair in
both orders with a rubric, report win rate with a confidence interval.

## Compare against the base model — and probe for regressions

Two comparisons matter, and most people only do the first:

1. **Task performance vs base.** Did the fine-tune beat the base model on your task metric? This is
   the win you were chasing.
2. **General-capability regression.** Did it get *worse* at things outside your task? This is
   catastrophic forgetting (Lesson 04) and it's invisible if you only measure your task. Keep a small
   **general-capability probe** — a handful of general-knowledge, reasoning, and instruction-following
   prompts — and run it every time. A fine-tune that gains 10 points on your task but tanks general
   instruction-following may be a net loss depending on how the model is used.

Report both. "Task +12%, general capability flat" is a shippable result; "Task +12%, general
capability −20%" is a decision, not a victory.

## Eval contamination: the silent invalidator

**Contamination** is when eval data (or a paraphrase of it) leaks into training. It makes the
fine-tune look great and the gain is fake — you measured memorization, not generalization. It's the
single most common reason a fine-tune that "crushed the eval" flops in production. Sources and
defenses:

- **Train/eval overlap.** Dedup eval against train, including near-duplicates and paraphrases, not
  just exact matches. Split on the natural unit (document, customer, conversation) so related rows
  don't straddle the split (Lesson 02).
- **Public-benchmark leakage.** Public benchmarks are often already in the base model's *pretraining*
  data, so a high score can reflect memorization from pretraining, not your fine-tune. Prefer a
  **private eval built from your own data** for the number you actually trust.
- **Synthetic-data leakage.** If you generated training data from the same prompts as your eval, you
  contaminated it. Keep eval prompts entirely separate from anything that seeded training data.
- **Iterative overfitting to the eval.** If you tune hyperparameters against the eval many times, you
  slowly overfit *to the eval itself*. Keep a final **held-out test set** you look at once, at the
  end — not during the tuning loop.

## A concrete judge rubric

Vague judge prompts give vague, noisy scores. Spell out the axes and force a discrete choice. A
usable pairwise template:

```text
You are comparing two assistant responses to the same user request.
Judge ONLY on these axes, in priority order:
1. Correctness — is the content accurate and complete?
2. Format — is it valid JSON matching the required schema?
3. Concision — no filler, no repetition.

Request: {prompt}
Response A: {a}
Response B: {b}

Think step by step, then output exactly one token on the last line:
"A", "B", or "TIE".
```

Run every pair twice with A and B swapped; if the verdict flips when you swap, that pair is a
position-bias artifact — count it as a tie. Aggregate to a **win rate** (wins / (wins + losses),
ties excluded, or counted as half). Fix the judge model, its prompt, and temperature 0 so the eval
is reproducible across runs.

## Is the improvement real, or noise?

A win rate of 55% on 40 examples is not a result — it's within the margin of coin-flipping. Two cheap
guards against fooling yourself:

- **Report an interval, not a point.** For a win rate `p` over `n` comparisons, the rough standard
  error is `sqrt(p(1-p)/n)`. At `n = 40`, that's ~8 points — so 55% ± 8 overlaps 50%, meaning "no
  detectable difference." At `n = 400` it shrinks to ~2.5 points and 55% becomes a real signal. This
  is exactly why the eval set has to be a few hundred, not a few dozen.
- **Eyeball the disagreements.** Read the 10 examples where base won and the 10 where the fine-tune
  won. If the "wins" are cosmetic and the "losses" are substantive, your metric is measuring the
  wrong thing regardless of what the aggregate says.

The same logic applies to a ground-truth metric: a 2-point exact-match gain on a 100-example eval is
inside the noise; make the eval big enough that the gain you care about clears the standard error.

## A minimal eval harness

```python
import json
from transformers import pipeline

def evaluate(model_path, eval_path):
    gen = pipeline("text-generation", model=model_path, max_new_tokens=256, do_sample=False)
    exact, valid, n = 0, 0, 0
    for row in map(json.loads, open(eval_path)):
        out = gen([{"role": "user", "content": row["prompt"]}])[0]["generated_text"][-1]["content"]
        n += 1
        try:
            pred = json.loads(out)                 # format-compliance metric
            valid += 1
            exact += int(pred == row["answer"])    # task metric (ground truth)
        except json.JSONDecodeError:
            pass
    return {"exact_match": exact / n, "valid_json": valid / n, "n": n}

base = evaluate("Qwen/Qwen2.5-7B-Instruct", "eval.jsonl")   # baseline, run once
ft   = evaluate("qwen-sft/merged",          "eval.jsonl")
print("base:", base, "\nfine-tuned:", ft)
```

Deterministic (`do_sample=False`), ground-truth-based where possible, base-vs-fine-tune side by side.
That's the spine; add an LLM-judge win rate for the open-ended parts and a general-capability probe
for regressions.

## Key takeaways

- **Build and freeze the eval before training**, and pin a **base-model baseline** — the delta is
  your only honest measure of improvement.
- Match the metric to the task: **ground-truth metrics (exact match, F1, schema-valid) whenever
  possible**; LLM-as-judge only for open-ended outputs. Perplexity/loss is a sanity check, not a task
  metric.
- **LLM-as-judge:** pairwise not absolute, a concrete rubric, swap positions to cancel position bias,
  a different strong judge, temperature 0, spot-checked against humans.
- Report **two numbers**: task gain vs base *and* general-capability regression (catastrophic
  forgetting is invisible if you only measure your task).
- **Eval contamination** (train/eval overlap, public-benchmark leakage, tuning against the eval)
  fakes gains — dedup, use a private eval, and keep a final test set you touch once.

## Try it

Build the eval harness before you touch the model. (1) Freeze a held-out eval set of ~100+ examples
from your task and dedup it against your training data (assert zero overlap, including near-matches).
(2) Run the *base* model and record your task metric — this is the baseline; write it down. (3) Run
your Lesson 04 fine-tune on the same eval and compute the delta. (4) Add an LLM-as-judge win rate:
for 30 open-ended prompts, generate from base and fine-tune, and have a strong judge pick the winner
in *both* orders — report the win rate and check whether swapping order changes the answer (that's
position bias in action). (5) Run a 10-prompt general-capability probe on both models and confirm the
fine-tune didn't regress. If your fine-tune wins on task, holds on general capability, and the win
survives a clean, uncontaminated eval — you have a shippable result and the evidence to prove it.
