# 10 — Evaluation: The Skill That Gets You Hired

Anyone can prompt a model and eyeball the output. What separates a practitioner from a hobbyist is
knowing whether the model is actually good — reproducibly, on the thing you care about, before it
reaches a user. Evaluation is the least glamorous and most valuable part of the job: it is how you
choose between two models, decide whether a prompt change helped or hurt, and catch a regression
before it ships. Teams that treat eval as an afterthought ship confidently and are wrong; teams that
build eval first move slower at the start and then never guess again. This module is the full stack:
the intrinsic metric (perplexity), the public benchmark canon and why it is decaying, contamination,
the LLM-as-judge and its biases, human eval, and — the part that actually matters at work — building
a product eval that gives you a number you can trust.

## Perplexity: what it measures and what it hides

The most basic intrinsic metric for a language model is **perplexity**, the exponential of the
average negative log-likelihood the model assigns to held-out text:

$$
\text{PPL} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i})\right)
$$

Read it as the model's *branching factor*: a perplexity of 10 means the model is, on average, as
uncertain as if choosing uniformly among 10 next tokens. Lower is better. It falls directly out of
the cross-entropy training loss ($\text{PPL} = e^{\text{loss}}$ in nats), so it is free to compute,
needs no labels, and is the right metric for tracking pretraining and comparing two models on the
*same tokenizer and corpus*.

Its limits are severe and worth stating plainly. Perplexity is **tokenizer-dependent** — two models
with different vocabularies produce perplexities that cannot be compared at all, because they are
averaging over different units. It rewards fitting the *distribution* of text, which is not the same
as being *useful, correct, or safe*: a model can have excellent perplexity and still fail at
instruction-following, reasoning, or refusing harmful requests, because those depend on behavior the
next-token objective only loosely constrains. And it is dominated by easy, frequent tokens. So
perplexity is a fine dashboard metric for pretraining and a useless one for "is this assistant good."
For that you need task benchmarks and, ultimately, your own eval.

## The benchmark canon and its saturation

A handful of public benchmarks became the industry's shared vocabulary. Know what each *measures*,
because interviewers name them and product decisions cite them:

- **MMLU / MMLU-Pro** — multiple-choice knowledge across 57 subjects. The default "how much does it
  know" number. MMLU is **saturated** (top models cluster in the high 80s–90s, inside the noise of
  its known label errors), which is why MMLU-Pro rebuilt it with harder, cleaner, ten-option
  questions to restore headroom.
- **GSM8K** — grade-school math word problems. Once a reasoning showcase, now saturated (>95%) and
  heavily contaminated; useful mainly as a sanity floor.
- **MATH** — competition mathematics, far harder than GSM8K, still discriminating for the reasoning
  models (see [reasoning](11-reasoning.md)).
- **HumanEval / MBPP** — code generation scored by executing unit tests (pass@k). Real, functional
  correctness rather than text overlap — but small and largely saturated; superseded in practice by
  harder suites.
- **GPQA** — "Google-proof" graduate science questions written by domain PhDs to resist lookup;
  designed as a hard knowledge/reasoning benchmark with real headroom.
- **SWE-bench (and Verified)** — resolve real GitHub issues in real repos, graded by the repo's own
  tests. The benchmark that best predicts practical coding-agent utility, which is why it is the one
  labs now compete on hardest.
- **MMMU** — multimodal (image + text) reasoning across disciplines; the multimodal analog of MMLU
  (see [multimodality](14-multimodality.md)).

The pattern across all of them is **saturation**: a benchmark is useful only while it discriminates,
and the moment models cluster near the top it stops informing decisions — the differences left are
noise, label errors, and contamination. Benchmarks have a half-life. Treat a leaderboard as a *coarse
filter* ("is this model in the right class?"), never as evidence it will do *your* task. HELM
(Liang et al., 2022) made the broader point that a single number is malpractice: evaluate along
*multiple* axes — accuracy, calibration, robustness, fairness, efficiency — because a model that wins
on accuracy can lose badly on cost or bias, and you need to see both.

## Contamination: why the leaderboard lies

**Contamination** is test data leaking into training data. Because pretraining scrapes much of the
public web, and benchmarks live on the public web, the questions (and sometimes answers) are often
*in the training set*. A contaminated model is doing recall, not reasoning, and its benchmark score
is inflated by an unknown amount — which is exactly why GSM8K at 95% tells you almost nothing.

Detection is imperfect but real: **n-gram overlap** between test items and the training corpus (when
you have access to it); the **canary-string** convention where benchmark files embed a GUID that
should never appear in a clean corpus; and behavioral tests like checking whether a model completes a
benchmark question verbatim from a partial prompt, or comparing accuracy on the original set versus a
freshly written, held-out variant — a large gap is the tell. The defensive posture that follows: prefer
**recent** benchmarks (authored after a model's cutoff), prefer **private held-out** sets, and above
all build your own eval on data the model has never seen, which is the only score you can fully trust.

## LLM-as-judge

Most things you care about — is this summary good, is this answer helpful, is this tone right — have
no automatic metric. The scalable substitute is **LLM-as-judge**: prompt a strong model to score or
compare outputs. It correlates well with human judgment, costs cents, and runs in seconds, which is
why it now underpins most generation eval. Two modes:

- **Pairwise** — show the judge two responses (A and B) and ask which is better. More reliable,
  because relative judgments are easier and more stable than absolute ones.
- **Rubric / pointwise** — score one response 1–5 against explicit criteria. Necessary when you have
  no comparison, but noisier; anchor it with a detailed rubric and few-shot examples or the scores
  drift.

The judge is itself a model, so it has **systematic biases** you must design around, not wish away:

- **Position bias.** The judge favors whichever answer appears *first* (or sometimes second) purely
  by position. The fix is mandatory: run both orderings (A,B) and (B,A) and only count a win if it is
  consistent across both; ties otherwise.
- **Length bias / verbosity.** Judges prefer longer, more elaborate answers even when the extra text
  adds nothing correct. This directly rewards padding and is why raw judge scores can be gamed by
  telling a model to write more. Control for length (report response length alongside win rate) or
  use a length-debiased metric — AlpacaEval 2.0's length-controlled win rate exists precisely for
  this.
- **Self-preference bias.** A judge tends to rate outputs from its *own* model family higher. Never
  let a model be the sole judge of its own family in a competitive eval; use a different judge, or an
  ensemble, and be suspicious of a lab grading itself.

Other traps: judges are lenient graders that miss subtle factual errors, and they inherit the
generator's blind spots. Judge eval is a real tool, not a free oracle — validate it against human
labels on a sample before you trust its verdicts at scale.

## Arena-style ELO

To rank many models against each other, **Chatbot Arena** (LMSYS) collects *human* pairwise votes on
blind head-to-head responses to real user prompts and aggregates them into **ELO ratings**, the same
system used for chess. Each model has a rating $R$; the expected win probability of A over B is

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

and ratings update after each vote toward the observed outcome. Its strengths are that prompts are
real and diverse, judgments are human, and the setup is hard to overfit because prompts are not
released as a fixed set. Its weaknesses: it measures *human preference* (which favors friendly,
confident, well-formatted answers — not necessarily correct ones), it has thin coverage of
specialized domains, and preference is gameable by style. Read Arena rank as "which model do people
like chatting with," which correlates with but is not identical to "which model is most correct or
most useful for my task." The same ELO machinery, with an LLM judge instead of humans, powers
automatic arenas like AlpacaEval — cheaper, faster, and subject to all the judge biases above.

## Human eval, done right

When the stakes justify it, humans are still the gold standard — but only if run like a measurement,
not a vibe check. That means: a **written rubric** with concrete criteria and examples so two
annotators agree on what "good" means; **blind** evaluation (annotators do not know which system
produced which output, to kill brand and order effects); **multiple annotators per item** with an
**inter-annotator agreement** number (Cohen's or Fleiss' kappa) reported — if your annotators do not
agree with each other, no model comparison built on their labels means anything; and enough items
for the difference you care about to clear statistical noise. Human eval is slow and expensive, so use
it to *calibrate your cheap automatic evals* (does the judge agree with humans on a 200-item sample?)
and to make the final high-stakes call, not for every iteration.

## Building a product eval

This is the section that earns the salary. A public benchmark tells you nothing about whether your
support bot resolves tickets. You build the eval, and it has three layers.

**1. The golden set.** A curated set of representative inputs with known-good expected outputs (or
grading criteria), built from real traffic, edge cases, and past failures — a few dozen to a few
hundred items, quality over quantity. It is **frozen**: you may *add* items, but rebalancing or
removing them invalidates every prior number and destroys comparability across runs (this is a
discipline, not a suggestion — see the eval-set rules that recur throughout this repo). Each item has
a grader: exact match or unit tests where the answer is objective, a validated LLM-judge with a rubric
where it is not.

**2. The regression gate.** Wire the golden set into CI. Every prompt change, model swap, or
parameter tweak reruns it, and the score must not drop below a threshold before it merges. This is
**eval-driven development** — the ML analog of test-driven development — and it is the difference
between "I think this prompt is better" and "this prompt scores 84% vs 81%, merge it." Without the
gate, every change is a gamble and regressions ship silently; with it, prompt engineering becomes
engineering. Track the score over time so you can see drift and attribute regressions to specific
changes.

**3. Online metrics.** Offline eval is necessary but never sufficient, because your golden set cannot
anticipate everything real users do. In production you close the loop with **online** signals: task
success (did the ticket get resolved, did the user accept the answer), thumbs up/down, escalation and
abandonment rates, latency and cost, and A/B tests between versions. The honest workflow is: offline
eval to filter changes fast and cheaply, online metrics to confirm the ones that pass actually help
real users. A change that wins offline and loses online means your golden set is missing something —
so you add those cases to it, and the eval gets better.

Wire it up with the community harnesses where they fit — `lm-eval-harness` for standard benchmarks,
`inspect` for structured agentic/graded evals (see [modern-stack](16-modern-stack.md)) — but the
golden set and its grader are yours to build, because they encode what *your* product means by good.

## Eval-driven development as the hiring signal

Read a job description for an applied LLM role and "build and own the evaluation" is in it, because
managers have learned the hard way that the bottleneck is not generating outputs — it is knowing
whether they got better. In an interview, "how would you know if your change helped?" is the question
that separates candidates: the weak answer is "I'd try some examples and see"; the strong answer names
a frozen golden set, a grader appropriate to the task, a regression gate in CI, judge-bias controls,
and an online metric to confirm offline wins. If you take one habit from this course into your work,
make it this: **build the eval before you build the thing.** It feels slower for a week and then it is
the only reason you can move fast without breaking things.

## What interviews ask here

- What is perplexity and when is it the wrong metric? — Exp of avg negative log-likelihood
  (branching factor); tokenizer-dependent and measures distribution-fit, not usefulness/correctness —
  useless for assistant quality.
- Why don't you trust a model's MMLU/GSM8K score? — Saturation (differences are noise) plus
  contamination (test data in training); benchmarks are a coarse filter, not proof it does your task.
- What is contamination and how do you detect it? — Test data leaked into training; detect via n-gram
  overlap, canary strings, and original-vs-fresh-variant accuracy gaps.
- Name three LLM-as-judge biases and their fixes. — Position (swap orderings, require consistency),
  length/verbosity (length-control the metric), self-preference (don't let a model judge its own
  family).
- How does Chatbot Arena rank models? — Blind human pairwise votes aggregated into ELO; measures human
  preference, which favors style and isn't the same as correctness.
- How would you know a prompt change helped? — Frozen golden set + task-appropriate grader + CI
  regression gate + online confirmation; the eval-driven-development answer.
- Why report length alongside judge win rate? — Judges reward verbosity, so win rate can be gamed by
  padding; length exposes it.

## Where this shows up on the job

- Choosing between models or providers for a feature: you cannot cite a leaderboard and be done, you
  build a task-specific eval and decide on *your* numbers, including cost and latency, not just
  accuracy.
- Owning the regression gate that guards a production prompt or pipeline, so a "small" prompt tweak
  can't silently degrade quality for users.
- Standing up LLM-as-judge for a subjective task (summary quality, tone, helpfulness) and validating
  it against human labels before trusting it at scale.
- Closing the offline-to-online loop after launch: reconciling why a change that won on the golden set
  under- or over-performed with real users, and folding those cases back into the eval.
