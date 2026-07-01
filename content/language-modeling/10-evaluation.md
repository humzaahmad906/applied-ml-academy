# 10 — Evaluation: Measuring Whether the Model Is Any Good

Training and inference are the easy part in the sense that they either run or they do not.
Evaluation is where people fool themselves. A number that looks like progress can be measurement
artifact, contamination, or a metric that does not track what you actually care about. It helps to
frame any eval as four decisions you make explicitly: **input design** (what you put in front of
the model), **model invocation** (how you run it — greedy, sampled, few-shot, chain-of-thought),
**output evaluation** (how you turn its output into a score), and **result interpretation** (what
the number actually licenses you to conclude). Most bad evaluations go wrong by leaving one of these
four implicit. This chapter is about measuring honestly.

## The intrinsic metric: perplexity

The direct objective a language model optimizes is the likelihood of the next token. Perplexity is
the standard summary of that: the exponential of the average negative log-likelihood per token on
held-out text. Lower is better. Intuitively, a perplexity of `p` means the model is on average as
uncertain as if it were choosing uniformly among `p` options at each position. The classic
language-modeling corpora for this — Penn Treebank, WikiText-103, the One Billion Word benchmark —
exist precisely to report perplexity on a fixed held-out distribution.

Perplexity is useful because it is cheap, continuous, and directly tied to the training objective,
so it is the right thing to watch during training and for comparing checkpoints of the same model.
Its limits: it is only comparable across models that use the same tokenizer (perplexity is
per-token, and different tokenizers cut text into different numbers of tokens, so you must at least
normalize per-byte or per-word to compare across tokenizers), it does not directly tell you
downstream task quality, and it says nothing about instruction-following or reasoning. Use it to
track a training run and compare siblings, not to rank different model families.

## Downstream benchmarks

To measure what the model can actually do, you run it on task benchmarks. The families to organize
around:

- **Knowledge multiple-choice** (MMLU: 57 academic subjects; and its successors): questions with
  answer choices, scored by which choice the model prefers. MMLU is saturating, so harder variants
  keep appearing — **MMLU-Pro** widens each question to 10 choices to reduce guessing and
  saturation, **GPQA** uses PhD-level expert-written questions designed to be Google-proof, and
  **HLE** ("Humanity's Last Exam") pushes toward the frontier with ~2,500 hard, often multimodal
  questions. The arms race exists because a benchmark stops discriminating once frontier models
  score near the ceiling.
- **Math and code with checkable answers** (GSM8K, MATH, HumanEval and successors): the answer is
  verifiable by exact match or by running the code against tests. This verifiability is gold,
  because you get an objective score with no judgment call, and it is exactly what makes these
  domains suitable for the RL methods in the alignment chapter.
- **Agentic and tool-use tasks:** multi-step problems requiring the model to plan, call tools, and
  integrate external results. Harder to score because success is a trajectory, not a single output,
  and because capability and safety intertwine (a capable cybersecurity agent is dual-use).
- **Long-form and open-ended generation:** no single correct answer, so you need either human
  judgment or a model judge (below).

How you score a multiple-choice question matters as much as which benchmark, and this is where the
"output evaluation" decision bites. You can score by the log-likelihood the model assigns to the
answer **letter** ("C"), or to the answer **text** ("Paris"), or by having it **generate** an
answer and parsing it. These give different numbers on the same model: text-likelihood favors
different models than letter-likelihood, and generation-plus-parsing depends on formatting
robustness. Length-normalizing the option likelihood (dividing by token count) changes rankings
again. Papers comparing models sometimes differ mostly in scoring protocol rather than model
quality. When you see a benchmark number, ask how it was scored before you believe a comparison.
This is what tools like the EleutherAI LM Evaluation Harness and HELM standardize — a fixed
prompt, few-shot format, and scoring rule per task — so that two models are at least measured the
same way.

## Contamination: the thing that quietly ruins everything

The most important evaluation problem at scale, and one worth treating as a first-class concern.
Pretraining corpora are scraped from the web. Many benchmarks are also on the web. If the
benchmark's questions and answers ended up in the training data, the model can score well by
memorization, and your evaluation measures leakage rather than capability. This is contamination,
and it is pervasive and hard to fully rule out.

Detection is itself a research problem. There are two families of approach: statistical tests
that exploit **exchangeability** — if a model has memorized a benchmark, it will assign
systematically higher likelihood to the canonical ordering of examples than to a shuffled ordering,
which a clean model would not — and simply **encouraging providers to disclose** the train/test
overlap statistics they measured internally. Defenses on your side: hold out freshly created
evaluation data the model could not have seen (dated after the training cutoff), decontaminate the
training set by removing documents that overlap with benchmark items (n-gram overlap detection —
this connects directly to the decontamination stage in the data-pipeline chapter), and treat suspiciously high
scores on public benchmarks with skepticism. When you build your own evals for a product, keep a
private held-out set that never touches training or hyperparameter tuning, because the moment an
eval influences your decisions it starts to leak into your model through your choices. This is the
eval analog of a test set: once you have looked at it enough to tune against it, it is no longer a
clean measurement.

## Instruction and chat evaluation

Base-model benchmarks do not capture whether a chat model is helpful, so a separate ecosystem
evaluates open-ended, instruction-following behavior. The main instruments:

- **LMSYS Chatbot Arena:** real users submit a prompt, get two anonymous model responses, and vote
  which is better; an Elo-style rating aggregates the pairwise votes. This is the closest thing to a
  ground-truth human-preference signal at scale, but it is slow, uncontrolled (users pick their own
  prompts), and gameable by style.
- **MT-Bench and AlpacaEval:** fixed sets of open-ended prompts scored by an LLM judge, cheaper and
  reproducible proxies for the Arena.

## LLM-as-judge

For open-ended outputs where there is no automatic metric, the common approach is to use a strong
model to score or compare outputs. It is scalable and correlates reasonably with human preference,
which is why MT-Bench, AlpacaEval, and most internal evals rely on it. Its failure modes are real
and you must design around them: judges have **position bias** (they favor the first or second
option in a pairwise comparison, so you randomize order and average both orderings), **length
bias** (they favor longer answers, so you control for length), **self-preference** (a judge favors
outputs from its own model family), and they can be gamed by confident formatting and markdown. Use
LLM judges for relative comparison and coarse quality signals, calibrate them against human ratings
on a sample, and do not treat their scores as ground truth. A cheap sanity check is to include a
few items where you know the answer and confirm the judge gets them right.

## The realism distinction

There is a line most benchmark discussions miss: the difference between **quiz-style**
evaluation, where the evaluator already knows the answer and is checking the model against it, and
**information-seeking** tasks, where a realistic user genuinely does not know the answer and derives
value from the response. Almost all standard benchmarks are quiz-style, which is convenient
(automatic scoring) but systematically unrepresentative of how the model is actually used. It also
folds cost into the picture: serious evaluation should report capability *and* inference price
together, because a model that is marginally better but 10x more expensive per
query is not obviously better for a product. Both points push you toward evaluating on your own
realistic task distribution rather than trusting leaderboard rank.

## Building an evaluation you can trust

The practical recipe, from your product-engineer hat:

1. Decide what "good" means for the actual task, in terms a user would recognize, before you look
   at any model outputs.
2. Build a held-out test set from real target-domain data (for you: real shipping labels, BOLs,
   invoices), labeled carefully, that never touches training or tuning.
3. Use automatic metrics where the task allows exact checking (field-level accuracy, exact match on
   structured extraction). Your 94% F1 versus a baseline is this kind of honest, checkable metric,
   which is exactly why it is credible.
4. Use a small, calibrated human or LLM-judge process only for the genuinely open-ended parts.
5. Track a few metrics, not one. A single number hides regressions; a model can improve average
   accuracy while getting worse on the hard subset you care about. Slice by document type,
   language, difficulty.
6. Re-check for contamination and distribution shift periodically, especially when a number jumps.

## Why lower loss does not always mean better product

Scaling laws are about loss, and benchmarks are about capability, and
those are correlated but not identical. A model can have lower perplexity but worse behavior on
your task because your task's distribution differs from the pretraining distribution, or because
the thing you care about (correct field extraction, calibrated refusals) is a small part of the
loss. Always close the loop with a task metric on your own held-out data. Loss is the thing the
model optimizes; the task metric is the thing you actually get paid for.

## Key takeaways

Every eval is four explicit decisions — input, invocation, output scoring, interpretation — and bad
evals leave one implicit. Perplexity is the cheap intrinsic metric, right for tracking a training
run and comparing same-tokenizer siblings, wrong for ranking different model families. Downstream
benchmarks measure capability; MMLU and its ever-harder successors (MMLU-Pro, GPQA, HLE) exist
because benchmarks saturate, and the checkable-answer ones (math, code, structured extraction) are
the most trustworthy because scoring is objective. How multiple-choice is scored — answer letter vs
text vs generation, length-normalized or not — changes rankings, which is why harnesses like
LM-Eval-Harness and HELM fix the protocol. Contamination is pervasive and quietly inflates scores;
detect it via exchangeability tests and n-gram overlap, and keep a private held-out set that never
influences training or tuning. Chat models need their own evals (Chatbot Arena, MT-Bench,
AlpacaEval); LLM-as-judge scales them but has position, length, and self-preference biases you must
control and calibrate. Prefer realistic information-seeking tasks and report cost alongside
capability, and track several task metrics on your own target-domain data, because lower loss does
not guarantee a better product.
