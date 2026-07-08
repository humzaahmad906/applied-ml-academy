# 05 — Preference Tuning: Beyond SFT with DPO

SFT teaches the model to imitate good answers. Its ceiling is that it only ever sees good answers —
it never learns to *prefer* a good response over a plausible-but-worse one, because it never sees the
worse one contrasted with the better. When SFT plateaus and the residual problem is a *quality axis*
("both answers are correct, but this one is better — more concise, safer, better formatted"),
preference tuning is the next rung. In 2026 the default tool for this is **DPO**, and this lesson is
how to run it in TRL, what data it needs, and — importantly — when it's *not* worth the trouble.

## The intuition, without the derivation

The full derivation is in the alignment chapter; here's the applied version. You collect **preference
pairs**: for a prompt `x`, a preferred response `y_w` ("winner") and a dispreferred one `y_l`
("loser"). Classic RLHF trains a separate reward model on these pairs, then runs PPO to maximize
reward under a KL leash to the reference model — powerful but heavy, with a reward model, a value
network, and a finicky on-policy RL loop to babysit.

**DPO (Direct Preference Optimization)** proves you don't need any of that machinery. The
KL-regularized RLHF objective has a known closed-form optimal policy, and inverting it collapses the
whole pipeline into a single supervised loss on preference pairs:

$$
L_{\text{DPO}} = - \log \sigma\!\left( \beta \left( \log\frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right)
$$

Read it plainly: **increase the log-probability of the winner relative to the frozen reference model,
decrease it for the loser.** `β` (the KL coefficient) controls how hard you're allowed to pull away
from the reference. There's no reward model and no RL loop — the policy *is* the implicit reward
model. That simplicity is why DPO (and cousins IPO, KTO, ORPO) is the default preference method for
practitioners now.

## The data: prompt, chosen, rejected

DPO needs a **preference dataset**, not an instruction dataset. Each row is a prompt plus two
completions labeled better/worse:

```python
{"prompt":   [{"role": "user", "content": "Explain a transformer in one sentence."}],
 "chosen":   [{"role": "assistant", "content": "A transformer maps a sequence to a sequence using self-attention to mix information across positions."}],
 "rejected": [{"role": "assistant", "content": "A transformer is a deep learning model that is very powerful and used for many tasks and works really well."}]}
```

TRL accepts both conversational (as above) and standard-text forms, and both **explicit-prompt**
(`prompt` + `chosen` + `rejected`) and **implicit-prompt** (just `chosen`/`rejected` with the context
implied) layouts — it can extract the shared prefix automatically. Where do the pairs come from?

- **From your SFT model's own outputs.** Sample two completions per prompt, have a human (or a strong
  LLM judge) pick the better one. This on-policy source is the most effective because you're teaching
  the model to prefer better versions of *its own* behavior.
- **Public preference datasets** (e.g. UltraFeedback, HH-style) for general helpfulness/harmlessness.
- **Constructed pairs** where you can define better/worse mechanically — e.g. the schema-valid output
  is `chosen`, a deliberately malformed one is `rejected`.

Quality matters as much as in SFT: the *contrast* must be meaningful and consistent. If your "winner"
and "loser" differ randomly rather than along the axis you care about, DPO learns noise.

## Running DPOTrainer

DPO must run **after SFT** — it refines a model that already produces reasonable outputs; it can't
bootstrap format and basic behavior from scratch. You start from your SFT checkpoint (or SFT adapter,
merged or loaded), and DPO also works with LoRA:

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "qwen-sft/merged"                 # your SFT'd model (or base + SFT adapter)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="bfloat16")

pref = load_dataset("json", data_files={"train": "prefs.jsonl"}, split="train")

cfg = DPOConfig(
    output_dir="qwen-dpo",
    beta=0.1,                             # KL strength: the main knob
    learning_rate=5e-6,                   # ~10-50x LOWER than SFT
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=1024, max_prompt_length=512,
    bf16=True, logging_steps=10, seed=42,
)

trainer = DPOTrainer(
    model=model, args=cfg, train_dataset=pref,
    processing_class=tok,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
```

Note what DPO handles for you: with a PEFT config, the **reference model** `π_ref` is the base with
the adapter disabled — TRL runs the same model twice (adapter on = policy, adapter off = reference),
so you don't pay for a second copy in memory. Without PEFT it loads a frozen reference copy.

## The knobs that differ from SFT

- **`beta`** — the KL coefficient, and the whole game. It controls how far the policy may drift from
  the reference. **Too low → reward hacking and mode collapse** (the model finds degenerate outputs
  that win the pairwise comparison but are bad); **too high → the model barely moves off SFT.** Start
  at `0.1`; `0.05`–`0.5` is the usual range. This is the one hyperparameter to sweep.
- **Learning rate is much lower** than SFT — `5e-6` to `5e-7`, roughly 10–50× smaller. DPO makes
  precise, small adjustments; a large LR blows past the good policy and collapses quality.
- **One epoch, usually.** DPO overfits even faster than SFT. More than one pass over preference data
  often degrades quality.

## Watching a DPO run

DPO logs metrics beyond loss that tell you if it's working:

- **`rewards/chosen`** and **`rewards/rejected`** — the implicit rewards (`β·log π/π_ref`). You want
  `chosen` to rise above `rejected`; the **`rewards/margins`** (their difference) should grow
  positive.
- **`rewards/accuracies`** — fraction of pairs where the model correctly prefers the chosen response.
  Should climb toward, but realistically not to, 1.0.
- If margins explode and the model's actual generations get weird or repetitive, `beta` is too low —
  you're reward-hacking the pairwise objective. Raise it.

## A nod to GRPO and verifiable rewards

DPO learns from *offline* preference pairs judged by humans or an LLM. When your task has a
**verifiable reward** — math with a known answer, code that passes tests, structured output you can
validate — you don't need preference pairs at all. You can run online RL directly against the
checkable reward. **GRPO (Group Relative Policy Optimization)** is the algorithm that made this cheap:
sample a group of responses per prompt, score them all, and use the group's own mean/std as the
baseline for the advantage — no separate value network. TRL's `GRPOTrainer` takes a reward function
(signature `(completions, **kwargs) -> list[float]`) and optimizes against it. It's what drove the
2024–2025 reasoning-model boom.

For most *applied product* fine-tunes, the ladder is: **SFT first** (90% of the win for narrow
tasks), **DPO** if there's a quality axis SFT can't capture, and **GRPO/expert-iteration** only when
you have a genuine verifiable reward and SFT has plateaued. Don't reach for GRPO because it's
fashionable — it's more moving parts, and for a well-scoped extraction or style task, good SFT plus a
touch of DPO is usually the whole answer.

## The DPO cousins: IPO, KTO, ORPO

DPO isn't the only offline preference method, and TRL implements the main variants. You rarely need
them, but knowing when each helps saves a stuck run:

- **IPO** fixes a DPO failure mode where, given noisy or near-tied pairs, DPO can over-optimize and
  push the winner's probability toward 1 while collapsing the loser — IPO adds a regularizer that
  bounds this, useful when your preference labels are weak or inconsistent.
- **KTO** needs only a *binary* signal per example (this output is good / this output is bad), not
  pairs. If your feedback is thumbs-up/down rather than A-vs-B comparisons, KTO fits the data you
  actually have and skips the cost of building matched pairs.
- **ORPO** folds preference optimization *into* SFT — one stage, no separate reference model, using a
  monolithic loss with an odds-ratio preference term. It trades a little quality for skipping the
  two-stage pipeline entirely; attractive when you want the whole thing in one pass.

Start with plain DPO. Move to IPO if margins explode on noisy data, KTO if your feedback is binary,
ORPO if you want to collapse SFT and preference tuning into a single run.

## How much preference data

Preference tuning needs less data than you'd guess because each pair carries a sharp, directional
signal. **A few hundred to a few thousand high-quality pairs** moves a model meaningfully; below ~100
the signal is usually too weak and noisy to help. As with SFT, the *contrast quality* dominates the
count — 300 pairs that cleanly isolate the axis you care about (concise-vs-verbose, valid-vs-invalid)
beat 3,000 where the winner and loser differ along random, irrelevant dimensions. On-policy pairs
(sampled from your own SFT model) at the low hundreds routinely outperform larger off-the-shelf sets
because they target exactly the behaviors your model currently gets wrong.

## When preference tuning is *not* worth it

- **SFT hasn't plateaued yet.** Fix the SFT data first — more/cleaner demonstrations often beat
  adding a DPO stage.
- **You can't articulate the quality axis.** DPO needs a consistent better/worse signal. "It's just
  not good enough" isn't one.
- **The gap is knowledge, not preference.** Back to Lesson 01 — that's RAG, not any fine-tune.
- **Tiny preference data.** A few dozen noisy pairs won't move a model meaningfully and can hurt.

## Key takeaways

- **DPO** turns RLHF into a single supervised loss on `(prompt, chosen, rejected)` pairs — no reward
  model, no value network, no RL loop. It's the default preference method in 2026.
- The loss pushes the winner's log-prob up and the loser's down, *relative to a frozen reference*;
  **`β` is the KL leash and the main knob** — too low reward-hacks, too high barely moves.
- Run DPO **after** SFT, with a **much lower LR (~5e-6)** and usually **one epoch**; it overfits fast.
  With PEFT, the reference is just the adapter turned off (no extra memory).
- Watch **`rewards/margins`** and **`rewards/accuracies`**, not just loss.
- **GRPO** (via `GRPOTrainer`, with a reward function) is for *verifiable* rewards — reach for it only
  when SFT has plateaued and correctness is checkable. For most product fine-tunes, SFT + optional
  DPO is enough.

## Try it

Build a tiny preference dataset and run DPO on the model you SFT'd in Lesson 04. (1) Take 50 held-out
prompts, sample two completions each from your SFT model (temperature ~0.8), and label the better one
— by hand, or with a strong LLM as judge using a fixed rubric (concise + correct + valid format).
Write them as `{prompt, chosen, rejected}`. (2) Run `DPOTrainer` for 1 epoch at `lr=5e-6`,
`beta=0.1`, logging `rewards/margins` and `rewards/accuracies`. (3) Sweep `beta ∈ {0.05, 0.1, 0.5}`
and, for each, generate on your fixed eval prompts — find the setting where the chosen-style behavior
strengthens *without* the outputs degenerating (repetition, truncation, weirdness). You'll feel
directly how `beta` trades preference-following against staying close to the reference — the single
most important thing to internalize about preference tuning.
