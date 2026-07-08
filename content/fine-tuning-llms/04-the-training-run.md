# 04 — The Training Run: A Real SFT Loop

You have a clean dataset (Lesson 02) and a LoRA/QLoRA config (Lesson 03). This lesson is the run
itself: a complete, current `SFTTrainer` script you could paste into a notebook, the hyperparameters
that matter and sane starting values, what to watch while it trains, and the failure modes that
quietly ruin a fine-tune — overfitting and catastrophic forgetting chief among them. The goal is not
to memorize magic numbers; it's to read the loss curves well enough to know whether the run is
working before you've burned the whole budget.

## The complete script

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, dtype=torch.bfloat16)

# messages-format dataset from Lesson 02, already split
ds = load_dataset("json", data_files={"train": "train.jsonl", "eval": "eval.jsonl"})

lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                  lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

cfg = SFTConfig(
    output_dir="qwen-sft",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,          # effective batch = 4 * 4 = 16
    learning_rate=2e-4,                     # LoRA wants ~10x a full-FT LR
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=2048,
    packing=True,                           # pack short examples to fill sequences
    assistant_only_loss=True,               # loss on assistant turns only
    bf16=True,                              # default on modern GPUs
    gradient_checkpointing=True,            # default in SFTConfig; saves activation memory
    logging_steps=10,
    eval_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=50, load_best_model_at_end=True,
    report_to="wandb",                      # watch curves live
    seed=42,
)

trainer = SFTTrainer(
    model=model, args=cfg, peft_config=lora,
    train_dataset=ds["train"], eval_dataset=ds["eval"], processing_class=tok,
)
trainer.train()
trainer.save_model("qwen-sft/adapter")      # saves the LoRA adapter only
```

That's a production-shaped run. Everything below explains the choices.

## The hyperparameters that matter

**Learning rate.** The single most important knob. LoRA adapters train at roughly **10× a full
fine-tune's LR** because only a few new parameters are learning — `2e-4` is the canonical LoRA
starting point (TRL's own guidance is `≈1e-4` for adapters; `1e-4`–`3e-4` is the usable band). Too
high and the loss spikes or diverges; too low and it barely moves. This is the first thing to sweep.

**Epochs.** Fewer than instinct suggests. **1–3 epochs** is the norm; on a few thousand quality
examples, 2 is a good default. Fine-tuning overfits fast because the model already knows the language
— you're nudging behavior, not teaching from scratch. More than ~3 epochs on a small dataset is how
you memorize the training set (see failure modes).

**Batch size and gradient accumulation.** Real batch size is limited by VRAM. You reach an
*effective* batch size by accumulating gradients over several micro-batches before stepping:
`effective = per_device_batch × grad_accum × num_gpus`. The script's `4 × 4 = 16` is a solid target;
effective batches of 16–64 are typical. Larger = smoother gradients but slower steps. If you OOM,
halve `per_device_train_batch_size` and double `gradient_accumulation_steps` — same math, less memory.

**Scheduler and warmup.** `cosine` decay with a short warmup (`warmup_ratio=0.03`) is the standard.
Warmup prevents an early large step from destabilizing the freshly-initialized adapter.

**`max_length`.** Set from your token-length distribution (Lesson 02), default 1024. Too short
truncates real content; too long wastes memory on padding. Sequences longer than this are truncated.

**Packing.** `packing=True` concatenates multiple short examples into one full-length sequence,
eliminating padding waste and speeding up training substantially when your examples are short
relative to `max_length`. Combine with `assistant_only_loss`/completion masking so the packing
doesn't blur which tokens carry loss.

## Watching the run: what the curves tell you

Log to Weights & Biases (`report_to="wandb"`) and watch, live:

- **Training loss** should fall smoothly then flatten. A jagged, spiking, or rising curve almost
  always means the learning rate is too high — kill the run and lower it. A curve that barely moves
  means it's too low.
- **Eval loss** is the one that matters. The healthy pattern: eval loss falls alongside train loss,
  then flattens. The moment **eval loss starts rising while train loss keeps falling, you are
  overfitting** — that inflection is your stopping point, which is exactly why the script sets
  `eval_strategy="steps"` and `load_best_model_at_end=True` (it restores the checkpoint at the eval
  minimum, not the last step).
- **`mean_token_accuracy`** (TRL logs it) is a friendlier read than loss — the fraction of tokens the
  model predicts correctly on the response. It should climb and plateau.
- **`grad_norm`** — sudden spikes signal instability (LR too high or a bad batch).

A loss curve is necessary but not sufficient. Loss going down means the model is fitting your data;
it does **not** mean the model got better at your *task*. That's what the held-out task eval in
Lesson 06 is for. Generate on a fixed set of held-out prompts before and after training and read the
outputs — a low loss with degenerate generations means something is wrong (bad masking, wrong
template, contaminated data).

## Failure modes, and how to catch them

**Overfitting.** Too many epochs or too-high LR on too-little data. The model memorizes training
examples and parrots them, generalizing poorly. Signal: eval loss rises while train loss falls;
generations echo training phrasings verbatim. Fixes: fewer epochs, more/ more-diverse data, higher
`lora_dropout`, lower LR, early stopping (`load_best_model_at_end`).

**Catastrophic forgetting.** The model gets good at your task but *loses* general capability it had
before — it forgets how to do things outside your narrow dataset. This is the sneakiest failure
because your task eval looks great while the model quietly got worse at everything else. LoRA is
inherently more resistant than full FT (the base weights are frozen; you're adding a small
correction), which is a real reason to prefer it. Further mitigations: keep LR and epochs modest,
keep `r` from being enormous, and — critically — **evaluate on general-capability probes too**, not
just your task. If you need the model to stay broadly capable, mix a small fraction of
general-instruction data into your training set.

**Wrong chat template / masking.** Loss goes down but generations are garbage or never stop. Almost
always a template mismatch (training format ≠ serving format) or the EOS token isn't aligned so the
model never learns to terminate. Re-check Lesson 02: render a training example and read the raw
string.

**Loss is NaN.** Usually fp16 instability — use `bf16=True` on modern GPUs (Ada/Hopper), which is the
`SFTConfig` default, and reserve fp16 for memory-pressure cases on older cards.

**Reproducibility.** Set `seed=42` (the script does). Fine-tune results vary run to run; a fixed seed
plus a frozen eval is what makes "it improved" a measurement instead of folklore.

## Checkpointing, resuming, and saving what you need

Long runs die — preemptible GPUs, OOMs, disconnects. The script's `save_strategy="steps"` writes
checkpoints periodically; resume with `trainer.train(resume_from_checkpoint=True)`, which restores
model, optimizer, and scheduler state so you continue exactly where you stopped. Two things worth
knowing:

- **`save_model` saves only the adapter** when you're training with PEFT — tens of MB, not the full
  model. That's the artifact you deploy or merge (Lesson 07). The base is unchanged and referenced by
  name.
- **`load_best_model_at_end=True` needs `eval_strategy` and `save_strategy` aligned** (same steps) and
  a `metric_for_best_model` (defaults to eval loss). It restores the best-eval checkpoint at the end,
  which is what you want — the *last* checkpoint is usually slightly overfit relative to the eval
  minimum.

Don't checkpoint too often on cloud storage — each save has I/O cost; every 50–100 steps is a
reasonable balance between safety and overhead.

## Gradient accumulation, concretely

Gradient accumulation is how you hit a large effective batch on a small GPU. Instead of one big
backward pass, you run `gradient_accumulation_steps` micro-batches, summing gradients, and step the
optimizer once at the end:

```text
effective_batch = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
                = 4 × 4 × 1 = 16
```

The gradient you apply is the average over all 16 examples, mathematically close to running batch
size 16 directly — but peak memory is set by the *micro-batch* of 4, not 16. This is the lever you
reach for on every OOM: halve `per_device_train_batch_size`, double `gradient_accumulation_steps`,
and the effective batch (and thus the learning dynamics) stays fixed while memory drops. It costs
wall-clock time (more forward/backward passes per step) but nothing in quality.

## When you hit an out-of-memory error

OOMs are the most common practical wall. Work the levers in this order, cheapest first: (1) lower
`per_device_train_batch_size` and raise `gradient_accumulation_steps` to keep the effective batch
fixed; (2) reduce `max_length` if your data allows (activation memory scales with sequence length);
(3) confirm `gradient_checkpointing=True` (the `SFTConfig` default) — it recomputes activations in
the backward pass to trade compute for memory; (4) switch to **QLoRA** (4-bit base) if you were
running LoRA on a bf16 base; (5) switch to **Unsloth**, whose kernels cut VRAM up to ~70% and often
turn an OOM run into one that fits. Only after exhausting these do you need a bigger GPU. Reach for
Unsloth or Axolotl specifically when a plain PEFT run won't fit or is too slow on the hardware you
have — same LoRA math, tighter memory and faster kernels.

## A sane run recipe

1. Start small — a 0.5B model, 100 examples, 1 epoch — and confirm the loss falls and
   before/after generations *change*. This validates the pipeline in minutes.
2. Scale to the real model and dataset with the defaults above (`lr=2e-4`, 2 epochs, effective
   batch 16, `packing=True`).
3. Read the eval-loss curve; if it turns up, you have too many epochs — cut them.
4. Sweep LR first (`1e-4`, `2e-4`, `3e-4`), then rank if quality plateaus.
5. Take the best-eval checkpoint (not the last), and evaluate it on your *task* eval, not just loss.

## Key takeaways

- Use `SFTTrainer` + a QLoRA config; `packing=True` and `assistant_only_loss=True` are the two
  efficiency/correctness switches most people miss.
- **LR ≈ 2e-4 for LoRA** (10× a full-FT LR) is the first thing to tune; **1–3 epochs**, effective
  batch 16–64 via gradient accumulation, cosine schedule with short warmup.
- Watch **eval loss**: the inflection where it rises while train loss falls is overfitting — stop
  there. `load_best_model_at_end=True` keeps the right checkpoint.
- Low loss ≠ better task performance. Always read before/after generations on held-out prompts.
- Guard against **catastrophic forgetting** (LoRA helps; also probe general capability and optionally
  mix in general data) and template/masking bugs (garbage generations despite falling loss).
- Set a seed, freeze the eval, start tiny to validate the pipeline before scaling.

## Try it

Run the small-model validation loop end to end. Load `Qwen/Qwen2.5-0.5B-Instruct`, take ~100 rows of
your Lesson 02 dataset, and train 2 epochs with `lr=2e-4`, `eval_strategy="steps"`, `eval_steps=20`,
and `report_to="none"` (or wandb if you have it). Do three things: (1) capture greedy generations on 5
fixed held-out prompts *before* training and *after* — do the outputs move toward your target
behavior? (2) Plot or print train vs eval loss and identify whether you overfit — then deliberately
crank epochs to 10 and watch eval loss turn upward to *see* overfitting happen. (3) Break the run on
purpose: set `lr=5e-3` and confirm the loss spikes/diverges so you recognize the signature. You'll
finish able to read a loss curve and diagnose a run without re-reading this lesson.
