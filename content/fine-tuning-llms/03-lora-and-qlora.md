# 03 — LoRA and QLoRA: Fine-Tuning on One GPU

Full fine-tuning updates every weight in the model. For a 7B model in bf16 that means holding the
weights (~14 GB), the gradients (another ~14 GB), and the optimizer state — Adam keeps two moments
per parameter, so ~56 GB more — plus activations. You're north of 80 GB before a single training
token, which is why full fine-tuning of even a "small" model needs a multi-GPU server. **LoRA** and
**QLoRA** are the techniques that collapse that to a single consumer 24 GB card, and they're how
essentially all applied fine-tuning is done in 2026. This lesson is the mechanism, the knobs, and
the memory math so you can size a run before you launch it.

## The idea: don't move the weights, add a small correction

**LoRA (Low-Rank Adaptation)** freezes the entire pretrained model and injects a small, trainable
correction into chosen weight matrices. For a frozen weight matrix `W` of shape `d × k`, instead of
learning an update `ΔW` of the same (huge) size, you learn a **low-rank factorization** of it:

$$
W' = W + \Delta W = W + \frac{\alpha}{r} \, B A
$$

where `A` is `r × k`, `B` is `d × r`, and the **rank** `r` is tiny — typically 8, 16, or 32. `A` and
`B` are the only trainable parameters; `W` never changes. During the forward pass the layer computes
`W x + (α/r)·B(A x)`.

This is the rank story from linear algebra made useful. A full `ΔW` has up to `min(d, k)` independent
directions; the low-rank product `BA` can only express `r` of them. The bet LoRA makes — and it holds
empirically — is that the *update* a fine-tune needs is intrinsically low-rank. You're nudging an
already-capable model toward a behavior, not rebuilding it, and that nudge lives in a small subspace.
So you pay for `r·(d + k)` parameters instead of `d·k`. For a 4096×4096 attention projection at
`r=16`, that's ~131K trainable params instead of ~16.8M — a ~128× reduction on that matrix, and it
compounds across every targeted layer.

Because only `A` and `B` train, gradients and optimizer state exist *only* for them. That's where the
memory savings come from: the 56 GB of Adam state for a full 7B fine-tune becomes a few hundred MB.

## QLoRA: 4-bit base + LoRA on top

**QLoRA** goes one step further. The frozen base weights are the biggest memory consumer (14 GB for
7B in bf16). QLoRA **quantizes the frozen base to 4-bit** (NF4 via `bitsandbytes`), cutting that to
~3.5 GB, while keeping the trainable LoRA adapters in bf16. Forward and backward passes dequantize
weights on the fly to bf16 for the matmul, so you get 4-bit *storage* with near-full-precision *compute*.

The result: a 7B model fine-tunes comfortably in **~8 GB of VRAM**; a 70B fits on a single 48 GB
A100. This is the single change that moved fine-tuning from "rent a cluster" to "run it on the GPU
you have." The base is frozen so its quantization error doesn't accumulate — the trainable adapters
learn in full precision on top of a slightly-lossy but fixed foundation.

Note the important asymmetry (Lesson 07): you quantize the *frozen base during training* with NF4,
but you do **not** merge an adapter into a 4-bit base afterward — that's a different, lossier
operation. Keep the distinction clear.

## The config, in real code

Two stacks dominate. **PEFT + TRL** is the standard, most portable path:

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit base (QLoRA). Drop this block for plain LoRA on a bf16 base.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # quantizes the quantization constants too
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb, dtype=torch.bfloat16,
)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",     # or an explicit list, see below
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(model=model, args=SFTConfig(output_dir="out"),
                     train_dataset=train_ds, peft_config=lora_cfg)
```

**Unsloth** is the faster path — custom Triton kernels give ~2× speed and up to ~70% less VRAM, with
a HuggingFace-compatible API you plug straight into the same `SFTTrainer`:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-7B-Instruct", max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)
# → same SFTTrainer(model=model, ...) call
```

Reach for Unsloth on a single consumer GPU where speed and memory matter; PEFT+TRL when you want the
most portable, best-documented path or multi-GPU; and **Axolotl** when you want the whole run in a
version-controlled YAML file instead of Python (a team-reproducibility choice, same LoRA underneath).

## Choosing rank, alpha, and target modules

- **rank `r`** — the capacity of the adapter. `r=8`–`16` is the sweet spot for most tasks; `32`–`64`
  for harder adaptation or larger, more diverse datasets. Higher `r` = more capacity *and* more
  overfitting risk and memory. Start at 16.
- **`lora_alpha`** — a scaling factor; the update is scaled by `α/r`. The common heuristic is
  **`α = r` or `α = 2r`**. Alpha and rank interact: doubling `r` while keeping `α` fixed halves the
  effective update scale. The practical advice is to fix the `α/r` ratio (e.g. always `α = 2r`) and
  tune `r` alone, so changing capacity doesn't silently change your effective learning rate.
- **`target_modules`** — which layers get adapters. The modern default is **`"all-linear"`** (every
  linear layer: attention `q/k/v/o` *and* the MLP `gate/up/down`), which the QLoRA paper showed
  matters more than raising rank. The minimal, cheapest choice is attention-only
  (`["q_proj","v_proj"]`) — fine for light style adaptation, but leaves capability on the table for
  harder tasks. When in doubt, target all linear layers.
- **`lora_dropout`** — `0.05`–`0.1` for regularization on small datasets.
- **DoRA** (weight-decomposed LoRA) is a drop-in variant (`use_dora=True`) that often squeezes out a
  bit more quality at a small speed cost; try it if plain LoRA plateaus.

## The memory math you should do before launching

A back-of-envelope for a 7B model tells you which GPU you need:

| Component | Full FT (bf16) | LoRA (bf16 base) | QLoRA (4-bit base) |
|---|---|---|---|
| Base weights | ~14 GB | ~14 GB | ~3.5 GB |
| Gradients | ~14 GB | ~0 (frozen) | ~0 (frozen) |
| Optimizer (Adam, 2 moments) | ~56 GB | tiny (adapters only) | tiny (adapters only) |
| Adapter params + their grad/opt | — | ~0.3 GB | ~0.3 GB |
| **Rough total (before activations)** | **~84 GB** | **~15 GB** | **~5 GB** |

Activations add on top and scale with `batch_size × sequence_length`; **gradient checkpointing**
(on by default in `SFTConfig`) trades compute to shrink them. The headline: full FT of 7B needs a
big multi-GPU box; LoRA fits a 24 GB card; QLoRA fits an 8 GB card. That table is why the entire
applied field standardized on parameter-efficient fine-tuning.

## A worked parameter count

Make the savings concrete. Take a 7B model with hidden size `d = 4096`; each attention block has four
projections (`q, k, v, o`), each a `4096 × 4096` matrix, plus three MLP projections (`gate, up, down`)
around `4096 × 11008`. Across ~28 layers, targeting all linear layers, that's on the order of 6.5
billion weights — the "7B."

A full fine-tune trains all ~6.5B. A LoRA adapter at `r = 16` adds, per targeted matrix of shape
`d × k`, exactly `r·(d + k)` parameters. For one `4096 × 4096` attention projection that's
`16 · (4096 + 4096) ≈ 131K` — versus `16.8M` in the full matrix, a ~128× cut. Summed over every
targeted matrix in the model, the adapter lands around **20–40M trainable parameters — roughly
0.3–0.6% of the model.** PEFT prints this for you (`print_trainable_parameters()`), and seeing "0.4%
trainable" is the moment the memory table below stops being abstract: you are optimizing a rounding
error's worth of parameters, which is why the optimizer state (the real memory hog in full FT) all
but vanishes.

Raising `r` scales that trainable count linearly — `r = 64` is ~4× the adapter of `r = 16` — while
the frozen base memory doesn't move at all. That decoupling (capacity knob separate from base cost)
is the practical heart of PEFT.

## When full fine-tuning still wins

LoRA is right ~95% of the time, but not always. Full fine-tuning can edge it out when you're adapting
*heavily* (a large domain shift, a lot of data, teaching substantial new behavior) and you have the
hardware — the low-rank constraint is a real ceiling if the needed update genuinely isn't low-rank.
For the applied practitioner on realistic data and budgets, that regime is rare; start with QLoRA,
and only reach for full FT if a well-run LoRA sweep plateaus below your target.

## Key takeaways

- **LoRA** freezes the base and trains a low-rank update `ΔW = (α/r)·BA`; only `A` and `B` train, so
  gradients and optimizer state shrink dramatically. It bets the needed update is intrinsically
  low-rank — and it usually is.
- **QLoRA** = 4-bit NF4 frozen base (`bitsandbytes`) + bf16 LoRA adapters, dequantized on the fly for
  compute. A 7B fine-tunes in ~8 GB.
- Defaults that work: **`r=16`, `α=2r`, `target_modules="all-linear"`, dropout 0.05.** Fix the `α/r`
  ratio and tune `r` alone. Targeting all linear layers beats just raising rank.
- Do the **memory math** before launching: full FT ~84 GB vs LoRA ~15 GB vs QLoRA ~5 GB for 7B
  (before activations). Gradient checkpointing shrinks activations further.
- **Unsloth** for speed/memory on one GPU, **PEFT+TRL** for portability, **Axolotl** for
  YAML-reproducible runs. Full FT only for heavy adaptation with the hardware to match.

## Try it

Instrument the memory difference yourself on a small model (`Qwen/Qwen2.5-0.5B-Instruct` runs
anywhere). Load it three ways and, for each, count trainable parameters and peak memory: (1) plain,
all params trainable (full FT); (2) with a `LoraConfig(r=16, target_modules="all-linear")`; (3) with
`BitsAndBytesConfig(load_in_4bit=True)` + the same LoRA. Use
`model.print_trainable_parameters()` (PEFT prints trainable / total and the percentage) and, on CUDA,
`torch.cuda.max_memory_allocated()`. Confirm LoRA trains well under ~1% of parameters, and that the
4-bit base slashes the resident weight memory. Then sweep `r ∈ {4, 16, 64}` and watch trainable-param
count scale linearly with `r` while base memory stays flat — the concrete picture of why the whole
field fine-tunes this way.
