# 02 — Resource Accounting: FLOPs, Memory, and Precision

Before you write a single line of the model you should be able to answer three questions for any
configuration on paper: how many floating-point operations a forward and backward pass cost, how
much memory the whole training step needs, and how those change when you switch precision. If you
can do this arithmetic, you can predict whether a run will fit and roughly how long it will take,
and you will stop being surprised by out-of-memory errors.

## The one rule everything is built on

For a Transformer, the compute is dominated by matrix multiplications. The core method is
almost mechanical: write down every matmul in the forward pass, then convert each to FLOPs with a
single rule.

> **Matmul rule.** Given `A ∈ ℝ^{m×n}` and `B ∈ ℝ^{n×p}`, the product `AB` costs `2·m·n·p` FLOPs.

Why the 2: each output entry `(AB)[i,j]` is a dot product of length `n`, which is `n` multiplies
and `n` adds (`2n` FLOPs), and there are `m·p` output entries, so `(2n)(mp) = 2mnp`. A single
multiply-accumulate is two FLOPs; hold onto that, it is the source of every factor of 2 below.

## Per-component FLOP table

Take one forward pass over a batch of `B` sequences of length `L`, with model dimension `d`
(`d_model`), feed-forward dimension `d_ff`, `n` layers (`n_layers`), and vocabulary `V`. Let
`T = B·L` be the total number of tokens in the batch. Every parameterized operation is a matmul,
so we apply the rule to each. (SwiGLU has three FFN weight matrices `W1, W3 ∈ ℝ^{d_ff×d}` and
`W2 ∈ ℝ^{d×d_ff}`; standard multi-head attention has combined QKV of `3d×d` and an output
projection of `d×d`.)

| Component (per layer, unless noted) | Matmul shape | Forward FLOPs |
|---|---|---|
| Token embedding lookup | (gather, not a matmul) | ~0 |
| Q, K, V projections | `(T×d)·(d×3d)` | `2 · T · d · 3d = 6 T d²` |
| Attention scores `QKᵀ` | `(L×d_head)·(d_head×L)` per head | `2 · B · L² · d` |
| Attention over values `·V` | `(L×L)·(L×d_head)` per head | `2 · B · L² · d` |
| Output projection | `(T×d)·(d×d)` | `2 T d²` |
| FFN up + gate (`W1`, `W3`) | `(T×d)·(d×d_ff)` twice | `2 · 2 T d d_ff = 4 T d d_ff` |
| FFN down (`W2`) | `(T×d_ff)·(d_ff×d)` | `2 T d_ff d` |
| **Per-layer subtotal (params)** | | `8 T d² + 6 T d d_ff` |
| **Per-layer subtotal (attention scores+values)** | | `4 B L² d` |
| Final LM head (once, not per layer) | `(T×d)·(d×V)` | `2 T d V` |

Summing the parameterized terms over `n` layers plus the head gives the forward-pass matmul cost:

```
forward_flops ≈ n · (8 T d² + 6 T d d_ff)      # attention proj + FFN, all params
              + n · (4 B L² d)                  # attention scores and value-weighting (no params)
              + 2 T d V                          # LM head
```

With the canonical SwiGLU sizing `d_ff = (8/3)d`, the FFN term `6 T d d_ff` becomes `16 T d²`, so
each layer's parameterized cost is `8 T d² + 16 T d² = 24 T d²`. Note this is the FLOP twin of the
parameter count we derive when we go through the architecture: the transformer body is ~`12 n d²`
parameters, and `2 × 12 n d² = 24 n d²` FLOPs per token — exactly the `2N` rule, arrived at
component by component.

## The 2N and 6N rules

Every parameter participates in roughly one multiply-accumulate per token, and a multiply-add is
2 FLOPs, so:

> A forward pass through a dense Transformer costs about `2·N` FLOPs per token, where `N` is the
> non-embedding parameter count. A 1B-parameter model is ~2 GFLOP per token forward.

The backward pass costs about twice the forward, because you compute gradients with respect to
both the activations (to keep propagating) and the weights. State it as `2×` forward plus
`4×` backward:

> Training costs about `6·N` FLOPs per token. Forward is `2N`, backward is `4N`.

That `6N` is the number you carry around. Total training compute is `C ≈ 6·N·D`, where `D` is the
number of tokens trained on. This is the equation the entire scaling-laws module is built on, so
internalize it now.

One more accounting item worth doing: **AdamW's own FLOPs**. The optimizer step is a handful
of elementwise operations per parameter (decay, two moment updates, the update itself), so it is
`O(N)` per step — negligible next to the `6ND` of the matmuls, but you should be able to say why.

## What the 2N rule ignores, and when it bites

The `2N`/`6N` rule counts the parameterized matmuls: attention projections, FFN, and the head.
It ignores attention's **score computation**, which is not parameterized but still costs FLOPs:
from the table, computing `QKᵀ` and applying softmax weights to `V` together cost `4 B L² d` per
layer. For short sequences relative to model size this is negligible and the rule holds. For long
context it stops being negligible: the parameter term grows linearly in `L` (through `T = B·L`)
while attention grows with `L²`, so past some context length the quadratic term dominates and your
effective cost per token climbs.

Make this concrete with GPT-2 XL (`d=1600, n=48`): at `L=1024` the attention-score term is
a small slice of total FLOPs and the FFN dominates, but push the context to `L=16384` and the
`L²` term balloons until it rivals or exceeds the parameterized cost. This is the entire reason
long context is expensive and why FlashAttention and its relatives matter.

## A worked numeric example: GPT-2 XL

Our reference config, using the modern (SwiGLU, no-bias) architecture:

```
vocab_size V     = 50,257
context L        = 1,024
n_layers n       = 48
d_model d        = 1,600
n_heads          = 25      (d_head = 64)
d_ff             = 4,288   (nearest multiple of 64 to (8/3)·1600)
```

Forward FLOPs for one sequence (`B=1`, `T=L=1024`), by the table:

```
attn proj  per layer = 8 · 1024 · 1600²          ≈ 3.36e10
FFN        per layer = 6 · 1024 · 1600 · 4288     ≈ 4.21e10
attn score per layer = 4 · 1 · 1024² · 1600       ≈ 6.71e9
per layer  total     ≈ 8.24e10
× 48 layers          ≈ 3.96e12
LM head              = 2 · 1024 · 1600 · 50257    ≈ 1.65e11
forward total        ≈ 4.12e12  FLOPs  (~4.1 TFLOP for one 1024-token sequence)
```

Two things fall out immediately. The FFN is the single largest parameterized consumer per layer
(a direct consequence of `d_ff > d`), and the attention-score term is only ~8% here — but rerun it
at `L=16384` and that 8% becomes the dominant cost. The LM head, a single big `d×V` matmul, is
also not free: at ~1.65e11 it is about 4% of the forward pass and grows with vocabulary.

Now put a training budget on it. At 50% MFU on one H100 (~495 TFLOP/s effective for the TF32-class
matmuls we assume here), training GPT-2 XL for 400k steps at batch size 1024 with the `6N` (here
`3×` forward) rule takes on the order of a thousand-plus GPU-hours — you compute it as
`(6 · forward_per_token · tokens) / (0.5 · peak)`. The point of the exercise is that you can price
a run before launching it.

## Memory: the four consumers during training

Training memory is not just the weights. The full accounting counts all four:

1. **Parameters.** `N` numbers.
2. **Gradients.** One per parameter, another `N`.
3. **Optimizer state.** AdamW keeps two moments per parameter (first and second), so `2N` more.
   This is why AdamW training needs so much more memory than the model itself.
4. **Activations.** Everything saved on the forward pass for use in the backward pass. This scales
   with batch size and sequence length, not just parameter count, and is often the largest and
   most variable consumer.

Memory of any tensor is exact: `bytes = numel · element_size` (`tensor.numel() *
tensor.element_size()`). In plain fp32, consumers 1–3 cost `4 · (N + N + 2N) = 16N` bytes. A 1B
model is 16 GB before you store a single activation. This is why nobody trains large models in
pure fp32 and why you must understand mixed precision.

## Activation memory, with a formula

Activations are the part the `18N`-style rules leave vague, so here is the shape of it. Per
transformer layer you store the inputs to each matmul and the attention intermediates. The
dominant, unavoidable terms scale as:

```
activation_bytes ≈ bytes_per_elt · n_layers · B · L · (c₁ · d + c₂ · L · n_heads)
```

The `c₁·d` part is the linear stack of stored activations (residual stream, FFN hidden of width
`d_ff`, the QKV/output projection inputs) — all `O(d)` per token. The `c₂·L·n_heads` part is the
attention score/probability matrix, which is `L × L` **per head per layer** and is the term that
blows up with sequence length. Two takeaways: activation memory is linear in `B·L·n_layers`, and
the attention-map term is quadratic in `L`, mirroring the FLOP story. You do not memorize `c₁`,
`c₂`; you measure activation memory for your config and use this to reason about which lever moves
it.

## Tensor and dtype accounting

The formats you need to know, with their bit layouts:

- **fp32**: 1 sign / 8 exponent / 23 mantissa, 4 bytes. The reference. Wide range, high precision.
- **fp16**: 1/5/10, 2 bytes. Only 5 exponent bits, so its dynamic range is small — it underflows
  small numbers to zero and overflows large ones. `torch.tensor([1e-8], dtype=torch.float16)`
  becomes exactly `0`. Precise within its range, but the range is the problem.
- **bf16**: 1/8/7, 2 bytes. Same 8 exponent bits as fp32, so the **same wide range**, but only 7
  mantissa bits (less precision). `torch.tensor([1e-8], dtype=torch.bfloat16)` preserves the
  value. This is what almost everyone trains in now, because range matters more than mantissa
  precision for stable training and bf16 rarely needs loss scaling.
- **fp8**: standardized in 2022 for ML. H100 supports two variants: **E4M3** (range ≈ [−448, 448])
  and **E5M2** (range ≈ [−57344, 57344]). Used for the matmuls on the newest hardware with careful
  per-tensor scaling.

## Mixed precision, concretely

The idea: do the expensive matmuls in a low-precision format (bf16 or fp8) for speed and memory,
but keep a high-precision master copy of the weights and do the accumulation and optimizer step in
fp32 so you do not lose the model to rounding. The one-liner: **use bf16/fp8 for the forward
pass (activations); use fp32 for the rest (a master copy of parameters, gradients, optimizer
state).**

So the memory picture for bf16 mixed-precision AdamW is roughly:

```
params (bf16)         = 2N
master params (fp32)  = 4N
gradients (fp32)      = 4N
adamw m, v  (fp32)    = 8N
--------------------------------
fixed per-model       ≈ 18N bytes  (order of magnitude; exact bytes depend on your setup)

activations           = f(B, L, n_layers, d) — measure it, do not guess
```

The lesson is that "mixed precision" does **not** halve everything: the fp32 master and fp32
optimizer state persist, and they are the bulk of the fixed cost. fp16 additionally needs **loss
scaling** — multiply the loss by a large constant before backward so small gradients do not
underflow in fp16, then divide it back out before the optimizer step. bf16 usually skips this
because its exponent range already covers the gradients. If you have ever seen a run go NaN and
someone mutter about loss scaling, this is what they meant.

## Activation memory and how to cut it

Activation memory is where you have the most control. Three levers:

1. **Batch size and sequence length.** Activations scale linearly with both (and quadratically
   with `L` through the attention map). Halving batch size halves the linear part — at the cost of
   noisier gradients, which you fix with gradient accumulation.
2. **Gradient (activation) checkpointing.** Instead of saving every intermediate for the backward
   pass, save only a few (say, layer boundaries) and recompute the rest during backward. This
   trades compute for memory: you pay roughly one extra forward pass (training goes from `6N` to
   about `8N` per token) in exchange for a large drop in activation memory. Almost always worth it
   for large models.
3. **Precision.** bf16 activations are half the size of fp32.

## Roofline and arithmetic intensity

Knowing FLOPs is half the story. The other half is whether the hardware can feed the compute units
fast enough, which depends on **memory bandwidth**, not FLOPs. Every kernel has an *arithmetic
intensity*: FLOPs performed per byte moved between HBM and the compute units. Plot achievable
FLOP/s against arithmetic intensity and you get the roofline: below a threshold intensity you are
**memory-bound** (a sloped ceiling set by bandwidth), above it you are **compute-bound** (a flat
ceiling set by peak FLOP/s).

Big dense matmuls have high arithmetic intensity and sit in the compute-bound region — this is why
you want large batches and large matrices, and why MFU is high when matmuls dominate. Attention's
softmax, elementwise ops, and small matmuls have low intensity and are memory-bound; this is
exactly the regime FlashAttention targets by keeping the `L×L` scores in fast on-chip memory
instead of round-tripping them to HBM.

The single summary number is **MFU (Model FLOPs Utilization)**: `MFU = (actual FLOP/s) / (peak
FLOP/s)`, ignoring communication and overhead. An A100 peaks around 312 TFLOP/s (bf16/fp16), an
H100 around 990 TFLOP/s (bf16 dense) or ~495 for the TF32-class path we use in the estimate above.
`MFU ≥ 0.5` is good, and it climbs when large matmuls dominate the run. FLOP counts give the lower
bound on time; roofline and MFU tell you how close you get to it.

## Key takeaways

Every parameterized op is a matmul, and a matmul is `2mnp` FLOPs. Component by component this
gives ~`24 T d²` per layer with SwiGLU sizing, i.e. the `2N`/`6N` rule (forward `2N`, backward
`4N`, training `6ND`). Attention scores add a `4 B L² d` term that is small at short context and
dominant at long context — the same `L²` that dominates activation memory. Training memory has
four consumers, and AdamW's optimizer state alone is `2N` parameters; mixed precision keeps an
fp32 master and fp32 optimizer state, so it does not halve total memory. bf16 buys fp32's dynamic
range at half the bytes and usually skips loss scaling; fp16 does not. Activation memory is your
main lever (batch, sequence, checkpointing, precision). Finally, FLOPs set a floor on time, but
whether you hit it is a roofline question — arithmetic intensity decides memory-bound vs
compute-bound, and MFU is the one number that scores it.
