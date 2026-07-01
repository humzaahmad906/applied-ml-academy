# 03 — The Modern Transformer, Block by Block

You are going to implement a decoder-only Transformer from primitives, so you need to know what
every piece does and why the current default choices beat the 2017 originals. This chapter walks
the forward pass of one decoder in order, gives the exact math for each component, then covers the
hyperparameter choices that decide model shape.

## The shape of the thing

A decoder-only language model is a stack of identical blocks between an input embedding and an
output projection. Tokens come in as integer ids `(batch, seq_len)`, get looked up in an
embedding table of shape `(vocab_size, d_model)` to become vectors of size `d` (the model
dimension), pass through `num_layers` identical blocks that preserve the shape
`(batch, seq_len, d_model)`, hit a final norm, and are projected back to vocabulary logits of
shape `(batch, seq_len, vocab_size)`. The model is trained to predict the next token, so at every
position it outputs a distribution over the vocabulary for the token that follows.

The core trick that makes this a language model rather than just a stack of MLPs is causal
attention: each position may attend to earlier positions but not later ones, so the prediction at
position `i` depends only on tokens `1..i`. This lets you train on every position of a sequence
in parallel with a single forward pass while still respecting the left-to-right structure.

## One block, in order

A modern **pre-norm** decoder block is two sublayers, each wrapped in a residual connection with
a normalization applied *inside* the branch:

```
y = x + Attention(RMSNorm(x))
z = y + FFN(RMSNorm(y))
```

Note the norm is inside the residual branch (pre-norm), not after the addition (post-norm). The
residual stream `x` itself is never normalized as it flows down the stack — each sublayer reads a
normalized copy, computes an update, and adds it back. This matters and I will come back to it.

### Normalization: RMSNorm, not LayerNorm

The original Transformer used LayerNorm, which subtracts the mean and divides by the standard
deviation of each vector, then applies a learned scale and bias. The current default is RMSNorm
(Zhang & Sennrich, eq. 4), which drops the mean-subtraction and the bias and just divides by the
root-mean-square of the vector, then applies a learned per-dimension gain `g`:

```
RMSNorm(a_i) = a_i / RMS(a) · g_i        where   RMS(a) = sqrt( (1/d) · Σ_i a_i²  +  ε )
```

`g ∈ ℝ^d` is a learned gain (one parameter per dimension, no bias), and `ε` is a small constant,
fixed at `1e-5` here. One implementation detail worth insisting on: **upcast the input
to fp32 before squaring** and downcast the result back to the original dtype afterward, because
squaring bf16 activations can overflow or lose precision:

```python
def forward(self, x):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return (x / rms * self.g).to(in_dtype)
```

RMSNorm works about as well as LayerNorm, has fewer operations, and fewer parameters. Almost
every recent model uses it (LLaMA, Qwen, Mistral). The reason it works without mean-centering is
not fully settled theoretically; empirically the re-centering turns out not to be necessary.

### Pre-norm vs post-norm

Pre-norm (norm inside the residual branch, as above) makes deep Transformers trainable without
careful warmup and learning-rate tuning, because the residual path stays a clean identity and
gradients flow through it unimpeded from the loss all the way to the embeddings. Post-norm (the
2017 original, `x = Norm(x + Sublayer(x))`) puts the norm after the addition, which sits directly
on the residual highway and can destabilize very deep networks. Every large model today is
pre-norm, plus an **extra final RMSNorm** before the output projection so the last activations are
properly scaled going into the head. Some models (certain Qwen and Gemma variants) add norms in
still more places for stability at scale; the base recipe is pre-norm sublayers plus a final norm.

### The feed-forward network: SwiGLU

Each block's second sublayer is a position-wise network that expands the dimension, applies a
nonlinearity, and contracts back. The original used two linear layers with a ReLU between them and
an inner width of `4d`. Modern models make two changes: a smoother activation and a **gate**.

The activation is SiLU (a.k.a. Swish), which is smooth at zero unlike ReLU:

```
SiLU(x) = x · σ(x) = x / (1 + e^{-x})
```

The gate is a Gated Linear Unit: the elementwise product of one linear projection (passed through
the activation) with a second, independent linear projection. Putting them together gives SwiGLU,
which we use with **no bias terms** (following PaLM and LLaMA):

```
FFN(x) = SwiGLU(x, W1, W2, W3) = W2 ( SiLU(W1 x) ⊙ W3 x )
```

with `x ∈ ℝ^d`, `W1, W3 ∈ ℝ^{d_ff × d}`, `W2 ∈ ℝ^{d × d_ff}`, and `⊙` elementwise product. There
are now **three** weight matrices (up, gate, down) instead of two.

Why `d_ff ≈ (8/3)·d`: the classic FFN has two `d×4d` matrices, i.e. `8d²` parameters. SwiGLU has
three matrices of size `d×d_ff`, i.e. `3·d·d_ff` parameters. Setting `3·d·d_ff = 8d²` gives
`d_ff = (8/3)d`, which keeps the parameter (and FLOP) budget matched to the old `4d` design while
adding the gate. In practice you round `d_ff` to a nearby multiple of 64 for hardware efficiency —
our GPT-2-XL-shaped config uses `d_ff = 4288` (the nearest multiple of 64 to `(8/3)·1600`),
and the small toy config uses `d_ff = 1344` (`(8/3)·512` rounded). SwiGLU consistently beats plain
ReLU MLPs at the same budget; Shazeer's paper famously "offers no explanation... we attribute
their success, as all else, to divine benevolence."

### Attention

Attention lets a position pull information from other positions. For each position you compute
three projections: a query `q`, a key `k`, and a value `v`. You split these into `h` heads of size
`d_head = d/h` and, within each head, apply scaled dot-product attention. First the primitive:

```
softmax(v)_i = exp(v_i - max_j v_j) / Σ_j exp(v_j - max_j v_j)
```

Subtracting the max is the numerical-stability trick — without it `exp` of a large score overflows
to `inf` and `inf/inf` is `NaN`. Then attention itself (Vaswani et al., eq. 11):

```
Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) V
```

with `Q ∈ ℝ^{n×d_k}`, `K ∈ ℝ^{m×d_k}`, `V ∈ ℝ^{m×d_v}`. The `1/sqrt(d_k)` keeps the dot products
from growing with dimension and saturating the softmax. In code, per head:

```python
scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)   # (L x L)
scores = scores.masked_fill(~causal_mask, float("-inf"))
weights = softmax(scores, dim=-1)
out = weights @ V                                    # (L x d_head)
```

Concatenate the heads and project back to `d` with the output matrix. The **causal mask** is what
makes it a decoder: a boolean mask `M ∈ {True, False}^{n×m}` where `True` at `(i, j)` means query
`i` may attend to key `j`; you set future positions to `False` (score `-inf`) so they get zero
weight after softmax. Query `i` attends only to keys `j ≤ i`.

### Positional information: RoPE

Attention as described is permutation-invariant: it has no idea what order the tokens are in. You
have to inject position. The original used additive sinusoidal or learned absolute embeddings. The
current default is **Rotary Position Embeddings (RoPE)**, which rotate `q` and `k` (never `v`) by
an angle that depends on absolute position, arranged so that the dot product between a query at
position `i` and a key at position `j` depends only on the relative offset `i − j`.

RoPE treats the `d_k`-dimensional query as `d_k/2` two-dimensional pairs and rotates each pair.
For a query `q^(i) = W_q x^(i)` at position `i`, it applies a block-diagonal rotation `R_i`, whose
`k`-th `2×2` block rotates the pair `(q_{2k-1}, q_{2k})` by angle

```
θ_{i,k} = i / Θ^{(2k-2)/d}          for k ∈ {1, ..., d/2}
```

```
R_i^k = [ cos θ_{i,k}   -sin θ_{i,k} ]
        [ sin θ_{i,k}    cos θ_{i,k} ]
```

`Θ` is the base frequency (`rope_theta`, conventionally `10000`). Different pairs `k` rotate at
geometrically spaced frequencies: low `k` rotate fast (fine-grained position), high `k` rotate
slowly (coarse position). You never build the full `d×d` matrix — you precompute `cos θ_{i,k}` and
`sin θ_{i,k}` into a buffer (registered non-persistent, since these are fixed, not learned), share
one RoPE module across all layers, and index the buffer by each token's position. RoPE adds **no
parameters**, applies to `q` and `k` *before* the score computation, extrapolates to longer
sequences better than learned absolute embeddings, and is the default in LLaMA, Qwen, and Mistral.

Long-context models often bump `Θ` or interpolate positions to stretch a model trained at, say, 4k
tokens out to 32k or beyond — a fine-tuning-time trick built on this same mechanism.

## Attention variants that save inference memory

Standard multi-head attention (MHA) gives each head its own keys and values. At inference you
cache all past keys and values (covered in the inference chapter), and that KV cache is
proportional to the number of heads. Two variants shrink it:

- **Multi-query attention (MQA):** all query heads share a single key head and a single value
  head. Cuts the KV cache by a factor of `h` but can lose quality.
- **Grouped-query attention (GQA):** a middle ground where groups of query heads share one key/
  value head. Most current models (LLaMA 2/3 70B, Qwen) use GQA with something like 8 query heads
  per KV head, recovering most of MHA's quality at a fraction of the KV cache. This is the default
  to reach for.

There is a family of even more aggressive variants (multi-head latent attention in DeepSeek, for
example) that compress the KV cache further with a low-rank projection. The theme is the same: the
KV cache is the inference bottleneck, and attention design is increasingly driven by shrinking it.

## Hyperparameters that set model shape

Given a target parameter count, you still choose the shape. The main knobs, with the heuristics
the field converges on:

- **Model dimension `d` and number of layers `num_layers`.** The two big ones. For a fixed budget,
  moderately deep and moderately wide beats extreme aspect ratios. A common heuristic is an aspect
  ratio `d / num_layers` around 100–200 for billion-parameter models — GPT-2 XL sits near
  `1600/48 ≈ 33` (older, deeper-and-narrower), while modern models push wider. Treat it as soft.
- **Number of heads `h` and head dimension `d_head`, with `h · d_head = d`.** Head dimension is
  almost always 64 or 128; our configs use `d_head = 64` (e.g. `d=1600`, `h=25`).
- **FFN hidden size `d_ff = (8/3)·d`, rounded to a multiple of 64.**
- **Vocabulary size**, from the tokenization chapter (our toy runs use 10k to keep the embedding cheap).
- **RoPE `Θ = 10000`.**

The parameter count of the transformer body (ignoring embeddings) is roughly `12 · num_layers ·
d²` per the standard block: attention projections are ~`4d²` (QKV + output) and the SwiGLU FFN
~`8d²` per layer with the `(8/3)d` hidden size. Add the embedding and output head (`2·V·d` if
untied, `V·d` if tied) to get the total. Being able to compute this in your head lets you hit a
target size without a spreadsheet — and it is exactly `N` in the `2N`/`6N` FLOP rule we counted
when accounting for resources.

### Optimization hyperparameters

The architecture pairs with a specific training recipe you should know by name:

- **Optimizer: AdamW**, decoupled weight decay. Betas are typically `(0.9, 0.999)` but LLM
  training often uses `(0.9, 0.95)`; `ε ≈ 1e-8`; a weight-decay rate `λ` you tune.
- **Learning-rate schedule: cosine annealing with linear warmup** (the LLaMA schedule). With
  current step `t`, peak `α_max`, floor `α_min`, warmup steps `T_w`, and final step `T_c`:

  ```
  t < T_w :            α_t = (t / T_w) · α_max                                  # linear warmup
  T_w ≤ t ≤ T_c :      α_t = α_min + ½(1 + cos(π · (t−T_w)/(T_c−T_w)))·(α_max−α_min)
  t > T_c :            α_t = α_min                                              # floor
  ```

  Warmup avoids the early instability of a large LR on a cold model; the cosine decay lets you
  take big steps early and settle into a minimum late. Set `T_c` to your total step count so decay
  finishes exactly at the end of training.
- **Gradient clipping.** After backward, compute the global gradient ℓ2-norm `‖g‖`; if it exceeds
  a max `M`, scale all gradients by `M / (‖g‖ + ε)` (with `ε ≈ 1e-6`). This caps the damage from
  the occasional pathological batch that would otherwise spike the loss.
- **Batch size and total tokens.** Our toy TinyStories run processes ~`3.3e8` tokens
  (`batch × steps × context_length`); larger models scale `D` per the `6ND` compute budget.

Folk wisdom worth verifying empirically: the best learning rate sits "at the edge of
stability" — just below the value at which training diverges.

## Weight tying, initialization, and the small stuff

- **Tied embeddings:** small models often share the input embedding matrix with the output
  projection to save the `V·d` parameters. Large models untie them. Tying helps most when the
  embedding is a large fraction of the model — exactly the small-model regime.
- **Initialization:** scale initial weights so activations neither blow up nor vanish through the
  stack; a common choice scales the residual-branch output projections down by `1/sqrt(2·num_layers)`
  so the residual-stream variance stays controlled as depth grows. Pre-norm is more forgiving here
  than post-norm, which is part of why it won.
- **No biases:** modern blocks drop the bias terms in the linear layers. They add parameters and
  do not help.

## Key takeaways

The modern decoder block is pre-norm RMSNorm, causal multi-head (really grouped-query) attention
with RoPE for position, and a SwiGLU FFN, all wrapped in residual connections, with a final
RMSNorm before the head. Each piece is a deliberate improvement over the 2017 original: RMSNorm
(divide by RMS, learned gain, no bias, fp32-upcast the square) over LayerNorm; pre-norm (norm
inside the branch, clean residual highway) over post-norm; RoPE (rotate `q`,`k` by `θ_{i,k} =
i/Θ^{(2k-2)/d}`, no parameters, relative position) over learned absolute; SwiGLU with `d_ff =
(8/3)d` over ReLU-4d; and GQA over MHA to shrink the inference KV cache. The body is about
`12·num_layers·d²` parameters — that is the `N` in the FLOP rules. Shape it with `d_head = 64`, an
aspect ratio in the low hundreds, and `d_ff` a multiple of 64; train it with AdamW, cosine LR with
warmup, and gradient clipping. You can now design and price a model to a target size by hand.
