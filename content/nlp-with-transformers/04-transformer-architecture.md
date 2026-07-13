# 04 — The Transformer Architecture

The transformer (Vaswani et al., 2017) won because it removed the sequential bottleneck. An RNN
processes token `t` only after token `t-1`, so it can't parallelize across a sequence and it has to
carry every long-range dependency through a single hidden state that keeps getting overwritten.
Attention replaces that recurrence with a single operation where every position looks at every other
position simultaneously — fully parallel on a GPU, and with a direct O(1)-hop path between any two
tokens regardless of distance. Everything modern — Llama 3/4, Qwen3, DeepSeek-V3, GPT-4o, Claude,
Gemini — is a stack of the same block with a handful of refinements. This module builds that block
from first principles, derives the parts that are actually derived (and flags the parts that are just
heuristics), and ends by counting the parameters of a small model so the numbers stop being abstract.

Tokens arrive as integers from the [tokenizer](03-tokenization.md), each becoming a `d`-dimensional
embedding vector (the same table idea as [word vectors](02-word-vectors.md), now learned jointly with
the model). From there it's attention and MLPs all the way up.

## Attention as a soft lookup

Think of a Python dict. You have a **query**, you compare it against every **key**, and you retrieve
the matching **value**. A hard lookup returns exactly one value. Attention is the *soft, differentiable*
version: compare the query to every key to get a similarity score, turn the scores into weights that
sum to 1, and return the weighted average of all values. Nothing is selected discretely, so gradients
flow.

Each token produces three vectors by multiplying its embedding by three learned matrices:

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

where `X` is the `n × d` sequence of embeddings. The query is "what am I looking for," the key is
"what do I offer as a match," the value is "what I actually pass along if matched." Similarity is the
dot product `q · k`, so the raw score matrix is `Q Kᵀ` (shape `n × n`). Softmax each row into weights,
multiply by `V`:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

The output at each position is a mix of all value vectors, weighted by how well that position's query
matched each key. That is the entire mechanism.

## Why divide by √d — the derivation

The `√d_k` scaling isn't cosmetic; it keeps training from stalling. Assume the components of a query
`q` and key `k` are independent, mean 0, variance 1. Their dot product is
`q · k = Σᵢ qᵢ kᵢ` over `d_k` terms. Each term `qᵢ kᵢ` has mean 0 and, for independent unit-variance
factors, variance 1. Summing `d_k` independent terms adds variances:

$$
\mathrm{Var}(q \cdot k) = \sum_{i=1}^{d_k} \mathrm{Var}(q_i k_i) = d_k
$$

So the dot product has standard deviation `√d_k`. For `d_k = 128` that's a spread of ~±11 before the
softmax. Softmax of logits with that magnitude is nearly a one-hot: it puts essentially all weight on
the single largest score, and its gradient in the saturated region is almost zero — attention stops
learning. Dividing by `√d_k` rescales the dot product back to unit variance, keeping the softmax in
its responsive range where gradients are healthy. That's the whole reason for the `√d_k`: normalize
the variance that dimension `d_k` injected.

## Multi-head attention

One attention operation can only represent one notion of "relevant." Real language needs several at
once — syntactic agreement, coreference, positional adjacency. **Multi-head attention** runs `h`
attention operations in parallel, each on its own `d/h`-dimensional slice, then concatenates and
projects:

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O,
\qquad \text{head}_i = \text{Attention}(X W_Q^i, X W_K^i, X W_V^i)
$$

Each head gets `d_k = d/h` dimensions, so total compute is roughly the same as one full-width head,
but the model gets `h` independent subspaces to specialize in. Empirically some heads learn crisp,
interpretable jobs — the induction heads and name-copying circuits you'll meet in
[interpretability](13-interpretability.md).

## Causal masking

A language model predicting token `t` must not see tokens `t+1…n` — that would be looking at the
answer. **Causal masking** enforces this by setting the pre-softmax scores for future positions to
`-∞` before the softmax, so their weights become exactly 0:

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (n, n)
mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float("-inf"))
attn = scores.softmax(dim=-1) @ V
```

Encoders (BERT-style) skip the mask — every token sees the whole sequence, which is why encoders are
for understanding, not generation. Decoders use the mask, which is why they generate left to right.

## Positional information: sinusoidal → learned → RoPE

Attention is a weighted average, and averages are **permutation-invariant**: shuffle the input tokens
and the outputs shuffle the same way but carry no signal about order. Yet "dog bites man" ≠ "man bites
dog." Position must be injected explicitly.

**Sinusoidal** (original transformer): add a fixed vector to each embedding, built from sines and
cosines at geometrically spaced frequencies. No parameters, extrapolates somewhat beyond training
length. **Learned absolute** (BERT, GPT-2): a trainable embedding per position — simple, but it can't
represent any position past the training length, so it hard-fails on longer inputs.

Both inject *absolute* position, but attention fundamentally cares about *relative* position — the
offset between a query and a key, not their absolute indices. **Rotary Position Embedding** (RoPE, Su
et al., 2021) encodes relative position directly and is what essentially every modern model uses.

### The RoPE rotation math

RoPE doesn't add anything. It **rotates** the query and key vectors by an angle proportional to their
absolute position. Take a 2D slice of the query at position `m`. Rotate it by angle `mθ`:

$$
R(m\theta) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
$$

Apply `R(mθ)` to the query at position `m` and `R(nθ)` to the key at position `n`. The attention score
is their dot product. The key algebraic fact about rotation matrices is `R(a)ᵀ R(b) = R(b − a)`, so:

$$
\big(R(m\theta)\,q\big)^\top \big(R(n\theta)\,k\big)
= q^\top R(m\theta)^\top R(n\theta)\, k
= q^\top R\big((n - m)\theta\big)\, k
$$

The absolute positions `m` and `n` **cancel**; the score depends only on the offset `n − m`. That's
the entire point: you applied an absolute rotation to each vector independently (cheap, no `n × n`
bias matrix), but the dot product that attention computes sees only the *relative* distance. The full
`d`-dimensional vector is split into `d/2` such 2D pairs, each rotated at its own frequency
`θᵢ = 10000^(−2i/d)` — high frequencies for fine local distinctions, low frequencies for long-range
position. Because RoPE is a rotation applied at score time, you can stretch its frequencies at
inference to extend context (position interpolation, YaRN) without retraining — covered in
[inference](12-inference-decoding.md).

## The residual stream and pre-norm

Wrap attention and the MLP each in a **residual connection**: `x + Sublayer(x)`. This is the single
most important structural choice for trainability — it gives gradients a direct path to every layer
and lets the network default to identity, so adding depth can't easily hurt. The modern mental model
(from interpretability) is the **residual stream**: a `d`-dimensional vector per token that flows
straight up the network, and each attention and MLP block *reads* from it and *writes* an additive
update back. Nothing is overwritten; information accumulates.

Where you put the LayerNorm matters. **Post-norm** (original) normalizes *after* the residual add:
`LayerNorm(x + Sublayer(x))`. This puts the normalization on the residual path and makes deep stacks
unstable — the original transformer needed a learning-rate warmup to train at all. **Pre-norm**
normalizes the *input* to each sublayer, keeping the residual path clean:

$$
x \leftarrow x + \text{Sublayer}\big(\text{Norm}(x)\big)
$$

Pre-norm trains stably without warmup tricks and is standard in every modern LLM. The one cost is that
the residual stream's magnitude grows with depth, usually handled with a final norm before the output.

## The FFN and SwiGLU

After attention mixes information *across* positions, a **feed-forward network** transforms each
position independently — this is where most of the model's parameters and much of its stored knowledge
live. The classic form expands to a wider inner dimension (typically `4d`) and back:
`FFN(x) = W₂ · ReLU(W₁ x)`. Modern models replace ReLU with a **gated** variant, **SwiGLU** (Shazeer,
2020):

$$
\text{SwiGLU}(x) = \big(\text{SiLU}(x W_1) \odot (x W_3)\big) W_2
$$

One projection is passed through SiLU and used to *gate* (elementwise-multiply) a second projection.
The gate lets the network modulate information flow per dimension, and it consistently beats plain
ReLU/GELU MLPs at equal compute. Because SwiGLU uses three matrices instead of two, the inner
dimension is usually shrunk to ~`8/3·d` to keep the parameter count matched. Llama and Qwen use
SwiGLU.

## Encoder, decoder, encoder-decoder

- **Encoder-only** (BERT, DeBERTa): bidirectional, no causal mask. Best for *understanding* — each
  token sees full left and right context, producing rich representations for classification, NER,
  retrieval embeddings. Can't generate. See [transfer learning](06-transfer-learning-tasks.md).
- **Decoder-only** (GPT, Llama, Qwen): causal mask, generates left to right. This architecture won for
  general-purpose LLMs — a single next-token objective scales, and the model can do any task framed as
  text continuation.
- **Encoder-decoder** (T5, BART): an encoder reads the input bidirectionally, a decoder generates while
  cross-attending to the encoder output. Natural fit for seq2seq (translation, summarization) where
  input and output are distinct sequences.

The field consolidated on decoder-only because it's the simplest thing that scales and unifies every
task — the "why decoder-only won" story is in [pretraining](05-pretraining.md).

## Complexity: the O(n²) tax

The score matrix `Q Kᵀ` is `n × n`, so attention is **O(n² · d)** in compute and **O(n²)** in memory
for the score matrix. Double the context and attention cost quadruples. This single fact drives most
of serving economics and long-context research: it's why 128K-token contexts are expensive, why
FlashAttention (which never materializes the full `n × n` matrix) matters, and why the KV cache
(covered in [inference](12-inference-decoding.md)) dominates memory at long context. The MLPs, by
contrast, are O(n · d²) — linear in sequence length — so at short context the MLP dominates cost and
at long context attention does.

## Modern refinements, one paragraph each

**RMSNorm** — LayerNorm subtracts the mean and divides by the standard deviation; RMSNorm (Zhang &
Sennrich, 2019) drops the mean-centering and just divides by the root-mean-square. It's cheaper, has
one fewer parameter, and works as well, so Llama/Qwen use it.

**GQA / MQA** — the KV cache stores keys and values for every past token for every head, and at long
context that memory dominates. **Multi-Query Attention** (Shazeer, 2019) shares *one* K/V head across
all query heads, shrinking the cache ~`h`×; **Grouped-Query Attention** (Ainslie et al., 2023) is the
middle ground — a few K/V heads shared among groups of query heads — trading a little quality for a
big cache saving. GQA is the modern default (Llama 3, Qwen3).

**Attention sinks** — models learn to dump a lot of attention weight onto the very first token(s),
apparently as a "no-op" place to park probability mass when nothing else is relevant. StreamingLLM (Xiao
et al., 2023) showed you must *keep* those sink tokens in the cache when you slide a long-context
window, or quality collapses — a practical gotcha for streaming inference.

## Parameter accounting for a small model

Make it concrete. A small decoder-only model with hidden size `d = 768`, `L = 12` layers, `h = 12`
heads, SwiGLU inner dim `≈ 2048`, vocab `V = 32000`. Per layer:

- **Attention** — four `d × d` projections (`Q, K, V, O`): `4 · 768² ≈ 2.36M`.
- **MLP (SwiGLU)** — three matrices between `768` and `2048`: `3 · 768 · 2048 ≈ 4.72M`.
- **Norms** — two RMSNorms, `2 · 768 ≈ 1.5K` (negligible).

Per layer ≈ `7.1M`, times 12 layers ≈ **85M** in the transformer blocks. Then the **embedding table**:
`V · d = 32000 · 768 ≈ 24.6M`, and the output projection is usually **tied** to it (shared weights),
so it's counted once. Total ≈ `85M + 24.6M ≈ 110M` — a GPT-2-base / BERT-base-scale model. Two things
to notice: the **MLP holds roughly twice the parameters of attention** per layer (`4.72M` vs `2.36M`),
which is why the FFN is where most knowledge lives; and for a *small* model the **embedding table is a
huge fraction** (~22%), which is exactly why vocabulary size is a real parameter-budget decision back
in [tokenization](03-tokenization.md). As models scale, the `L · d²` block terms grow faster than the
`V · d` embedding, so at 7B+ the embedding is a rounding error and the blocks dominate. You can now
sanity-check any config: FLOPs per token ≈ `6N` in training (the rule you'll use in
[pretraining](05-pretraining.md)), and `N` is what you just computed.

## What interviews ask here

- **Why divide by √d_k?** Dot product of two unit-variance `d_k`-vectors has variance `d_k`; unscaled,
  softmax saturates near one-hot and its gradient vanishes. Dividing by `√d_k` restores unit variance
  and healthy gradients.
- **Why does RoPE encode relative position?** It rotates Q and K by angles proportional to absolute
  position; since `R(mθ)ᵀR(nθ) = R((n−m)θ)`, the score depends only on the offset `n−m`.
- **Why decoder-only over encoder-decoder for LLMs?** Simplest architecture that scales, single
  next-token objective unifies all tasks; encoders can't generate, encoder-decoder adds complexity for
  seq2seq-specific gains.
- **What's the complexity of attention and why does it matter?** O(n²·d) compute and O(n²) memory in
  sequence length — drives long-context cost, FlashAttention, and KV-cache pressure.
- **Pre-norm vs post-norm?** Post-norm puts normalization on the residual path and is unstable in deep
  stacks (needs warmup); pre-norm normalizes each sublayer's input, keeps the residual clean, trains
  stably — the modern default.
- **What do GQA/MQA solve?** KV-cache memory at long context, by sharing K/V heads across query heads;
  GQA is the quality/memory sweet spot.

## Where this shows up on the job

- **Sizing and serving models.** You'll use parameter and FLOP accounting to predict memory, choose a
  GPU, and reason about why long context is expensive — the O(n²) tax and KV cache set your latency and
  cost budget.
- **Reading and modifying model code.** Every open-weight model is this block with small variations
  (RoPE flavor, GQA group count, SwiGLU dims); knowing the pieces lets you patch attention, extend
  context, or debug a custom architecture.
- **Debugging training instability.** Norm placement, residual scaling, and softmax saturation are the
  usual suspects when a run diverges or won't learn, and they're all in this module.
- **Interview design rounds.** "How would you extend this model to 128K context?" is answered with
  RoPE scaling, GQA, and attention-complexity reasoning — the vocabulary of this module.
