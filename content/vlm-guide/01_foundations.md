# Foundations

Everything downstream is a variation on what's in this chapter. The goal here is that when a paper says "we use a decoder-only transformer with pre-norm RMSNorm, SwiGLU, and RoPE," you don't read it as jargon — you read it as a precise spec you could implement.

---

## 0. The math you actually need (and why)

You don't need a math degree. You need fluency in five things, because they recur literally everywhere:

- **Matrix multiplication and shapes.** 90% of reading a model is tracking tensor shapes. If you can keep `[batch, seq, dim]` straight in your head and know what each matmul does to those dims, you can follow any architecture. The single most useful habit: annotate every line of a forward pass with its output shape.
- **The dot product as similarity.** `q · k` is large when `q` and `k` point the same way. Attention, retrieval, contrastive learning — all of them are "dot product = relevance." Internalize this and three separate topics collapse into one.
- **Softmax.** `softmax(x)_i = exp(x_i) / Σ exp(x_j)`. Turns a vector of scores into a probability distribution. It's the bridge between "raw scores" and "weights that sum to 1." Appears in attention, in the output head, in routing.
- **The chain rule / gradients.** You don't need to derive backprop by hand, but you need to viscerally believe "loss is differentiable w.r.t. every weight, and we nudge each weight opposite its gradient." That's all training is.
- **Expectation / sampling.** RL post-training (covered in the LLM chapter) is "maximize expected reward." Decoding (also covered there) is "sample from a distribution." Comfort with `E[·]` and with "draw a sample from a categorical distribution" covers it.

Probability distributions over a vocabulary, log-likelihood, and KL divergence (`KL(p‖q)` = how far q is from p; appears as a regularizer in RLHF) round it out. That's the whole toolkit.

---

## 1. Tokenization — how text becomes integers

A model never sees characters or words. It sees integer IDs. Tokenization is the (lossy, consequential) map from a string to a sequence of integers.

**Why not just use words?** Vocabulary would be huge and you'd choke on rare/unseen words (OOV). **Why not characters?** Sequences get very long and the model wastes capacity learning to spell. The answer is **subword tokenization**: frequent words become single tokens, rare words get split into pieces.

The three schemes you'll see:

- **BPE (Byte-Pair Encoding).** Start with characters (or bytes). Repeatedly merge the most frequent adjacent pair into a new token. Stop at a target vocab size. GPT-2/3/4, Llama, most models use a byte-level BPE variant — operating on UTF-8 *bytes* means there are no OOV tokens ever, since worst case you fall back to single bytes.
- **WordPiece.** Like BPE but merges are chosen to maximize training-corpus likelihood rather than raw frequency. BERT-era.
- **Unigram / SentencePiece.** Start with a big vocab and *prune* down, keeping tokens that best explain the corpus under a unigram language model. SentencePiece is the *library/wrapper* (treats text as a raw stream, handles whitespace as a real token `▁`), commonly used with the Unigram algorithm. T5, Llama tokenizers.

**Things that bite you in practice** (and show up in papers):
- Numbers and code tokenize badly under naive BPE (e.g. "12345" might be three weird tokens), which is part of why arithmetic is hard for LLMs and why some models add digit-level handling.
- The same word with/without a leading space is a *different token*. This matters for prompting and for few-shot formatting.
- **Tokenizer = a fixed asset of a model.** You cannot mix a model's weights with a different tokenizer. When fine-tuning or converting to mobile (MNN/MLX), tokenizer mismatches are a classic source of silent garbage output.
- **Context length is measured in tokens, not words.** Rough English ratio: ~0.75 words/token, ~4 chars/token. Wildly different for code, other languages, and structured data.

The output of tokenization is a sequence of integer IDs `[t_1, ..., t_n]`, each in `[0, vocab_size)`. That's the only thing the model ingests.

---

## 2. Embeddings — integers become vectors

Each token ID indexes into an **embedding matrix** `E ∈ R^{vocab_size × d_model}`. Token `t_i` becomes the row `E[t_i]`, a `d_model`-dimensional vector. This is a lookup, not a matmul (though it's equivalent to multiplying a one-hot vector by `E`).

`d_model` (the "hidden size" or "model dimension") is *the* width of the model — the size of the vector that flows through every layer (the "residual stream"). Common values: 768 (small), 4096 (7–8B class), 8192+ (large).

Key facts:
- Embeddings are **learned** during pretraining. Semantically similar tokens end up with similar vectors (the famous "king − man + woman ≈ queen" property emerges naturally).
- Many models **tie weights**: the input embedding matrix `E` and the output projection (from final hidden state back to vocab logits) share parameters. Saves `vocab × d_model` params and tends to help.
- The embedding is *position-agnostic* on its own. "dog bites man" and "man bites dog" produce the same multiset of embeddings. Position information must be injected separately (§6).

**Token embeddings vs. sentence/retrieval embeddings:** Don't confuse the two. The embedding *matrix* above gives per-token vectors *inside* the model. A "text embedding" for retrieval (covered in the RAG chapter) is a single vector for a whole span, produced by an embedding *model* (often a separate encoder, or pooled hidden states). Same word, different object.

---

## 3. The Transformer — the whole thing in one pass

Introduced in *Attention Is All You Need* (Vaswani et al., 2017). The original was **encoder–decoder** (for translation). Three families descend from it:

- **Encoder-only** (BERT): bidirectional attention, good for understanding/classification/embeddings. Not generative.
- **Decoder-only** (GPT, Llama, Qwen, virtually all modern LLMs): causal (left-to-right) attention, trained as next-token predictors. **This is the one that matters most for everything that follows.**
- **Encoder–decoder** (T5, BART, original Transformer): encoder reads input bidirectionally, decoder generates while cross-attending to the encoder. Still used in some translation/summarization and some VLMs.

### The decoder-only forward pass, end to end

Given token IDs `[t_1, ..., t_n]`:

1. **Embed:** `x = E[t]` → shape `[n, d_model]`. Add/inject positional info (§6).
2. **Stack of N identical transformer blocks.** Each block does two things, each wrapped in a residual connection and a normalization:
   - **Attention sub-layer:** `x = x + Attention(Norm(x))`
   - **FFN sub-layer:** `x = x + FFN(Norm(x))`
   (This is **pre-norm**: normalize *before* the sub-layer. The original was post-norm; pre-norm is now standard because it trains far more stably at depth.)
3. **Final norm**, then **unembed/LM head:** project the final `[n, d_model]` hidden states to `[n, vocab_size]` logits via the output matrix (often tied to `E`).
4. **Softmax** over the vocab dimension gives, for each position, a probability distribution over the next token.

The **residual stream** (the `x` that gets added to repeatedly) is the central object. Every sub-layer *reads* from it and *writes* a delta back into it. Thinking of the model as "a shared communication channel (the residual stream) that each layer reads and edits" is the single most useful mental model for interpretability and for understanding architectural changes.

### What each block is *for*
- **Attention** = move information *between positions* (mixing across the sequence).
- **FFN** = process information *within each position* (mixing across the feature dimension / per-token computation). The FFN holds the bulk of the parameters and is where a lot of "knowledge" is stored.

That's the whole architecture. Modern models change the *details* of attention, FFN, normalization, and positional encoding — but never this skeleton.

---

## 4. Attention — the core mechanism, with the math

Attention lets each position pull in information from other positions, weighted by relevance.

For each token, produce three vectors via learned linear projections of the hidden state `x`:
- **Query** `Q = x W_Q` — "what am I looking for?"
- **Key** `K = x W_K` — "what do I offer as a match target?"
- **Value** `V = x W_V` — "what do I actually contribute if matched?"

Then:

```text
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

Step by step:
- `Q Kᵀ` → `[n, n]` matrix of raw scores; entry `(i,j)` = how much token `i` should attend to token `j` (it's a dot product = similarity).
- `/ √d_k` → scale by the square root of the key dimension. Without this, large `d_k` makes dot products huge, pushing softmax into saturated regions with tiny gradients. (This is *the* reason for the scaling factor — keep variance ~1.)
- `softmax(...)` row-wise → each row sums to 1; these are the **attention weights**.
- `× V` → weighted sum of value vectors. Each token's output is a blend of all tokens' values, weighted by relevance.

### Causal masking (the thing that makes it a language model)
In a decoder, token `i` must not see tokens `j > i` (the future), or training would be trivial cheating. So before the softmax, set scores for `j > i` to `−∞` (they become 0 after softmax). This **causal mask** is what makes autoregressive generation valid: the prediction for position `i` depends only on positions `≤ i`.

### Multi-Head Attention (MHA)
Instead of one attention with dimension `d_model`, split into `h` **heads**, each operating in dimension `d_model/h`, run in parallel, then concatenate and project. Why: different heads can specialize (one tracks syntax, one tracks coreference, one is an "induction head" copying repeated patterns, etc.). Each head has its own `W_Q, W_K, W_V`.

```text
head_i = Attention(x W_Q^i, x W_K^i, x W_V^i)
MHA(x) = Concat(head_1, ..., head_h) W_O
```

### Complexity — the central problem of the field
The `Q Kᵀ` matrix is `[n, n]`. So attention is **O(n²)** in compute and memory w.r.t. sequence length `n`. This quadratic cost is *the* bottleneck that drives a huge fraction of all research: long-context techniques, FlashAttention (in the inference chapter), linear attention and SSMs (in the LLM chapter), etc. Whenever a paper is about efficiency or long context, it's almost always attacking this `n²`.

### Cross-attention
Same mechanism, but `Q` comes from one sequence and `K, V` from another. Decoder attending to encoder (encoder–decoder models), or text tokens attending to image tokens (some VLMs, covered in the VLM chapter). "Self-attention" = Q,K,V all from the same sequence; "cross-attention" = Q from one, K/V from another.

---

## 5. FFN, normalization, activations — the rest of the block

### Feed-Forward Network (FFN / MLP)
Applied independently to each position. Classic form: two linear layers with a nonlinearity:
```text
FFN(x) = W_2 · σ(W_1 · x)
```
The hidden dimension `d_ff` is typically **4× `d_model`** (an expand-then-contract). This is where most parameters live in a dense model.

**SwiGLU** (now standard, from Llama onward) replaces the simple MLP with a *gated* variant:
```text
FFN(x) = W_2 · ( SiLU(W_1 · x) ⊙ (W_3 · x) )
```
Three matrices instead of two; one branch acts as a learned gate (element-wise multiply `⊙`) on the other. Empirically better per-parameter. (To keep param count constant, `d_ff` is scaled down, often to ~⅔·4·`d_model`.) When a paper says "SwiGLU FFN," this is it.

### Normalization
Stabilizes activations so deep stacks train. Two you'll see:
- **LayerNorm:** subtract mean, divide by std (over the feature dim), then learned scale + shift. BERT/GPT-2 era.
- **RMSNorm** (now dominant): skip the mean-centering, just normalize by the root-mean-square and apply a learned scale. Cheaper, no bias term, works as well. Llama, Qwen, most modern models use RMSNorm.

**Pre-norm vs post-norm:** pre-norm (normalize the *input* to each sub-layer) is standard now because it gives a clean residual path and trains stably at 100+ layers. Post-norm (original) needs careful warmup and is fragile at depth.

### Activation functions
- **ReLU** `max(0, x)` — old reliable.
- **GELU** — smooth ReLU, BERT/GPT era.
- **SiLU/Swish** `x · sigmoid(x)` — used inside SwiGLU. Smooth, non-monotonic.
The trend is smooth, gated activations. Not a place where much hinges day-to-day, but you should recognize the names.

### Putting a modern block together
A 2024–2026-era decoder block, in words:
```text
h = x + MHA_or_GQA( RMSNorm(x) )      # with RoPE applied to Q,K
out = h + SwiGLU_FFN( RMSNorm(h) )    # or MoE FFN; see the LLM chapter
```
If you can read that line and picture every operation and tensor shape, you've got the foundation. Everything in the LLM and VLM chapters is a swap-out of one of these pieces.

---

## 6. Positional encoding — and RoPE in depth

Attention is **permutation-invariant**: scramble the tokens and (without positional info) you get the same result. So we must inject order. This has evolved a lot, and RoPE deserves real detail because it's in nearly every modern model and shows up constantly in long-context papers.

### The lineage
- **Sinusoidal (absolute):** original Transformer. Add fixed sine/cosine vectors (different frequencies per dimension) to embeddings. No parameters, but doesn't generalize well past trained lengths.
- **Learned absolute:** a trainable vector per position (BERT, GPT-2). Hard cap at max trained position; no extrapolation.
- **Relative position bias:** bias attention scores by `(i − j)`. T5 uses learned relative buckets; **ALiBi** adds a simple linear distance penalty (`−m·|i−j|`) directly to attention scores and extrapolates decently.
- **RoPE (Rotary Position Embedding):** the modern default. Used by Llama, Qwen, Mistral, DeepSeek, Gemma, and most others.

### RoPE — the actual idea
RoPE encodes position by **rotating** the query and key vectors by an angle proportional to their absolute position, *before* the dot product. The genius is that when you then take `q_m · k_n`, the result depends only on the *relative* offset `(m − n)`, not on absolute `m` and `n` separately.

Mechanically: split each head's vector into 2D pairs `(x_0,x_1), (x_2,x_3), ...`. For a token at position `m`, rotate pair `k` by angle `m·θ_k`, where the frequencies `θ_k = base^(−2k/d)` (base typically 10000) range from fast (early pairs) to slow (later pairs). Different pairs rotate at different rates — like clock hands of different speeds — so the pattern of rotations uniquely (within a range) encodes position, multi-scale.

The key algebraic property:
```text
⟨ RoPE(q, m), RoPE(k, n) ⟩  depends only on (m − n)
```
That's why RoPE simultaneously gives you *absolute* position injection and *relative* position behavior in attention scores, with **zero learned parameters**.

### Why RoPE matters so much in practice (and in papers)
- **It's applied to Q and K only, inside attention, per head** — not added to the residual stream. (A common confusion: RoPE is not an additive embedding.)
- **KV-cache friendly:** keys are rotated by their absolute position once and cached; new queries rotate by their position; the dot product still collapses to the relative offset. So cached keys stay valid — critical for fast generation (covered in the inference chapter).
- **Length extrapolation is the pain point.** A model trained to 4k tokens degrades past that because high-frequency rotations "wrap around" / alias at distances unseen in training. This spawned a whole sub-literature you'll meet constantly:
  - **Position Interpolation (PI):** linearly squeeze positions so a longer context fits in the trained rotation range.
  - **NTK-aware scaling:** change the rotation base instead of uniformly squeezing, preserving high-frequency resolution.
  - **YaRN:** a refined NTK/interpolation combo, the common way models advertise "extended to 128k context."
  When a paper says "we extend context from 32k to 128k via YaRN," it's manipulating RoPE frequencies, not retraining from scratch.
- **fp16 precision** on the trig terms can cause drift at long lengths; many implementations compute RoPE in fp32. Minor but a real gotcha.
- **2D/multimodal RoPE:** for images/video, position is 2D or 3D, so variants (axial RoPE, M-RoPE in Qwen-VL) apply rotations along multiple axes. The "MRoPE split bug" class of issues in mobile conversion lives here (covered in the VLM chapter).

If you understand RoPE — rotation in 2D subspaces, relative-from-absolute, frequency spectrum, and how context-extension tricks perturb it — you'll comprehend a startling fraction of architecture and long-context papers with no further effort.

---

## 7. Self-check: can you read this block spec?

> *"A 32-layer pre-norm decoder, `d_model=4096`, 32 query heads / 8 KV heads (GQA), RMSNorm, SwiGLU FFN with `d_ff≈14336`, RoPE base 500000, tied embeddings, 128k context via YaRN."*

If you can now picture: the residual stream of width 4096; 32 stacked blocks each doing normed grouped-query attention then a gated FFN; rotary position applied to Q/K with a high base for long context; the same matrix used for input embedding and output logits — then you have the foundation, and the LLM chapter is just filling in *why* each choice was made and what the alternatives are.
