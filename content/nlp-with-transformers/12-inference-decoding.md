# 12 — Inference and Decoding: Sampling, KV Cache, and Speculative Decoding

Training gets the attention, but the model spends its whole deployed life doing inference, and that is
where the money goes. Two questions dominate the job: *how do you turn the model's next-token
distribution into text* (decoding), and *how do you serve that fast and cheap* (the systems). This
module owns the NLP-facing half — decoding strategies, the KV-cache memory math, speculative decoding,
long-context position scaling, and constrained decoding — and hands off the deep serving-systems
treatment (kernels, PagedAttention internals, parallelism) to [inference](../language-modeling/09-inference.md).
The framing throughout: every choice here is a knob on the accuracy/cost/latency triangle.

## Decoding as search over sequences

At each step the model gives you a probability distribution $P(x_t \mid x_{<t})$ over the vocabulary.
Generating text is choosing a token from it, appending, and repeating. You are approximately searching
for a high-probability *sequence*, and the exact argmax over all sequences is intractable
($V^T$ options), so every real method is a greedy or stochastic approximation. The strategies differ
in how much they exploit (pick likely tokens) versus explore (sample diverse ones).

### Greedy

Take $\arg\max$ every step. Deterministic, fastest, and correct-feeling for tasks with one right
answer — extraction, classification, short factual QA, code where you want the single best token. Its
failure mode on open-ended text is degeneration: it locks into repetitive loops ("the the the") and
bland phrasing because the single most-likely token at each step does not compose into a
high-probability *sequence*. Use it when there is one answer; avoid it for anything creative.

### Beam search

Keep the `b` highest-probability partial sequences at each step instead of one. It finds
higher-total-probability sequences than greedy and is the right call for tasks with a well-defined
correct output where you want to maximize likelihood: **machine translation, summarization, and other
seq2seq** tasks (the [transfer-learning tasks](06-transfer-learning-tasks.md) module). But beam search
**dies for open-ended generation**. Maximizing sequence probability there produces short, generic,
repetitive text — human language is deliberately *not* the highest-probability string (Holtzman et
al., 2019, "The Curious Case of Neural Text Degeneration"). Beam also multiplies cost by `b`. In 2026,
chat and creative generation use sampling; beam survives mainly in classic seq2seq and
constrained-output settings.

### Temperature

Reshape the distribution before sampling by dividing logits by $T$:

$$
P_T(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

$T \to 0$ approaches greedy (sharpens onto the top token); $T = 1$ is the raw distribution; $T > 1$
flattens it toward uniform (more random). Practical bands: **0 for deterministic/factual/code, 0.7 for
balanced chat, 1.0–1.3 for creative/brainstorming.** Temperature alone still samples from the long
tail, so it is almost always paired with a truncation method below.

### Top-k, top-p (nucleus), min-p

These truncate the tail so temperature does not occasionally emit garbage:

- **Top-k** — keep the `k` most probable tokens, renormalize, sample. Simple, but `k` is a fixed count
  that ignores the shape of the distribution: when the model is confident (one token at 0.95), a large
  `k` drags in junk; when it is uncertain (flat), a small `k` chops off valid options.
- **Top-p / nucleus** (Holtzman et al., 2019) — keep the smallest set of tokens whose cumulative
  probability exceeds `p` (e.g. 0.9), then sample. This adapts to the distribution's shape: few tokens
  when confident, many when uncertain. The standard default for chat.
- **Min-p** — keep tokens whose probability is at least `min_p × p_max` (a fraction of the top token's
  probability). It scales the threshold with model confidence and tends to preserve coherence better
  than top-p at high temperature, so it pairs well when you want *both* diversity and reliability.

### Repetition penalties

Independent of the above: down-weight tokens already generated (`repetition_penalty`, or
`no_repeat_ngram_size` to hard-block repeated n-grams). Useful for smaller/base models that loop;
overuse hurts fluency by penalizing legitimately common words. Reach for it only when you actually see
looping.

**Production defaults.** Deterministic tasks (extraction, classification, structured output):
`temperature=0`, greedy. Chat/assistant: `temperature≈0.7, top_p≈0.9` (or min-p ≈ 0.05–0.1). Creative:
`temperature≈1.0` with top-p or min-p. Pin these in config and log them with outputs — irreproducible
generations are almost always an unlogged sampling-param change.

## The KV cache: the memory that dominates serving

Attention at position `t` needs the keys and values of *all* previous positions. Recomputing them
every step is $O(t^2)$ over a generation; instead you **cache** each layer's K and V as you go and
reuse them — decode becomes $O(t)$ per step. The KV cache is the single most important object in LLM
serving, because its size, not the weights, is usually what caps how many requests you can batch.

The memory, per token, summed over the whole model:

$$
\text{KV bytes/token} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times \text{bytes}
$$

The `2` is K and V; `L` is layers; $n_{\text{kv}}$ is the number of key/value heads (fewer than query
heads under GQA — see [transformer architecture](04-transformer-architecture.md)); $d_{\text{head}}$
is head dimension; `bytes` is the dtype width (2 for fp16/bf16). Worked example — a Llama-3-8B-class
model (`L=32`, 8 KV heads, `d_head=128`, bf16):

$$
2 \times 32 \times 8 \times 128 \times 2 = 131072 \text{ bytes} \approx 128 \text{ KB/token}
$$

So a single 8K-token sequence holds ~1 GB of KV cache; batch 32 of them and that is ~32 GB — more than
the 16 GB of weights. This is why **GQA/MQA exist** (cut $n_{\text{kv}}$ to shrink the cache), why
**KV-cache quantization** (fp8/int8) is common, and why serving throughput is a memory-bandwidth game.
PagedAttention (vLLM) exists precisely to stop fragmentation from wasting this cache; the depth of that
is in [inference](../language-modeling/09-inference.md).

## Speculative decoding: draft-verify, and why it is lossless

Decode is *memory-bandwidth bound*: generating one token reads the entire weight matrix from HBM but
does little arithmetic, so the GPU sits idle. The wasteful part is that verifying `k` tokens in one
forward pass costs almost the same as generating one — the weights get read once either way.
Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) exploits this.

Mechanism: a small, cheap **draft** model proposes `k` tokens autoregressively. The big **target**
model then does *one* forward pass over all `k` proposals in parallel, yielding its own probability at
each position. You accept the longest prefix that passes an acceptance test and generate one bonus
token from the target's distribution at the first rejection. One expensive forward pass can thus emit
several tokens.

**Why it is exactly lossless** — this is the part interviewers push on. The acceptance test is
constructed so the accepted sequence is distributed *identically* to sampling from the target model
alone. For a draft-proposed token `x` with draft prob $q(x)$ and target prob $p(x)$: accept with
probability $\min(1, p(x)/q(x))$; on rejection, resample from the normalized residual distribution
$\propto \max(0, p(x) - q(x))$. This is a rejection-sampling scheme whose stationary distribution is
exactly $p$. The draft model's only job is to *guess*; every token is validated against the target, so
the output distribution is provably unchanged. The draft can be bad and you still get correct samples
— just fewer accepted tokens.

The speedup is set by the **acceptance rate** $\alpha$ (how often draft tokens survive). Expected
tokens per target pass with draft length `k` is $\tfrac{1 - \alpha^{k+1}}{1 - \alpha}$; at $\alpha=0.8,
k=4$ that is ~3.4 tokens per big-model pass, roughly a 2–3× wall-clock win once you net out draft cost.
The draft must be well-aligned with the target (same tokenizer, similar training) or $\alpha$ collapses.
Self-speculation variants (Medusa's extra heads, EAGLE) drop the separate draft model.

## Long context: RoPE scaling

A model trained with RoPE (rotary position embeddings, [transformer architecture](04-transformer-architecture.md))
at 4K context does not automatically work at 32K — positions beyond training produce rotation
frequencies the model never saw, and quality falls off a cliff. Two scaling techniques extend it
without full retraining:

- **Position Interpolation** (Chen et al., 2023) — linearly *scale down* position indices so a 32K
  sequence maps into the 4K range the model was trained on. A short fine-tune adapts it. Simple, but
  compresses every frequency uniformly, blurring the high-frequency (local, short-range) information
  the model relies on for adjacent-token relationships.
- **YaRN** (Peng et al., 2023) — interpolate frequency-dependently: leave high-frequency (local)
  dimensions nearly untouched and interpolate mainly the low-frequency (long-range) ones, plus a small
  attention-temperature correction. Reaches longer contexts with far less fine-tuning and less quality
  loss than plain PI, which is why most 2026 long-context open models (Qwen, Llama) ship YaRN-style
  scaling.

Context length is not free: attention is $O(n^2)$ and the KV cache grows linearly, so 128K context is
expensive per request. The recurring engineering decision is **long context vs RAG** — stuff
everything in the window (simple, expensive, and subject to lost-in-the-middle degradation) versus
retrieve the relevant slice (cheaper, but a retrieval miss means the answer is absent). See
[RAG and agents](09-rag-agents.md); the honest answer is usually a hybrid.

## Constrained decoding

When output must be machine-parseable — JSON matching a schema, a SQL query, a token from a fixed
enum — hope is not a strategy. **Constrained decoding** masks the logits at each step to zero out any
token that would violate the grammar, so only structurally-valid continuations can be sampled. A
JSON-schema or context-free grammar is compiled (e.g. Outlines, XGrammar, llama.cpp GBNF) into a state
machine that, at each position, tells the sampler which tokens are legal. The output is *guaranteed*
to parse — you are not asking the model to be well-behaved, you are making malformed output
impossible. Costs and caveats: a small per-step masking overhead, occasional tokenizer-boundary
friction, and the real risk that over-constraining *hurts content quality* — forcing a rigid structure
can push the model off the manifold it reasons well on. Prefer the lightest constraint that guarantees
parseability, and validate semantics separately.

## Latency anatomy: TTFT vs throughput

Inference has two distinct phases with different bottlenecks, and conflating them is the classic
mistake:

- **Prefill** — process the entire prompt in one parallel forward pass to build the KV cache. This is
  *compute-bound* (big matmuls, GPU well-utilized) and sets **TTFT (time to first token)**. TTFT scales
  with prompt length, so a 100K-token RAG context makes users wait before *anything* appears.
- **Decode** — generate output tokens one at a time, each a forward pass reading all weights. This is
  *memory-bandwidth-bound* and sets **inter-token latency / tokens-per-second** (throughput).

They pull in opposite directions. A user-facing chat app optimizes **TTFT** (feel responsive) and
per-user tokens/sec; a batch pipeline optimizes aggregate **throughput** (total tokens/sec across all
requests) and does not care about any single request's latency. **Continuous batching** (a.k.a.
in-flight batching) is the lever that reconciles them: instead of waiting for a whole batch to finish,
the server swaps completed sequences out and new ones in every step, keeping the GPU full and
massively raising throughput without stalling short requests behind long ones. This — plus
PagedAttention for the KV cache — is what vLLM and SGLang do; the systems-level depth (kernels,
scheduling, tensor/pipeline parallelism) lives in [inference](../language-modeling/09-inference.md).

The number you report depends on who is asking: an SRE wants p99 TTFT and tokens/sec/GPU; a cost model
wants tokens/sec/dollar; a user wants the stream to start fast and not stutter.

## What interviews ask here

- Why does beam search fail for open-ended generation? — Human text is not the max-probability
  sequence; maximizing likelihood yields short, generic, repetitive output. Beam is for seq2seq/MT.
- Top-k vs top-p vs min-p? — Top-k keeps a fixed count (ignores shape); top-p keeps a cumulative-mass
  nucleus (shape-adaptive); min-p thresholds relative to the top token (best coherence at high T).
- How big is the KV cache and why does it matter? — `2·L·n_kv·d_head·bytes` per token; it caps batch
  size and often exceeds weight memory, so serving is memory-bandwidth bound (motivates GQA, KV quant).
- Why is speculative decoding lossless? — The accept/reject test (accept w.p. `min(1,p/q)`, resample
  from `max(0,p−q)`) makes accepted tokens distributed exactly as sampling from the target.
- What is TTFT vs throughput and what drives each? — TTFT = prefill (compute-bound, scales with prompt
  length); throughput = decode (memory-bandwidth-bound); continuous batching raises throughput.
- What does YaRN do that plain position interpolation doesn't? — Interpolates RoPE frequencies
  non-uniformly (preserves high-frequency local info), extending context with less fine-tuning and loss.

## Where this shows up on the job

- Picking and pinning sampling params per feature (greedy for extraction, top-p/min-p for chat) and
  logging them so generations are reproducible.
- Sizing GPU memory and max batch size from the KV-cache formula before a deployment, and choosing
  GQA / KV-quantization to fit more concurrent requests.
- Turning on speculative decoding (or vLLM continuous batching) to cut cost/latency, and debugging a
  low acceptance rate from a mismatched draft model.
- Deciding long-context (YaRN-scaled window) vs RAG for a document feature, and enforcing JSON/schema
  output with constrained decoding in an extraction or tool-calling pipeline.
