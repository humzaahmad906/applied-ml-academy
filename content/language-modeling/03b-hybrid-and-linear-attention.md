# 03b — Hybrid and Linear Attention

The block you built in the last chapter has one structural flaw that no amount of RMSNorm, RoPE, or
SwiGLU tuning fixes: attention costs `O(n²)` compute and its inference state grows without bound.
For most of the Transformer era that was tolerable because context windows were short. It stopped
being tolerable when the field went after 128k, 1M, and multi-million-token contexts. This chapter
covers the architectural responses — linear attention, state-space models, and local attention —
and the design that actually won production in 2024–2026: **hybrids** that interleave a few full
attention layers into an otherwise-subquadratic stack. This is now first-class architecture
knowledge, not an inference footnote, and by the end you should be able to reason about when to
reach for each.

## The `O(n²)` wall

Recall the score computation from the last chapter: `scores = Q @ Kᵀ` produces an `n × n` matrix
for a sequence of length `n`. Both the compute and (in the naive form) the memory of that matrix
scale quadratically. FlashAttention removes the `O(n²)` *memory* by never materializing the full
matrix, but it cannot remove the `O(n²)` *compute* — the work of comparing every query to every
key is intrinsic to softmax attention. Double the context and you quadruple the attention FLOPs.

Worse, at inference the problem changes shape. During autoregressive decode you keep a KV cache
(chapter 09) so you do not recompute past keys and values — but that cache grows *linearly* with
sequence length, and every new token must attend over all of it. Decode is memory-bandwidth-bound,
and the bandwidth you spend is proportional to how much KV you have accumulated. At a million
tokens the KV cache dwarfs the model weights. So attention has two problems at long context:
quadratic training compute, and linear-and-unbounded inference state. Every architecture in this
chapter is an attempt to break one or both.

## Linear attention: the kernel reformulation

Start from causal softmax attention for a single query at position `i`, written as a weighted sum
over the keys and values up to `i`:

$$
o_i = \frac{\sum_{j \le i} \exp(q_i^\top k_j)\, v_j}{\sum_{j \le i} \exp(q_i^\top k_j)}
$$

(dropping the `1/√d` into the scores for brevity). The `exp(q_iᵀk_j)` term is what forces the
`n × n` matrix: it couples every query with every key nonlinearly, so you cannot factor it. **Linear
attention** (Katharopoulos et al., 2020) replaces the exponential with a kernel that *is*
factorable — a dot product of feature maps `φ(·)`:

$$
\exp(q_i^\top k_j) \;\longrightarrow\; \phi(q_i)^\top \phi(k_j)
$$

Now the numerator becomes `Σⱼ (φ(q_i)ᵀφ(k_j)) vⱼ`, and because `φ(q_i)` does not depend on `j` you
can pull it out of the sum by associativity of matrix multiplication:

$$
o_i = \frac{\phi(q_i)^\top \sum_{j \le i} \phi(k_j)\, v_j^\top}{\phi(q_i)^\top \sum_{j \le i} \phi(k_j)}
$$

This is the whole trick. Define a running **state** matrix and a running normalizer:

$$
S_i = \sum_{j \le i} \phi(k_j)\, v_j^\top \in \mathbb{R}^{d \times d}, \qquad
z_i = \sum_{j \le i} \phi(k_j) \in \mathbb{R}^{d}
$$

Both admit a trivial recurrence — each is the previous value plus a rank-one (or vector) update:

$$
S_i = S_{i-1} + \phi(k_i)\, v_i^\top, \qquad z_i = z_{i-1} + \phi(k_i), \qquad
o_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
$$

So linear attention is a **linear RNN**. Training is a prefix-sum (parallel scan) over the sequence,
which is `O(n)` in the sequence length instead of `O(n²)`. Inference is even better: the state `S`
is a **fixed-size** `d × d` matrix, so each decode step is `O(1)` in context length and the memory
never grows. No KV cache. The Performer variant (FAVOR+, Choromanski et al., 2020) picks `φ` as a
random-feature map that provably *approximates* the softmax kernel, so you can even take a
pretrained softmax model as the target; the linear-transformer variant just picks a simple positive
`φ` (e.g. `elu(x)+1`) and trains from scratch.

**What you lose.** Softmax can put almost all its weight on one key — it selects. A fixed-size `d ×
d` state cannot: it is a lossy running summary of everything seen, and once two pieces of
information collide in that state you cannot cleanly pull one back out. This is why plain linear
attention underperforms softmax on tasks that need precise **recall** — "what was the phone number
15,000 tokens ago" is exactly the query a compressed state answers badly. That single weakness is
the thread running through the rest of this chapter.

## State-space models: S4 → Mamba → Mamba-2

State-space models (SSMs) arrive at the same linear-recurrence structure from control theory rather
than from attention. A continuous linear system maps an input signal to an output through a hidden
state; discretized to a sequence, it becomes

$$
h_t = \bar{A}\, h_{t-1} + \bar{B}\, x_t, \qquad y_t = C\, h_t \; (+\, D x_t)
$$

with hidden state `h_t ∈ ℝ^N`. This is again a linear RNN, but because it is **linear
time-invariant (LTI)** — `Ā, B̄, C` are the same at every position — the whole sequence can also be
computed as a single long **convolution**, which is what makes it parallelizable at training time.
**S4** (Gu et al., 2021) made this work at depth by structuring `A` with HiPPO theory so the state
compresses history in a principled way and the convolution kernel is stable over thousands of
steps.

The problem with LTI is that the dynamics cannot depend on the content. A model whose `A, B, C` are
fixed treats "the" and a rare named entity identically; it cannot decide to *remember this token
and forget that one*. **Mamba** (Gu & Dao, 2023) fixes this with **selectivity**: it makes `B`, `C`,
and the discretization step `Δ` **functions of the input** `x_t`. The step size `Δ` acts as a
soft gate — a large `Δ` makes the model write the current input strongly and overwrite state (focus),
a small `Δ` lets input pass through while history persists (skip). This input-dependence is exactly
what linear attention lacked, and it is why Mamba is called a *selective* SSM.

Selectivity breaks LTI: with input-dependent parameters the convolution trick no longer applies,
because there is no single fixed kernel. Mamba recovers parallel training with a **hardware-aware
selective scan** — a parallel prefix-scan implementation fused into a single GPU kernel that keeps
the expanded state in fast SRAM and never writes it to HBM. This co-design is not optional polish;
a selective SSM without the fused scan is correct but too slow to train. The payoff is a model that
matches or beats similarly-sized Transformers on language modeling while decoding in **constant
state**.

**Mamba-2** (Dao & Gu, 2024) is the piece that connects everything in this chapter. Its central
result is **structured state-space duality (SSD)**: if you restrict the SSM's state matrix to a
scalar times the identity, `A_t = a_t I`, then the SSM is *mathematically equivalent* to a form of
masked attention whose mask is a **1-semiseparable** matrix (a lower-triangular matrix built from
the cumulative products of the `a_t` gates). Concretely, the same sequence transformation has two
faces:

- a **linear-time recurrence** — `O(n)`, constant state, ideal for inference; and
- a **quadratic-time "attention" form** — `O(n²)`, expressed as big dense matmuls, ideal for
  training because it saturates tensor cores.

SSD lets Mamba-2 train with a **chunked** algorithm: quadratic (attention-style) *within* a chunk,
linear recurrence *across* chunks, getting the best of both. Practically this let Mamba-2 use a much
larger state dimension than Mamba-1 and run substantially faster. The conceptual takeaway is the one
to keep: **SSMs and (linear) attention are two algorithmic realizations of the same underlying
linear operator.** They are not rival families; they are dual views.

## Sliding-window and local attention

The cheapest fix keeps softmax attention but restricts its reach. **Sliding-window attention (SWA)**
lets each token attend only to the previous `W` tokens instead of all `n`, dropping compute from
`O(n²)` to `O(n·W)` and — crucially for inference — bounding the KV cache to a fixed rolling buffer
of size `W`, independent of sequence length. Longformer (Beltagy et al., 2020) introduced the local
window plus a handful of **global tokens** that see everything, for document tasks. Mistral 7B
(2023) showed a plain `W = 4096` window works at LLM scale with no meaningful quality loss, making
SWA a production standard.

The subtlety is receptive field. Stacking `L` windowed layers grows the *effective* receptive field
to roughly `L·W` — information can hop window-to-window up the stack, exactly like a CNN — so a deep
model with a 4k window nominally reaches over 100k tokens. But that reach is diluted: information
must survive many hops, and the model cannot sharply retrieve a specific distant token the way full
attention can. So SWA alone has the same recall ceiling as the other subquadratic methods. This is
why current models (Gemma 3, GPT-OSS, and many hybrids) **interleave** a few full-attention layers
among the windowed ones rather than going pure-local — which is the pattern the next section
generalizes.

## The hybrid recipe

Every subquadratic method above hits the same wall: a compressed or windowed state is efficient but
recalls poorly. Pure Mamba/Mamba-2 models measurably lag on associative-recall tasks — five-shot
MMLU, needle-in-a-haystack, synthetic "phonebook" copying — precisely because a fixed-size state
cannot hold and index arbitrary key–value pairs. Full attention is the opposite: expensive, but its
growing KV cache *is* a perfect content-addressable memory.

The hybrid recipe is to **interleave a small fraction of full-attention layers into a mostly-SSM (or
mostly-linear) stack**. The empirical finding, reproduced across labs, is striking: adding only
~7–8% attention layers to a Mamba-2 backbone closes the recall gap *and* the resulting model
slightly **exceeds** a pure Transformer on average across standard benchmarks — while keeping most
of the SSM's efficiency. The SSM layers do the cheap bulk sequence mixing; the few attention layers
provide the sharp long-range recall the SSM can't.

There are two ways to combine them:

- **Block-level (layer interleaving):** whole layers are either attention or SSM, stacked in a
  pattern. Jamba, Samba, Zamba, and Nemotron-H are block-level.
- **Head-level (parallel hybrid):** within a *single* layer, some heads are attention and some are
  SSM, computed in parallel and fused. Hymba and Falcon-H1 are head-level.

Real models shipped in 2024–2026 (mechanisms as reported by their authors — verify exact configs
against each model card before relying on numbers):

- **Jamba / Jamba-1.5 (AI21, 2024)** — the first at-scale hybrid: Transformer + Mamba + MoE, an
  attention-to-Mamba layer ratio of **1:7**, MoE applied on every other layer. 52B total / ~12B
  active parameters.
- **MiniMax-01 (2025)** — hybrid **lightning attention** (a linear-attention variant) with softmax
  attention at a **1:7** softmax-to-linear ratio, plus MoE; 456B total / ~45.9B active, targeting
  contexts into the millions. The authors note that *pure* lightning attention fails
  needle-in-a-haystack retrieval, which is why the softmax layers are retained.
- **Nemotron-H (NVIDIA, 2025)** — replaces roughly **92%** of attention layers with Mamba-2 blocks
  at 8B/47B/56B scales, reporting up to ~3× throughput over comparable Transformers at matched
  accuracy.
- **Falcon-H1 (TII, 2025)** — parallel (head-level) hybrid mixing attention and Mamba heads within
  each layer, released across scales from 0.5B to 34B.
- **IBM Granite 4.0, Zamba, Hymba, Samba** — additional block- or head-level Mamba/attention
  hybrids in the same period. Qwen's newer line (e.g. Qwen3-Next) also adopts a hybrid of gated
  linear attention with periodic full attention. *(I have not verified every one of these configs
  to the layer; treat the specific ratios as illustrative and confirm per model.)*

The convergence is the signal: independent teams landed on "mostly subquadratic, a pinch of full
attention," differing mainly on which subquadratic primitive (Mamba-2 vs a linear-attention flavor)
and block- vs head-level mixing.

## Inference implications

This is where the architecture pays off, and it ties directly to chapter 09. Recall the KV-cache
size there scales with `n_layers · n_kv_heads · d_head · L_seq`. In a hybrid, only the
**attention** layers contribute a growing KV cache; every SSM or linear-attention layer carries a
**fixed-size recurrent state** independent of `L_seq`. So:

- **Memory at long context** is dominated by the handful of attention layers you kept. Cut attention
  from every layer to one-in-eight and you cut the length-dependent KV memory by roughly the same
  factor. Many hybrids further make those retained attention layers *sliding-window*, so even they
  contribute only a bounded buffer — giving a model whose total decode memory is effectively
  constant in context length.
- **Decode compute per token** for an SSM layer is `O(1)` in context (a fixed state update), versus
  `O(L)` for a full-attention layer that must read the whole KV cache. At long context the SSM
  layers are nearly free and the attention layers are the cost.

This is exactly the "architectural escape from the memory-bound decode bottleneck" flagged at the
end of chapter 09. The training-time duality (SSD) and the inference-time recurrence are the same
coin: you train with the parallel/quadratic form to feed tensor cores, then serve with the linear
recurrence to get constant-state decode.

## How to reason about picking an architecture

Match the primitive to the bottleneck, not to fashion:

- **Short context, quality-first, small effort budget:** a plain GQA Transformer is still the
  simplest correct answer, and the tooling is the most mature. Do not add an SSM you do not need.
- **Long-context, throughput-bound serving** (summarization, high-QPS long documents, long-running
  agents): a hybrid with a mostly-Mamba/linear stack and a few (ideally windowed) attention layers
  gives near-constant decode memory and multi-x throughput. This is the regime that drove the 2025
  hybrids.
- **Recall-heavy workloads** (in-context retrieval, many-shot prompts, tool schemas, exact copying):
  keep more full attention. Pure SSM/linear will silently fail needle-in-a-haystack; a hybrid tuned
  for recall (or a windowed-plus-global scheme) is the floor.
- **Cross-cutting practicalities:** SSM kernels and serving stacks are less mature than the deeply
  optimized attention ecosystem, though closing fast; a hybrid's few attention layers still need a
  standard KV cache, so your serving system must handle *two* kinds of state at once. And the
  training/inference duality means you generally cannot bolt an SSM on as an afterthought — you
  choose the mix up front.

The mental model to leave with: attention is a memory that grows and can be queried exactly but
costs `O(n)` to read; SSM/linear attention is a fixed-size memory that is cheap forever but forgets.
Hybrids buy exactness only where it is worth paying for, and take the cheap memory everywhere else.

## Key takeaways

Softmax attention's `O(n²)` compute and unbounded KV cache are what long context broke. **Linear
attention** replaces `exp(qᵀk)` with a factorable kernel `φ(q)ᵀφ(k)`, turning attention into a linear
RNN with a fixed `d × d` state — `O(n)` train, `O(1)`-state decode — at the cost of precise recall.
**SSMs** reach the same linear recurrence from control theory: S4 is LTI (and thus a convolution),
**Mamba** adds input-dependent selectivity (the `Δ` gate) trained with a hardware-aware scan, and
**Mamba-2**'s SSD duality proves the SSM and a 1-semiseparable masked attention are the same
operator with two algorithms — quadratic for training, linear for inference. **Sliding-window
attention** bounds cost to `O(n·W)` and the KV cache to a rolling buffer, but its stacked receptive
field recalls distant tokens weakly. Because every subquadratic method compresses or windows its
state, all share a recall ceiling — so the winning 2024–2026 designs are **hybrids** that interleave
~1-in-8 full-attention layers (Jamba, MiniMax-01, Nemotron-H, Falcon-H1, Granite 4.0, and kin),
recovering recall while keeping near-constant decode memory. Pick per bottleneck: plain Transformer
for short/quality, hybrid for long/throughput, more attention for recall.

## You can now

- explain the two distinct costs of softmax attention at long context — quadratic training compute
  and a linear, unbounded KV cache — and which architectural response attacks which.
- derive linear attention from softmax by kernelizing `exp(qᵀk)` into `φ(q)ᵀφ(k)`, and write the
  running-state recurrence `S_i = S_{i-1} + φ(k_i)v_iᵀ`, `o_i = φ(q_i)ᵀS_i / φ(q_i)ᵀz_i`.
- trace the SSM lineage: LTI S4 (convolution) → selective Mamba (input-dependent `B,C,Δ`, fused
  scan) → Mamba-2 SSD, and state the SSM↔masked-attention duality and why it matters for train-vs-serve.
- describe sliding-window attention's cost and receptive-field behavior and why it alone does not
  fix recall.
- state the hybrid recipe (~7–8% full-attention layers), distinguish block-level from head-level
  mixing, and name real shipped hybrids with their rough attention:SSM ratios.
- reason about a model's long-context inference memory as "growing KV from the attention layers,
  constant state from the SSM layers," and pick an architecture from the workload's bottleneck.

## Try it

Take your 1.3B config from chapter 03 and convert it to a Jamba-style hybrid: keep the same `d`,
`num_layers`, and `d_head`, but make only every 8th layer full attention and the rest Mamba-2
blocks. First, estimate the KV-cache memory at a 128k context for (a) the all-attention original and
(b) the hybrid, using the chapter-09 formula with `n_layers` set to only the attention layers for
(b) — you should see roughly an 8× reduction in the length-dependent term. Then reason through
decode: for a single new token at position 128k, count the KV reads for one attention layer versus
the fixed-state update for one Mamba-2 layer, and explain in one sentence why the hybrid's decode is
nearly context-length-independent while the original's is not.
