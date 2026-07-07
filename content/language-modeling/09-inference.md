# 09 — Inference: Serving the Model Cheaply

Training happens once. Inference happens forever. For any deployed model the inference cost
dominates the lifetime cost, and inference has a completely different performance profile from
training. We frame the whole subject as an accounting problem: count the
FLOPs, count the bytes moved, and the ratio between them (the arithmetic intensity) tells you which
hardware limit you are hitting and therefore what optimization will actually help. This chapter is
closest to your day job, so it goes deeper on the parts that decide real-world latency and
throughput, including quantization.

## The two phases: prefill and decode

Autoregressive generation has two distinct phases with opposite characteristics. Put bluntly:
prefill is **compute-limited (parallelizable)**, decode is **memory-limited (sequential)**.

**Prefill** processes the entire prompt at once. All prompt tokens go through the model in a single
forward pass, in parallel, because you already know all of them. This phase is compute-bound: it is
one big batched matmul over many tokens, high arithmetic intensity, good GPU utilization. Prefill
latency is what determines time-to-first-token.

**Decode** generates the output one token at a time. Each new token requires a full forward pass,
but that pass processes only a single new token (plus the cached context). A single-token forward
pass is memory-bound: you load the entire model's weights from HBM to produce one token's worth of
compute, so arithmetic intensity is terrible and the GPU runs at a fraction of its capability.
Decode latency, per token, is dominated by how fast you can read the weights from memory, not by
FLOPs. This is why decode is often described as memory-bandwidth-bound and why the relevant
hardware spec for generation speed is memory bandwidth, not peak FLOPs.

The reason lives in arithmetic intensity — FLOPs performed per byte read. Work this out
per operation. For the big MLP matmuls, prefill has intensity proportional to the number of tokens
processed together (`B × T`), which is large, so those matmuls saturate the tensor cores. In
decode you process one token per sequence, so the intensity collapses toward the batch size `B`
alone: you read a full weight matrix and do a single vector-matrix product against it. Attention
sits in between, with intensity scaling like $\frac{S\cdot T}{S + T}$ where `S` is the cached length and `T`
the new tokens. Every hardware has a break-even intensity (the ridge point of its roofline, roughly
peak-FLOPs ÷ memory-bandwidth); below it you are memory-bound, above it compute-bound. Prefill is
comfortably above; single-stream decode is far below.

This split explains a lot of observed behavior. A long prompt with a short answer is dominated by
prefill (compute). A short prompt with a long answer is dominated by decode (memory bandwidth). On
your M2 Max, generation speed for a given model is set almost entirely by its memory bandwidth
divided by the bytes of weights read per token, which is why quantization (fewer bytes per weight)
directly buys decode speed.

## The KV cache

Naively, generating token `t` repeats attention over all previous tokens, recomputing their keys
and values every step. That is quadratic waste. The KV cache fixes it: after processing each
token you store its keys and values, and on the next step you only compute the query, key, and
value for the one new token and attend against the cached keys and values. This turns per-token
decode from recomputing everything to computing one token's worth plus reading the cache.

The cost is memory. The KV cache size is:

$$
\text{KV cache} = \underbrace{2}_{\text{keys \& values}}\cdot n_{layers}\cdot n_{kv\_heads}\cdot d_{head}\cdot L_{seq}\cdot \text{batch}\cdot \text{bytes\_per\_element}
$$

It grows linearly with sequence length and batch size, and for long contexts and many concurrent
users it can exceed the size of the model weights themselves. A worked example makes the scale
concrete: a 30-layer model with 32 KV heads of dimension 128 in bf16 spends
`2 × 30 × 32 × 128 × 2 = ~491 KB` per token of context. At 8 K context that is ~4 GB for a single
sequence, and a batch of 32 such sequences is ~128 GB — larger than the weights of most models you
would run. Managing this memory is the central problem of a serving system.

This is why the attention variants we met in the architecture chapter are inference-economics
decisions, not training niceties. **GQA/MQA** shrink `n_kv_heads`, cutting the KV cache by exactly the factor N/K (query
heads over KV heads) and letting you fit more users or longer contexts. **Multi-head latent
attention (MLA)**, as in DeepSeek-V2, goes further: it compresses the per-token key/value state
into a small latent of dimension C instead of storing N×H, a large constant-factor reduction.
**Local (sliding-window) attention** interleaved with occasional global layers makes most layers'
KV cache size independent of sequence length, because each local layer only keeps a fixed window.
Each of these attacks the same term in the formula above.

## Batching for throughput

A single decode step underutilizes the GPU badly (tiny batch, no occupancy, memory-bound). The
fix is to batch many users' requests together so one weight-read from HBM serves many
sequences at once, amortizing the memory-bound weight load across the batch. Recall the decode
arithmetic intensity was proportional to `B`: raising the batch size is literally how you climb the
roofline toward the compute-bound regime. This is the single biggest lever for serving throughput.

The complication is that users' requests start and finish at different times and have different
lengths, so a static batch wastes work waiting for the slowest sequence. **Continuous batching**
(also called in-flight batching) solves this: the server maintains a running batch and, as soon as
any sequence finishes, evicts it and slots in a waiting request, so the batch stays full. Worth
noting too is **selective batching** — different operations in the layer (the token-wise matmuls
versus the per-sequence attention) are batched differently because they have different shapes for
variable-length inputs. This is what vLLM and similar servers do, and it is why they get far higher
throughput than naive batched generation.

**PagedAttention** (the vLLM innovation) manages the KV cache the way an operating system manages
virtual memory. The cache is split into fixed-size blocks (pages) allocated non-contiguously, so
the fragmented, variable-length caches of a continuous batch pack into memory efficiently instead
of each reserving a worst-case contiguous block. The OS analogy is exact: a shared system prompt is
stored once and referenced by many sequences (shared pages), branching or sampling multiple
completions uses copy-on-write at the block level, and the attention kernel is fused to read the
scattered blocks. The payoff is near-zero KV fragmentation, which directly translates to a larger
serviceable batch.

There is a throughput-latency tradeoff here. Bigger batches give higher total throughput
(tokens/second across all users) but each individual user may wait longer. You tune this for your
product: an interactive assistant cares about per-user latency, a bulk document-processing pipeline
cares about total throughput. Your VisionSDK batch-processing architecture is squarely in the
second camp, so batch aggressively.

## Speculative decoding

Decode is memory-bound, which means the GPU has spare compute while waiting on weight reads.
Speculative decoding uses that spare compute to generate faster by exploiting a key asymmetry:
**generation is sequential and expensive, verification is parallel and cheap.** A
small, cheap "draft" model proposes several tokens ahead, then the large target model verifies all
of them in a single parallel forward pass (which is compute-bound and nearly free on the otherwise-
idle compute). A modified rejection-sampling rule accepts every proposed token the target agrees
with, corrects the first disagreement, and continues.

It is worth stressing that the scheme is **lossless**: it produces exact samples from the target model's
distribution, not an approximation. The math guarantees the accepted-plus-corrected sequence has
the same distribution as sampling from the target alone. If the draft model is good (high
acceptance rate), you get several tokens per expensive target-model pass instead of one; typical
speedups are 2–3x with no quality change. The cost is running two models and the verification
complexity. Variants avoid a separate model by drafting from the target's own earlier layers or a
small attached head (Medusa, EAGLE).

## Quantization for inference (your area)

Since decode is memory-bound on reading weights, storing weights in fewer bytes directly speeds up
decode and shrinks memory. This is why inference quantization is one of the highest-leverage things
you can do, and why it is the core of your on-device work. The precision ladder, by
bytes per element: bf16 at 2 bytes is the standard inference baseline; fp8 at 1 byte is aggressive
and has a narrow range (E4M3 is roughly [−240, 240]); int8 at 1 byte is the post-training
quantization workhorse; int4 at 0.5 bytes is maximum compression but needs care.

The landscape, roughly from least to most aggressive:

- **Weight-only quantization to int8 or int4.** Store weights as low-bit integers with per-group
  scales, dequantize on the fly into the matmul. Halves or quarters the weight memory and the
  weight-read traffic, so decode gets faster. Activations stay in higher precision. This is the
  workhorse for on-device. **AWQ** (activation-aware weight quantization) is the method to
  highlight: it selects and protects the small fraction of weight channels that matter most,
  identified by their activation magnitudes, and quantizes the rest aggressively. GPTQ is the other
  well-known post-training method, picking scales to minimize layerwise reconstruction error.
- **Weight-and-activation quantization (int8 both, or fp8).** Quantize activations too so the
  matmul itself runs in low precision on integer/fp8 tensor cores. Faster still, but harder,
  because activations have outliers — a few channels with huge magnitudes that wreck naive
  quantization. **LLM.int8()** handles this by extracting those outlier channels and keeping them
  in higher precision (mixed precision); SmoothQuant instead migrates the difficulty from
  activations into weights.
- **KV cache quantization.** Since the KV cache can dwarf the weights at long context, quantizing
  it (to int8 or int4) is a direct win for context length and batch size — it attacks the
  `bytes_per_element` term in the KV formula above.

Two things you already know from your QAT work but that belong here for completeness. Post-training
quantization (PTQ) quantizes an already-trained model and is cheap but loses more accuracy at low
bit-widths. Quantization-aware training (QAT) simulates the quantization during training so the
model learns weights robust to it, recovering most of the accuracy at int4 and below, at the cost
of a training run. The layer-sensitivity idea generalizes: not all layers tolerate low precision
equally, so mixed-precision schemes that keep sensitive layers higher and push tolerant layers
lower get the best accuracy-per-byte. This is exactly the sensitivity-map approach, and it is the
right instinct.

The practical decode-speed model to carry: tokens per second is roughly memory bandwidth divided
by (bytes per weight times active parameters). Halve the bytes per weight with int4 and, if you are
memory-bound, you roughly double decode speed. That is the whole reason low-bit quantization
matters for on-device serving.

## Latency and throughput metrics

Serving is measured on several axes that trade off against each other, and it is worth carefully
separating them:

- **Time-to-first-token (TTFT)** — how long until the user sees anything. Set by prefill, so it
  scales with prompt length and is compute-bound. This is the number an interactive user feels
  first.
- **Inter-token latency (ITL) / time-per-output-token** — the gap between successive tokens once
  generation starts. Set by decode, so it is memory-bandwidth-bound and improved by quantization,
  GQA, and (per accepted token) speculative decoding.
- **Throughput** — total tokens/second across all concurrent sequences. Improved by batching, which
  raises decode arithmetic intensity.

The central tension: batching and long queues raise throughput but hurt per-user TTFT and ITL. An
interactive product optimizes latency (small batches, low queue depth); a bulk pipeline optimizes
throughput (large batches, deep queues). Some stacks also do **disaggregated serving**, running
prefill and decode on separate hardware because they have opposite compute profiles — compute-heavy
prefill on one pool, bandwidth-heavy decode on another — so neither starves the other.

## Putting the phases together in a server

A production serving stack combines all of the above: quantized weights (and maybe KV cache) to
shrink memory and speed decode, continuous batching with paged KV cache to keep the GPU busy across
many users, prefill and decode possibly disaggregated because they have opposite compute profiles,
and optionally speculative decoding to squeeze the memory-bound decode phase. Beyond squeezing the
transformer, the field is also exploring architectural escapes from the autoregressive
memory-bound bottleneck entirely — state-space models (S4, Mamba, BASED), linear/full-attention
hybrids (MiniMax-01, a 456B MoE), and diffusion-style parallel token generation. Each piece attacks
a specific bottleneck you can now name.

## Key takeaways

Inference has two phases: prefill (compute-bound, sets time-to-first-token) and decode (memory-
bandwidth-bound, sets inter-token latency). Which limit you hit is decided by arithmetic intensity
— FLOPs per byte — and single-stream decode sits far below the hardware's roofline ridge point,
which is why reading weights, not doing math, is the bottleneck. The KV cache makes decode linear
instead of quadratic but its memory grows with context and batch and can exceed the weights, which
is why GQA/MQA, MLA, local attention, and KV-cache quantization all matter. Batching, especially
continuous batching with paged KV cache, is the biggest throughput lever because it raises decode
arithmetic intensity and amortizes the weight read across many sequences. Speculative decoding uses
decode's spare compute to get several tokens per expensive pass, losslessly. Quantization is the
highest-leverage on-device technique because decode speed scales with bytes-per-weight; weight-only
int4/int8 (AWQ, GPTQ) is the workhorse, activation quantization is harder due to outliers
(LLM.int8, SmoothQuant), and QAT plus layer-sensitivity mixed precision recovers accuracy at low
bit-widths. Report TTFT, inter-token latency, and throughput separately, because optimizing one
usually costs another.

## You can now

- distinguish prefill (compute-bound, sets TTFT) from decode (memory-bandwidth-bound, sets inter-token latency) by their arithmetic intensity.
- compute KV-cache size from the config and explain why GQA/MQA, MLA, local attention, and KV-cache quantization each attack a specific term in that formula.
- explain why continuous batching with a paged KV cache is the single biggest throughput lever, and why it works.
- describe speculative decoding, why verification is cheap on the otherwise-idle compute, and why the scheme is lossless.
- pick a quantization strategy (weight-only int4/int8 with AWQ/GPTQ, activation quantization, KV-cache quantization, QAT) for a memory-bound decode target, and predict the decode speedup from bytes-per-weight.

## Try it

Take a small model you can run locally and measure decode throughput (tokens/second) at batch sizes 1, 4, and 16. Confirm total throughput climbs with batch size — you are raising decode arithmetic intensity toward the compute-bound regime — while per-user inter-token latency degrades, making the throughput-versus-latency tradeoff concrete. Then quantize the weights to int4 and re-measure single-stream decode speed, and check the result against the `bandwidth / (bytes_per_weight × active_params)` prediction: if you were memory-bound, halving the bytes per weight should roughly double decode speed.
