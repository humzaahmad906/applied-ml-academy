# 18 — Interview Bank IV: Breadth & Rapid-Fire — Part 1 of 4: Tokenization, MoE internals & GPUs

The first three interview banks went deep on estimation, systems, inference, and post-training. This
one goes *wide*. These are the breadth questions a frontier-lab interviewer fires to map the edges
of what you know — the "ML breadth" round where they hop across tokenization, MoE internals, the
roofline, scaling-law *fitting*, data pipelines, evaluation protocol, sampling, long context, and
alignment mechanics — plus a rapid-fire "intuition/trivia" round to calibrate how quickly you can
retrieve a fact and say *why* it is true. The questions here deliberately avoid the topics already
drilled in the estimation-and-design bank (back-of-envelope, the decoder block, quadratic
attention/MLA, MFU debugging, FlashAttention, Triton-vs-compile, the parallelism ladder, two-phase
inference/serving/spec-decoding, quantization int4/PTQ-QAT, Chinchilla-deployment, the GRPO recipe,
eval-is-lying, fine-tune-vs-RAG). They fill the gaps around those.

Each question has a worked answer at the depth an interviewer wants. Read it, then close it and
reproduce the reasoning, because the point is to *derive* the answer live, not recite it. A one-line
note after most answers flags what the question is really probing.

This is part 1 of 4 of this breadth bank. Here we cover tokenization, Mixture-of-Experts internals,
and GPUs & the roofline; the remaining parts cover scaling laws, data, evaluation, sampling, long
context, alignment mechanics, and a rapid-fire round.

---

## Part A — Tokenization

**A1. For a sub-billion-parameter model, where do the parameters actually live, and why does that constrain vocabulary size?**

The transformer body is roughly `12 · num_layers · d²` parameters (attention ~`4d²`, SwiGLU FFN
~`8d²` per layer), but the embedding table and output head are each `V × d`. When the model is small,
`V × d` does not shrink with the body: at `V = 128k` and `d = 2048` the embedding alone is 262M
parameters, which for a sub-billion model can be a *third or more* of the whole budget. So a large
vocabulary is a luxury small models cannot afford — the embedding dominates. This is exactly why
on-device and toy models use smaller vocabularies (a typical toy run uses `V = 10000`): you spend your
parameter budget on the transformer, not on a giant lookup table where most rows are rarely updated.
The tradeoff is real in both directions: a *bigger* vocab shortens sequences (cheaper attention,
more text per context window) but grows the embedding and starves rare tokens of gradient updates.
*Probes: whether you can locate parameters in a model, not just quote a total.*

**A2. What is fertility, and why does an English-trained tokenizer make multilingual and code serving expensive?**

Fertility is the average number of tokens a word gets split into — near 1 for English on an
English-trained BPE tokenizer, much higher out of domain. BPE bakes in the *statistics of its
training corpus*: it learns merges for the byte sequences that were frequent, so English words
collapse to one or two tokens while Urdu, Chinese, or code identifiers — whose byte patterns the
merges never learned, and which are multi-byte in UTF-8 to begin with — fragment into many small
tokens. Two costs follow. First, cost per document scales with token count, so the same paragraph in
a fragmented language costs several times more compute and money to process. Second, empirically the
model is a bit *worse* at over-fragmented inputs. This is why serving multiple languages or code is
a data-mix and tokenizer decision, not just a runtime one: if your product runs on shipping labels
and part numbers, a generic web tokenizer over-fragments exactly your domain. Always report fertility
per language before you commit.
*Probes: connecting the tokenizer to real serving cost and quality, not just compression.*

**A3. Compare BPE, WordPiece, and Unigram. When would you not use BPE?**

All three are subword schemes; they differ in how they *choose* the vocabulary. **BPE** is bottom-up
and greedy: start from bytes, repeatedly merge the most frequent adjacent pair, record the merges in
order, and at encode time replay those merges earliest-first. **WordPiece** (BERT) is also
merge-based but picks the merge that most increases the *likelihood* of the corpus under a unigram
language model rather than the raw pair count — a slightly more principled selection. **Unigram**
(SentencePiece) goes top-down: start from a large candidate vocabulary and *prune* the tokens whose
removal costs the least likelihood, keeping a probabilistic model that can score *multiple*
segmentations of the same string, which enables subword regularization (sampling segmentations at
training time). For LLMs, byte-level BPE won because it is simple, deterministic, always
representable (base vocab is the 256 bytes, so nothing is ever "unknown"), and round-trips losslessly.
You reach for Unigram/SentencePiece when you want segmentation sampling as regularization or clean
multilingual handling; WordPiece is mostly a BERT-era artifact now.
*Probes: knowing there is a family, and the merge-vs-prune distinction, not just "BPE."*

**A4. Why is the tokenizer frozen once pretraining starts, and what breaks if you change it?**

Every embedding row and output-head row is trained *against a specific integer-to-string mapping*.
The id `4021` means whatever string the tokenizer assigned it, and the model has learned an embedding
for exactly that meaning. Change the tokenizer and the ids now point to different strings, so every
embedding is attached to the wrong concept — the model is effectively reading scrambled input. There
is no cheap patch: you would have to re-learn the embedding table and output head at minimum, and
realistically re-pretrain, because the whole model was optimized around this vocabulary's
segmentation statistics. This is why tokenization is a frozen, upfront decision that "sets a floor on
cost and a ceiling on which inputs your model handles well." The only safe extensions are *adding*
special tokens (new ids appended, new rows randomly initialized and trained) — you never renumber or
redefine existing ones.
*Probes: understanding the embedding is a lookup keyed on token id, so the map is load-bearing.*

**A5. Explain the leading-space / pre-tokenization gotcha. Why do `"cat"` and `" cat"` tokenize differently?**

Real BPE never runs merges across the raw byte stream; it first splits text into pre-tokens with the
GPT-2 regex, then merges *within* each chunk and never across chunk boundaries. That regex attaches
an optional leading space to the following word (` ?\p{L}+`), so ` cat` (mid-sentence, with its
space) and `cat` (start of line, no space) fall into different pre-tokens and therefore get different
token ids. This is deliberate: it means concatenating decoded token strings reproduces the original
spacing exactly, and it stops the tokenizer from learning a token that glues `dog` to a trailing `.`
in some contexts but not others. The practical gotcha is that when you hand-build a prompt or few-shot
example, whether you put a space before a word changes its tokenization — and a model that saw ` Paris`
during training may score the answer differently if you feed it `Paris`. It bites hardest in
multiple-choice scoring and in stop-sequence handling.
*Probes: awareness that pre-tokenization exists and has downstream prompt-formatting consequences.*

**A6. Why are LLMs historically bad at arithmetic, and how much of that is the tokenizer?**

A large part is tokenization. A number like `1234567` is not seen digit-by-digit; BPE merges frequent
digit runs into chunks, so `1234567` might become `[123][4567]` or some other corpus-dependent split,
and crucially `327` and `3271` may share no token structure at all. That destroys the positional,
place-value regularity arithmetic depends on — the model cannot easily align "the hundreds digit of A
with the hundreds digit of B" when the digits are glued into inconsistent chunks. The fixes are
tokenizer-level: force single-digit tokenization (split every digit), or tokenize numbers in fixed
groups (e.g. always 3 digits, right-to-left), which is what several recent models do. Once digits are
consistent and aligned, arithmetic accuracy jumps well before any change to the transformer. It is a
clean example of the general point that the tokenizer sets a ceiling: no amount of parameters fixes a
representation that scrambles the structure of the task.
*Probes: recognizing that a "model" failure can be a representation failure upstream.*

---

## Part B — Mixture-of-Experts internals

**B1. What is routing collapse, why does it happen, and what is the standard fix?**

The router picks the top-`k` experts by a learned linear layer, and it is trained *implicitly*: you
only get gradients through the softmax gate weights on the chosen experts, so an expert picked early
gets trained more, gets better, and gets picked more — a self-reinforcing loop. Left alone, a handful
of experts absorb all the traffic and the rest are dead parameters that never train (collapse). The
standard fix is the **auxiliary load-balancing loss** (Switch/GShard):
$L_{\text{aux}} = \alpha \cdot E \cdot \sum_i (f_i \cdot P_i)$, where `f_i` is the *fraction of tokens* routed to expert `i` (a
hard count) and `P_i` is the *mean router probability* for expert `i` (soft). The product is
minimized at uniform load (`f_i = P_i = 1/E`). The clever part: `f_i` is non-differentiable, so the
gradient flows only through `P_i` — the loss nudges the router's probabilities toward *under*-loaded
experts. `α` is small (~1e-2): too strong forces uniform routing and kills specialization, too weak
lets collapse win. First thing to check when an MoE underperforms a dense model of equal *active*
size: the per-expert token histogram.
*Probes: the discrete-routing instability and the exact form of the balancing term.*

**B2. Explain expert capacity and token dropping. What does the capacity factor buy?**

In token-choice routing you cap how many tokens each expert accepts per batch:
$\text{capacity} = \text{capacity\_factor} \cdot (T / E)$. Tokens beyond an expert's capacity are *dropped* — they skip
the expert layer and pass through unchanged via the residual connection. The reason you need a cap at
all is systems, not modeling: when experts are sharded across GPUs (expert parallelism), a hot expert
that receives triple its share stalls the whole all-to-all collective while other GPUs idle, so you
bound the imbalance. The capacity factor (typically 1.0–2.0) trades drops against waste: higher means
fewer dropped tokens but more memory and FLOPs spent on padded, unused slots. The alternative that
sidesteps dropping is **expert-choice** routing — each expert picks its top tokens — which guarantees
perfect balance and zero drops but lets a token be processed by zero or many experts.
*Probes: that MoE balance is a systems constraint, and the drop-vs-waste tradeoff.*

**B3. Shared experts versus fine-grained experts — what problem does each solve?**

Both are refinements DeepSeek converged on, and they compose. **Fine-grained experts:** instead of a
few large experts, split each into several smaller ones and route to proportionally more (e.g. 256
small experts, top-8, instead of 8 large, top-2). The active parameter count is identical, but the
router now has vastly more *combinations* to compose (`C(256,8)` versus `C(8,2)`), which empirically
sharpens specialization. **Shared experts:** designate one or a few experts that *every* token always
uses, on top of the routed ones. The shared expert absorbs the common, always-needed computation, so
the routed experts stop each re-learning the basics and are freed to specialize. DeepSeek-V2/V3 use
both: many fine-grained routed experts plus a small number of shared ones. The unifying idea is
letting the router spend its capacity on *what differs* between tokens rather than on redundant
common work.
*Probes: that MoE design is about specialization pressure, not just parameter count.*

**B4. How does the 6ND rule apply to an MoE, and why must you separate total from active parameters?**

Two counts matter and you must never conflate them. **Total parameters** = all experts summed;
determines the memory to hold the model and loosely its knowledge capacity. **Active parameters** =
those actually used per token (the `k` selected experts plus shared attention/embeddings); determines
FLOPs per token. The `6ND` training-compute rule uses `N = active` parameters, because compute tracks
what runs, not what is stored. So Mixtral "8x7B" is ~47B total (attention and embeddings are shared,
only MLPs replicate — it is *not* 56B) but activates ~13B per token, and it costs like a 13B model to
train and serve while carrying the capacity of something much larger. DeepSeek-V3 is 671B total,
~37B active. When someone quotes an MoE's "size," your first question is *which number* — the
training-cost estimate and the memory-footprint estimate use different ones.
*Probes: the total-vs-active split, the single most common MoE misconception.*

**B5. Why is MoE a datacenter technique and a poor fit for on-device?**

Because its whole selling point — more capacity at fixed active FLOPs — is bought with *memory* and
*communication*, which are exactly what a device lacks. All experts must be stored resident, so total
memory is large even though active FLOPs are small; a phone or a Mac mini simply cannot hold a 671B
(or even 47B) parameter footprint. And when experts are sharded across GPUs, every token is routed to
whichever GPU holds its chosen expert and the result sent back — an all-to-all pattern that is
expensive and acutely sensitive to load imbalance, and that assumes a fast interconnect. On a single
device you have neither the memory to hold all experts nor a fabric to shard them across. So MoE wins
when you are throughput- or knowledge-bound on a well-connected multi-GPU system with memory to
spare; it loses on the edge, which is why on-device models are almost always dense.
*Probes: reasoning about where a technique's costs land relative to the deployment constraint.*

**B6. What is loss-free load balancing and why did DeepSeek move to it?**

The auxiliary balancing loss works, but its gradient term slightly *degrades* model quality — you are
adding a training signal orthogonal to the actual language-modeling objective, pulling the router
toward uniformity for its own sake. DeepSeek-V3's **loss-free balancing** drops the auxiliary loss
entirely and instead keeps a per-expert **bias** `b_i` that is added to the routing logits *only for
the top-k selection* — not for the gate weights that combine the outputs. After each step, `b_i` is
nudged *down* for overloaded experts and *up* for underloaded ones, so load equalizes over time
without ever injecting a balancing gradient into the model weights. It is a controller on the
selection, not a term in the loss. The payoff is balance without the small quality tax of the
auxiliary loss, which is why the newest large MoEs adopted it.
*Probes: awareness that the standard fix has a cost and that the frontier moved past it.*

---

## Part C — GPUs & the roofline

**C1. Derive arithmetic intensity and explain the roofline in your own words.**

Arithmetic intensity is `FLOPs performed / bytes moved to and from HBM` — how much math you do per
byte you fetch. Every GPU has two ceilings: a peak compute rate (FLOP/s) and a peak memory bandwidth
(bytes/s). Their ratio is a break-even intensity, the FLOPs-per-byte where the two ceilings meet:
`break-even = peak FLOP/s ÷ peak bytes/s`. If your op's intensity is *below* break-even you finish
moving the bytes before you finish the math, so the compute units idle waiting on data — you are
**memory-bound**. Above it, the arithmetic units are the limit — **compute-bound**. Plot achievable
performance against intensity and you get the roofline: a sloped line (`performance = intensity ×
bandwidth`, the memory limit) rising to meet a flat line (the compute limit) at the break-even point.
Any kernel sits under whichever piece of the roof its intensity puts it below. The diagnostic use:
measure an op's achieved FLOP/s, place it on the plot, and you instantly know whether to chase fewer
HBM bytes (under the slope) or more compute efficiency (under the flat).
*Probes: whether you can construct the mental model, not just name "memory-bound."*

**C2. Give me a memory-bound op and a compute-bound op and explain the difference in intensity.**

Compute-bound: a **large matrix multiply** (the MLP, the big attention projections). An `M×K` by
`K×N` matmul does `2MNK` FLOPs and moves on the order of `MK + KN + MN` elements, so each element
loaded participates in many multiply-adds and intensity scales with the shared dimension — comfortably
above break-even, which is exactly what tensor cores are for. Memory-bound: **softmax** (and
elementwise ops generally). Softmax loads each element, does a handful of operations (max, subtract,
exp, sum, divide), and writes back, so intensity is near 1 — two to three orders of magnitude below
break-even, and the naive version makes several separate passes over HBM. The contrast is the whole
point made earlier: a transformer forward pass is a few compute-bound matmuls interleaved with many
memory-bound elementwise/reduction/attention ops, and the *time* goes to the HBM round-trips of the
memory-bound ones, not the FLOPs of the matmuls. That is why fusion and low precision, not more
compute, are the levers.
*Probes: naming concrete ops on each side and explaining via intensity, not by rote.*

**C3. Walk up the memory hierarchy from registers to HBM. Why does the hierarchy dictate kernel design?**

Fastest and smallest first: **registers** (per-thread, single-cycle, where working values live);
**shared memory / L1** (per-block, on-chip, fast, up to ~192 KB per SM on A100/H100, where threads in
a block cooperate); **L2 cache** (chip-wide, tens of MB, slower); **HBM** (the 40–80 GB you think of
as "GPU memory," large but far — where weights and activations live). The analogy: HBM is a warehouse,
huge but far away; shared memory and registers are the factory floor, small but where work actually
happens. Every trip to the warehouse costs you. So the entire game of a fast kernel is to move data
from HBM as few times as possible, do as much work as possible while it is on-chip, and ship back as
little as possible. This is *why* fusion wins (one HBM round-trip instead of one per op) and why
FlashAttention tiles the computation to keep the `L×L` scores on-chip — the hierarchy is the reason
data movement, not arithmetic, is the thing to minimize.
*Probes: connecting the hierarchy to the design principle, not just listing levels.*

**C4. What is occupancy, and why do larger batches run more efficiently per element?**

A GPU hides memory latency by keeping far more warps resident than it can execute at once: when one
warp stalls on an HBM read, the scheduler instantly swaps in another whose data is ready. Occupancy
is the ratio of active warps to the maximum an SM can hold, and it is limited by resources — each
block consumes registers and shared memory, so a kernel that hogs either fits fewer blocks per SM and
exposes less parallelism to hide latency behind. Larger batches (and longer sequences) help two ways:
they give the GPU more independent work to keep occupancy high, and they make the matmuls larger and
therefore higher-intensity. This is the systems reason a single-token, single-user decode runs the
GPU at a fraction of its capability — the matmul degenerates to a skinny matrix-vector product with
intensity near 1 — and it is the entire motivation for inference batching.
*Probes: latency-hiding as the mechanism, linking occupancy to why batching matters.*

**C5. Why must elementwise ops be fused, and what does `torch.compile` do about it?**

An elementwise op (activation, RMSNorm scale, residual add, dropout) has intensity near 1: it reads
each element from HBM, does one or a few operations, and writes it back. Run a chain of them
unfused — activation, then norm, then residual — and each is a separate kernel that streams the *same*
tensor out to HBM and back in again, so you pay the memory round-trip once per op while doing almost
no math. Fusion collapses the chain into one kernel: the data makes a single trip from HBM, all the
elementwise work happens on-chip, and it returns once. Since these ops are memory-bound, cutting the
round-trips is nearly a linear speedup on that part of the graph. `torch.compile` does exactly this
automatically for most elementwise chains — it is the first optimization to reach for — and you only
hand-write a kernel when profiling shows a specific hot chain it failed to fuse.
*Probes: the "same tensor streamed repeatedly" waste that fusion eliminates.*

**C6. Quote the compute-vs-bandwidth numbers for A100 and H100 and tell me what they imply.**

A100: FP32 on general CUDA cores is ~19.5 TFLOP/s, but bf16/fp16 on tensor cores is ~312 TFLOP/s —
roughly 16× higher — against ~1.5–2.0 TB/s of HBM (1.5 on the 40 GB card, ~2.0 on the 80 GB).
Break-even intensity is ~312e12 / 2.0e12 ≈ 156 FLOPs/byte. H100: ~990 TFLOP/s bf16 dense (more with
FP8 and sparsity) against ~3.35 TB/s of HBM3, so break-even ≈ 990e12 / 3.35e12 ≈ 295 FLOPs/byte. Two
implications. First, running matmuls in fp32 leaves ~15/16 of the chip idle — low precision is not
just a memory trick, it is the only way to reach peak compute. Second, break-even has *climbed*
generation over generation because compute grew faster than bandwidth, so the bar to be compute-bound
keeps rising and more workloads fall into the memory-bound regime, which is why bandwidth is the spec
that governs decode speed.
*Probes: carrying real numbers and drawing the fp32-penalty and rising-break-even conclusions.*

## You can now

- Locate where a small model's parameters live, explain fertility, and compare BPE/WordPiece/Unigram by how each chooses a vocabulary.
- Explain why the tokenizer is frozen at pretraining time, and diagnose the leading-space and digit-tokenization gotchas.
- Explain routing collapse, expert capacity, shared vs. fine-grained experts, and why `6ND` uses active — not total — parameters.
- Explain why MoE is a datacenter technique and how loss-free load balancing replaces the auxiliary balancing loss.
- Derive the roofline model, name compute- vs. memory-bound ops, and use real A100/H100 numbers to explain fusion, occupancy, and batching.

