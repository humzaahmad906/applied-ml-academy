# 18 — Interview Bank IV: Breadth & Rapid-Fire

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

---

## Part D — Scaling laws (fitting, not lore)

**D1. Name the three ways to fit a scaling law and when you would use each.**

There are three. **(1) Minimum over training curves.** Train several model sizes, each on
many token counts, record loss along each curve; for every compute level `C`, take the *minimum* loss
across all runs hitting that `C`, and the `(N, D)` achieving it is compute-optimal at that `C`. Trace
the minima and fit `N_opt ∝ C^a`. Cheap because it reuses points along curves you trained anyway.
**(2) IsoFLOP profiles.** Pick several fixed compute budgets; within each, train several model sizes
`N`, giving each the matching `D = C/(6N)` so they all consume the *same* compute; plot final loss vs
`N` — it is a U-shaped bowl whose bottom is the optimal `N` for that budget (fit a parabola in
log-space, don't just take the lowest sampled point). Collect the bowl-bottoms and fit `N_opt ∝ C^a`.
This is the cleanest and the one the scaling-laws build centers on. **(3) Parametric fit.** Fit
$L(N,D) = E + A/N^\alpha + B/D^\beta$ to *all* your `(N, D, loss)` points at once (nonlinear regression,
Huber loss on log-loss for robustness), then compute the optimum analytically via `a = β/(α+β)`. Most
data-efficient, most sensitive to fitting choices. The discipline: all three must *agree* before you
trust an extrapolation.
*Probes: that you fit a scaling law empirically three ways, not derive it from theory.*

**D2. Kaplan said grow the model; Chinchilla said grow both. Why did they disagree?**

Both fit power laws; they got different *allocation* answers because of methodology. Kaplan (2020)
concluded `N ∝ C^0.73` — spend most of a new budget on a bigger model, barely grow the data — which
shaped GPT-3 (175B on only ~300B tokens, under 2 tokens/param). Chinchilla (2022) redid it and got
`N ∝ C^~0.5` — parameters and tokens scale *equally*, ~20 tokens/param. The discrepancy was that
Kaplan used a largely *fixed* learning-rate schedule and step count across model sizes, which
systematically *under-trained* the smaller models and biased the fit toward "bigger is better."
Chinchilla varied token counts properly and tuned the LR schedule to each run's length. The verdict
was stark: their 70B Chinchilla, trained compute-optimally on 1.4T tokens, *beat* the 280B Gopher
trained on 300B — a 4× smaller model won because it saw the right amount of data. GPT-3-era models
were over-parameterized and under-trained.
*Probes: that the "law" is only as good as the training discipline behind the fit.*

**D3. What do the exponents in `L(N,D) = E + A/N^α + B/D^β` mean, and why do both terms decay at comparable rates?**

`E` is the irreducible loss — the entropy of the data itself, the floor no model can beat (Chinchilla
fit ≈ 1.69 nats/token). `A/N^α` is the penalty for a model too small to represent the function; `B/D^β`
is the penalty for too little data. `α` and `β` are the power-law exponents governing how fast each
penalty shrinks as you grow parameters or tokens (Chinchilla fit α ≈ 0.34, β ≈ 0.28). They come out
*comparable* — neither dominates — which is precisely why the compute-optimal allocation grows both
together: substitute `D = C/(6N)` and minimize, and you get `N_opt ∝ C^(β/(α+β)) ≈ √C` and
`D_opt ∝ √C`, i.e. roughly 20 tokens/param. If one exponent were far larger, you would pour the budget
into whichever resource shrinks loss faster. (Note the Chinchilla α,β differ from the Kaplan
single-variable exponents α≈0.076, β≈0.095 — the two papers parameterize differently, part of why they
disagreed. Don't overread the exact constants; the *structure* is the durable insight.)
*Probes: reading the parametric form as "floor plus two decaying penalties" and why √C falls out.*

**D4. What are the ways extrapolating a scaling law loses you money?**

Four classic pitfalls. **Over-reaching the measured range:** a fit is trustworthy within and modestly
beyond where you measured; predicting 100× past your largest run is faith — hold out your largest
affordable point and check the fit predicts it before betting further. **A changing recipe:** the law
is fit for a *fixed* architecture, data distribution, optimizer, and schedule; change the data mix,
tokenizer, or LR schedule and you are on a different curve, so you cannot mix runs with different data
into one fit (better data shifts the whole line *down*, which is exactly why data work pays). **Under-
tuned small runs:** the Kaplan mistake in miniature — if your small models use a schedule tuned for
large ones, the small end is biased and the extrapolation tilts, so tune per run length. **Reading the
wrong bowl-bottom:** IsoFLOP bowls are flat near the minimum, so the naive lowest *sampled* point is
noisy — fit the parabola.
*Probes: the specific failure modes, especially "different data = different curve."*

**D5. You have a fixed FLOPs budget for a real run. Walk me through planning it.**

State the budget as `C` FLOPs. If I have scaling-law fits for this recipe, read `N_opt ∝ C^a` and
`D_opt = C/(6 N_opt)` straight off, sanity-checking that all three fitting methods agreed and that the
target `C` is not wildly beyond my measured range — if it is, I run a hold-out point first. Then I ask
the question the Chinchilla optimum ignores: *is this compute-optimal or deployment-optimal?* If the
model will be served heavily, especially on-device, I deliberately *overtrain* a smaller model past
its Chinchilla point — the loss bowl is flat near the minimum, so shifting toward a smaller,
cheaper-to-serve model costs almost nothing in quality while cutting inference cost on every forward
pass forever (LLaMA 3 8B on ~15T tokens, ~2000 tokens/param, is the example). Finally I budget for
reality: real runs lose time to restarts, data loading, and evaluation, and the recipe must stay fixed
or the fit no longer applies.
*Probes: turning a budget into `(N, D)` and immediately flagging deployment-vs-compute-optimal.*

---

## Part E — Data

**E1. Walk me through the Common Crawl pipeline from WARC to trainable tokens.**

The stages, each dropping a large fraction: **(1) Text extraction** — pull real content out of raw
WARC HTML (extract from WARC, *not* the crude WET files; the difference in the final model is large),
using tools like resiliparse/trafilatura, dropping nav, ads, boilerplate. **(2) Language ID** — score
each doc with a fastText language classifier and keep the languages you want above a threshold.
**(3) Quality filtering** — Gopher/C4 heuristics (symbol-to-word ratio, mean word length, fraction of
lines ending in punctuation, duplicate-line fraction) *plus*, increasingly, a fastText classifier
trained to distinguish curated high-quality text from raw web, which substantially outperforms
heuristics; plus a toxicity/NSFW filter. **(4) Deduplication** — exact (hash lines/docs) and fuzzy
(MinHash + LSH). **(5) PII removal** — detect and mask emails, phone numbers, IPs. **(6)
Decontamination** — remove docs overlapping your eval benchmarks by n-gram overlap so eval measures
capability, not memorization. Dedup and model-based quality filtering are the highest-return stages.
The order matters — you dedup and decontaminate after you have clean text to compare.
*Probes: knowing the full pipeline as an ordered sequence, not a bag of tricks.*

**E2. Explain MinHash + LSH dedup and the S-curve. What are you tuning?**

The goal is to find document pairs with high **Jaccard similarity** (|intersection| / |union| of their
n-gram sets) without the O(n²) all-pairs comparison over billions of docs. **MinHash:** represent each
doc as its set of word n-grams, apply many independent hash functions, and for each keep the *minimum*
value. The key theorem: the probability two docs share a given MinHash value equals their Jaccard
similarity — so a signature of `k` MinHashes estimates Jaccard as the fraction of matching entries, an
unbiased estimate, and each doc is now a fixed-length signature. **LSH (banding):** split the `k`-entry
signature into `b` bands of `r` rows each (`k = b·r`); two docs are *candidates* if they match exactly
in at least one band. The probability a pair with true Jaccard `s` becomes a candidate is
$1 - (1 - s^r)^b$, an S-curve in `s`. You tune `b` and `r` to place the steep threshold where you want
the cutoff (e.g. catch pairs above ~0.8, skip below): raising `r` sharpens and raises the threshold
(fewer false positives, more misses), raising `b` lowers it. Only candidate pairs get an exact
comparison, collapsing the quadratic blowup. That single S-curve is the whole tuning story.
*Probes: the Jaccard-equals-collision theorem and the b/r knobs on the S-curve.*

**E3. How does classifier-based quality filtering work, and what is its risk?**

You train a lightweight classifier — typically fastText, because it is fast enough to run over
petabytes — to distinguish "high quality" from "random web." Positives are drawn from curated
high-quality sources (Wikipedia-referenced pages, OpenWebText-style links, a known-good corpus);
negatives from raw Common Crawl. Then you keep documents the classifier scores above a threshold.
The empirical point is that this *substantially outperforms* heuristic rules — it is the lesson
behind DCLM and Nemotron-CC and the current frontier of open data. The risk is that the classifier
bakes in the biases of whatever you *called* "high quality": if your positive set is Wikipedia-like,
you quietly narrow the distribution toward encyclopedic prose and penalize legitimate text (forums,
dialects, code, non-Western sources) that simply does not look like the reference. Filtering is
destructive and you cannot undo it, so you inspect samples of what you keep *and* drop, because a
too-aggressive filter can silently delete all your code and math.
*Probes: knowing the method beat heuristics, and that "quality" is a value judgment with a cost.*

**E4. What is data mixing, and why is the mixture a tuned hyperparameter with a schedule?**

After you have clean data from many sources — web, books, code, arXiv, Wikipedia, synthetic — you
decide how much of each to include, and those weights meaningfully move the model: more code improves
reasoning and structured output *even for non-code tasks*, more math helps quantitative reasoning, but
too much of any narrow source narrows the model. So the mixture is tuned like any other
hyperparameter, on smaller runs, using scaling-law transfer. It is not a single decision but a
*schedule* (a curriculum): front-load broad filtered web in pre-training, raise the fraction of
high-quality curated / math / code toward mid-training, and often finish with an **annealing** phase
on a small, very-high-quality set. The three-stage view (pre-, mid-, post-training) has a *rising*
quality bar, and the same raw source can feed different stages at different filtering thresholds.
Domain sources also get domain handling — GitHub code needs license filtering (permissive only) and
its own dedup.
*Probes: mixture as a tuned, scheduled hyperparameter, not a fixed recipe.*

**E5. Quality versus quantity — how do you think about the tradeoff, and where does dedup sit?**

The field's verdict is that data *quality* moves the loss curve more than almost any architectural
change — two models with equal architecture and compute but different data are not close, which is why
labs guard data pipelines more than architecture. But you also need *volume*: trillions of tokens for
pretraining, so the pre-training stage runs the most permissive quality bar precisely because you
cannot be too picky at that scale. Deduplication is the elegant resolution of the tension: it improves
*quality* (heavy duplication drives memorization and degrades loss-per-unique-token) while *reducing*
wasted quantity (you stop re-learning the same article thousands of times), so it improves models at
fixed compute across multiple papers — a rare stage that costs you nothing to want. The general
instinct: aggressive dedup and model-based filtering first (highest return), then worry about squeezing
more raw volume.
*Probes: that quality dominates but volume is a floor, and dedup improves both.*

**E6. How does contamination sneak into a corpus despite a decontamination stage, and how do you catch it?**

Decontamination removes documents that overlap your *known* benchmarks by n-gram overlap — but it only
catches what you thought to check against. Contamination sneaks in through benchmarks you did not
decontaminate against, through *paraphrased* or reformatted versions of test items that dodge exact
n-gram matching, through synthetic data generated by a model that itself memorized the benchmark, and
through benchmarks released *after* your decontamination pass. Detection has two families: statistical
**exchangeability** tests — a model that memorized a benchmark assigns systematically higher likelihood
to the canonical ordering of examples than to a shuffled ordering, which a clean model would not — and
simply *encouraging providers to disclose* their measured train/test overlap. Your own defense is a
private held-out set created *after* the training cutoff that never touches training or tuning, plus
treating suspiciously high public-benchmark scores with skepticism. The moment an eval influences your
decisions, it starts leaking into the model through your choices.
*Probes: that decontamination is incomplete by construction and how exchangeability tests work.*

---

## Part F — Evaluation

**F1. What does perplexity measure, and what are its limits?**

Perplexity is the exponential of the average negative log-likelihood per token on held-out text —
lower is better, and intuitively a perplexity of `p` means the model is on average as uncertain as if
choosing uniformly among `p` options at each position. It is the direct summary of the training
objective, so it is cheap, continuous, and the right thing to watch during training and to compare
checkpoints or siblings of the *same* model. Its limits: it is only comparable across models with the
*same tokenizer* (perplexity is per-token, and different tokenizers cut text into different token
counts, so you must at least normalize per-byte or per-word to compare across families); it does not
directly measure downstream task quality; and it says nothing about instruction-following or reasoning.
Use it to track a run and rank siblings, never to rank different model families off the shelf.
*Probes: the per-token, same-tokenizer caveat, the most common perplexity misuse.*

**F2. How is a multiple-choice benchmark like MMLU actually scored, and why does the protocol change rankings?**

You do not usually let the model free-generate. The common protocol scores by *log-likelihood*: for
each option, compute the likelihood the model assigns to it, and pick the highest. But there are
distinct variants that give *different numbers on the same model*: score the likelihood of the answer
*letter* ("C"), or of the answer *text* ("Paris"), or *generate* an answer and parse it — and
text-likelihood favors different models than letter-likelihood, while generate-and-parse depends on
formatting robustness. Length-normalizing the option likelihood (dividing by token count) changes
rankings *again*, because otherwise longer options are penalized simply for having more tokens. This
is why two papers can differ mostly in *scoring protocol* rather than model quality, and why harnesses
like the EleutherAI LM-Eval-Harness and HELM exist — to fix a prompt, few-shot format, and scoring
rule per task so two models are at least measured the same way. When you see an MMLU number, ask how it
was scored before believing a comparison.
*Probes: that MC is log-likelihood-scored and the protocol is a hidden confound.*

**F3. Why do base and chat models need different evals?**

A base model is a next-token predictor; a chat model has been aligned to follow instructions, so they
fail different ways and you measure them with different instruments. Base models are evaluated on
knowledge/reasoning benchmarks scored by option likelihood (MMLU and its harder successors MMLU-Pro,
GPQA, HLE) and on checkable math/code (GSM8K, MATH, HumanEval) — the log-likelihood scoring works
because you are probing what the model *knows*. Chat models need to be judged on open-ended,
instruction-following behavior a base model cannot even format: Chatbot Arena (real users vote pairwise,
aggregated to Elo — the closest thing to ground-truth human preference at scale, but slow, uncontrolled,
and gameable by style), and cheaper reproducible proxies MT-Bench and AlpacaEval scored by an LLM
judge. Running a base-model MMLU protocol on a chat model, or an open-ended judge eval on a base model,
measures the wrong thing.
*Probes: the base-vs-chat split and the right instrument for each.*

**F4. What are the biases of LLM-as-judge, and how do you control each?**

LLM judges scale open-ended evaluation and correlate reasonably with human preference, but they have
systematic biases you must design around. **Position bias:** they favor the first (or second) option
in a pairwise comparison — control by randomizing order and averaging both orderings. **Length bias:**
they favor longer answers regardless of quality — control by matching or controlling for length.
**Self-preference:** a judge favors outputs from its own model family — control by using a judge from a
different family, or an ensemble. They can also be gamed by confident formatting and markdown. The
general discipline: use LLM judges for *relative* comparison and coarse signal, calibrate them against
human ratings on a sample, and never treat their scores as ground truth. A cheap sanity check is to
slip in a few items where you know the answer and confirm the judge gets them right.
*Probes: naming position/length/self-preference and the specific control for each.*

**F5. Explain Chatbot Arena and Elo. Why is it valued despite its flaws?**

In Chatbot Arena, real users submit their own prompt, receive two anonymous model responses, and vote
which is better; the pairwise votes are aggregated into an Elo-style rating (the chess system: each
model has a rating, a win against a higher-rated model moves you up more, and the ratings converge to
reflect pairwise win probabilities). It is valued because it is the closest thing to a *ground-truth
human-preference* signal at scale, on *real* prompts users actually cared about — unlike a fixed quiz,
it reflects genuine, diverse, information-seeking use. Its flaws: it is slow, uncontrolled (users pick
their own prompts, so the distribution drifts and cannot be held fixed), and gameable by *style* — a
chattier, more confidently formatted model can win votes without being more correct. So you read it as
a strong human-preference proxy, not as a capability measurement, and you pair it with checkable
benchmarks that Arena cannot provide.
*Probes: understanding Elo aggregation and why "real prompts, human votes" is both its strength and its weakness.*

**F6. Why do benchmarks saturate, and what does the field do about it?**

A benchmark discriminates between models only while they score meaningfully below the ceiling. Once
frontier models cluster near 100%, the remaining gap is noise, formatting, and mislabeled items rather
than capability, so the benchmark stops *ranking* — it saturates. Two things drive it: models genuinely
get better, and contamination inflates scores on public benchmarks that leaked into training. The field
responds with an arms race of harder successors: MMLU saturated, so MMLU-Pro widened each question to
10 choices to cut guessing, GPQA moved to PhD-level Google-proof expert-written questions, and HLE
("Humanity's Last Exam") pushes ~2,500 hard, often multimodal, frontier questions. The deeper fix is to
stop trusting leaderboard rank and evaluate on your *own* realistic, information-seeking task
distribution with a private held-out set — and to report capability alongside cost, since a marginally
better but 10× pricier model is not obviously better for a product.
*Probes: saturation as loss of discriminative power, the successor arms race, and the real fix.*

---

## Part G — Sampling & decoding

**G1. Greedy versus sampling — when does each make sense?**

Greedy decoding takes the argmax token at every step: deterministic, reproducible, and best when there
is one correct continuation — code, math, structured extraction, anything you will verify against
ground truth. Its weakness is that it is repetitive and bland on open-ended text and can get stuck in
loops, because always taking the single most likely token collapses diversity and can walk into
degenerate high-probability cycles. Sampling draws from the model's distribution, which gives variety
and more human-like text for creative and conversational generation, at the cost of determinism and
occasional low-probability mistakes. The rule of thumb: greedy (or low-temperature) for tasks with a
right answer, sampling for tasks where diversity is the point — and for RL rollouts and self-consistency
you *want* sampling, because you need multiple distinct completions per prompt.
*Probes: matching the decode strategy to whether the task has a verifiable answer.*

**G2. Explain temperature, top-k, top-p, and min-p, and how they interact.**

All reshape the next-token distribution before sampling. **Temperature** `T` divides the logits before
softmax: `T < 1` sharpens (more greedy, safer), `T > 1` flattens (more diverse, riskier), `T → 0` is
greedy. **Top-k** keeps only the `k` highest-probability tokens and renormalizes — a fixed-count
truncation. **Top-p (nucleus)** keeps the smallest set of tokens whose cumulative probability exceeds
`p` and renormalizes — an *adaptive* truncation that keeps few tokens when the model is confident and
many when it is unsure, which is why it usually beats top-k. **Min-p** keeps tokens whose probability
is at least `min_p × (max token probability)` — scaling the threshold to the peak, so it is
permissive when the distribution is flat and strict when there is a clear favorite. They *compose* and
order matters: you typically truncate first (top-k/top-p/min-p) then apply temperature to what
remains, and stacking an aggressive temperature on top of a wide nucleus can reintroduce the junk the
nucleus was meant to cut.
*Probes: the mechanics of each truncation and that they combine, with top-p's adaptivity as the key insight.*

**G3. What do repetition and frequency penalties do, and when do they backfire?**

They fight the degenerate looping that greedy and low-temperature decoding fall into. A **repetition
penalty** divides (or subtracts from) the logit of any token that has already appeared, making it less
likely to be repeated. A **frequency penalty** scales the penalty by *how many times* the token
appeared, and a **presence penalty** applies a flat penalty once a token has appeared at all. They
backfire when the task legitimately requires repetition: code has many repeated tokens (`for`, `=`,
indentation, variable names), structured output repeats keys and delimiters, and a language with
limited vocabulary naturally reuses words — an aggressive penalty there degrades correctness or
produces contorted phrasing as the model avoids the natural token. So they are a creative-text tool;
turn them off (or way down) for code and structured extraction.
*Probes: knowing the penalties and that they are harmful on repetitive-by-nature tasks.*

**G4. What is beam search and why do LLMs rarely use it?**

Beam search keeps the `b` highest-probability *partial sequences* at each step, expanding all of them
and pruning back to the top `b`, to approximate the globally most-likely sequence rather than the
greedy locally-most-likely one. It was standard in machine translation, where there is a single best
faithful output and a slightly higher-probability full sequence is genuinely better. LLMs rarely use
it for two reasons. First, on open-ended generation, maximizing sequence probability produces bland,
repetitive, degenerate text — the highest-probability continuation is often the most generic one, and
beam search *amplifies* that pathology. Second, it is expensive (`b` parallel hypotheses) and interacts
badly with sampling. For open-ended text, nucleus sampling gives better output than chasing the mode;
for verifiable tasks, greedy or best-of-n sampling with a verifier beats beam search. Beam survives
mainly in constrained/structured decoding where the objective really is the most probable valid string.
*Probes: that "most probable sequence" is the wrong objective for open-ended LM generation.*

**G5. How do you make sampled generation reproducible, and why might it still drift?**

Fix the random seed for the sampler so the same logits produce the same draws, and pin temperature,
top-k/top-p, and any penalties — with a fixed seed and greedy (`T=0`) decoding you should get identical
output run to run. But it can still drift for reasons *outside* the sampler: floating-point
non-associativity means that changing the batch size, sequence padding, kernel, GPU model, or library
version reorders the reductions inside matmuls and softmax, producing slightly different logits, and
near a tie those tiny differences flip the argmax and the whole continuation diverges. Continuous
batching makes this worse because a request's effective batch composition varies with what else is
in flight. So bit-exact reproducibility across hardware/engine versions is not guaranteed even at
`T=0`; for true determinism you fix the seed *and* pin the batch shape, kernels, and versions, or
accept run-to-run variation as a property of the serving stack.
*Probes: seeding for the sampler plus the deeper floating-point/batching source of non-determinism.*

---

## Part H — Long context

**H1. Why does naively feeding a longer context to a model trained on short context fail?**

Two failures. First, **positional**: RoPE encodes position as a rotation angle `θ_{i,k} = i / Θ^{...}`
that grows with absolute position `i`. A model trained to 4k has only ever seen rotation angles up to
that range, so at position 20k the query/key rotations are in a regime it never learned — the relative-
position signal the attention dot product depends on is out of distribution, and quality collapses.
Second, **the KV cache**: cost grows linearly with sequence length, so at long context one sequence's
cache can exceed the model weights (a 30-layer, 32-KV-head, d_head-128 bf16 model spends ~491 KB per
token — ~4 GB at 8k, ~10 GB at 32k), which caps batch size and can OOM you before quality even matters.
So you cannot just pass more tokens; you have to *extend* the position encoding (interpolation, below)
and *manage* the cache (GQA/MLA, KV quantization, local attention). Long-context
models "bump `Θ` or interpolate positions" as a fine-tuning-time trick on the same RoPE mechanism.
*Probes: naming both the out-of-distribution position problem and the KV-memory wall.*

**H2. What is position interpolation, and how do NTK-aware / YaRN scaling improve on it?**

The RoPE angle grows with position, so to stretch a 4k model to 32k you can **interpolate positions**:
scale every position index down by the extension factor (here 8×) so position 32k maps back into the
0–4k *angle* range the model was trained on — trading angular resolution for reach — then briefly
fine-tune. Plain linear interpolation works but uniformly compresses all frequencies, blurring the
high-frequency (fine-grained, local) position signal the model relies on for nearby tokens.
**NTK-aware scaling** fixes this by changing the RoPE base `Θ` instead of the positions, scaling
frequencies *non-uniformly* — stretch the low-frequency (long-range) components a lot and the
high-frequency (local) components little, so local precision is preserved while range extends.
**YaRN** refines this further with a frequency-dependent interpolation schedule (per-band ramp) plus
an attention-temperature adjustment, reaching longer contexts with less fine-tuning and less quality
loss than either plain interpolation or naive NTK. The through-line: extend where you can afford to
lose resolution (long range), preserve it where you cannot (local).
*Probes: interpolation as trading angular resolution for reach, and why NTK/YaRN scale frequencies non-uniformly.*

**H3. What is the needle-in-a-haystack eval, and what does it and its variants actually test?**

You plant a specific fact (the "needle" — e.g. a random sentence with a magic number) at a controlled
depth inside a long distractor context (the "haystack"), then ask a question only answerable from the
needle, sweeping both context length and needle depth. It tests *retrieval* over long context: can the
model actually attend to and use information anywhere in the window, or does it only see the start and
end (the "lost in the middle" failure, where accuracy sags for needles buried mid-context)? Its limit
is that single-needle retrieval is *easy* — a model can pass it while still failing real long-context
reasoning — so harder variants (multi-needle, needle requiring aggregation across several planted
facts, or reasoning over the retrieved content) are used to distinguish "can find one fact" from "can
reason over the whole context." Passing single-needle is necessary, not sufficient, evidence of usable
long context.
*Probes: knowing the eval construction and that it measures retrieval, not long-context reasoning.*

**H4. What are attention sinks and streaming attention, and what problem do they solve?**

Serving an effectively infinite stream (a long chat) means you cannot keep the whole KV cache — it
grows without bound — so the obvious move is a sliding window that evicts the oldest tokens. Naively
doing that *tanks* quality, and the reason is the **attention sink**: models learn to dump excess
attention probability onto the very first few tokens (softmax must sum to 1, so when no later token
deserves the mass it parks it on the initial tokens as a no-op). Evict those initial tokens and the
softmax distribution is corrupted, so quality collapses. **StreamingLLM's** fix is to *always retain*
the first few "sink" tokens' KV plus a sliding window of recent tokens, discarding the middle. This
keeps the sink the model relies on while bounding the cache, giving stable generation over arbitrarily
long streams without fine-tuning — though note it *forgets* the evicted middle, so it enables endless
*streaming*, not true long-context *recall*.
*Probes: the counterintuitive sink phenomenon and that streaming ≠ long-context recall.*

**H5. At long context, what dominates inference cost — and which levers attack it?**

The **KV cache**, not the weights. It grows linearly with sequence length and batch ($2 \cdot n_{\text{layers}} \cdot n_{\text{kv\_heads}} \cdot d_{\text{head}} \cdot L_{\text{seq}} \cdot \text{batch} \cdot \text{bytes}$), so at long context a single sequence's cache can exceed
the model weights and, across concurrent users, becomes the binding memory constraint that caps batch
size — and since decode is memory-bandwidth-bound, reading that cache also eats bandwidth. The levers
all attack terms in that formula. **GQA/MQA** shrink `n_kv_heads`, cutting the cache by the query-to-KV
ratio. **MLA** compresses the per-token key/value into a small latent, a larger constant-factor
reduction. **Local (sliding-window) attention** interleaved with occasional global layers makes most
layers' cache independent of sequence length. **KV-cache quantization** to int8/int4 attacks the
bytes-per-element term directly. **PagedAttention** does not shrink the cache but eliminates
fragmentation so more of it fits. At long context, KV-cache economics *is* the inference-design problem.
*Probes: that KV cache, not weights, dominates at long context, and mapping each lever to the formula.*

---

## Part I — Alignment mechanics

**I1. Why does SFT mask the prompt, and what exactly is the masked loss?**

You fine-tune on prompt-response pairs with the next-token objective, but you compute loss *only on
the response tokens*. Mechanically you build a `response_mask` that is 0 over prompt tokens and 1 over
response tokens, then average the per-token NLL only over the masked positions:
$\text{loss} = - \sum_t \text{mask}_t \cdot \log p_\theta(y_t \mid y_{<t}) \,/\, \sum_t \text{mask}_t$. The reason: you do not want to spend model
capacity learning to *generate the user's question* — that is input you will always be given, not
behavior you want to produce — only to generate the answer *conditioned on* it. Training on the prompt
tokens would waste gradient on modeling the instruction distribution and can actively hurt. This pairs
with **chat templates**: the pair is wrapped in the model's role markers and turn delimiters, and for
reasoning models structural tags like `<think>...</think>` / `<answer>...</answer>` — which is not
cosmetic, because the reward function later *parses those tags*, so SFT is teaching the exact structure
the reward will grade.
*Probes: the masked-NLL form and *why* the prompt is excluded, plus the template-reward link.*

**I2. How is a reward model trained from preference pairs?**

From comparisons. You collect, for a prompt `x`, a preferred response `y_w` and a dispreferred `y_l`
(judged by humans or an AI), and train a scalar reward model `R(x, y)` under the **Bradley-Terry**
model, which says the probability a human prefers `y_1` over `y_2` is the sigmoid of the reward
difference: $P(y_1 \succ y_2 \mid x) = \sigma(R(x, y_1) - R(x, y_2))$. You fit it by maximum likelihood —
equivalently minimizing $-\log \sigma(R(x, y_w) - R(x, y_l))$ over the pairs. Architecturally it is usually
the same transformer with a scalar head replacing the vocab projection. The crucial property is that
Bradley-Terry only ever sees reward *differences* between two responses to the same prompt, so the
absolute scale of `R` is unidentified — which is exactly the property DPO later exploits to cancel the
intractable partition function.
*Probes: Bradley-Terry as sigmoid-of-difference and the difference-only property.*

**I3. In PPO-style RLHF, what is the KL-to-reference term for, and what breaks without it?**

The objective is $\max_\pi \mathbb{E}_{y \sim \pi}[R(x,y)] - \beta \cdot \mathrm{KL}(\pi(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x))$: maximize reward while staying
close to the frozen SFT reference. The KL term is the leash. Without it, optimization **reward-hacks**
— it finds degenerate outputs the reward model scores highly but humans hate, drifting arbitrarily far
from sensible language because a *learned* reward model is only accurate near the distribution it was
trained on and is exploitable off it. The KL penalty anchors the policy near its SFT starting point
where the reward model is trustworthy. `β` is the main knob in the whole pipeline: too low and you get
reward hacking and mode collapse; too high and the model barely moves off the SFT policy. It also
matters for *safety* — push the policy too far chasing reward and you can knock out lightly-reinforced
refusal behaviors.
*Probes: KL as the anti-reward-hacking anchor and `β` as the central knob.*

**I4. DPO versus PPO — what does DPO trade, and when would you still run PPO?**

DPO exploits that the KL-regularized objective has a *closed-form* optimal policy
$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \cdot \exp(R(x,y)/\beta)$. Invert it to express the implicit reward as
$\beta \log(\pi/\pi_{\text{ref}}) + \beta \log Z(x)$, substitute into the Bradley-Terry loss, and the intractable partition
`Z(x)` cancels (Bradley-Terry sees only differences) — leaving a plain supervised loss on preference
pairs, no reward model and no RL loop:
$L_{\text{DPO}} = - \log \sigma(\beta \cdot (\log \pi(y_w)/\pi_{\text{ref}}(y_w) - \log \pi(y_l)/\pi_{\text{ref}}(y_l)))$. The policy *is* the reward model.
DPO trades away PPO's separate reward model, value network, and finicky on-policy loop for a stable,
simple offline loss that reaches comparable quality on many tasks — the default now. You still reach
for PPO (or online RL) when you need *on-policy* improvement — generating fresh samples, scoring them,
and learning from the model's *own current* outputs — which DPO's fixed offline preference set cannot
give, and which matters when you want the model to explore past the demonstrated/preferred data.
*Probes: the DPO derivation at a high level and the offline-vs-on-policy tradeoff.*

**I5. What is RLVR, and why did verifiable rewards change the field?**

RLVR is reinforcement learning with **verifiable rewards**: where correctness is *checkable* — math
with a known answer, code that passes tests, structured output you can validate — you skip the learned
reward model entirely and reward the objective outcome directly (did the answer match, did the tests
pass). It changed everything because it breaks the dependence on human-labeled or learned rewards,
which are expensive, biased, and *hackable*. A verifiable reward is unlimited and incorruptible, so you
can push *hard* on it without the reward-hacking that plagues learned rewards, and the model can
improve by practicing far past the level of its demonstrations ("SFT memorizes, RL generalizes"). This
is why the reasoning frontier concentrated in checkable domains — math, code, formal tasks — and drove
the 2024–2025 reasoning-model boom via GRPO. The open problem is extending it to domains *without* a
clean verifier, where you are back to learned rewards or clever proxies. For any task with a checkable
answer (does the extracted field match ground truth), this is the move.
*Probes: verifiable = unhackable = can push hard, and the "checkable domains" restriction.*

**I6. What is reward hacking, and how does it differ between learned and verifiable rewards?**

Reward hacking is the policy finding outputs that score high under the *proxy* reward while failing the
*true* objective. On a **learned** reward it is fatal and unbounded: the reward model is only accurate
near its training distribution, so optimization drifts off into degenerate outputs the RM loves and
humans hate — this is precisely what the KL leash exists to prevent. On a **verifiable** reward it is
bounded (the model *does* have to produce a correct answer to get the reward) but it still shows up as
exploiting the *shape* of the reward: the classic case is **length** — models learn longer chains
correlate with correctness and inflate reasoning with degenerate padding — or exploiting a *lenient
answer parser* that accepts near-misses. That is why even with a verifiable reward you keep the KL
leash and the clip, design the reward as format-*plus*-correctness with a *strict* validator (a lenient
validator is a reward-hacking invitation), and watch for length blowup and padding.
*Probes: that verifiable rewards bound but don't eliminate hacking, and the length/lenient-parser failure modes.*

---

## Part J — Rapid-fire

Short probes an interviewer fires to calibrate breadth. One or two sentences each; know the *why*.

**J1. Why bytes, not characters, for BPE?** Base vocab is 256 byte values, so nothing is ever
"unknown" and every input round-trips; a character vocab is huge (~150k Unicode code points) and still
splits emoji/CJK poorly.

**J2. Why RMSNorm over LayerNorm?** Mean-centering is unnecessary, so dropping it is cheaper with no
quality loss.

**J3. Why pre-norm over post-norm?** It keeps the residual path an identity, making deep stacks
trainable without warmup gymnastics.

**J4. Why is `d_ff ≈ (8/3)·d` for SwiGLU?** SwiGLU uses three weight matrices instead of two, so
shrinking the hidden size to 8/3·d keeps the FFN parameter count matched to a standard 4·d ReLU MLP.

**J5. Why does RoPE rotate q and k but never v?** Position must enter the *score* (the q·k dot product,
which becomes relative-offset-dependent); v is the content being mixed and carries no position.

**J6. Total vs active params in an MoE — which goes in 6ND?** Active. FLOPs track what runs per token,
not what is stored.

**J7. Why does int4 weight-only quantization speed up decode ~4×?** Decode is memory-bound on reading
weights; a quarter the bytes means ~a quarter the weight-read traffic.

**J8. Why is decode memory-bound but prefill compute-bound?** Prefill processes all prompt tokens in
one high-intensity batched matmul; decode reads the whole weight matrix to produce a single token
(intensity near the batch size).

**J9. Why is speculative decoding lossless?** The target verifies drafted tokens in one parallel pass
and a modified rejection rule accepts/corrects them so the output distribution is exactly the target's.

**J10. What determines on-device tokens/sec?** Memory bandwidth ÷ (bytes-per-weight × active
parameters) — bandwidth-bound, which is why quantization is the lever.

**J11. Why is activation quantization harder than weight quantization?** Activations have outlier
channels with huge magnitudes; one scale either clips them or crushes everything else. Weights are
better-behaved.

**J12. Chinchilla's tokens-per-parameter?** ~20. But deployment models overtrain far past it (LLaMA 3
8B ≈ 2000) to cut lifetime inference cost.

**J13. Why does compute-optimal differ from deployment-optimal?** Chinchilla minimizes *training* cost;
serving pays inference on every forward pass forever, so a smaller overtrained model is cheaper overall.

**J14. What's the highest-return data stage?** Deduplication and model-based (classifier) quality
filtering.

**J15. What does the fuzzy-dedup S-curve $1-(1-s^r)^b$ control?** The Jaccard-similarity threshold
above which document pairs become candidates; tune `b`, `r` to place the steep cutoff.

**J16. Why extract from WARC, not WET?** WET's crude pre-extracted text is low quality; re-extracting
main content from raw HTML materially improves the final model.

**J17. Why can't you compare perplexity across model families?** It is per-token and tokenizer-
dependent; different tokenizers cut text into different token counts.

**J18. Why does the multiple-choice scoring protocol matter?** Letter-likelihood vs text-likelihood vs
generate-and-parse, and length normalization, all change rankings on the *same* model.

**J19. Top-p over top-k — why?** Top-p is adaptive: few tokens when the model is confident, many when
unsure, matching the truncation to the distribution's shape.

**J20. When do you turn off repetition penalties?** Code and structured output, which legitimately
repeat tokens (keywords, delimiters, keys).

**J21. Why do LLMs avoid beam search?** Maximizing sequence probability yields bland, degenerate text
on open-ended generation; the mode is not what you want.

**J22. What's an attention sink?** Models park excess softmax mass on the first few tokens; evicting
them in a sliding window corrupts the distribution, so StreamingLLM keeps them.

**J23. What does needle-in-a-haystack test — and not test?** Long-context *retrieval* of one fact; it
does *not* test long-context reasoning or aggregation.

**J24. Why mask the prompt in SFT?** You want capacity spent on generating the answer given the
question, not on generating the question.

**J25. In DPO, what is the implicit reward?** $\beta \cdot \log(\pi/\pi_{\text{ref}})$ — the policy *is* the reward model,
which is how the reward model and RL loop are removed.

**J26. What does GRPO drop versus PPO?** The learned value network; the group's own reward mean/std is
the baseline instead.

**J27. What is the KL leash's job?** To anchor the policy near the SFT reference and prevent reward
hacking; `β` is the main alignment knob.

**J28. Why did MoE routing collapse without balancing?** Selection is self-reinforcing — an expert
picked early trains more, gets better, gets picked more — until a few experts hog all traffic.

**J29. Where do a small model's parameters mostly live?** The embedding + output head (`V×d` each),
which don't shrink with the transformer body — hence small vocab for small models.

**J30. What separates two same-size models today?** Data quality and post-training, not architecture —
architecture has largely converged on the standard pre-norm decoder.

---

## How to practice

Breadth rounds reward *coverage* and *speed of retrieval*, so practice differently from the deep-dive
banks. Take one Part at a time, close the answers, and try to give the two-sentence core of each
question in under thirty seconds — the interviewer is sampling how much of the field you hold and how
fast you can reach it, then following up on whatever you fumble. For the rapid-fire round, drill until
the *why* comes out with the *what*: "top-p over top-k — because it's adaptive to the distribution's
shape" is a pass; "top-p is better" is a fail. When you can answer a Part cold, have someone (or a
model) fire the questions out of order and follow each answer with "why?" one more level down, because
the real breadth interview is not the first question — it is the third follow-up that finds the edge of
what you actually understand. Then go back to the earlier interview banks and notice how these
breadth facts are the foundation the deep questions stand on.
