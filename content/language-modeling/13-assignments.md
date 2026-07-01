# 13 — The Five Builds

Reading the chapters gives you the map. These five builds are the territory. They are ordered so
each produces a component you reuse in the next. Do them in order. Each is scoped so it runs on
modest hardware; scale up only if you want to.

Structure each build as a `uv`-managed repo with a `pytest` **adapter** pattern: keep the tests
fixed, and implement functions in `tests/adapters.py` that wire your code into them, so
`uv run pytest` is your green/red signal throughout. Several builds end in an optimization target —
a held-out metric you drive down. Track runs in **wandb**. Below, each build lists what you
implement, what is genuinely hard, and the deliverable.

For every build: write it from primitives where the point is understanding (do not call the library
function you are supposed to be implementing), test correctness against the adapter suite before you
optimize, and keep a short log of what you measured. The log is where the learning consolidates.

## Build 1 — Basics: BPE, Transformer, training loop

**Goal.** Every core component of an LM from scratch: a byte-level BPE tokenizer, a decoder-only
Transformer, AdamW, and a training loop — then drive down validation perplexity under a fixed
compute budget.

**Do.**
- **BPE training:** byte-level initialization, pre-tokenization with a regex split, frequency
  counting of adjacent pairs, iterative merging to a target vocab, recording ordered merges. Then
  `encode` (apply merges in learned order) and `decode` (ids → bytes → UTF-8), plus special tokens
  (at least end-of-document) and serialize/reload.
- **The Transformer block:** RMSNorm, causal multi-head attention with RoPE, SwiGLU
  MLP, pre-norm residuals, final norm, output projection. No `torch.nn.Transformer`, no
  `torch.nn.MultiheadAttention` — write the attention math yourself.
- **Optimizer and loop:** AdamW from the update rule, cross-entropy next-token loss, learning-rate
  warmup and cosine decay, gradient clipping, gradient accumulation, checkpoint save/load.
- Train on **TinyStories** (coherent output appears fast) and the **OpenWebText** sample, using your
  own tokenizer.

**What's hard.** Making BPE training fast enough on a real corpus — naive pair recounting on every
merge is quadratic; you must cache pair counts and update only the counts a merge touches. And
getting the attention/RoPE/norm details exactly right so the adapter tests pass numerically. Then
the fixed compute budget turns it into an optimization problem: best validation loss under that
budget, which forces you to actually tune.

**Deliverable.** Code passing the `uv run pytest` adapter suite (tokenizer round-trips losslessly,
block matches reference tensors) plus your best OpenWebText perplexity under the compute budget.
Report compression (bytes/token) and fertility (tokens/word) in- and out-of-domain — tokenize some
Urdu or code and watch fertility blow up; that is the lesson.

## Build 2 — Systems: FlashAttention2, fused kernels, DDP

**Goal.** Take the Build-1 model and make it fast, by attacking the memory-bound parts and by
distributing it. Start from your Build-1 code and add a systems layer on top of it.

**Do (kernels).**
- **Benchmark and profile** the model to find the memory-bound hot spots (elementwise chains,
  attention) — produce timings, do not guess.
- Write a **fused Triton kernel** for something memory-bound and confirm the speedup.
- Implement **FlashAttention-2 forward *and* backward in Triton:** tile queries and keys/values, use
  the online softmax with running max and normalizer, never materialize the full score matrix. The
  backward is the hard, graded part — you derive and implement the recomputation. Verify it matches
  your Build-1 attention numerically.

**Do (distribution).**
- Implement **distributed data parallel (DDP)** with PyTorch process groups: run the collectives
  (`all_reduce`, `all_gather`, `reduce_scatter`) yourself, then build a DDP wrapper that all-reduces
  gradients, and benchmark naive-per-parameter vs bucketed/flattened communication.
- Implement **optimizer state sharding** (the ZeRO-1 idea): shard optimizer state across ranks and
  watch per-device memory drop. If you have two GPUs, run it for real; if one, reason through what
  is sharded and gathered when.

**What's hard.** The Triton backward for FlashAttention — the index math and the recompute — is the
crux of the whole course's systems content. Getting collectives and gradient bucketing correct so
DDP actually matches single-GPU training (and is faster) is the other.

**Deliverable.** Code passing the systems test suite, plus benchmarks: fused-kernel and
FlashAttention speedup vs naive as sequence length grows (the win grows with `L`), peak memory
before/after (quadratic → linear in `L`), and DDP scaling with the bucketing optimization.

## Build 3 — Scaling laws

**Goal.** Predict the compute-optimal model size for a budget you do not run, by fitting a scaling
law from small runs — against a real training API with a hard FLOPs budget.

**Do.**
- Query a **training API** (a service that runs the training and returns losses) under a fixed total
  **FLOPs budget** — you cannot just run everything, so you must *choose* your queries.
- Build **IsoFLOP profiles:** at each of a few fixed compute budgets, train several model sizes, each
  on the token count that keeps `C ≈ 6ND` constant, and find the bottom of each loss-vs-size bowl
  (the compute-optimal size for that budget).
- **Fit** how the optimal size (and the optimal tokens-per-parameter ratio) scales with compute, and
  **extrapolate** to a larger target budget you were never allowed to run.

**What's hard.** It is a constrained experimental-design problem, not just curve-fitting: the budget
means every run you spend on a bad configuration is gone, so you must plan the IsoFLOP grid to
extract the law from sparse, noisy points. Fitting the bowl minima and the power law through them
robustly is the graded skill.

**Deliverable.** A fitted prediction for the compute-optimal configuration at the target budget, with
your IsoFLOP curves and the fitted exponents. Expect the tokens-per-parameter ratio to be noisy and
not land exactly at 20 on a toy setup — the point is the method and seeing the bowl appear.

## Build 4 — Data: filter and dedup Common Crawl

**Goal.** Turn raw web crawl into training-quality tokens, and prove it by training on your output.

**Do.**
- Work from **Common Crawl WET files** (extracted text). Implement, in a fresh data module:
  **language identification** (`is_english`), **quality filters** (heuristic
  Gopher/C4-style rules plus a trained **quality classifier**), **PII removal** (emails, phone
  numbers, IPs), and **deduplication** — both exact-document hashing and approximate/fuzzy
  **MinHash + LSH**.
- Emit the survivors as tokenized `*.bin` shards (GPT-2 tokenization) ready for training.

**What's hard.** Scale and reproducibility: the pipeline runs over thousands of WET files (a service
like Modal works well for distributed download and multi-GPU training on the output), so your
filters and especially MinHash dedup must be performant, and everything must be deterministic so
comparisons are fair. The judgment call — quality vs corpus size — is the real lesson: over-filter
and you starve the model, under-filter and you feed it junk.

**Deliverable.** Filtered, deduplicated, tokenized `*.bin` data, scored by the validation loss of a
model trained on it. Report how much data survives each stage and inspect what you keep and drop.

## Build 5 — Alignment: SFT, Expert Iteration, GRPO

**Goal.** Turn a base model into a math reasoner through the full post-training ladder, on
**Qwen2.5-Math-1.5B** against **GSM8K** and **MATH**.

**Do.**
- **SFT** (`algs/sft.py`): fine-tune on correct reasoning traces with the response
  masked so loss is computed only on response tokens (`masked_normalize` over a `response_mask`),
  using the R1-Zero chat template (`<think>...</think><answer>...</answer>`). Effective batch via
  gradient accumulation.
- **A reward function** (`r1_zero_reward_fn`): format reward (correct `<think>`/`<answer>`
  structure) plus answer reward (parse the answer, check math correctness); full reward 1 when both
  are right.
- **Expert Iteration:** sample several responses per prompt with **vLLM**, keep only trajectories
  with reward 1, SFT on those, repeat (default ~5 EI steps, 100 SFT steps each). Watch accuracy
  climb as the model bootstraps onto problems it could not solve before.
- **GRPO** (`test_grpo.py`, the graded core): sample a group of `G=8` responses per prompt, set each
  advantage to the group-normalized reward `A_i = (r_i − mean)/(std + ε)`, and optimize the
  per-token clipped surrogate `min(ratio·A, clip(ratio, 1±0.2)·A)` with `ratio = π/π_old`. Implement
  the naive, unclipped, and clipped modes to see the difference; compute reference/old log-probs
  under `torch.no_grad()`.
- **Optional:** DPO and the supplemental safety/RLHF track.

**What's hard.** Wiring the rollout → reward → advantage → clipped-loss loop correctly (the
`no_grad` reference log-probs, per-token loss, gradient accumulation) so `test_grpo.py` passes, and
managing the vLLM-generate / train-loop interleave. Reward design is the subtle part: a lenient
answer parser is a reward-hacking invitation.

**Deliverable.** Code passing the adapter/`test_grpo` suite, plus held-out MATH/GSM8K accuracy after
each of SFT, expert iteration, and GRPO — expert iteration should beat SFT, and GRPO should beat
expert iteration. Keep that held-out set clean the whole time.

## After the five builds

You will have built, from primitives, every major component of a modern language model: the
tokenizer, the Transformer, AdamW and the training loop, the fused Triton kernels including
FlashAttention-2 (forward and backward), DDP and optimizer-state sharding, the scaling-law
methodology, the Common Crawl data pipeline, and the full post-training ladder (SFT → expert
iteration → GRPO). At that point the frontier models stop being magic and become a known list of
engineering choices you could reproduce given the compute. That is the entire point.

The natural next step for your work specifically is to take Build 2 (kernels, memory optimization)
and Build 5's alignment ladder and apply them to your on-device VLM: fused dequant-matmul kernels for
your quantized models, and expert-iteration-style outcome training on your verifiable
structured-extraction task. Those two are where this curriculum touches your day job most directly.
