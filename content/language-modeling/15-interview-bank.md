# 15 — Interview Bank: Frontier-Lab Level

These are the kinds of questions asked for senior applied ML and inference roles at labs like
OpenAI, Anthropic, DeepMind, and the strong application layer (Fireworks, Together, Baseten). They
are not trivia. They test whether you can reason from first principles, do arithmetic under
pressure, make tradeoffs out loud, debug from symptoms, and design a system end to end. Each
question has a worked answer at the depth an interviewer wants to hear. Read the answer, then close
it and try to reproduce the reasoning, because the interview tests the reasoning, not the memorized
conclusion.

The answers assume the concepts from the earlier chapters. Where an answer names a specific number,
either it is derived on the spot or it is a known result you should be able to cite.

---

## Part A — Back-of-envelope estimation

Labs love these because they reveal whether you actually understand the cost structure or just
know vocabulary. Always narrate the assumptions.

**A1. Roughly how much compute to train a 7B model on 2T tokens, and how long on 1000 H100s?**

Training compute is `6 * N * D = 6 * 7e9 * 2e12 = 8.4e22` FLOPs. An H100 does on the order of
1e15 FLOP/s in bf16 at realistic utilization (peak is higher; assume ~40-50% MFU, so call it
~5e14 effective). 1000 of them give ~5e17 effective FLOP/s. Time is `8.4e22 / 5e17 ≈ 1.7e5`
seconds, about 2 days. State the MFU assumption explicitly because it swings the answer by 2-3x,
and note that real runs lose time to restarts and data loading. The interviewer wants to see you
carry the `6ND` rule, plug a realistic (not peak) FLOP number, and flag the utilization
assumption.

**A2. A 70B model in bf16. Will it fit for inference on one 80GB H100? For training?**

Inference weights: `70e9 * 2 bytes = 140 GB`. Does not fit on one 80GB card even before the KV
cache, so you need at least two GPUs (tensor parallel) or aggressive quantization. At int4 it is
~35GB, which fits with room for a KV cache. Training is far worse: roughly `18N`
bytes for parameters, gradients, fp32 master, and Adam state, so `18 * 70e9 ≈ 1.26 TB` before
activations, which is ~16 H100s of memory just for state. So single-GPU inference needs
quantization, and training needs FSDP sharding across many GPUs. The move is to compute the four
memory consumers out loud.

**A3. Estimate the KV cache size for a 70B model serving one user at 32k context.**

You need the config: say 80 layers, GQA with 8 KV heads, head dim 128, bf16. Per token per layer
the KV is `2 (K and V) * 8 heads * 128 * 2 bytes = 4096` bytes. Times 80 layers is ~328 KB per
token. Times 32k tokens is ~10.5 GB for a single user's cache. This is the punchline that surprises
people: at long context, one user's KV cache is a large fraction of, or exceeds, a quantized copy
of a smaller model, and it is why KV-cache quantization and GQA/MLA matter and why concurrency at
long context is memory-bound on the cache, not the weights.

---

## Part B — Architecture

**B1. Walk me through a modern decoder block and justify each choice against the 2017 original.**

Pre-norm RMSNorm, causal multi-head (really grouped-query) attention with RoPE, SwiGLU MLP,
residuals, final norm. Justifications: RMSNorm over LayerNorm because the mean-centering is
unnecessary and it is cheaper; pre-norm over post-norm because it keeps the residual path an
identity and makes deep stacks trainable without warmup gymnastics; RoPE over learned absolute
positions because it encodes relative position directly in the attention dot product, adds no
parameters, and extrapolates to longer context better; SwiGLU over ReLU-MLP because the gated
variant is consistently better per parameter, with hidden size set near `8/3 d` to match the
three-matrix parameter count; GQA over MHA because it cuts the inference KV cache by the query-to-
KV-head ratio while keeping almost all the quality. The interviewer is checking that you know not
just what but why, and that you connect GQA to inference cost unprompted.

**B2. Why does attention scale quadratically, and what have people done about it?**

The score matrix is `L x L` per head: every query attends to every key, so compute and (naively)
memory grow with `L²`. Responses fall into three buckets. First, do not change the math, just the
IO: FlashAttention tiles the computation and uses an online softmax so the `L x L` matrix never
hits HBM, cutting memory to `O(L)` and running faster, but the compute is still quadratic. Second,
approximate attention: sparse, linear, and low-rank attention variants trade exactness for
sub-quadratic cost, but they historically underperform and have not displaced full attention at the
frontier. Third, sidestep it at inference: the KV cache means decode only attends new queries
against cached keys, so per-step decode is linear in context, and MLA/GQA shrink that cache. The
honest senior answer is that FlashAttention plus GQA/MLA plus KV caching is what actually shipped,
and the sub-quadratic-attention research is promising but not yet the default.

**B3. What is MLA and why did DeepSeek use it over GQA?**

Multi-head latent attention compresses the keys and values into a low-rank latent vector, and that
compressed latent is what gets cached, then it is projected back up per head at use time. GQA
reduces the number of KV heads (coarse); MLA reduces the dimensionality of what is stored (finer),
so it shrinks the KV cache more aggressively while retaining more of full MHA's expressiveness than
GQA does at the same cache budget. DeepSeek chose it because at their scale and context lengths the
KV cache dominates inference memory and MLA gave a better quality-per-cache-byte tradeoff, and they
built FlashMLA to make it fast. Connecting it to the KV-cache-is-the-bottleneck theme is the point.

---

## Part C — Systems, GPUs, kernels

**C1. My training run is at 20% MFU. Where do I look?**

MFU is achieved FLOPs over peak, so 20% means the compute units idle 80% of the time, which
means you are memory-bound or stalled somewhere, not compute-bound. Checklist, roughly
in order: are the matmuls actually in bf16/fp8 hitting tensor cores, or accidentally fp32; is
`torch.compile` on and fusing the elementwise chains, or are RMSNorm/residual/activation each a
separate memory-bound kernel; is the batch large enough to give the GPU occupancy to hide memory
latency; is the data loader starving the GPU (CPU-bound preprocessing, check whether GPU util
drops between steps); is communication (all-reduce/all-gather) not overlapping with compute in your
distributed setup; are tensor shapes multiples of 8/16 so tensor cores are not padding. Profile
first (torch profiler or nsys), do not guess. The interviewer wants a systematic memory-vs-compute
diagnosis, not a random list.

**C2. Explain FlashAttention as if implementing it. Where do people get it wrong?**

You tile queries into blocks and keys/values into blocks. For each query block you iterate over
key/value blocks, computing partial scores on-chip, and you accumulate the output using an online
softmax that maintains a running max `m` and running normalizer `l`. When a new key block reveals a
larger max, you rescale the accumulated output and normalizer by `exp(m_old - m_new)` so the
normalization stays correct. The full `L x L` scores never touch HBM. It is exact, not
approximate. Where people get it wrong: the rescaling correction on the running max, and the
backward pass, where you recompute scores on the fly rather than storing them (trading compute for
memory, which is fine because attention was memory-bound). If you can write the running-max update
loop from memory, you are done.

**C3. When would you write a custom Triton kernel versus using `torch.compile`?**

`torch.compile` fuses most memory-bound elementwise chains automatically and is the first move.
You write a custom kernel when profiling shows a specific hot spot it did not fuse, or when you
need a fused op that does not decompose into simple pieces the compiler recognizes, like a custom
attention mask or a fused dequantize-matmul for a quantized model where you want the int4-to-bf16
dequant fused into the GEMM rather than run as a separate memory-bound pass. For most work the
order is: framework fused ops (FlashAttention via SDPA, Liger kernels), then `torch.compile`, then
profile, then hand-write only what remains. Naming the fused dequant-matmul as a real reason shows
you have done quantization work.

---

## Part D — Distributed training

**D1. A model does not fit on one GPU. Walk me up the parallelism ladder.**

First, FSDP2: shard parameters, gradients, and optimizer state across the data-parallel group so
each device holds `~18N / num_devices`, all-gathering each layer's params just in time and
discarding them after. This alone fits models many times a single GPU's memory and keeps the data-
parallel programming model. If a single layer is too big or too slow, add tensor parallelism within
a node, because it needs an all-reduce per layer and that high-volume communication must ride
NVLink, not the network. If the model is too deep to fit even sharded, add pipeline parallelism
across nodes, using micro-batches to shrink the bubble, because pipeline only passes activations
between adjacent stages so its communication is low-volume and network-tolerant. If sequences are
too long, add sequence/context parallelism. If it is an MoE, add expert parallelism and accept the
all-to-all cost. The governing principle stated explicitly: keep high-volume communication on fast
intra-node links, low-volume across nodes.

**D2. Why does FSDP save memory, and what does it cost?**

Plain data parallelism replicates the full `~18N` bytes of training state on every device, which is
wasteful. FSDP shards that state so each device stores only its slice, dropping per-device memory
to roughly `18N / num_devices`. The cost is communication: it all-gathers each layer's parameters
before the forward and backward compute of that layer and reduce-scatters the gradients after, so
you trade memory for bandwidth. In practice it hides most of that communication behind computation,
so throughput stays high, but on a slow interconnect the exposed communication can dominate, which
is why FSDP scales best within fast-networked clusters.

**D3. What is the pipeline bubble and how do you shrink it?**

If you split layers across devices and feed one batch, device 0 works while all others idle, then
device 1 works while the rest idle: that idle time is the bubble, and it scales with the number of
stages. You shrink it by splitting the batch into micro-batches and feeding them in so all stages
work on different micro-batches simultaneously, like an assembly line. With enough micro-batches
the bubble becomes a small fraction of runtime. You never eliminate it, and there are schedules
(1F1B, interleaved) that reduce it and the peak activation memory further. Naming a schedule signals
depth.

---

## Part E — Inference and serving (go deep here)

**E1. Explain the two-phase nature of inference and why the hardware bottleneck differs.**

Prefill processes the whole prompt in one parallel pass: it is a big batched matmul, high
arithmetic intensity, compute-bound, and it sets time-to-first-token. Decode generates one token
per forward pass, and a single-token pass reads the entire model's weights from HBM to do a tiny
amount of compute, so its arithmetic intensity is terrible and it is memory-bandwidth-bound; it
sets per-token latency. The consequence: decode speed scales with memory bandwidth divided by bytes
of weights read per token, which is why quantization directly speeds up decode and why the relevant
spec for generation speed is memory bandwidth, not peak FLOPs. This split also motivates
disaggregated serving (Dynamo), running prefill and decode on differently-provisioned hardware
because they want opposite things.


**E2. Design an inference server for a chat product with thousands of concurrent users.**

Start from the bottleneck. Decode is memory-bound and single requests underutilize the GPU, so the
core is continuous (in-flight) batching to keep the batch full and amortize the weight read across
many sequences, with a paged KV cache so the variable-length caches pack into memory without
fragmentation: this is vLLM. If the product has a large shared system prompt across users, prefix
caching (SGLang RadixAttention) reuses that prefill and cuts time-to-first-token. Quantize weights
(int4/int8, or FP8 on new hardware) to shrink memory and speed decode, and quantize the KV cache if
context is long. Add speculative decoding (EAGLE) to spend decode's idle compute on multiple tokens
per pass. Tune the throughput-latency tradeoff to the product: interactive chat caps batch size for
per-user latency, bulk processing maximizes batch for throughput. If one GPU cannot hold the model,
tensor-parallel within a node. Name each piece against the bottleneck it attacks; that structure is
the answer.

**E3. Your p99 latency is fine but throughput is low and GPU util is 60%. What is wrong?**

60% util with low throughput and acceptable tail latency points to under-batching: the GPU is
starved for parallel work because you are not packing enough concurrent requests onto it, so decode
runs memory-bound at low occupancy. Likely causes: static batching that waits for the slowest
sequence instead of continuous batching, a batch-size cap set too conservatively for latency, or KV
cache fragmentation limiting how many requests fit. Fixes: continuous batching, paged KV cache,
raise the batch cap while watching p99, and if the cache is the limit, quantize it or shrink it with
GQA/MLA. The diagnostic move is recognizing that low util plus low throughput means the engine is
not keeping the GPU fed, which is a batching and KV-cache-memory problem, not a model problem.

**E4. Why is speculative decoding free, and when does it stop helping?**

Decode is memory-bound, so the compute units are idle while waiting on the weight read. A small
draft model proposes several tokens, and the target model verifies all of them in one parallel
forward pass, which is compute-bound and uses the otherwise-idle compute. Every token the target
agrees with is accepted, and the first disagreement is corrected, so the output is provably
identical to plain target decoding. It helps in proportion to the draft's acceptance rate and the
spare compute available. It stops helping when the batch is already large (then decode is no longer
so memory-bound because the weight read is amortized, so there is no idle compute to reclaim), when
the draft is poor (low acceptance means wasted verification), or when the verification overhead
exceeds the savings. So it is most valuable at low-to-moderate batch sizes, which is exactly
latency-sensitive interactive serving.

---

## Part F — Quantization (your zone; expect depth)

**F1. Why does weight-only int4 speed up decode, and why not activations too by default?**

Decode is bottlenecked on reading weights from HBM (E1). int4 weights are a quarter the bytes of
bf16, so the memory-bound weight read is ~4x faster, roughly ~4x decode speedup when fully
memory-bound. Activations are harder because they contain outliers: a few channels with magnitudes
far larger than the rest, so a single scale for the whole tensor either clips the outliers or
crushes the precision of everything else. Weights are better-behaved and quantize cleanly.
Activation quantization needs outlier handling (SmoothQuant migrates the difficulty into the
weights; LLM.int8 keeps outlier channels in higher precision) and only pays off when you are
compute-bound enough that running the matmul itself in low precision helps, which is more the
prefill and large-batch regime. So weight-only is the default for memory-bound decode; add
activation quantization when compute-bound and willing to handle outliers.

**F2. PTQ versus QAT, and how do you decide which layers to keep in higher precision?**

PTQ quantizes a trained model with a calibration set, is cheap (no training), and is fine down to
int8 and often int4 for weights with good methods (GPTQ, AWQ), but accuracy degrades as you push
lower. QAT simulates the quantization in the forward pass during (continued) training so the model
learns weights robust to it, recovering most of the accuracy at int4 and below, at the cost of a
training run. You choose QAT when PTQ's accuracy loss at your target bit-width is unacceptable and
you can afford the training. For which layers stay higher precision, you build a sensitivity map:
perturb or quantize each layer in isolation and measure the downstream accuracy hit, then keep the
sensitive layers (typically attention and certain input projections) at higher precision and push
the tolerant layers (typically MLP) lower. Mixed precision by sensitivity gets the best accuracy
per byte. This is exactly the approach that works in practice, and describing the sensitivity-map
methodology concretely is the senior signal.

**F3. You quantize to int4 and accuracy drops on one document type but not others. Debug it.**

The uneven drop says the quantization interacts with something distribution-specific, not a
uniform degradation. Hypotheses to check: that document type may exercise layers or channels that
your sensitivity analysis (done on a different distribution) marked as tolerant but that are
actually sensitive for this input, so re-run sensitivity on the failing distribution. The
calibration set for PTQ may not have represented this document type, so the chosen scales are
wrong for it: recalibrate with data that includes it. There may be activation outliers specific to
this input that your weight-only scheme does not touch but that interact with the quantized
weights. And confirm it is quantization at all by comparing to the bf16 model on the same
documents, isolating the quantization delta. The debugging discipline (localize, isolate the
variable, re-measure on the failing distribution) is what is being tested.

---

## Part G — Scaling, data, post-training

**G1. Chinchilla says 20 tokens per parameter. Why do deployed models ignore it?**

Chinchilla minimizes loss for a fixed training budget. It says nothing about inference. If you
serve a model heavily, you pay its inference cost on every forward pass forever, and a smaller
model is cheaper at every one. So it is rational to overtrain a smaller model far past its
Chinchilla-optimal token count: you spend extra training compute once to get a smaller model that
is cheaper to serve billions of times, minimizing lifetime cost rather than training cost. LLaMA 3
8B on ~15T tokens (nearly 2000 tokens per parameter) is the example. For on-device deployment the
inference argument dominates completely, so small-and-heavily-overtrained is correct. Interviewers
like this because it separates people who memorized "20:1" from those who understand what it
optimizes.

**G2. Walk me through the 2026 post-training recipe and what changed from RLHF.**

Three stages: SFT for format and cold-start behavior by imitation; preference optimization (DPO and
variants) to prefer better over worse responses; then RL with verifiable rewards for reasoning. The
change from a year ago is the third stage. Classic RLHF trained a reward model on human preferences
and optimized against it with PPO, which is complex and the reward model is hackable. The shift is
to verifiable rewards: where correctness is checkable (math answers, passing tests, validating
structured output), the reward is objective and unhackable, so you can push hard on it. GRPO made
this cheap by dropping PPO's value network and using a group of sampled responses as its own
baseline. DAPO refined it for long chain-of-thought with Clip-Higher, dynamic sampling, token-level
loss, and overlong reward shaping, and the trend (DAPO, Dr. GRPO, GSPO) is to drop the KL term
entirely because verifiers reduce the distributional-shift risk it guarded against and dropping it
frees the reference model from memory. "SFT memorizes, RL generalizes" is the one-line summary of
why the RL stage matters.

**G3. What is GRPO's advantage estimate, and what are its failure modes?**

For each prompt you sample a group of responses, score them all with the reward, and the advantage
of a response is its reward minus the group mean (often normalized by the group std). That group
mean replaces PPO's learned value-network baseline, which is what removes the value network and
halves the memory. Failure modes: entropy collapse (the policy becomes overconfident and stops
exploring, so training stalls; Clip-Higher in DAPO fights this), advantage collapse (when all
responses in a group get the same reward the advantage is zero and there is no gradient, so dynamic
sampling filters those batches), and KL drift or reward hacking if the reward is not truly
verifiable. Naming entropy collapse and the specific DAPO fix is the depth signal.

**G4. Where does the real quality difference between two same-size models come from now?**

Data and post-training, not architecture. Architecture has largely converged on the standard
pre-norm decoder, the optimizer is Adam-family, and scaling laws are known, so at fixed size and compute the
separation comes from pretraining data quality (extraction, filtering, deduplication, mixture,
synthetic augmentation) and the post-training recipe. This is why the best labs guard their data
pipelines more than their architecture. For an applied role, the actionable version is that you get
more from improving your domain data and your SFT/verifiable-RL loop than from architectural
tinkering, which for your structured-extraction work means high-quality demonstrations plus
expert-iteration on the checkable extraction reward.

---

## Part H — Open-ended and judgment

These have no single right answer; they test how you think.

**H1. You have a fixed budget to improve a deployed extraction model's accuracy. What do you try, in what order?**

Order by expected return per unit effort. First, error analysis on a clean held-out set sliced by
document type and field, because you cannot improve what you have not localized, and the failures
are usually concentrated. Then data: more and higher-quality SFT demonstrations for the failing
slices, since data quality is the dominant lever and demonstrations are cheap relative to
architecture work. Then, because extraction has a verifiable reward (does the field match, does the
output validate), expert iteration: sample multiple extractions, keep the ones that validate,
fine-tune on those, which often beats more demonstrations because it uses the model's own correct
outputs. Then GRPO on the same verifiable reward if expert iteration plateaus. Only after
exhausting data and post-training would I touch architecture or a bigger model, because those are
expensive and the deployment likely has a size constraint. Throughout, keep the held-out set clean.
The reasoning (localize, then cheapest-highest-return first, then verifiable-reward RL because the
task supports it) is the answer.

**H2. Should you fine-tune a model or use retrieval for a knowledge-heavy task?**

Retrieval for knowledge that is large, changing, or needs to be cited and updated without
retraining, because you can swap the corpus without touching the model and you get provenance.
Fine-tuning for behavior, format, style, and skills the model should internalize, and for
narrowing a general model to a domain's patterns. They compose: retrieve the facts, fine-tune the
behavior of using them. The failure mode to name is trying to fine-tune facts into a model that
change frequently, which is expensive and goes stale, or trying to retrieve your way to a behavior
change that the model simply does not do well regardless of context. The judgment is matching the
tool to whether the gap is knowledge or behavior.

**H3. How would you know if your evaluation is lying to you?**

Contamination first: if the benchmark could be in pretraining data, high scores may be
memorization, so check with freshly-created held-out data dated after the training cutoff and
decontaminate the training set. Then leakage through your own process: any eval you tune against
stops being a clean measurement, so keep a private held-out set that never influences training or
hyperparameters. Then metric-task mismatch: lower loss or higher benchmark average can hide
regressions on the hard slice you actually care about, so track several sliced metrics, not one.
Then judge bias if using LLM-as-judge: position, length, and self-preference biases, which you
control by randomizing order, controlling for length, and calibrating against human labels on a
sample. The meta-point is that evaluation is where you fool yourself, so you design against your
own incentives.

---

## How to practice these

Do not memorize answers. For each question, close the answer and reconstruct the reasoning chain
out loud, because interviews test whether you can derive it live and adapt when the interviewer
changes an assumption ("now the model is an MoE," "now the interconnect is 10x slower," "now the
context is 1M tokens"). The strongest signal you can give is to state your assumptions, do the
arithmetic, name the current tool, and articulate the tradeoff, then notice out loud where your
answer would change if an assumption flipped. That last habit, holding the tradeoff space in view
rather than committing to one answer, is what a senior interviewer at a frontier lab is actually
listening for.
