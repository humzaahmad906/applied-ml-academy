# 14 — The 2026 Stack and Recent Techniques

The core chapters teach the durable concepts. This one is the perishable layer: the actual tools,
libraries, and paper techniques a candidate is expected to know by name in a 2026 interview at a
frontier lab. Concepts get you understanding; naming the right tool and the right recent result is
what signals you have shipped, not just read. Everything here is current as of mid-2026; treat the
specific version claims as a snapshot, not a permanent ranking.

## Training stack

PyTorch is still the substrate; nobody is asking you to write raw CUDA in an interview. What
changed is the layer above it.

`torch.compile` is the default now, not an optimization you reach for. It traces your model,
fuses the memory-bound elementwise chains automatically, and generates Triton
kernels. If someone asks how you would speed up a training loop, "turn on `torch.compile` and
profile what it did not fuse" is the correct first answer before you talk about hand-written
kernels.

TorchTitan is PyTorch's native reference for production pretraining: it wires together FSDP2,
tensor parallelism, pipeline parallelism, and FP8 in one place so you compose the parallelism axes
with config rather than plumbing. FSDP2 is the current sharded-data-parallel
implementation (the successor to the original FSDP), and it is what you say when asked how you fit
a model too big for one GPU without reaching for Megatron. Megatron-LM and DeepSpeed are still the
heavyweight options for the largest runs and are worth being able to name; the distinction to
articulate is that TorchTitan is PyTorch-native and composable, Megatron is battle-tested at
extreme scale, DeepSpeed pioneered ZeRO.

FP8 training is the live frontier. bf16 is still the safe default, but FP8 for the GEMMs (the
matmuls) gives roughly 1.5x training throughput at matched quality, and 2025 work pushed toward
fully-FP8 transformer blocks (all GEMMs in forward and backward in FP8) by redesigning the
architecture to suppress the outlier activations that break naive FP8. The interview-relevant
point: FP8's problem is not the matmul, it is the outlier activations in sensitive layers
(attention projections), and the solutions are either keeping those layers higher precision or
architecturally reducing the outliers. This is the same outlier story as activation quantization
for inference, which is a nice connection to draw unprompted.

Optimizers moved too. AdamW is still standard, but Muon (a matrix-aware optimizer that
orthogonalizes the update) has real traction for pretraining efficiency and is the current "do you
follow the literature" name to drop. muP (maximal update parameterization) is the technique for
making learning rates and other hyperparameters transfer across model scale, so you tune on a
small model and the settings hold at the large one; this connects directly to the scaling-law
transfer idea covered earlier and saves enormous sweep compute.

## Attention and architecture, current defaults

GQA is the de facto standard for KV cache reduction; you should know the
typical ratio (8 KV heads for 32 query heads) and name that LLaMA, Mistral, and Qwen all use it.

Multi-head latent attention (MLA), from DeepSeek-V2 and carried into V3, is the technique to know
beyond GQA. Instead of reducing the number of KV heads, MLA compresses the full keys and values
into a low-rank latent vector that is what actually gets cached, then projects back up when needed.
It shrinks the KV cache further than GQA while keeping more of MHA's expressiveness, and DeepSeek
shipped FlashMLA, a dedicated kernel for it. If asked "how do modern models attack the KV cache
bottleneck," the full answer is MQA to GQA to MLA, in increasing order of sophistication and
recency.

FlashAttention has kept moving. FA2 is the workhorse. FA3 added warp specialization and FP8
support for a further 1.5-2x on Hopper-class hardware. FA4 targets Blackwell (B200) and works
around the fact that softmax special-function units did not scale with tensor-core throughput.
The takeaway to carry: FlashAttention is now the default attention backend inside vLLM, SGLang,
Transformers, and TensorRT-LLM, so you rarely call it directly, but you should be able to explain
the online-softmax tiling because that is the "explain FlashAttention" question and
it comes up constantly.

## Kernels

Triton is the language you name for custom kernels, and the trend is toward Triton backends that
run on multiple vendors (AMD as well as NVIDIA) rather than proprietary CUDA, because vendor
lock-in is a real production concern. Liger-kernel is a drop-in set of fused Triton kernels for
common Transformer ops (fused RMSNorm, SwiGLU, cross-entropy) that cut memory and speed up training
with no model changes, and it is the pragmatic "I did not write my own, I used the community fused
kernels" answer.

## Serving stack (your zone, and where the hiring signal is strongest)

This is the part to know cold, because inference is where cost lives and where a lot of applied ML
hiring is concentrated.

vLLM is the default and the industry standard for dynamic workloads. Its innovation is
PagedAttention (KV cache managed like virtual memory in fixed pages), and it has the
widest model support and no compilation step, so fastest time-to-first-inference. When in doubt,
vLLM.

SGLang is the one to know for structured generation and shared-prefix workloads. Its innovation is
RadixAttention: it caches shared prompt prefixes in a radix tree, so multi-turn chat and agentic
workloads that reuse the same system prompt across requests get large time-to-first-token savings.
If your workload has heavy prefix sharing (agents, RAG with a fixed instruction block), SGLang is
the right pick and saying so signals you understand workload-shaped decisions.

TensorRT-LLM gives the best raw throughput on NVIDIA hardware but requires a compilation step
(tens of minutes) and locks you to NVIDIA. The tradeoff to articulate: you pay setup time and
vendor lock-in for peak performance, worth it for a stable model at high concurrency, not worth it
for rapid iteration or model flexibility.

TGI is now in maintenance mode; Hugging Face themselves point people to vLLM or SGLang. Knowing
this signals you are current, because a candidate recommending TGI in 2026 dates themselves.

For orchestration above the engine there is NVIDIA Dynamo for disaggregated serving (splitting
prefill and decode across different hardware, which follows directly from their opposite compute
profiles), and Ray Serve / Triton Inference Server for multi-model fleets. For the
edge, which is your world, the names are llama.cpp (GGUF quantized models on CPU/Metal), MLX
(Apple silicon, which you already use), MNN, ONNX Runtime, and CoreML. The through-line: the same
concepts (paged KV cache, continuous batching, quantization) show up everywhere, but the edge
runtimes trade throughput features for footprint and portability.

The feature floor to state confidently: any serious 2026 engine has continuous (in-flight)
batching, paged KV cache, and FP8 quantization built in. The surface-level feature gap has largely
closed, so the decision is workload shape (prefix sharing, concurrency, latency target), hardware,
and operational tolerance, not a feature checklist.

## Quantization (your specialty, stated in interview terms)

Post-training weight-only: GPTQ (second-order, layer-wise) and AWQ (activation-aware, protects the
salient weight channels) are the two names for int4 weight quantization. For weight-and-activation,
SmoothQuant (migrates activation difficulty into weights) and the LLM.int8 outlier-handling idea
are the ones to cite, and the core problem is always activation outliers. FP8 is increasingly the
default low-precision format on new hardware for both training and inference because it keeps a
usable exponent range. KV cache quantization (int8/int4 KV) is the lever for long context and large
batch. TorchAO (which you already use) is the PyTorch-native library for all of this including QAT.
The senior framing: PTQ is cheap and loses accuracy at low bits; QAT recovers most of it at int4
and below at the cost of a training run; mixed-precision by layer sensitivity (keep attention/input
projections higher, push MLP lower) gets the best accuracy per byte. That last sentence is
essentially your sensitivity-map work, and it is exactly the kind of concrete, earned detail that
lands in an interview.

## Speculative decoding, current forms

The concept was covered under inference. The names to know: a separate small draft model is the classic form;
Medusa attaches multiple prediction heads to the target model so it drafts itself; EAGLE drafts in
the model's feature space and is the current strong one, reaching up to roughly 3.6x generation
speedup with no quality loss. Interview framing: speculative decoding spends decode's idle compute
(decode is memory-bound) to verify multiple drafted tokens in one pass, and it is exact, not
approximate.

## Post-training / RL stack (the fastest-moving area, know it well)

The 2026 recipe is three stages: SFT for format and cold-start, then preference optimization
(DPO and its variants SimPO, ORPO, KTO), then RL with verifiable rewards for reasoning. This is
the biggest shift from a year ago: RLHF with human preference labels is no longer the frontier;
verifiable-reward RL is.

GRPO (from DeepSeekMath, scaled in DeepSeek-R1) is the standard RL algorithm: drop PPO's value
network, sample a group of responses per prompt, use the group mean reward as the baseline. Know
its failure modes because they are common interview follow-ups: entropy collapse (the model stops
exploring), advantage collapse, and KL drift.

DAPO (ByteDance/Tsinghua) is the important refinement to name. It fixes GRPO's instabilities on
long chain-of-thought with four specific techniques: Clip-Higher (raise the upper clipping bound to
preserve exploration and prevent entropy collapse), Dynamic Sampling (drop batches with no gradient
signal), token-level policy-gradient loss (so long sequences do not get vanishing gradients from
sequence-level averaging), and overlong reward shaping. Being able to list those four is a strong
signal. GSPO and Dr. GRPO are further variants; the notable common thread is that DAPO, Dr. GRPO,
and GSPO all remove the KL term, arguing it holds the model too close to the base and that
rule-based verifiers make the distributional-shift risk that KL guards against less relevant.
Removing KL also lets you offload the reference model and save memory.

The libraries: verl (ByteDance, implements HybridFlow, scales to hundreds of GPUs and 600B+
models, supports GRPO/DAPO/GSPO/Dr.GRPO and VLM RL) is the production-scale name. TRL (Hugging
Face) is the accessible default for SFT/DPO/GRPO on Transformers. OpenRLHF, NeMo-RL (NVIDIA),
slime (Megatron+SGLang), and torchforge (Meta, PyTorch-native) round out the fragmented landscape.
Naming verl and TRL and knowing verl is the scale option covers most interviews.

The single most-cited recent result to have an opinion on: "SFT memorizes, RL generalizes." The
finding that RL post-training generalizes to held-out distributions where SFT overfits is a
frequent discussion prompt, and the nuanced take (RL generalizes but needs an SFT cold-start, and
only works where you have a verifier) shows depth.

## Data stack

FineWeb and FineWeb-Edu are the reference open pretraining datasets and the names to cite for
"where would you get quality pretraining data": they are the result of exactly the data
pipeline covered earlier (Common Crawl extracted, filtered, deduplicated, with an education-quality classifier for
FineWeb-Edu). datatrove is the library for running that pipeline at scale. For your world, the
synthetic-data connection is that the same filtering-and-classifier discipline applies to synthetic
generation.

## How to use this in an interview

Do not recite this list. The move is: answer the conceptual question from the core modules, then
ground it with one current name and one tradeoff. "I would use FSDP2 in TorchTitan because the
model does not fit on one GPU, and the cost is the all-gather of parameters into every layer, which
overlaps with compute so it is usually fine." That pattern (concept, current tool, tradeoff) is
what separates a candidate who has read from one who has shipped. The interview banks that follow
drill exactly these.

## Key takeaways

The durable concepts live in the earlier chapters; this is the current tooling that proves you apply
them. Training: `torch.compile` by default, FSDP2/TorchTitan for sharding, FP8 for throughput,
Muon and muP as the follow-the-literature names. Attention: GQA standard, MLA the sophisticated KV
compression, FlashAttention 2/3/4 as the default backend. Serving: vLLM default, SGLang for
prefix-sharing, TensorRT-LLM for compiled peak throughput, TGI dead, Dynamo for disaggregation,
and llama.cpp/MLX/ONNX/CoreML for edge. Quantization: GPTQ/AWQ/SmoothQuant/FP8/TorchAO with layer-
sensitivity mixed precision. Post-training: SFT then DPO then GRPO/DAPO verifiable-reward RL, run
in verl or TRL, with the KL-removal trend and the four DAPO tricks as the depth signals.
