# Inference & Efficiency

This chapter is "how to make a trained model fast, small, and cheap to serve." It applies to both LLMs and VLMs. A large share of all engineering-oriented papers and most production work lives here. The unifying fact: **inference has two phases with opposite bottlenecks**, and almost every technique targets one of them.

---

## 1. The two phases of inference (internalize this first)

Autoregressive generation splits into:

- **Prefill (prompt processing):** process the entire input prompt in one parallel forward pass; compute and cache K/V for every prompt token. This is **compute-bound** (big matmuls, GPU-friendly). Determines **TTFT (time to first token)**.
- **Decode (generation):** generate one token at a time, each step attending to the whole KV cache. Each step is a tiny matmul but must *read the entire KV cache + weights from memory* → **memory-bandwidth-bound**, GPU compute mostly idle. Determines **TPOT (time per output token)** / inter-token latency.

This asymmetry explains everything: decode is slow not because of FLOPs but because of *memory movement*. So decode optimizations attack memory bandwidth and cache size; prefill optimizations attack compute and the `n²` attention. Modern systems even **disaggregate** prefill and decode onto different hardware.

Serving metrics to know: **TTFT**, **TPOT/ITL**, **throughput (tokens/sec, aggregate)**, **goodput (requests meeting SLO)**, **P50/P95/P99 latency**. There's a fundamental **latency–throughput tradeoff**: bigger batches = more throughput but worse per-request latency.

---

## 2. The KV cache — the central object

During decode, you don't recompute past tokens' Keys and Values — you cache them. The KV cache grows with every generated token:

```text
KV cache size ≈ 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_elem
```

It is usually the **memory bottleneck** at inference (often dwarfing the weights for long contexts / large batches), and reading it is the **bandwidth bottleneck** of decode. Hence:

- **Shrink it architecturally:** GQA/MQA/MLA (the attention variants from the LLM chapter). MLA's low-rank latent cache is the most aggressive.
- **Quantize it:** store K/V in FP8 / INT4 — see §2.1 below. Halves or quarters cache size; quality must be checked.
- **Manage it efficiently:** PagedAttention (§4).

### 2.1 KV-cache quantization

Quantizing the KV cache is distinct from quantizing weights — you're attacking the *decode bandwidth bottleneck*, not just the memory footprint. The key insight from **KIVI** is that Keys and Values have structurally different outlier patterns and need different quantization axes:

- **Keys: quantize per-channel.** Key vectors have *stable outlier channels* — the same channel positions blow up across all tokens. Per-channel quantization places each channel's scale factor separately, containing the outliers without polluting the rest. This makes 2-bit Keys feasible.
- **Values: quantize per-token.** Value vectors have *per-token outliers* — a specific token's value vector may have large magnitude, but the outlier positions shift across tokens. Per-token quantization handles this correctly.

This **asymmetric** strategy (different granularity for K vs V) is the reason 2-bit KV quantization is achievable with minimal quality loss; symmetric per-tensor quantization at 2-bit is not.

**Layer-discriminative mixed precision** goes further: not all layers tolerate the same KV quantization level. **MiniKV** and **KVTuner** measure per-layer sensitivity (typically via calibration data) and assign tighter quantization (INT4 or FP8) to the less sensitive middle layers while keeping early and late layers at higher precision. Early layers — where the residual stream has not yet built up rich compositional representations — are disproportionately sensitive; aggressively quantizing them degrades quality far more than quantizing the same layers mid-network.

**FP8 KV is the default production setting on H100+.** The H100's native FP8 support makes FP8 KV a near-free win over BF16: ~2× memory reduction, same decode throughput target, quality impact below measurement noise on standard benchmarks. When you read an H100 inference paper and they don't mention KV quantization, they almost certainly still use FP8 KV as a baseline.

Reading takeaway: KV-cache quantization is not one choice but three: (a) the bit-width, (b) the quantization axis (per-channel vs per-token vs per-tensor), and (c) whether it's uniform or layer-discriminative. Papers that elide these details are hiding the hard part.

- **Evict/compress it:** drop low-value tokens. **StreamingLLM / attention sinks** (keep the first few "sink" tokens + a recent window — turns out models dump attention onto the first tokens, so keeping them stabilizes infinite streaming). **DuoAttention** (only "retrieval heads" get the full cache; "streaming heads" get a small one) → big VRAM savings, up to ~2× decode speedup.
- **Reuse it:** **prefix caching** — if many requests share a prompt prefix (system prompt, few-shot examples, a long document), cache and reuse its KV across requests. Huge win for agents/RAG with fixed system prompts.

---

## 3. FlashAttention — making attention memory-efficient

Standard attention materializes the full `n×n` score matrix in slow GPU memory (HBM), so it's bottlenecked by memory traffic, and its memory is `O(n²)`.

**FlashAttention** computes the *same* attention but **never materializes the full matrix**. It tiles Q/K/V into blocks that fit in fast on-chip SRAM, computes attention block-by-block, and uses the **online softmax** trick (running max + running normalizer) to combine blocks correctly without storing all scores. Result: attention memory drops to **O(n)**, with 2–4× speedups and far less HBM traffic — and it's *exact*, not an approximation.

Versions: **FA-2** (better parallelization/work partitioning), **FA-3** (Hopper/H100: overlaps matmul and softmax via warp specialization, uses FP8, ~1.5–2× over FA-2, ~75% H100 utilization). Related kernels: FlashDecoding (parallelize over KV length during decode), FlashInfer (block-sparse, production kernels). You won't write these, but you'll see "we use FlashAttention-2/3" in nearly every systems paper, and you should know it's an *exact, IO-aware* kernel, not a model change.

---

## 4. PagedAttention & vLLM — serving memory like an OS

Naive serving allocates one big contiguous KV buffer per sequence sized to the max length → massive **fragmentation and waste** (you reserve for 8k tokens but generate 200).

**PagedAttention** (the core of **vLLM**) borrows OS virtual memory: split the KV cache into fixed-size **blocks**, allocate them on demand, and keep a per-sequence **block table** (indirection layer) pointing to non-contiguous blocks. Near-zero waste, and blocks can be **shared** across sequences (e.g. common prefixes, parallel samples from one prompt = copy-on-write). This is what enables **continuous batching** (§5) at high utilization. vLLM reports ~14–24× throughput over naive HuggingFace generation.

This pattern — treat the KV cache as paged, shareable memory — is now standard across serving stacks.

### 4.1 Prefix caching and RadixAttention

PagedAttention's block sharing becomes dramatically more valuable when combined with **automatic prefix reuse** across requests. The observation: in production agent and RAG workloads, a large fraction of requests share a common prefix — a system prompt, a few-shot preamble, a retrieved document, or a long multi-turn conversation history. Re-running the prefill for that shared prefix on every request is pure waste.

**SGLang's RadixAttention** generalizes prefix caching into a full **radix tree over KV blocks**. Every completed request's KV blocks are retained in the radix tree, keyed by their token content. When a new request arrives, the engine performs **automatic longest-prefix matching** against the tree — it finds the deepest node matching the new request's prefix, reuses those KV blocks directly (no compute, just memory reads), and only runs prefill for the novel suffix. **Copy-on-write** handles the case where two concurrent requests diverge from a shared prefix: blocks are shared read-only until a write is needed, then copied — the same mechanism as OS fork.

The numbers at scale: **60–85% cache hit rates** on agent and RAG workloads, translating to **5–12× cost reduction** in prefill compute and TTFT. Cursor's adoption of SGLang is the canonical industry reference — coding agents have extremely stable system prompts and file-context prefixes that repeat across thousands of concurrent requests, making the hit rate especially high.

**Who shares prefixes:**
- **System prompts** — every request in a deployment shares the same opening instructions.
- **Few-shot preambles** — the same examples prepended to every query.
- **Multi-turn agents** — earlier turns in a long conversation are a shared prefix relative to new turns.
- **n-sample decoding** — sampling N responses to the same prompt shares the prompt prefix across all N.
- **RAG with a common retrieved document** — many queries about the same document share its context prefix.

The implementation detail that makes this work at scale: the radix tree is managed by the same paged block allocator as PagedAttention, so blocks in the tree count against the same pool and can be evicted under memory pressure (LRU eviction from the tree), with the engine dynamically trading off cache hit rate against available memory for new requests.

### 4.2 Disaggregated prefill/decode and NVIDIA Dynamo

Prefill is **compute-bound** (big parallel matmuls over the prompt). Decode is **memory-bandwidth-bound** (one token per step, reading the entire KV cache + weights). Running both phases on the same GPU means the machine is simultaneously under-utilizing its compute (during decode) and under-utilizing its memory bandwidth (during prefill) — a fundamental resource mismatch that continuous batching only partially resolves.

**Disaggregated prefill/decode** addresses this directly: **prefill runs on dedicated compute-optimized instances; decode runs on dedicated bandwidth-optimized instances.** The KV cache computed during prefill is transferred to the decode pool over **RDMA or NVLink** (via **NIXL**, a low-latency KV transfer library). Each pool can be independently scaled and optimized — you can run prefill on fewer, compute-dense GPUs and decode on more, bandwidth-rich ones, matching the bottleneck to the hardware.

**NVIDIA Dynamo** (open-sourced, GA March 2026) is the reference implementation of disaggregated serving at the orchestration layer. It provides:
- **KV-aware routing:** routes incoming requests to prefill workers that already hold relevant cached KV blocks (integrating with prefix caching), then routes to decode workers with bandwidth headroom.
- **SLO-based planner:** dynamically adjusts the prefill/decode pool ratio to meet latency and throughput SLOs under varying load.

**SGLang + Mooncake on GB200 NVL72** demonstrated **2.7× decode throughput** versus a baseline non-disaggregated setup — the clearest published number on the hardware's bandwidth-compute asymmetry being exploited correctly.

**Mooncake** (Kimi's serving stack, **FAST 2025 Best Paper**, >100B tokens/day) is the canonical production case study. Mooncake separates KV cache management into a distributed object store, enabling KV migration between nodes with minimal overhead and achieving high cache hit rates across a large fleet — the paper is worth reading in full for anyone building at that scale.

Reading takeaway: when a paper describes "prefill–decode disaggregation," the key questions are (a) how KV is transferred between pools and at what cost, (b) how routing is aware of prefix-cached KV, and (c) what the scaling ratio between prefill and decode pools looks like at steady state.

---

## 5. Batching & scheduling

- **Static batching:** wait for a full batch, run it together. Wastes time when sequences finish at different lengths (everyone waits for the longest).
- **Continuous / in-flight batching:** the scheduler adds and removes sequences from the running batch *every step* as requests arrive and finish. Keeps the GPU saturated; the single biggest throughput lever for real traffic. Enabled by paged KV.
- **Chunked prefill / split-fuse:** break long prefills into chunks and interleave them with ongoing decodes so a giant prompt doesn't stall everyone (DeepSpeed Dynamic SplitFuse, vLLM chunked prefill).

---

## 6. Quantization — fewer bits per weight/activation

Reduce numerical precision to cut memory and bandwidth (and sometimes compute). The dominant compression technique. Notation: **W**eight bits **A**ctivation bits, e.g. **W4A16** = 4-bit weights, 16-bit activations.

- **Why it works:** weights are redundant and tolerant; you can often go to 4-bit weights with minor quality loss because decode is memory-bound, so smaller weights = faster *and* smaller.
- **PTQ (Post-Training Quantization):** quantize an already-trained model, no/low retraining. The common path:
  - **GPTQ:** layer-wise, uses second-order (Hessian) info to minimize error; 3–4 bit weights. Calibration-set based.
  - **AWQ (Activation-aware):** protects the most salient weight channels (those multiplying large activations) by scaling, then quantizes the rest. Often better quality at 4-bit, fast.
  - **SmoothQuant:** migrates activation outliers into weights so both can be INT8 (W8A8) — good for compute-bound regimes.
  - **GGUF (llama.cpp):** the CPU/consumer format with many k-quant levels (Q4_K_M, Q5_K_M, ...). What you run locally on a Mac/CPU.
  - **MLX / CoreML quant** (Apple silicon), **AutoRound**, **bitsandbytes** (NF4 — the 4-bit used in QLoRA).
- **QAT (Quantization-Aware Training):** *simulate* quantization during training/fine-tuning (fake-quant in the forward, full-precision gradients via straight-through estimator) so the model learns weights robust to low precision. More expensive but recovers more quality at very low bits, and lets you build a **layer sensitivity map** — some layers (attention projections, gating, certain in-projections) are far more sensitive than others (many MLP layers tolerate aggressive quantization), so you quantize mixed-precision accordingly. Essential when targeting on-device int4/int8.
- **KV-cache quantization:** quantize the *cache* (FP8/INT4), separate from weight quantization — attacks the decode bandwidth bottleneck directly.

The reading takeaway: classify any quant paper by (a) PTQ vs QAT, (b) what bits for weights/activations/KV, (c) what trick handles **outliers** (the universal enemy — a few large-magnitude values that wreck naive quantization).

### 6.1 Outlier handling: the rotation trick

Activation outliers — a handful of channels with magnitudes 10–100× larger than the rest — are the central obstacle to sub-4-bit weight+activation quantization. SmoothQuant (from the quantization section above) migrates outlier energy from activations into weights. A more powerful and now-standard approach: **rotate the model's weight matrices so outliers don't exist in the first place.**

**QuaRot** applies random **Hadamard rotations** to the weight matrices at each layer boundary. A Hadamard matrix is orthogonal (norm-preserving) and extremely fast to compute (O(n log n) via the Fast Walsh-Hadamard Transform). Multiplying activations by a random Hadamard matrix redistributes outlier energy *uniformly across all channels* — the few huge-magnitude channels become many medium-magnitude channels. The model's output is identical (the rotation is applied consistently to both weights and activations so they cancel), but the resulting activation tensor is quantization-friendly at 4-bit or lower. No retraining required for the rotation itself; only a short calibration pass to set quantization scales.

**SpinQuant** takes the same insight further: instead of a fixed random Hadamard, it learns the rotation matrix end-to-end with gradient descent, **constrained to lie on the Stiefel manifold** (the space of orthogonal matrices) using Cayley SGD. The manifold constraint ensures the rotation stays norm-preserving throughout optimization. Learned rotations find the *optimal* orientation for quantization-friendliness rather than relying on the statistical argument that random Hadamards work on average.

**DartQuant (2025–2026)** extends the rotation framework to handle the residual outliers that survive Hadamard/SpinQuant in very deep models, applying layer-adaptive rotation strengths rather than a single global transform.

The rotation trick is now the **standard recipe for sub-4-bit weight+activation quantization (W4A4 and below)**. The practical pipeline: apply rotations (QuaRot-style for speed, SpinQuant for quality) → calibrate quantization scales → optionally QAT-fine-tune for a few steps to recover residual loss. This is what the checklist item "what trick handles outliers" is pointing at — if a sub-4-bit quant paper doesn't mention rotations or an equivalent, it's missing the key ingredient.

---

## 7. Distillation & pruning — other ways to shrink

- **Knowledge Distillation:** train a small **student** to mimic a large **teacher** — match the teacher's output distribution (soft labels carry more info than hard labels), or its hidden states, or just SFT the student on teacher-generated data (**sequence-level distillation**, how R1's reasoning went into small models, as covered in the post-training material). The cheapest way to get a capable small model.
- **Pruning:** remove weights/structures. **Unstructured** (zero individual weights — high sparsity but needs special kernels to actually speed up) vs **structured** (remove whole heads/neurons/layers — actually faster on normal hardware). **Depth pruning** (drop layers) and **width pruning** are common; usually followed by a short heal/distill step.
- **MoE** (the Mixture-of-Experts approach from the LLM chapter) is itself a compute-reduction technique (activate few params), often combined with the above.

---

## 8. Speculative decoding — break the one-token-at-a-time barrier

Decode is sequential and bandwidth-bound, leaving GPU compute idle. **Speculative decoding** exploits the idle compute:

1. A cheap **draft** model (or the model's own extra heads) proposes the next `k` tokens quickly.
2. The big **target** model verifies all `k` in a **single parallel forward pass** (verification is compute, which is free here).
3. Accept the longest correct prefix; on a mismatch, fall back to the target's token. **The output distribution is provably identical to the target's** — it's pure speedup, no quality change.

Typically 2–3× decode speedup. Variants: **Medusa** (extra prediction heads on the target, no separate draft), **EAGLE / EAGLE-3** (predict at the feature level, current SOTA), **lookahead decoding**, **n-gram/prompt-based drafts**. Note: it helps most at *small batch sizes* (where you're bandwidth-bound and compute is idle); at huge batches the GPU is already saturated and the gain shrinks.

---

## 9. On-device / edge inference

Running models on phones, laptops, embedded — memory, bandwidth, thermal, and battery constrained. The stack you'll encounter:

- **Formats/runtimes:** **GGUF + llama.cpp** (CPU/Metal/CUDA, ubiquitous local), **MLX** (Apple silicon, with **MLX Swift** for iOS), **CoreML** (Apple Neural Engine), **ONNX Runtime**, **MNN** (Alibaba; Android, including a VLM-capable chat app fork), **ExecuTorch** (PyTorch edge), **TensorRT-LLM** (NVIDIA server/edge), **LiteRT/LiteRT-LM** (formerly TFLite; Gemma-on-mobile).
- **What dominates:** aggressive quantization (int4/int8, often QAT-derived), small models (sub-4B, MoE-on-device emerging), and architecture choices that shrink the KV cache (GQA, sliding-window) since memory is the wall.
- **Known edge gotchas:** immature GPU kernels for newer ops (e.g. gated-delta/linear-attention kernels) forcing CPU fallback; tokenizer mismatches during conversion producing silent garbage; positional-encoding bugs in multimodal RoPE during export; per-op support gaps between runtimes. Conversion is where most of the pain is, not the model.
- **Server-class edge:** devices like DGX Spark vs a beefy laptop (M-series Max) trade off differently — for **agentic/coding loops dominated by TTFT** (many short turns, prefill-heavy), faster prefill compute can matter more than raw decode throughput.

---

## 10. Reading-an-efficiency-paper checklist

- **Which phase / bottleneck?** Prefill/compute (TTFT) or decode/bandwidth (TPOT)? KV-cache memory? Weight memory?
- **Lossless or lossy?** FlashAttention, PagedAttention, speculative decoding are *exact*. Quantization, pruning, KV-eviction, linear-attention are *approximations* — demand the quality numbers and the failure mode.
- **What's the actual win, on what hardware, at what batch size?** Speedups are regime-dependent (batch size, context length, GPU generation). A 2× that only holds at batch=1 on an H100 is not a 2× for your serving load.
- **Outliers / sensitivity:** for quant, how are activation outliers and layer sensitivity handled? That's where the quality lives.
