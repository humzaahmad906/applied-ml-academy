# 06 — Advanced Topics: Everything Else Worth Knowing — Part 1 of 5: Distributed Systems, Training, and Inference Optimization


The prior files cover foundations through F50 specialization and portfolio projects. This file covers the remaining body of knowledge that distinguishes a strong senior MLOps engineer from a competent mid-level one — distributed systems theory, training/inference optimization at the kernel level, model compression, security, LLM internals, RL and agentic systems, observability deep dives, and the operational concerns that show up only after you've shipped real systems.

Treat this as a **post-graduate curriculum**. You don't sit and march through it. You consult it as you build projects, hit walls, and prepare for interviews.

**How to use this chapter:** 18 phases. Phases 1–5 are foundational — work through sequentially. Phases 6+ are specialized — read in any order based on the role you target.

---

## Phase 1 — Distributed Systems Foundations for ML

Most MLOps problems are distributed systems problems wearing ML clothes. Without this foundation, you'll re-learn the same lessons by hitting the same walls.

### What to Learn

#### CAP, PACELC, and what they mean for ML systems

**CAP theorem:** In a partition, choose Consistency or Availability.

**PACELC:** Even without partitions, choose Latency or Consistency.

For ML serving:
- A feature store online cache that lets reads return slightly stale values (eventually consistent) — AP/EL. Cheaper, faster, often correct enough.
- A model promotion that must be globally consistent before traffic switches — CP. Slower, atomic.
- Predictions on an event stream — often AP at-least-once with idempotent consumers, not exactly-once.

#### Consistency Models

- **Linearizability** — every op appears instantaneous. Strongest. What you want for model promotion.
- **Sequential consistency** — operations per client are ordered; global order can be reordered.
- **Causal consistency** — causally related ops are ordered; unrelated ones not.
- **Read-your-writes** — common online feature store guarantee.
- **Eventual** — replicas eventually agree. Most online feature reads.

#### Replication Strategies

- **Single-leader** — Postgres replication, most relational systems. Writes to leader, reads from anywhere with lag.
- **Multi-leader** — geo-distributed writes. Conflict resolution is hard.
- **Leaderless** — Cassandra, DynamoDB. Quorum writes; R + W > N for strong reads.

For ML: training data warehouses are usually single-leader (the lake is the source of truth). Online feature stores are often leaderless or multi-leader for low-latency global reads.

#### Partitioning (Sharding)

- **Range** — partition by ordered key. Good for ranges; bad for hotspots.
- **Hash** — partition by hash of key. Even; bad for ranges.
- **Composite** — hash + range (DynamoDB PK + SK).
- **Geo** — by user region; required for data residency.

Feature stores typically hash on entity ID. Training data partitions typically range on date.

#### Consensus (Conceptually)

Multiple nodes agreeing on a value despite failures. Raft is the modern, understandable consensus algorithm. Where you see it: Kafka's controller election (KRaft), etcd, every distributed control plane.

### The Reading

1. **Designing Data-Intensive Applications** by Martin Kleppmann, chapters 5–9. Single most important book.
2. The **Dynamo paper** (Amazon, 2007).
3. The **Spanner paper** (Google, 2012).
4. The **Borg paper** (Google) — foundational for Kubernetes.

### Exercises

1. Sketch the architecture of three systems you've used (Postgres, Kafka, Redis, your model registry — pick any). For each, identify: leader topology, replication, consistency model, partitioning.
2. Write 500 words on why exactly-once across heterogeneous systems is hard.
3. Pick a paper from [Papers We Love](https://github.com/papers-we-love/papers-we-love) and write a one-page summary.

---

## Phase 2 — Training Optimization at Depth

Beyond the foundations of mixed precision and DDP, here's what frontier-lab training looks like.

### Memory Math, Precisely

For a Transformer with **P** parameters, batch size **B**, sequence length **L**, hidden size **H**, number of layers **N**:

| Component | Memory (mixed precision Adam) |
|---|---|
| Parameters (FP16 + FP32 master) | 6P bytes |
| Gradients (FP16) | 2P bytes |
| Optimizer state (Adam: m, v in FP32) | 8P bytes |
| Activations (depends on checkpointing) | proportional to B × L × H × N |
| Workspace + buffers | GBs (NCCL, cuDNN, allocator fragmentation) |

For Llama-2-7B in mixed precision Adam:
- Parameters: 6 × 7B = 42 GB
- Gradients: 2 × 7B = 14 GB
- Optimizer: 8 × 7B = 56 GB
- Total **before** activations: 112 GB
- That's why you need 2+ A100-80GBs for full fine-tuning even of a 7B model.

LoRA fixes this: you train ~0.5% of parameters, so optimizer state is tiny, gradients are tiny. Same 7B model fine-tunes on a single 24GB GPU.

### ZeRO Stages (DeepSpeed) / FSDP Sharding

- **ZeRO-1:** Shard optimizer state. ~4x memory reduction for optimizer.
- **ZeRO-2:** Shard optimizer + gradients. ~8x reduction.
- **ZeRO-3:** Shard parameters too. ~Nx reduction where N is the data-parallel degree. Equivalent to PyTorch FSDP `FULL_SHARD`.

Trade-off: more sharding = more communication. ZeRO-3 vs ZeRO-2 doubles inter-GPU traffic. In NVLink-connected nodes that's fine. Across slower interconnects it can dominate.

### Activation Recomputation (Gradient Checkpointing)

Instead of storing activations during forward for backward, recompute them. ~30% slower step time, sometimes 5x memory savings. Default for any sizable model.

### Sequence Parallelism

For very long sequences (long-context LLMs), shard the sequence dimension across GPUs. Megatron and Ring Attention popularized this. Required for million-token contexts.

### Tensor Parallelism Within a Node

Split a single layer (e.g., a linear's weight matrix) across GPUs. Each GPU computes part of the layer, then all-reduces to share. Requires fast interconnect (NVLink) because of frequent communication.

The 2026 frontier setup:

- Tensor parallel within a node (e.g., 8-way TP on 8 GPUs over NVLink)
- Pipeline parallel across nodes (e.g., 8-stage PP across 8 nodes)
- Data parallel / FSDP across pipeline replicas

That's **3D parallelism**. For trillion-parameter models, 4D (add sequence parallelism).

### FlashAttention and Its Successors

Standard attention reads/writes O(L^2) memory in HBM. FlashAttention reorders the computation to keep things in SRAM, achieving 2–4x speedup and lower memory. FlashAttention-2 and FlashAttention-3 (Hopper-optimized) extend this.

You won't implement it. You'll enable it: `attn_implementation="flash_attention_2"` in HuggingFace Transformers, or use PyTorch's `torch.nn.functional.scaled_dot_product_attention` which dispatches to FA when available.

### Compilation

`torch.compile` traces your model into a graph, optimizes it, generates fused kernels (Triton). 1.5–3x speedup on many models. Two pain points: long first-call compile time, occasional graph breaks that fall back to eager.

For LLM inference, the production-grade compiled stacks:

- **TensorRT-LLM** (NVIDIA) — best on NVIDIA hardware
- **vLLM**'s built-in optimized kernels (PagedAttention, FlashAttention)
- **MAX engine** (Modular)
- **OpenAI Triton** kernels (different from NVIDIA Triton — confusingly)

### Hyperparameter Optimization (Done Right)

- **Successive halving / Hyperband / ASHA** — start many trials, kill underperformers early
- **Bayesian optimization (Optuna, Ax)** — model the objective; pick the next trial intelligently
- **Population-based training (PBT)** — DeepMind's approach for online HPO during long runs
- **Multi-fidelity:** train on small data first, only commit to full training for promising configs

For LLM training, HPO matters less than data and architecture. For classical ML it dominates.

### Exercises

1. Take a 1B-param model. Compute peak memory by hand for mixed precision Adam. Verify with `nvidia-smi` during a training step.
2. Enable gradient checkpointing. Measure memory + time. Quantify the trade-off.
3. Switch FlashAttention on/off. Measure throughput + memory.
4. `torch.compile` a small model. Time first call and steady-state.

---

## Phase 3 — Inference Optimization at Depth

The other half of the optimization story. For LLM inference especially, this is where the money is.

### Quantization

Reducing numerical precision of weights and/or activations to save memory and increase compute throughput.

| Format | Precision | Common use |
|---|---|---|
| FP32 | 32-bit | Training original |
| FP16 | 16-bit float | Training, inference (some accuracy loss) |
| BF16 | 16-bit float, wider exponent | Training (preferred), inference |
| FP8 | 8-bit float (Hopper+) | Inference (cutting edge) |
| INT8 | 8-bit int | Common inference quantization |
| INT4 | 4-bit int | LLM inference (GPTQ, AWQ, GGUF) |

Two flavors:

1. **Post-training quantization (PTQ)** — quantize a trained model. Easy, slight quality loss.
2. **Quantization-aware training (QAT)** — train with simulated quantization. More work, less quality loss.

For LLMs, **AWQ** and **GPTQ** are the dominant INT4 schemes. Quality loss is typically <2% on benchmarks for many models. Marlin kernels make INT4 ops competitive with FP16 in throughput. Memory savings: 4x.

### Pruning

Remove weights / heads / layers. Unstructured pruning rarely helps inference speed (sparsity is hard to exploit). Structured pruning (remove whole heads, neurons, layers) gives real speedups but requires fine-tuning to recover quality.

### Distillation

Train a small "student" model on the outputs of a large "teacher" model. Often gets within 95% of teacher quality at 10–100x lower inference cost. Big lever for production.

The 2026 default LLM cost play: take GPT-4o or Claude as teacher, distill into a 7B or 13B open-weights student via synthetic data generation + SFT. Massive cost reduction with most of the quality.

### KV Cache Management

In LLM inference, the KV cache stores attention key/value tensors for previous tokens. Its size is $2 \times N \times L \times H \times \text{bytes\_per\_element}$ (N layers, L sequence length, H hidden size, bytes per element by precision).

For Llama-2-7B with 4096 context, FP16: ~2GB per request. At 100 concurrent requests: 200GB. The KV cache, not parameters, is the memory bottleneck for LLM inference.

**PagedAttention** (vLLM's signature feature) pages the KV cache like virtual memory. Eliminates fragmentation. Allows much higher concurrency.

**Prefix caching** reuses KV state for shared prefixes (system prompt, RAG context). Free if your traffic shares prefixes.

**Quantized KV cache** (INT8 / FP8) halves or quarters cache memory.

### Continuous Batching

Standard batching waits for N requests, runs together, returns together. With LLMs (variable output length), the longest request bottlenecks the batch.

**Continuous batching** (also "in-flight batching"): treat each token generation as a step; new requests join the batch as old ones finish, token by token. Throughput typically 5–10x of static batching. vLLM, TGI, SGLang, TensorRT-LLM all do this.

### Speculative Decoding

A small "draft" model proposes the next N tokens; the big model verifies them in parallel. If most are accepted, you get N tokens for the cost of one big-model forward pass.

Variants:

- **Standard speculative decoding** — Leviathan et al.
- **Medusa** — multiple parallel decoding heads
- **EAGLE** — improves on Medusa
- **Speculative + tree decoding** — multiple branches; tree-shaped acceptance
- **Lookahead decoding** — no draft model; uses N-gram caches

Typical speedup: 2–4x for chat-style workloads.

### Architectural Tricks

- **Multi-query attention (MQA)** and **grouped-query attention (GQA)** — share keys/values across heads. Massively reduces KV cache. Llama-2-70B onwards uses GQA.
- **Mixture of Experts (MoE)** — only a subset of parameters active per token. Larger total capacity, similar inference cost. Mixtral, DeepSeek-V3, GPT-4-class architectures.
- **Sliding window attention** — bound attention to a local window. Mistral-style.
- **State-space models (Mamba, RWKV)** — alternative to attention; linear in sequence length. Interesting; not yet dominant.

### Inference Stack Choices in 2026

For LLM serving on NVIDIA GPUs:

- **vLLM** is the default open-source choice
- **TensorRT-LLM + Triton** is the highest-throughput / lowest-latency choice if you can afford the operational complexity
- **SGLang** for structured-output-heavy or program-over-LLM workloads
- **TGI** for HuggingFace-aligned shops
- **MAX** for the cutting edge of throughput on CPU and accelerators

For non-NVIDIA: AWS Inferentia2/Trn1, Apple Silicon (MLX), Google TPUs, AMD MI300X (now competitive on inference, ROCm has matured). Each has its own stack.

### Exercises

1. Quantize a 7B LLM to INT4 with AWQ or GPTQ. Measure latency, throughput, and quality on a small eval set.
2. Serve it with vLLM. Drive load to its saturation point with a tokenizer-aware load tool.
3. Implement prefix caching with a fixed system prompt. Measure throughput improvement.
4. Distill GPT-4o into a 7B student on a small task. Quantify the cost reduction.

---

## Phase 4 — Model Compression and Edge Inference

### When Edge Matters

- Mobile / on-device — privacy, latency, cost
- IoT / embedded — battery, bandwidth
- Browser / WASM — zero-server cost
- Air-gapped / regulated environments

### The Toolkit

- **ONNX** — universal model interchange format
- **TensorRT** — NVIDIA edge inference
- **CoreML** (Apple), **TensorFlow Lite** (everywhere), **Executorch** (PyTorch's mobile path)
- **GGUF / llama.cpp** — CPU and consumer-GPU LLM inference; the de facto standard for local LLM
- **MLX** (Apple Silicon), **ONNX Runtime Web** (browser via WASM/WebGPU)

### What You Care About at the Edge

- Model size on disk (download cost, app size budget)
- Memory at runtime
- Latency on the target hardware
- Battery / thermal — sustained inference must not throttle
- Cold start time

Quantization (INT8 / INT4) and structured pruning are the headline techniques. For LLMs, GGUF-quantized models running in llama.cpp give a phone or laptop competitive performance with a small server GPU.

### Exercises

1. Convert a model to ONNX. Run it via ONNX Runtime with the right execution provider for your machine.
2. Run a 7B LLM locally on a laptop via llama.cpp. Compare tokens/sec across Q4_K_M, Q5_K_M, Q8_0 quantizations.
3. (If Apple Silicon) Run the same model via MLX. Compare.

---

## You can now

- Map a distributed-systems consistency choice (CAP/PACELC, replication, partitioning, consensus) onto the right ML component — feature cache, model promotion, or event stream.
- Compute training memory by hand for a given model and optimizer, and choose the right ZeRO stage / FSDP sharding and 3D-parallelism layout for the interconnect you have.
- Select the correct inference optimization for a bottleneck — quantization (AWQ/GPTQ/FP8), PagedAttention KV-cache paging, continuous batching, speculative decoding, or an architectural trick like GQA/MoE.
- Choose the right edge-inference toolkit (ONNX, TensorRT, CoreML, GGUF/llama.cpp, MLX) and the compression techniques (quantization, structured pruning) that make a model viable on-device.
