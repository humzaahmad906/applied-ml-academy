# 06 — Advanced Topics: Everything Else Worth Knowing

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

## Phase 5 — RL, RLHF, DPO, and Modern Alignment

For LLM training and recommendation/decision systems, RL is increasingly part of the MLOps stack.

### Classical RL — The Conceptual Backbone

- **Agent, environment, action, state, reward** — the basic loop
- **Policy** — the function that maps state to action distribution
- **Value functions** — V (state value), Q (state-action value)
- **On-policy vs off-policy** — does the training data come from the current policy?
- **PPO** — proximal policy optimization; the workhorse algorithm
- **DDPG, SAC, TD3** — actor-critic methods for continuous actions

You don't need to implement these from scratch. You need to recognize them and know when each is used.

### RLHF Specifically

For aligning LLMs:

1. **SFT (Supervised Fine-Tuning)** — train on instruction → response pairs
2. **Reward modeling** — train a model to predict human preferences
3. **PPO** — fine-tune the LLM to maximize predicted reward, with a KL constraint to the reference model

PPO is finicky. Reward hacking is real. Most modern alignment has moved past it.

### DPO (Direct Preference Optimization)

Skip the reward model. Given pairs (prompt, chosen, rejected), train the model to prefer chosen directly:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left( \beta \log \frac{\pi(\text{chosen} \mid \text{prompt})}{\pi_{\text{ref}}(\text{chosen} \mid \text{prompt})} - \beta \log \frac{\pi(\text{rejected} \mid \text{prompt})}{\pi_{\text{ref}}(\text{rejected} \mid \text{prompt})} \right)
$$

No reward model, no PPO. Much more stable. Often equivalent results. The 2026 default.

### Successors: ORPO, KTO, IPO, SimPO, GRPO

- **ORPO** — combines SFT and DPO in one loss. Simpler pipeline.
- **KTO** — uses single-rating (not pairs) preference data.
- **IPO** — robust to preference noise.
- **SimPO** — reference-model-free DPO variant.
- **GRPO** (Group Relative Policy Optimization) — used in DeepSeek-R1 and successors. No critic, no reference model in some variants; uses group statistics for advantage.

The field moves quickly. Internalize SFT + DPO + one PPO-variant; track the rest in the literature.

### RL for Other ML Systems

- **Contextual bandits** for recommendations — A/B testing's smarter cousin
- **Off-policy evaluation** — given logged data from policy A, estimate policy B's performance without deploying it
- **RL in pricing, inventory, routing** — when the action affects the next state and the reward, RL beats supervised learning

### Exercises

1. Implement a simple DPO loop on a small preference dataset.
2. Read the DPO paper and the GRPO/DeepSeek-R1 papers.
3. Experiment with TRL (HuggingFace's RL library) — SFT then DPO on a 1B model on a small task.

---

## Phase 6 — Agentic Systems

Multi-step LLM applications with tool use. Increasingly the F50 frontier.

### What Makes a System "Agentic"

- **Multi-step:** the model decides what to do next based on intermediate results
- **Tool use:** the model can call external functions / APIs / databases
- **Memory:** state persists across turns or even sessions
- **Self-correction:** the model can detect and recover from its own errors

### The Patterns

- **ReAct** — interleave reasoning ("I should look up X") and acting ("call tool X"). The classic.
- **Plan-and-execute** — generate a plan, then execute steps. Better for long horizons.
- **Tool-calling APIs** — OpenAI / Anthropic function-calling; the modern standard.
- **Code execution as the tool** — let the LLM write Python; execute it in a sandbox; feed back the output. Very general, very powerful, very risky.

### Frameworks

- **LangChain / LangGraph** — most mature; criticized for over-abstraction.
- **LlamaIndex** — strong for RAG-flavored agents.
- **Haystack** — search-flavored.
- **Autogen** (Microsoft) — multi-agent collaboration.
- **CrewAI** — role-based multi-agent.
- **Custom Python** — most production agents at frontier labs are bespoke.

For F50 portfolio work, **build at least one agent from scratch in plain Python** before reaching for a framework. Frameworks hide the design space you need to understand.

### Operational Challenges

- **Cost** — agents can spin in loops; bound the iteration count, the token budget per session
- **Latency** — many round trips; user-facing agents need streaming + intermediate status
- **Safety** — tool calls with side effects (sending email, modifying databases) need confirmation, dry-run modes, allowlists
- **Observability** — log every step; visualize the trace. Langfuse, Weave, LangSmith are designed for this.
- **Evaluation** — much harder than single-turn; end-to-end success rate, per-step quality, recovery rate from errors

### Agent Observability and AgentOps

Agent observability is qualitatively different from LLM observability. A single LLM call has one prompt and one response. An agent has a multi-step trace, a tool-call tree, branches that were considered and rejected, retries on failure, and potentially parallel sub-agents. Standard LLM tracing tools show you a flat list of spans; agent debugging requires a tree.

**What to capture per agent step:**

- Prompt version (which system prompt, which few-shot examples)
- Tool name, arguments passed, result returned
- Step latency (time to decision + time in tool)
- Token cost for this step (accumulating across the session)
- Decision rationale (the model's "thought" in ReAct; the plan step in plan-and-execute)
- Whether this step was a retry and why the previous attempt failed

**Tools:**

- **AgentOps** (MIT-licensed, 400+ LLM integrations) — wraps agent frameworks and provides a session replay UI. Time-travel debugging: step through the agent's execution, see what the model saw at each decision point, replay from any intermediate state. This is the capability that matters most when debugging an agent that took a wrong branch 12 steps into a 30-step run.
- **Braintrust** — 1M free spans per month; first-class support for multi-step traces with nested tool calls; eval integration means you can catch the same failure in CI before it reaches production.
- **Langfuse** — open-source, self-hostable; session-level traces with parent/child span hierarchy; good for teams that can't send data to a managed service.
- **OTel GenAI semconvs** — the underlying standard. LangChain, CrewAI, and AutoGen emit OTel-compliant agent spans; the `gen_ai.tool.name` and `gen_ai.tool.call.id` attributes on child spans give you the tree structure that maps to the agent's actual decision tree. Covered in the LLMOps and monitoring sections of this course.

**The distinct failure modes to instrument for:**

- **Loop detection** — agent revisits the same tool call with the same arguments N times; budget exceeded without progress
- **Tool error propagation** — a tool returns an error; does the agent recover, retry with different args, or hallucinate a result?
- **Context window pressure** — as the conversation grows, earlier context gets truncated; the agent "forgets" earlier reasoning
- **Latency per step** — a single slow tool call cascades through every downstream step; per-step latency attribution is essential

The ops rule: any agent that touches a production system (sends email, writes to a database, calls an external API) must have a **dry-run mode** that logs all intended tool calls without executing them. Run dry-run on every new prompt version before enabling live execution.

### Exercises

1. Build a 3-tool agent in plain Python (calculator, web search, file read). Implement ReAct manually.
2. Add tracing — every tool call logged with inputs, outputs, latency.
3. Build an eval harness with 30+ test scenarios. Measure success rate.
4. Instrument the agent with AgentOps or Langfuse. Replay a failed session. Identify the exact step where the agent went wrong.
5. Inject a tool that intermittently fails (returns an error 30% of the time). Observe how the agent handles it. Add detection for loop behavior.

---

## Phase 7 — Vector Search and Retrieval Internals

### ANN Algorithms in Depth

**HNSW** (Hierarchical Navigable Small World):

- Build a multi-layer graph; each layer is a "small-world" navigable graph
- Search top-down: at each layer, navigate to the nearest neighbor; descend
- Parameters: M (graph degree), efConstruction (build effort), efSearch (search effort)
- Trade-off: higher M / ef → better recall, more memory, slower

**IVF** (Inverted File Index):

- Cluster the vectors (k-means). Vectors stored by cluster.
- Search: find nearest centroids; scan only those clusters.
- Parameter: nlist (clusters), nprobe (clusters scanned at query). Higher nprobe → better recall, slower.

**PQ** (Product Quantization):

- Split each vector into subvectors; quantize each independently with a small codebook
- Massive memory savings (16x typical); some recall loss
- Often combined: **IVF-PQ** for billion-scale on cheap hardware

**DiskANN** — disk-backed; great for billion-scale at modest cost.

### Hybrid Search Math

$$
\text{score} = \alpha \cdot \text{normalize}(\text{vector\_score}) + (1 - \alpha) \cdot \text{normalize}(\text{bm25\_score})
$$

How to pick α? Use a labeled eval set. Tune α to maximize NDCG@10 or recall@10. Often α=0.5–0.7 works.

**Reciprocal Rank Fusion (RRF)** — alternative score combination that doesn't require score normalization. Robust default.

### Reranking

A cross-encoder (BERT-class model, fine-tuned for relevance) scores each candidate against the query. Way slower per pair but applied only to top 50–200 candidates from first-pass retrieval. Big quality lift.

### Embedding Model Choice

In 2026:

- **OpenAI text-embedding-3-large** — strong, expensive
- **Cohere embed-english-v3** — strong, expensive, multilingual
- **BGE-large-v1.5**, **E5-large-v2** — strong open-source baselines
- **Domain-specific** — fine-tune on your data with contrastive learning (in-batch negatives, hard negative mining)

Dimensionality matters: 1536-dim has more capacity but higher storage; 256-dim or 384-dim is often sufficient and 4–6x cheaper.

### Exercises

1. Build an HNSW index from scratch (or use `hnswlib`). Tune M / ef. Measure recall vs latency.
2. Build IVF-PQ index. Compare to HNSW on a 1M-vector corpus.
3. Fine-tune an embedding model on your domain. Use contrastive loss with in-batch negatives.

---

## Phase 8 — Streaming Internals (Flink, Kafka)

### Watermarks

Flink's answer to "when have I seen all events for a window?"

```
watermark = max_event_time_seen - allowed_lateness
```

When the watermark passes the end of a window, the window closes. Late events go to a side output.

### State Backends

- **HashMapStateBackend** — JVM heap; fast; size-limited
- **EmbeddedRocksDBStateBackend** — disk-spillable; TB-scale state; slower per access
- **Remote state** (Flink 1.18+) — emerging

### Checkpointing

Periodic asynchronous snapshots of all state to durable storage (S3). On failure, restore from the last checkpoint. Exactly-once via the Chandy-Lamport algorithm.

### Savepoints

User-triggered durable snapshots. Stop the job, change code, restart from savepoint. The feature that makes Flink production-ready.

### Kafka KRaft

Kafka removed Zookeeper. KRaft mode is Kafka's own Raft-based metadata layer. Standard for new deployments.

### Exactly-Once Across Kafka and a Sink

Two-phase commit: prepare → barrier flows through job → commit on completion. Requires sink support (Iceberg, Kafka transactional producer, JDBC with transactions).

### Exercises

1. Run a Flink job that processes a Kafka topic and writes to Iceberg with exactly-once semantics. Trigger a failure mid-run. Verify no duplicates.
2. Trigger a savepoint. Modify the job. Restore.
3. Read [Streaming 101](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-101/) and [102](https://www.oreilly.com/radar/the-world-beyond-batch-streaming-102/) by Tyler Akidau.

---

## Phase 9 — Kubernetes for ML at Depth

### GPU Scheduling

- **NVIDIA Device Plugin** — exposes `nvidia.com/gpu` as a schedulable resource
- **MIG (Multi-Instance GPU)** — partition an A100/H100 into smaller instances. Schedule like multiple GPUs.
- **GPU Operator** — installs drivers + plugin + monitoring in one operator
- **Time-slicing** — allow multiple pods to share a GPU (no isolation; for non-prod)

### Networking for ML

- **NCCL** for GPU collective comms; needs proper IPC and shared-memory setup
- **InfiniBand / RoCE** via SR-IOV or Multus
- **Topology-aware scheduling** — co-locate pods that need to communicate
- **NodePort vs LoadBalancer vs Ingress** — picking the right service type for serving

### Storage

- **PV / PVC abstractions**
- **CSI drivers** for cloud storage (EBS, EFS, GCS, FSx for Lustre)
- **Lustre / FSx Lustre / WekaFS** for high-throughput training data
- **Tiered storage**: hot SSD per-node + warm shared filesystem + cold S3

### Autoscaling

- **HPA** — pod replicas by CPU/memory or custom metric
- **VPA** — pod resource requests
- **KEDA** — event-driven autoscaling (Kafka lag, queue depth, custom)
- **Cluster Autoscaler** — node count
- **Karpenter** (AWS) — more advanced node provisioning

### Operators You'll Touch

- Kubeflow Training Operator (PyTorchJob, TFJob, XGBoostJob, ...)
- KubeRay (RayCluster, RayJob, RayService)
- KServe (InferenceService)
- Spark Operator (SparkApplication)
- Flink Operator (FlinkDeployment)
- Argo Workflows (Workflow CRD)

### Multi-Tenancy

- Namespaces + RBAC
- Network Policies (Calico, Cilium)
- ResourceQuotas, LimitRanges
- Hierarchical Namespaces (HNC) for organizational structure
- Pod Security Standards
- vCluster for true multi-tenant Kubernetes-in-Kubernetes

### Exercises

1. Set up MIG on a single A100 (or simulate). Schedule three small pods on three MIG instances.
2. Run Karpenter in a kind cluster (with a karpenter-on-kind setup). Watch it provision nodes for a workload.
3. Build a GitOps stack: Argo CD watching a Git repo; every commit reconciles the cluster.

---

## Phase 10 — Observability for ML, Done Properly

### The Stack

- **Metrics:** Prometheus + Mimir (long retention)
- **Logs:** Loki + Vector (collector)
- **Traces:** Tempo + OpenTelemetry SDKs
- **Profiles:** Pyroscope + Grafana for continuous profiling
- **Dashboards / alerting:** Grafana + Grafana OnCall / PagerDuty

All Grafana-stack — the dominant open-source observability stack.

### What to Instrument

In an ML service, in addition to standard request/error/duration:

- Model version label on every metric
- Feature distribution histograms (per feature, per window)
- Prediction distribution histograms
- Per-slice metrics (group, country, segment)
- Feature freshness (max age of features at prediction time)
- Cache hit/miss for features and predictions
- LLM-specific: tokens in, tokens out, time-to-first-token, tokens/sec, KV cache utilization
- GPU-specific: utilization, memory, power, temperature

### Slicing

Always slice. Aggregate hides:

- Geo slice — a region degraded
- Tenant / customer slice — one big tenant broken
- Device slice — mobile vs desktop
- Cohort slice — new users vs existing

### SLOs and Error Budgets

- Define SLOs: availability, latency, freshness, accuracy
- Burn-rate alerting: alert when current burn rate would exhaust the budget early
- Tie deployment freezes to error budget exhaustion. Forces the team to invest in reliability.

### LLM-Specific Observability

- Token cost per request, per endpoint, per tenant
- Time-to-first-token (latency-critical for streaming UX)
- Generation length distribution
- Cache hit rates (semantic + exact)
- Refusal / safety filter triggers
- Eval scores over time (drift in LLM quality is real)

Tools: Langfuse, W&B Weave, Braintrust, Helicone.

### Exercises

1. Instrument your serving service with Prometheus + OpenTelemetry.
2. Build a Grafana dashboard with at least 12 panels covering system + model + business metrics.
3. Define 3 SLOs. Implement burn-rate alerts.
4. For an LLM project, add Langfuse tracing. Look at traces; identify a slow span.

---

## Phase 11 — Security for ML

### The ML Attack Surface

- **Model extraction** via API querying
- **Adversarial examples** — perturbations that flip decisions
- **Membership inference** — figure out whether a record was in the training set
- **Model inversion** — reconstruct training data from outputs
- **Data poisoning** — bad training data corrupting the model
- **Prompt injection** — LLM-specific; user input bypasses the system prompt
- **Indirect prompt injection** — malicious instructions in retrieved content
- **Tool abuse** in agents — getting the model to take harmful actions
- **Supply chain** — malicious pretrained weights, pip packages, container images

### Mitigations

- **Authentication and rate limiting** on every inference endpoint
- **Differential privacy** in training, where threat model warrants — adds calibrated noise; bounds membership inference. Implementation: Opacus (PyTorch), TF-Privacy.
- **Federated learning** when data can't leave its source (healthcare, banking).
- **Watermarking** model outputs (e.g., text watermarking for LLMs; image watermarking for diffusion).
- **Input sanitization** for LLMs — prompt injection detection, PII redaction.
- **Output filtering** — toxicity, hallucination detection.
- **Sigstore for ML artifacts** — sign model weights; verify on load.
- **SBOMs** for ML pipelines.

### Secrets Management

- Never in Git, never in container images, never in DAG code
- AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault
- External Secrets Operator on Kubernetes
- OIDC for short-lived cloud credentials in CI

### Regulatory Specifics for ML

- **EU AI Act** — high-risk systems require risk management, technical docs, data governance, transparency, human oversight, robustness, post-market monitoring.
- **NYC Local Law 144** — annual bias audit for employment AI.
- **NIST AI RMF** — voluntary risk framework; increasingly a baseline.
- **GDPR Article 22** — restrictions on solely automated decisions.
- **HIPAA** for health — PHI protection, audit logging, BAA.
- **Colorado AI Act, California regulations** — emerging US state-level.

### Exercises

1. Add authentication (API key) to your serving service. Add per-key rate limiting.
2. Train a small model with Opacus. Compare quality and DP-noise trade-off.
3. Try a prompt-injection attack on your LLM project. Then build a detector.

---

## Phase 12 — Governance and Compliance

### Model Inventory

Every production model registered with:

- Owner (team, individual)
- Purpose (the problem it solves)
- Training data lineage (which sources, versions)
- Performance per slice (overall and per protected attribute where applicable)
- Known limitations (out-of-distribution behavior, failure modes)
- Deployment scope (which services, which traffic)
- Approval status and reviewers

### Model Cards

For each model, a public-ish document:

- Intended use, intended users
- Out-of-scope uses
- Training data summary (sources, period, sample size, demographic breakdown)
- Performance overall and per slice
- Known biases and limitations
- Ethical considerations
- Last updated, version, changelog

Format: see [HuggingFace model cards](https://huggingface.co/docs/hub/model-cards) for a public-facing example. For internal, your registry can host them.

### Audit Trails

For regulated domains, log:

- Every promotion (who, when, which version, evidence pack)
- Every prediction (sampled for high-volume; full for low-volume regulated decisions like credit)
- Every label feedback
- Every model card change
- Every contract change

Retain 7+ years for finance/healthcare. Make queryable.

### Approval Workflows

Bake into the registry. Promotion to Production for high-risk models requires:

- Bias audit passed
- Security review
- Risk committee sign-off
- Tested rollback plan

### Explanation and Recourse

Where applicable (credit, insurance, employment):

- SHAP / LIME / counterfactual explanations
- Human-readable reason codes
- Appeal / manual-review path

### Exercises

1. Add a model card template to your project. Generate one for your model.
2. Add an audit log table; log promotions and a sampled fraction of predictions.
3. Add a SHAP-based explanation endpoint. Show how a specific prediction is explained.

---

## Phase 13 — Modeling Beyond the Defaults

### Causal Inference

Most ML predicts; some problems need to *intervene*. "If I show this ad, will the user buy?" is causal, not predictive.

- **Randomized experiments** (A/B tests) — the gold standard
- **Propensity score matching** — for observational data
- **Difference-in-differences** — natural experiments
- **Synthetic control** — comparing affected unit to a weighted average of unaffected units
- **Uplift modeling / treatment-effect estimation** — predict treatment effects, not outcomes
- **Doubly Machine Learning (DML)** — combines ML with causal estimation; semi-parametric

Tools: EconML (Microsoft), CausalML (Uber), DoWhy.

### Bandits and Online Learning

- **Multi-armed bandits** — Thompson sampling, UCB, ε-greedy
- **Contextual bandits** — when context matters
- **Reinforcement learning** — when actions affect future states

Used heavily in recommendations, ads, dynamic pricing.

### Time-Series Forecasting

- **Classical:** ARIMA, ETS, Prophet
- **DL:** N-BEATS, NHITS, Temporal Fusion Transformers, TimesNet
- **Foundation models for time series:** TimeGPT, Chronos, Lag-Llama — the emerging frontier

For most business forecasting, simple beats fancy. Always try a strong baseline (last-week-this-day, sliding mean, Prophet) before reaching for transformers.

### Graph ML

- **GNNs** — message-passing networks for graph data
- **Use cases:** fraud rings, recommendation, drug discovery, supply-chain
- **Frameworks:** PyTorch Geometric, DGL

For most companies, graph problems are solved with graph databases (Neo4j) + heuristics. GNNs come in when you have lots of labeled graph data and the patterns are subtle.

### Tabular Deep Learning

Despite the deep learning revolution, **gradient boosting (XGBoost, LightGBM, CatBoost) still wins on most tabular problems** in 2026. Tabular DL has matured (TabNet, FT-Transformer, SAINT) but hasn't displaced GBT for most use cases.

Senior engineers know when to *not* reach for deep learning.

---

## Phase 14 — Backup, Disaster Recovery, Business Continuity for ML

### RPO and RTO for ML Artifacts

- **RPO (Recovery Point Objective):** how much can you lose? For training data, often hours; for model artifacts, ideally zero.
- **RTO (Recovery Time Objective):** how fast back online? For real-time serving, minutes.

### What Needs Backup

- Model artifacts (registry contents)
- Feature definitions and historical features
- Training data
- Configuration (DAGs, prompts, feature definitions, deployment manifests in Git)
- Online feature store (with reconstruction-from-offline as fallback)

### Multi-Region for ML Serving

- Active/active for stateless serving (with cross-region model replication)
- Active/passive for training (cheaper; failover for DR only)
- Online feature stores: replicated or rebuildable from offline
- Avoid cross-region inference at request time (latency, cost)

### What Can Go Wrong

- A single AZ outage
- A region outage (rare but real)
- A bad model deploy (most common; mitigated by canary + automated rollback)
- A feature pipeline regression (silently wrong predictions; the worst kind of outage)
- A vendor outage (OpenAI / Anthropic down → your LLM app down). Multi-provider routing is the mitigation.

### Runbooks

For each system, a one-pager runbook:

- How to detect a failure
- Immediate mitigations (rollback model, switch traffic, route to backup provider)
- Investigation steps
- Communication plan
- Post-incident review template

### Multi-Region LLM Serving

Multi-region serving for LLMs is harder than multi-region serving for classical ML. The model is stateless; the compliance constraints are not.

**Control-plane / data-plane split:**

The control plane (routing decisions, quota enforcement, configuration) can be centralized. The data plane (actual inference) must be regional. A request from Frankfurt should never cross to us-east-1 for inference — latency and residency both prohibit it. The AI gateway layer (covered in the LLMOps section) handles the routing; the inference fleet is deployed per-region.

**The six data residency leak surfaces:**

Data residency is not just about where inference runs. A compliant system must trace all six paths:

1. **Inference** — prompt and response processed in the correct region (the obvious one)
2. **Telemetry** — OTel spans, logs, and metrics must export to a regional endpoint, not a global collector
3. **Eval pipelines** — if eval jobs use production prompts as test cases, those prompts cannot leave the region for evaluation compute
4. **Prompt caches** — semantic caches must be regional; a cached response from a EU user must not be retrievable by a US instance
5. **Fine-tune feedback loops** — if production interactions feed back into fine-tuning datasets, that data movement must respect residency
6. **Observability** — dashboards and alerting platforms that aggregate cross-region data must not expose individual prompts to operators in the wrong jurisdiction

Most teams get #1 right and miss #2–6. An architecture review for EU AI Act or GDPR compliance should walk all six explicitly.

**Region-specific model availability:**

Model availability is not uniform across regions. AWS Bedrock, Azure OpenAI, and Vertex AI all have different model catalogs per region. Claude 3.5 Sonnet may be available in us-east-1 but not eu-central-1 at a given point in time. Your gateway's model routing table must be region-aware, with fallback logic that respects both residency constraints and availability.

**Failover automation and brownout testing:**

Standard health checks detect hard failures (503, timeout). LLM serving also degrades gracefully: a provider may be available but returning degraded quality or elevated latency. Brownout testing — deliberately injecting latency or error rates on one region's provider path — validates that your failover logic triggers on soft failures, not just hard ones. Run brownout tests quarterly; they catch routing bugs that health checks miss.

**Research grounding:**

The cross-region load balancing problem for LLM serving has been formalized in recent work. SkyWalker (arXiv:2505.24095) and GORGO (arXiv:2602.11688) both address the optimization of cross-region request routing under latency, cost, and capacity constraints — useful reading if you're designing the routing policy for a large fleet.

---

## Phase 15 — Cost and FinOps for ML

### The Cost Model

| Layer | Drivers | Typical % of bill |
|---|---|---|
| Training | GPU-hours, dataset prep compute | 20–40% |
| Inference | GPU-hours, request volume, model size | 30–60% |
| Storage | TB-months across hot/warm/cold | 5–15% |
| Data egress | Cross-region, cross-cloud | 5–15% |
| Vendors | LLM API spend, observability, registries | 5–25% |
| Engineering tooling | W&B, MLflow Cloud, etc. | 1–5% |

### The Levers

- **Inference:** quantize, distill, batch, route, cache, scale-to-zero, right-size hardware
- **Training:** spot instances, early stopping, smarter HPO, smaller search spaces, transfer learning
- **Storage:** tier old data; compress; lifecycle to cheap storage
- **Egress:** co-locate data and compute; VPC endpoints
- **Vendors:** negotiate annual commits, route to cheaper provider when quality allows
- **People-process:** chargeback (not showback), per-team budgets, weekly cost review

### LLM Cost Specifically

- Token cost dominates. Compress prompts (LLMLingua), shorter outputs, JSON not prose.
- Self-host the most-used models. The break-even at 2026 prices is roughly: if you spend >$10K/month on a hosted LLM, evaluate self-hosting.
- Distill expensive models into cheap ones for specific tasks.
- Cache aggressively (semantic cache catches more than exact-match).

### Exercises

1. Pick one of your projects. Build a per-day cost breakdown across training / inference / storage / egress / vendors.
2. Identify the top 3 cost drivers. Reduce one by 50%. Document.

---

## Phase 16 — Architectural Patterns for ML

### The Outbox Pattern

When the app makes a prediction *and* publishes an event ("we predicted X for user Y"), do both in the same DB transaction (write to an outbox table), then a separate process reads outbox and publishes to Kafka. Prevents inconsistency between app state and event stream. CDC reads outbox directly.

### Event Sourcing

Store the log of events as the source of truth. Current state is a fold of events. Kafka + Iceberg is event-sourcing-shaped. Multiple read projections (warehouse, online feature store, OLAP) is CQRS.

### Reverse ML

The pattern of pushing model outputs back into operational tools. Predictions to Salesforce, scores to marketing automation, segments to ad platforms. Tools: Hightouch, Census, Polytomic.

### Lambda vs Kappa for ML

- **Lambda:** two paths (batch + speed), merge for queries. Complex; two codebases.
- **Kappa:** one streaming path; reprocess history by replaying. Simpler; needs stream retention.

Most modern ML platforms are Kappa-ish over Iceberg/Delta. Batch is just "view the streaming-written table."

### Model as a Product

The mindset shift: a model isn't a side effect of training; it's a deliverable with consumers, SLAs, contracts, documentation, versioning, deprecation policy. This is what makes a "model platform" work at scale (Project 7).

### Federated Learning

Train a global model on data that stays distributed. Each node trains locally; gradients (or models) aggregated centrally. Used in healthcare, finance, mobile (Google's keyboard). Frameworks: Flower, FedML, NVIDIA FLARE.

---

## Phase 17 — The Bookshelf

### Tier 1 — Read These

1. **Designing Machine Learning Systems** — Chip Huyen. The textbook. Twice.
2. **Designing Data-Intensive Applications** — Martin Kleppmann. The distributed systems bible.
3. **Machine Learning Design Patterns** — Lakshmanan, Robinson, Munn.
4. **Reliable Machine Learning** — Chen et al. (O'Reilly).

### Tier 2 — Strongly Recommended

5. **Practical MLOps** — Noah Gift.
6. **Building Machine Learning Powered Applications** — Emmanuel Ameisen.
7. **The Hundred-Page Machine Learning Book** — Andriy Burkov (foundations refresher).
8. **Deep Learning** — Goodfellow, Bengio, Courville (the formal text).
9. **Streaming Systems** — Akidau, Chernyak, Lax.
10. **Database Internals** — Alex Petrov.

### Tier 3 — LLM-Specific

11. **Hands-On Large Language Models** — Jay Alammar, Maarten Grootendorst.
12. **Build a Large Language Model (From Scratch)** — Sebastian Raschka.
13. **Generative Deep Learning** (2nd ed) — David Foster.

### Papers Worth Reading

- **Attention Is All You Need** (Vaswani et al.) — Transformers.
- **GPT-3, GPT-4 technical reports** — what scale does.
- **LLaMA, LLaMA-2, LLaMA-3 papers** — open model design.
- **InstructGPT** — RLHF for alignment.
- **DPO paper** — the simpler alternative.
- **FlashAttention 1/2/3** — attention optimization.
- **vLLM paper** — efficient LLM serving.
- **Chinchilla** — scaling laws.
- **Mixtral / DeepSeek-V3** — MoE in practice.
- **Constitutional AI** (Anthropic) — RLAIF.
- **The DeepSeek-R1 paper** — GRPO and reasoning training.
- **Mamba / RWKV** — alternatives to attention.

### Blogs and Newsletters

- **Chip Huyen's blog**
- **Eugene Yan**
- **Sebastian Raschka's Magazine**
- **Lilian Weng's blog** (OpenAI)
- **Hugging Face blog**
- **Anthropic / OpenAI / DeepMind research blogs**
- **The Latent Space podcast and newsletter**
- **MLOps Community Slack and newsletter**
- **The Pragmatic Engineer** (Gergely Orosz) for engineering culture
- **Databricks / Snowflake / NVIDIA engineering blogs**

### Conferences (Watch Talks Online)

- **NeurIPS, ICML, ICLR** — research; pick MLOps-adjacent papers
- **MLOps World**
- **NVIDIA GTC** — infrastructure at scale
- **KubeCon + AI/ML Day**
- **Data + AI Summit** (Databricks) — practical at scale
- **Ray Summit** — Ray + distributed ML

---

## Phase 18 — A Closing Note

You'll never finish this curriculum. New tools appear monthly. The point isn't to know everything; it's to internalize the **underlying patterns** so deeply that any new tool slots into your mental model in a day.

Patterns that recur:

- Storage/compute separation, reinvented every cycle
- Lazy evaluation and predicate pushdown in every modern engine
- Eventually-consistent replication with strong-consistency islands
- Idempotency as the primary defense against distributed failures
- Schema evolution as a first-class operational concern
- Cost as a function of bytes scanned, parameters, tokens, compute time
- The training-serving skew bug, in some new disguise, every few months
- "Just batch it" as the answer to most throughput problems
- Caching at every layer

Master these and the rest is vocabulary.

---

## What to Do Next

You've now seen the full landscape. The honest path forward:

1. **Finish the foundations through specialization chapters.** Work through them sequentially.
2. **Pick a specialization** from the next-steps chapter and go deep.
3. **Build two portfolio projects** from the Fortune 50 portfolio chapter. Slowly. Deeply.
4. **Use this chapter as reference** when problems push you into new territory.
5. **Read Designing Machine Learning Systems and DDIA at least twice each.**

The compound interest on solid fundamentals over 12–18 months is genuinely transformative. Most candidates skip them and stay mid-level forever. Don't be most candidates.

When you're ready to think about the role *above* senior IC, continue to the ML architect track.

---

## You can now

- Compute training memory by hand for a given model and optimizer, and choose the right ZeRO stage / FSDP sharding and 3D-parallelism layout for the interconnect you have.
- Select the correct inference optimization for a bottleneck — quantization (AWQ/GPTQ/FP8), PagedAttention KV-cache paging, continuous batching, speculative decoding, or an architectural trick like GQA/MoE.
- Map a distributed-systems consistency choice (CAP/PACELC, replication, partitioning, consensus) onto the right ML component — feature cache, model promotion, or event stream.
- Reason about the six data-residency leak surfaces for multi-region LLM serving and design a control-plane/data-plane split that survives an EU AI Act or GDPR review.
- Diagnose an agentic system through a tool-call trace tree — loop detection, tool-error propagation, context-window pressure — and enforce dry-run mode on side-effecting tools.
- Recognize, from the recurring patterns, which advanced topic to consult when a real production system pushes you into unfamiliar territory.
