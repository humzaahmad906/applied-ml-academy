# Reading Papers, Canonical List, Glossary & Staying Current

This is the reference layer: a method for reading any paper fast, the canonical papers that define the field (read these and 90% of new work is recognizable as variations), a dense glossary, the benchmark landscape, and how to keep up without drowning.

---

## 1. How to read an ML paper (a three-pass method)

You almost never read a paper linearly start-to-finish. Do this instead:

**Pass 1 — 5 minutes, decide if it's worth more:**
Title → abstract → figures (especially the architecture diagram and the main results table) → conclusion. After this you should be able to state: *what problem, what's the one idea, what's the headline result.* If you can't, the paper is either bad or you're missing a prerequisite (go fill it from the earlier chapters).

**Pass 2 — 30 minutes, understand the contribution:**
Read intro, method, and results carefully; skim related work; ignore proof details. Run the four-primitive classification from the knowledge map: which primitive (sequence model / encoder / training signal / control loop)? What baseline, what axis, what tradeoff? Now you understand the paper well enough to talk about it.

**Pass 3 — only for papers you'll build on:**
Re-derive the method as if you were the author. Question every choice. Find the unstated assumptions and the hidden costs. This is where real understanding (and the ability to spot flaws) lives.

**The five questions to answer for every paper:**
1. What's the **one-sentence idea**?
2. What does it **beat**, on **what axis** (quality/speed/memory/cost/context/data)?
3. What did it **trade away** (find the cost even if they hide it)?
4. Does it **survive** at a different scale / hardware / base model?
5. Is the **evaluation honest** (right baselines, no contamination, ablations that isolate the claimed cause)?

**Red flags:** no ablations (so you can't tell what actually caused the gain), only one seed / no variance, suspiciously strong results vs a weak baseline, benchmark contamination, "SOTA" on a metric nobody cares about, missing cost/latency numbers.

---

## 2. The canonical reading list

Read (or deeply understand the idea of) these and the field stops being intimidating. Grouped by area; roughly chronological within each.

**Foundations / architecture**
- *Attention Is All You Need* (Vaswani 2017) — the Transformer. Non-negotiable.
- *BERT* (2018) — encoder-only, masked LM, the pretraining-then-finetune paradigm.
- *GPT-2 / GPT-3* (2019/2020) — decoder-only scaling, in-context/few-shot learning.
- *RoFormer / RoPE* (Su 2021) — rotary position embeddings.
- *Chinchilla* (Hoffmann 2022) — compute-optimal scaling laws (~20 tokens/param).
- *LLaMA* (2023) — the open-model recipe (RMSNorm, SwiGLU, RoPE) that everyone copied.
- *Mixtral* / *Switch Transformer* — sparse MoE.
- *Mamba* (Gu & Dao 2023) — selective state-space models.
- *DeepSeek-V3* (2024) — MLA + MoE + FP8 + MTP; the modern efficient-flagship template.

**Training / alignment / reasoning**
- *InstructGPT* (Ouyang 2022) — RLHF (SFT → RM → PPO).
- *DPO* (Rafailov 2023) — preference optimization without RL.
- *Chain-of-Thought prompting* (Wei 2022) and *Self-Consistency* (2022).
- *DeepSeek-R1* (2025) — RLVR + GRPO; reasoning emerges from pure RL. The most important post-training paper of the era.
- *Tülu 3* (2024/25) — open, documented full post-training pipeline incl. RLVR.

**Efficiency / inference**
- *FlashAttention* 1/2/3 — IO-aware exact attention.
- *PagedAttention / vLLM* (Kwon 2023) — paged KV cache serving.
- *GPTQ*, *AWQ*, *QLoRA* — quantization (PTQ + 4-bit fine-tuning).
- *Speculative decoding* (Leviathan 2023) and *EAGLE* — draft-and-verify.
- *LoRA* (Hu 2021) — low-rank adapters.

**Vision-language**
- *ViT* (Dosovitskiy 2020) — transformers for images.
- *CLIP* (Radford 2021) and *SigLIP* (2023) — contrastive image-text encoders.
- *Flamingo* (2022) — cross-attention fusion, few-shot multimodal.
- *BLIP-2* (2023) — Q-Former.
- *LLaVA* (2023) — the projector + instruction-tuning recipe most open VLMs use.
- *Qwen2-VL* — native dynamic resolution + M-RoPE.
- *Chameleon* / *Emu3* — early-fusion, unified token-space multimodal.

**RAG / retrieval**
- *RAG* (Lewis 2020) — the original.
- *DPR* (2020), *ColBERT* (2020) — dense retrieval, late interaction.
- *HyDE* (2022), *RAPTOR* (2024), *Self-RAG* (2023).
- *GraphRAG* (Edge 2024) — graph-structured corpus-level RAG.

**Agents**
- *ReAct* (Yao 2022) — reason+act interleaving.
- *Reflexion* (Shinn 2023) — verbal self-improvement.
- *Toolformer* (2023), *Tree of Thoughts* (2023).
- *Voyager* (2023) — skill-library / lifelong learning agent.
- (Standard, evolving) MCP spec; SWE-bench (eval).

You don't need to read all of these end-to-end. Understanding each one's *core idea* (which the earlier chapters give you) is the goal; read in full only the ones in your build path.

---

## 3. Dense glossary (the words papers assume you know)

**Architecture**
- **Autoregressive** — generates one token at a time, each conditioned on all previous.
- **Causal mask** — prevents attending to future tokens; makes decoding valid.
- **Residual stream** — the running hidden-state vector each layer reads from and writes to.
- **d_model / hidden size** — width of the residual stream. **d_ff** — FFN inner width (~4×).
- **Head** — one of the parallel attention sub-spaces. **head_dim** = d_model / n_heads.
- **MHA / MQA / GQA / MLA** — attention variants by how K/V heads are shared/compressed (LLM chapter).
- **MoE / experts / router / top-k / active params** — sparse FFN (LLM chapter).
- **Dense model** — all params active per token (opposite of MoE).
- **SSM / Mamba / linear attention / hybrid** — sub-quadratic sequence mixers (LLM chapter).
- **RoPE / ALiBi / NTK / YaRN / PI** — positional encoding & context-extension (foundations).
- **RMSNorm / LayerNorm / pre-norm** — normalization (foundations).
- **SwiGLU / GELU / SiLU** — activations/gated FFNs (foundations).
- **Tied embeddings** — sharing input embedding and output projection weights.

**Training**
- **Pretraining / base model** — next-token training on raw text.
- **Cross-entropy / NLL / perplexity** — the LM loss and its exponentiated form.
- **SFT / instruction tuning** — supervised fine-tuning on instruction-response pairs.
- **LoRA / QLoRA / PEFT** — parameter-efficient fine-tuning.
- **RLHF / reward model / PPO / KL penalty** — RL from human feedback (LLM chapter).
- **DPO (and IPO/KTO/ORPO/SimPO)** — direct preference optimization, no RL loop.
- **RLVR** — RL with verifiable (rule-based) rewards.
- **GRPO** — critic-free, group-relative RL algorithm (DeepSeek-R1). **Dr. GRPO** — debiased variant.
- **PRM / ORM** — process vs outcome reward model (per-step vs final-answer).
- **Distillation (soft labels / sequence-level)** — train a small student from a big teacher.
- **Scaling laws / Chinchilla-optimal** — predictable loss-vs-(params, data, compute).
- **Emergent abilities** — capabilities that appear sharply at scale (contested but common term).
- **Catastrophic forgetting** — losing prior abilities when fine-tuning.

**Reasoning / decoding**
- **CoT (chain-of-thought)** — explicit step-by-step reasoning tokens.
- **Test-time / inference-time compute scaling** — spending more compute at inference for better answers.
- **Reasoning model** — trained to think (long CoT) before answering (o1/R1/QwQ).
- **Self-consistency / best-of-N / ToT / MCTS** — ways to spend test-time compute.
- **Temperature / top-k / top-p / min-p / greedy / beam** — decoding strategies (LLM chapter).
- **Constrained / structured decoding** — force valid JSON/grammar output.

**Inference / systems**
- **Prefill vs decode** — compute-bound prompt processing vs bandwidth-bound generation (inference chapter).
- **TTFT / TPOT / ITL / throughput / goodput** — serving latency metrics.
- **KV cache** — cached keys/values of past tokens; the memory/bandwidth bottleneck.
- **FlashAttention** — IO-aware, memory-linear *exact* attention.
- **PagedAttention / continuous batching / prefix caching** — vLLM serving tech (inference chapter).
- **Quantization: PTQ/QAT, W4A16, GPTQ/AWQ/GGUF/NF4/MLX, outliers, KV-quant** (inference chapter).
- **Speculative decoding / draft model / Medusa / EAGLE** — exact decode speedup.
- **Tensor/pipeline/data/expert parallelism, ZeRO/FSDP** — distributed train/serve.

**VLM**
- **ViT / patches / [CLS] token** — vision transformer basics.
- **CLIP / SigLIP / DINO / InternViT** — vision encoders by pretraining objective.
- **Projector / connector: MLP, Q-Former, Perceiver resampler, pixel-shuffle** — vision→LLM bridge.
- **Visual tokens** — encoded image pieces fed to the LLM (continuous or discrete/VQ).
- **Fusion: late / cross-attention / prefix-concat / early-native** — modality interaction spectrum (VLM chapter).
- **AnyRes / native dynamic resolution / M-RoPE** — high-res handling.
- **VQ-VAE / VQGAN** — discrete image tokenizers (enable image generation).
- **Object hallucination / POPE / grounding / referring** — VLM-specific eval/capabilities.

**RAG**
- **Dense vs sparse (BM25) vs hybrid; RRF** — retrieval modes and rank fusion.
- **Bi-encoder vs cross-encoder vs late-interaction (ColBERT)** — retrieval vs reranking models.
- **ANN: HNSW / IVF / PQ / DiskANN** — approximate nearest-neighbor search.
- **Chunking: fixed/recursive/semantic/parent-child/late** — document splitting.
- **HyDE / multi-query / query rewriting / decomposition** — query-side transforms.
- **Reranking** — precision-focused second-stage scoring.
- **RAPTOR / GraphRAG / HippoRAG / Self-RAG / CRAG** — advanced RAG architectures.
- **RAGAS / faithfulness / context precision-recall / nDCG / MRR** — RAG evaluation.

**Agents**
- **Agent loop / policy / scaffolding** — the control loop around the model.
- **ReAct / Reflexion / plan-execute / ToT / LATS** — agent reasoning patterns.
- **Tool / function calling / tool schema / MCP / CodeAct** — acting on the world.
- **Working vs long-term memory; episodic/semantic/procedural; reflection** — memory.
- **Context engineering / compression / offloading / context rot** — managing the window (agents chapter).
- **Orchestrator-worker / role-based multi-agent / context isolation** — multi-agent.
- **SWE-bench / GAIA / WebArena / τ-bench / AgentBench** — agent benchmarks.
- **LLM-as-judge** — using an LLM to score outputs (with known biases).
- **Long-horizon reliability / compounding error** — the central agent failure mode.

---

## 4. Benchmarks — what they actually measure

Knowing benchmarks lets you read results tables critically.
- **General knowledge/reasoning:** MMLU / **MMLU-Pro** (multiple-choice across subjects; saturating/contaminated — treat with suspicion), GPQA (hard, "Google-proof" science), BIG-Bench Hard.
- **Math:** GSM8K (grade school, mostly saturated), MATH, **AIME** (competition; the current reasoning-model headline), Olympiad-level sets.
- **Code:** HumanEval / MBPP (function completion, saturated), LiveCodeBench (contamination-resistant, fresh problems), **SWE-bench / SWE-bench Verified** (real repo issues — the agentic-coding standard).
- **Long context:** Needle-in-a-Haystack (retrieval at depth), RULER, LongBench, ∞Bench.
- **Instruction following / chat:** IFEval, **Chatbot Arena / Elo** (human pairwise preference — the most trusted general signal), MT-Bench, AlpacaEval.
- **VLM:** MMMU (multimodal reasoning), MathVista, DocVQA / ChartQA / TextVQA (document/chart/OCR), POPE (hallucination), MMBench.
- **RAG:** RAGAS metrics, KILT, plus task QA sets (Natural Questions, HotpotQA for multi-hop).
- **Agents:** GAIA, WebArena, τ-bench, AgentBench, PlanBench.

Two standing cautions: **(1) contamination** — many benchmarks leak into training data, inflating scores; prefer fresh/held-out/contamination-resistant ones. **(2) saturation** — once everyone scores ~95%, the benchmark stops discriminating; watch what the frontier has *moved on to*.

---

## 5. How to stay current without drowning

- **Read the architecture-comparison and "state of" syntheses**, not every paper. Well-maintained architecture-comparison write-ups (the recurring "big LLM architecture comparison," "state of LLMs," and "state of RL for reasoning" syntheses) are the single highest-leverage way to track what changed and why — someone else reads the papers so you can read one post. Vision-encoder surveys and the periodic agent/RAG surveys on arXiv serve the same role.
- **Track model technical reports**, which are now the most information-dense source: DeepSeek (V3/R1), Qwen (Qwen3/Qwen3-VL), Llama, Gemma, Kimi, Mistral. The "what's new" section of a frontier report teaches more than ten incremental papers.
- **Curated lists** for depth-on-demand: Awesome-LLM-Inference, Awesome-GraphRAG, agent-memory paper lists, the RLHF/post-training survey on arXiv.
- **Filter by the four-primitive framing:** when a new thing trends, immediately ask "which primitive does it touch, and what's the tradeoff." Most hype collapses under that question; the few that don't are worth real time.
- **Build the habit, not the backlog:** one paper read *properly* (three passes, the five questions) teaches more than ten skimmed. Depth compounds; breadth-without-depth evaporates.

---

## 6. The meta-point

The field looks vast because of vocabulary and pace, not because of conceptual depth. There are maybe a dozen core ideas (next-token prediction, attention and its `n²`, the residual stream, KV caching, the SFT→preference→verifiable-reward ladder, test-time compute, the encoder→projector→LLM VLM template, retrieve-then-ground, the reason-act loop) and everything else is **composition and optimization of those ideas under different constraints** (scale, hardware, modality, latency, reliability). Once the core chapters are internalized, a new paper is almost always a recognizable move in a known game — and *that* recognition, not memorized facts, is what "understanding almost everything" actually means.
