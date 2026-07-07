# The Progression: A Chained Lineage (2017 → 2026) — Part 2 of 2: Efficiency, Reasoning, Multimodal, RAG & Agents Lineages

This is part 2 of the progression timeline. Part 1 covered the prehistory and the main architecture/pretraining/alignment spine. Here we trace five parallel tracks that overlap in time with the main spine and reference each other — efficiency/inference, reasoning/test-time compute, vision/multimodal, RAG/retrieval, and agents — plus the meta-pattern for placing any new paper on these chains.

## Part 2 — Efficiency & Inference Lineage

The whole track is the war on attention's `O(n²)` and the KV-cache memory wall (the attention material in the foundations, and the inference chapter).

**KV-cache / attention shape:**
**MHA (2017)** full but huge cache ⇒ **MQA (Shazeer 2019)** one shared KV head, tiny cache but quality drop ⇒ **GQA (Ainslie 2023)** grouped KV heads, the tunable sweet spot (Llama 2/3, most models) ⇒ **MLA (DeepSeek-V2 2024)** compress KV to a low-rank latent — MHA-level quality at a fraction of the cache, with a decoupled-RoPE trick. (LLM chapter)

**Sparse capacity (MoE):**
**Outrageously Large NN (Shazeer 2017)** MoE on LSTMs, decouple params from compute ⇒ **GShard (2020)** MoE for Transformers + sharding ⇒ **Switch Transformer (2021)** top-1 routing, trillion params, simpler ⇒ **Mixtral (2023)** first big *open* MoE ⇒ **DeepSeekMoE / V3 (2024)** fine-grained + shared experts, *auxiliary-loss-free* balancing — the modern standard. (LLM chapter)

**Sub-quadratic sequence mixing:**
**Linear/efficient attention & RWKV (2020–22)** kernelize away `n²`, often quality gaps ⇒ **Mamba (2023)** selective SSM, strong but weak at exact retrieval ⇒ **Hybrids (Qwen3-Next/Qwen3.5, Kimi Linear, Nemotron 3, MiniMax-M1 lightning attention, 2025–26)** mix cheap linear/SSM blocks with periodic full-attention to get both. (LLM chapter)

**Exact-attention IO efficiency:**
naive attention materializes the `n×n` matrix (memory-bound) ⇒ **FlashAttention (Dao 2022)** tile + online-softmax, never materialize it, `O(n)` memory, *exact* ⇒ **FA-2 (2023)** better parallelism ⇒ **FA-3 (2024)** Hopper warp-specialization + FP8 ⇒ FlashDecoding/FlashInfer for decode + sparsity. (inference chapter)

**Serving:**
contiguous per-request KV → huge fragmentation ⇒ **PagedAttention / vLLM (Kwon 2023)** paged, shareable KV like OS virtual memory, enabling **continuous batching** and **prefix caching** ⇒ FP8 KV-quant, chunked prefill, prefill/decode disaggregation. (inference chapter)

**Compression:**
**LoRA (2021)** low-rank adapters, train <1% of params ⇒ **QLoRA (2023)** LoRA on a 4-bit-quantized base, big models on one GPU ⇒ PTQ **GPTQ (2022)** / **AWQ (2023)** 4-bit weights, activation-aware ⇒ **SmoothQuant** (W8A8 via outlier migration), **GGUF** (llama.cpp consumer), **QAT** + layer-sensitivity maps for on-device int4. (inference chapter)

**Decode acceleration:**
sequential one-token decode wastes idle GPU compute ⇒ **Speculative decoding (Leviathan 2023)** draft-and-verify in parallel, *identical* output ⇒ **Medusa (2024)** extra heads, no draft model ⇒ **EAGLE / EAGLE-3 (2024–25)** feature-level drafting, current SOTA. (inference chapter)

---

## Part 3 — Reasoning & Test-Time-Compute Lineage

How "think before answering" went from a prompt trick to a trained capability and a scaling axis (the reasoning-model material in the LLM chapter).

**Scratchpads (2021)** show your work in intermediate tokens ⇒ **Chain-of-Thought prompting (Wei 2022)** "think step by step" unlocks latent reasoning at scale ⇒ **Self-Consistency (2022)** sample many chains, majority-vote ⇒ **STaR (Zelikman 2022)** bootstrap: keep CoTs that reach correct answers, fine-tune on them, repeat — *train* reasoning from its own correct traces ⇒ **Let's Verify Step by Step / PRMs (OpenAI 2023)** reward *each step*, not just the final answer (process vs outcome supervision) ⇒ **Tree-of-Thoughts / LATS (2023)** search over reasoning branches with backtracking ⇒ **Quiet-STaR (2024)** learn to think between *every* token ⇒ **o1 (OpenAI 2024)** RL-trained long CoT + **test-time compute scaling** as a first-class, secret method ⇒ **DeepSeek-R1 / GRPO / RLVR (Jan 2025)** open it: verifiable rewards + critic-free group-relative RL; reasoning *emerges* from pure RL (R1-Zero), then distill into small models ⇒ **Dr. GRPO / DAPO (2025)** fix length/difficulty biases in GRPO's normalization; **process reward models** and turn-level credit assignment mature ⇒ **DeepSeekMath-V2 (late 2025)** *self-verifiable* reasoning (the model generates *and* checks its own proofs), extending verifiable-reward RL to domains lacking cheap external verifiers. Open debate throughout: does RLVR *create* new reasoning or *elicit* what pretraining already had, and when should the model *stop* thinking (overthinking hurts).

---

## Part 4 — Vision & Multimodal Lineage

How pixels learned to talk to language models (the VLM chapter). Two feeder threads (vision backbones, image generation) converge into VLMs.

**Vision backbones:** **AlexNet (2012)** CNNs win vision ⇒ **ResNet (2015)** residual connections train very deep nets (and, note, *residuals are what the Transformer later borrows*) ⇒ **ViT (Dosovitskiy 2020)** drop convolutions: cut the image into patches and run a *Transformer* — unifies the architecture with NLP (the vision-encoder material in the VLM chapter).

**Language-aligned encoders:** **CLIP (Radford 2021)** + **ALIGN (2021)** contrastively align image and text encoders on web-scale pairs → vision features *already in language space*, zero-shot classification ⇒ **SigLIP (2023) / SigLIP 2 (2025)** sigmoid loss, better/multilingual, the default VLM encoder ⇒ **DINO / DINOv2 (2021–23)** self-supervised, strong *spatial/geometric* features where CLIP is weak. (VLM chapter)

**Vision → LLM fusion (the main VLM line):** **Flamingo (DeepMind 2022)** inject frozen-LLM **cross-attention** + a Perceiver resampler, few-shot multimodal (the deep-fusion end of the VLM fusion spectrum) ⇒ **BLIP / BLIP-2 (2022–23)** the **Q-Former** compresses vision to a few learned query tokens ⇒ **LLaVA (2023)** the dominant simplification: a tiny **MLP projector** maps patches to visual tokens, *concatenate* with text into one decoder, two-stage train (align → instruction-tune) — "visual instruction tuning" (the projector and training-recipe material in the VLM chapter) ⇒ **Qwen2-VL / InternVL (2024)** native **dynamic resolution** + **M-RoPE** + token compression for documents/charts ⇒ **early/native fusion: Chameleon (Meta 2024), Emu3 (2024)** drop the bolted-on encoder entirely — tokenize images (often via VQ) into the *same vocabulary* and train one model on interleaved streams, enabling **image generation** too ⇒ **Llama 4 / Gemini / GPT-4o-style (2025–26)** natively-multimodal frontier (vision stacks largely undisclosed). The arc: *late fusion (CLIP) → cross-attention (Flamingo) → prefix-concat (LLaVA) → early-native (Chameleon)* — barely-interact to one-unified-model. (VLM chapter)

**Image generation (feeds unified models):** **DDPM (Ho 2020)** diffusion: generate by iterative denoising ⇒ **Latent Diffusion / Stable Diffusion (2022)**, DALL·E 2 — high-quality text-to-image ⇒ this capability gets folded into discrete-token unified models (Chameleon/Emu3) and 2025–26 "any-to-any" systems (e.g. Gemini-based image models).

---

## Part 5 — RAG & Retrieval Lineage

Giving frozen models fresh/private knowledge and grounding (the RAG chapter).

**DrQA (2017)** open-domain QA = retrieve Wikipedia + read ⇒ **DPR (Karpukhin 2020)** learn **dense** bi-encoder retrieval (beats BM25 on semantics) (RAG chapter) ⇒ **REALM (2020)** retrieval-*augmented pretraining* (learn to retrieve end-to-end) ⇒ **RAG (Lewis 2020)** the name and the pattern: retrieve passages, condition generation on them ⇒ **FiD / Fusion-in-Decoder (2021)** encode many passages, fuse in the decoder for multi-doc answers ⇒ **ColBERT (2020) / v2 (2021)** **late interaction** (per-token vectors, max-sim) — between bi- and cross-encoder ⇒ **HyDE (2022)** embed a *hypothetical answer* instead of the question (closer to real answer passages) (the query-transformation material in the RAG chapter) ⇒ **Self-RAG (2023) / CRAG (2024)** the model *decides when to retrieve*, *critiques* passages, and *self-corrects* (the advanced-RAG material) ⇒ **RAPTOR (2024)** recursively cluster+summarize chunks into a tree for multi-level retrieval ⇒ **GraphRAG (Microsoft 2024)** extract an entity/relation **knowledge graph** + community summaries to answer *corpus-wide/global* questions vanilla chunk-retrieval can't (the advanced-RAG material) ⇒ **HippoRAG / HippoRAG 2 (2024–25)** graph + personalized PageRank for cheap multi-hop "memory" ⇒ **Agentic RAG (Search-o1, Search-R1, Graph-R1, 2025)** make retrieval a *reasoned, iterative tool decision*, often **RL-trained** end-to-end (the agentic-RAG material) — converging with the agents track and productized as **Deep Research**.

---

## Part 6 — Agents Lineage

From "answer a question" to "drive a loop and act" (the agents chapter).

**WebGPT (2021)** an LLM that browses to answer ⇒ **ReAct (Yao Oct 2022)** interleave **Thought → Act → Observation**, grounding reasoning in real tool results — the substrate of nearly every agent (the ReAct material in the agents chapter) ⇒ **Toolformer (2023)** the model *self-teaches* which APIs to call and when ⇒ **Reflexion (Mar 2023)** verbal self-critique of a failed attempt, retry with the lesson in context ("verbal RL," no weight updates) (the planning material) ⇒ **OpenAI function calling (Jun 2023)** standardized **structured tool calls** (schema → JSON call → observation) (the tool-use material) ⇒ **AutoGPT / BabyAGI (2023)** autonomous goal-pursuit hype — exposed how brittle long-horizon loops are ⇒ **Generative Agents (Stanford 2023)** memory + **reflection** (consolidate experiences into higher-level insights) (the memory material); **Voyager (2023)** a growing **skill library** (lifelong learning) ⇒ **MetaGPT / ChatDev / AutoGen (2023)** **multi-agent** role teams (planner/coder/critic) (the multi-agent material) ⇒ **Tree-of-Thoughts / LATS (2023)** search-based planning ⇒ **Computer use (Anthropic Oct 2024)** agents that operate a GUI like a human ⇒ **MCP (Anthropic Nov 2024)** an open standard to expose any tool/data uniformly — the "USB-C for tools" (the tool-use material) ⇒ **Coding agents (Devin 2024, OpenAI Codex agent May 2025, Claude Code 2025)** long-horizon software engineering, evaluated on **SWE-bench Verified** ⇒ **RL-trained agents + context engineering (2025–26)** turn-level credit assignment for multi-turn tasks, self-evolving memory (ReasoningBank, Mem0, A-MEM), and **context engineering** (compression/offloading/isolation) as the dominant practical lever (the context-engineering material). The binding constraint throughout, still unsolved: **long-horizon reliability** — per-step success compounds badly over many steps (the reliability material).

---

## Part 7 — The meta-pattern: placing any 2026+ paper on these chains

When a new paper lands, you no longer ask "what is this" — you ask "which chain, and what's the next link":

1. **Which track?** Architecture/training spine (§1), efficiency (§2), reasoning (§3), multimodal (§4), RAG (§5), or agents (§6)? Usually one, occasionally a merge (agentic RAG = §5×§6).
2. **What's the predecessor it's beating, and on what axis?** Every entry above improved a *specific* prior link on a *specific* axis (quality / `n²` cost / KV memory / reasoning / context / reliability / data). Name both.
3. **Which recurring move is it?** Almost everything is one of: *compress something* (KV→MLA, weights→quant, vision→Q-Former), *make a quadratic thing linear* (linear attn/SSM/hybrids), *decouple capacity from compute* (MoE), *change the training signal* (SFT→DPO→RLVR→self-verify), *spend compute at inference* (CoT→search→o1), *add a control loop* (RAG, ReAct, agents), or *unify modalities/interfaces* (T5 text-to-text, Chameleon tokens-for-everything).
4. **What did it trade away?** Find the cost even when hidden — it's the predecessor's strength (e.g. linear attention trades exact retrieval; MoE trades VRAM; reasoning trades latency/cost; agents trade reliability).
5. **Will it survive at a different scale / hardware / base model?** The senior judgment. Most links don't generalize; knowing which do is the skill this whole timeline is meant to build.

If you can run those five on a fresh paper in a couple of minutes, you have what the original request asked for: the ability to read almost anything in this field and understand it as the next move in a game you already know.

---

## Concepts now covered here (that the earlier chapters assumed)

For completeness, the prehistory and tracks above fold in the foundational pieces the rest of the guide built on top of without deriving: **distributed word embeddings (Word2Vec/GloVe), RNNs/LSTMs, seq2seq, the original (Bahdanau) attention, ELMo** (Part 0); the **MoE history** (Shazeer→GShard→Switch, §2); **scaling-law and emergence framing** (entries 7, 11, 12); **instruction tuning and Constitutional AI/RLAIF** (entries 9, 14); the **CNN→ResNet→ViT** vision backbone path and **diffusion/image-generation** thread (§4); and the **process-reward / STaR reasoning lineage** and **tool-use/MCP/computer-use** agent lineage (§3, §6). Combined with the rest of the guide, there should be no load-bearing concept in a modern paper that isn't introduced somewhere in this guide.

---

## You can now

- Trace each parallel track's own chain of fixes: the KV-cache/MoE/sub-quadratic-mixing efficiency lineage (§2), the CoT→search→RL reasoning lineage (§3), the vision-backbone→fusion-spectrum multimodal lineage (§4), the retrieval lineage (§5), and the tool-use/agent-loop lineage (§6).
- Name the recurring architectural axis each track optimizes: §2 fights `O(n²)` and the KV-cache wall, §3 turns "think step by step" into a trained and scaled capability, §4 goes from barely-interacting encoders to one unified model, §5 grounds frozen weights in fresh/private data, §6 turns single answers into acting loops.
- Apply the five-question meta-pattern (§7) to a paper you haven't seen before: which track, what predecessor and axis it beats, which recurring move it makes, what it trades away, and whether it survives at a different scale.
- Explain how the tracks cross-reference each other in practice (e.g. agentic RAG = §5×§6, MoE from §2 feeding the main-spine efficient-flagship template) rather than evolving in isolation.
