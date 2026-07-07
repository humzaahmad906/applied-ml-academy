# The Progression, Explained for a Junior ML Engineer — Part 2 of 2: Efficiency, Reasoning, Multimodal, RAG & Agents — Explained

This is part 2 of the companion to the progression timeline. Part 1 covered the vocabulary, the prehistory, and the main architecture/pretraining/alignment spine. Here we slow down and explain the mechanics behind the five parallel tracks — efficiency & inference, reasoning & test-time compute, vision & multimodal, RAG & retrieval, and agents — plus the meta-pattern for reading any new paper, and a cheat sheet tying it all together.

## Table of contents (Part 2 of 2)

- [Part 2 — Efficiency & Inference](#part-2--efficiency--inference)
- [Part 3 — Reasoning & Test-Time Compute](#part-3--reasoning--test-time-compute)
- [Part 4 — Vision & Multimodal](#part-4--vision--multimodal)
- [Part 5 — RAG & Retrieval](#part-5--rag--retrieval)
- [Part 6 — Agents](#part-6--agents)
- [Part 7 — The meta-pattern](#part-7--the-meta-pattern-how-to-read-any-new-paper)
- [Cheat sheet](#cheat-sheet)

---

## Part 2 — Efficiency & Inference

The entire track is one long war on two enemies introduced by the Transformer:
1. **Attention's `O(n²)` cost** — quadratic in sequence length.
2. **The KV-cache memory wall** — explained below.

**First, what is the KV cache?** When generating text autoregressively, at each new token you'd recompute attention over all previous tokens. That's wasteful — the Keys and Values of past tokens don't change. So you **cache** them. The KV cache stores K and V for every token in the context. The problem: it grows linearly with context length *and* batch size, and it lives in precious GPU memory. For long contexts it becomes the dominant memory cost and the main thing limiting how many users you can serve at once. Much of this track is "shrink the KV cache."

### KV-cache / attention shape

- **MHA (Multi-Head Attention, 2017)** — the original. Full quality, but every head stores its own K and V → huge cache.
- **MQA (Multi-Query Attention, 2019)** — all heads *share a single* K and V head. Cache shrinks dramatically. *Cost:* a quality drop.
- **GQA (Grouped-Query Attention, 2023)** — the compromise that won. Group the heads; each *group* shares one K/V. A tunable knob between MHA (best quality) and MQA (smallest cache). Used by Llama 2/3 and most models.
- **MLA (Multi-head Latent Attention, DeepSeek-V2 2024)** — compress K and V into a small **low-rank latent** vector, decompress on the fly. MHA-level quality at a fraction of the cache, using a "decoupled RoPE" trick to keep positional info intact.

### Sparse capacity (Mixture-of-Experts)

The core MoE idea: **decouple the number of parameters from the compute per token.** Instead of one big FFN that every token passes through, have *many* expert FFNs and a **router** that sends each token to only a few. The model "has" huge knowledge (many params) but each token only "uses" a small slice (cheap compute).

- **Outrageously Large Neural Networks (Shazeer 2017)** — MoE on LSTMs; introduced the decouple-params-from-compute idea.
- **GShard (2020)** — MoE for Transformers, plus the sharding to train it across many devices.
- **Switch Transformer (2021)** — simplified routing to **top-1** (send each token to just *one* expert), reached trillion params, easier to train.
- **Mixtral (2023)** — the first big *open* MoE; made the technique mainstream.
- **DeepSeekMoE / V3 (2024)** — fine-grained experts + shared experts + **auxiliary-loss-free** load balancing (keep experts evenly used without a hacky penalty term). The modern standard.

### Sub-quadratic sequence mixing

Goal: replace `O(n²)` attention with something cheaper.
- **Linear/efficient attention & RWKV (2020–22)** — mathematically reorganize attention so it's `O(n)`. Often a quality gap vs full attention.
- **Mamba (2023)** — selective state-space model; strong, but weak at *exact* retrieval.
- **Hybrids (Qwen3-Next/Qwen3.5, Kimi Linear, Nemotron 3, MiniMax-M1 "lightning attention", 2025–26)** — interleave cheap linear/SSM blocks with periodic full-attention blocks. Get both cheap cost and exact retrieval. The current frontier.

### Exact-attention IO efficiency

This thread keeps attention *mathematically exact* but makes it run faster on real hardware by being smart about memory movement.
- The problem: naive attention **materializes** the full `n×n` score matrix in slow GPU memory (HBM). It's *memory-bandwidth-bound* — the GPU spends its time shuffling data, not computing.
- **FlashAttention (Dao 2022)** — never materialize the full matrix. Process it in **tiles** that fit in fast on-chip SRAM, using an **online softmax** (compute softmax incrementally) so you get the *exact* same answer with `O(n)` memory and far less data movement. Huge real-world speedup, zero accuracy loss.
- **FlashAttention-2 (2023)** — better parallelism across GPU warps.
- **FlashAttention-3 (2024)** — exploits Hopper-GPU features (warp specialization + FP8).
- **FlashDecoding / FlashInfer** — variants tuned for the decode phase and sparsity.

### Serving (systems layer)

How you run inference for many users efficiently.
- The problem: storing each request's KV cache in one contiguous memory block causes huge **fragmentation** (wasted gaps).
- **PagedAttention / vLLM (Kwon 2023)** — store the KV cache in fixed-size **pages**, like an operating system's virtual memory. Pages can be shared and allocated on demand. This enables:
  - **Continuous batching** — swap requests in and out of a batch as they finish, keeping the GPU full.
  - **Prefix caching** — if many requests share a prompt prefix (e.g. a system prompt), compute it once and share.
- Plus: **FP8 KV quantization**, **chunked prefill**, **prefill/decode disaggregation** (run the two phases on different hardware tuned for each).

### Compression (fit big models on small hardware)

- **LoRA (2021)** — Low-Rank Adaptation. Freeze the giant pretrained weights; train only tiny *low-rank adapter* matrices added alongside them. You train <1% of the params and get most of the benefit of full fine-tuning, cheaply.
- **QLoRA (2023)** — do LoRA on top of a base model **quantized to 4-bit**. Lets you fine-tune very large models on a *single* GPU.
- **GPTQ (2022) / AWQ (2023)** — **post-training quantization** (PTQ): squeeze trained weights down to 4-bit *after* training. AWQ is "activation-aware" (protects the weights that matter most).
- **SmoothQuant** (8-bit weights *and* activations via outlier migration), **GGUF** (the llama.cpp format for running models on consumer CPUs/laptops), **QAT** (quantization-*aware* training) + layer-sensitivity maps for on-device int4.

*(Quantization = representing weights with fewer bits — fp16 → int8 → int4 — to save memory and speed up inference, trading a little accuracy.)*

### Decode acceleration

- The problem: generating one token at a time *underuses* the GPU — it's built for parallel work but decode is sequential.
- **Speculative decoding (Leviathan 2023)** — a small fast "draft" model guesses the next several tokens; the big model **verifies them all in parallel** in one pass. Accepted guesses are free; rejected ones fall back. Output is **provably identical** to the big model alone — just faster.
- **Medusa (2024)** — add extra prediction "heads" to the model itself so it drafts its own future tokens; no separate draft model.
- **EAGLE / EAGLE-3 (2024–25)** — draft at the *feature* level (more accurate guesses → more accepted). Current state of the art.

---

## Part 3 — Reasoning & Test-Time Compute

How "think before answering" went from a prompt trick → a trained capability → a whole scaling axis.

The chain:
- **Scratchpads (2021)** — let the model write intermediate working in tokens before the answer.
- **Chain-of-Thought prompting (Wei 2022)** — "think step by step" unlocks latent reasoning; emerges at scale.
- **Self-Consistency (2022)** — sample many reasoning chains, take the majority answer. Trades compute for accuracy.
- **STaR (Zelikman 2022)** — *bootstrap* reasoning: generate chains, **keep only the ones that reached the correct answer**, fine-tune on those, repeat. The model learns to reason from its *own* correct traces.
- **Let's Verify Step by Step / PRMs (OpenAI 2023)** — reward *each step* of the reasoning (a **Process Reward Model**), not just the final answer. Catches reasoning that gets the right answer for the wrong reasons. (Process supervision vs outcome supervision.)
- **Tree-of-Thoughts / LATS (2023)** — don't commit to one chain; *search* over a tree of reasoning branches with backtracking.
- **Quiet-STaR (2024)** — learn to generate a silent "thought" between *every* token.
- **o1 (OpenAI 2024)** — RL-trained long CoT + **test-time compute scaling** as a first-class method (but secret).
- **DeepSeek-R1 / GRPO / RLVR (2025)** — open the recipe: **verifiable rewards** + **critic-free group-relative RL**; reasoning *emerges* from pure RL (R1-Zero), then distill into small models.
- **Dr. GRPO / DAPO (2025)** — fix subtle biases in GRPO's normalization (it was unintentionally favoring certain lengths/difficulties); process reward models and turn-level credit assignment mature.
- **DeepSeekMath-V2 (late 2025)** — *self-verifiable* reasoning: the model generates **and checks** its own proofs, extending verifiable-reward RL to domains that have no cheap external verifier.

**The open debate to be aware of:** does RLVR *create* genuinely new reasoning ability, or merely *elicit* what pretraining already buried in the weights? And: when should the model *stop* thinking? (Overthinking measurably hurts on easy problems.)

---

## Part 4 — Vision & Multimodal

How pixels learned to talk to language models. Two feeder threads (vision backbones, image generation) converge into VLMs (Vision-Language Models).

### Vision backbones (how to turn an image into features)

- **AlexNet (2012)** — CNNs (convolutional neural nets) decisively win image classification; the deep-learning vision era begins.
- **ResNet (2015)** — **residual connections** let you train *very* deep networks (the gradient has a shortcut path). Side note worth remembering: residuals are exactly what the Transformer later borrows.
- **ViT (Vision Transformer, 2020)** — drop convolutions entirely: cut the image into fixed patches (e.g. 16×16 pixels), treat each patch as a "token," and run a plain **Transformer**. This *unifies* vision with NLP — same architecture for both.

### Language-aligned encoders (put image features in the same space as words)

- **CLIP (2021) + ALIGN (2021)** — train an image encoder and a text encoder **contrastively** on web-scale image-caption pairs: pull matching image/text vectors together, push mismatched ones apart. Result: image features that *already live in language space* → zero-shot classification (classify an image by comparing it to text label embeddings).
- **SigLIP (2023) / SigLIP 2 (2025)** — swap CLIP's softmax contrastive loss for a **sigmoid** loss (simpler, scales better, multilingual). The default VLM encoder today.
- **DINO / DINOv2 (2021–23)** — **self-supervised** (no captions needed) encoders with strong *spatial/geometric* features — good where CLIP is weak (precise localization, structure).

### Vision → LLM fusion (the main VLM line)

This is the heart of the track — *how* you connect a vision encoder to a language model. The arc goes from "barely interact" to "one unified model":

- **Flamingo (DeepMind 2022)** — keep the LLM **frozen**, inject vision via **cross-attention** layers + a "Perceiver resampler" (compresses many image features into a fixed few). Few-shot multimodal. This is **deep fusion** (vision injected throughout the LLM's layers).
- **BLIP / BLIP-2 (2022–23)** — the **Q-Former**: a small module that compresses the image into a handful of learned "query" tokens the LLM can ingest.
- **LLaVA (2023)** — the dominant *simplification* and the one to know. Just use a tiny **MLP projector** to map vision patches into "visual tokens," then **concatenate** them with the text tokens and feed the whole thing into one decoder LLM. Two-stage training: (1) align the projector, (2) instruction-tune. This is "**visual instruction tuning**." Simple, effective, widely copied.
- **Qwen2-VL / InternVL (2024)** — add **native dynamic resolution** (handle images at their real aspect ratio/size instead of forcing a fixed square), **M-RoPE** (multimodal rotary positions), and token compression — crucial for documents and charts.
- **Early / native fusion: Chameleon (Meta 2024), Emu3 (2024)** — drop the separate vision encoder entirely. **Tokenize images** (often via VQ — vector quantization, turning image patches into discrete codebook tokens) into the *same vocabulary* as text, and train one model on interleaved image+text streams. This also enables **image generation** (the model can output image tokens too).
- **Llama 4 / Gemini / GPT-4o-style (2025–26)** — natively-multimodal frontier models (their exact vision stacks are mostly undisclosed).

**The arc to remember:** *late fusion (CLIP) → cross-attention (Flamingo) → prefix-concat (LLaVA) → early-native (Chameleon)* — going from two barely-connected models to one unified model.

### Image generation (feeds the unified models)

- **DDPM (Ho 2020)** — diffusion: generate an image by starting from noise and **iteratively denoising** it.
- **Latent Diffusion / Stable Diffusion (2022), DALL·E 2** — do the diffusion in a compressed *latent* space → fast, high-quality text-to-image.
- This capability then gets **folded into** discrete-token unified models (Chameleon/Emu3) and the 2025–26 "any-to-any" systems (e.g. Gemini-based image models).

---

## Part 5 — RAG & Retrieval

The problem: a pretrained model's knowledge is **frozen** at training time and it can't see your *private* data. **RAG (Retrieval-Augmented Generation)** fixes this: fetch relevant documents at query time and feed them into the prompt, so the model answers *grounded* in fresh/private facts (and can cite them, reducing hallucination).

The chain:
- **DrQA (2017)** — open-domain QA = *retrieve* Wikipedia passages + *read* them for the answer.
- **DPR (Dense Passage Retrieval, 2020)** — learn a **dense bi-encoder**: embed the question and each passage into vectors, retrieve by nearest-neighbor. Beats keyword search (BM25) on *semantic* match.
- **REALM (2020)** — retrieval-*augmented pretraining*: learn the retriever end-to-end with the model.
- **RAG (Lewis 2020)** — coined the name and the pattern: retrieve passages, condition generation on them.
- **FiD (Fusion-in-Decoder, 2021)** — encode many passages separately, **fuse them in the decoder** to answer from multiple documents.
- **ColBERT (2020) / v2 (2021)** — **late interaction**: keep a vector *per token* and match with "MaxSim." A middle ground between cheap bi-encoders and accurate-but-slow cross-encoders.
- **HyDE (2022)** — embed a *hypothetical answer* (let the LLM imagine one) instead of the question, because a fake answer sits closer to the real answer passages in vector space.
- **Self-RAG (2023) / CRAG (2024)** — the model *decides when* to retrieve, *critiques* the retrieved passages, and *self-corrects* if they're bad.
- **RAPTOR (2024)** — recursively cluster + summarize chunks into a **tree**, so you can retrieve at multiple levels of abstraction.
- **GraphRAG (Microsoft 2024)** — extract an entity/relation **knowledge graph** + community summaries, enabling *corpus-wide / global* questions that plain chunk-retrieval can't answer ("what are the main themes across all these docs?").
- **HippoRAG / HippoRAG 2 (2024–25)** — graph + personalized PageRank for cheap multi-hop "memory."
- **Agentic RAG (Search-o1, Search-R1, Graph-R1, 2025)** — make retrieval a *reasoned, iterative tool decision* (search, read, search again), often **RL-trained** end-to-end. This converges with the agents track and is productized as **Deep Research**.

---

## Part 6 — Agents

The shift from "answer a question" to "drive a loop and *act* in the world."

The chain:
- **WebGPT (2021)** — an LLM that browses the web to answer questions.
- **ReAct (2022)** — interleave **Thought → Act → Observation**: the model reasons, takes an action (calls a tool), observes the result, repeats. This grounds reasoning in *real* tool results and is the substrate of nearly every agent since.
- **Toolformer (2023)** — the model *self-teaches* which APIs to call and when, by inserting API calls into training text and keeping the ones that help.
- **Reflexion (2023)** — after a failed attempt, the model writes a **verbal self-critique** ("I failed because..."), keeps it in context, and retries. "Verbal RL" — improvement with no weight updates.
- **OpenAI function calling (2023)** — standardized **structured tool calls**: define a tool with a JSON schema, the model emits a JSON call, you run it and return the observation.
- **AutoGPT / BabyAGI (2023)** — autonomous goal-pursuit hype. Mostly *exposed* how brittle long-horizon loops are (they wander, loop, and fail).
- **Generative Agents (Stanford 2023)** — memory + **reflection** (consolidate experiences into higher-level insights). **Voyager (2023)** — a growing **skill library** (lifelong learning in Minecraft).
- **MetaGPT / ChatDev / AutoGen (2023)** — **multi-agent** role teams (planner / coder / critic collaborate).
- **Tree-of-Thoughts / LATS (2023)** — search-based planning (shared with the reasoning track).
- **Computer use (Anthropic 2024)** — agents that operate a **GUI** (mouse/keyboard/screenshots) like a human.
- **MCP (Model Context Protocol, Anthropic 2024)** — an open standard to expose any tool or data source uniformly. The "USB-C for tools" — write a tool once, any model can use it.
- **Coding agents (Devin 2024, OpenAI Codex agent 2025, Claude Code 2025)** — long-horizon software engineering, benchmarked on **SWE-bench Verified** (real GitHub issues).
- **RL-trained agents + context engineering (2025–26)** — turn-level credit assignment for multi-turn tasks; self-evolving memory (ReasoningBank, Mem0, A-MEM); and **context engineering** (compressing/offloading/isolating what's in the context window) as the dominant practical lever.

**The binding constraint, still unsolved:** **long-horizon reliability.** Per-step success rates *compound*: if each step is 95% reliable, a 20-step task succeeds only ~36% of the time (0.95²⁰). This is *the* wall agents keep hitting.

---

## Part 7 — The meta-pattern: how to read any new paper

This is the actual *skill* the whole guide builds toward. When a 2026+ paper lands, don't ask "what is this." Ask "which chain, and what's the next link." Run these five questions:

1. **Which track?** Architecture/training spine (§1), efficiency (§2), reasoning (§3), multimodal (§4), RAG (§5), or agents (§6)? Usually one, occasionally a merge (agentic RAG = §5×§6).

2. **What predecessor is it beating, and on what axis?** Every entry above improved a *specific* prior link on a *specific* axis: quality / `n²` cost / KV memory / reasoning / context length / reliability / data efficiency. Name both the predecessor and the axis.

3. **Which recurring move is it?** Almost everything is one of a handful of moves:
   - *compress something* (KV → MLA, weights → quantization, vision → Q-Former)
   - *make a quadratic thing linear* (linear attention / SSM / hybrids)
   - *decouple capacity from compute* (MoE)
   - *change the training signal* (SFT → DPO → RLVR → self-verify)
   - *spend compute at inference* (CoT → search → o1)
   - *add a control loop* (RAG, ReAct, agents)
   - *unify modalities/interfaces* (T5 text-to-text, Chameleon tokens-for-everything)

4. **What did it trade away?** There's always a hidden cost, and it's usually the *predecessor's strength*. Linear attention trades exact retrieval. MoE trades VRAM. Reasoning trades latency/cost. Agents trade reliability. Find the cost even when the paper hides it.

5. **Will it survive at a different scale / hardware / base model?** This is the senior judgment. Most links *don't* generalize — a trick that helps a 1B model may vanish at 100B, or depend on one specific GPU. Knowing which ones survive is the real skill.

If you can run those five on a fresh paper in a couple of minutes, you can read almost anything in the field and understand it as the next move in a game you already know.

---

## Cheat sheet

**The one-line spine:** recurrence → attention (2017) → pretrain (2018) → scale + in-context learning (2020) → Chinchilla correction (2022) → RLHF/DPO alignment (2022–23) → open recipe LLaMA (2023) → efficient MoE/MLA/FP8 (2024) → reasoning via RLVR (2024–25) → hybrids + agentic (2025–26).

**The seven recurring moves:** compress · linearize the quadratic · decouple capacity from compute (MoE) · change the training signal · spend compute at inference · add a control loop · unify modalities.

**The three enemies the field keeps fighting:** attention's `O(n²)` cost · the KV-cache memory wall · long-horizon agent reliability.

**The alignment ladder:** SFT → RLHF (reward model + PPO) → DPO (no RL) → RLVR (verifiable rewards) → self-verifiable reasoning.

**The fusion arc (vision):** late (CLIP) → cross-attention (Flamingo) → prefix-concat (LLaVA) → early-native (Chameleon).

**The numbers worth remembering:** Chinchilla ≈ 20 tokens/param (compute-optimal training); modern models *overtrain* to ~1000+ tokens/param (inference-optimal); hybrid attention ≈ 3:1 linear-to-full blocks.

When in doubt, return to the framing: **every paper is a fix for a specific pain in a specific predecessor, and it creates a new pain that the next paper attacks.** Learn the pains, and the field becomes a story instead of a list.

---

## You can now

- Explain mechanically why the KV cache exists, what MHA→MQA→GQA→MLA trades off, why MoE routers need load-balancing, and why FlashAttention is exact (not an approximation) — the efficiency track's core mechanics.
- Walk the reasoning chain mechanically: what STaR's bootstrap does, what a Process Reward Model adds over an Outcome Reward Model, and what GRPO changes about PPO.
- Explain the vision-fusion arc mechanically — what Flamingo's cross-attention, BLIP-2's Q-Former, and LLaVA's MLP projector each do differently — and how RAG's retrieval chain (DPR → RAG → HyDE → GraphRAG) fixes successive problems.
- Explain why "long-horizon reliability" compounds (e.g. 0.95^20) and trace the agent lineage from ReAct through MCP to context engineering.
- Apply the five-question meta-pattern and the cheat sheet's recurring moves to a paper you haven't seen before, using the mechanics explained here rather than just the names from the timeline.
