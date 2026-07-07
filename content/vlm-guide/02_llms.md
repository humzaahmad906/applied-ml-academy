# Large Language Models — Part 1 of 3: Architecture & Variants

This is the core chapter. It assumes the foundations. Structure: (1) what an LLM *is* as a trained object, (2) the modern architectural variants you'll see in every model card, (3) pretraining + scaling laws + data, (4) post-training — SFT, RLHF, DPO, and the RLVR/GRPO reasoning revolution, (5) reasoning models and test-time compute, (6) decoding, (7) long context. This part (1 of 3) covers what an LLM is and the architectural variants; part 2 covers pretraining and post-training; part 3 covers reasoning, decoding, and long context.

---

## 1. What an LLM actually is

A large language model is a decoder-only transformer (from the foundations) trained to predict the next token, scaled to billions of parameters on trillions of tokens. That's the whole definition. The "intelligence" is an emergent consequence of being very good at next-token prediction over a huge, diverse corpus.

Three properties to hold in mind:
- **It's a probability distribution.** Given a prefix, it outputs `P(next token | all previous tokens)`. Generation = repeatedly sample/pick a token, append, repeat (autoregressive).
- **It's stateless across calls.** No memory between separate generations except what's in the context window. "Memory" in agents and RAG is engineered on top.
- **The weights are frozen at inference.** All "learning" within a conversation is in-context (the model conditions on the prompt), not weight updates. This is why prompting, RAG, and context engineering matter so much.

The lifecycle: **pretraining** (learn language/knowledge from raw text via next-token loss) → **post-training** (SFT + preference/RL to make it follow instructions, be helpful, reason) → **inference** (decoding + serving, covered in the inference chapter).

---

## 2. The decoder-only skeleton (recap) and where models differ

From the foundations: embed → N × [norm → attention → residual; norm → FFN → residual] → final norm → LM head. Every modern model is this, and differs only in:
1. **Attention variant** (how Q/K/V are shaped and cached) — §3.1
2. **FFN variant** (dense vs Mixture-of-Experts) — §3.2
3. **Sequence-mixing variant** (full attention vs linear attention / SSM hybrids) — §3.3
4. Normalization/activation/positional details (the normalization and positional-encoding material from the foundations) — minor knobs

Read any model card or architecture-comparison paper through this lens and it decomposes instantly.

---

## 3. Modern architectural variants (the heart of "reading model cards")

### 3.1 Attention variants — the KV-cache problem

During generation, you cache the Keys and Values of all past tokens (the **KV cache**, detailed in the inference chapter) so you don't recompute them each step. The KV cache is the dominant memory cost at inference and scales with `(layers × heads × head_dim × seq_len × 2)`. Shrinking it is the motivation behind the attention-variant zoo:

- **MHA (Multi-Head Attention):** the original. `h` query heads, `h` key heads, `h` value heads. Best quality, biggest KV cache.
- **MQA (Multi-Query Attention):** `h` query heads but **1** shared K head and **1** shared V head. Slashes KV cache by ~`h×`. Some quality loss. (PaLM, early efficiency work.)
- **GQA (Grouped-Query Attention):** the **standard compromise** today. `h` query heads grouped to share `g` KV heads (e.g. 32 query / 8 KV). Tunable between MHA (`g=h`) and MQA (`g=1`). Llama 2/3, Qwen, Mistral. When you see "32 query heads, 8 KV heads," that's GQA with 4 query heads per KV head.
- **MLA (Multi-head Latent Attention):** DeepSeek's innovation (V2/V3). Instead of caching full K and V, **compress them into a low-rank latent vector** and cache *that*; reconstruct K/V on the fly. Dramatically smaller KV cache than even GQA while keeping near-MHA quality. The decompression is fused with RoPE handling (a "decoupled RoPE" trick because you can't naively rotate a compressed latent). When a paper mentions MLA, the point is "MHA-level quality at a fraction of the KV memory."

Mental model: MHA → MQA → GQA → MLA is a ladder of *KV-cache-vs-quality* tradeoffs. Almost every flagship since 2024 sits at GQA or MLA.

### 3.2 FFN variants — Mixture of Experts (MoE)

Dense models run *every* parameter for *every* token. MoE decouples **total parameters** from **compute per token**.

- Replace the single FFN with **`E` expert FFNs** plus a small **router** (gating network).
- For each token, the router scores the experts and sends the token to only the **top-k** (commonly top-2, sometimes top-8). Only those experts run.
- So a model can have, say, 235B *total* params but activate only ~22B per token (e.g. Qwen3-235B-A22B; "A22B" = 22B active). DeepSeek-V3: 671B total, ~37B active. Kimi K2: similar, more experts.

Why it works: capacity (knowledge storage) scales with total params; cost scales with active params. You get a big model's quality at a smaller model's inference compute.

Key MoE concepts you'll meet:
- **Routing / gating:** the learned function that picks experts. Usually a linear layer + softmax + top-k.
- **Load balancing:** without pressure, the router collapses to a few favorite experts. An **auxiliary load-balancing loss** (or, in DeepSeek-V3, an *auxiliary-loss-free* bias-adjustment scheme) spreads tokens across experts. This is a recurring topic in MoE papers.
- **Shared experts:** a few experts that *every* token always uses (capturing common patterns), alongside the routed ones. DeepSeek, Qwen.
- **Fine-grained experts:** many small experts instead of few large ones — better specialization, but routing cost grows (an active research area).
- **Expert parallelism:** experts live on different GPUs; tokens are routed across the network. A serving/training-systems concern.

Tradeoff: MoE is memory-hungry (must hold *all* experts in VRAM) and trickier to train and serve, but compute-efficient. The dense-vs-MoE split (e.g. Qwen3 ships both) is a deployment-target decision.

### 3.3 Beyond attention — linear attention, SSMs, and hybrids (the 2025–2026 shift)

The `O(n²)` attention cost (from the foundations) drove a search for **sub-quadratic sequence mixers**. This moved from fringe to mainstream in 2025.

- **State Space Models (SSMs) / Mamba:** model the sequence like a continuous linear dynamical system with a *fixed-size hidden state* that gets updated token-by-token. Cost is **O(n)** in time and **O(1)** in state size per step — no growing KV cache. Mamba's key trick is making the state-update *input-dependent* (selective), recovering much of attention's content-routing ability. Great at long sequences, weaker at exact content-based retrieval ("what was that specific token 10k ago").
- **Linear attention:** rewrite `softmax(QKᵀ)V` with a kernel feature map so you can compute it as a running sum (associativity), turning `O(n²)` into `O(n)`. Variants: Lightning Attention (MiniMax-M1), Gated DeltaNet, Kimi Delta Attention. They maintain a small "fast-weight" memory matrix updated via a delta rule, with learned gates controlling how fast old memory decays.
- **The hybrid pattern (now the winning recipe):** *don't* remove attention entirely. Interleave mostly-cheap linear/SSM blocks with a few full-attention blocks (a common ratio is 3:1 — three linear blocks per one attention block). The cheap blocks carry the long-context load with flat memory; the periodic full-attention blocks restore exact retrieval that linear blocks are bad at. **Qwen3-Next / Qwen3.5** (Gated DeltaNet + gated attention) and **Kimi Linear** (Kimi Delta Attention + gated MLA) are the prominent flagships proving this works at scale. This is one of the most important recent architectural trends — when you see "hybrid attention" or "3:1 pattern," this is it.

Why you care: a growing share of new models and long-context papers are about these mixers. The framing to keep: *full attention = exact but quadratic; linear/SSM = cheap but lossy at retrieval; hybrids = have it both ways.*

### 3.4 Multi-token prediction (MTP)

Standard next-token training generates exactly one training target per forward pass. **Multi-token prediction (MTP)** adds extra transformer heads that predict token `t+2`, `t+3`, etc., trained jointly with the main next-token objective — more training signal from the same data.

The **mechanism**: after the main model stream produces a hidden state for position `t`, a lightweight transformer head takes that hidden state (conditioned on the main stream, not independent of it) and predicts the token two steps ahead. In the **cascaded/sequential design** that **DeepSeek-V3 (Dec 2024)** introduced, each additional head is conditioned on the *output* of the previous head, forming a chain: head₁ predicts `t+2`, head₂ takes head₁'s output and predicts `t+3`. This cascaded design is strictly better than the earlier **parallel/independent design** (Meta's research) where each head predicts independently of the others — parallel heads cannot leverage the structure that "knowing `t+2` helps predict `t+3`." DeepSeek-V3's approach has since been adopted by **Qwen3, MiMo, GLM, and LongCat**.

**Why it works:**
- **Denser training signal / data efficiency:** each token participates in predicting multiple future targets instead of one, effectively multiplying the useful gradient signal per token.
- **+12–17% on code generation** benchmarks, because code has especially predictable structure (matching braces, boilerplate, function signatures) that multi-step lookahead captures better than single-step prediction.
- **Loss weighting:** the extra heads typically receive a downweighted loss coefficient (e.g. 0.1–0.3× the main loss) to prevent them from distorting the main language modeling objective.

**The inference bonus — self-speculation:** the MTP head, already trained to predict `t+2`, is a natural **draft model** for speculative decoding (covered in the inference chapter) — you get a free draft model with no extra architecture or training cost. The MTP head proposes ahead; the main model verifies in one parallel pass. Because the MTP head shares the main model's representations (not a separate smaller model), alignment is near-perfect and acceptance rates are high. This "self-speculation" approach collapses draft-model maintenance and matches or exceeds external draft model setups.

Reading takeaway: when a paper says "MTP" or "multi-token prediction heads," it's buying both *training efficiency* and *free speculative decoding at inference* in one mechanism. The cascaded-vs-parallel distinction is the load-bearing design choice.

### 3.5 DeepSeek V4-Pro architecture (Apr 2026)

After DeepSeek-V3 set the open reference design, **DeepSeek V4-Pro** (Apr 2026) is the current benchmark — 1.6T total parameters, 49B active, trained on Huawei Ascend (the first high-performance open model trained outside CUDA). Three structural innovations you won't find in prior work:

**mHC (manifold-constrained hyper-connections).** Hyper-connections modify how a layer's output residual is combined with the stream — essentially a learned, dynamic alternative to the fixed `x + f(x)` residual. The problem: naive hyper-connections cause *residual signal amplification*, where the effective scale of the residual stream grows by a factor of ~3,000× during training, causing instability and gradient pathologies. DeepSeek V4-Pro's fix is **doubly-stochastic projection via Sinkhorn-Knopp normalization**: the hyper-connection matrix is constrained to lie on the Sinkhorn manifold (rows and columns each sum to 1), forcing it to redistribute signal rather than amplify it. This brings amplification from ~3,000× down to ~1.6× — close to the scale-neutral identity residual — making deep MoE training stable without extra loss terms or gradient clipping.

**Engram conditional memory.** Separates two computationally distinct operations that transformer FFNs currently conflate: **static knowledge retrieval** (looking up factual associations baked into weights — an operation whose cost should be O(1), not proportional to sequence length or reasoning depth) from **dynamic reasoning computation** (forming new conclusions by combining retrieved facts, which legitimately scales with reasoning). Engram implements the static lookup via key-value memory banks accessed in O(1) per query, reserving the full transformer depth for dynamic computation. You care because this is a principled answer to "why is the FFN doing two different jobs, and should it?"

**Hybrid CSA+HCA attention.** **Compressed Sparse Attention (CSA)** covers most positions: attends only over a compressed, strided selection of past KV entries rather than all of them. **Head-wise Cross Attention (HCA)** — full-context attention — fires on a small fraction of attention heads per layer where CSA's approximation would cost quality. The result: up to **−73% inference FLOPs** for attention vs full MHA, while retaining the quality of full attention where it matters. Unlike the linear-attention hybrid (§3.3) which trades *architecture families*, CSA+HCA operates within the attention family — familiar training dynamics, no gated-state machinery, just an efficient sparsity pattern.

Mental model: V4-Pro is the "new open reference design" for large MoE — what V3 was in late 2024 but with stabilized deep residuals, a cleaner compute-knowledge separation, and aggressive attention sparsity at scale.

Also now real, not niche: **diffusion language models** (generate tokens by iterative denoising instead of left-to-right). **LLaDA** (2025) reached autoregressive parity at 8B params; **Gemini Diffusion** (Google I/O 2025) shipped in production at ~1,479 tok/s — roughly 5× faster generation than comparable AR models, though still weaker on hard reasoning. **Block diffusion** (BD3-LM) bridges the two: AR over blocks, diffusion within a block. Treat this as a genuine third decoding paradigm you'll see in papers, with the tradeoff *parallel generation speed vs left-to-right reasoning quality*.

---

## You can now

- Explain what an LLM actually is: a decoder-only transformer trained via next-token prediction, and why frozen inference-time weights are exactly why prompting, RAG, and context engineering matter so much.
- Decompose any modern LLM into its architectural axes — attention variant (MHA/MQA/GQA/MLA), FFN variant (dense vs MoE), sequence-mixer variant (full attention vs linear attention/SSM/hybrid) — and read a model card as a set of deliberate tradeoffs.
- Explain the KV-cache pressure that drives the MHA→MQA→GQA→MLA ladder, and why MLA's compressed latent needs a "decoupled RoPE" trick.
- Describe how MoE routing/gating and load balancing work, and why MoE decouples total parameters from compute-per-token.
- Explain multi-token prediction (cascaded vs parallel design) and place DeepSeek V4-Pro's mHC, Engram, and CSA+HCA innovations within the architecture landscape.

## Try it

Take a model you've actually used (e.g. Llama 3, Mistral, DeepSeek-V3, Qwen3) and work out: how many query heads vs KV heads does it use (is that MHA/GQA/MLA)? Is it dense or MoE — and if MoE, what are its total vs active params? Does it use any sub-quadratic sequence mixing (linear attention/SSM/hybrid), or is it full attention throughout? If any of this isn't disclosed in the model card, note that as a finding too — knowing what's hidden is part of reading these cards.

