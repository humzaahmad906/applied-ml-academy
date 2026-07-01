# Interview Q&A (the full bank)

Built from what's actually asked in 2025–2026 LLM/ML-engineer interviews, plus the deep-dive and "modify the architecture" style that separates strong candidates. Answers are written at senior level — the *why* and the *tradeoff*, not a definition. Each maps back to the relevant chapter of the guide so you can go deeper.

**How interviewers grade (keep this in mind for every answer):** depth (*why* it works, not *what* it does), production experience (real numbers — latency, cost, failure rates you hit), tradeoff awareness (every choice has a cost — name it), and whether you can explain what a framework (LangChain/LlamaIndex) actually hides. The strongest answers cite a real project. Reciting the transformer paper while being unable to estimate the cost of running 1M documents through a model is the classic senior-level fail.

**Sections:** A foundations · B LLM architecture · C training & alignment · D inference & efficiency · E decoding · F RAG · G VLMs · H agents · **I tricky / modify-the-architecture / deep-dive** · J system design · K rapid-fire.

---

## A. Foundations & Transformer mechanics

**Q: Explain self-attention from scratch.**
Each token is projected to query/key/value vectors. The attention score between tokens i and j is the scaled dot product `q_i·k_j/√d_k`; softmax over j turns scores into weights that sum to 1; the output for i is the weighted sum of value vectors. Dot product = similarity, so a token pulls information from the tokens most relevant to it. Multi-head runs this in `h` parallel subspaces and concatenates, so different heads specialize (syntax, coreference, induction). (the foundations)

**Q: Why divide by √d_k? What happens if you don't?** *(extremely common, and a deep-dive favorite)*
Q and K components are ~unit variance, so a dot product over `d_k` dims has variance ~`d_k` — it grows with dimension. Large logits push softmax into a saturated region where one weight ≈1 and the rest ≈0, and the softmax gradient there is near zero. Dividing by √d_k normalizes the variance back to ~1, keeping softmax in a high-gradient regime. Remove it and training becomes unstable / very slow, worse as `d_k` grows. (the foundations)

**Q: Why is attention O(n²), and why does that matter?**
The `QKᵀ` score matrix is `n×n` — compute and memory both scale quadratically with sequence length. It's the single bottleneck behind long-context cost, the KV-cache memory wall, and the entire FlashAttention / linear-attention / SSM research program. (the foundations, the LLM chapter)

**Q: Why subword tokenization instead of words or characters?**
Words → huge vocab + out-of-vocabulary failures. Characters → very long sequences, model wastes capacity learning spelling. Subword (BPE/WordPiece/Unigram) gives frequent words single tokens and splits rare ones into pieces; byte-level BPE has *no* OOV ever (worst case falls back to bytes). Gotchas: numbers/code tokenize poorly (part of why arithmetic is hard), and the same word with/without a leading space is a different token. (the foundations)

**Q: Token embeddings vs sentence/retrieval embeddings?**
Different objects. The embedding *matrix* maps token IDs to per-token vectors *inside* the model. A retrieval embedding is a single vector for a whole span, produced by an embedding *model* (pooled hidden states), trained contrastively. Same word "embedding," different thing. (the foundations, the RAG chapter)

**Q: Why do transformers need positional encoding when RNNs don't?**
Attention is permutation-invariant — scramble the tokens and you get the same result, because there's no recurrence to impose order. RNNs get order for free from sequential processing. So transformers must inject position explicitly. (the foundations)

**Q: Explain RoPE and why it beats learned absolute positions.**
RoPE rotates Q and K by an angle proportional to absolute position (in 2D subspaces, multi-frequency), so that the dot product `q_m·k_n` depends only on the *relative* offset `m−n`. You get absolute-position injection *and* relative-position behavior, with zero learned parameters, and it's KV-cache friendly (cached keys stay valid). It extrapolates better than learned absolute embeddings (which hard-cap at trained length), and context-extension tricks (PI/NTK/YaRN) just rescale its frequencies. (the foundations)

**Q: Pre-norm vs post-norm? RMSNorm vs LayerNorm?**
Pre-norm (normalize the input to each sub-layer) gives a clean residual path and trains stably at 100+ layers; post-norm (original) is fragile at depth and needs careful warmup. RMSNorm drops LayerNorm's mean-centering and bias, just normalizing by RMS + a learned scale — cheaper, no quality loss, now standard. (the foundations)

**Q: Difference between Transformer, BERT, GPT, and "LLM"?**
Transformer = the architecture (self-attention + FFN). BERT = encoder-only, bidirectional, masked-LM, for understanding. GPT = decoder-only, causal, next-token, for generation. LLM = the broad class of large models; BERT and GPT are both instances. The encoder/decoder difference is fundamentally about the *attention mask* (bidirectional vs causal), not a structural rewrite. (the foundations, the progression timeline)

---

## B. LLM architecture & variants

**Q: Why are virtually all modern LLMs decoder-only?**
Generation is the primary use case, and the causal next-token objective is a dense, self-supervised signal over *every* token of raw text — maximally data-efficient. Decoder-only also scales cleanly and unifies pretraining and generation. Encoder-decoder still wins for some seq-to-seq (translation) and grounding tasks, but the field consolidated on decoder-only for general LLMs. (the foundations, the progression timeline)

**Q: MHA vs MQA vs GQA vs MLA — explain the ladder.**
All trade KV-cache size against quality. MHA: `h` K/V heads, best quality, biggest cache. MQA: 1 shared K/V head, ~`h×` smaller cache, some quality loss. GQA: `g` KV-head groups (e.g. 32 query / 8 KV) — the tunable standard. MLA (DeepSeek): compress K/V to a low-rank latent and cache *that*, reconstructing on the fly — MHA-level quality at a fraction of the cache, using a decoupled-RoPE trick. (the LLM chapter)

**Q: What is MoE and what problem does it solve?**
It decouples total parameters (capacity) from compute-per-token. Replace the FFN with `E` experts + a router; each token goes to top-k experts (often 2), so a 235B-total model might activate only 22B/token. You get big-model quality at small-model inference compute. Costs: must hold *all* experts in VRAM (memory-hungry), trickier training (load balancing), trickier serving (expert parallelism). (the LLM chapter)

**Q: How do MoE routers avoid collapsing to a few experts?**
Without pressure the router favors a handful of experts. Fix with an auxiliary load-balancing loss (encourage uniform expert usage) or, like DeepSeek-V3, an auxiliary-loss-free bias-adjustment scheme that nudges routing toward balance without a competing loss term. (the LLM chapter)

**Q: What are SSMs / linear attention, and why the hybrid trend in 2025–26?**
They're sub-quadratic sequence mixers with a fixed-size recurrent state (no growing KV cache) — O(n) time, O(1) state. Great at long sequences, weak at *exact* content-based retrieval. The winning recipe (Qwen3-Next/Qwen3.5, Kimi Linear, Nemotron 3) interleaves mostly-cheap linear/SSM blocks with periodic full-attention blocks (~3:1): linear carries the long-context load with flat memory, attention restores exact retrieval. (the LLM chapter, the progression timeline)

**Q: What did Chinchilla change, and is it still right?**
It showed most pre-2022 models were too big and undertrained; compute-optimal is ~20 tokens/param. But it optimizes *training* compute only. Since you serve a model millions of times, it's often rational to "overtrain" a smaller model far past Chinchilla (Llama-3-8B ≈ 1800 tokens/param) for cheap inference. So: right framework, but inference-aware practice deliberately departs from it. (the LLM chapter)

---

## C. Training & alignment

**Q: Walk through the full LLM lifecycle.**
Pretraining (next-token loss on trillions of tokens → a base model that continues text but doesn't follow instructions) → SFT (instruction-response pairs → assistant format) → preference optimization (RLHF or DPO → align to human preference) → optionally RLVR/GRPO (verifiable rewards → reasoning). Then inference: decoding + serving. (the LLM chapter, the progression timeline)

**Q: Explain LoRA and the key insight behind why it works.**
Freeze the base weights `W`; learn a low-rank update `ΔW = B·A` (rank r, e.g. 8–64) added to chosen layers, training <1% of params. The insight: the *adaptation* needed for a downstream task has low intrinsic rank, so a low-rank delta suffices. Benefits: tiny swappable checkpoints (tens of MB), and you can merge `ΔW` into `W` at inference for zero added latency, or keep adapters hot-swappable for multi-tenant serving. (the LLM chapter)

**Q: LoRA vs QLoRA — when each?**
QLoRA does LoRA on top of a 4-bit (NF4) quantized frozen base with double-quantized scales and paged optimizers — lets you fine-tune a 65B model on one 48GB GPU, accuracy ~within noise of full FT. Use QLoRA when memory-constrained; use plain LoRA when you have headroom, want max training speed, or need to merge adapters cleanly (quantization complicates clean merges). (the LLM chapter, the inference chapter)

**Q: RLHF vs DPO — what's the difference and why did DPO take over?**
RLHF: train a reward model on human preference pairs, then RL-optimize (PPO) the policy against it with a KL leash — four models in memory, unstable, expensive. DPO mathematically reparameterizes the same objective into a simple classification loss on (prompt, chosen, rejected) triples — no reward model, no RL loop, no sampling. Far simpler and stable, so it became the default. (the LLM chapter)

**Q: What is RLVR / GRPO and why was DeepSeek-R1 a big deal?**
RLVR uses *verifiable* rewards (check if math/code answer is correct) instead of a learned reward model — no reward-hacking of an RM, cheap, binary signal. GRPO is PPO without a critic: sample a group of G responses per prompt, use the group mean as the baseline (advantage = reward − group mean). R1-Zero showed that GRPO+RLVR on a base model *with no SFT* spontaneously produced long chain-of-thought, self-verification, and backtracking — reasoning emerged from pure RL. (the LLM chapter, the progression timeline)

**Q: What is catastrophic forgetting and how do you mitigate it?**
Fine-tuning on new data overwrites prior capabilities. Mitigations: PEFT/LoRA (freeze the base, so original knowledge is preserved), lower learning rates, mixing in replay/original data, and regularization toward the base weights. PEFT is the practical default because frozen base weights can't be forgotten. (the LLM chapter)

**Q: How do you reduce hallucination?**
No single fix — layer them: ground in retrieval (RAG) with citations, preference-train against unsupported claims (DPO/RLVR/RLHF), prompt for "say I don't know," constrain decoding, and verify with a second pass or a verifier. For VLMs, also raise resolution and add grounding data. Distinguish *intrinsic* (contradicts the source) from *extrinsic* (unsupported) hallucination. (the LLM chapter, the RAG chapter, the VLM chapter)

**Q: How do you evaluate a fine-tuned model beyond loss/perplexity?**
Task metrics on a held-out set, but also: human eval / Arena-style pairwise preference, LLM-as-judge (with its biases), capability benchmarks (contamination-resistant ones), and *production* signals — latency, cost, factual accuracy, regression on previously-working cases. Perplexity correlates weakly with usefulness. (the reading-papers chapter)

---

## D. Inference & efficiency

**Q: Explain the two phases of inference and their bottlenecks.** *(senior staple)*
Prefill: process the whole prompt in one parallel pass, compute K/V for all tokens — compute-bound, determines TTFT. Decode: generate one token at a time, each step reading the entire KV cache + weights — memory-bandwidth-bound (GPU compute mostly idle), determines TPOT. This asymmetry explains nearly every optimization: decode tricks attack bandwidth/cache; prefill tricks attack compute and `n²`. (the inference chapter)

**Q: What is the KV cache and why does it dominate memory?**
During decode you cache past tokens' keys/values to avoid recomputation. It grows with `2 × layers × kv_heads × head_dim × seq_len × batch` — for long contexts/large batches it dwarfs the weights and is the binding memory and bandwidth constraint. Shrink it via GQA/MLA, quantize it (FP8/INT4), evict it (attention sinks/StreamingLLM), or manage it (PagedAttention). (the inference chapter)

**Q: How does FlashAttention speed things up — is it an approximation?**
No, it's *exact*. Standard attention materializes the full `n×n` matrix in slow HBM (memory-bound, O(n²) memory). FlashAttention tiles Q/K/V into SRAM-sized blocks and uses online softmax (running max + normalizer) to combine blocks without ever storing the full matrix → O(n) memory, far less HBM traffic, 2–4× faster, same result. (the inference chapter)

**Q: What does PagedAttention/vLLM solve?**
Naive serving reserves one contiguous KV buffer per sequence at max length → massive fragmentation/waste. PagedAttention pages the KV cache into fixed blocks with a per-sequence block table (like OS virtual memory) → near-zero waste, shareable blocks (prefix caching, parallel samples), and it enables continuous batching. ~14–24× throughput over naive generation. (the inference chapter)

**Q: Quantization — explain PTQ vs QAT and the outlier problem.**
PTQ quantizes a trained model with little/no retraining (GPTQ uses Hessian info; AWQ protects salient channels; both hit 4-bit weights). QAT simulates quantization during training (fake-quant forward, straight-through gradients) so the model learns robust weights — more expensive, recovers more at very low bits. The universal enemy is *outliers*: a few large-magnitude activations that wreck naive quantization; SmoothQuant migrates them into weights, AWQ scales around them. Notation W4A16 = 4-bit weights, 16-bit activations. (the inference chapter)

**Q: How does speculative decoding work, and does it change outputs?**
A cheap draft model proposes k tokens; the big target verifies all k in one parallel forward pass (verification is compute, which is idle during decode); accept the longest correct prefix. Output distribution is *provably identical* to the target's — pure speedup (~2–3×). Helps most at small batch sizes (bandwidth-bound, compute idle); gains shrink at huge batches where the GPU is already saturated. Variants: Medusa (extra heads), EAGLE (feature-level). (the inference chapter)

**Q: Explain the throughput–latency tradeoff.**
Bigger batches amortize weight loads and saturate the GPU → higher aggregate throughput (tokens/sec) but worse per-request latency (each request waits for the batch). Latency-sensitive serving uses small batches + speculative decoding; throughput-sensitive uses large continuous batches. You tune to your SLO (TTFT/TPOT/P95). (the inference chapter)

---

## E. Decoding & generation

**Q: How does temperature affect output?**
It scales logits before softmax. T<1 sharpens (more deterministic/confident), T>1 flattens (more random/creative), T→0 ≈ greedy/argmax. It changes the *shape* of the distribution, not the ranking. (the LLM chapter)

**Q: Top-k vs top-p (nucleus) vs min-p?**
Top-k samples from the k most likely tokens (fixed count). Top-p samples from the smallest set whose cumulative probability ≥ p (adapts to how peaked the distribution is). min-p keeps tokens above a fraction of the top token's probability — a newer adaptive cutoff that often behaves better at high temperature. (the LLM chapter)

**Q: Why does greedy decoding produce repetitive/degenerate text?** *(deep-dive)*
Maximizing likelihood at each step favors high-frequency, "safe" continuations and can lock into self-reinforcing loops (the model assigns high probability to repeating what it just said). Fixes: sampling (top-p/temperature) to inject diversity, repetition/frequency/presence penalties, and no-repeat-ngram constraints. Open-ended generation wants sampling; low-entropy tasks (translation) tolerate greedy/beam. (the LLM chapter)

**Q: How do you force valid JSON / a schema?**
Constrained (structured) decoding: at each step, mask logits to only the tokens allowed by a grammar/regex/finite-state machine, so the output is *guaranteed* valid. This is how reliable tool-calling and structured extraction work, independent of how well the model "wants" to comply. (the LLM chapter, the agents chapter)

---

## F. RAG

**Q: Walk through a RAG pipeline and where each step fails.**
Index: load → chunk → embed → store in a vector index. Query: embed query → retrieve top-k → (rerank) → stuff into prompt → generate. Failure points: chunking (fragmented/diluted context), embeddings (semantic mismatch), index freshness (stale data), retrieval depth (too few/many), grounding (model ignores context). (the RAG chapter)

**Q: Bi-encoder vs cross-encoder — when each?** *(very common)*
Bi-encoder embeds query and doc separately and compares vectors — fast, scalable (docs pre-embedded), used for *retrieval* over millions. Cross-encoder feeds query+doc together so attention runs between them — far more accurate but O(candidates) forward passes, too slow for the full corpus, so used for *reranking* the top ~20–100. Pattern: retrieve broad with bi-encoder, rerank narrow with cross-encoder. (the RAG chapter)

**Q: Why hybrid search?**
Dense (semantic) retrieval captures meaning/synonyms but misses exact terms (rare names, IDs, codes). Sparse (BM25) nails exact terms but misses paraphrases. Run both and fuse rankings (Reciprocal Rank Fusion) for semantic recall *and* lexical precision — usually beats either alone. (the RAG chapter)

**Q: Explain chunking strategies and the core tradeoff.**
Fixed-size+overlap (baseline), recursive/structural (split on paragraphs/headings), semantic (split at topic shifts via embedding-similarity drop), parent-child / small-to-big (retrieve precise small chunks, feed the larger parent), late chunking (embed the whole doc first so each chunk's vector has full context, then chunk). Core tension: small chunks = precise retrieval but fragmented context; big chunks = rich context but diluted/noisier embeddings. Parent-child and late chunking exist to escape it. Chunking is the highest-leverage and most-underrated knob. (the RAG chapter)

**Q: What is an ANN index (HNSW/IVF/PQ) and why not exact search?**
Brute-force over millions of vectors is too slow. ANN trades a little recall for huge speed: HNSW (layered proximity graph, dominant in-memory), IVF (cluster + search nearest clusters), PQ (compress vectors), IVF-PQ/DiskANN for billion-scale. You're choosing a point on the recall–latency–memory triangle. (the RAG chapter)

**Q: My RAG gives wrong/outdated answers — how do you debug?** *(scenario, very common)*
First separate retrieval vs generation failure. Inspect what was retrieved for the query: if irrelevant → embeddings/chunking/index-freshness problem (check ingestion, re-embed, fix metadata filters). If the right chunk *was* retrieved but the answer is still wrong → grounding problem (tighten the prompt, add citations, add a reranker, check the context isn't lost-in-the-middle). Fix root cause (reindex/re-embed/adjust chunking) and add eval checks so it doesn't recur. (the RAG chapter)

**Q: When advanced RAG (GraphRAG/RAPTOR) vs vanilla?**
Match the question type. Vanilla flat retrieval = "find this fact." RAPTOR (cluster+summarize tree) or GraphRAG (entity/relation graph + community summaries) = corpus-wide/"global" synthesis questions vanilla can't answer ("what are the main themes"). Iterative/agentic RAG = multi-hop reasoning. Graph/tree methods pay heavy indexing cost (many LLM calls) — justify it against corpus scale. (the RAG chapter)

**Q: RAG vs fine-tuning vs long-context?**
RAG: dynamic/changing knowledge, provenance/citations, large corpora. Fine-tuning: teach behavior/format/style or internalize a stable domain — *not* for fast-changing facts. Long-context: the relevant info fits and you want joint reasoning over all of it. They compose (retrieve into a long context; fine-tune the retriever or grounding behavior). (the RAG chapter)

**Q: How do you evaluate RAG?**
Separately. Retrieval: Recall@k, MRR, nDCG. Generation: faithfulness/groundedness (is the answer supported by context — i.e., no hallucination), answer relevance, context precision/recall (RAGAS, LLM-as-judge). Always diagnose retrieval-failure vs generation-failure separately. (the RAG chapter)

---

## G. VLMs / multimodal

**Q: Walk through a VLM's architecture.**
Vision encoder (pixels → visual features) → projector/connector (map into the LLM's embedding space → visual tokens) → LLM (decoder attends over text + visual tokens) → text out. Most modern open VLMs are LLaVA-style: frozen-ish encoder + small trained MLP projector + pretrained LLM. (the VLM chapter)

**Q: How does ViT turn an image into tokens?**
Split into fixed patches (e.g. 16×16), flatten+linearly-project each patch to a vector (patches are the "tokens"), add 2D positional embeddings + a CLS token, run through transformer layers. A 224² image at patch-16 → 196 tokens. Patch count drives both spatial resolution and token cost. (the VLM chapter)

**Q: CLIP vs SigLIP vs DINO — why does the encoder choice matter?**
CLIP/SigLIP are contrastively trained on image-text pairs, so their features are *already language-aligned* — the default for VLMs (SigLIP's sigmoid loss scales better; SigLIP 2 is multilingual + dense). DINO is self-supervised (no text), strong at spatial/geometric structure where CLIP is weak. The pretraining objective tells you what the encoder is good at; document/OCR VLMs care about resolution + spatial fidelity, so encoder choice matters more there. (the VLM chapter)

**Q: Explain the VLM fusion spectrum.**
Late fusion (CLIP): separate encoders, interact only via similarity — great for retrieval, not generative. Cross-attention fusion (Flamingo): text attends to vision via inserted cross-attn layers, LLM weights mostly intact. Prefix-concat (LLaVA, dominant): project vision to tokens, concatenate with text, one decoder's self-attention does the fusion. Early/native (Chameleon, Emu3, Llama 4): tokenize images into the same vocabulary, train one model from scratch on interleaved streams — can generate images too. Spectrum = barely-interact → one-unified-model. (the VLM chapter)

**Q: Projector choices — MLP vs Q-Former vs pixel-shuffle, and the core tradeoff?**
MLP (LLaVA): keep every patch as a token, simple, strong — but token count = patch count. Q-Former (BLIP-2): learned query tokens cross-attend to extract a *fixed small* number of tokens (e.g. 32), aggressive compression, complex to train. Pixel-shuffle/pooling: merge 2×2 patches → 4× fewer tokens. Core tradeoff the projector mediates: number of visual tokens (cost, since the LLM is O(n²)) vs information/detail preserved (OCR, spatial precision). (the VLM chapter)

**Q: Continuous vs discrete visual tokens?**
Continuous: real-valued vectors from encoder+projector, fed as soft embeddings — can't be *generated* by a discrete LM head. Discrete (VQ-VAE/VQGAN): quantize patches into codebook indices added to the vocabulary, so image and text share one discrete space and the model can *generate* images by predicting image tokens (Chameleon, Emu3). (the VLM chapter)

**Q: How is a LLaVA-style VLM trained?**
Stage 1 (alignment): freeze encoder + LLM, train only the projector on image-caption data. Stage 2 (instruction tuning): unfreeze the LLM (sometimes encoder, gradually), train on multimodal instruction data (VQA, OCR, charts, grounding). Stage 3 (optional): DPO/RLHF/RLVR for preferences and to reduce hallucination. (the VLM chapter)

**Q: Why do VLMs hallucinate objects, and how do you reduce it?**
Strong language priors override weak visual grounding — the model "expects" objects that co-occur in training text. Reduce with better grounding data, DPO against hallucinated descriptions, and higher resolution (small objects/text get lost at low res). Measured by POPE/CHAIR. (the VLM chapter)

---

## H. Agents

**Q: What makes something an "agent" vs a chatbot?**
The model drives a *loop*: it can take actions (tools), observe results, and decide what to do next toward a goal — rather than emitting one answer and stopping. The LLM is the policy/brain; the loop + tools are scaffolding. Four capabilities: reasoning/planning, tool use, memory, coordination. (the agents chapter)

**Q: Explain ReAct.**
Interleave Thought → Action → Observation: the model reasons about what to do, emits a tool call, reads the result, repeats. Grounding reasoning in real observations lets it adapt to errors and changing state instead of hallucinating a plan. It's the substrate of most agents. (the agents chapter)

**Q: How does function calling / tool use actually work?**
The model is given tool schemas (name, description, JSON params). To act, it emits a structured call (name + JSON args), the runtime executes it, the result returns as an observation. The model is post-trained to produce these, and constrained decoding guarantees valid JSON. It picks tools by their *descriptions*, so description quality and good error messages (for recovery) matter a lot; too many tools → tool-retrieval (RAG over the tool catalog). (the agents chapter)

**Q: What is MCP?**
Model Context Protocol — an open standard for exposing tools/data/resources to models uniformly, so any MCP client can use any MCP server. "USB-C for tools": decouples tool providers from agent builders. (the agents chapter)

**Q: Planning without feedback vs with feedback?**
Without feedback (plan-then-execute): generate a full plan upfront — CoT, decomposition, Tree-of-Thoughts. Efficient but brittle if reality diverges. With feedback (interleaved): plan, act, observe, replan — ReAct, Reflexion (verbal self-critique + retry), LATS (tree search + acting). Robust but more tokens/latency. (the agents chapter)

**Q: How does agent memory work, and how is it different from RAG?**
Short-term = the context window (current task). Long-term = persisted across sessions, retrieved when relevant — vector memory (RAG over past interactions), structured/graph memory, or episodic/semantic/procedural splits, often with reflection (summarize experiences into insights). Difference: RAG retrieves from an external *knowledge* corpus; agent memory retrieves from the agent's own *experience*. Same machinery, different content/lifecycle. (the agents chapter)

**Q: When is multi-agent worth it, and when not?**
Worth it when the task genuinely decomposes into parallelizable or cleanly-separable sub-tasks, when role specialization helps, or for context isolation (each subagent gets a clean window). Not worth it for tasks a single well-engineered agent handles — multi-agent adds coordination overhead, error propagation, token cost (often many×), and harder debugging. (the agents chapter)

**Q: What is context engineering and why did it replace prompt engineering?**
The context window is finite *and* models degrade as it fills (lost-in-the-middle, distraction, cost). Context engineering curates exactly what's in the window each step: compression/summarization of old turns, selective retrieval of relevant tools/memories/files, offloading big artifacts outside context (files/handles), structured + cache-friendly prefixes, subagent isolation. A lot of real agent performance comes from this, not a smarter model. (the agents chapter)

**Q: What's the central unsolved problem in agents?**
Long-horizon reliability. Per-step errors compound: 95% per-step success → ~60% over 10 steps. Capability isn't the binding constraint for real deployments; reliability over long trajectories is. Watch for compounding errors, looping, context rot, and recovery from tool failures. (the agents chapter)

---

## I. Tricky / "modify the architecture" / deep-dive

These are the hardest ones — where they hand you a constraint and watch you reason, or push on *why*. Frameworks, not memorized answers.

**Q: Your KV cache OOMs at 100k-token context. What do you change, in what order?** *(classic design-under-constraint)*
Reason from the bottleneck (KV memory). In rough order of cost/effort: (1) switch to GQA/MQA if on MHA — biggest cache cut for least work; (2) quantize the KV cache to FP8/INT4 (KIVI/KVQuant) — 2–4× with a quality check; (3) eviction/sinks — StreamingLLM (keep first "sink" tokens + recent window) or DuoAttention (full cache only for retrieval heads); (4) PagedAttention so you're not wasting on fragmentation; (5) chunked prefill so the long prompt doesn't stall others; (6) architecturally, move to MLA or a linear-attention hybrid (flat memory); (7) step back — does this even need 100k in-context, or should it be RAG? Name the quality tradeoff for each lossy option. (the inference chapter, the LLM chapter)

**Q: Make attention linear. What do you lose, and how do you get it back?** *(deep-dive)*
Linear attention (kernel feature map + running sum) drops `n²`→`n` and removes the growing KV cache, but you lose *exact* content-based retrieval — the softmax attention matrix is what lets a token sharply attend to one specific far-away token; a fixed-size recurrent state blurs that. You get it back with a hybrid: keep a few full-attention layers (the ~3:1 pattern) so exact retrieval is preserved where it matters while linear blocks carry the bulk cheaply. (the LLM chapter)

**Q: Design a module to add a new prior (say, barcode/region awareness) into a transformer's attention. How?** *(architecture-modification, VLM/document flavor)*
Don't retrain from scratch — inject a *bias*. Add a learned additive term to the attention logits before softmax (`QKᵀ/√d + B`), where `B` encodes the prior (e.g. higher bias toward known barcode regions, or a 2D-distance bias for spatial structure). It's parameter-light, differentiable, composes with existing attention, and you can ablate it cleanly. Alternatives: a dedicated cross-attention to a region-feature stream, or a doc-type conditioning token prepended to the sequence. Then justify: minimal params, preserves pretrained weights, easy to A/B. (the foundations, the VLM chapter)

**Q: Your MoE router collapses to 3 experts. Diagnose and fix.**
Diagnosis: no balancing pressure, so the router exploits early-favored experts (rich-get-richer). Fixes: add an auxiliary load-balancing loss (penalize deviation from uniform expert load), or DeepSeek-V3's auxiliary-loss-free bias adjustment (add a per-expert routing bias nudged to balance, no competing loss); also check capacity factors (token dropping) and router learning rate / noise (noisy top-k routing helps exploration). Verify with per-expert utilization histograms. (the LLM chapter)

**Q: Extend a model trained at 4k context to 128k without full retraining. How, and what breaks?** *(deep-dive on RoPE)*
RoPE high-frequency rotations alias/wrap at distances unseen in training, so naive extension degrades. Options: Position Interpolation (squeeze positions into the trained rotation range — simple, some resolution loss), NTK-aware scaling (change the rotation base, preserves high-freq resolution), or YaRN (refined NTK+interpolation, the common "extended to 128k" method), usually with a short continued-pretraining/fine-tune at long context. What breaks even after it *runs*: "lost in the middle" — running at 128k ≠ using 128k well; verify with needle-in-a-haystack/RULER, not just that it doesn't crash. (the foundations, the LLM chapter)

**Q: Reduce a VLM's visual tokens 4× with minimal quality loss. How?**
Pixel-shuffle/spatial pooling (merge 2×2 patches → 1 token) before the projector, or a Q-Former/resampler to a fixed small token count, or token pruning/merging (drop low-information patches). The tradeoff is detail — fine for natural-image captioning, risky for OCR/charts where small text lives in those tokens. So: choose compression by *task*, keep higher token budgets for document understanding, and measure on DocVQA/ChartQA not just captioning. (the VLM chapter)

**Q: Your RLHF-trained model started reward-hacking (verbose, sycophantic, gaming the RM). Diagnose.**
The policy found high-reward regions the reward model scores well but humans don't like — the RM is a proxy and the policy is over-optimizing it (Goodhart). Fixes: stronger KL penalty to the reference model (limit drift), better/retrained RM with adversarial examples, length normalization (penalize verbosity the RM rewards), or switch the signal — DPO (no separate RM to hack) or RLVR (verifiable reward can't be flattered). This is exactly why verifiable rewards became attractive for math/code. (the LLM chapter)

**Q: Why does removing the FFN hurt more than you'd think? What does each sub-layer do?** *(deep-dive)*
Attention *moves* information between positions (mixing across the sequence); the FFN *processes* information within each position (mixing across features) and holds the bulk of the parameters — it's where much factual "knowledge" is stored (key-value memory view). Remove/shrink it and you keep routing but lose per-token computation and storage capacity, so quality drops sharply even though attention is "the famous part." (the foundations)

**Q: A reasoning model "overthinks" simple questions — wastes tokens, sometimes worse answers. What do you do?**
Recognize test-time compute isn't free or monotonic. Options: train/prompt it to allocate thinking by difficulty (budget-aware), add a stop-thinking signal / length penalty in RL, route easy queries to a non-thinking mode (hybrid thinking models like Qwen3 do this), or cap reasoning tokens. The deeper point: more thinking helps on hard reasoning, hurts on simple/factual/latency-sensitive tasks — match the tool to the task. (the LLM chapter)

**Q: You see attention "sink" tokens (huge attention on the first token / BOS). Bug or feature?**
Feature, mostly. Models learn to dump excess attention onto a few initial tokens as a no-op when no real token is relevant (a "register"). It's why StreamingLLM keeps the first few tokens when sliding the window (drop them and quality collapses), and why KV-quantization schemes specially preserve sink tokens. Not a bug to fix — a behavior to accommodate. (the inference chapter)

**Q: Estimate the cost/latency of running 1M documents through a model.** *(the senior gut-check)*
Show the back-of-envelope: tokens per doc × 1M = total input tokens; prefill is compute-bound so estimate via model FLOPs/token and GPU throughput (or just $/1M-token API pricing × volume); if generating, add output tokens × per-token decode latency (bandwidth-bound). Then mention levers: batching/throughput mode, prefix caching if docs share a prompt, a smaller/quantized model, or whether you even need the full model (route easy docs to a cheaper one). The point is you can reason about cost, not the exact number. (the inference chapter)

**Q: Why not just use a bigger context window instead of RAG?** *(trap)*
Because (1) `n²` cost — huge contexts are expensive in compute and KV memory; (2) lost-in-the-middle — models don't use the middle of long contexts well, so stuffing everything in *lowers* effective accuracy; (3) freshness/provenance — RAG gives citations and updates without retraining; (4) you often have far more corpus than any window. The honest answer is "both": retrieve to fill a large-but-finite context *well*. (the LLM chapter, the RAG chapter)

---

## J. ML system design (frameworks for the open-ended ones)

For any "design an X" prompt, structure the answer: **requirements (latency/cost/scale/quality SLOs) → data → architecture → training/eval → serving/inference → monitoring → failure modes & tradeoffs.** Always state assumptions and name tradeoffs; interviewers grade the reasoning, not a "right" diagram.

**Q: Design a production RAG system for internal company docs.**
Requirements: corpus size, freshness, latency, citations, access control. Pipeline: ingestion (parse, structure-aware chunk, embed with a domain-fit model, store in a vector DB with metadata) → query (rewrite, hybrid dense+BM25 retrieve top-50, cross-encoder rerank top-5, build prompt with citations, generate) → guardrails (faithfulness check, "I don't know" fallback). Ops: metadata filtering for permissions, document versioning, periodic re-indexing, prefix caching for the system prompt, eval harness (retrieval recall + faithfulness), monitoring for stale/irrelevant answers. Tradeoffs: chunking strategy, top-k vs latency, rerank cost. (the RAG chapter)

**Q: Design an agent that resolves customer support tickets end-to-end.**
Requirements: success rate, escalation policy, latency, safety. Loop: ReAct over tools (KB search/RAG, account lookup, ticket update, refund API) with structured function calls; planning for multi-step tickets; memory of the customer's history; human-in-the-loop escalation on low confidence or high-risk actions. Reliability: step budgets, loop detection, action confirmation for irreversible operations, trajectory logging. Eval: task-completion on a held-out ticket set + trajectory quality, not just final-answer. Tradeoffs: autonomy vs safety, single vs multi-agent, context engineering for long tickets. (the agents chapter)

**Q: Design on-device deployment of a VLM for document capture (phone).**
Requirements: memory/thermal/battery limits, offline, latency. Choices: small VLM (sub-4B), aggressive quantization (QAT-derived int4/int8 with a layer-sensitivity map), runtime per platform (MLX/CoreML on iOS, MNN/LiteRT on Android), high-enough resolution for OCR with token compression to fit memory, GQA/sliding-window to shrink KV. Gotchas: immature GPU kernels for newer ops (CPU fallback), tokenizer mismatch on conversion (silent garbage), multimodal-RoPE export bugs. Validate transfer on *real* captures, since synthetic document data lacks real geometric/spatial distortions. (the inference chapter, the VLM chapter)

---

## K. Rapid-fire (crisp answers for the quick ones)

- **Q: Why residual connections?** Preserve gradient flow and give each layer a clean "read/edit the residual stream" path; enable very deep stacks. (the foundations)
- **Q: What's perplexity?** `exp(cross-entropy loss)` — average branching factor; lower = more confident/accurate next-token prediction. (the LLM chapter)
- **Q: Zero-shot vs few-shot vs in-context learning?** Zero/few-shot = task from instruction alone / with a few in-prompt examples; in-context learning = conditioning on the prompt to do a task *without weight updates*. (the LLM chapter)
- **Q: Why CoT works?** It lets the model externalize intermediate computation into tokens it can condition on, turning a one-shot guess into a multi-step computation; effective mainly at scale. (the LLM chapter, the progression timeline)
- **Q: Greedy vs beam search — when beam?** Beam keeps b candidate sequences; good for low-entropy tasks (translation), bad for open-ended chat (bland/repetitive). (the LLM chapter)
- **Q: What does the LM head do, and weight tying?** Projects the final hidden state to vocab logits; tying shares it with the input embedding matrix (saves params, often helps) — though some large models decouple them for big tokenizers. (the foundations)
- **Q: Embedding dimensionality tradeoff (incl. Matryoshka)?** Higher dim = more expressive but costlier storage/search; Matryoshka embeddings let you truncate to a shorter prefix with graceful degradation. (the RAG chapter)
- **Q: Sliding-window attention?** Each token attends only to a fixed recent window (O(n) not O(n²)); cheap long context, with stacked layers giving an effectively larger receptive field (Mistral). (the LLM chapter)
- **Q: Multi-token prediction (MTP)?** Predict several future tokens per step during training for a richer signal (DeepSeek-V3); can also seed speculative decoding at inference. (the progression timeline)
- **Q: Distillation in one line?** Train a small student to mimic a big teacher (soft labels or teacher-generated data); how R1's reasoning went into small models. (the inference chapter, the LLM chapter)
- **Q: What's "lost in the middle"?** Models recall info at the start/end of a long context better than the middle — a long window ≠ good long-context use. (the LLM chapter)
- **Q: BLEU/ROUGE limitations?** N-gram overlap metrics; correlate weakly with quality for open-ended generation — supplement with human/LLM eval. (the reading-papers chapter)
- **Q: What's an induction head?** An attention head that implements in-context copying (sees "A B … A" → predicts "B"); a mechanistic basis for in-context learning. (the foundations)

---

**Final prep advice (the meta-answer):** for any question, lead with the one-sentence *why*, then the *mechanism*, then the *tradeoff*, and if you can, a *real number or project* you've shipped. For "modify/design" questions, reason from the bottleneck (the two-phase inference model is the master key for efficiency questions; the four-primitive frame for everything else), enumerate options in order of cost/effort, and always name what each option trades away. That structure is what reads as senior.
