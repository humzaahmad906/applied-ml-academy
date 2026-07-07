# Interview Q&A (the full bank) — Part 1 of 2: Foundations through RAG

Built from what's actually asked in 2025–2026 LLM/ML-engineer interviews, plus the deep-dive and "modify the architecture" style that separates strong candidates. Answers are written at senior level — the *why* and the *tradeoff*, not a definition. Each maps back to the relevant chapter of the guide so you can go deeper. This part (1 of 2) covers sections A–F: foundations through RAG. Part 2 (`09b_interview_qa.md`) covers VLMs, agents, the tricky/deep-dive bank, system design, and rapid-fire.

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

## You can now

- Answer the foundations questions (self-attention from scratch, why √d_k, why O(n²), tokenization tradeoffs, RoPE, pre/post-norm) with the *why* and *mechanism*, not just a definition.
- Answer the architecture-and-training bank (MHA/MQA/GQA/MLA ladder, MoE routing, LoRA/QLoRA, RLHF vs DPO, RLVR/GRPO, catastrophic forgetting, hallucination mitigation) at the depth interviewers grade for.
- Answer inference/efficiency and decoding questions (prefill vs decode bottlenecks, KV cache, FlashAttention, PagedAttention, quantization, speculative decoding, temperature/top-p/min-p) with real numbers where you have them.
- Walk a RAG pipeline end to end (bi- vs cross-encoder, hybrid search, chunking tradeoffs, ANN indexes, RAG vs fine-tuning vs long-context) and debug a "wrong answer" scenario by separating retrieval failure from generation failure.
