# Module 06 — Retrieval & RAG Systems

## Why this module matters

Retrieval is the connective tissue of every 2026 LLM product: RAG assistants, agent tool-search, memory systems, code search, and the candidate-generation stage of recommendations (covered in the classic-ML chapter) are all the same retrieval stack wearing different hats. The naive RAG of 2023 (chunk → embed → top-k → stuff prompt) is now considered a baseline; interviews probe whether you know what breaks it and what the modern stack looks like.

## 1. The retrieval stack, layer by layer

**Embeddings.** Bi-encoder models map queries and documents into one vector space (check MTEB, but trust your own task eval — MTEB overfitting is real). Modern niceties: instruction-prefixed embeddings (different prompts for query vs document), **Matryoshka representation learning** (truncate 1024-d vectors to 256-d with graceful degradation — a free index-size/latency knob), and domain fine-tuning of embedders with contrastive pairs mined from your own logs (one of the highest-ROI fine-tunes that juniors never propose).

**Chunking.** The unglamorous decision that dominates RAG quality. Defaults: 256–1024 tokens, 10–20% overlap, split on structural boundaries (headings, paragraphs, functions) not raw character counts. Two upgrades worth knowing: **contextual retrieval** — prepend an LLM-generated document-level context blurb to each chunk before embedding, cutting retrieval failures dramatically by de-orphaning chunks; and **parent-child retrieval** — embed small chunks for precision, return their larger parent sections for context.

**Indexes (ANN).** Exact search is O(N); approximate nearest neighbor makes it sublinear:

- **HNSW** — multi-layer navigable small-world graph; the default for in-memory indexes. Knobs: `M` (graph degree → memory/recall), `efConstruction`, `efSearch` (query-time beam → recall/latency dial).
- **IVF-PQ** — cluster into nlist cells, probe nprobe at query time, compress vectors with product quantization; the memory-frugal choice for 100M+ vectors.
- **DiskANN/Vamana** — SSD-resident graphs for billion-scale on modest RAM.
- GPU options (FAISS-GPU, CAGRA) for extreme QPS. Know the recall@latency tradeoff story rather than vendor names: pgvector (operationally simplest, lives in your Postgres), Qdrant/Milvus/Weaviate (dedicated engines: filtering, hybrid, scale-out), LanceDB/Turbopuffer (object-storage-native, cheap at huge scale).

**Hybrid retrieval.** Dense vectors miss exact identifiers, rare terms, and part numbers; lexical (BM25) misses paraphrase. Production default: run both, merge with **Reciprocal Rank Fusion (RRF)** — simple, tuning-free, robust. Learned sparse (SPLADE) is the middle path.

**Reranking.** A cross-encoder (or LLM listwise reranker) jointly attends over (query, candidate) pairs for the top 50–100 candidates from the cheap stage — typically the single biggest quality jump per engineering-hour in the whole stack. This retrieve-then-rerank shape is the same multi-stage funnel as recommendation systems (see the classic-ML chapter); say that in interviews.

**Late interaction & multimodal.** **ColBERT** stores per-token vectors and scores via MaxSim — quality between bi- and cross-encoders with index-time precomputation. Its descendant **ColPali/ColQwen** applies the same trick to *document page images* via VLMs — retrieving over rendered pages directly, skipping brittle OCR/layout pipelines entirely. For document-heavy products this is the most important retrieval development of the last two years.

## 2. Beyond single-shot RAG

- **Query understanding:** rewriting (decontextualize "what about its battery?" using chat history), decomposition of multi-hop questions, HyDE (embed a hypothetical answer), metadata-filter extraction (date ranges, doc types) — cheap LLM pre-steps that fix a large share of failures.
- **Agentic retrieval (the 2026 default for hard questions):** instead of one retrieve-then-answer pass, the model gets search as a *tool* and iterates — query, read, refine, query again. Subsumes multi-hop tricks; costs latency/tokens; pairs with the agent-loop design in the agentic-systems chapter. Simple lookup → classic RAG; investigative questions → agentic search.
- **GraphRAG:** LLM-extracted entity/relation graph + community summaries; answers corpus-global questions ("what are the recurring themes across these 10k reports?") that chunk-level top-k structurally cannot. Expensive to build; use when questions are about the corpus, not in it.
- **Long context vs RAG:** million-token contexts didn't kill RAG — cost scales with tokens, latency with prefill, and effective use of mid-context information remains unreliable ("lost in the middle", "context rot"). The synthesis: retrieval narrows from millions of documents to the relevant dozens; long context lets you be generous with those dozens. They compose; they don't compete.

## 3. Evaluating RAG (where most teams are weakest)

Evaluate the two stages separately — end-to-end answer quality alone can't tell you *which* stage failed:

- **Retrieval metrics:** recall@k (did the gold evidence appear?), MRR/nDCG. Build a golden set: 100–300 (question → gold-evidence-chunks) pairs, partly human-written, partly synthetic (LLM generates questions *from* sampled chunks — automatic ground truth) — then audited.
- **Generation metrics (the RAG triad):** **faithfulness/groundedness** (is every claim supported by retrieved context? — the hallucination metric), **answer relevance**, **context precision/utilization**. LLM-as-judge implementations work but inherit judge biases — calibrate against a human-labeled subset (covered in the evaluation chapter).
- **Operational metrics:** retrieval latency p99, index freshness lag, cost/query, and the hit-rate of "I don't know" on questions genuinely outside the corpus (abstention quality — under-measured everywhere).

## 4. ICL example selection as retrieval

Dynamic few-shot selection is a retrieval problem wearing a prompting hat — and recognizing that connection lets you reuse the entire retrieval stack described earlier without building anything new.

**The idea.** Instead of a static set of hand-picked few-shot examples in your system prompt, maintain an **example bank**: a corpus of (input, ideal output) pairs, each embedded with your standard bi-encoder. At inference time, embed the live query, retrieve the top-k most similar examples from the bank, and insert them as the few-shot context before the user's input. The model sees examples that are semantically close to the current task, rather than generic ones chosen by whoever wrote the prompt.

**Why it beats static examples.** Static few-shot examples are chosen to cover an imagined average case. Real query distributions have long tails — unusual document types, rare intents, edge-case formats — where the static examples provide little signal. Dynamic selection retrieves the examples most relevant to this specific query, including edge cases the prompt author didn't anticipate. Empirically, dynamic few-shot selection consistently outperforms static selection on tasks with high input diversity, without any additional training.

**It's just RAG over a different corpus.** The infrastructure is identical: embed the example bank offline, store in an ANN index, retrieve at query time, inject into context. The only differences are corpus content (input/output pairs instead of documents), chunk granularity (one example = one chunk), and the injection point (before the user query, in the few-shot position). You already have this infrastructure if you have a RAG pipeline — the example bank is just another index.

**When it matters most.** Dynamic example selection is highest-value when: (1) fine-tuning is not on the table (data volume too low, latency constraints, cost of retraining); (2) the task distribution is broad or shifts over time (new document types, new user intents — update the example bank by adding new examples, no retraining required); (3) you are using a strong base model and want to steer it toward task-specific output format without SFT.

**Operational notes.** The example bank needs curation — low-quality examples retrieved at high similarity will degrade outputs faster than no examples at all. Version the bank like a dataset, gate changes through the eval suite, and monitor per-example retrieval frequency (popular examples should be high-quality; stale examples that never retrieve can be pruned). At high QPS, the retrieval latency adds to TTFT — keep the bank index small (hundreds to low thousands of examples) and the bi-encoder fast, or pre-cache the query embedding if the query arrives in a predictable form.

## Going deeper

- The retrieval stack layers cleanly — embeddings, chunking, ANN index, hybrid dense+lexical fusion, reranking — and each layer has a well-studied set of tradeoffs. Learn the recall@latency story for HNSW, IVF-PQ, and disk-resident graphs rather than memorizing vendor names.
- Late-interaction retrieval (per-token vectors scored by MaxSim) and its vision-model descendants (retrieving over rendered document page images directly) are the most important recent developments for document-heavy products.
- Contextual chunk headers, parent-child retrieval, and query rewriting/decomposition each fix a distinct class of retrieval failure — study which failure each one targets.
- The RAG triad (faithfulness, answer relevance, context precision) and the "lost in the middle" degradation are the two evaluation facts that matter most; the Project below builds a real eval harness around them.

## Project 06 — A RAG system with a real eval harness

Corpus: ~5k arXiv abstracts+intros (or your own document set). Build: (1) baseline — fixed-size chunks, one embedding model, HNSW (FAISS or Qdrant), top-k → answer with a small local LLM; (2) golden set — 150 synthetic QA pairs generated from sampled chunks + 30 hand-written hard ones (multi-hop, exact-identifier, out-of-corpus); (3) measure retrieval recall@5/@20 and judge-scored faithfulness; (4) ablate one axis at a time and chart deltas: + BM25 hybrid w/ RRF, + cross-encoder reranker, + contextual chunk headers, chunk size 256 vs 1024, efSearch sweep (recall vs latency curve); (5) report which intervention bought the most per millisecond added. Stretch: index 200 PDF pages as *images* with ColPali and compare against the OCR-text pipeline on visually-rich pages (tables, figures).

## Interview Q&A

**Q1. Walk through HNSW's main parameters and their tradeoffs.**
**A.** HNSW builds a hierarchy of proximity graphs — sparse upper layers for coarse navigation, dense bottom layer for precision; queries greedily descend then beam-search the bottom layer. `M` (max neighbors per node) sets graph connectivity: higher M → better recall and robustness on hard distributions, more memory (the index is often larger than the raw vectors) and slower builds. `efConstruction` is build-time beam width: higher → better graph quality, slower indexing — a one-time cost usually worth paying. `efSearch` is the query-time beam width and the *operational* dial: it trades recall directly against latency per query and can be tuned live per traffic class (e.g., 95% recall for autocomplete, 99.5% for legal search). Two more things worth volunteering: deletes are awkward (tombstones + periodic rebuilds), and metadata filtering interacts badly with graph traversal (pre-filtering can strand the search), which is exactly the problem dedicated vector DBs compete on solving.

**Q2. Bi-encoder vs cross-encoder vs late interaction — what's the actual tradeoff?**
**A.** A **bi-encoder** encodes query and document independently into single vectors; all document vectors are precomputed and indexed, so search is one query-encode plus an ANN lookup — scales to billions, but the single-vector bottleneck loses fine-grained term interactions. A **cross-encoder** concatenates query+document and attends jointly — far more accurate, but nothing is precomputable, so it's O(model-forward) per candidate: usable only to rerank a short list. **Late interaction (ColBERT)** is the negotiated middle: per-token document embeddings precomputed offline; at query time, score = sum of each query token's max similarity over document tokens (MaxSim) — most of cross-attention's term-level matching at near-bi-encoder serving cost, paying instead in index size (per-token vectors, mitigated by compression in ColBERTv2). The production pattern composes them: bi-encoder (+BM25) retrieves hundreds, cross-encoder or late-interaction reranks to ten.

**Q3. Your RAG bot gives a confidently wrong answer. Debug it.**
**A.** Localize the failure stage with the eval harness rather than prompt-twiddling. (1) **Retrieval check:** does the gold evidence exist in the corpus, did it survive chunking intact, and did it appear in top-k? If not — diagnose embedding miss (query/document vocabulary mismatch → hybrid BM25, embedder fine-tune, query rewriting), chunking orphaned the fact (→ contextual headers, parent-child), or stale index. (2) **If retrieval succeeded, generation check:** did the model ignore the context (position — "lost in the middle"; → rerank to put best evidence first/last), get contradicted by its parametric knowledge, or face *conflicting* retrieved chunks (→ recency/source-authority metadata in the prompt)? (3) **If the question is outside the corpus**, the failure is missing abstention — add explicit "answer only from context / say you don't know" behavior and *measure* abstention on out-of-corpus questions. Then add the failing case to the golden set so the fix is regression-protected — turning bugs into evals is the discipline interviewers want to hear.

**Q4. RAG vs long context vs fine-tuning — how do you decide?**
**A.** They solve different problems and compose. **Fine-tuning** teaches *behavior* — format, style, domain reasoning patterns, tool conventions; it is a poor and unmaintainable mechanism for storing *facts* (update = retrain, no citations, hallucination-prone recall). **RAG** injects *knowledge* — fresh, per-user, access-controlled, citable; updating = re-indexing a document. **Long context** is a capacity question, not a knowledge strategy: it determines how generously you can pass what retrieval found, but stuffing a whole corpus per query fails on cost (you pay prefill for every token, every request), latency, and degraded mid-context attention. Decision pattern: knowledge that changes or is user-scoped → RAG; consistent behavior/skill on a narrow task at volume → fine-tune (often distillation); both → both — a fine-tuned model *operating over* retrieved context is the standard production shape. Caching of stable long prefixes (see the serving chapter) shifts the economics toward "more context" for shared corpora, which is worth mentioning as the 2026 nuance.

**Q5. Design semantic search over 200M product listings, p99 < 150 ms, with filters.**
**A.** Two-stage funnel. **Offline:** fine-tune a bi-encoder on click/purchase pairs from search logs (in-batch negatives + hard negatives mined from BM25); embed listings (Matryoshka-truncate to 256–512d); index with IVF-PQ or sharded HNSW — at 200M vectors, memory math decides: 200M × 512d × fp16 ≈ 200 GB raw → PQ compression or disk-resident (DiskANN) per shard; shard by category or hash, replicate for QPS. **Online:** query → light normalization + metadata-filter extraction → parallel dense + BM25 → RRF merge of ~500 → cross-encoder rerank of top 100 (small, distilled, batched on GPU, ~20–40 ms) → business-rule re-rank (availability, margin, diversity). Filters: pre-filter inside the ANN engine where selective (price range, category), post-filter when broad. **Freshness:** new/updated listings flow through a streaming embed pipeline into a small fresh-index tier merged at query time, with nightly compaction into the main index. **Eval:** recall@k against logged purchases offline, interleaving experiments online (see the evaluation chapter), and a latency budget per stage stated explicitly: 5 retrieval + 35 rerank + 10 merge ≈ well inside p99.
