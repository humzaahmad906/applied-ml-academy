# Retrieval-Augmented Generation (RAG)

RAG = give the LLM relevant external knowledge at inference time by **retrieving** it and putting it in the context, instead of relying only on the model's frozen weights. It fixes the three core LLM weaknesses: stale knowledge, no access to private/specific data, and hallucination from missing facts. Depends lightly on the foundations and the LLM chapter. The whole field is "retrieve the right stuff, then generate grounded on it" — every technique improves either the *retrieve* or the *ground* half.

---

## 1. The naive RAG pipeline (the baseline everything improves on)

**Indexing (offline):**
1. **Load** documents.
2. **Chunk** them into passages (§4).
3. **Embed** each chunk into a vector with an embedding model (§2).
4. **Store** vectors in a vector index (§3).

**Querying (online):**
5. **Embed the query** with the same model.
6. **Retrieve** the top-k chunks by vector similarity (nearest neighbors).
7. **Augment**: stuff the retrieved chunks into the prompt.
8. **Generate**: the LLM answers using the provided context.

This works and is the right starting point. Everything in §4–7 exists because each step here fails in specific, predictable ways.

---

## 2. Embeddings & retrieval — the core mechanic

An **embedding model** maps a text span to a single dense vector such that *semantically similar* texts have nearby vectors (high cosine similarity / low distance). This is the "dot product = similarity" idea from the foundations applied to whole passages.

- **What it is:** usually an encoder (BERT-style or an LLM adapted for embedding) whose token outputs are **pooled** (mean-pool, or the last token / a special token) into one vector, trained **contrastively** (pull query close to relevant doc, push away irrelevants — same family as the CLIP training in the VLM chapter). Often **instruction-tuned** ("represent this query for retrieval:").
- **What to know about choosing one:** dimensionality (384–4096), max input length (chunk must fit), domain match, and the **MTEB** benchmark (the standard leaderboard — but treat it as a *floor*, not a ranking: top entries are partly benchmark-tuned/contaminated). **Matryoshka embeddings** are now standard: truncate the vector to a shorter length with graceful degradation (256d at ~2–3% loss — big storage/search savings).
- **The 2025–2026 shift:** open models passed the APIs — **Qwen3-Embedding-8B** (70.6 MTEB) outscores every proprietary API embedding. And **BGE-M3** / **Cohere embed-v4** emit *dense and sparse vectors in a single call*, collapsing the two-model hybrid-search pipeline (§5) into one encoder pass. If you're designing retrieval in 2026, "one model, both vector types" is the default to check first.
- **Dense vs sparse:**
  - **Dense (semantic):** the vectors above. Captures meaning/synonyms; can miss exact terms (rare names, IDs, codes).
  - **Sparse / lexical (BM25, TF-IDF, SPLADE):** keyword matching. Catches exact terms dense misses. BM25 is a shockingly strong baseline.
  - **Hybrid** (§5) combines both — usually the right answer.
- **Bi-encoder vs cross-encoder (crucial distinction):**
  - **Bi-encoder:** embed query and doc *separately*, compare vectors. Fast (docs pre-embedded), scalable to millions — used for *retrieval*.
  - **Cross-encoder:** feed query+doc *together* into a model that outputs a relevance score. Much more accurate (full attention between query and doc) but `O(candidates)` forward passes — too slow to run over the whole corpus, so used for **reranking** a small candidate set (§6).
  - **Late interaction (ColBERT):** a middle ground — store per-token vectors, compute fine-grained max-similarity at query time. More accurate than single-vector, cheaper than cross-encoder.

---

## 3. Vector search / ANN & vector databases

You can't compare the query to millions of vectors exactly (brute-force is `O(N·d)`). **Approximate Nearest Neighbor (ANN)** search trades a little recall for huge speed:

- **HNSW (Hierarchical Navigable Small World):** a layered proximity graph you greedily traverse. The dominant in-memory algorithm — fast, high recall, but memory-heavy.
- **IVF (inverted file):** cluster vectors, search only the nearest clusters.
- **PQ (Product Quantization):** compress vectors into codes for memory savings (often combined: IVF-PQ).
- **DiskANN:** ANN that spills to SSD for billion-scale corpora.

**Vector databases** (FAISS — a library, not a full DB; plus Milvus, Qdrant, Weaviate, pgvector, Pinecone, Chroma, LanceDB) wrap an ANN index with metadata filtering, CRUD, persistence, and hybrid search. Key practical features: **metadata filtering** (restrict by date/source/permissions — important and surprisingly tricky to do efficiently alongside ANN) and **hybrid retrieval** support.

Reading takeaway: when a paper/system mentions HNSW/IVF/PQ, it's choosing a point on the **recall vs latency vs memory** triangle. ANN is "exact-enough nearest neighbors, fast."

---

## 4. Chunking — quietly the highest-leverage knob

How you split documents dominates real-world RAG quality, and it's underappreciated.

- **Fixed-size** (N tokens, with **overlap** to avoid cutting facts at boundaries): simple baseline.
- **Recursive / structural:** split on natural boundaries (paragraphs, headings, sentences, code blocks) before resorting to size.
- **Semantic chunking:** split where embedding similarity between adjacent sentences drops (topic shift) — keeps coherent ideas together.
- **Document-structure-aware:** respect markdown headers, tables, sections; keep tables/code intact.
- **Hierarchical / parent-child (small-to-big):** retrieve on *small* precise chunks but feed the LLM the *larger* parent passage for context. Very effective.
- **Late chunking:** embed the whole long document first (so each chunk's vector has full-document context), *then* chunk — preserves context that naive chunking destroys.

The core tension: **small chunks = precise retrieval but fragmented context; big chunks = rich context but diluted, noisier embeddings.** Parent-child and semantic chunking exist to escape this tradeoff. If a RAG system underperforms, chunking is the first thing to revisit.

---

## 5. Hybrid search & query transformation

**Hybrid search:** run dense (semantic) *and* sparse (BM25) retrieval, then fuse the rankings — commonly **Reciprocal Rank Fusion (RRF)**, which combines by rank position without needing comparable scores. Gets semantic recall *and* exact-term precision. Usually beats either alone.

**Query-side transformations** (the query is often a bad search key):
- **Query rewriting / expansion:** the LLM rewrites a messy/conversational query into clean search queries.
- **Multi-query:** generate several paraphrases, retrieve for each, union the results.
- **HyDE (Hypothetical Document Embeddings):** have the LLM *write a hypothetical answer* to the query, then embed *that* and retrieve — because a hypothetical answer is embedding-closer to real answer passages than the question is. Clever and effective.
- **Decomposition:** break a complex/multi-hop question into sub-questions, retrieve for each.

---

## 6. Reranking — the cheap, high-impact second stage

Retrieval (bi-encoder/ANN) optimizes recall: get the relevant docs *somewhere* in the top ~50–100. **Reranking** then reorders that candidate set with a more powerful, slower model to get precision in the top ~3–5 you actually feed the LLM.

- Usually a **cross-encoder** (§2) — query+doc together, full attention, accurate relevance score. (Cohere Rerank, BGE-reranker, etc.)
- **LLM-as-reranker:** prompt an LLM to score/order candidates.
- Why it's worth it: it's run on a small set, so it's affordable, and it sharply improves the final context quality. **Retrieve broad (recall) → rerank narrow (precision)** is the standard two-stage pattern.

---

## 7. Advanced RAG architectures (what modern papers are about)

Naive RAG fails on: multi-hop questions, "summarize the whole corpus" questions, queries needing reasoning, and noisy/irrelevant retrievals poisoning the answer. The responses:

- **RAPTOR:** recursively cluster and **summarize** chunks into a *tree*; retrieve at multiple abstraction levels. Good for questions needing both detail and big-picture synthesis.
- **GraphRAG (Microsoft):** use an LLM to extract **entities and relationships** into a **knowledge graph**, cluster it into communities, and **pre-summarize each community**. Then answer corpus-wide / "global" questions (e.g. "what are the main themes") that vanilla chunk-retrieval simply can't, by reasoning over graph structure and community summaries. The trade: expensive indexing (lots of LLM calls to build the graph). Variants optimize that cost (KET-RAG, E²GraphRAG, LightRAG, PathRAG).
- **HippoRAG / HippoRAG 2:** knowledge-graph + personalized PageRank for cheap multi-hop associative retrieval ("memory"-style continual knowledge).
- **Self-RAG / CRAG (Corrective RAG):** the model **decides whether to retrieve at all**, **critiques** retrieved passages for relevance, and **self-corrects** (re-retrieve, web-search fallback) — adding reflection to the loop.
- **Speculative RAG, MemoRAG, RankRAG**, and many others: each tweaks when/what/how to retrieve and verify.

Mental model: advanced RAG moves from *one-shot flat retrieval* toward *structured (tree/graph) knowledge* and *iterative, self-critiquing* retrieval. Which you need depends on the question type: flat retrieval for "find this fact," graph/tree for "synthesize across the whole corpus," iterative for "multi-hop reasoning."

---

## 8. Agentic RAG (the convergence with agents)

The frontier (and "2025 was the year of agentic RAG"): make retrieval a **decision the model reasons about**, not a fixed pre-step. The LLM acts as an agent (in the sense covered in the agents chapter) that decides *when* to retrieve, *what* to query, whether the results suffice, and whether to retrieve *again* — interleaving reasoning and retrieval.

- **ReAct / Self-Ask / Search-o1** patterns: the model emits a thought, decides to search, reads results, decides if it needs more, then answers. Iterative retrieval beats one-shot for multi-hop.
- **Search/retrieval as a tool** the agent calls (often the same machinery as web-search and tool-use covered in the agents chapter).
- **RL-trained search agents** (Search-R1, Graph-R1): train the model *with RL (RLVR-style, from the post-training material)* to use retrieval well end-to-end, rewarding correct final answers.
- **Multi-agent RAG:** separate planner/retriever/reranker/synthesizer agents.

This is where retrieval and agents merge: "agentic RAG" is just an agent whose primary tool is retrieval. **Deep Research**-type products are the productized version (an agent doing many rounds of search + reasoning + synthesis).

---

## 9. Evaluation — how to know if your RAG works

Evaluate **retrieval** and **generation** separately, then end-to-end:

- **Retrieval:** Recall@k, Precision@k, **MRR** (mean reciprocal rank), **nDCG** (rank-quality with graded relevance). "Did the right chunk get retrieved, and ranked high?"
- **Generation (grounded):** **faithfulness/groundedness** (is the answer supported by the retrieved context, i.e. no hallucination?), **answer relevance**, **context precision/recall**. The **RAGAS** framework and LLM-as-judge are standard for these.
- **End-to-end:** task accuracy / human eval.

The two failure modes to always separate: **retrieval failure** (right info never retrieved → no amount of generation tricks helps) vs **generation failure** (right info retrieved but the model ignored/misused it). Diagnosing which one is the single most useful debugging skill in RAG.

---

## 10. Reading-a-RAG-paper checklist

- **Which half does it improve** — retrieval (embeddings, ANN, chunking, hybrid, rerank) or generation/grounding (reflection, faithfulness)?
- **What query type is it for** — single-fact lookup, corpus-wide synthesis, or multi-hop reasoning? (Determines whether flat/tree/graph/agentic is the right tool.)
- **Static pipeline or agentic** (does the model decide when/what to retrieve)?
- **Indexing cost** — graph/tree methods pay heavily upfront (many LLM calls); is the quality worth it at the corpus scale they test?
- **Evaluation** — did they separate retrieval vs generation quality, and use faithfulness, not just answer accuracy?
- **The one-sentence contribution and its cost** (usually indexing cost, latency, or added complexity).

> **RAG vs long-context vs fine-tuning** — the perennial design question. *RAG*: dynamic/changing knowledge, need citations/provenance, large corpora. *Long context* (the long-context material from the LLM chapter): the relevant info fits and you want the model to reason over all of it jointly. *Fine-tuning* (the SFT/PEFT material from the LLM chapter): teach *behavior/format/style* or deeply internalize a stable domain — not for injecting fast-changing facts. In practice these compose (e.g. retrieve into a long context; fine-tune the retriever or the generator's grounding behavior).
