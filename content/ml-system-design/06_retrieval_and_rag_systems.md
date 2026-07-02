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

### Eval harness sketch

Build the harness before the first ablation — it prevents you from chasing noise.

**Golden set construction.** Sample chunks uniformly, ask an LLM "write one question whose complete answer is in this passage", and record the source chunk ID as ground truth. Hand-write 20–30 adversarial pairs: multi-hop questions, exact identifiers, and out-of-corpus questions for abstention testing. Audit the synthetic pairs before freezing — LLM-generated questions occasionally reference facts not present in the chunk.

```python
import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class GoldenPair:
    question: str
    gold_chunk_ids: list[str]   # ≥1 chunk must appear for a retrieval hit
    reference_answer: str        # used only for faithfulness calibration


def recall_at_k(
    golden: list[GoldenPair],
    retrieve_fn: Callable[[str, int], list[str]],  # (query, k) -> [chunk_id, ...]
    k: int = 5,
) -> float:
    """Fraction of golden pairs where ≥1 gold chunk appears in the top-k result."""
    hits = sum(
        bool(set(retrieve_fn(pair.question, k)) & set(pair.gold_chunk_ids))
        for pair in golden
    )
    score = hits / len(golden)
    logger.info("recall@%d = %.3f  (%d/%d)", k, score, hits, len(golden))
    return score
```

**Faithfulness (claim-by-claim).** Decompose the model's answer into atomic claims with a short LLM call — "list every factual claim in this sentence as a JSON array". Then for each claim ask the judge: "Is this claim supported by the context below? Answer yes or no." Faithfulness = (supported claims) / (total claims). This is deliberately pessimistic: one unsupported claim in a three-claim answer scores 0.67, which is the right sensitivity for hallucination detection.

```python
def faithfulness_score(
    claims: list[str],
    context_chunks: list[str],
    judge_fn: Callable[[str, str], bool],  # (claim, context) -> grounded?
) -> float:
    """Claim-level groundedness. Decompose the answer into claims before calling this."""
    if not claims:
        return 1.0
    context = "\n\n".join(context_chunks)
    supported = sum(judge_fn(claim, context) for claim in claims)
    return supported / len(claims)
```

Calibrate the judge against 50–100 human-labeled (claim, context, grounded) triples before trusting it at scale — LLM judges systematically over-rate "supported" on longer contexts (covered in the evaluation chapter).

**Ablation protocol.** Run recall@5, recall@20, and mean faithfulness across the full golden set for each variant — record absolute numbers, not just deltas:

| Variant | recall@5 | recall@20 | faithfulness |
| --- | --- | --- | --- |
| Baseline: dense-only, 512-token chunks | — | — | — |
| + BM25 hybrid (RRF k=60) | — | — | — |
| + cross-encoder reranker | — | — | — |
| + contextual chunk headers | — | — | — |
| Chunk size 256 vs 1024 | — | — | — |
| efSearch sweep (32 → 128) | latency curve | — | — |

The reranker typically moves the faithfulness needle more than it moves recall@20 — it does not surface new evidence; it promotes the right evidence to positions the model uses.

## 4. ICL example selection as retrieval

Dynamic few-shot selection is a retrieval problem wearing a prompting hat — and recognizing that connection lets you reuse the entire retrieval stack described earlier without building anything new.

**The idea.** Instead of a static set of hand-picked few-shot examples in your system prompt, maintain an **example bank**: a corpus of (input, ideal output) pairs, each embedded with your standard bi-encoder. At inference time, embed the live query, retrieve the top-k most similar examples from the bank, and insert them as the few-shot context before the user's input. The model sees examples that are semantically close to the current task, rather than generic ones chosen by whoever wrote the prompt.

**Why it beats static examples.** Static few-shot examples are chosen to cover an imagined average case. Real query distributions have long tails — unusual document types, rare intents, edge-case formats — where the static examples provide little signal. Dynamic selection retrieves the examples most relevant to this specific query, including edge cases the prompt author didn't anticipate. Empirically, dynamic few-shot selection consistently outperforms static selection on tasks with high input diversity, without any additional training.

**It's just RAG over a different corpus.** The infrastructure is identical: embed the example bank offline, store in an ANN index, retrieve at query time, inject into context. The only differences are corpus content (input/output pairs instead of documents), chunk granularity (one example = one chunk), and the injection point (before the user query, in the few-shot position). You already have this infrastructure if you have a RAG pipeline — the example bank is just another index.

**When it matters most.** Dynamic example selection is highest-value when: (1) fine-tuning is not on the table (data volume too low, latency constraints, cost of retraining); (2) the task distribution is broad or shifts over time (new document types, new user intents — update the example bank by adding new examples, no retraining required); (3) you are using a strong base model and want to steer it toward task-specific output format without SFT.

**Operational notes.** The example bank needs curation — low-quality examples retrieved at high similarity will degrade outputs faster than no examples at all. Version the bank like a dataset, gate changes through the eval suite, and monitor per-example retrieval frequency (popular examples should be high-quality; stale examples that never retrieve can be pruned). At high QPS, the retrieval latency adds to TTFT — keep the bank index small (hundreds to low thousands of examples) and the bi-encoder fast, or pre-cache the query embedding if the query arrives in a predictable form.

## 5. RAG failure modes

Every failure maps to a retrieval stage, a generation stage, or an operational gap. The table makes the debug decision tree faster — pair each row with the localization step in the eval harness above.

| Failure mode | Symptom | Root cause | Fix |
| --- | --- | --- | --- |
| Retrieval miss — embedding gap | Gold evidence exists; BM25 finds it, dense misses | Query/document vocabulary mismatch; domain distribution shift | BM25 hybrid + RRF; mine hard negatives from logs; fine-tune embedder on domain pairs |
| Chunk boundary splits the answer | Neither of two adjacent chunks is sufficient alone | Fixed-size splitting cuts mid-sentence or mid-table | Sentence/paragraph-boundary splitting; 15–20% overlap; parent-child retrieval |
| Lost in the middle | Top-k retrieved correctly; model ignores middle chunks | Attention degrades for tokens far from context boundaries | Rerank: put the highest-scoring chunk first and last; pass fewer but higher-quality chunks |
| Stale index | Queries return outdated or deleted content | Indexing lag; deleted docs still retrievable | Streaming embed pipeline + fresh-index tier merged at query time; tombstone on delete |
| Out-of-corpus hallucination | Model fabricates confidently; no relevant chunk retrieved | No abstention instruction; no OOC signal | Explicit "say I don't know if not in context"; measure abstention on a held-out OOC set |

In an interview, naming these five failure modes and pairing each with its localization step — retrieval recall vs. faithfulness score — is the senior signal. Most candidates debug by guessing at prompts.

## Going deeper

- The retrieval stack layers cleanly — embeddings, chunking, ANN index, hybrid dense+lexical fusion, reranking — and each layer has a well-studied set of tradeoffs. Learn the recall@latency story for HNSW, IVF-PQ, and disk-resident graphs rather than memorizing vendor names.
- Late-interaction retrieval (per-token vectors scored by MaxSim) and its vision-model descendants (retrieving over rendered document page images directly) are the most important recent developments for document-heavy products.
- Contextual chunk headers, parent-child retrieval, and query rewriting/decomposition each fix a distinct class of retrieval failure — study which failure each one targets.
- The RAG triad (faithfulness, answer relevance, context precision) and the "lost in the middle" degradation are the two evaluation facts that matter most; the Project below builds a real eval harness around them.

## Project 06 — A RAG system with a real eval harness

Corpus: ~5k arXiv abstracts+intros (or your own document set). Build: (1) baseline — fixed-size chunks, one embedding model, HNSW (FAISS or Qdrant), top-k → answer with a small local LLM; (2) golden set — 150 synthetic QA pairs generated from sampled chunks + 30 hand-written hard ones (multi-hop, exact-identifier, out-of-corpus); (3) measure retrieval recall@5/@20 and judge-scored faithfulness; (4) ablate one axis at a time and chart deltas: + BM25 hybrid w/ RRF, + cross-encoder reranker, + contextual chunk headers, chunk size 256 vs 1024, efSearch sweep (recall vs latency curve); (5) report which intervention bought the most per millisecond added. Stretch: index 200 PDF pages as *images* with ColPali and compare against the OCR-text pipeline on visually-rich pages (tables, figures).

### Stage-by-stage walkthrough

Each stage has a concrete expected output so you can confirm correctness before moving on.

**Stage 1 — Chunking.** The naive mistake is fixed-size character splitting: it cuts sentences and tables mid-thought, handing the embedder incoherent fragments. Paragraph-boundary splitting with overlap is the baseline to beat.

```python
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def split_paragraphs_with_overlap(
    text: str,
    doc_id: str,
    max_words: int = 400,
    overlap_words: int = 60,
) -> list[Chunk]:
    """
    Split at paragraph boundaries; merge short paragraphs to stay near max_words;
    carry an overlap window into each new chunk to preserve cross-boundary context.
    Fixed-size character splits measurably hurt retrieval recall on domain corpora
    vs. paragraph-aligned chunking — the overlap preserves context that a hard
    boundary would sever.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_wc = 0

    def flush() -> None:
        if buffer:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_{len(chunks)}",
                    doc_id=doc_id,
                    text="\n\n".join(buffer),
                )
            )

    for para in paragraphs:
        wc = len(para.split())
        if buffer_wc + wc > max_words and buffer:
            flush()
            tail: list[str] = []
            tail_wc = 0
            for prev in reversed(buffer):
                pw = len(prev.split())
                if tail_wc + pw > overlap_words:
                    break
                tail.insert(0, prev)
                tail_wc += pw
            buffer, buffer_wc = tail, tail_wc
        buffer.append(para)
        buffer_wc += wc
    flush()
    return chunks
```

*Expected output:* 5k arXiv abstracts yield roughly 8k–12k chunks, median ~300 words, with ~15% of content duplicated across overlap windows. Spot-check a random sample — chunks should begin and end on sentence boundaries.

**Stage 2 — Embedding + HNSW index.** `normalize_embeddings=True` puts vectors on the unit sphere so FAISS inner product equals cosine similarity.

```python
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def build_hnsw_index(
    chunks: list[Chunk],
    model_name: str = "BAAI/bge-small-en-v1.5",  # representative 2026 — verify on MTEB for your domain
    index_path: Path | None = None,
) -> tuple[faiss.Index, list[str]]:
    """Encode chunks and build an HNSW index with cosine similarity via inner product."""
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    logger.info("Encoding %d chunks with %s", len(texts), model_name)
    vecs = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    vecs = np.array(vecs, dtype=np.float32)

    dim = vecs.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vecs)
    logger.info("Index built: %d vectors, dim=%d", index.ntotal, dim)
    if index_path:
        faiss.write_index(index, str(index_path))
    return index, chunk_ids
```

*Expected output:* build time roughly 2–5 min on CPU for 10k vectors; `index.ntotal == len(chunks)`. Sanity query: an exact phrase from a known chunk should return that chunk in top-3 at default `efSearch`.

**Stage 3 — Hybrid search (BM25 + dense + RRF).** Build the `BM25Okapi` index from tokenized chunk texts before calling `hybrid_retrieve`.

```python
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def reciprocal_rank_fusion(
    *ranked_lists: list[str], k: int = 60
) -> list[tuple[str, float]]:
    """k=60 is the standard constant from the original RRF paper (Cormack et al., 2009)."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(
    query: str,
    dense_index: faiss.Index,
    chunk_ids: list[str],
    bm25_index: BM25Okapi,
    encoder: SentenceTransformer,
    top_n: int = 50,
) -> list[str]:
    qvec = encoder.encode([query], normalize_embeddings=True).astype(np.float32)
    _, dense_idxs = dense_index.search(qvec, top_n)
    dense_ranked = [chunk_ids[i] for i in dense_idxs[0] if i != -1]

    tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokens)
    sparse_ranked = [chunk_ids[i] for i in np.argsort(bm25_scores)[::-1][:top_n]]

    merged = reciprocal_rank_fusion(dense_ranked, sparse_ranked)
    return [cid for cid, _ in merged[:top_n]]
```

*Expected output:* a query containing an exact model number or author name should rank its source chunk higher with hybrid than with dense alone. That exact-identifier test is the cheapest sanity check that hybrid is working correctly.

**Stage 4 — Cross-encoder reranker.** Run only over the top 50 candidates from Stage 3; the cross-encoder reads the full (query, chunk) pair jointly and is not precomputable.

```python
from sentence_transformers import CrossEncoder


def rerank(
    query: str,
    candidate_ids: list[str],
    chunk_store: dict[str, str],   # chunk_id -> text
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",  # representative 2026 — check current
    top_k: int = 10,
) -> list[str]:
    reranker = CrossEncoder(model_name)
    valid_ids = [cid for cid in candidate_ids if cid in chunk_store]
    pairs = [(query, chunk_store[cid]) for cid in valid_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:top_k]]
```

*Expected output:* the reranker should place a clearly relevant chunk at position #1 for the majority of easy questions in the golden set. Track latency — a small cross-encoder on CPU takes roughly 200–500 ms for 50 candidates; batch on GPU for production use.

**Stage 5 — Generation.** Pass reranked chunks best-first; "sandwich" ordering (best chunk first and last) mitigates lost-in-the-middle for longer contexts.

```python
from typing import Callable


def generate_answer(
    query: str,
    top_chunk_texts: list[str],   # ordered best-first from the reranker
    generate_fn: Callable[[str], str],
) -> str:
    context = "\n\n---\n\n".join(top_chunk_texts)
    prompt = (
        "Answer using only the context below. "
        'If the answer is not in the context, say "I don\'t know."\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return generate_fn(prompt)
```

*Expected output:* on the 150-pair golden set, the full pipeline (all five stages) should reach roughly faithfulness > 0.75 and recall@5 > 0.60 on a clean arXiv corpus. If it falls short, trace back through the stage-level sanity checks before touching the generation prompt.

For the end-to-end RAG case study — serving architecture, latency budget breakdown, online eval loop, and production incident analysis — see the case-studies chapter (module 13).

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
