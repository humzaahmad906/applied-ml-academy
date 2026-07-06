# Lab 5 — Build a RAG Pipeline End to End

This lab makes the [RAG chapter](05_rag.md) concrete. Build every component from scratch — word-overlap chunker, dense bi-encoder, FAISS index, cross-encoder reranker, local instruct generator, BM25 hybrid retrieval, and a hit-rate eval — so the mechanics are fully visible before reaching for a framework.

## Setup

```bash
pip install torch transformers accelerate sentence-transformers faiss-cpu rank_bm25
# alternatives: pip install llama-index llama-index-embeddings-huggingface
#               pip install langchain langchain-community langchain-huggingface
#               pip install qdrant-client chromadb
```

Models: `all-MiniLM-L6-v2` (~90 MB, 384-D bi-encoder), `cross-encoder/ms-marco-MiniLM-L-6-v2` (~85 MB), `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB generator). All run on CPU; MPS/CUDA speeds up embedding and generation.

---

## 1. Corpus and chunking

```python
import random
from typing import Any

import faiss
import numpy as np
import torch

torch.manual_seed(42); np.random.seed(42); random.seed(42)

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"device: {device}")

CORPUS = [
    {"doc_id": "transformers-attn",
     "text": "Transformers model all-position dependencies via Q/K/V self-attention (weights = softmax(QKᵀ/√d)V). Unlike RNNs all positions process in parallel, removing the sequential bottleneck and enabling large model scales."},
    {"doc_id": "lora-finetuning",
     "text": "LoRA injects trainable low-rank matrices A,B into attention projections (ΔW=AB, rank 4-64). Only A and B are trained, cutting trainable parameters by orders of magnitude and GPU memory by 10x vs full fine-tuning with no quality loss."},
    {"doc_id": "flash-attention",
     "text": "FlashAttention rewrites attention to avoid materializing the N×N matrix in HBM by tiling into SRAM. Memory bandwidth drops from O(N²) to O(N), cutting wall-clock time 2-4x. Output is mathematically identical to standard attention."},
    {"doc_id": "kvcache-inference",
     "text": "The KV cache stores K and V tensors for all previous tokens, avoiding recomputation during autoregressive generation. Without it a 1000-token sequence costs O(N²) recompute; with it each new token costs O(N). Cache size grows with sequence length and batch."},
    {"doc_id": "moe-architecture",
     "text": "MoE replaces the dense FFN with N expert FFNs and a linear router. Each token activates top-k experts (e.g. k=2 of 64 in Mixtral), keeping FLOPs constant while scaling parameters. All expert weights must reside in memory simultaneously."},
    {"doc_id": "rag-overview",
     "text": "RAG improves LLM factual accuracy by retrieving relevant documents at inference time and conditioning the generator on them. The generator answers from retrieved context only, reducing hallucination. Effective for domain-specific or frequently updated knowledge bases."},
]


def chunk_words(text: str, chunk_size: int = 64, overlap: int = 16) -> list[str]:
    words = text.split()
    result, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        result.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return result


chunks: list[dict[str, Any]] = []
for doc in CORPUS:
    for i, txt in enumerate(chunk_words(doc["text"])):
        chunks.append({"doc_id": doc["doc_id"], "chunk_id": f"{doc['doc_id']}-{i}", "text": txt})

print(f"Corpus: {len(CORPUS)} docs → {len(chunks)} chunks")
```

Chunk size and overlap are the primary retrieval quality knobs. Too large: chunks span multiple topics. Too small: context splits across boundaries. 64-128 words with 16-20 word overlap is a good prose starting point.

---

## 2. Embed and index with FAISS

```python
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embeddings  = embed_model.encode(
    [c["text"] for c in chunks], batch_size=32, convert_to_numpy=True
)
print(f"Embeddings: {embeddings.shape}  dtype={embeddings.dtype}")  # (N, 384), float32

normed = embeddings.copy()
faiss.normalize_L2(normed)                      # cosine sim = inner product after L2 norm
index  = faiss.IndexFlatIP(embeddings.shape[1]) # exact brute-force
index.add(normed.astype(np.float32))
print(f"FAISS: {index.ntotal} vectors, dim={index.d}")
```

`all-MiniLM-L6-v2` is a bi-encoder: query and document encoded independently, retrieval is a dot-product scan — O(1) per query against the prebuilt index. `IndexFlatIP` is exact; for >100k docs swap to `IndexIVFFlat` or `IndexHNSWFlat` for approximate ANN.

---

## 3. Dense retrieval + cross-encoder reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

TEST_Q = "How does LoRA reduce GPU memory requirements during fine-tuning?"


def dense_retrieve(query: str, k: int = 10) -> list[dict[str, Any]]:
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    return [{"score": float(scores[0][i]), "chunk": chunks[int(idxs[0][i])]} for i in range(k)]


def rerank(query: str, candidates: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
    pairs  = [(query, c["chunk"]["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_n]]


top3 = rerank(TEST_Q, dense_retrieve(TEST_Q, k=10), top_n=3)
print("\nTop-3 after reranking:")
for r in top3:
    print(f"  ({r['chunk']['doc_id']}) {r['chunk']['text'][:100]}...")
```

The cross-encoder sees `[query; document]` jointly — it models their interaction and ranks more accurately than cosine similarity. But it is O(k) forward passes per query. The pattern is: fast bi-encoder to get top-k candidates, cross-encoder to rerank them.

## 4. Generate with a local instruct model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

gen_id    = "Qwen/Qwen2.5-0.5B-Instruct"
gen_tok   = AutoTokenizer.from_pretrained(gen_id)
gen_model = AutoModelForCausalLM.from_pretrained(gen_id, torch_dtype=torch.bfloat16).to(device)
gen_model.eval()

def rag_generate(query: str, context_chunks: list[dict[str, Any]], max_new_tokens: int = 256) -> str:
    context  = "\n\n".join(c["chunk"]["text"] for c in context_chunks)
    user_msg = (
        "Answer based solely on the context below. If the answer is not there, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    messages = [{"role": "user", "content": user_msg}]
    prompt   = gen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = gen_tok(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        out = gen_model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    n_prompt = inputs["input_ids"].shape[1]
    return gen_tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()


answer = rag_generate(TEST_Q, top3)
print(f"\nQ: {TEST_Q}\nA: {answer}")
```

## 5. Hybrid BM25 + dense retrieval

```python
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])


def hybrid_retrieve(query: str, k: int = 5, alpha: float = 0.5) -> list[dict[str, Any]]:
    n = len(chunks)
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    dense_scores, dense_idxs = index.search(q_emb, n)
    dense_arr = np.zeros(n, dtype=np.float32)
    dense_arr[dense_idxs[0]] = dense_scores[0]
    bm25_arr  = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)

    def norm01(a: np.ndarray) -> np.ndarray:
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-9)

    combined = alpha * norm01(dense_arr) + (1.0 - alpha) * norm01(bm25_arr)
    top_idxs  = np.argsort(combined)[::-1][:k]
    return [{"score": float(combined[i]), "chunk": chunks[int(i)]} for i in top_idxs]
```

BM25 handles exact-match keywords — acronyms, version numbers, named entities — that dense embeddings sometimes blur. Hybrid consistently outperforms either alone on technical queries.

## 6. Eval — hit-rate and groundedness

```python
QA_EVAL = [
    {"query": "How does LoRA reduce GPU memory during fine-tuning?",          "doc_id": "lora-finetuning"},
    {"query": "What makes transformers faster to train than RNNs?",           "doc_id": "transformers-attn"},
    {"query": "How does FlashAttention reduce memory bandwidth?",             "doc_id": "flash-attention"},
    {"query": "What is the KV cache and why does it matter for generation?",  "doc_id": "kvcache-inference"},
    {"query": "How does the MoE router select experts for each token?",       "doc_id": "moe-architecture"},
]


def hit_rate_at_k(qa_pairs: list[dict], k: int = 5, hybrid: bool = False) -> float:
    hits = 0
    for pair in qa_pairs:
        fn   = hybrid_retrieve if hybrid else dense_retrieve
        ids  = {r["chunk"]["doc_id"] for r in fn(pair["query"], k=k)}
        hits += pair["doc_id"] in ids
    return hits / len(qa_pairs)


def groundedness(answer: str, context_chunks: list[dict[str, Any]]) -> float:
    """Word-overlap proxy for faithfulness. Use LLM-as-judge in production."""
    ctx   = set(" ".join(c["chunk"]["text"] for c in context_chunks).lower().split())
    words = answer.lower().split()
    return sum(1 for w in words if w in ctx) / max(len(words), 1)


hr_dense  = hit_rate_at_k(QA_EVAL, k=5, hybrid=False)
hr_hybrid = hit_rate_at_k(QA_EVAL, k=5, hybrid=True)
g_score   = groundedness(answer, top3)

print(f"\nRetrieval hit-rate @5:  dense={hr_dense:.0%}  hybrid={hr_hybrid:.0%}")
print(f"Groundedness (word overlap): {g_score:.1%}")
print(f"Answer snippet: {answer[:200]}...")
```

Hit-rate @k is the primary retrieval metric: did the correct document appear in the top-k? It is the number to watch when tuning chunk size, overlap, and retrieval strategy. Groundedness measures whether answer words are found in the retrieved context — a cheap faithfulness proxy before investing in an LLM judge.

---

## Stacks & alternatives

### LlamaIndex

```python
# pip install llama-index llama-index-embeddings-huggingface
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = None
li_index = VectorStoreIndex.from_documents(
    [Document(text=d["text"], doc_id=d["doc_id"]) for d in CORPUS]
)
print(li_index.as_query_engine(similarity_top_k=5).query("How does LoRA reduce GPU memory?"))
```

LlamaIndex owns chunking, embedding, indexing, and prompting — less code, faster iteration on retrieval parameters. Use it when the pipeline is standard. Use from-scratch when you need a custom eval harness, non-standard chunking, or full visibility into every intermediate tensor.

### LangChain

```python
# pip install langchain langchain-community langchain-huggingface faiss-cpu
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LC_FAISS

db        = LC_FAISS.from_texts([d["text"] for d in CORPUS], HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
retriever = db.as_retriever(search_kwargs={"k": 5})
# LCEL: retriever | format_docs | prompt | llm | StrOutputParser()
```

LangChain excels when RAG is one step in a larger agent or multi-step chain (see Lab 6). More boilerplate than LlamaIndex for pure RAG; pick it when you know you'll need chaining or routing downstream.

### Qdrant, Chroma — FAISS alternatives

```python
# Qdrant: persistent, metadata filtering, HTTP API, horizontal scale
# pip install qdrant-client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")   # or url="http://localhost:6333" for persistence
client.create_collection("rag", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.upsert("rag", points=[PointStruct(id=i, vector=normed[i].tolist(),
    payload={"text": chunks[i]["text"], "doc_id": chunks[i]["doc_id"]}) for i in range(len(chunks))])

# Chroma: simplest local start, zero config
# pip install chromadb
import chromadb
col = chromadb.Client().create_collection("rag")
col.add(documents=[c["text"] for c in chunks], ids=[c["chunk_id"] for c in chunks])
```

**FAISS**: fastest ANN, in-process, manual persistence (`faiss.write_index`). **Qdrant**: payload-filtered retrieval and production scale. **Chroma**: least setup for a local prototype. **LanceDB**: columnar Arrow-native storage, good when you co-locate structured metadata with vectors.

---

## What you built

- Chunked a corpus with a word-overlap sliding window and embedded it with `all-MiniLM-L6-v2` into a FAISS `IndexFlatIP`
- Retrieved top-k candidates via bi-encoder cosine search and reranked with a cross-encoder
- Built a context-grounded prompt and generated answers with `Qwen2.5-0.5B-Instruct` via `apply_chat_template`
- Added BM25 hybrid retrieval with linear score interpolation between normalized dense and sparse scores
- Measured retrieval hit-rate @5 and word-overlap groundedness on five labeled Q→doc pairs
- Showed equivalent pipelines in LlamaIndex and LangChain, and Qdrant and Chroma as FAISS alternatives

## Build it further

Load a real document: chunk a PDF with `pypdf`, embed it, and build a Q&A CLI that streams tokens (`TextStreamer`). Then measure: (a) hit-rate @3 with dense-only vs hybrid on 10 labeled Q→doc pairs you write yourself, (b) faithfulness with the LLM-as-judge pattern — prompt `Qwen2.5-0.5B-Instruct` to output `{"grounded": true/false, "reason": "..."}` for each answer. Log chunk count, hit-rate, and mean groundedness to wandb. Write a two-sentence finding on where hybrid retrieval helped or hurt.
