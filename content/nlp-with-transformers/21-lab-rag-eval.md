# 21 — Lab 5: RAG with a Real Eval Harness

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/nlp-with-transformers/notebooks/21-lab-rag-eval.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup.

Most RAG demos stop at "it returned a plausible answer." That is not a system — it is a screenshot. The engineering work is the eval harness: a golden set, retrieval metrics you trust, a faithfulness check, and a regression gate that fails loudly when a change makes things worse. This lab builds the full pipeline over a small inline corpus — chunk, embed, index, retrieve, rerank, generate — and wraps it in measurement at every stage. You will see hit@k and MRR move when you add a reranker, watch a lexical retriever get fooled by a keyword-stuffed distractor, and end with an assert-style gate you could drop into CI.

The RAG mechanisms here are covered in depth in [rag-agents](09-rag-agents.md); the eval discipline in [evaluation](10-evaluation.md). This lab is where both become code.

## Setup

```bash
pip install -q sentence-transformers faiss-cpu rank-bm25 transformers accelerate
```

Runs on a free Colab T4 (or CPU) in under 25 minutes; peak GPU memory stays under 3 GB. Three models download once: `all-MiniLM-L6-v2` (90 MB) for embeddings, `ms-marco-MiniLM-L-6-v2` (90 MB) for reranking, and `Qwen2.5-0.5B-Instruct` (~1 GB) for generation and judging. Seeds are fixed at 42. No API keys.

---

## Part A — Build a corpus and chunk it

A synthetic product-docs corpus defined inline: 34 short passages for a fictional support platform, "Helix." Inline means zero download flakiness and full control over the facts — which matters when you want a distractor to be *plausibly* wrong later.

```python
import random, re, collections
import numpy as np
import torch

random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

DOCS = [
    ("plan-free", "Helix Free includes 3 agent seats and 500 tickets per month. It is intended for small teams evaluating the platform."),
    ("plan-pro", "Helix Pro costs 29 dollars per agent per month billed annually. It removes the ticket cap and unlocks automation rules and CSAT surveys."),
    ("plan-enterprise", "Helix Enterprise uses custom pricing negotiated per contract. It adds SSO, a dedicated success manager, and a signed uptime SLA."),
    ("retention", "Ticket data is retained for 90 days on the Free plan, 24 months on Pro, and indefinitely on Enterprise. Deleted tickets are purged from backups within 30 days."),
    ("api-rate", "The Helix REST API allows 100 requests per minute per token on standard plans. Enterprise plans raise this to 1000 requests per minute on request."),
    ("api-auth", "API requests authenticate with a bearer token generated under Settings, API Keys. Tokens can be scoped to read-only or read-write."),
    ("languages", "Automated reply suggestions support 12 languages, including English, Spanish, German, French, Portuguese, and Japanese."),
    ("integrations", "Helix integrates natively with Slack, Microsoft Teams, and Salesforce. A one-time importer migrates existing tickets from Zendesk."),
    ("mobile", "The Helix mobile app requires iOS 15 or later and Android 10 or later. Push notifications for new tickets are on by default."),
    ("sla-uptime", "Enterprise contracts include a 99.9 percent monthly uptime SLA. Credits are issued automatically when uptime falls below the threshold."),
    ("twofa", "Two-factor authentication is available using a TOTP authenticator app or SMS codes. Admins can require 2FA for all members of a workspace."),
    ("residency", "Customer data can be pinned to the US, EU, or APAC region at workspace creation. The region cannot be changed after setup."),
    ("routing", "Incoming tickets are auto-routed to queues by a machine-learning classifier trained on your past assignments. Routing rules can override the model."),
    ("csat", "CSAT surveys are sent automatically when a ticket is marked solved. Responses use a 1 to 5 scale and feed the analytics dashboard."),
    ("webhooks", "Webhooks fire on ticket created, updated, and solved events. Payloads are signed with an HMAC secret you set per endpoint."),
    ("export", "Ticket history can be exported as CSV or JSON from the Reports page. Exports over 100000 rows are delivered as an emailed download link."),
    ("roles", "Helix defines three roles: admin, agent, and viewer. Viewers can read tickets and reports but cannot reply or change status."),
    ("kb", "The knowledge base hosts public help articles with version history. Articles can be surfaced to customers by the chat widget as suggested reading."),
    ("widget", "The chat widget is a single JavaScript snippet added before the closing body tag. It supports light and dark themes and custom brand colors."),
    ("business-hours", "Business-hours calendars pause SLA timers outside working hours. Each queue can use its own calendar and time zone."),
    ("custom-fields", "Up to 50 custom fields can be added to tickets on Pro and Enterprise. Field types include text, number, dropdown, and date."),
    ("macros", "Macros apply a set of canned actions to a ticket in one click, such as inserting a reply and changing status. Macros can be shared team-wide."),
    ("sso", "Single sign-on supports SAML 2.0 and OpenID Connect and is available only on Enterprise. Just-in-time provisioning creates agent accounts on first login."),
    ("audit-log", "The audit log records every configuration change with actor, timestamp, and before-and-after values. Logs are retained for 12 months."),
    ("attachments", "Attachments up to 25 megabytes per file can be added to tickets. Common formats are previewed inline without download."),
    ("spam", "A spam filter automatically diverts suspected junk tickets to a spam view. Messages in the spam view are deleted after 30 days."),
    ("sla-policy", "SLA policies set a first-response target and a resolution target per priority level. Breached targets trigger an alert to the queue owner."),
    ("tags", "Tickets can carry free-form tags used for filtering and reporting. A tag report shows volume and trend per tag over time."),
    ("api-pagination", "List endpoints in the API return 50 results per page by default and up to 200 with the per_page parameter. Pagination uses cursor tokens."),
    ("status-page", "A public status page shows current system health and incident history. Customers can subscribe to email or webhook status updates."),
    ("gdpr", "On request Helix will export or permanently delete all personal data for a given end user to support GDPR and CCPA requests. Deletion completes within 30 days."),
    ("onboarding", "New workspaces get a guided setup checklist covering branding, team invites, and the first integration. Setup typically takes under an hour."),
    ("reporting", "The analytics dashboard reports ticket volume, first-response time, resolution time, and CSAT. Metrics can be filtered by queue, agent, and tag."),
    ("trial", "Every paid plan starts with a 14-day free trial that requires no credit card. Unused trial data carries over if you subscribe."),
]

def chunk_text(text, max_chars=200):
    """Sentence-aware chunking: pack sentences up to a character budget."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

CHUNKS = []  # each: {"text": ..., "doc_id": ...}
for doc_id, text in DOCS:
    for c in chunk_text(text):
        CHUNKS.append({"text": c, "doc_id": doc_id})

print(f"{len(DOCS)} docs -> {len(CHUNKS)} chunks")
```

Each chunk carries its source `doc_id`. Retrieval is scored at the document level: a retrieved chunk counts as a hit if its `doc_id` matches the gold document. That is the honest unit — a user cares whether the right *source* was found, not the exact substring.

---

## Part B — Embed into FAISS, and a BM25 baseline

Dense retrieval encodes every chunk with a bi-encoder and finds nearest neighbours by cosine similarity (inner product on L2-normalised vectors). BM25 is the sparse lexical baseline — no model, just term statistics. You want both, because they fail differently: dense retrieval matches meaning, BM25 matches exact terms.

```python
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
emb = embedder.encode([c["text"] for c in CHUNKS], normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=False).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])   # inner product == cosine on normalised vectors
index.add(emb)

tokenized = [c["text"].lower().split() for c in CHUNKS]
bm25 = BM25Okapi(tokenized)

def dense_search(query, k):
    qv = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, idx = index.search(qv, k)
    return [(int(i), float(s)) for i, s in zip(idx[0], scores[0])]

def bm25_search(query, k):
    scores = bm25.get_scores(query.lower().split())
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top]

print("dense top-3:", [CHUNKS[i]["doc_id"] for i, _ in dense_search("how much does the paid plan cost", 3)])
print("bm25  top-3:", [CHUNKS[i]["doc_id"] for i, _ in bm25_search("how much does the paid plan cost", 3)])
```

`IndexFlatIP` is a brute-force exact index — correct and fast enough for tens of thousands of chunks. At millions of vectors you switch to an approximate index (HNSW, IVF-PQ); the retrieval interface stays identical.

---

## Part C — Golden set and retrieval metrics

The golden set is 15 questions, each with a short expected answer and the one document that contains it. This is the frozen artifact the whole harness rests on — every metric below is computed against it.

**hit@k** is the fraction of questions whose gold document appears in the top-k retrieved chunks. **MRR** (mean reciprocal rank) rewards ranking the gold document *high*, not just present:

$$\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the position of the first gold chunk (and $1/\text{rank}_q = 0$ if it never appears).

```python
GOLDEN = [
    {"q": "How many agent seats does the free plan include?", "answer": "3", "gold_doc": "plan-free"},
    {"q": "How much does Helix Pro cost per agent?", "answer": "29 dollars per agent per month", "gold_doc": "plan-pro"},
    {"q": "How long is ticket data retained on the Pro plan?", "answer": "24 months", "gold_doc": "retention"},
    {"q": "What is the default API rate limit?", "answer": "100 requests per minute", "gold_doc": "api-rate"},
    {"q": "How many languages do automated reply suggestions support?", "answer": "12", "gold_doc": "languages"},
    {"q": "What is the minimum iOS version for the mobile app?", "answer": "iOS 15", "gold_doc": "mobile"},
    {"q": "What uptime does the Enterprise SLA guarantee?", "answer": "99.9 percent", "gold_doc": "sla-uptime"},
    {"q": "Which regions can customer data be pinned to?", "answer": "US, EU, or APAC", "gold_doc": "residency"},
    {"q": "What is the maximum attachment file size?", "answer": "25 megabytes", "gold_doc": "attachments"},
    {"q": "How many custom fields can a ticket have?", "answer": "50", "gold_doc": "custom-fields"},
    {"q": "What roles does Helix define?", "answer": "admin, agent, and viewer", "gold_doc": "roles"},
    {"q": "Which single sign-on protocols are supported?", "answer": "SAML 2.0 and OpenID Connect", "gold_doc": "sso"},
    {"q": "How long are audit logs retained?", "answer": "12 months", "gold_doc": "audit-log"},
    {"q": "How long is the free trial?", "answer": "14 days", "gold_doc": "trial"},
    {"q": "What is the default page size for API list endpoints?", "answer": "50", "gold_doc": "api-pagination"},
]

def retrieved_docs(results, chunk_list=CHUNKS):
    return [chunk_list[i]["doc_id"] for i, _ in results]

def hit_at_k(retriever, k):
    hits = sum(g["gold_doc"] in retrieved_docs(retriever(g["q"], k)) for g in GOLDEN)
    return hits / len(GOLDEN)

def mrr(retriever, k):
    total = 0.0
    for g in GOLDEN:
        docs = retrieved_docs(retriever(g["q"], k))
        for rank, d in enumerate(docs, 1):
            if d == g["gold_doc"]:
                total += 1.0 / rank
                break
    return total / len(GOLDEN)

def rrf(query, k, pool=10, c=60):
    """Reciprocal rank fusion of dense + BM25 — the cheap hybrid."""
    d = {i: r for r, (i, _) in enumerate(dense_search(query, pool), 1)}
    s = {i: r for r, (i, _) in enumerate(bm25_search(query, pool), 1)}
    fused = collections.defaultdict(float)
    for i, r in d.items(): fused[i] += 1.0 / (c + r)
    for i, r in s.items(): fused[i] += 1.0 / (c + r)
    ranked = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(i, fused[i]) for i in ranked]

K = 3
for name, ret in [("BM25", bm25_search), ("Dense", dense_search), ("Hybrid (RRF)", rrf)]:
    print(f"{name:15s}  hit@{K}={hit_at_k(ret, K):.2f}  MRR={mrr(ret, K):.3f}")
```

Read the table, not the vibes. Dense retrieval usually beats BM25 here because the questions paraphrase the docs ("how much does Pro cost" vs "costs 29 dollars"), but BM25 wins whenever a rare exact term is the signal. Hybrid fusion takes the union and typically matches or beats the better of the two — that is the value of hybrid: it covers both failure modes for almost no cost.

---

## Part D — Rerank with a cross-encoder

The bi-encoder embeds query and chunk *independently*, then compares vectors — fast, but it never lets the query and chunk attend to each other. A cross-encoder feeds the pair jointly through a transformer and scores relevance directly. It is far more accurate and far too slow to run over the whole corpus, so the pattern is: retrieve a cheap pool of ~10 with the bi-encoder, then rerank that pool with the cross-encoder and keep the top k.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

def rerank(query, candidates, k, chunk_list=CHUNKS):
    pairs = [[query, chunk_list[i]["text"]] for i, _ in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    order = np.argsort(scores)[::-1][:k]
    return [(candidates[o][0], float(scores[o])) for o in order]

def dense_then_rerank(query, k, pool=10):
    return rerank(query, dense_search(query, pool), k)

print(f"Dense            hit@{K}={hit_at_k(dense_search, K):.2f}  MRR={mrr(dense_search, K):.3f}")
print(f"Dense + rerank   hit@{K}={hit_at_k(dense_then_rerank, K):.2f}  MRR={mrr(dense_then_rerank, K):.3f}")
```

hit@k can only improve up to what the pool already contained — reranking cannot retrieve a document the bi-encoder missed at pool size 10. Where it pays off is **MRR**: it pulls the correct chunk from rank 3 to rank 1, so the generator sees the right evidence first. On a lost-in-the-middle-prone model, that reordering alone changes answers.

---

## Part E — Generate grounded answers

Now the generation stage. A small instruct model reads the retrieved context and answers, instructed to use only what it was given and to abstain otherwise. Greedy decoding keeps it reproducible.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN = "Qwen/Qwen2.5-0.5B-Instruct"
gen_tok = AutoTokenizer.from_pretrained(GEN)
gen_model = AutoModelForCausalLM.from_pretrained(
    GEN, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

def generate(prompt, system="You are a support assistant. Answer only from the provided context.", max_new_tokens=96):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = gen_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = gen_tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=gen_tok.eos_token_id)
    return gen_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def build_context(results, chunk_list=CHUNKS):
    return "\n".join(f"[{n}] {chunk_list[i]['text']}" for n, (i, _) in enumerate(results, 1))

def rag_answer(query, retriever, k=3, chunk_list=CHUNKS):
    results = retriever(query, k)
    ctx = build_context(results, chunk_list)
    prompt = (f"Context:\n{ctx}\n\nQuestion: {query}\n"
              f"Answer in one sentence using only the context. If the answer is not in the context, say 'I don't know.'")
    return generate(prompt), results

ans, _ = rag_answer("How much does Helix Pro cost per agent?", dense_then_rerank)
print(ans)
```

---

## Part F — Faithfulness with an LLM judge (and its limits)

Retrieval metrics tell you the evidence was present; they say nothing about whether the *answer* is grounded in it. Faithfulness asks: is every claim in the answer supported by the retrieved context? The scalable proxy is LLM-as-judge — here, the same 0.5B model grading its own outputs.

Be honest about what this is. A 0.5B judge is weak: it agrees with itself, misses subtle contradictions, and is sensitive to phrasing. Its verdicts correlate loosely with truth, not tightly. Treat this as a **smoke detector**, not a scale — it catches gross ungroundedness (the answer invents a number the context never mentions), and that is genuinely useful as a regression signal. Production faithfulness scoring uses a stronger judge, multiple samples, or human review on a sampled slice.

```python
def judge_faithful(question, context, answer):
    prompt = (f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
              f"Is every claim in the Answer directly supported by the Context? "
              f"Reply with exactly one word: YES or NO.")
    verdict = generate(prompt, system="You are a strict grader.", max_new_tokens=4)
    return verdict.strip().upper().startswith("YES")

def faithfulness_rate(retriever, k=3):
    faithful = 0
    for g in GOLDEN:
        ans, results = rag_answer(g["q"], retriever, k)
        faithful += judge_faithful(g["q"], build_context(results), ans)
    return faithful / len(GOLDEN)

print(f"faithfulness (dense+rerank): {faithfulness_rate(dense_then_rerank):.2f}")
```

---

## Part G — A failure mode, and the reranker fixing it

Now inject a distractor: a passage stuffed with the query's keywords that answers a *different* question. It talks about the API "default" and the number "100" — but 100 *seconds of timeout*, not 100 requests per minute. BM25 counts terms, so the keyword-stuffed distractor outranks the real answer, and the generator dutifully reports the wrong number. The cross-encoder reads the pair jointly, recognises the distractor does not answer *this* question, and demotes it.

```python
distractor = {"text": ("The API default request timeout is 100 seconds. API rate limit default settings and "
                       "the API default retry policy are documented in the API changelog per API token."),
              "doc_id": "api-changelog"}

aug_chunks = CHUNKS + [distractor]
aug_emb = np.vstack([emb, embedder.encode([distractor["text"]], normalize_embeddings=True,
                                           convert_to_numpy=True).astype("float32")])
aug_index = faiss.IndexFlatIP(aug_emb.shape[1]); aug_index.add(aug_emb)
aug_bm25 = BM25Okapi([c["text"].lower().split() for c in aug_chunks])

def aug_bm25_search(q, k):
    s = aug_bm25.get_scores(q.lower().split()); top = np.argsort(s)[::-1][:k]
    return [(int(i), float(s[i])) for i in top]

def aug_dense_then_rerank(q, k, pool=10):
    qv = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    sc, idx = aug_index.search(qv, pool)
    cands = [(int(i), float(s)) for i, s in zip(idx[0], sc[0])]
    return rerank(q, cands, k, chunk_list=aug_chunks)

q = "What is the default API rate limit?"

bad = aug_bm25_search(q, 3)
print("BM25 top doc:", aug_chunks[bad[0][0]]["doc_id"])
print("BM25 answer:", generate(f"Context:\n{build_context(bad, aug_chunks)}\n\nQuestion: {q}\n"
                                f"Answer in one sentence using only the context."))

good = aug_dense_then_rerank(q, 3)
print("\nReranked top doc:", aug_chunks[good[0][0]]["doc_id"])
print("Reranked answer:", generate(f"Context:\n{build_context(good, aug_chunks)}\n\nQuestion: {q}\n"
                                    f"Answer in one sentence using only the context."))
```

You should see BM25 surface `api-changelog` first and the answer come back with "100 seconds" (or a confused blend), while dense-plus-rerank surfaces `api-rate` first and the answer corrects to "100 requests per minute." This is exactly why a lexical-only retriever is dangerous in production: an adversarial or simply badly-written document with the right keywords poisons the context, and the model has no way to know.

---

## Part H — A golden-set regression gate

Wrap the whole thing in a function that recomputes the metrics and *asserts* they clear a floor. Run it in CI on every change to chunking, embeddings, the prompt, or the model. A change that quietly drops hit@3 from 0.93 to 0.60 stops being invisible — the build goes red.

```python
def regression_gate(min_hit=0.80, min_mrr=0.75, min_faith=0.50, k=3):
    hit = hit_at_k(dense_then_rerank, k)
    m = mrr(dense_then_rerank, k)
    faith = faithfulness_rate(dense_then_rerank, k)
    print(f"hit@{k}={hit:.2f} (floor {min_hit})  MRR={m:.3f} (floor {min_mrr})  "
          f"faithfulness={faith:.2f} (floor {min_faith})")
    assert hit >= min_hit, f"retrieval regression: hit@{k}={hit:.2f} < {min_hit}"
    assert m >= min_mrr, f"ranking regression: MRR={m:.3f} < {min_mrr}"
    assert faith >= min_faith, f"faithfulness regression: {faith:.2f} < {min_faith}"
    print("REGRESSION GATE PASSED")

regression_gate()
```

The floors are set below the current numbers with headroom, not at them — a gate pinned to the exact current score fails on ordinary run-to-run noise and gets disabled within a week. Set floors you are willing to defend, and raise them deliberately when you have a real improvement to lock in.

---

## What you built

- A chunked, document-tagged corpus with a frozen 15-question golden set.
- Three retrievers — dense (FAISS), sparse (BM25), and a reciprocal-rank-fusion hybrid — scored on hit@k and MRR.
- A cross-encoder reranker, with the before/after MRR gain measured, not asserted.
- A grounded generator on a small local instruct model and an LLM-as-judge faithfulness check, with its weakness stated plainly.
- A reproducible failure mode (keyword distractor) and its fix (reranking), then an assert-style regression gate you could ship to CI.

## Exercises

1. Add five adversarial questions to the golden set whose answers are *not* in the corpus (e.g., "What is the Helix phone-support number?"). Measure the abstention rate — how often does the model correctly say "I don't know"? Report it as a separate metric alongside faithfulness.
2. Sweep the retrieval pool size (5, 10, 20) feeding the reranker and plot hit@3 and MRR against it. Where does adding candidates stop helping, and what does that cost in reranker latency?
3. Replace the fusion weighting in `rrf` with a weighted sum of normalised dense and BM25 scores. Find a weight that beats plain RRF on this golden set, then argue why it may not transfer to a different corpus.
4. Swap the judge model for `Qwen2.5-1.5B-Instruct` and re-run faithfulness on the same answers. Report how many verdicts flip, and which direction they flip — this is a concrete measurement of judge reliability.
5. Add a second distractor targeting a different golden question and confirm the reranker still recovers. Then construct a distractor the reranker *cannot* fix, and explain what property made it robust to reranking.

## What interviews ask here

- Why retrieve a pool with a bi-encoder and then rerank with a cross-encoder, instead of using one of them for everything? (Cost vs accuracy: independent embeddings are cheap and cacheable but coarse; joint attention is precise but O(pool) forward passes per query.)
- What does hit@k miss that MRR captures, and when do you care about the difference? (hit@k ignores position; MRR rewards ranking gold first, which matters when the generator is position-sensitive or context is truncated.)
- How would you evaluate faithfulness without a golden answer for every query? (LLM-as-judge over retrieved context, its biases — self-preference, length, position — and why you sample human review on top.)
- When does BM25 beat a dense retriever, and why keep it around? (Rare exact tokens: IDs, error codes, names; hybrid fusion covers the lexical gap dense embeddings smooth over.)
- What goes in a RAG regression gate, and why set floors below current metrics rather than at them? (Retrieval + faithfulness metrics on a frozen golden set; floors with headroom survive run-to-run noise instead of being disabled after the first flaky failure.)
