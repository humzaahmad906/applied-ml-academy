# 09 — RAG and Agents: Grounding Models in the World

A pretrained model knows only what was in its training data, frozen at some cutoff, blended
into weights you cannot inspect or edit. That is a problem the moment you deploy. It cannot
answer questions about your company's docs, last week's incident, or a customer's account. It
will confidently invent a plausible answer instead — the failure mode that kills most first LLM
projects. **Retrieval-augmented generation (RAG)** fixes this by fetching relevant text at query
time and putting it in the prompt, so the model reasons over evidence instead of memory.
**Agents** go further: they let the model *act* — call tools, run searches, take multiple steps —
to gather what it needs. This module is the mechanism of both, and their characteristic failure
modes, which is what you will actually spend your time debugging. For the engineering build-out —
chunking pipelines, vector-DB ops, production agent loops — see the sibling
[RAG](../vlm-guide/05_rag.md) and [agents](../vlm-guide/06_agents.md) guides; here we own the
*why* and the *what-breaks*.

## Why retrieve at all

Three reasons, and it helps to keep them separate because they imply different designs.

- **Knowledge cutoff.** Weights are stale the day training ends. Retrieval injects fresh text
  without retraining. Cheaper and faster than continued pretraining, and auditable.
- **Grounding.** Even for facts in-distribution, a retrieved passage lets you *cite* a source and
  lets the model quote rather than confabulate. This is the single biggest hallucination
  reduction available to you (see [risks](15-risks-and-safety.md) for the taxonomy).
- **Private data.** Your wiki, contracts, and tickets were never in anyone's pretraining corpus
  and never should be. Retrieval is how a general model uses proprietary knowledge without that
  knowledge leaking into a shared checkpoint.

The alternative to retrieval is fine-tuning the knowledge in (see
[prompting-peft](08-prompting-peft.md) for the decision framework). The rule of thumb: fine-tune
to teach *behavior and format*, retrieve to supply *facts*. Facts change; behavior is stable.
You do not fine-tune every time a doc is edited.

## Sparse vs dense retrieval

The retrieval half of RAG is an information-retrieval problem older than transformers. Two
families, and modern systems use both.

**Sparse (lexical) retrieval — BM25.** Represent each document as a bag of terms and score it
against the query by term overlap, weighted. BM25 is the canonical scorer:

$$
\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot
\frac{f(t, d) \,(k_1 + 1)}{f(t, d) + k_1 \,(1 - b + b \cdot |d| / \overline{|d|})}
$$

where $f(t,d)$ is the term frequency in the document, $|d|$ its length, $\overline{|d|}$ the
average length, and $\text{IDF}(t)$ down-weights common terms. The $k_1$ term (typically ~1.2–2.0)
saturates term frequency so a word appearing 50 times does not score 50× a word appearing once;
the $b$ term (~0.75) normalizes for document length so long docs do not win by sheer size. BM25 is
a strong, zero-training baseline that never quietly fails: it matches exact tokens, so product
codes, error strings, names, and rare acronyms are retrieved reliably. Its weakness is the
**vocabulary mismatch** — a query for "car" will not match a document that only says "automobile."

**Dense retrieval.** Encode query and document into vectors with a trained embedding model and
score by similarity (cosine or dot product). Because the encoder learned semantics, "car" and
"automobile" land near each other. Dense retrieval handles paraphrase and synonymy that BM25 is
blind to. Its weakness is the mirror image: it can miss an exact token match, and it fails
silently — a bad embedding returns *something*, just the wrong thing, with a confident score.

The practical answer is **hybrid**: run both, and fuse the rankings. Reciprocal rank fusion (RRF)
is the standard, parameter-light fuser — for each document, sum $1/(k + \text{rank}_i)$ over each
retriever $i$ (with $k \approx 60$), and re-sort. Hybrid beats either alone on almost every real
corpus because queries are a mix of "find this exact string" and "find this idea."

## Embeddings and vector search

The embedding model is a bi-encoder (see [transfer-learning-tasks](06-transfer-learning-tasks.md)
for bi- vs cross-encoders): it maps a chunk of text to a fixed vector, independent of the query, so
you can embed the whole corpus once, offline, and store the vectors. At query time you embed only
the query and search for nearest neighbors. In 2026 the workhorses are models like the E5, BGE,
GTE, and Nomic families, plus API embeddings; pick by your language mix and the MTEB leaderboard,
but do not over-index on a half-point of average score — retrieval quality on *your* corpus is what
matters, and it often disagrees with the leaderboard.

Exact nearest-neighbor search is O(N) per query and does not scale past a few hundred thousand
vectors. Production uses **approximate nearest neighbor (ANN)**, and the dominant algorithm is
**HNSW** (Hierarchical Navigable Small World): a multi-layer proximity graph where the top layers
are sparse "express lanes" for coarse jumps and the bottom layer is dense for fine search. A query
greedily walks the graph toward closer neighbors, descending layers. It gives roughly logarithmic
search time at ~95–99% recall, tunable via `M` (edges per node) and `efSearch` (candidate breadth) —
higher means better recall and slower queries. This is what FAISS, Qdrant, Weaviate, pgvector, and
friends implement under the hood.

## The RAG pipeline, end to end

A production RAG system is a pipeline, and each stage has a failure mode. In order:

1. **Chunking.** Documents are split into passages, because you retrieve and fit *chunks*, not
   whole files, into the context window. Chunk too large and you dilute the signal and waste
   tokens; too small and you sever the context a passage needs to make sense. Sensible defaults
   are ~200–500 tokens with ~10–20% overlap, split on semantic boundaries (headings, paragraphs)
   rather than blindly every N characters. Chunking is boring and it is where most RAG quality is
   won or lost — a table split down the middle or a heading orphaned from its section is a
   retrieval miss you will never see in the logs.
2. **Indexing.** Embed every chunk and build the ANN index (plus a BM25 index for hybrid). Done
   offline; re-run on doc changes. Store the source metadata (URL, section, timestamp) with each
   vector — you need it for citations and for filtering.
3. **Retrieval.** Embed the query, run dense + sparse, fuse. Return top-`k` candidates, usually
   `k = 20–50` at this stage — deliberately over-fetch, because the next stage will prune.
4. **Reranking.** Pass the query and each candidate together through a **cross-encoder** — a model
   that reads query and passage *jointly* and scores relevance directly. It is far more accurate
   than bi-encoder similarity (it can attend across the pair) but too slow to run over the whole
   corpus, which is exactly why you only rerank the top-`k` the retriever already narrowed. Keep
   the top ~3–8 after reranking. Reranking is the highest-leverage single addition to a naive RAG
   system; skipping it is the most common reason a demo works and production does not.
5. **Context assembly.** Format the surviving chunks into the prompt with their sources, add the
   instruction ("answer only from the context; if it is not there, say so"), and generate. Order
   matters (see lost-in-the-middle below). Instructing the model to cite the chunk it used both
   improves faithfulness and gives you something to audit.

## RAG failure modes

When RAG is wrong, it is almost always one of these, and diagnosing *which* is the job.

- **Retrieval miss.** The answer is in the corpus but the right chunk was not retrieved — bad
  chunking, vocabulary mismatch, or a query phrased unlike the document. The generator then either
  confabulates or (if well-instructed) abstains. Measured by retrieval **recall@k** and **hit@k**;
  if hit@k is low, no amount of prompt engineering on the generator will help — fix retrieval.
- **Lost-in-the-middle.** Even when the right chunk *is* in context, models attend most strongly to
  the beginning and end of a long context and neglect the middle (Liu et al., 2023). A relevant
  passage buried at position 10 of 20 can be effectively invisible. Mitigate by retrieving fewer,
  better chunks (reranking) and placing the strongest evidence first or last.
- **Conflicting evidence.** Two retrieved chunks disagree — an old policy and its update, two
  regions' rules. The model has no inherent way to know which wins and often silently blends or
  picks the wrong one. Mitigate with recency/authority metadata, dedup, and prompting the model to
  surface the conflict rather than resolve it invisibly.
- **Right retrieval, wrong generation.** Retrieval succeeded but the model ignored the context and
  answered from parametric memory anyway, or over-summarized. Caught by faithfulness eval (below).

The discipline is to **measure retrieval and generation separately**. A single end-to-end accuracy
number cannot tell you whether to fix the retriever or the prompt, and those are entirely different
fixes.

## From retrieval to action: agents

RAG does one retrieval and generates. An **agent** runs a loop: the model decides what to do, does
it, observes the result, and decides again — until it has an answer. The canonical scaffold is
**ReAct** (Reason + Act; Yao et al., 2022), which interleaves reasoning traces with actions:

```
Thought: I need the customer's current plan before I can check eligibility.
Action: get_account(customer_id="4417")
Observation: {"plan": "pro", "renewed": "2026-05-01"}
Thought: Pro plan is eligible. Now check the refund window.
Action: get_policy(topic="refund_window")
Observation: "Refunds within 30 days of renewal."
Thought: Renewed 2026-05-01, today is 2026-07-13 — 73 days, outside the window.
Answer: This renewal is outside the 30-day refund window, so it is not eligible.
```

Each `Thought` is generated text (chain-of-thought, see [reasoning](11-reasoning.md)) that lets the
model plan in the open; each `Action` is a tool call the runtime executes; each `Observation` is the
result, fed back into context for the next step. The loop is just the model generating,
the runtime executing and appending, and the model generating again.

## Tool-calling mechanics

Modern models are post-trained (see [post-training](07-post-training.md)) to emit **structured tool
calls**, not free-text you have to parse. You supply each tool as a JSON schema — name, description,
typed parameters:

```python
tools = [{
    "name": "get_account",
    "description": "Look up a customer's account by ID. Use when you need plan or billing state.",
    "input_schema": {
        "type": "object",
        "properties": {"customer_id": {"type": "string", "description": "Numeric account ID"}},
        "required": ["customer_id"],
    },
}]
```

The model, when it decides to act, emits a structured call (`{"name": "get_account", "input":
{"customer_id": "4417"}}`) instead of prose. Your runtime executes the real function, appends the
result as an observation, and continues the loop. The **description fields are the interface**: the
model chooses tools by reading them, so a vague description is the most common cause of a model
calling the wrong tool or hallucinating arguments. When steps are independent — fetching three
users' records — capable models emit **parallel tool calls** in a single turn, and you execute them
concurrently, which cuts latency dramatically over a serial loop. Planning ranges from implicit
(the model figures out order on the fly, as in ReAct) to an explicit plan-then-execute phase for
long tasks; explicit planning helps on tasks with many steps where drifting off-course is costly.

## Why agents are fragile: error compounding

The defining risk of multi-step agents is that **errors multiply**. If each step succeeds
independently with probability $p$, an $n$-step task succeeds with roughly $p^n$. At a very good
per-step reliability of $p = 0.95$, a 10-step task lands at $0.95^{10} \approx 0.60$; at 20 steps,
$\approx 0.36$. And steps are not truly independent — one wrong observation poisons every
subsequent reasoning step, so real degradation is often worse than the geometric floor. This is why
agent demos dazzle on 3-step tasks and disappoint on 15-step ones. The engineering response is to
*shorten the critical path* (fewer, more powerful tools beat many primitive ones), add
verification/retry at each step, checkpoint so a failure does not restart the whole task, and cap
the loop so a confused agent cannot burn your budget spinning. Treat step count as a reliability
budget you spend deliberately, not a free dimension.

## Agentic RAG

Fold the two together and you get **agentic RAG**: instead of one fixed retrieval, the model treats
search as a *tool* and decides when and what to retrieve. It can reformulate a bad query, decompose
a multi-hop question ("which of our EU customers signed before the policy change?") into sub-queries,
retrieve iteratively, and stop when it has enough. This handles questions naive single-shot RAG
cannot — anything requiring more than one lookup or a query the user did not phrase retrievably — at
the cost of the latency, expense, and compounding-error risk of the agent loop. Reach for it when
questions are genuinely multi-hop; a single well-built retrieval + rerank still wins on cost and
reliability for the common single-fact case.

## Evaluating RAG

You cannot improve what you do not measure, and RAG has two things to measure separately.
**Retrieval** metrics are classic IR: recall@k, hit@k, MRR, nDCG — does the right chunk come back,
and how high. **Generation** metrics center on **faithfulness** (is every claim in the answer
supported by the retrieved context — the anti-hallucination metric) and **answer relevance** (does
it actually address the question). Faithfulness is typically scored with an LLM-as-judge that checks
each answer sentence against the context, and frameworks like RAGAS package these. The deeper
treatment of judges, their biases, and building a real eval harness is [evaluation](10-evaluation.md);
the one rule to carry from here is that a good RAG eval reports retrieval and generation quality on
their own axes, because they fail for different reasons and get fixed in different places.

## MCP: the tool-integration standard

Every agent needs tools, and before 2025 every team wired every tool to every model by hand — an
N×M integration mess. The **Model Context Protocol (MCP)**, introduced by Anthropic in late 2024 and
now broadly adopted across the ecosystem, standardizes that connection: an open client-server
protocol where a "server" exposes tools, resources, and prompts through a uniform interface, and any
MCP-capable client (an IDE, a chat app, an agent runtime) can consume them without bespoke glue. In
practice it means the Slack, GitHub, Postgres, or filesystem connector you write once works with any
model that speaks MCP, turning tool integration from N×M into N+M. For an agent engineer in 2026 it
is the default way to expose capabilities to a model rather than reinventing a tool-calling bridge
per project.

## What interviews ask here

- When do you use RAG vs fine-tuning? — Retrieve for facts (change often, need citations, private);
  fine-tune for behavior/format. They compose.
- BM25 vs dense retrieval — trade-offs? — Sparse nails exact tokens and never fails silently; dense
  handles paraphrase/synonymy but can miss exact matches; hybrid + RRF beats either.
- Why add a reranker if you already have embeddings? — Cross-encoder reads query+passage jointly for
  far better relevance; too slow for the whole corpus, so rerank only the retriever's top-k.
- Your RAG gives wrong answers — how do you debug? — Split retrieval (hit@k) from generation
  (faithfulness); a low hit@k means fix retrieval, prompt tweaks won't help.
- What is lost-in-the-middle and how do you mitigate it? — Models neglect mid-context evidence;
  retrieve fewer/better chunks via reranking and place strongest evidence first/last.
- Why do long agent tasks fail? — Per-step errors compound (~$p^n$); shorten the path, verify per
  step, checkpoint, cap the loop.
- What is MCP? — Open protocol standardizing tool/resource exposure to models; turns N×M tool
  integration into N+M.

## Where this shows up on the job

- Building an internal knowledge assistant over company docs — the single most common applied-LLM
  project, and it lives or dies on chunking + reranking, not the generator model choice.
- Debugging "the bot made something up": tracing whether retrieval missed or the model ignored
  context, and instrumenting faithfulness so you catch it before users do.
- Standing up an agent that touches real systems (tickets, databases, code) where tool schemas,
  parallel calls, and loop caps decide whether it is reliable enough to ship.
- Cost and latency reviews: deciding when single-shot RAG suffices versus paying for an agentic loop,
  and sizing `k`, rerank depth, and step budgets against a latency SLA.
