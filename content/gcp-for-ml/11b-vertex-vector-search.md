# 11b — Vertex AI Vector Search

Module 11 ended with retrieval-augmented generation: to ground Gemini in *your* data you first turn documents into embeddings, then find the ones nearest to a user's question and inject them into the prompt. That "find the nearest embeddings" step is the job of a **vector database**, and on Google Cloud the managed, purpose-built option is **Vertex AI Vector Search** (formerly **Matching Engine**). It is the piece that was implicit in the RAG Engine of module 11 — this module opens it up so you can build and operate it yourself.

> **Naming note (2026).** Google has been consolidating the Vertex AI umbrella under new branding — recent docs refer to the **Gemini Enterprise Agent Platform**, and the next-generation API surface (single **Collection** objects that hold vectors, metadata, and content together) is being marketed as **Vector Search 2.0 / "Agent Retrieval."** The concepts and the classic `google-cloud-aiplatform` workflow below are unchanged and still current, but the console labels and top-level product name are in flux — **check current docs** for the exact name you see.

## When you need a vector database

A vector database exists to answer one question fast, at scale: *given this query vector, which of my millions of stored vectors are closest?* You need one whenever your retrieval key is **meaning**, not an exact value:

- **RAG** — retrieve the passages most relevant to a question before calling Gemini (module 11).
- **Semantic search** — "find documents about chargeback disputes" matching on meaning, not keywords.
- **Recommendations / similarity** — "items like this one," near-duplicate detection, dedup.

A regular index (BigQuery, a B-tree) finds exact matches. A vector index finds *nearest* matches in high-dimensional space, and doing that exactly over millions of vectors is too slow — so vector databases use **approximate nearest neighbor (ANN)** search, trading a sliver of recall for orders-of-magnitude lower latency.

## Vector Search concepts

Vertex AI Vector Search is Google's productized version of **ScaNN** (Scalable Nearest Neighbors), the same research library that powers Google Search and YouTube. Four objects make up the mental model:

- **Index** — the built data structure holding your vectors. You create it from embedding files in Cloud Storage (batch) or by streaming vectors in.
- **Index endpoint** — the always-on serving resource that hosts one or more deployed indexes. Can be **public** or inside your **VPC** (Private Service Connect / VPC peering).
- **Deployed index** — an index actually loaded onto an endpoint's machines and ready to serve queries. Creating an index does not serve it; you must deploy it.
- **The algorithm** — Vector Search offers two index types:
  - **tree-AH** — ScaNN's tree + Asymmetric Hashing. A partitioning "tree" narrows the search to a few clusters; "AH" quantizes each vector into compact codes for fast distance approximation. This is the production choice: low latency at high recall on large corpora.
  - **brute force** — exact linear scan, 100% recall, high latency. Not for production; you build a small brute-force index to measure the *ground-truth* recall of your tree-AH index.

## The workflow

Four stages: **embed → build index → deploy → query.** Note the SDK split introduced in module 11 — you generate embeddings with the **`google-genai`** client, but you create and manage indexes and endpoints with the **`google-cloud-aiplatform`** SDK (`from google.cloud import aiplatform`), which is *not* deprecated.

### 1. Generate embeddings

Embed your corpus with a Vertex text-embedding model (`text-embedding-005` / `gemini-embedding`), using `task_type="RETRIEVAL_DOCUMENT"` for the corpus and `"RETRIEVAL_QUERY"` for queries. For a whole corpus use a **batch** embedding job, not a loop of online calls. Vector Search ingests embeddings as JSONL in Cloud Storage, one object per line with an `id` and a `embedding` (feature vector) field:

```json
{"id": "doc-001", "embedding": [0.021, -0.145, 0.318, ...]}
{"id": "doc-002", "embedding": [0.114, 0.002, -0.271, ...]}
```

### 2. Build the index

```python
from google.cloud import aiplatform

aiplatform.init(project="myco-fraud-dev", location="us-central1",
                staging_bucket="gs://myco-fraud-vectors")

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="fraud-policy-index",
    contents_delta_uri="gs://myco-fraud-vectors/embeddings/",  # folder of JSONL
    dimensions=768,                       # must match your embedding model
    approximate_neighbors_count=150,      # required for tree-AH
    distance_measure_type="DOT_PRODUCT_DISTANCE",  # default; see tuning below
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=7,
    index_update_method="STREAM_UPDATE",  # or "BATCH_UPDATE"
)
```

`BATCH_UPDATE` rebuilds from files in Cloud Storage — right for nightly refreshes of a large corpus. `STREAM_UPDATE` lets you `upsert_datapoints(...)` and remove individual vectors that become searchable within seconds — right for a live corpus (new documents, deletions). **Gotcha:** the update method is fixed at creation. You cannot convert a batch index to streaming; you must create a new index. Building the index takes minutes to hours depending on size, and it runs asynchronously.

For the ground-truth companion, `aiplatform.MatchingEngineIndex.create_brute_force_index(...)` takes the same embeddings with no algorithm tuning fields.

### 3. Deploy to an endpoint

```python
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="fraud-policy-endpoint",
    public_endpoint_enabled=True,          # or network=VPC_NETWORK for private
)

endpoint.deploy_index(
    index=index,
    deployed_index_id="fraud_policy_v1",   # letters/digits/underscore only
    machine_type="e2-standard-16",
    min_replica_count=1,
    max_replica_count=3,                   # autoscaling ceiling
)
```

Deployment also takes many minutes. One endpoint can host several deployed indexes (e.g. blue/green versions) sharing its network.

### 4. Query — find_neighbors

For a **public** endpoint, query with `find_neighbors`; embed the user's question first (as a `RETRIEVAL_QUERY`) and pass the raw vector:

```python
neighbors = endpoint.find_neighbors(
    deployed_index_id="fraud_policy_v1",
    queries=[query_vector],   # list of embeddings; each is a list[float]
    num_neighbors=10,
)
for n in neighbors[0]:
    print(n.id, n.distance)   # look up the source text by id in your own store
```

**Gotcha:** `find_neighbors` is the method for **public** endpoints. For a **VPC / Private Service Connect** endpoint the equivalent is `endpoint.match(...)`. Both accept metadata **filtering** (via `Namespace` restricts) and support **hybrid** dense+sparse queries. Vector Search returns only ids and distances — you keep the mapping from id to document text/metadata in your own store (Cloud Storage, BigQuery, or a database).

You can hit the same public endpoint with plain REST, authenticating with a short-lived token (no API key):

```bash
curl -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  "https://PUBLIC_ENDPOINT_DOMAIN/v1/projects/PROJ/locations/us-central1/indexEndpoints/ENDPOINT_ID:findNeighbors" \
  -d '{"deployed_index_id": "fraud_policy_v1", "queries": [{"datapoint": {"feature_vector": [0.02, -0.14, ...]}, "neighbor_count": 10}]}'
```

## Tuning: recall, latency, and distance

- **Approximate vs exact.** tree-AH is approximate. Measure **recall** by comparing its results against a brute-force index on a fixed query set. Target **recall ≥ 0.95** for most production use. If you are below that, raise `leaf_nodes_to_search_percent` (search more clusters) or `approximate_neighbors_count` — both improve recall at the cost of latency. This is the core knob: **recall vs latency.**
- **Distance measure.** `DOT_PRODUCT_DISTANCE` (default), `COSINE_DISTANCE`, `SQUARED_L2_DISTANCE`, `L1_DISTANCE`. It must match how your embedding model was trained — the text-embedding models are tuned for dot-product / cosine. Google recommends **`DOT_PRODUCT_DISTANCE` combined with `UNIT_L2_NORM`** normalization rather than `COSINE_DISTANCE` directly (they are equivalent, and the former is faster).
- **Shard size** (`SHARD_SIZE_SMALL` 2 GiB / `MEDIUM` 20 GiB / `LARGE` 50 GiB) and **machine type** determine how your vectors are partitioned across nodes.

## Cost: the always-on gotcha

This is the single most important operational fact about Vector Search: **a deployed index endpoint bills per node-hour, continuously, whether or not anyone queries it.** Unlike a Gemini API call (pay-per-token) or even a scale-to-zero Cloud Run service, index-serving nodes run 24/7. Two cost components:

- **Index build** — a one-time charge roughly proportional to data size (on the order of `examples × dimensions × 4 bytes × a per-GB rate`).
- **Serving** — `machine_type × replica_count × node-hours`, running as long as the index is deployed, so even a tiny index carries a fixed monthly floor of hundreds of dollars if left up.

The practical rules: **undeploy** indexes you are not using (`endpoint.undeploy_index(...)`), keep `min_replica_count` low in dev, and never leave an experiment endpoint running over a weekend. Exact rates vary by machine type and region — **check current pricing**. Watch **quotas** too: index/endpoint counts and online query QPS are quota-limited per project and region, and a new project may need a quota bump before a large deployment. IAM: index and endpoint operations require **`roles/aiplatform.user`** (or finer Vector Search permissions), and a VPC endpoint needs the networking set up in advance.

## The managed alternative: pgvector on AlloyDB / Cloud SQL

You do not always need a dedicated vector service. If your data already lives in **PostgreSQL** — or you want vectors *alongside* transactional rows so you can filter with ordinary SQL and join to other tables — use the **pgvector** extension on **Cloud SQL for PostgreSQL** or **AlloyDB**. AlloyDB adds Google's **ScaNN index** for Postgres (plus HNSW and IVFFlat) and an `embedding()` SQL function that calls Vertex embedding models inline:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS alloydb_scann;      -- AlloyDB only

-- ScaNN index for fast ANN (num_leaves ≈ sqrt(row count))
CREATE INDEX ON docs USING scann (embedding vector_cosine_ops)
  WITH (num_leaves = 1000, max_num_levels = 2);

-- Query: nearest neighbors to an embedding generated in-database
SELECT id, content
FROM docs
WHERE category = 'chargeback-policy'                -- ordinary SQL filter
ORDER BY embedding <=> CAST(
  embedding('gemini-embedding-001', 'card-not-present dispute rules') AS vector)
LIMIT 5;
```

The pgvector operators are `<=>` (cosine), `<->` (L2), `<#>` (inner product).

**When to pick pgvector.** Choose it for smaller-to-medium scale (thousands to low tens of millions of vectors), when your vectors belong next to relational/transactional data, when rich SQL filtering matters, and when you want to avoid a second always-on serving system — you are already paying for the database. AlloyDB's ScaNN scales further (documented to billions of vectors); HNSW suits higher-dimensional data that fits in memory. **Choose dedicated Vertex AI Vector Search** for very large corpora, the lowest tail latency at high QPS, and streaming upserts at scale — accepting the always-on endpoint cost as the price.

## How it slots into a RAG pipeline

Tie this back to module 11. A production RAG flow on GCP is: embed your corpus (Vertex embeddings) → **index it in Vector Search** (or pgvector) → at query time, embed the question, `find_neighbors` to retrieve the top-k passage ids → fetch the passage text → inject it into a **Gemini** prompt via `google-genai`. The **Vertex AI RAG Engine** automates exactly this and can use Vector Search as its managed backend (alongside Pinecone, Weaviate, or Feature Store), so module 11's RAG Engine was orchestrating the very index you now know how to build and tune by hand — the right level to operate at when you need control over recall, cost, and freshness.

## Key takeaways

- A **vector database** answers "nearest embeddings" queries fast via **approximate nearest neighbor (ANN)** search — the retrieval half of RAG, semantic search, and recommendations.
- **Vertex AI Vector Search** (formerly Matching Engine) is ScaNN-as-a-service. The workflow is **embed → build index → deploy to an endpoint → query with `find_neighbors`** (public) or `match` (VPC). Use `google-cloud-aiplatform` for indexes/endpoints, `google-genai` for the embeddings.
- Prefer **tree-AH** for production; keep a small **brute-force** index to measure recall. Tune **recall vs latency** with `leaf_nodes_to_search_percent` / `approximate_neighbors_count`, target **recall ≥ 0.95**, and use `DOT_PRODUCT_DISTANCE` + `UNIT_L2_NORM`. Pick `STREAM_UPDATE` vs `BATCH_UPDATE` at creation — it cannot be changed later.
- **The endpoint is always-on and always billing.** Undeploy idle indexes, keep replica counts low in dev, mind per-region quotas and `aiplatform.user` IAM. **Check current pricing** for exact node-hour rates.
- For smaller scale or vectors alongside transactional data, use **pgvector on AlloyDB / Cloud SQL** (AlloyDB adds a ScaNN index and in-SQL `embedding()`), avoiding a second always-on system.

## CLI / SDK cheat-sheet

```bash
# --- google-cloud-aiplatform SDK (pip install google-cloud-aiplatform) ---
#   aiplatform.init(project=..., location="us-central1", staging_bucket="gs://...")
#   aiplatform.MatchingEngineIndex.create_tree_ah_index(
#       display_name=..., contents_delta_uri="gs://.../embeddings/",
#       dimensions=768, approximate_neighbors_count=150,
#       distance_measure_type="DOT_PRODUCT_DISTANCE",
#       index_update_method="STREAM_UPDATE")           # or BATCH_UPDATE (fixed at create)
#   aiplatform.MatchingEngineIndex.create_brute_force_index(...)   # ground-truth recall
#   ep = aiplatform.MatchingEngineIndexEndpoint.create(display_name=..., public_endpoint_enabled=True)
#   ep.deploy_index(index=..., deployed_index_id="v1", machine_type="e2-standard-16",
#                   min_replica_count=1, max_replica_count=3)
#   ep.find_neighbors(deployed_index_id="v1", queries=[vec], num_neighbors=10)   # public
#   ep.match(...)                                       # VPC endpoint
#   ep.undeploy_index(deployed_index_id="v1")           # STOP THE BILLING

# --- Raw findNeighbors REST (public endpoint) ---
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://PUBLIC_ENDPOINT_DOMAIN/v1/projects/PROJ/locations/us-central1/indexEndpoints/EP_ID:findNeighbors" \
  -d '{"deployed_index_id":"v1","queries":[{"datapoint":{"feature_vector":[...]},"neighbor_count":10}]}'

# --- pgvector alternative (AlloyDB / Cloud SQL for PostgreSQL) ---
#   CREATE EXTENSION vector;  CREATE EXTENSION alloydb_scann;      # AlloyDB
#   CREATE INDEX ON docs USING scann (embedding vector_cosine_ops) WITH (num_leaves=1000);
#   SELECT id FROM docs ORDER BY embedding <=> CAST(embedding('gemini-embedding-001','q') AS vector) LIMIT 5;
```

## Try it

Build a retrieval layer end to end and feel the cost model:

1. Embed ~1,000 short documents with `text-embedding-005` (batch, `RETRIEVAL_DOCUMENT`) and write them as JSONL to Cloud Storage.
2. Create a **tree-AH** index and a small **brute-force** index over the same data; create a public index endpoint and deploy both.
3. Embed a handful of test questions (`RETRIEVAL_QUERY`) and call `find_neighbors` on both indexes. Compute **recall** of tree-AH against brute force, then raise `leaf_nodes_to_search_percent` and watch recall and latency both climb.
4. Wire the top-k results into a **Gemini** call (module 11's `google-genai` client) so the model answers strictly from the retrieved passages — a working RAG loop you built the retrieval half of yourself.
5. **`undeploy_index` and delete the endpoint when done** — then repeat the exercise using **pgvector on AlloyDB** and compare the effort, the query ergonomics (SQL filters!), and the cost of leaving each running.
