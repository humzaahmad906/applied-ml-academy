# 08b — Vector Search

Retrieval-augmented generation (RAG) is only as good as its retrieval step. When a user asks a question, the system embeds it, finds the most semantically similar chunks of your own documents, and hands those chunks to the model as grounding context. That "find the most similar chunks" step is a **nearest-neighbor search over embedding vectors**, and it needs a store built for it — a plain relational `WHERE` clause cannot rank a thousand-dimensional vector by cosine similarity at speed. Module 08 covered the data services that hold your source records; this module covers the retrieval layer you reach for when you roll your own RAG instead of letting a managed service do it. AWS gives you three realistic options — OpenSearch Serverless, pgvector on Aurora/RDS, and the newer S3 Vectors — plus Bedrock Knowledge Bases, which wires one of them up for you.

## Why you need a vector store

An embedding model (Amazon Titan Text Embeddings V2, Cohere Embed, or similar) turns each text chunk into a fixed-length float array — 1,024 dimensions for Titan V2 by default, configurable down to 512 or 256. Retrieval means: embed the query the same way, then find the stored vectors with the smallest distance (cosine or Euclidean) to it. Doing this exactly over millions of vectors is too slow, so vector stores use **approximate nearest neighbor (ANN)** indexes — HNSW (Hierarchical Navigable Small World) or IVF (Inverted File) — that trade a sliver of recall for orders-of-magnitude speed. The store's job is to hold the vectors, build the ANN index, and answer top-k queries with optional metadata filters. The three AWS options differ mainly in *where that index lives and what you pay to keep it warm*.

## The three options and when each fits

- **OpenSearch Serverless (vector search collection)** — the scale-and-features choice. It runs the OpenSearch k-NN engine (HNSW and IVF, cosine/L2/dot-product, up to 16,000 dimensions) and, crucially, supports **hybrid search** — combining lexical BM25 scoring with vector similarity in one query, which pure vector stores can't do. Reach for it when you have large corpora, need filtering plus full-text plus vector in one place, or want binary-vector support (only OpenSearch offers it). The catch is cost floor (below).
- **pgvector on Aurora/RDS PostgreSQL** — the "I already have Postgres" choice. If your transactional data and your embeddings can live in the same database, you get vector search with a familiar `SELECT`, join vectors to your relational rows, and pay nothing extra beyond the instance you were already running. Best for smaller-to-medium corpora and teams that value operational simplicity over specialized scale.
- **S3 Vectors** — the cost-optimized choice, GA since December 2025. A native vector store built into S3 with a storage-first architecture: no cluster, no always-on compute, billed on storage plus per-query API calls. It supports up to 2 billion vectors per index and ~100 ms warm-query latency, at up to ~90% lower TCO than cluster-based stores for large, infrequently-queried datasets. Best for dev/test, cost-sensitive production, and archival-scale RAG where sub-second (not single-digit-ms) latency is acceptable.

## Worked example: OpenSearch Serverless

A vector search collection is created through the `opensearchserverless` control-plane API, but you must lay down three security policies first — encryption, network, and data access — or the collection creation fails. This trips up nearly everyone the first time.

```bash
COLL=ml-rag

# 1. Encryption policy (AWS-owned key here; use kmsKeyArn for CMK)
aws opensearchserverless create-security-policy --name rag-enc --type encryption \
  --policy '{"Rules":[{"ResourceType":"collection","Resource":["collection/'$COLL'"]}],"AWSOwnedKey":true}'

# 2. Network policy (public access for the demo; use a VPC endpoint in prod)
aws opensearchserverless create-security-policy --name rag-net --type network \
  --policy '[{"Rules":[{"ResourceType":"collection","Resource":["collection/'$COLL'"]},
             {"ResourceType":"dashboard","Resource":["collection/'$COLL'"]}],"AllowFromPublic":true}]'

# 3. Data-access policy — grants an IAM principal index/document permissions
aws opensearchserverless create-access-policy --name rag-data --type data \
  --policy '[{"Rules":[{"ResourceType":"index","Resource":["index/'$COLL'/*"],"Permission":["aoss:*"]}],
             "Principal":["arn:aws:iam::123456789012:role/RagRole"]}]'

# 4. Now create the VECTORSEARCH collection
aws opensearchserverless create-collection --name $COLL --type VECTORSEARCH
aws opensearchserverless list-collections --query 'collectionSummaries[0].id'
```

Data-plane work (create index, index vectors, query) goes through the OpenSearch REST API, signed with SigV4. The service name for serverless is `aoss`. Use `opensearch-py` with `AWSV4SignerAuth`:

```python
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers

region = "us-west-2"
host = "abc123.us-west-2.aoss.amazonaws.com"   # collection endpoint, no https://
auth = AWSV4SignerAuth(boto3.Session().get_credentials(), region, "aoss")
client = OpenSearch(hosts=[{"host": host, "port": 443}], http_auth=auth,
                    use_ssl=True, verify_certs=True,
                    connection_class=RequestsHttpConnection)

# Create a k-NN index: knn_vector field sized to the embedding model, HNSW + cosine
client.indices.create(index="docs", body={
    "settings": {"index.knn": True},
    "mappings": {"properties": {
        "embedding": {"type": "knn_vector", "dimension": 1024,
                      "method": {"name": "hnsw", "engine": "faiss",
                                 "space_type": "cosinesimil"}},
        "text": {"type": "text"}}}})
```

Embeddings come from Bedrock. Invoke Titan Text Embeddings V2, then bulk-load with `helpers.bulk`:

```python
import json
bedrock = boto3.client("bedrock-runtime", region_name=region)

def embed(text):
    r = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0",
                             body=json.dumps({"inputText": text, "dimensions": 1024}))
    return json.loads(r["body"].read())["embedding"]

chunks = ["Aurora auto-scales storage.", "DynamoDB is single-digit-ms key-value.", "..."]
helpers.bulk(client, ({"_index": "docs", "_source": {"text": c, "embedding": embed(c)}}
                      for c in chunks))

# Query: embed the question, ask for the k nearest chunks
q = embed("Which AWS database scales storage automatically?")
res = client.search(index="docs", body={"size": 3,
        "query": {"knn": {"embedding": {"vector": q, "k": 3}}}})
for hit in res["hits"]["hits"]:
    print(hit["_score"], hit["_source"]["text"])
```

That top-3 text is what you paste into the prompt as grounding context. To make it hybrid, wrap the `knn` clause and a `match` clause in a `bool`/`should` query so lexical and semantic scores combine.

## The pgvector alternative

If a Postgres instance already holds your data (see module 08 for `create-db-cluster`), pgvector adds vector search without a second system. Enable the extension, add a `vector` column, and build an index. pgvector 0.8.0 is current on Aurora PostgreSQL (requires engine 17.4+/16.8+/15.12+/14.17+/13.20+ or newer) and RDS PostgreSQL.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE docs (id bigserial PRIMARY KEY, text text, embedding vector(1024));

-- HNSW is the production default; IVFFlat is cheaper to build but needs tuning.
-- vector_cosine_ops matches a cosine embedding; use vector_l2_ops for Euclidean.
CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops);

-- Retrieve the 3 nearest chunks. <=> is cosine distance; <-> is L2, <#> is inner product.
SELECT id, text FROM docs ORDER BY embedding <=> '[0.12, -0.04, ...]'::vector LIMIT 3;
```

You still generate the embedding with Bedrock in your app, then pass the array into the query. The payoff is the join: `... WHERE tenant_id = $1 ORDER BY embedding <=> $2 LIMIT 3` filters by a relational column and ranks by similarity in one statement, with full transactional consistency. The limit is scale — beyond tens of millions of vectors, HNSW build time and memory pressure on the instance push you toward OpenSearch or S3 Vectors.

## S3 Vectors

S3 Vectors uses a dedicated `s3vectors` boto3 client and a new bucket type. You create a **vector bucket**, then a **vector index** with a fixed dimension and distance metric (both immutable after creation, like the index name and any non-filterable metadata keys), then `put_vectors` and `query_vectors`.

```python
s3v = boto3.client("s3vectors", region_name="us-west-2")

s3v.create_vector_bucket(vectorBucketName="rag-embeddings")
s3v.create_index(vectorBucketName="rag-embeddings", indexName="docs",
                 dataType="float32", dimension=1024, distanceMetric="cosine",
                 metadataConfiguration={"nonFilterableMetadataKeys": ["source_text"]})

s3v.put_vectors(vectorBucketName="rag-embeddings", indexName="docs", vectors=[
    {"key": "chunk-1", "data": {"float32": embed("Aurora auto-scales storage.")},
     "metadata": {"source_text": "Aurora auto-scales storage.", "topic": "db"}}])

res = s3v.query_vectors(vectorBucketName="rag-embeddings", indexName="docs",
                        queryVector={"float32": embed("which db scales storage?")},
                        topK=3, filter={"topic": "db"},
                        returnDistance=True, returnMetadata=True)
```

Metadata keys are filterable by default; only keys you name as non-filterable (typically the raw text you carry along for the prompt) are excluded from filter evaluation. There is no cluster to provision and nothing to keep warm.

## Bedrock Knowledge Bases: rolling your own vs. managed

Everything above is the roll-your-own path: you own chunking, embedding calls, indexing, and query assembly. **Bedrock Knowledge Bases** collapses all of it. You point it at an S3 data source, pick an embedding model, and pick a vector store — and it chunks documents, generates embeddings, writes them to the store, and exposes a single `retrieve` (or `retrieve_and_generate`) API. As of 2026 the supported stores include OpenSearch Serverless and managed clusters, Aurora PostgreSQL (pgvector), S3 Vectors, Neptune Analytics, and third parties (Pinecone, MongoDB Atlas, Redis). "Quick create" will stand up an OpenSearch Serverless collection, an Aurora Serverless v2 vector store, or an S3 vector bucket for you; "bring your own" lets you attach a store you built with the code above.

Rule of thumb: use Knowledge Bases when you want RAG plumbing to disappear and standard chunking is fine. Roll your own when you need custom chunking, a hybrid-search ranking you control, embeddings from a model Knowledge Bases doesn't offer, or the vectors sitting next to transactional data in your own Postgres.

## Cost and ops gotchas

- **OpenSearch Serverless has a real floor.** Classic collections keep a minimum of OCUs running even when idle — a vector collection cannot share OCUs with other collection types, so your first one spins up its own set, roughly $175/mo (dev) to $350/mo (2-OCU production) at ~$0.24/OCU-hour *regardless of traffic*. The next-generation OpenSearch Serverless (GA May 2026) scales compute to zero after ~10 idle minutes — verify which mode your account/region uses, since it changes the cost story entirely. **Check current docs for exact OCU minimums and pricing.**
- **pgvector cost is just your instance** — no extra service bill, but HNSW indexes consume RAM; size the instance for the index, not just the rows.
- **S3 Vectors has zero idle cost** — billed on storage, API calls, and data processed. The trade is latency (sub-second cold, ~100 ms warm) versus OpenSearch's single-digit-ms, so it fits cost-sensitive and bursty workloads, not tight-SLA interactive search.
- **IAM:** OpenSearch Serverless needs the data-access policy naming your principal (an IAM allow alone is not enough — this is the most common "403 despite permissions" cause). S3 Vectors and Bedrock KB each need their own resource permissions plus `bedrock:InvokeModel` for the embedding model.
- **Dimensions are immutable** in S3 Vectors indexes and are baked into the OpenSearch mapping and the pgvector column type. Changing embedding models with a different output size means a new index and a full re-embed.

## How this fits the whole ML solution

The vector store is the retrieval half of RAG — the bridge between the data lake (module 08) and the generative model (the Bedrock module). Documents land in S3, get chunked and embedded, and the embeddings live in one of these three stores; at query time you embed the user's question, retrieve the top-k grounding chunks, and pass them to the LLM so it answers from your data instead of hallucinating. Pick the store by scale and cost shape: OpenSearch for large hybrid-search workloads, pgvector when embeddings belong next to transactional data, S3 Vectors when idle cost must be zero — or hand the whole loop to Bedrock Knowledge Bases and let it choose and manage the store for you.

## Key takeaways

- RAG's retrieval step is approximate nearest-neighbor search over embeddings; you need a purpose-built vector store, not a plain `WHERE`.
- OpenSearch Serverless scales and uniquely supports hybrid (lexical + vector) search, but carries an always-on OCU cost floor unless you're on the next-gen scale-to-zero mode.
- pgvector on Aurora/RDS (0.8.0, HNSW, `<=>`/`<->` operators) is the cheap, simple choice when embeddings can live beside relational data at moderate scale.
- S3 Vectors (GA Dec 2025) is the cost-optimized, zero-idle-cost store — up to 2B vectors/index, ~100 ms warm queries — for large or bursty RAG.
- Bedrock Knowledge Bases automates chunking, embedding, indexing, and retrieval over any of these stores; roll your own only when you need control the managed path doesn't give.

## Try it

Chunk a handful of your own documents and generate embeddings with Titan Text Embeddings V2 via Bedrock. Load the same embeddings into two stores: a pgvector table on an Aurora Serverless v2 cluster and an S3 Vectors index. Run the same top-3 query against both, compare the returned chunks and latency, then check the day's cost for each (Aurora instance-hours vs. S3 Vectors storage + API calls). Finally, stand up a Bedrock Knowledge Base with "quick create" over the same S3 documents, call `retrieve`, and compare its results to your hand-built pipeline — noting where its default chunking differs from yours.
