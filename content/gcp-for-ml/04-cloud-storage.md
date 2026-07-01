# 04 — Storage: Cloud Storage

Cloud Storage is the object store at the center of nearly every ML system on Google Cloud. Your raw data lands here, your training sets live here, your model checkpoints and artifacts are written here, and your batch prediction inputs and outputs pass through here. It is cheap, effectively infinite, and durable, and it is the "data lake" tier that sits underneath the BigQuery warehouse and the Vertex AI tooling. Fluency with buckets, storage classes, and lifecycle rules directly controls both the performance and the cost of your ML pipelines.

## Buckets and objects

The unit of Cloud Storage is the **bucket** — a container for **objects** (files). Bucket names are **globally unique** across all of Google Cloud, so teams prefix them with the project or company (`myco-fraud-data`, `myco-fraud-models`). Objects are immutable blobs identified by a key; the `/` in a key is just a naming convention that tools render as folders, but there are no real directories underneath. You address everything with `gs://bucket/path/to/object` URIs, which every Google Cloud ML tool understands natively.

The current command-line tool is **`gcloud storage`**, which supersedes the older `gsutil`. It is faster (better parallelism for the large, many-file transfers ML workloads generate) and has a cleaner grammar. Prefer it.

```bash
# Create a regional bucket co-located with your compute
gcloud storage buckets create gs://myco-fraud-data \
  --location=us-central1 \
  --uniform-bucket-level-access

# Upload/download; -r for recursive, and it parallelizes automatically
gcloud storage cp -r ./dataset gs://myco-fraud-data/datasets/v1/
gcloud storage ls gs://myco-fraud-data/datasets/v1/
gcloud storage cp gs://myco-fraud-models/best/model.pt ./model.pt
```

## Location types

When you create a bucket you choose a **location type**, and this decision affects latency, availability, and cost:

- **Region** (`us-central1`) — data in one region. Lowest latency and egress cost for compute in that same region, which is exactly what you want for training and serving. **This is the default choice for ML.** Co-locating a regional bucket with your GPUs/TPUs avoids cross-region reads that slow training and add network charges.
- **Dual-region** — a pair of regions for higher availability with predictable performance.
- **Multi-region** (`US`, `EU`, `ASIA`) — highest availability, served from anywhere in a broad geography; good for widely-accessed serving assets, less ideal for a training set that a single-region cluster streams.

## Storage classes and Autoclass

Every object has a **storage class** that trades retrieval cost and access latency against storage price:

- **Standard** — frequent access; your active training data and hot artifacts.
- **Nearline** — access less than about once a month; older dataset versions.
- **Coldline** — access roughly once a quarter; archived experiments.
- **Archive** — rarely accessed, long-term retention; compliance backups.

Colder classes are dramatically cheaper to store but charge more to retrieve and have minimum storage durations. Rather than manage classes by hand, enable **Autoclass** on a bucket and Cloud Storage automatically moves each object between classes based on its actual access pattern — objects that go cold slide toward cheaper tiers and jump back to Standard on access. For ML data lakes with mixed hot/cold access, Autoclass is the low-effort way to control storage cost.

```bash
gcloud storage buckets create gs://myco-fraud-archive \
  --location=us-central1 --enable-autoclass
```

## Lifecycle management

**Lifecycle rules** apply automated actions to objects based on age, version count, or class. For ML this is how you keep storage bills from creeping: delete raw ingestion files after they have been processed into the warehouse, downgrade old model versions, or expire intermediate pipeline artifacts.

```json
{
  "rule": [
    {
      "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
      "condition": {"age": 30}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"age": 365, "matchesPrefix": ["tmp/", "staging/"]}
    }
  ]
}
```

```bash
gcloud storage buckets update gs://myco-fraud-data \
  --lifecycle-file=lifecycle.json
```

## Access control and security

Turn on **uniform bucket-level access** at creation (shown above). It disables per-object ACLs and makes IAM the single source of truth for who can read and write the bucket — far simpler to reason about and audit than a mix of object ACLs and IAM. Grant workloads the least-privilege object roles from the security module: `roles/storage.objectViewer` for a training job that only reads data, `roles/storage.objectAdmin` for one that also writes checkpoints.

For controlled, time-limited external access without granting IAM, generate a **signed URL** — a URL that carries its own expiring authorization, useful for letting a client upload a batch-prediction input or download a result:

```bash
gcloud storage sign-url gs://myco-fraud-data/uploads/input.jsonl \
  --duration=1h
```

For high-security setups, combine no-external-IP compute, Private Google Access, and a VPC Service Controls perimeter so bucket data can never be read to anywhere outside your trusted boundary.

## Reading data for ML

There are two main ways training code consumes Cloud Storage:

- **The Python client library** for programmatic reads and writes, and for streaming objects into a data loader:

```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket("myco-fraud-data")
blob = bucket.blob("datasets/v1/train.parquet")
blob.download_to_filename("/local/train.parquet")

# Upload a checkpoint written during training
bucket.blob("checkpoints/step-1000.pt").upload_from_filename("step-1000.pt")
```

- **Cloud Storage FUSE**, which mounts a bucket as a local filesystem so training code can `open()` objects with ordinary file paths — no rewrite of your data-loading code. Vertex AI custom training mounts your buckets via Cloud Storage FUSE automatically (your training container sees them under `/gcs/<bucket>`), which is the idiomatic way large training jobs stream sharded datasets and write checkpoints. For the newest large-scale workloads, buckets with **hierarchical namespace** enabled give real directory semantics and faster rename/list operations, which improves FUSE and checkpointing performance.

## How this fits the whole solution

Cloud Storage is the substrate the whole pipeline shares. Streaming ingestion lands raw files here; Dataflow and BigQuery read and write here; Vertex AI training reads datasets and writes models and the model registry points at artifacts here; batch prediction reads inputs and writes outputs here; and CI/CD stages build artifacts here. Choosing regional buckets co-located with compute, enabling Autoclass and lifecycle rules for cost, enforcing uniform IAM for security, and mounting via FUSE for training are the storage decisions that make the end-to-end system fast, cheap, and safe.

## Key takeaways

- Buckets are **globally-named** containers of immutable objects addressed by `gs://` URIs; use the modern **`gcloud storage`** CLI, not `gsutil`.
- Default to **regional** buckets **co-located with your compute** to minimize training latency and egress; use multi-region only for broadly-accessed assets.
- Control cost with **storage classes**, **Autoclass** (automatic tiering), and **lifecycle rules** that downgrade or delete aged objects.
- Enforce **uniform bucket-level access** with least-privilege IAM; use **signed URLs** for time-limited external access; consume data via the **client library** or **Cloud Storage FUSE** (auto-mounted in Vertex training).

## Try it

Build the storage layer an ML pipeline actually uses:

1. Create a regional bucket in your compute region with uniform bucket-level access, and a second bucket with Autoclass enabled.
2. Upload a small dataset with `gcloud storage cp -r` and confirm it with `ls`.
3. Apply a lifecycle rule that deletes anything under `tmp/` after 7 days and moves everything else to Nearline after 30.
4. Grant a `training-sa` service account `objectViewer` on the data bucket and `objectAdmin` on a separate models bucket, reflecting read-only data and read-write artifacts.
5. From Python, download an object with the client library, then generate a 1-hour signed URL for a different object and fetch it with `curl` to see time-limited access in action.
