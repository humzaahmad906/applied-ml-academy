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

# Upload/download; --recursive for directories, and it parallelizes automatically
gcloud storage cp --recursive ./dataset gs://myco-fraud-data/datasets/v1/
gcloud storage ls gs://myco-fraud-data/datasets/v1/
gcloud storage cp gs://myco-fraud-models/best/model.pt ./model.pt
```

The workhorse commands you run daily go beyond `cp` and `ls`. `rsync` mirrors a source to a destination and only transfers what changed, which is how you keep a local dataset and a bucket in sync without re-uploading everything; `mv` renames or relocates objects; `rm` deletes; `du` reports space usage so you can see what a prefix actually costs; and `describe` dumps the metadata of a single object or an entire bucket. For big many-file ML transfers, `--recursive` plus the automatic parallelism is what makes uploading a sharded dataset fast.

```bash
# Mirror a local training tree into the bucket; --delete-unmatched-destination-objects
# removes objects that no longer exist locally (leave it off to only add/update)
gcloud storage rsync --recursive ./dataset gs://myco-fraud-data/datasets/v1/

# Move/rename and delete objects or whole prefixes
gcloud storage mv gs://myco-fraud-staging/tmp/run-42/ gs://myco-fraud-staging/archive/run-42/
gcloud storage rm --recursive gs://myco-fraud-staging/tmp/run-42/

# How much space is a prefix using? (add --readable-sizes for human units)
gcloud storage du --summarize --readable-sizes gs://myco-fraud-data/datasets/

# Inspect metadata: one object, or the bucket's whole config
gcloud storage objects describe gs://myco-fraud-models/best/model.pt
gcloud storage buckets describe gs://myco-fraud-data
gcloud storage buckets list --format="table(name, location, storageClass)"
```

Two transfer knobs matter for ML data. `--gzip-in-flight-all` (or `--gzip-in-flight=csv,json,...` for specific extensions) compresses uploads in memory and on the wire only — the local files and stored objects stay uncompressed — which cuts transfer time and egress on compressible text datasets. And you can set a storage class per copy with `--storage-class`, so an archival copy lands cold without a follow-up lifecycle move.

```bash
# Compress-on-the-wire for a large JSONL upload
gcloud storage cp --gzip-in-flight-all --recursive ./logs gs://myco-fraud-data/logs/

# Write a copy directly into a colder class
gcloud storage cp ./old-run.tar gs://myco-fraud-archive/2025/old-run.tar \
  --storage-class=COLDLINE
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

You can also change an existing object's class in place with `objects update --storage-class` — useful for hand-tiering a specific artifact without touching a lifecycle rule:

```bash
gcloud storage objects update gs://myco-fraud-models/2024/*.pt \
  --storage-class=NEARLINE
```

**Gotchas that cost real money.** Colder classes have **minimum storage durations** (Nearline 30 days, Coldline 90, Archive 365) — delete or rewrite an object before that window and you are billed for the remainder anyway, so do not lifecycle churn short-lived data into cold tiers. Cold classes also charge **per-operation retrieval and higher request fees**, so a training job that repeatedly scans a Coldline dataset can cost more in retrieval than you saved on storage. And **cross-region egress is billed**: reading a `US` multi-region bucket from a `us-central1` cluster, or copying between regions, incurs network charges that a co-located regional bucket avoids entirely.

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

## Data protection: versioning, soft delete, and retention

Three settings protect you from the two ways ML teams lose data: an overwrite that clobbers a good checkpoint, and a `rm --recursive` that hits the wrong prefix. **Object versioning** keeps prior generations of an object when it is overwritten or deleted, so you can roll back a model artifact to yesterday's weights. **Soft delete** is on by default and retains deleted objects (including in unversioned buckets) for a configurable window — a safety net for accidental deletes — and you tune it with `--soft-delete-duration`. **Retention policies** enforce a minimum age before any object can be deleted, which is how you satisfy compliance holds on training data. All are set on the bucket.

```bash
# Turn on versioning so overwrites/deletes keep prior generations
gcloud storage buckets update gs://myco-fraud-models --versioning

# List including noncurrent versions, then restore or purge a specific generation
gcloud storage ls --all-versions gs://myco-fraud-models/best/model.pt
gcloud storage rm gs://myco-fraud-models/best/model.pt#1712000000000000

# Extend the soft-delete window to 30 days (0d disables it)
gcloud storage buckets update gs://myco-fraud-data --soft-delete-duration=30d

# Enforce a 7-year minimum retention for compliance data
gcloud storage buckets update gs://myco-fraud-compliance --retention-period=7y
```

Note that a **non-empty bucket cannot be deleted** — you must clear objects first (`gcloud storage rm --recursive gs://bucket/**`) or use `buckets delete` on an emptied bucket. Cloud Storage is **strongly consistent**: once a write or metadata update returns, every subsequent read sees it, so you never have to code around read-after-write staleness the way you would on an eventually-consistent store.

## Composite objects

Cloud Storage lets you build one object out of up to 32 existing objects with `objects compose`, server-side and without re-uploading. The parallel-composite-upload path uses this under the hood to upload a large checkpoint as chunks and stitch them together, and you can use it directly to concatenate sharded outputs. All sources must live in the same bucket and share a storage class.

```bash
gcloud storage objects compose \
  gs://myco-fraud-data/shards/part-0000 \
  gs://myco-fraud-data/shards/part-0001 \
  gs://myco-fraud-data/merged/train.parquet
```

## Access control and security

Turn on **uniform bucket-level access** at creation (shown above). It disables per-object ACLs and makes IAM the single source of truth for who can read and write the bucket — far simpler to reason about and audit than a mix of object ACLs and IAM. Grant workloads the least-privilege object roles from the security module: `roles/storage.objectViewer` for a training job that only reads data, `roles/storage.objectAdmin` for one that also writes checkpoints. You grant these at the bucket scope with `buckets add-iam-policy-binding`, which is more surgical than a project-level grant:

```bash
# Read-only data access for the training service account
gcloud storage buckets add-iam-policy-binding gs://myco-fraud-data \
  --member="serviceAccount:training-sa@myco-fraud-dev.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# Read-write on the models bucket for the same job's checkpoint writes
gcloud storage buckets add-iam-policy-binding gs://myco-fraud-models \
  --member="serviceAccount:training-sa@myco-fraud-dev.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

For datasets you share with external teams, **Requester Pays** shifts egress and operation costs to the accessing project instead of yours — the caller must pass their billing project on every request. It is the standard way to publish a large dataset without absorbing everyone's download bill.

```bash
gcloud storage buckets update gs://myco-fraud-public --requester-pays
# Callers then read with: gcloud storage cp gs://... . --billing-project=THEIR_PROJECT
```

You can also set object **metadata** — `Content-Type`, `Cache-Control`, and custom headers — at upload or after the fact. `Cache-Control` matters for serving assets fetched by clients or CDNs, and correct `Content-Type` keeps downstream tools from mis-parsing your files.

```bash
gcloud storage objects update gs://myco-fraud-models/serve/labels.json \
  --content-type=application/json \
  --cache-control="public, max-age=3600"
```

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

- **Cloud Storage FUSE**, which mounts a bucket as a local filesystem so training code can `open()` objects with ordinary file paths — no rewrite of your data-loading code. Vertex AI custom training mounts your buckets via Cloud Storage FUSE automatically (your training container sees them under `/gcs/<bucket>`), which is the idiomatic way large training jobs stream sharded datasets and write checkpoints. For the newest large-scale workloads, buckets with **hierarchical namespace** enabled give real directory semantics and faster rename/list operations, which improves FUSE and checkpointing performance. Hierarchical namespace must be set at creation and requires uniform bucket-level access:

```bash
gcloud storage buckets create gs://myco-fraud-hns \
  --location=us-central1 \
  --uniform-bucket-level-access \
  --enable-hierarchical-namespace
```

For very high aggregate read throughput — many GPU/TPU nodes hammering the same dataset — **Anywhere Cache** provisions an SSD-backed zonal read cache in front of a bucket, cutting read latency and cross-zone egress for training that repeatedly streams the same shards. To move large datasets *into* Cloud Storage on a schedule (from another cloud, an S3 bucket, or on-prem), use the managed **Storage Transfer Service** rather than scripting `cp` loops; it handles retries, incremental sync, and bandwidth control.

## How this fits the whole solution

Cloud Storage is the substrate the whole pipeline shares. Streaming ingestion lands raw files here; Dataflow and BigQuery read and write here; Vertex AI training reads datasets and writes models and the model registry points at artifacts here; batch prediction reads inputs and writes outputs here; and CI/CD stages build artifacts here. Choosing regional buckets co-located with compute, enabling Autoclass and lifecycle rules for cost, enforcing uniform IAM for security, and mounting via FUSE for training are the storage decisions that make the end-to-end system fast, cheap, and safe.

## Key takeaways

- Buckets are **globally-named** containers of immutable objects addressed by `gs://` URIs; use the modern **`gcloud storage`** CLI, not `gsutil`.
- Default to **regional** buckets **co-located with your compute** to minimize training latency and egress; use multi-region only for broadly-accessed assets.
- Control cost with **storage classes**, **Autoclass** (automatic tiering), and **lifecycle rules** that downgrade or delete aged objects.
- Enforce **uniform bucket-level access** with least-privilege IAM; use **signed URLs** for time-limited external access; consume data via the **client library** or **Cloud Storage FUSE** (auto-mounted in Vertex training).

## CLI cheat-sheet

```bash
# --- Buckets: create / inspect / configure ---
gcloud storage buckets create gs://BUCKET --location=us-central1 --uniform-bucket-level-access
gcloud storage buckets create gs://BUCKET --location=us-central1 --enable-autoclass
gcloud storage buckets create gs://BUCKET --location=us-central1 \
  --uniform-bucket-level-access --enable-hierarchical-namespace
gcloud storage buckets list --format="table(name, location, storageClass)"
gcloud storage buckets describe gs://BUCKET
gcloud storage buckets update gs://BUCKET --versioning        # or --no-versioning
gcloud storage buckets update gs://BUCKET --soft-delete-duration=30d
gcloud storage buckets update gs://BUCKET --retention-period=7y
gcloud storage buckets update gs://BUCKET --requester-pays
gcloud storage buckets update gs://BUCKET --lifecycle-file=lifecycle.json
gcloud storage buckets delete gs://BUCKET                     # must be empty first

# --- Objects: move data ---
gcloud storage cp --recursive ./dir gs://BUCKET/prefix/
gcloud storage cp --gzip-in-flight-all --recursive ./logs gs://BUCKET/logs/
gcloud storage cp ./f gs://BUCKET/f --storage-class=COLDLINE
gcloud storage rsync --recursive --delete-unmatched-destination-objects ./src gs://BUCKET/dst/
gcloud storage mv gs://BUCKET/a/ gs://BUCKET/b/
gcloud storage rm --recursive gs://BUCKET/tmp/
gcloud storage ls --all-versions gs://BUCKET/path
gcloud storage du --summarize --readable-sizes gs://BUCKET/prefix/

# --- Objects: metadata / class / compose ---
gcloud storage objects describe gs://BUCKET/obj
gcloud storage objects update gs://BUCKET/obj --storage-class=NEARLINE
gcloud storage objects update gs://BUCKET/obj --content-type=application/json \
  --cache-control="public, max-age=3600"
gcloud storage objects compose gs://B/part-0 gs://B/part-1 gs://B/merged

# --- Access ---
gcloud storage buckets add-iam-policy-binding gs://BUCKET \
  --member="serviceAccount:SA@PROJECT.iam.gserviceaccount.com" --role="roles/storage.objectViewer"
gcloud storage sign-url gs://BUCKET/obj --duration=1h
```

## Try it

Build the storage layer an ML pipeline actually uses:

1. Create a regional bucket in your compute region with uniform bucket-level access, and a second bucket with Autoclass enabled.
2. Upload a small dataset with `gcloud storage cp -r` and confirm it with `ls`.
3. Apply a lifecycle rule that deletes anything under `tmp/` after 7 days and moves everything else to Nearline after 30.
4. Grant a `training-sa` service account `objectViewer` on the data bucket and `objectAdmin` on a separate models bucket, reflecting read-only data and read-write artifacts.
5. From Python, download an object with the client library, then generate a 1-hour signed URL for a different object and fetch it with `curl` to see time-limited access in action.
