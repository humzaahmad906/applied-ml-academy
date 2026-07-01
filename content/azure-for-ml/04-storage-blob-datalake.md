# 04 — Storage: Blob and Data Lake

Data is the input to every ML system, and on Azure the default home for that data is object storage. Training sets, validation splits, model checkpoints, exported artifacts, batch-scoring inputs and outputs, RAG source documents — all of it lands in **Blob Storage** or its analytics-oriented variant, **Azure Data Lake Storage Gen2 (ADLS Gen2)**. In the end-to-end solution, storage is the hub every other service reads from and writes to: ingestion drops raw data here, feature pipelines read it and write features back, training jobs mount it, the model registry stores artifacts on it, batch endpoints stream inputs and outputs through it, and monitoring exports logs to it. Understanding tiers, structure, and access is what keeps that hub fast and cheap.

## The storage account and its services

Everything starts with a **storage account** — a globally unique, DNS-named container for several storage services. For ML you almost always want a **general-purpose v2 (StorageV2)** account with **standard** performance (backed by HDD/hybrid, cost-efficient) rather than premium (SSD, for very high IOPS). The account exposes several services; the one you use most is **Blob** (object storage), organized as **containers** holding **blobs** (objects). Blobs come in three types, but for ML you overwhelmingly use **block blobs** — arbitrary files from a few KB to terabytes.

```bash
# Create a standard general-purpose v2 account for ML data
az storage account create \
  --name stmlxdata --resource-group rg-mlx-dev \
  --location eastus2 --sku Standard_LRS --kind StorageV2 \
  --enable-hierarchical-namespace true   # this flips it to Data Lake Gen2
```

The `--sku` sets **redundancy**: `LRS` (locally redundant, three copies in one data center) is the cheap default for reproducible data you could regenerate; `ZRS` (zone-redundant) survives a data-center loss; `GRS`/`GZRS` replicate to a second region for disaster recovery. For a training dataset you can re-derive, LRS is fine; for the one copy of hand-labeled ground truth you cannot recreate, pay for ZRS or GRS.

## Blob vs Data Lake Gen2: one product, one flag

ADLS Gen2 is not a separate service — it is **Blob Storage with the hierarchical namespace (HNS) enabled**, set at account creation with `--enable-hierarchical-namespace true`. That single flag changes the semantics in ways that matter for analytics and ML:

- **Real directories.** Without HNS, blob "folders" are a naming illusion — `data/train/x.parquet` is one flat key with slashes. Renaming or deleting a "folder" means touching every object. With HNS, directories are first-class, so a rename or a `move` of a million-file partition is a single atomic metadata operation. This is why big Spark, Fabric, and Synapse jobs run dramatically faster on HNS accounts — listing and reorganizing partitioned data stops being O(files).
- **POSIX-style ACLs.** HNS supports fine-grained access-control lists on directories and files, on top of RBAC, for the granular data governance analytics workloads expect.

The trade-off: a handful of niche Blob features behave differently on HNS, and per-operation pricing differs slightly. For an ML data platform the answer is almost always **enable HNS** — you want the directory semantics and ACLs, and the analytics engines you will layer on (Fabric, Synapse, Databricks) assume it. Both worlds share the same underlying storage, so you get lifecycle management, tiering, replication, soft delete, and Event Grid triggers regardless.

## Access tiers: matching cost to access pattern

Blob storage has four **access tiers** that trade storage cost against access cost and retrieval latency. Choosing correctly is one of the highest-leverage cost levers in an ML data platform:

- **Hot** — highest storage cost, lowest access cost. For actively used data: the current training set, features being read every epoch, live batch-scoring inputs.
- **Cool** — lower storage cost, higher access cost, 30-day minimum. For data queried weekly/monthly: last quarter's training snapshots, recent experiment artifacts.
- **Cold** — an online tier optimized for rarely accessed data that still needs fast retrieval; even lower storage cost, higher access cost, 90-day minimum. For older snapshots you occasionally reload.
- **Archive** — an offline tier, cheapest storage, but retrieval takes *hours* (you must "rehydrate") and has a 180-day minimum. For raw data you must keep for compliance or reproducibility but expect never to touch.

Set the tier per blob, and — the important part — automate transitions with a **lifecycle management policy** so old data ages down automatically instead of sitting expensive in Hot forever:

```bash
# Auto-tier and expire training snapshots by age
cat > lifecycle.json <<'EOF'
{
  "rules": [{
    "name": "age-out-snapshots",
    "enabled": true,
    "type": "Lifecycle",
    "definition": {
      "filters": { "blobTypes": ["blockBlob"], "prefixInclude": ["snapshots/"] },
      "actions": { "baseBlob": {
        "tierToCool":    { "daysAfterModificationGreaterThan": 30 },
        "tierToArchive": { "daysAfterModificationGreaterThan": 180 },
        "delete":        { "daysAfterModificationGreaterThan": 730 }
      }}
    }
  }]
}
EOF
az storage account management-policy create \
  --account-name stmlxdata -g rg-mlx-dev --policy @lifecycle.json
```

## Structuring data for ML: the medallion layout

A durable convention for the data lake is the **medallion (bronze/silver/gold)** layout, which maps cleanly onto ML stages:

- **Bronze** — raw, immutable, as-ingested data (JSON events, CSV dumps, images). Never overwrite; append.
- **Silver** — cleaned, validated, deduplicated, schema-enforced data, typically stored as **Parquet** or **Delta** for columnar efficiency.
- **Gold** — curated, aggregated, ML-ready feature tables and training sets.

A concrete container layout: `raw/` (bronze), `curated/` (silver), `features/` and `datasets/` (gold), `models/` (registry-backed artifacts), `scoring/inputs/` and `scoring/outputs/` (batch I/O). Prefer **Parquet** over CSV for tabular data — columnar, compressed, typed, and read far faster by training jobs and analytics engines. Partition large datasets by a natural key (date, region) so jobs read only the slices they need.

## Accessing storage from ML code

Authenticate with a **managed identity**, not account keys or SAS tokens embedded in code. Grant the platform identity **Storage Blob Data Reader** (or Contributor to write) scoped to the account or a single container, then use `DefaultAzureCredential`:

```python
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

svc = BlobServiceClient(
    account_url="https://stmlxdata.blob.core.windows.net",
    credential=DefaultAzureCredential(),
)
container = svc.get_container_client("datasets")
with open("train.parquet", "wb") as f:
    f.write(container.download_blob("fraud/train.parquet").readall())
```

Inside Azure Machine Learning you rarely download by hand. You register the storage account as a **datastore** and reference data as a **data asset** with a URI; jobs then **mount** or **download** the data automatically, and lineage is tracked:

```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

data_asset = Data(
    name="fraud-train",
    version="3",
    type=AssetTypes.URI_FOLDER,
    path="azureml://datastores/stmlxdata/paths/datasets/fraud/train/",
    description="Curated fraud training set, Parquet, partitioned by date",
)
ml_client.data.create_or_update(data_asset)
```

Versioned data assets are what make training reproducible: a job records exactly which data version it consumed, so you can always answer "what data produced this model."

## How storage fits the whole solution

Storage is the shared backbone. Streaming ingestion (Event Hubs) and batch ingestion (Data Factory) land raw data in **bronze**. Feature pipelines (Fabric/Synapse/Spark) read bronze, write **silver** and **gold**. Training jobs mount gold datasets; the **model registry** persists trained artifacts to blob. **Batch endpoints** read scoring inputs and write outputs to dedicated containers. **Monitoring** exports inference logs and drift baselines back to the lake. Because every service authenticates with the same managed identity and reads the same account, the lake is the single source of truth that keeps the whole system consistent — and correct tiering keeps it affordable at scale.

## Key takeaways

- Use a **general-purpose v2** account with **HNS enabled** (that flag *is* Data Lake Gen2) for ML data — you get real directories, ACLs, and fast analytics.
- Match **access tiers** (Hot / Cool / Cold / Archive) to access frequency, and automate aging with a **lifecycle policy** — the biggest storage cost lever.
- Structure the lake with the **medallion (bronze/silver/gold)** layout; prefer **Parquet/Delta** and partition large datasets.
- Access storage via **managed identity + `DefaultAzureCredential`**, never embedded keys; inside Azure ML use **datastores and versioned data assets** for reproducibility and lineage.
- Storage is the **hub** the entire end-to-end system reads from and writes to; getting its structure and tiering right pays off in every downstream service.

## Try it

Create a StorageV2 account with hierarchical namespace enabled. Create containers named `raw`, `curated`, and `datasets`, upload a small Parquet file to `datasets/demo/`, and set its tier to Cool. Then write and apply a lifecycle policy that tiers anything under `raw/` to Cool after 30 days and Archive after 180. Finally, grant your managed identity `Storage Blob Data Reader` on just the `datasets` container and read the Parquet file back with `DefaultAzureCredential` — confirming the whole path works with zero keys in your code.
