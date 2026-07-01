# 04 — Storage: S3 (and EBS)

Data is the input, the output, and the memory of every ML system, and on AWS the default home for data is Amazon S3. Object storage behaves differently from a filesystem, and those differences shape how you lay out datasets, feed training jobs, and control cost. This module covers S3 in depth, then places it against the block and file storage options — EBS, instance store, EFS, and FSx for Lustre — so you know which storage to use at each stage of an ML pipeline.

## S3 fundamentals

S3 stores **objects** (a blob of bytes plus metadata) inside **buckets** (globally-unique named containers scoped to a Region). There are no real directories; a key like `datasets/images/train/0001.jpg` is one flat string, and the "folders" you see are just key prefixes. This matters for ML because listing millions of objects under a prefix is a real operation with real latency — how you name keys affects throughput.

S3 is durable (designed for 11 nines of durability), effectively unlimited in size, and accessed over HTTP. It is the natural landing zone for raw data, curated datasets, model artifacts, and inference outputs.

```bash
aws s3 mb s3://my-ml-data                     # make bucket
aws s3 cp ./train/ s3://my-ml-data/train/ --recursive
aws s3 sync ./features s3://my-ml-data/features   # only copies changes
```

For large files, S3 **multipart upload** splits the object into parts uploaded in parallel and reassembled — the CLI does this automatically above a threshold, and it is what makes multi-GB dataset and checkpoint uploads fast and resumable. **Transfer Acceleration** routes uploads through the nearest edge location for cross-continent transfers.

## Storage classes: matching cost to access pattern

S3 charges for storage, requests, and retrieval, and the classes trade these against each other:

- **S3 Standard** — frequent access, highest storage cost, no retrieval fee. Active training data.
- **S3 Intelligent-Tiering** — automatically moves objects between tiers based on access, no retrieval fees. The safe default when access patterns are unknown.
- **S3 Standard-IA / One Zone-IA** — infrequent access, cheaper storage, per-GB retrieval fee. One Zone-IA drops the multi-AZ redundancy for a lower price (fine for re-creatable data).
- **S3 Glacier Instant Retrieval** — archive pricing with millisecond access; good for rarely-touched data you still occasionally need immediately.
- **S3 Glacier Flexible Retrieval / Deep Archive** — lowest storage cost, retrieval in minutes to hours. Cold model archives, compliance data.
- **S3 Express One Zone** — a high-performance, single-AZ class delivering single-digit-millisecond access, up to ~10x faster than Standard with ~50% lower request costs. This is the class to feed high-throughput training that hammers many small objects.

Newer analytical features are worth knowing: **S3 Tables** store tabular data natively in Apache Iceberg format, queryable by Athena, Redshift, and Spark; and **S3 Vectors** offers native vector storage and query for AI/RAG workloads. **Mountpoint for Amazon S3** exposes a bucket as a local filesystem so training code can read S3 objects with ordinary file APIs.

## Lifecycle policies and versioning

A **lifecycle policy** automates transitions — for example, move training logs to Standard-IA after 30 days and Glacier after 90, and expire temporary artifacts after 7. This is the single biggest lever on storage bills for a maturing ML platform:

```json
{
  "Rules": [{
    "ID": "archive-old-artifacts",
    "Filter": { "Prefix": "artifacts/" },
    "Status": "Enabled",
    "Transitions": [
      { "Days": 30, "StorageClass": "STANDARD_IA" },
      { "Days": 90, "StorageClass": "GLACIER" }
    ]
  }]
}
```

**Versioning** keeps every version of an object, protecting against accidental overwrites and deletes — valuable for datasets and model artifacts where reproducibility matters. Combine it with lifecycle rules to expire old versions so they do not accumulate cost forever.

## Block and file storage: EBS, instance store, EFS, FSx for Lustre

S3 is object storage; sometimes you need a real disk or a shared filesystem.

- **EBS (Elastic Block Store)** is a network-attached virtual disk for a single EC2 instance — think of it as the instance's hard drive. Volume types: **gp3** (general-purpose SSD, the default; you provision IOPS and throughput independently), **gp2** (older SSD), **io2 Block Express** (highest IOPS for demanding databases), and **st1/sc1** (throughput-optimized HDD for cheap sequential data). Use EBS for the OS, code, and per-instance scratch.
- **Instance store** is physically-attached NVMe that is extremely fast but **ephemeral** — it vanishes when the instance stops. Perfect for shuffling/temp data during a training epoch, never for anything you must keep.
- **EFS (Elastic File System)** is a managed NFS filesystem shared across many instances and AZs — convenient for shared notebooks or moderate shared data, but not built for extreme throughput.
- **FSx for Lustre** is a high-performance parallel filesystem that can deliver hundreds of GB/s and millions of IOPS, and it links directly to an S3 bucket. This is the standard choice when a large distributed training job must feed thousands of GPU cores from a huge dataset without S3 request overhead dominating.

## Choosing storage for each ML stage

Raw and curated data lives in **S3** (Standard or Intelligent-Tiering). For training, either stream from S3 directly, use **S3 Express One Zone** for many-small-object throughput, or hydrate **FSx for Lustre** from S3 for the largest jobs. Per-instance scratch during a job uses **instance store** or **gp3 EBS**. Model artifacts and inference outputs go back to **S3**, aging into Glacier via lifecycle rules.

## How this fits the whole ML solution

S3 is the hub the entire system revolves around: ingestion writes to it, the data lake catalogs it, training reads from it and writes artifacts back, batch inference reads and writes it, and monitoring archives predictions into it. Because nearly every other service reads and writes S3, your bucket layout and storage-class discipline quietly determine both the throughput of training and the size of the monthly bill. Get the data plane right and everything downstream gets simpler.

## Key takeaways

- S3 is durable, effectively unlimited object storage with a flat key namespace; prefixes are not real folders.
- Match storage classes to access: Standard/Intelligent-Tiering for active data, Glacier for archives, S3 Express One Zone for high-throughput training.
- Lifecycle policies and versioning are your primary tools for cost control and reproducibility.
- EBS = per-instance disk (gp3 default), instance store = fast but ephemeral scratch, EFS = shared NFS, FSx for Lustre = extreme parallel throughput linked to S3.
- Feed large distributed training from FSx for Lustre or S3 Express One Zone; keep raw and archival data in S3.

## Try it

Create a bucket, enable versioning, and upload a dataset directory with `aws s3 sync`. Overwrite one file and confirm both versions exist. Apply a lifecycle policy that transitions a `logs/` prefix to Standard-IA after 30 days and expires it after 90. Finally, compare `aws s3 cp` throughput of a multi-GB file into S3 Standard versus S3 Express One Zone and note the difference — that gap is why training data placement is an engineering decision, not an afterthought.
