# 04 — Storage: S3 (and EBS)

Data is the input, the output, and the memory of every ML system, and on AWS the default home for data is Amazon S3. Object storage behaves differently from a filesystem, and those differences shape how you lay out datasets, feed training jobs, and control cost. This module covers S3 in depth, then places it against the block and file storage options — EBS, instance store, EFS, and FSx for Lustre — so you know which storage to use at each stage of an ML pipeline.

## S3 fundamentals

S3 stores **objects** (a blob of bytes plus metadata) inside **buckets** (globally-unique named containers scoped to a Region). There are no real directories; a key like `datasets/images/train/0001.jpg` is one flat string, and the "folders" you see are just key prefixes. This matters for ML because listing millions of objects under a prefix is a real operation with real latency — how you name keys affects throughput.

S3 is durable (designed for 11 nines of durability), effectively unlimited in size, and accessed over HTTP. It is the natural landing zone for raw data, curated datasets, model artifacts, and inference outputs.

```bash
aws s3 mb s3://my-ml-data --region us-east-1     # make bucket (high-level)
aws s3 cp ./train/ s3://my-ml-data/train/ --recursive
aws s3 sync ./features s3://my-ml-data/features   # only copies changes
aws s3 ls s3://my-ml-data/train/ --recursive --human-readable --summarize
aws s3 rm s3://my-ml-data/tmp/ --recursive       # delete a prefix
aws s3 rb s3://my-ml-data --force                # remove bucket (empties first)
```

The CLI has two layers, and you use both constantly. The high-level `aws s3` commands (`mb`, `ls`, `cp`, `sync`, `rm`, `rb`) are ergonomic and handle multipart and parallelism for you. The low-level `aws s3api` commands map one-to-one onto the REST API and expose every configuration knob — versioning, encryption, policies, lifecycle — that the high-level commands do not. Almost everything past "move bytes" happens through `s3api`:

```bash
aws s3api create-bucket --bucket my-ml-data --region us-east-1   # us-east-1 needs NO LocationConstraint
# every other Region REQUIRES the constraint or the call fails
aws s3api create-bucket --bucket my-ml-data --region eu-west-1 \
  --create-bucket-configuration LocationConstraint=eu-west-1
aws s3api head-bucket --bucket my-ml-data          # exists + you have access?
aws s3api list-objects-v2 --bucket my-ml-data --prefix train/ --max-items 100
```

The `us-east-1` special case is a classic gotcha: it is the only Region where `--create-bucket-configuration` must be *omitted*; every other Region requires a `LocationConstraint` matching `--region` or the request errors with `IllegalLocationConstraintException`.

For large files, S3 **multipart upload** splits the object into parts uploaded in parallel and reassembled — the CLI does this automatically above the `multipart_threshold` (8 MB by default), and it is what makes multi-GB dataset and checkpoint uploads fast and resumable. When you stream data into the CLI from stdin, S3 cannot see the size, so pass `--expected-size` (in bytes) so it picks a part size large enough to stay under the 10,000-part limit. Interrupted uploads leave orphaned parts that you still pay for — list and abort them, or set a lifecycle rule to clean them up:

```bash
# stream a large object from a producer, hinting the size so multipart sizing is right
some-producer | aws s3 cp - s3://my-ml-data/big.bin --expected-size 5368709120
aws s3api list-multipart-uploads --bucket my-ml-data      # find orphaned parts
aws s3api abort-multipart-upload --bucket my-ml-data --key big.bin --upload-id <id>
```

**Transfer Acceleration** routes uploads through the nearest edge location for cross-continent transfers; enable it per bucket, then target the accelerate endpoint:

```bash
aws s3api put-bucket-accelerate-configuration --bucket my-ml-data \
  --accelerate-configuration Status=Enabled
aws s3 cp big.tar s3://my-ml-data/ --endpoint-url https://s3-accelerate.amazonaws.com
```

### Tuning transfer throughput

The default CLI concurrency (10 parallel requests) leaves throughput on the table when moving a dataset directory to or from a large instance. The transfer settings live under the `s3` config namespace and are the difference between a `sync` that saturates a 25 Gbps NIC and one that trickles. Bump concurrency and the multipart chunk size on beefy hosts:

```bash
aws configure set default.s3.max_concurrent_requests 40
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.multipart_threshold 128MB
aws configure set default.s3.max_bandwidth 500MB/s   # cap it if sharing a link
```

`aws s3 sync` is the workhorse for dataset movement, and its filter flags are what make it precise. Order matters: filters apply left to right, so exclude broadly then re-include:

```bash
# push only images, drop everything else, and mirror deletes to the destination
aws s3 sync ./dataset s3://my-ml-data/dataset \
  --exclude "*" --include "*.jpg" --include "*.png" --delete
aws s3 sync ./dataset s3://my-ml-data/dataset --dryrun   # preview before committing
```

`--delete` makes the destination an exact mirror of the source (it removes objects absent locally) — powerful and dangerous, so `--dryrun` first is a habit worth keeping.

## Storage classes: matching cost to access pattern

S3 charges for storage, requests, and retrieval, and the classes trade these against each other:

- **S3 Standard** — frequent access, highest storage cost, no retrieval fee. Active training data.
- **S3 Intelligent-Tiering** — automatically moves objects between tiers based on access, no retrieval fees. The safe default when access patterns are unknown.
- **S3 Standard-IA / One Zone-IA** — infrequent access, cheaper storage, per-GB retrieval fee. One Zone-IA drops the multi-AZ redundancy for a lower price (fine for re-creatable data).
- **S3 Glacier Instant Retrieval** — archive pricing with millisecond access; good for rarely-touched data you still occasionally need immediately.
- **S3 Glacier Flexible Retrieval / Deep Archive** — lowest storage cost, retrieval in minutes to hours. Cold model archives, compliance data.
- **S3 Express One Zone** — a high-performance, single-AZ class delivering single-digit-millisecond access, up to ~10x faster than Standard with ~50% lower request costs. This is the class to feed high-throughput training that hammers many small objects.

You set the class at write time or transition into it later. On `cp`/`sync` use `--storage-class`; the accepted values are `STANDARD`, `STANDARD_IA`, `ONEZONE_IA`, `INTELLIGENT_TIERING`, `GLACIER_IR`, `GLACIER`, `DEEP_ARCHIVE`, and `EXPRESS_ONEZONE`:

```bash
aws s3 cp cold-archive.tar s3://my-ml-data/archive/ --storage-class DEEP_ARCHIVE
aws s3 cp model.tar.gz s3://my-ml-data/models/ --storage-class GLACIER_IR
aws s3 sync ./raw s3://my-ml-data/raw --storage-class INTELLIGENT_TIERING
```

Objects in `GLACIER` and `DEEP_ARCHIVE` are not directly readable — you must **restore** them first (this creates a temporary copy in Standard for the number of days you specify), and the restore itself takes minutes to hours depending on the retrieval tier. This is the gotcha that surprises people who archived a dataset and then tried to train on it:

```bash
aws s3api restore-object --bucket my-ml-data --key archive/cold-archive.tar \
  --restore-request '{"Days":7,"GlacierJobParameters":{"Tier":"Standard"}}'
aws s3api head-object --bucket my-ml-data --key archive/cold-archive.tar  # watch Restore status
```

Intelligent-Tiering has a nuance: it automatically manages the frequent/infrequent tiers for free, but the deep **Archive Access** and **Deep Archive Access** tiers are opt-in and configured per bucket. Turn them on only if you want objects that go untouched for 90+ days to sink into asynchronous-retrieval tiers automatically:

```bash
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-ml-data --id ArchiveAfter90 \
  --intelligent-tiering-configuration '{
    "Id":"ArchiveAfter90","Status":"Enabled","Filter":{"Prefix":"datasets/"},
    "Tierings":[{"Days":90,"AccessTier":"ARCHIVE_ACCESS"},
                {"Days":180,"AccessTier":"DEEP_ARCHIVE_ACCESS"}]}'
```

Newer analytical features are worth knowing: **S3 Tables** store tabular data natively in Apache Iceberg format, queryable by Athena, Redshift, and Spark; and **S3 Vectors** offers native vector storage and query for AI/RAG workloads. **Mountpoint for Amazon S3** exposes a bucket as a local filesystem so training code can read S3 objects with ordinary file APIs — you mount it and point a dataloader at the mountpoint as if it were local disk, which is the low-friction alternative to FastFile mode when your training code insists on POSIX paths.

```bash
# S3 Tables: a purpose-built table bucket, namespace, and Iceberg table via CLI
aws s3tables create-table-bucket --name analytics
aws s3tables create-namespace \
  --table-bucket-arn arn:aws:s3tables:us-east-1:123456789012:bucket/analytics \
  --namespace ml_features
# Mountpoint for S3 (installed separately): mount a bucket as a filesystem
mount-s3 my-ml-data /mnt/s3-data
```

**S3 Express One Zone** uses **directory buckets**, which are a distinct bucket type with a required name suffix encoding the Availability Zone — `--bucket <name>--<azid>--x-s3`. They live in a single AZ for latency and are created with a `create-bucket-configuration` naming a location:

```bash
aws s3api create-bucket --bucket ml-fast--use1-az4--x-s3 \
  --create-bucket-configuration \
  '{"Location":{"Type":"AvailabilityZone","Name":"use1-az4"},
    "Bucket":{"Type":"Directory","DataRedundancy":"SingleAvailabilityZone"}}'
```

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

You apply the policy with `put-bucket-lifecycle-configuration`, passing the rules as a file. A production rule set usually also expires *noncurrent* versions and aborts incomplete multipart uploads — the two silent cost leaks on a versioned bucket:

```bash
aws s3api put-bucket-lifecycle-configuration --bucket my-ml-data \
  --lifecycle-configuration file://lifecycle.json
aws s3api get-bucket-lifecycle-configuration --bucket my-ml-data   # read it back
```

```json
{
  "Rules": [{
    "ID": "tidy-versions-and-uploads",
    "Filter": { "Prefix": "artifacts/" },
    "Status": "Enabled",
    "NoncurrentVersionExpiration": { "NoncurrentDays": 30 },
    "AbortIncompleteMultipartUpload": { "DaysAfterInitiation": 7 }
  }]
}
```

**Versioning** keeps every version of an object, protecting against accidental overwrites and deletes — valuable for datasets and model artifacts where reproducibility matters. Combine it with lifecycle rules to expire old versions so they do not accumulate cost forever. Versioning is off by default and, once enabled, can only be suspended (never fully removed), so decide deliberately:

```bash
aws s3api put-bucket-versioning --bucket my-ml-data \
  --versioning-configuration Status=Enabled
aws s3api get-bucket-versioning --bucket my-ml-data
aws s3api list-object-versions --bucket my-ml-data --prefix models/   # see every version + delete markers
# fetch a specific historical version
aws s3api get-object --bucket my-ml-data --key models/m.tar --version-id <vid> old.tar
```

A subtle gotcha: on a versioned bucket, deleting an object does not remove it — it writes a **delete marker** and hides the current version. To truly reclaim space you either expire noncurrent versions via lifecycle or delete each `--version-id` explicitly. This is why versioned training buckets quietly grow even when the visible object count is flat.

## Encryption and access control

Every ML bucket holds data you must not leak, so encryption and access control are not optional polish. S3 encrypts all new objects at rest by default (SSE-S3, AES-256), but for regulated data you typically want **SSE-KMS** so access is gated by a KMS key policy and every decrypt is logged in CloudTrail. Set the default at the bucket level so nobody has to remember per object; enabling **bucket keys** with KMS cuts the per-request KMS cost dramatically on high-volume training reads:

```bash
# SSE-S3 (AES256) default
aws s3api put-bucket-encryption --bucket my-ml-data \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
# SSE-KMS default with a bucket key (recommended for sensitive datasets)
aws s3api put-bucket-encryption --bucket my-ml-data \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms",
      "KMSMasterKeyID":"arn:aws:kms:us-east-1:123456789012:key/abcd-1234"},
    "BucketKeyEnabled":true}]}'
```

**Block Public Access (BPA)** is the guardrail that keeps a fat-fingered ACL or policy from exposing a dataset to the internet. Turn on all four settings at the bucket level (account-level BPA is even stronger). This should be reflexive for every ML bucket:

```bash
aws s3api put-public-access-block --bucket my-ml-data \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

**Bucket policies** are resource-based IAM: they express "who can do what to this bucket" and are how you grant a SageMaker execution role read access, or enforce TLS-only access. A common hardening policy denies any request not made over HTTPS:

```bash
aws s3api put-bucket-policy --bucket my-ml-data --policy file://policy.json
aws s3api get-bucket-policy --bucket my-ml-data --query Policy --output text
```

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*",
    "Action": "s3:*",
    "Resource": ["arn:aws:s3:::my-ml-data", "arn:aws:s3:::my-ml-data/*"],
    "Condition": { "Bool": { "aws:SecureTransport": "false" } }
  }]
}
```

When you share a public dataset and want downloaders to pay their own request/egress costs, enable **Requester Pays** — standard practice for large open ML datasets. Callers must then acknowledge with `--request-payer requester` or the request is denied:

```bash
aws s3api put-bucket-request-payment --bucket my-ml-data \
  --request-payment-configuration Payer=Requester
aws s3api get-object --bucket my-ml-data --key file --request-payer requester out
```

**Presigned URLs** grant time-limited access to a single object without handing out credentials — the clean way to let a labeler or a downstream service fetch one artifact:

```bash
aws s3 presign s3://my-ml-data/models/m.tar --expires-in 3600
```

## Replication and event notifications

**Cross-Region (or Same-Region) Replication** asynchronously copies new objects to another bucket — used for disaster recovery, or to co-locate a dataset near training compute in a second Region. It requires versioning on both buckets and an IAM role S3 can assume:

```bash
aws s3api put-bucket-replication --bucket my-ml-data \
  --replication-configuration file://replication.json
aws s3api get-bucket-replication --bucket my-ml-data
```

**Event notifications** are the trigger mechanism that turns S3 into the front of a pipeline: a new object under `raw/` can fire a Lambda, drop a message on SQS, or publish to SNS to kick off ingestion or trigger a training job. This is how "data landed → pipeline runs" is wired without polling:

```bash
aws s3api put-bucket-notification-configuration --bucket my-ml-data \
  --notification-configuration file://notify.json
```

## Block and file storage: EBS, instance store, EFS, FSx for Lustre

S3 is object storage; sometimes you need a real disk or a shared filesystem.

- **EBS (Elastic Block Store)** is a network-attached virtual disk for a single EC2 instance — think of it as the instance's hard drive. Volume types: **gp3** (general-purpose SSD, the default; you provision IOPS and throughput independently), **gp2** (older SSD), **io2 Block Express** (highest IOPS for demanding databases), and **st1/sc1** (throughput-optimized HDD for cheap sequential data). Use EBS for the OS, code, and per-instance scratch.
- **Instance store** is physically-attached NVMe that is extremely fast but **ephemeral** — it vanishes when the instance stops. Perfect for shuffling/temp data during a training epoch, never for anything you must keep.
- **EFS (Elastic File System)** is a managed NFS filesystem shared across many instances and AZs — convenient for shared notebooks or moderate shared data, but not built for extreme throughput.
- **FSx for Lustre** is a high-performance parallel filesystem that can deliver hundreds of GB/s and millions of IOPS, and it links directly to an S3 bucket. This is the standard choice when a large distributed training job must feed thousands of GPU cores from a huge dataset without S3 request overhead dominating.

The practical difference from S3 is that these are provisioned resources you create, size, and attach. With **gp3** you provision IOPS (up to 16,000) and throughput (up to 1,000 MB/s) *independently* of size — the reason gp3 replaced gp2, where both scaled only with capacity. Create a volume, attach it to an instance in the same AZ (volumes are AZ-bound), and snapshot it to S3 for backup or to clone across AZs:

```bash
aws ec2 create-volume --availability-zone us-east-1a \
  --volume-type gp3 --size 500 --iops 6000 --throughput 500
aws ec2 attach-volume --volume-id vol-0abc --instance-id i-0def --device /dev/sdf
aws ec2 create-snapshot --volume-id vol-0abc --description "training-scratch backup"
aws ec2 modify-volume --volume-id vol-0abc --iops 10000   # scale IOPS live, no downtime
```

**EFS** is created once and mounted by many instances over NFS across AZs — no capacity to provision, it grows elastically:

```bash
aws efs create-file-system --performance-mode generalPurpose --throughput-mode elastic
aws efs create-mount-target --file-system-id fs-0abc --subnet-id subnet-0aaa \
  --security-group sg-0efs
```

**FSx for Lustre** is created with a link to an S3 data repository so the dataset is lazy-loaded from S3 on first access and results can be exported back — this is the "hydrate from S3" pattern for the largest training jobs. Choose `PERSISTENT_2` for durable throughput or `SCRATCH_2` for cheap ephemeral scratch:

```bash
aws fsx create-file-system --file-system-type LUSTRE \
  --storage-capacity 4800 --subnet-ids subnet-0aaa \
  --lustre-configuration \
  'DeploymentType=PERSISTENT_2,PerUnitStorageThroughput=250,DataRepositoryConfiguration={ImportPath=s3://my-ml-data/train/}'
```

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

## CLI cheat-sheet

```bash
# --- Buckets ---
aws s3 mb s3://BUCKET --region REGION                 # high-level make
aws s3api create-bucket --bucket BUCKET --region us-east-1          # no LocationConstraint here
aws s3api create-bucket --bucket BUCKET --region eu-west-1 \
  --create-bucket-configuration LocationConstraint=eu-west-1        # every other Region
aws s3api head-bucket --bucket BUCKET                 # exists / access check
aws s3 rb s3://BUCKET --force                         # empty + remove

# --- Move data ---
aws s3 cp SRC s3://BUCKET/KEY [--recursive]
aws s3 sync ./dir s3://BUCKET/pre --exclude "*" --include "*.jpg" --delete --dryrun
aws s3 ls s3://BUCKET/pre --recursive --human-readable --summarize
aws s3 rm s3://BUCKET/pre --recursive
aws s3 presign s3://BUCKET/KEY --expires-in 3600      # time-limited URL

# --- Storage classes / archive ---
aws s3 cp F s3://BUCKET/ --storage-class INTELLIGENT_TIERING   # STANDARD_IA GLACIER_IR GLACIER DEEP_ARCHIVE EXPRESS_ONEZONE
aws s3api restore-object --bucket BUCKET --key KEY \
  --restore-request '{"Days":7,"GlacierJobParameters":{"Tier":"Standard"}}'
aws s3api put-bucket-intelligent-tiering-configuration --bucket BUCKET --id ID \
  --intelligent-tiering-configuration file://it.json

# --- Versioning ---
aws s3api put-bucket-versioning --bucket BUCKET --versioning-configuration Status=Enabled
aws s3api list-object-versions --bucket BUCKET --prefix pre/
aws s3api get-object --bucket BUCKET --key KEY --version-id VID out

# --- Lifecycle ---
aws s3api put-bucket-lifecycle-configuration --bucket BUCKET --lifecycle-configuration file://lc.json
aws s3api get-bucket-lifecycle-configuration --bucket BUCKET

# --- Encryption / access ---
aws s3api put-bucket-encryption --bucket BUCKET \
  --server-side-encryption-configuration file://enc.json
aws s3api put-public-access-block --bucket BUCKET \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-policy --bucket BUCKET --policy file://policy.json
aws s3api put-bucket-request-payment --bucket BUCKET --request-payment-configuration Payer=Requester

# --- Replication / events / notifications ---
aws s3api put-bucket-replication --bucket BUCKET --replication-configuration file://rep.json
aws s3api put-bucket-notification-configuration --bucket BUCKET --notification-configuration file://notify.json

# --- Multipart cleanup ---
aws s3api list-multipart-uploads --bucket BUCKET
aws s3api abort-multipart-upload --bucket BUCKET --key KEY --upload-id UID

# --- Transfer tuning ---
aws configure set default.s3.max_concurrent_requests 40
aws configure set default.s3.multipart_chunksize 64MB
aws configure set default.s3.max_bandwidth 500MB/s

# --- S3 Express One Zone (directory buckets) ---
aws s3api create-bucket --bucket NAME--use1-az4--x-s3 \
  --create-bucket-configuration \
  '{"Location":{"Type":"AvailabilityZone","Name":"use1-az4"},"Bucket":{"Type":"Directory","DataRedundancy":"SingleAvailabilityZone"}}'

# --- S3 Tables ---
aws s3tables create-table-bucket --name NAME
aws s3tables create-namespace --table-bucket-arn ARN --namespace NS

# --- Block / file storage ---
aws ec2 create-volume --availability-zone AZ --volume-type gp3 --size 500 --iops 6000 --throughput 500
aws ec2 attach-volume --volume-id vol-x --instance-id i-y --device /dev/sdf
aws ec2 create-snapshot --volume-id vol-x
aws efs create-file-system --throughput-mode elastic
aws fsx create-file-system --file-system-type LUSTRE --storage-capacity 4800 --subnet-ids subnet-x \
  --lustre-configuration 'DeploymentType=PERSISTENT_2,PerUnitStorageThroughput=250'
```

## Try it

Create a bucket, enable versioning, and upload a dataset directory with `aws s3 sync`. Overwrite one file and confirm both versions exist. Apply a lifecycle policy that transitions a `logs/` prefix to Standard-IA after 30 days and expires it after 90. Finally, compare `aws s3 cp` throughput of a multi-GB file into S3 Standard versus S3 Express One Zone and note the difference — that gap is why training data placement is an engineering decision, not an afterthought.
