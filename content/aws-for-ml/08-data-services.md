# 08 — Data Services

Models are only as good as the data plumbing behind them. Before a feature ever reaches a training job it has usually passed through operational databases, been catalogued in a data lake, queried and aggregated, and materialized into a feature store. AWS offers a distinct service for each of these jobs, and knowing which to reach for — and how they compose into a data lake and feature pipeline — is what separates a demo from a production ML system. This module surveys the data services an ML engineer touches most.

## Relational data: RDS and Aurora

**Amazon RDS** is managed relational databases — PostgreSQL, MySQL, MariaDB, SQL Server, Oracle — where AWS handles patching, backups, and failover. **Amazon Aurora** is AWS's cloud-native engine compatible with PostgreSQL and MySQL, with higher performance and storage that auto-scales. **Aurora Serverless v2** scales capacity up and down automatically, which suits spiky or unpredictable workloads. For ML, RDS/Aurora is where transactional application data lives — the source of truth you extract features from, and often the online store for low-latency lookups when a feature store is overkill.

Provisioning a plain RDS instance takes one call; the practitioner details are the flags around it. `--allocated-storage` with `--max-allocated-storage` turns on storage autoscaling so you do not get paged at 2am for a full disk; `--multi-az` gives you a synchronous standby for failover; and you should almost never pass `--master-user-password` on the command line (it lands in shell history and process listings) — prefer `--manage-master-user-password`, which lets RDS generate and rotate the credential in Secrets Manager for you.

```bash
# Single-AZ Postgres instance with storage autoscaling and Secrets Manager-managed creds
aws rds create-db-instance \
  --db-instance-identifier ml-app-db \
  --db-instance-class db.r6g.large \
  --engine postgres --engine-version 16.4 \
  --allocated-storage 100 --max-allocated-storage 1000 \
  --manage-master-user-password \
  --master-username mladmin \
  --backup-retention-period 7 --multi-az

aws rds describe-db-instances --db-instance-identifier ml-app-db \
  --query 'DBInstances[0].Endpoint'
```

Aurora is a two-step model that trips people up: you create a **cluster** (the storage/endpoint layer) and then add **instances** to it, unlike RDS where the instance *is* the database. Aurora Serverless v2 is expressed as a capacity range in **ACUs** (Aurora Capacity Units, ~2 GiB RAM each) via `--serverless-v2-scaling-configuration`; setting `MinCapacity=0` lets a v2 cluster pause to zero when idle (a relatively recent addition worth knowing for dev/test cost).

```bash
# Aurora PostgreSQL Serverless v2 cluster, then attach a serverless instance
aws rds create-db-cluster \
  --db-cluster-identifier ml-aurora \
  --engine aurora-postgresql --engine-version 16.4 \
  --manage-master-user-password --master-username mladmin \
  --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=16

aws rds create-db-instance \
  --db-instance-identifier ml-aurora-1 \
  --db-cluster-identifier ml-aurora \
  --engine aurora-postgresql \
  --db-instance-class db.serverless
```

A gotcha for ML extraction jobs: reading heavily from the primary competes with the application's writes. Point analytical reads at the **reader endpoint** (or an Aurora replica) instead of the writer, and for one-off dataset pulls take a snapshot and restore it rather than hammering production.

## Key-value at scale: DynamoDB

**Amazon DynamoDB** is a fully managed NoSQL key-value and document database with single-digit-millisecond latency at any scale. You choose **on-demand** capacity (pay per request, scales instantly — the safe default) or **provisioned** capacity (cheaper for steady, predictable load). In ML systems DynamoDB shines as an **online feature store** or a low-latency lookup layer: an inference request arrives, you fetch the entity's precomputed features from DynamoDB by key in milliseconds, and pass them to the model.

Create the table with `--billing-mode PAY_PER_REQUEST` (on-demand) so you never have to guess capacity — this is the right default for feature-lookup traffic that follows model demand. You define only the **key attributes** (partition key, optional sort key); everything else is schemaless. The single hard limit to remember is a **400 KB max item size**, so store large feature vectors in S3 and keep a pointer in DynamoDB.

```bash
# On-demand table keyed by user_id (plus an optional sort key for versioned features)
aws dynamodb create-table \
  --table-name user-features \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# Enable TTL so stale precomputed features self-expire (ML gotcha: keep online = offline)
aws dynamodb update-time-to-live --table-name user-features \
  --time-to-live-specification "Enabled=true,AttributeName=expires_at"
```

The common operations are a small set. `put-item`/`get-item` for single rows, `query` to pull all rows for one partition key (never `scan` in a hot path — it reads the whole table), and `batch-write-item` to bulk-load a materialized feature set (25 items per call, so paginate). Switching a provisioned table to on-demand under load is done live with `update-table --billing-mode PAY_PER_REQUEST`.

```bash
# Write and read a feature row
aws dynamodb put-item --table-name user-features \
  --item '{"user_id":{"S":"u_123"},"events_7d":{"N":"42"},"expires_at":{"N":"1735689600"}}'
aws dynamodb get-item --table-name user-features \
  --key '{"user_id":{"S":"u_123"}}' --consistent-read

# Bulk backfill (batch-write-item takes up to 25 put/delete requests per call)
aws dynamodb batch-write-item --request-items file://features-batch.json
```

```python
import boto3
ddb = boto3.resource("dynamodb").Table("user-features")
item = ddb.get_item(Key={"user_id": "u_123"})["Item"]   # ms-latency feature fetch
```

For read-heavy inference at extreme scale, front the table with **DynamoDB Accelerator (DAX)**, an in-memory cache that drops reads to microseconds, and use **global tables** for multi-region low-latency serving.

## Query on S3: Athena

**Amazon Athena** runs standard SQL directly over data sitting in S3, using a serverless Presto/Trino engine — no cluster to manage, billed per data scanned. It reads CSV, JSON, Parquet, ORC, and Iceberg tables. For ML, Athena is the workhorse for exploratory analysis, building training datasets with SQL, and computing aggregate features over historical data. Two practices control cost and speed: store data in **columnar Parquet** and **partition** it (for example by date), so a query scans only the relevant slices instead of the whole dataset.

```sql
SELECT user_id, COUNT(*) AS events_7d
FROM events
WHERE dt BETWEEN date_add('day', -7, current_date) AND current_date
GROUP BY user_id;   -- partitioned on dt, columnar Parquet: scans only 7 partitions
```

From the CLI, Athena is asynchronous: you `start-query-execution`, poll `get-query-execution` until the state is `SUCCEEDED`, then pull rows with `get-query-results` or read the result CSV that Athena wrote to S3. Every query **must** have a `--result-configuration` output location (Athena writes results there regardless), and `--work-group` is how you attach cost guardrails — a per-query data-scanned limit that kills a runaway `SELECT *` before it scans a terabyte.

```bash
# Kick off a query, capturing the execution id
QID=$(aws athena start-query-execution \
  --query-string "SELECT user_id, COUNT(*) c FROM events WHERE dt='2026-07-01' GROUP BY user_id" \
  --query-execution-context Database=ml_lake \
  --work-group ml-analysts \
  --result-configuration OutputLocation=s3://my-ml-results/athena/ \
  --query 'QueryExecutionId' --output text)

# Poll status, then fetch results (also check bytes scanned = cost)
aws athena get-query-execution --query-execution-id "$QID" \
  --query 'QueryExecution.{State:Status.State,Bytes:Statistics.DataScannedInBytes}'
aws athena get-query-results --query-execution-id "$QID"
```

Two gotchas beyond partitioning: after adding partitions to S3 you must register them (run `MSCK REPAIR TABLE` or, better, enable **partition projection** so Athena computes partition paths without a metastore round-trip), and `CREATE TABLE AS SELECT` (**CTAS**) is the idiomatic way to write a partitioned-Parquet training set directly from a query — one statement to build the dataset the next stage reads.

## ETL and cataloging: Glue

**AWS Glue** is the serverless ETL and metadata layer. The **Glue Data Catalog** is a central schema registry that Athena, Redshift, and Spark all read from — it is what turns a pile of S3 files into queryable "tables." **Glue crawlers** infer schema from S3 data and populate the catalog automatically. **Glue jobs** run Spark (or Python shell) to transform data at scale — cleaning, joining, and reshaping raw data into curated, feature-ready datasets. Glue is the standard tool for the "T" in the ELT that feeds training.

The catalog side is three verbs: create a **database** (a logical namespace of tables), point a **crawler** at an S3 prefix to infer tables into it, and read the result with `get-tables`. A crawler is convenient but has real gotchas — it will happily create a *separate* table per subfolder if your layout is inconsistent, and it can misinfer types on messy CSV, so for stable pipelines many teams define tables explicitly (or via Athena CTAS) and skip the crawler.

```bash
aws glue create-database --database-input Name=ml_lake

aws glue create-crawler \
  --name events-crawler --role AWSGlueServiceRole-ml \
  --database-name ml_lake \
  --targets '{"S3Targets":[{"Path":"s3://my-ml-data/raw/events/"}]}'
aws glue start-crawler --name events-crawler
aws glue get-tables --database-name ml_lake --query 'TableList[].Name'
```

The transformation side is **jobs**. You register a job (its script in S3, a role, and a `--glue-version`/worker configuration), then trigger runs and poll them. The knobs that matter for cost and speed are the worker type and count: `--worker-type G.1X`/`G.2X`/`G.4X` (increasing vCPU and memory per worker) and `--number-of-workers`. A frequent surprise is the DPU-hour billing with a minimum billed duration, so batch small transforms rather than firing hundreds of tiny jobs.

```bash
aws glue create-job \
  --name curate-events \
  --role AWSGlueServiceRole-ml \
  --command '{"Name":"glueetl","ScriptLocation":"s3://my-ml-data/scripts/curate.py","PythonVersion":"3"}' \
  --glue-version "4.0" \
  --worker-type G.1X --number-of-workers 10

RUN=$(aws glue start-job-run --job-name curate-events \
  --arguments '{"--source":"s3://my-ml-data/raw/","--dest":"s3://my-ml-data/curated/"}' \
  --query 'JobRunId' --output text)
aws glue get-job-run --job-name curate-events --run-id "$RUN" \
  --query 'JobRun.JobRunState'
```

## Analytics warehouse: Redshift

For heavy analytical queries and joins across large curated datasets, **Amazon Redshift** is the columnar data warehouse; **Redshift Serverless** runs it without managing clusters. In an ML platform Redshift often holds the modeled, warehouse-grade data that feature pipelines read from, and it integrates with the same Glue Catalog and S3 so the lake and warehouse share one view of the data.

Serverless is the two-object model most new projects want: a **namespace** (the database, its admin credentials, and IAM roles) and a **workgroup** (the compute, sized in **RPUs** — Redshift Processing Units — via `--base-capacity`). As with RDS, let Redshift manage the admin password in Secrets Manager rather than typing it. You attach an IAM role at the namespace level so Redshift can `COPY` from and `UNLOAD` to S3, which is how training data flows in and out.

```bash
# Serverless: namespace (data + creds) then a workgroup (compute)
aws redshift-serverless create-namespace \
  --namespace-name ml-warehouse \
  --admin-username mladmin --manage-admin-password \
  --iam-roles arn:aws:iam::123456789012:role/RedshiftSpectrumRole

aws redshift-serverless create-workgroup \
  --workgroup-name ml-wg \
  --namespace-name ml-warehouse \
  --base-capacity 32 \
  --subnet-ids subnet-aaa subnet-bbb --security-group-ids sg-123
```

If you need a fixed, always-on cluster (predictable heavy load, reserved-instance pricing), the provisioned path is `aws redshift create-cluster` instead. The recurring practitioner win is **Redshift Spectrum**: with the Glue Catalog attached you query S3 data in place with `SELECT`, joining lake data to warehouse tables without loading it first — the same "one view of the data" that keeps the lake and warehouse in sync. `UNLOAD ... TO 's3://...' FORMAT PARQUET` is the idiomatic way to export a query result as a partitioned training set.

```bash
# Provisioned alternative when you want a fixed cluster
aws redshift create-cluster \
  --cluster-identifier ml-warehouse \
  --node-type ra3.xlplus --number-of-nodes 2 \
  --master-username mladmin --manage-master-password \
  --db-name analytics
```

## Feature data: SageMaker Feature Store

**Amazon SageMaker Feature Store** is the purpose-built home for ML features, solving the notorious train/serve skew problem. It has two synchronized faces: an **online store** for low-latency reads at inference time, and an **offline store** (in S3) for building training datasets and backfills. You write a feature once and read the *same* definition in both training and serving, so the features a model trains on match the features it sees in production. Feature groups are versioned and time-stamped, enabling point-in-time-correct training sets that avoid leakage.

At the CLI level the surface is small — a feature group declares a record identifier, an event-time feature, and whether the online and/or offline store is enabled; you then `put-record` to write and `get-record` to read the online store. (Module 18 goes deep on Feature Store alongside the Model Registry and Model Monitor; this is the overview.)

```bash
aws sagemaker create-feature-group \
  --feature-group-name user-features \
  --record-identifier-feature-name user_id \
  --event-time-feature-name event_time \
  --feature-definitions '[{"FeatureName":"user_id","FeatureType":"String"},{"FeatureName":"event_time","FeatureType":"String"},{"FeatureName":"events_7d","FeatureType":"Integral"}]' \
  --online-store-config EnableOnlineStore=true \
  --offline-store-config S3StorageConfig={S3Uri=s3://my-ml-data/feature-store/} \
  --role-arn arn:aws:iam::123456789012:role/SageMakerRole
```

## The data lake pattern

These services compose into the standard AWS data lake: **raw and curated data in S3**, **schema in the Glue Data Catalog**, **SQL access via Athena and Redshift**, **transformation via Glue jobs**, and governance via **AWS Lake Formation**, which centralizes fine-grained (table/column/row-level) permissions across all of them. Newer tabular formats — **S3 Tables** with native Apache Iceberg — bring ACID transactions and schema evolution to lake data, so ML datasets can be updated safely without rewriting everything. The lake is the shared foundation; the feature store is the ML-specific layer built on top of it.

Lake Formation works in two moves. First you `register-resource` to bring an S3 location under Lake Formation's management (delegating access to its service role). Then, instead of hand-writing S3 bucket policies and IAM per table, you `grant-permissions` on catalog objects — and this is where the payoff for ML shows up: you can grant `SELECT` on only certain **columns** of a table, so a data scientist gets the feature columns but not the raw PII. The classic gotcha is forgetting that Lake Formation permissions layer *on top of* IAM — a principal needs both the LF grant and IAM access to the underlying API, and if a query returns "insufficient permissions" despite an IAM allow, an unregistered path or a missing LF grant is usually why.

```bash
# Bring an S3 prefix under Lake Formation management
aws lakeformation register-resource \
  --resource-arn arn:aws:s3:::my-ml-data/curated \
  --use-service-linked-role

# Grant column-level SELECT to a data-science role (PII columns withheld)
aws lakeformation grant-permissions \
  --principal DataLakePrincipalIdentifier=arn:aws:iam::123456789012:role/DataScientist \
  --permissions SELECT \
  --resource '{"TableWithColumns":{"DatabaseName":"ml_lake","Name":"events","ColumnNames":["user_id","events_7d","region"]}}'
```

## Choosing a data service

- Transactional app data, source of features → **RDS / Aurora**.
- Millisecond key-value lookups, online features → **DynamoDB**.
- Ad-hoc SQL and dataset building over S3 → **Athena**.
- Large-scale transformation and cataloging → **Glue**.
- Heavy analytics and warehouse joins → **Redshift**.
- Consistent train/serve features, point-in-time correctness → **SageMaker Feature Store**.

## How this fits the whole ML solution

This is the fuel supply for everything downstream. Ingestion lands data in S3; Glue catalogs and transforms it; Athena and Redshift query it; the feature store materializes features that both training and the live endpoint consume. Because training reads offline features and the endpoint reads online features from the *same* definitions, the whole system avoids the classic failure where a model that looked great in training degrades in production due to mismatched features. Getting this layer right is the difference between a model that generalizes and one that silently rots.

## Key takeaways

- RDS/Aurora hold transactional source data; DynamoDB serves millisecond key-value lookups and online features.
- Athena runs serverless SQL on S3 — partition and use Parquet to control cost; Glue catalogs and transforms data at scale.
- Redshift is the analytical warehouse; Lake Formation governs fine-grained access across the lake.
- SageMaker Feature Store gives synchronized online/offline features, preventing train/serve skew and enabling point-in-time-correct datasets.
- The data lake (S3 + Glue Catalog + Athena/Redshift, increasingly S3 Tables/Iceberg) is the shared foundation the feature store sits on.

## CLI cheat-sheet

```bash
# --- RDS / Aurora ---
aws rds create-db-instance --db-instance-identifier ml-db --engine postgres \
  --db-instance-class db.r6g.large --allocated-storage 100 --max-allocated-storage 1000 \
  --manage-master-user-password --master-username mladmin
aws rds create-db-cluster --db-cluster-identifier ml-aurora --engine aurora-postgresql \
  --manage-master-user-password --master-username mladmin \
  --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=16
aws rds create-db-instance --db-cluster-identifier ml-aurora \
  --db-instance-identifier ml-aurora-1 --engine aurora-postgresql --db-instance-class db.serverless
aws rds describe-db-instances --db-instance-identifier ml-db
aws rds create-db-snapshot --db-instance-identifier ml-db --db-snapshot-identifier ml-db-snap

# --- DynamoDB (online features) ---
aws dynamodb create-table --table-name user-features --billing-mode PAY_PER_REQUEST \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH
aws dynamodb update-table --table-name user-features --billing-mode PAY_PER_REQUEST
aws dynamodb put-item --table-name user-features --item file://item.json
aws dynamodb get-item --table-name user-features --key '{"user_id":{"S":"u_123"}}' --consistent-read
aws dynamodb query --table-name user-features \
  --key-condition-expression "user_id = :u" \
  --expression-attribute-values '{":u":{"S":"u_123"}}'
aws dynamodb batch-write-item --request-items file://batch.json   # <=25 items/call

# --- Athena (SQL on S3) ---
aws athena start-query-execution --query-string "SELECT ..." \
  --query-execution-context Database=ml_lake --work-group ml-analysts \
  --result-configuration OutputLocation=s3://my-ml-results/athena/
aws athena get-query-execution --query-execution-id <id>   # poll for SUCCEEDED + bytes scanned
aws athena get-query-results --query-execution-id <id>

# --- Glue (catalog + ETL) ---
aws glue create-database --database-input Name=ml_lake
aws glue create-crawler --name events-crawler --role AWSGlueServiceRole-ml \
  --database-name ml_lake --targets '{"S3Targets":[{"Path":"s3://my-ml-data/raw/"}]}'
aws glue start-crawler --name events-crawler
aws glue get-tables --database-name ml_lake
aws glue create-job --name curate --role AWSGlueServiceRole-ml --glue-version "4.0" \
  --worker-type G.1X --number-of-workers 10 \
  --command '{"Name":"glueetl","ScriptLocation":"s3://.../curate.py","PythonVersion":"3"}'
aws glue start-job-run --job-name curate
aws glue get-job-run --job-name curate --run-id <id>

# --- Redshift ---
aws redshift-serverless create-namespace --namespace-name ml-warehouse \
  --admin-username mladmin --manage-admin-password
aws redshift-serverless create-workgroup --workgroup-name ml-wg \
  --namespace-name ml-warehouse --base-capacity 32
aws redshift create-cluster --cluster-identifier ml-wh --node-type ra3.xlplus \
  --number-of-nodes 2 --master-username mladmin --manage-master-password   # provisioned

# --- Lake Formation (governance) ---
aws lakeformation register-resource --resource-arn arn:aws:s3:::my-ml-data/curated \
  --use-service-linked-role
aws lakeformation grant-permissions \
  --principal DataLakePrincipalIdentifier=arn:aws:iam::123456789012:role/DataScientist \
  --permissions SELECT \
  --resource '{"TableWithColumns":{"DatabaseName":"ml_lake","Name":"events","ColumnNames":["user_id","events_7d"]}}'

# --- Feature Store (overview; see module 18) ---
aws sagemaker create-feature-group --feature-group-name user-features \
  --record-identifier-feature-name user_id --event-time-feature-name event_time \
  --online-store-config EnableOnlineStore=true --role-arn <role> \
  --feature-definitions '[{"FeatureName":"user_id","FeatureType":"String"}, ...]'
aws sagemaker-featurestore-runtime put-record --feature-group-name user-features --record file://rec.json
aws sagemaker-featurestore-runtime get-record --feature-group-name user-features \
  --record-identifier-value-as-string u_123
```

## Try it

Land a raw CSV dataset in S3, run a Glue crawler to catalog it, then query it with Athena. Convert it to partitioned Parquet with a Glue job and re-query, comparing bytes scanned and cost. Then define a feature group in SageMaker Feature Store, ingest records into it, and read the same features from both the offline store (for a training set) and the online store (for a simulated inference lookup). Confirm the values match — that consistency is the entire point of a feature store.
