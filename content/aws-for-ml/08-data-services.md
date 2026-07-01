# 08 — Data Services

Models are only as good as the data plumbing behind them. Before a feature ever reaches a training job it has usually passed through operational databases, been catalogued in a data lake, queried and aggregated, and materialized into a feature store. AWS offers a distinct service for each of these jobs, and knowing which to reach for — and how they compose into a data lake and feature pipeline — is what separates a demo from a production ML system. This module surveys the data services an ML engineer touches most.

## Relational data: RDS and Aurora

**Amazon RDS** is managed relational databases — PostgreSQL, MySQL, MariaDB, SQL Server, Oracle — where AWS handles patching, backups, and failover. **Amazon Aurora** is AWS's cloud-native engine compatible with PostgreSQL and MySQL, with higher performance and storage that auto-scales. **Aurora Serverless v2** scales capacity up and down automatically, which suits spiky or unpredictable workloads. For ML, RDS/Aurora is where transactional application data lives — the source of truth you extract features from, and often the online store for low-latency lookups when a feature store is overkill.

## Key-value at scale: DynamoDB

**Amazon DynamoDB** is a fully managed NoSQL key-value and document database with single-digit-millisecond latency at any scale. You choose **on-demand** capacity (pay per request, scales instantly — the safe default) or **provisioned** capacity (cheaper for steady, predictable load). In ML systems DynamoDB shines as an **online feature store** or a low-latency lookup layer: an inference request arrives, you fetch the entity's precomputed features from DynamoDB by key in milliseconds, and pass them to the model.

```python
import boto3
ddb = boto3.resource("dynamodb").Table("user-features")
item = ddb.get_item(Key={"user_id": "u_123"})["Item"]   # ms-latency feature fetch
```

## Query on S3: Athena

**Amazon Athena** runs standard SQL directly over data sitting in S3, using a serverless Presto/Trino engine — no cluster to manage, billed per data scanned. It reads CSV, JSON, Parquet, ORC, and Iceberg tables. For ML, Athena is the workhorse for exploratory analysis, building training datasets with SQL, and computing aggregate features over historical data. Two practices control cost and speed: store data in **columnar Parquet** and **partition** it (for example by date), so a query scans only the relevant slices instead of the whole dataset.

```sql
SELECT user_id, COUNT(*) AS events_7d
FROM events
WHERE dt BETWEEN date_add('day', -7, current_date) AND current_date
GROUP BY user_id;   -- partitioned on dt, columnar Parquet: scans only 7 partitions
```

## ETL and cataloging: Glue

**AWS Glue** is the serverless ETL and metadata layer. The **Glue Data Catalog** is a central schema registry that Athena, Redshift, and Spark all read from — it is what turns a pile of S3 files into queryable "tables." **Glue crawlers** infer schema from S3 data and populate the catalog automatically. **Glue jobs** run Spark (or Python shell) to transform data at scale — cleaning, joining, and reshaping raw data into curated, feature-ready datasets. Glue is the standard tool for the "T" in the ELT that feeds training.

## Analytics warehouse: Redshift

For heavy analytical queries and joins across large curated datasets, **Amazon Redshift** is the columnar data warehouse; **Redshift Serverless** runs it without managing clusters. In an ML platform Redshift often holds the modeled, warehouse-grade data that feature pipelines read from, and it integrates with the same Glue Catalog and S3 so the lake and warehouse share one view of the data.

## Feature data: SageMaker Feature Store

**Amazon SageMaker Feature Store** is the purpose-built home for ML features, solving the notorious train/serve skew problem. It has two synchronized faces: an **online store** for low-latency reads at inference time, and an **offline store** (in S3) for building training datasets and backfills. You write a feature once and read the *same* definition in both training and serving, so the features a model trains on match the features it sees in production. Feature groups are versioned and time-stamped, enabling point-in-time-correct training sets that avoid leakage.

## The data lake pattern

These services compose into the standard AWS data lake: **raw and curated data in S3**, **schema in the Glue Data Catalog**, **SQL access via Athena and Redshift**, **transformation via Glue jobs**, and governance via **AWS Lake Formation**, which centralizes fine-grained (table/column/row-level) permissions across all of them. Newer tabular formats — **S3 Tables** with native Apache Iceberg — bring ACID transactions and schema evolution to lake data, so ML datasets can be updated safely without rewriting everything. The lake is the shared foundation; the feature store is the ML-specific layer built on top of it.

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

## Try it

Land a raw CSV dataset in S3, run a Glue crawler to catalog it, then query it with Athena. Convert it to partitioned Parquet with a Glue job and re-query, comparing bytes scanned and cost. Then define a feature group in SageMaker Feature Store, ingest records into it, and read the same features from both the offline store (for a training set) and the online store (for a simulated inference lookup). Confirm the values match — that consistency is the entire point of a feature store.
