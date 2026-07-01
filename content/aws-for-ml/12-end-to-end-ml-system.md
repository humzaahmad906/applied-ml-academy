# 12 — Building an End-to-End ML System on AWS

Every prior topic was a component. This module wires them into one production ML system and shows how many AWS services compose into a whole. The goal is the mental model of a complete solution — data flowing from ingestion through a lake, into features, into training, through a registry, out to deployment, and back through monitoring into retraining — plus a reference architecture and infrastructure-as-code you can adapt. We also fold in the generative-AI path via Amazon Bedrock, since real systems increasingly serve both classical models and foundation models.

## The reference architecture

Read the system as five planes, each a stage in the lifecycle:

**1. Ingestion and streaming.** Data arrives two ways. Batch data lands in S3 directly or via scheduled pulls. Streaming data flows through **Amazon Kinesis Data Streams** or **Amazon MSK** (managed Kafka), with **Amazon Data Firehose** delivering it into S3, Redshift, or Iceberg tables with minimal code. An S3 `ObjectCreated` event or an EventBridge rule kicks off downstream processing.

**2. Data lake and warehouse.** Raw data lands in an S3 lake, is catalogued by the **Glue Data Catalog**, transformed by **Glue** jobs into curated Parquet/Iceberg (**S3 Tables**), queried with **Athena**, and modeled in **Amazon Redshift** for heavy analytics. **AWS Lake Formation** governs fine-grained access across all of it.

**3. Features.** Curated data feeds feature pipelines (Glue or SageMaker Processing) that write to **SageMaker Feature Store** — the offline store (S3) for building point-in-time-correct training sets, the online store (backed by DynamoDB-class latency) for millisecond lookups at inference. Same definitions in both places, so no train/serve skew.

**4. Train, register, deploy.** A **SageMaker Pipeline** trains (managed spot, distributed as needed), evaluates, gates on a quality/bias threshold via **Clarify**, and registers the winner in the **Model Registry**. Deployment reads the approved version and serves it through the right SageMaker inference option — **real-time** behind the API, **serverless** for spiky internal use, **asynchronous** for heavy jobs, **batch transform** for nightly scoring. For generative AI, **Amazon Bedrock** provides managed foundation models (Anthropic Claude, Amazon Nova, Meta Llama, Mistral, and others) via the unified **Converse** API, with **Knowledge Bases** for RAG, **Guardrails** for safety, and **Agents** for orchestration — no model to host yourself.

**5. Serving front door and monitoring.** **API Gateway** exposes the system; **Lambda** authenticates, validates, and routes — calling the SageMaker endpoint or Bedrock as appropriate. **CloudWatch** collects metrics and logs; **SageMaker Model Monitor** watches for data and prediction drift; a drift alarm fires an **EventBridge** event that restarts the pipeline. The loop closes.

Threaded through all five planes: **IAM** roles per workload (least privilege), a **VPC** with private subnets and S3/PrivateLink **endpoints** keeping traffic private, **KMS** encryption on data at rest, **Secrets Manager** for credentials, cost tags on everything, and CI/CD deploying the whole stack.

```
                    ┌─────────── EventBridge (schedule / drift trigger) ───────────┐
                    │                                                              │
 sources ─▶ Kinesis/MSK ─▶ Firehose ─▶  S3 data lake  ─▶ Glue/Athena/Redshift      │
 batch ───────────────────────────────▶ (Glue Catalog, Lake Formation, S3 Tables) │
                                              │                                    │
                                              ▼                                    │
                                     SageMaker Feature Store                       │
                                     (offline S3 / online low-latency)             │
                                              │                                    │
                                              ▼                                    │
                             SageMaker Pipeline: train ▶ eval ▶ gate ▶ register ───┘
                                              │ (Model Registry, approved)
                                              ▼
            ┌───────── real-time / serverless / async / batch endpoints ──────────┐
 client ─▶ API Gateway ─▶ Lambda ─▶ ┤  or  Amazon Bedrock (Claude/Nova, RAG, Guardrails) │
            └──────────────────────── CloudWatch + Model Monitor (drift) ─────────┘
                  everything inside a VPC · IAM roles · KMS · tags · CI/CD
```

## The request path, concretely

At inference time for a classical model: a client hits **API Gateway**, which invokes a **Lambda** that authenticates, fetches the entity's features from the **Feature Store online store**, calls the **SageMaker real-time endpoint**, logs the request/response for **Model Monitor**, and returns the prediction. For a generative feature, the same Lambda instead calls **Bedrock's Converse API** — optionally grounded by a **Knowledge Base** and filtered by a **Guardrail** — and streams the response back. One front door, two model backends, shared auth and observability.

## Infrastructure as code

None of this should be built by clicking. Codify it with **AWS CloudFormation**, the **AWS CDK** (CloudFormation in Python/TypeScript), or **Terraform**, so the whole environment is versioned, reviewable, and reproducible across dev/staging/prod. A CDK snippet provisioning core pieces — the data bucket, a governed table, and an endpoint's scaling — reads like ordinary code:

```python
from aws_cdk import App, Stack, RemovalPolicy, aws_s3 as s3, aws_iam as iam
from aws_cdk import aws_sagemaker as sm
from constructs import Construct

class MlPlatformStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kw):
        super().__init__(scope, cid, **kw)

        data = s3.Bucket(self, "MlData",
            versioned=True,
            encryption=s3.BucketEncryption.KMS_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[s3.LifecycleRule(
                transitions=[s3.Transition(
                    storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                    transition_after=Duration.days(30))])])

        exec_role = iam.Role(self, "SmExecRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"))
        data.grant_read_write(exec_role)   # least-privilege, scoped to this bucket

        # Endpoint config / autoscaling, pipeline, monitoring, VPC endpoints,
        # and the API Gateway + Lambda front door follow the same declarative style.

App().synth()
```

Deploy it with `cdk deploy`; tear down a whole environment with `cdk destroy`. The same repository that holds the model code holds the infrastructure, so a code review covers both.

## Design principles that hold the system together

- **One artifact, many stages.** A DLC-based container in ECR is used for training and serving; a registered model version is the single source of truth for what deploys.
- **Same features everywhere.** The feature store's shared definitions are what keep training and production honest.
- **Everything triggered, nothing manual.** EventBridge schedules and drift alarms drive retraining; the system runs itself.
- **Private by default, least privilege always.** VPC endpoints, per-workload roles, KMS, and Secrets Manager are non-negotiable, not add-ons.
- **Cost and observability are first-class.** Tags, CloudWatch dashboards, and the right inference option per workload are part of the design, not a cleanup pass.

## How this fits the whole ML solution

This *is* the whole ML solution — the module where the components stop being a catalog and become a system. The value is not any single service but the wiring: ingestion feeding the lake, the lake feeding features, features feeding training, the registry gating deployment, the endpoint (or Bedrock) serving, and monitoring feeding retraining, all under one security and cost umbrella, all deployed as code. Once you can draw this diagram from memory and map each arrow to a service, you can design an ML platform, not just train a model.

## Key takeaways

- A production ML system is five planes — ingest/stream, lake/warehouse, features, train/register/deploy, serve/monitor — composed from many AWS services.
- Streaming (Kinesis/MSK/Firehose) and batch both land in an S3 lake governed by Glue Catalog, Athena/Redshift, and Lake Formation.
- The feature store's shared online/offline definitions eliminate train/serve skew; the model registry is the deploy gate.
- Serve classical models via SageMaker inference options and generative features via Bedrock (Converse, Knowledge Bases, Guardrails) behind one API Gateway/Lambda front door.
- Build it all with IaC (CloudFormation/CDK/Terraform), inside a VPC, with least-privilege IAM, KMS, tags, CI/CD, and a drift-driven retraining loop.

## Try it

Draw the reference architecture for a real use case of your own, labeling every arrow with the AWS service that carries it. Then implement a thin vertical slice with the CDK: an S3 data bucket, a SageMaker execution role scoped to it, one endpoint, and an API Gateway + Lambda front door that calls the endpoint. Add a second Lambda route that calls Bedrock's Converse API for a generative response. Deploy with `cdk deploy`, exercise both routes, then `cdk destroy` — proving you can stand the whole environment up and tear it down as code.
