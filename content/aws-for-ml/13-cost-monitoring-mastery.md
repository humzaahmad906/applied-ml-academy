# 13 — Cost, Monitoring, and the Well-Architected Mindset

The difference between an ML engineer who can train a model and one who can *own* a system in production is operational maturity: seeing what the system is doing, keeping the bill sane, and reasoning about the whole thing against a shared standard. This final module covers observability with CloudWatch, the cost controls that keep ML economical, and the AWS Well-Architected Framework — ending with a mastery checklist that ties the entire course together.

## Observability with CloudWatch

**Amazon CloudWatch** is the observability backbone. Four capabilities matter for ML:

- **Metrics** — numeric time series. AWS services emit them automatically (endpoint invocations, latency, instance CPU/GPU utilization, error rates), and you publish custom ones (prediction confidence, feature staleness).
- **Logs** — every training job, Lambda, and endpoint writes logs to CloudWatch Logs. **Logs Insights** lets you query them with a purpose-built query language — indispensable when a training job fails or an endpoint returns errors.
- **Alarms** — thresholds on metrics that trigger actions: page an on-call, scale an endpoint, or fire an EventBridge event to start retraining when Model Monitor reports drift.
- **Dashboards** — a single pane showing the health of the whole pipeline: ingestion volume, training success, endpoint latency and error rate, drift status, and spend.

```bash
# Alarm when endpoint p99 latency exceeds 500 ms for 3 consecutive minutes
aws cloudwatch put-metric-alarm \
  --alarm-name fraud-scorer-latency \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=fraud-scorer Name=VariantName,Value=AllTraffic \
  --statistic p99 --period 60 --evaluation-periods 3 \
  --threshold 500000 --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:us-east-1:<acct>:ml-oncall
```

For ML specifically, watch GPU utilization on endpoints — a chronically underutilized GPU endpoint is money burning, and the fix is often inference components or a smaller instance. Pair CloudWatch with **Model Monitor** so data/prediction drift shows up alongside infrastructure metrics.

## Cost control

ML is expensive in ways web apps are not — GPUs, large data transfer, always-on endpoints — so cost discipline is an engineering skill, not a finance afterthought.

- **Cost Explorer** visualizes spend over time and by dimension; **AWS Budgets** sets thresholds that alert (or act) before you overspend; the **Cost and Usage Report** gives the raw detail for deep analysis.
- **Tagging** is the foundation of all cost attribution. Tag every resource with team, project, and environment, and you can answer "what did this model cost to train and serve?" Untagged resources are invisible in cost reports — enforce tags with policy.
- **Pricing model per workload** is the biggest lever: **Spot / managed spot** for checkpointed training (up to ~90% off), **Savings Plans** for steady baseline inference, **Capacity Blocks** to guarantee scarce GPUs, and **On-Demand** only for unpredictable bursts.
- **Right inference option** matters as much: serverless and asynchronous inference scale to zero, so intermittent workloads should never sit on an always-on real-time endpoint. Multi-model endpoints and inference components pack many models onto shared instances.
- **Storage lifecycle** moves cold data to Glacier and expires temporary artifacts automatically.
- **VPC endpoints** cut NAT data-processing charges on large training reads.

The recurring failure mode is the forgotten resource: an idle GPU endpoint, an oversized notebook instance, an un-lifecycled bucket of old checkpoints. Budgets and tagged dashboards catch these before the bill does.

## The Well-Architected Framework

AWS distills operational wisdom into the **Well-Architected Framework**, built on **six pillars**:

1. **Operational Excellence** — run and improve systems; automate, monitor, learn from failure. (Your CI/CD and IaC live here.)
2. **Security** — protect data, models, and infrastructure. (IAM least privilege, VPC, KMS, Secrets Manager.)
3. **Reliability** — recover from failure and meet demand. (Multi-AZ, checkpointing, autoscaling, retries.)
4. **Performance Efficiency** — use resources efficiently. (Right instance families, accelerators, input modes, GPU utilization.)
5. **Cost Optimization** — deliver value at the lowest price. (Everything in the section above.)
6. **Sustainability** — minimize environmental impact. (Right-sizing, efficient accelerators like Trainium/Inferentia, Spot to use spare capacity, scaling to zero.)

AWS also publishes **lenses** that apply the pillars to specific domains — including a **Machine Learning Lens**, a **Generative AI Lens**, and a **Responsible AI Lens** — which translate each pillar into concrete ML practices (data lineage, bias evaluation, model monitoring, safe deployment). Treat the pillars as a review checklist: for any design, ask what each pillar demands and whether you have addressed it.

## Mastery checklist

You have mastered AWS for ML when you can, for a new project, confidently:

- Pick the Region and confirm the accelerator capacity you need exists there.
- Design per-workload IAM roles with least privilege, and keep secrets in Secrets Manager and data encrypted with KMS.
- Lay out an S3 data lake with the right storage classes, lifecycle rules, and a Glue-catalogued, partitioned Parquet/Iceberg structure.
- Put workloads in a VPC with private subnets and S3/PrivateLink endpoints, and know why that is both safer and cheaper.
- Choose the compute — EC2 vs SageMaker vs containers vs Lambda — and the pricing model (Spot, Savings Plans, Capacity Blocks) that fit each stage.
- Run reproducible, tracked training (managed spot, distributed or HyperPod, MLflow) that writes a registered, gated artifact.
- Choose the right inference option per workload (real-time, serverless, async, batch, or Bedrock) and autoscale it, including to zero.
- Orchestrate the whole lifecycle with SageMaker Pipelines / Step Functions and close the loop with drift-triggered retraining.
- Deploy it all as code, observe it with CloudWatch dashboards and alarms, attribute cost with tags, and review it against the six pillars.

## How this fits the whole ML solution

Cost and observability are not a final chapter bolted on — they are the feedback signals that let the system be operated safely over time. CloudWatch tells you the system is healthy; drift monitoring tells you the model still fits reality; cost tools tell you it is economical; the Well-Architected pillars tell you whether the design will hold under security, reliability, and scale pressure. Together they turn the reference architecture into something you can run in production for years, not just demo once. That operational ownership is the actual mastery this course was building toward.

## Key takeaways

- CloudWatch (metrics, logs, alarms, dashboards) plus Model Monitor gives full observability; alarms can trigger scaling and retraining.
- Control cost with tagging, Cost Explorer/Budgets, the right pricing model (Spot/Savings Plans/Capacity Blocks), scale-to-zero inference, storage lifecycle, and VPC endpoints.
- The forgotten idle GPU endpoint is the classic ML cost leak — dashboards and budgets catch it early.
- The Well-Architected Framework's six pillars (Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimization, Sustainability) plus the ML/GenAI/Responsible AI lenses are your design review checklist.
- Mastery is the ability to reason about a full ML system across all pillars and operate it economically over time.

## Try it

Build a CloudWatch dashboard for a deployed system showing endpoint invocations, latency, error rate, and GPU utilization, and set alarms on latency and error rate. Enable cost allocation tags, tag your ML resources by project, and use Cost Explorer to break down spend by tag. Then run a personal Well-Architected review of your end-to-end system from the previous work: write one concrete improvement for each of the six pillars. Implement the cheapest, highest-impact one — most often, switching an idle real-time endpoint to serverless or asynchronous inference.
