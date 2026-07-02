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

For ML specifically, watch GPU utilization on endpoints — a chronically underutilized GPU endpoint is money burning, and the fix is often inference components or a smaller instance. Pair CloudWatch with **Model Monitor** so data/prediction drift shows up alongside infrastructure metrics. (This module keeps observability at the level needed to *act on cost*; metrics, EMF, Logs Insights, and X-Ray get their own dedicated module.)

Two CLI habits carry most of the operational weight. `put-dashboard` builds the single-pane view as code (a JSON body you keep in the repo, not clicks in the console), so a new environment gets the same dashboard on deploy. And when triaging a failed job or a spike, `logs start-query` runs a Logs Insights query across log groups and returns a query id you poll with `get-query-results` — far faster than scrolling raw log streams.

```bash
# Publish a dashboard from a versioned JSON body
aws cloudwatch put-dashboard \
  --dashboard-name ml-platform \
  --dashboard-body file://dashboard.json

# Find the endpoints that errored in the last hour (Logs Insights)
QID=$(aws logs start-query \
  --log-group-name /aws/sagemaker/Endpoints/fraud-scorer \
  --start-time $(date -v-1H +%s) --end-time $(date +%s) \
  --query-string 'fields @timestamp, @message | filter @message like /Error/ | limit 50' \
  --query queryId --output text)
aws logs get-query-results --query-id "$QID"
```

## Cost control

ML is expensive in ways web apps are not — GPUs, large data transfer, always-on endpoints — so cost discipline is an engineering skill, not a finance afterthought.

- **Cost Explorer** visualizes spend over time and by dimension; **AWS Budgets** sets thresholds that alert (or act) before you overspend; the **Cost and Usage Report** gives the raw detail for deep analysis.

Everything Cost Explorer shows in the console is available from `aws ce`, which is what lets you script a weekly cost report or wire spend into a dashboard. `get-cost-and-usage` is the workhorse: it takes a `--time-period`, a `--granularity` (DAILY/MONTHLY), one or more `--metrics` (`UnblendedCost` is the number on your bill), and a `--group-by` that is the whole game — group by `Type=DIMENSION,Key=SERVICE` to see which service dominates, or by `Type=TAG,Key=Project` to answer "what did this model cost." `get-cost-forecast` projects the rest of the month so a budget alert isn't the first warning, and `get-dimension-values` enumerates what you can filter on (e.g. every `SERVICE` value that has spend). A gotcha worth internalizing: the Cost Explorer API bills roughly $0.01 per paginated request, so a tight polling loop over it is itself a (small) cost line — cache results, don't hammer it.

```bash
# Month-to-date spend broken down by service
aws ce get-cost-and-usage \
  --time-period Start=2026-07-01,End=2026-07-31 \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# "What did this project cost?" — group by a cost-allocation tag
aws ce get-cost-and-usage \
  --time-period Start=2026-07-01,End=2026-07-31 \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=TAG,Key=Project

# Project the rest of the month before the bill surprises you
aws ce get-cost-forecast \
  --time-period Start=2026-07-03,End=2026-07-31 \
  --metric UNBLENDED_COST --granularity MONTHLY
```
- **Tagging** is the foundation of all cost attribution. Tag every resource with team, project, and environment, and you can answer "what did this model cost to train and serve?" Untagged resources are invisible in cost reports — enforce tags with policy.

There is a two-step subtlety that trips up almost everyone: applying a tag to a resource is *not* enough for it to appear as a `--group-by TAG` dimension in Cost Explorer. You must also **activate** the tag key as a cost-allocation tag with `ce update-cost-allocation-tags-status`, and — this is the sharp edge — activation is not retroactive, so cost data only splits by that tag from the activation date forward. Activate your `Project`/`Team`/`Environment` keys on day one. To find and fix the resources that slipped through untagged, `resourcegroupstaggingapi get-resources --tag-filters` audits what carries a given tag, and `tag-resources` bulk-applies tags by ARN across services in one call.

```bash
# Bulk-apply tags across services by ARN
aws resourcegroupstaggingapi tag-resources \
  --resource-arn-list <endpoint-arn> <training-job-arn> \
  --tags Project=fraud,Team=risk,Environment=prod

# Audit: which resources are missing the Project tag
aws resourcegroupstaggingapi get-resources \
  --tag-filters Key=Project

# Activate the keys as cost-allocation tags (NOT retroactive — do this early)
aws ce update-cost-allocation-tags-status \
  --cost-allocation-tags-status \
  TagKey=Project,Status=Active TagKey=Team,Status=Active TagKey=Environment,Status=Active
```
- **Pricing model per workload** is the biggest lever: **Spot / managed spot** for checkpointed training (up to ~90% off), **Savings Plans** for steady baseline inference, **Capacity Blocks** to guarantee scarce GPUs, and **On-Demand** only for unpredictable bursts.
- **Right inference option** matters as much: serverless and asynchronous inference scale to zero, so intermittent workloads should never sit on an always-on real-time endpoint. Multi-model endpoints and inference components pack many models onto shared instances.
- **Storage lifecycle** moves cold data to Glacier and expires temporary artifacts automatically.
- **VPC endpoints** cut NAT data-processing charges on large training reads.

The recurring failure mode is the forgotten resource: an idle GPU endpoint, an oversized notebook instance, an un-lifecycled bucket of old checkpoints. Budgets and tagged dashboards catch these before the bill does.

A budget is the guardrail that turns "we should watch spend" into an automatic alert. `budgets create-budget` takes the budget definition (amount, period, and any filters) plus one or more `--notifications-with-subscribers` that fire an SNS or email alert at, say, 80% and 100% of the threshold — and, critically, you can alert on *forecasted* spend so you hear about a runaway training run mid-month, not after. `describe-budgets` lists what's configured.

```bash
# Alert at 80% actual and 100% forecasted of a $5k monthly ML budget
aws budgets create-budget \
  --account-id <acct> \
  --budget '{"BudgetName":"ml-monthly","BudgetLimit":{"Amount":"5000","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80},"Subscribers":[{"SubscriptionType":"SNS","Address":"arn:aws:sns:us-east-1:<acct>:ml-cost"}]},{"Notification":{"NotificationType":"FORECASTED","ComparisonOperator":"GREATER_THAN","Threshold":100},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"ml-oncall@example.com"}]}]'

aws budgets describe-budgets --account-id <acct>
```

Two services turn "is anything oversized?" from a guess into a report. **Compute Optimizer** analyzes utilization and recommends right-sizing — `get-ec2-instance-recommendations` returns, per instance, whether it is `Overprovisioned`/`Underprovisioned`/`Optimized` and the cheaper instance it suggests, which is the fastest way to catch the notebook or training box that is three sizes too big. **Trusted Advisor** (via the `support` API, which requires a Business or Enterprise support plan) has checks for idle load balancers, underutilized instances, and unassociated EIPs — its cost-optimization checks are exactly the "forgotten resource" hunt automated.

```bash
# Right-sizing: which instances are over/under-provisioned
aws compute-optimizer get-ec2-instance-recommendations

# Trusted Advisor cost checks (needs Business+ support; API lives under `support`)
aws support describe-trusted-advisor-checks --language en
aws support describe-trusted-advisor-check-result --check-id <cost-check-id>
```

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

## CLI cheat-sheet

```bash
# ── Cost Explorer: where the money goes ──
aws ce get-cost-and-usage --time-period Start=2026-07-01,End=2026-07-31 \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE          # spend by service
aws ce get-cost-and-usage --time-period Start=2026-07-01,End=2026-07-31 \
  --granularity MONTHLY --metrics UnblendedCost \
  --group-by Type=TAG,Key=Project                # spend by project tag
aws ce get-cost-forecast --time-period Start=2026-07-03,End=2026-07-31 \
  --metric UNBLENDED_COST --granularity MONTHLY  # projected month-end
aws ce get-dimension-values --time-period Start=2026-07-01,End=2026-07-31 \
  --dimension SERVICE                            # what you can filter/group on

# ── Budgets: automatic guardrails (alert on actual AND forecasted) ──
aws budgets create-budget --account-id ACCT \
  --budget '{"BudgetName":"ml-monthly","BudgetLimit":{"Amount":"5000","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers file://notifications.json
aws budgets describe-budgets --account-id ACCT

# ── Cost-allocation tags: apply, audit, then ACTIVATE (not retroactive) ──
aws resourcegroupstaggingapi tag-resources --resource-arn-list ARN1 ARN2 \
  --tags Project=fraud,Team=risk,Environment=prod
aws resourcegroupstaggingapi get-resources --tag-filters Key=Project
aws ce update-cost-allocation-tags-status \
  --cost-allocation-tags-status TagKey=Project,Status=Active

# ── Right-sizing & waste hunt ──
aws compute-optimizer get-ec2-instance-recommendations
aws support describe-trusted-advisor-checks --language en     # Business+ support
aws support describe-trusted-advisor-check-result --check-id CHECK_ID

# ── Observability to act on cost (full observability = its own module) ──
aws cloudwatch put-metric-alarm --alarm-name idle-endpoint \
  --namespace AWS/SageMaker --metric-name Invocations \
  --dimensions Name=EndpointName,Value=fraud-scorer \
  --statistic Sum --period 3600 --evaluation-periods 6 \
  --threshold 1 --comparison-operator LessThanThreshold \
  --alarm-actions arn:aws:sns:us-east-1:ACCT:ml-cost   # near-zero traffic = idle spend
aws cloudwatch put-dashboard --dashboard-name ml-platform --dashboard-body file://dashboard.json
aws cloudwatch get-metric-data --metric-data-queries file://queries.json \
  --start-time 2026-07-01T00:00:00Z --end-time 2026-07-02T00:00:00Z
aws logs start-query --log-group-name /aws/sagemaker/Endpoints/fraud-scorer \
  --start-time START --end-time END \
  --query-string 'fields @timestamp,@message | filter @message like /Error/'
aws logs get-query-results --query-id QID
```

## Try it

Build a CloudWatch dashboard for a deployed system showing endpoint invocations, latency, error rate, and GPU utilization, and set alarms on latency and error rate. Enable cost allocation tags, tag your ML resources by project, and use Cost Explorer to break down spend by tag. Then run a personal Well-Architected review of your end-to-end system from the previous work: write one concrete improvement for each of the six pillars. Implement the cheapest, highest-impact one — most often, switching an idle real-time endpoint to serverless or asynchronous inference.
