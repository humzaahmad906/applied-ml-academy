# 13 — Cost, Monitoring, and Best Practices

You can build a technically excellent ML system on Google Cloud and still fail — because it costs more than it earns, or because it silently degrades in production and nobody notices until a customer complains. Cost discipline and observability are not afterthoughts; they are what separate a demo from a production system you can defend in a review. This final module ties together the operational practices that make everything you have built sustainable, and ends with a mastery checklist.

## Cloud Monitoring and Logging

Google Cloud's operations suite — **Cloud Monitoring** and **Cloud Logging** — is the observability layer for everything you run. Monitoring collects metrics automatically from every service (Compute Engine CPU, Cloud Run request latency, Vertex AI prediction counts) and lets you build dashboards, alerting policies, and uptime checks. Logging aggregates structured logs from your workloads and lets you turn log patterns into metrics.

For ML endpoints specifically, Vertex AI exposes built-in metrics under the `aiplatform.googleapis.com` namespace — `prediction/online/prediction_count`, `prediction/online/error_count`, prediction latency, and accelerator (GPU) duty cycle and memory. These are the vital signs of a serving system: throughput, error rate, latency, and utilization.

```bash
# Create an alerting policy from a definition file
gcloud monitoring policies create --policy-from-file=high-latency-policy.json

# An uptime check against a serving endpoint's health route
gcloud monitoring uptime create serve-health \
  --resource-type=uptime-url \
  --hostname=fraud-serve-xyz.a.run.app --path=/healthz
```

Alerting policies fire on metric thresholds (latency over X ms), log-based conditions (a spike in error logs), or PromQL expressions, and notify via email, Slack, PagerDuty, Pub/Sub, or webhooks. **Log-based metrics** let you count occurrences of a pattern (for example, "model returned low-confidence prediction") and alert on it. Note that alerting is a billable feature, so scope policies to what actually matters.

## Model monitoring: the ML-specific observability

Infrastructure metrics tell you the endpoint is *up*; they do not tell you the model is still *right*. Models decay because the world changes — this is **drift** and **skew**, and catching it is uniquely an ML concern. **Vertex AI Model Monitoring** watches deployed models for:

- **Training-serving skew** — the distribution of features arriving in production differs from the distribution the model was trained on (a sign your training data was stale or unrepresentative).
- **Prediction drift** — feature distributions shift over time in production compared to an earlier window (the world moved on).
- **Feature attribution drift** — the relative importance of features changes, detected via Explainable AI.

Under the hood it compares distributions using statistical distances — **L-infinity distance** for categorical features and **Jensen-Shannon divergence** for numerical ones — against a baseline, and alerts when a per-feature threshold is exceeded. You configure a baseline (your training data), attach monitoring to an endpoint, set thresholds, and receive alerts when production input distributions drift. This is the early-warning system that tells you to retrain *before* accuracy visibly collapses. Pair it with a periodic offline eval on a frozen test set to confirm real accuracy.

## Cost controls

ML workloads are the most expensive thing most teams run on Google Cloud, and cost gets out of hand quietly. The controls, from foundational to advanced:

- **Budgets and alerts.** Set a budget on the billing account with threshold alerts (50/90/100%). Budgets *notify*; they do not cap. To actually stop spend, wire a budget's **Pub/Sub notification** to automation that disables billing or scales resources down.
- **Labels for attribution.** The labeling convention established early (`team`, `env`, `component`, `model`) flows into **billing export to BigQuery**, so you can answer "what did training cost last month, by model?" with SQL. Without labels, cost is an undifferentiated blob.
- **Right-size accelerators.** The single biggest lever. Do not serve on an A4 (B200) what an L4 (G2) handles. Do not train on eight H100s what one A100 finishes overnight. Match the chip to the job.
- **Spot VMs** for interruptible work (training with checkpointing, batch prediction, preprocessing) — up to ~90% off. Never for online serving.
- **Committed-use discounts** for steady baseline capacity (always-on serving) — up to ~40% off with 1–3 year commitments.
- **Scale to zero.** Cloud Run services (even GPU-backed) scale to zero when idle; Vertex endpoints can scale their minimum replicas down. Idle accelerators are pure waste.
- **BigQuery query cost.** Partition and cluster tables, prune columns, and prefer capacity slots for heavy, predictable workloads. A single `SELECT *` on a wide table can cost real money on on-demand billing.
- **Storage lifecycle.** Autoclass and lifecycle rules move cold data to cheaper tiers and delete transient artifacts automatically.
- **Turn things off.** Stop idle GPU VMs and development notebooks; they bill by the second whether or not they compute.

```sql
-- With billing export enabled, attribute last month's cost by label
SELECT
  (SELECT value FROM UNNEST(labels) WHERE key = 'component') AS component,
  ROUND(SUM(cost), 2) AS cost_usd
FROM `myco-fraud-dev.billing.gcp_billing_export_v1_XXXXXX`
WHERE DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY component ORDER BY cost_usd DESC;
```

## Best practices for production ML on Google Cloud

Pulling the course together, the habits that make a system production-grade:

- **Reproducibility.** Containers in Artifact Registry, pipelines defined in code, infrastructure as Terraform. Anyone should be able to rebuild the system from source.
- **Least privilege everywhere.** Dedicated service accounts per workload, no key files (impersonation / Workload Identity Federation), secrets in Secret Manager.
- **Co-locate compute and data** in one region; keep sensitive data inside a VPC Service Controls perimeter with private networking.
- **Automate the lifecycle.** CI/CD builds and tests images; pipelines orchestrate train → evaluate → register → deploy; monitoring triggers retraining.
- **Freeze your eval set** and gate deployment on it, so "the model improved" is a measured fact, not folklore.
- **Observe both layers** — infrastructure metrics *and* model drift — and alert on both.
- **Attribute and cap cost** with labels, budgets, and automation from day one.

## How this fits the whole solution

Cost and monitoring are the closed loop around the entire end-to-end system. Monitoring watches the ingestion, training, and serving stages you built; model monitoring watches the models themselves; alerts feed back into retraining and incident response; and cost controls with labeled billing keep the whole thing economically viable. A system without this loop is running blind and spending unbounded. With it, you have something you can operate, defend, and scale — which is what mastery of Google Cloud for ML actually means.

## Key takeaways

- **Cloud Monitoring/Logging** covers infrastructure vitals (latency, errors, GPU duty cycle) via dashboards, alerting policies, and log-based metrics; Vertex exposes prediction metrics under `aiplatform.googleapis.com`.
- **Vertex AI Model Monitoring** is the ML-specific layer: it detects **training-serving skew, prediction drift, and attribution drift** (via L-infinity and Jensen-Shannon distances) and alerts you to retrain before accuracy visibly drops.
- **Cost control** is dominated by right-sizing accelerators, **Spot** for interruptible work, **committed-use** for steady serving, **scale-to-zero**, BigQuery partitioning, and storage lifecycle — with **budgets + labeled billing export** for visibility and Pub/Sub automation for enforcement.
- Production mastery = **reproducibility (IaC + containers + pipelines), least privilege, co-located and private data, an automated lifecycle, a frozen eval gate, dual-layer observability, and cost attribution** from day one.

## Try it

Close the loop on a system you have deployed:

1. Enable **billing export to BigQuery**, then run a query that breaks down the last 30 days of cost by your `component` label. Identify the single most expensive component.
2. Create a **budget** with alerts at 50/90/100% and, as a stretch, wire its Pub/Sub notification to a function that logs a warning.
3. Build a Cloud Monitoring **dashboard** for a serving endpoint showing request count, error rate, p95 latency, and GPU duty cycle, and add an **alerting policy** on p95 latency.
4. Enable **Vertex AI Model Monitoring** on a deployed model with your training data as the baseline, set a drift threshold on one feature, then send skewed inputs and confirm you get a drift alert. Reflect on what retraining trigger you would wire to that alert.
