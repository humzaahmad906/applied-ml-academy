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

Alerting policies fire on metric thresholds (latency over X ms), log-based conditions (a spike in error logs), or PromQL expressions, and notify via email, Slack, PagerDuty, Pub/Sub, or webhooks. **Log-based metrics** let you count occurrences of a pattern (for example, "model returned low-confidence prediction") and alert on it. Note that alerting is a billable feature, so scope policies to what actually matters. (Module 15 covers Cloud Monitoring, Logging, Error Reporting, and Trace in depth; this module keeps observability at the summary altitude and focuses on cost.)

## Model monitoring: the ML-specific observability

Infrastructure metrics tell you the endpoint is *up*; they do not tell you the model is still *right*. Models decay because the world changes — this is **drift** and **skew**, and catching it is uniquely an ML concern. **Vertex AI Model Monitoring** watches deployed models for:

- **Training-serving skew** — the distribution of features arriving in production differs from the distribution the model was trained on (a sign your training data was stale or unrepresentative).
- **Prediction drift** — feature distributions shift over time in production compared to an earlier window (the world moved on).
- **Feature attribution drift** — the relative importance of features changes, detected via Explainable AI.

Under the hood it compares distributions using statistical distances — **L-infinity distance** for categorical features and **Jensen-Shannon divergence** for numerical ones — against a baseline, and alerts when a per-feature threshold is exceeded. You configure a baseline (your training data), attach monitoring to an endpoint, set thresholds, and receive alerts when production input distributions drift. This is the early-warning system that tells you to retrain *before* accuracy visibly collapses. Pair it with a periodic offline eval on a frozen test set to confirm real accuracy. (Model monitoring is summarized here for completeness; module 17 covers the Vertex AI Feature Store and Experiments that feed and complement it in depth.)

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

### Budgets: notify, and automate the enforcement

Budgets live on the **billing account**, not the project, and you create them from the CLI so they are reproducible. A budget always *notifies*; to *enforce*, you wire its Pub/Sub notification to automation. The key flags: `--budget-amount` (a fixed amount, or `--last-period-amount` to track the previous period), one or more `--threshold-rule` entries (a percent plus a `basis` of `current-spend` or `forecasted-spend`), scoping with `--filter-projects` and/or `--filter-labels`, and `--notifications-rule-pubsub-topic` to emit a machine-readable message on every threshold crossing.

```bash
# Forecast- and actual-based thresholds, scoped to prod, publishing to Pub/Sub
gcloud billing budgets create \
  --billing-account=0X0X0X-0X0X0X-0X0X0X \
  --display-name="fraud-prod monthly" \
  --budget-amount=5000USD \
  --filter-projects=projects/myco-fraud-prod \
  --filter-labels=env=prod \
  --threshold-rule=percent=0.5,basis=current-spend \
  --threshold-rule=percent=0.9,basis=current-spend \
  --threshold-rule=percent=1.0,basis=forecasted-spend \
  --notifications-rule-pubsub-topic=projects/myco-fraud-prod/topics/budget-alerts
```

The Pub/Sub message is what turns a budget from a warning into a kill-switch: a function subscribed to `budget-alerts` can, at 100%, cap Vertex endpoint replicas, stop idle notebook VMs, or (in extreme cases) detach the billing account. Budgets *notify* by default; automation is what makes them *cap*.

### Billing export: the SQL you actually run

With **billing export to BigQuery** enabled, the `gcp_billing_export_v1_*` table is your source of truth. The label-attribution query is the starting point; the ones you run in practice slice by service, by SKU, by credits, and forward in time.

```sql
-- Cost by service last 30 days (where is the money actually going?)
SELECT service.description AS service,
       ROUND(SUM(cost), 2) AS cost_usd
FROM `myco-fraud-prod.billing.gcp_billing_export_v1_XXXXXX`
WHERE DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY service ORDER BY cost_usd DESC;

-- Net cost after credits (CUD/committed-use and promotional credits net out here)
SELECT sku.description AS sku,
       ROUND(SUM(cost), 2) AS gross,
       ROUND(SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)), 2) AS credits,
       ROUND(SUM(cost) + SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)), 2) AS net
FROM `myco-fraud-prod.billing.gcp_billing_export_v1_XXXXXX`
WHERE DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY sku ORDER BY net DESC;

-- Attribute last month's cost by component label (the original view)
SELECT
  (SELECT value FROM UNNEST(labels) WHERE key = 'component') AS component,
  ROUND(SUM(cost), 2) AS cost_usd
FROM `myco-fraud-prod.billing.gcp_billing_export_v1_XXXXXX`
WHERE DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY component ORDER BY cost_usd DESC;

-- Naive 30-day forecast from a 7-day daily run-rate
SELECT ROUND(AVG(daily) * 30, 2) AS projected_month_usd FROM (
  SELECT DATE(usage_start_time) AS d, SUM(cost) AS daily
  FROM `myco-fraud-prod.billing.gcp_billing_export_v1_XXXXXX`
  WHERE DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
  GROUP BY d);
```

### Committed-use discounts and the recommenders

For a steady baseline — an always-on serving endpoint, a persistent training rig — a **committed-use discount (CUD)** trades a 1- or 3-year commitment for a large discount (up to ~40% resource-based, more for some GPUs). Resource-based CUDs are purchased per region against vCPU/memory (and GPU/Local SSD) quantities:

```bash
gcloud compute commitments create fraud-serving-cud \
  --region=us-central1 --plan=twelve-month \
  --resources=vcpu=16,memory=64GB
```

You do not have to guess where the waste is. The **Recommender** service continuously analyzes usage and surfaces cost recommendations you can list and act on — idle VMs, over-provisioned machine types, idle Cloud SQL instances, unattached disks, and CUD purchase opportunities:

```bash
# Idle VMs (delete/stop candidates)
gcloud recommender recommendations list \
  --project=myco-fraud-prod --location=us-central1-a \
  --recommender=google.compute.instance.IdleResourceRecommender

# Machine-type rightsizing (over-provisioned VMs)
gcloud recommender recommendations list \
  --project=myco-fraud-prod --location=us-central1-a \
  --recommender=google.compute.instance.MachineTypeRecommender

# CUD purchase recommendations
gcloud recommender recommendations list \
  --project=myco-fraud-prod --location=us-central1 \
  --recommender=google.compute.commitment.UsageCommitmentRecommender
```

### Quotas, pricing estimates, and per-second billing

Two more levers round out cost hygiene. **Quotas** are a proactive cap, not just a safety limit: capping the GPU or vCPU quota on a dev project makes it *impossible* to accidentally spin up a fleet of A4 (B200) instances, which is a stronger guarantee than a budget alert that fires after the spend. List and audit account/quota state with `gcloud billing accounts list` and the project's quota page. Before committing to an architecture, price it with the **Google Cloud Pricing Calculator** (`cloud.google.com/products/calculator`) — an hour there is cheaper than a surprise invoice. And remember that Compute Engine, Cloud Run, GPUs, and notebooks bill by the **second** (with a small minimum), so an idle GPU VM left running overnight is pure, continuous waste — the single most common avoidable cost. Storage and BigQuery levers are covered in the storage and BigQuery modules; the same partitioning, lifecycle, and `--maximum_bytes_billed` disciplines apply here as first-class cost controls.

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

## CLI cheat-sheet

```bash
# --- Budgets (on the BILLING ACCOUNT; notify, then automate to enforce) ---
gcloud billing accounts list
gcloud billing budgets create --billing-account=0X0X0X-0X0X0X-0X0X0X \
  --display-name="fraud-prod monthly" --budget-amount=5000USD \
  --filter-projects=projects/myco-fraud-prod --filter-labels=env=prod \
  --threshold-rule=percent=0.5,basis=current-spend \
  --threshold-rule=percent=0.9,basis=current-spend \
  --threshold-rule=percent=1.0,basis=forecasted-spend \
  --notifications-rule-pubsub-topic=projects/myco-fraud-prod/topics/budget-alerts
gcloud billing budgets list --billing-account=0X0X0X-0X0X0X-0X0X0X

# --- Cost recommenders (find idle/oversized resources and CUD opportunities) ---
gcloud recommender recommendations list --project=myco-fraud-prod \
  --location=us-central1-a --recommender=google.compute.instance.IdleResourceRecommender
gcloud recommender recommendations list --project=myco-fraud-prod \
  --location=us-central1-a --recommender=google.compute.instance.MachineTypeRecommender
gcloud recommender recommendations list --project=myco-fraud-prod \
  --location=us-central1 --recommender=google.compute.commitment.UsageCommitmentRecommender

# --- Committed-use discount for steady baseline capacity ---
gcloud compute commitments create fraud-serving-cud --region=us-central1 \
  --plan=twelve-month --resources=vcpu=16,memory=64GB

# --- Turn idle spend off (bills by the second) ---
gcloud compute instances stop dev-notebook --zone=us-central1-a

# --- Monitoring (summary; see module 15 for depth) ---
gcloud monitoring policies create --policy-from-file=high-latency-policy.json
gcloud monitoring uptime create serve-health --resource-type=uptime-url \
  --hostname=fraud-serve-xyz.a.run.app --path=/healthz
```

```sql
-- Billing export: cost by service, and net-of-credits, last 30 days
SELECT service.description AS service, ROUND(SUM(cost),2) AS cost_usd
FROM `myco-fraud-prod.billing.gcp_billing_export_v1_XXXXXX`
WHERE DATE(usage_start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY service ORDER BY cost_usd DESC;
```

## Try it

Close the loop on a system you have deployed:

1. Enable **billing export to BigQuery**, then run a query that breaks down the last 30 days of cost by your `component` label. Identify the single most expensive component.
2. Create a **budget** with alerts at 50/90/100% and, as a stretch, wire its Pub/Sub notification to a function that logs a warning.
3. Build a Cloud Monitoring **dashboard** for a serving endpoint showing request count, error rate, p95 latency, and GPU duty cycle, and add an **alerting policy** on p95 latency.
4. Enable **Vertex AI Model Monitoring** on a deployed model with your training data as the baseline, set a drift threshold on one feature, then send skewed inputs and confirm you get a drift alert. Reflect on what retraining trigger you would wire to that alert.
