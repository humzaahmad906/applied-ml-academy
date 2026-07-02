# 15 — Observability: Cloud Monitoring, Logging, and Error Reporting

Module 13 looked at monitoring from a cost-and-drift altitude; this module treats the observability suite — Cloud Logging, Cloud Monitoring, Error Reporting, Cloud Trace, and Cloud Profiler — as a first-class topic in its own right. The reason it deserves its own module is that a deployed fraud-scoring model is a *production service*, and production services fail in production-service ways: latency creeps up, a serving container OOMs, GPU utilization silently drops to zero, error rates spike after a deploy. None of that is model drift, and none of it shows up in an offline eval. It shows up in logs and metrics. This module is about the infrastructure and application observability that tells you the system is *healthy*; for the *cost* angle and for Vertex Model Monitoring (training-serving skew and prediction drift) refer to module 13 — we do not re-teach drift here.

## Cloud Logging: the query language

Every log entry Google Cloud ingests is a structured record with a `resource.type` (what produced it), a `severity`, a timestamp, and a payload that is either `textPayload` (a plain string) or `jsonPayload` (structured fields). The **Logging query language** filters on exactly those fields, and fluency with it is the difference between finding the one failing request and scrolling forever. You compare fields with `=`, `!=`, `>=`, `:` (has/substring), and combine them with `AND`/`OR`.

```bash
# All ERROR-and-above logs from a Cloud Run service in the last hour
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="fraud-scorer"
   AND severity>=ERROR' \
  --freshness=1h --format="table(timestamp, severity, textPayload)"

# Filter on a structured field your serving code emitted
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND jsonPayload.latency_ms>500' \
  --freshness=30m --limit=50 --format=json
```

`--freshness` bounds how far back to look, `--format` reshapes the output (`json`, `table(...)`, `value(...)`), and `--limit` caps the count.

## Log buckets, retention, and reading training logs

Ingested logs land in **log buckets**. Every project has two by default: `_Required` (holds admin/audit logs, a fixed 400-day retention, and cannot be modified or deleted) and `_Default` (everything else, 30-day retention, which you *can* extend). You can create additional buckets with custom retention for logs you must keep longer. Vertex AI training and prediction write to Cloud Logging automatically, so you read a training run's logs with the same tool by filtering on the job resource:

```bash
# Read logs from a specific Vertex custom training job
gcloud logging read \
  'resource.type="ml_job"
   AND resource.labels.job_id="7788990011"' \
  --freshness=2h --format="value(timestamp, textPayload)"

# Extend _Default retention to 90 days
gcloud logging buckets update _Default --location=global --retention-days=90
```

## The Log Router and sinks

Ingested logs are matched against the **Log Router**, which sends copies to destinations via **sinks**. This is how you get logs *out* of Cloud Logging for long-term retention or analysis: a sink routes matching entries to BigQuery (to join prediction logs against labels for analysis), Cloud Storage (cheap long-term archival), or Pub/Sub (to stream into an external SIEM or pipeline). A sink is a filter plus a destination, and after you create it you must grant its auto-generated writer identity access to write to the destination.

```bash
# Route all fraud-scorer prediction logs into BigQuery for offline analysis
gcloud logging sinks create fraud-preds-to-bq \
  bigquery.googleapis.com/projects/myco-fraud-prod/datasets/serving_logs \
  --log-filter='resource.type="cloud_run_revision"
    AND resource.labels.service_name="fraud-scorer"
    AND jsonPayload.event="prediction"'

# The command prints a writerIdentity; grant it on the destination, e.g.
gcloud projects add-iam-policy-binding myco-fraud-prod \
  --member="serviceAccount:<writerIdentity>" --role="roles/bigquery.dataEditor"
```

## Log-based metrics and exclusion filters

Two operations turn raw logs into signal and savings. A **log-based metric** converts matching log entries into a time series you can chart and alert on — a **counter** metric counts matches (how many `severity>=ERROR` entries per minute), while a **distribution** metric buckets a numeric value extracted from the logs (a histogram of `jsonPayload.latency_ms`). An **exclusion filter** does the opposite: it drops matching entries *before* ingestion so you never pay to store them — the primary lever for controlling logging cost, since ingestion is billed.

```bash
# Counter metric: prediction errors per unit time
gcloud logging metrics create fraud_pred_errors \
  --description="Fraud scorer prediction errors" \
  --log-filter='resource.type="cloud_run_revision"
    AND resource.labels.service_name="fraud-scorer" AND severity>=ERROR'

# Distribution metric (with value extractor + buckets) via a config file
gcloud logging metrics create fraud_latency_dist --config-from-file=latency-metric.yaml

# Stop paying to store chatty health-check logs
gcloud logging sinks update _Default \
  --add-exclusion=name=drop-healthchecks,filter='httpRequest.requestUrl:"/healthz"'
```

## Structured logging from ML code

Emit **structured** logs, not `print` statements. When your training or serving code logs JSON, every field becomes queryable in the query language and usable as a log-based metric — `jsonPayload.latency_ms`, `jsonPayload.model_version`, `jsonPayload.prediction`. The `google-cloud-logging` library installs a handler into standard Python `logging` so `logger.info(...)` with a dict `extra` becomes a `jsonPayload`:

```python
import logging
import google.cloud.logging

client = google.cloud.logging.Client()
client.setup_logging()  # routes stdlib logging -> Cloud Logging as structured entries
logger = logging.getLogger(__name__)

logger.info(
    "prediction",
    extra={"json_fields": {
        "event": "prediction",
        "model_version": "v7",
        "latency_ms": 42,
        "score": 0.91,
    }},
)
```

## Cloud Monitoring: metric types

Cloud Monitoring collects time-series **metrics**. Google Cloud emits built-in ones automatically — for a Vertex endpoint you get `aiplatform.googleapis.com/prediction/online/prediction_count`, `.../prediction_latencies`, `.../error_count`, and the GPU signal `.../accelerator/duty_cycle`; Compute VMs emit CPU and disk; Cloud Run emits request count and container utilization. You query metrics with **MQL** (Monitoring Query Language) or **PromQL** (for Prometheus-style metrics, including the GPU DCGM metrics under `prometheus.googleapis.com/`). For anything the platform does not measure — say a business KPI like fraudulent-transactions-caught — you write a **custom metric** from code with `google-cloud-monitoring`, or emit via OpenTelemetry:

```python
from google.cloud import monitoring_v3
import time

client = monitoring_v3.MetricServiceClient()
project = "projects/myco-fraud-prod"

series = monitoring_v3.TimeSeries()
series.metric.type = "custom.googleapis.com/fraud/flagged_count"
series.resource.type = "global"
now = time.time()
point = monitoring_v3.Point({
    "interval": {"end_time": {"seconds": int(now)}},
    "value": {"int64_value": 3},
})
series.points = [point]
client.create_time_series(name=project, time_series=[series])
```

## Dashboards, alerting policies, and notification channels

You compose metrics into **dashboards** and wrap them in **alerting policies**. The ordering matters and trips people up: an alerting policy references a **notification channel** (email, Slack, PagerDuty), and *the channel must already exist* — create the channel first, then reference its id in the policy. All three are declarative JSON/YAML you check into source control.

```bash
# 1. Create the notification channel FIRST, capture its id
gcloud monitoring channels create --channel-content-from-file=email-channel.yaml

# 2. Create the dashboard and the alerting policy from config files
gcloud monitoring dashboards create --config-from-file=fraud-serving-dashboard.json
gcloud monitoring policies create --policy-from-file=high-error-rate-policy.yaml
```

A representative policy fires when the Vertex endpoint's error rate stays high for five minutes and pings the channel above. You also get **uptime checks** (synthetic probes against your serving URL from multiple locations) and, at a higher level, **SLOs/SLIs** — you define a service-level indicator (e.g. p99 latency < 300 ms) and an objective (99.9% of the time), and Monitoring tracks your error budget against it. For multi-project setups, a **metrics scope** lets one monitoring project observe metrics across several projects (e.g. `myco-fraud-dev` and `myco-fraud-prod` on one dashboard).

## Error Reporting, Cloud Trace, and Cloud Profiler

Three companion tools close the loop. **Error Reporting** automatically parses stack traces out of your logs and *aggregates* them — a hundred instances of the same exception become one counted, deduplicated issue you can alert on, rather than a hundred log lines. **Cloud Trace** captures distributed latency: when a request flows Cloud Run → Vertex endpoint, Trace shows a waterfall of where the milliseconds went across services, which is how you find the slow hop. **Cloud Profiler** continuously samples CPU and heap of a running serving container at negligible overhead, so you can see which function is burning cycles in production. The through-line for all three is **OpenTelemetry**: instrument your code once with OTel and export traces and metrics to these backends, rather than wiring each tool by hand.

## What to observe for an ML serving endpoint

Concretely, a fraud-scoring endpoint should surface:

- **Latency** at p50/p95/p99 — tail latency is what your callers actually feel; the average hides it.
- **Error rate** — 5xx and prediction failures, alerted per the policy above.
- **Prediction volume** — request/prediction count, to catch a caller that has silently stopped or a flood.
- **GPU duty cycle** (`accelerator/duty_cycle`) — a GPU pinned at 0% is money burning; one pinned at 100% needs more replicas.
- **Token counts** — for a Gemini-backed path (module 11), log input/output token counts, since they drive both latency and cost.

Beyond health, log the **prediction inputs and outputs as structured entries** and route them to BigQuery with a sink. That table is the raw material for the training-serving skew and drift analysis that **Vertex Model Monitoring** (module 13) performs — observability here feeds the drift detection there.

## How this fits the whole solution

Observability is the feedback layer over everything the course has built. The ingestion pipeline, the BigQuery jobs of module 08, the Vertex training runs of module 09, and the endpoints of module 10 all write logs and metrics you read here; the serving service of modules 06 and 10 is where latency, error-rate, and GPU-duty-cycle alerts actually protect your SLO; the Gemini path of module 11 needs its token counts logged for both latency and cost. Sinks route prediction logs to BigQuery, where module 13's Vertex Model Monitoring turns them into drift and skew signals, and the alerting-and-cost story of module 13 rides on the same metrics and exclusion filters introduced here. Without this layer you are flying a production ML system blind.

## Key takeaways

- **Cloud Logging is queryable structure**: filter on `resource.type`, `severity`, and `jsonPayload.*`; read with `gcloud logging read --freshness/--format`; emit structured JSON from code via `google-cloud-logging`, never `print`.
- **Route and shape logs**: **sinks** export to BigQuery/Storage/Pub/Sub for analysis and archival, **log-based metrics** (counter/distribution) turn logs into alertable series, and **exclusion filters** cut ingestion cost.
- **Cloud Monitoring** exposes built-in metrics (`prediction/online/prediction_latencies`, `accelerator/duty_cycle`) plus custom metrics; compose **dashboards** and **alerting policies**, but create the **notification channel first** — it must exist before the policy references it.
- **Error Reporting, Trace, and Profiler** add exception aggregation, cross-service latency waterfalls, and production CPU/heap profiling — unify them with **OpenTelemetry**, and watch for billable ingestion and custom-metric **cardinality explosions**.

## CLI cheat-sheet

```bash
# --- Cloud Logging: read & filter ---
gcloud logging read 'resource.type="cloud_run_revision"
  AND resource.labels.service_name="fraud-scorer" AND severity>=ERROR' \
  --freshness=1h --limit=50 --format="table(timestamp, severity, textPayload)"
gcloud logging read 'resource.type="ml_job" AND resource.labels.job_id="ID"' --freshness=2h

# Buckets & retention (_Required and _Default are the built-ins)
gcloud logging buckets list --location=global
gcloud logging buckets update _Default --location=global --retention-days=90

# Sinks (export) + log-based metrics + exclusions
gcloud logging sinks create NAME bigquery.googleapis.com/projects/P/datasets/D --log-filter='...'
gcloud logging sinks create NAME storage.googleapis.com/BUCKET --log-filter='...'
gcloud logging metrics create NAME --description="..." --log-filter='...'   # counter
gcloud logging metrics create NAME --config-from-file=metric.yaml           # distribution
gcloud logging sinks update _Default --add-exclusion=name=X,filter='...'

# --- Cloud Monitoring ---
gcloud monitoring channels create --channel-content-from-file=channel.yaml   # FIRST
gcloud monitoring dashboards create --config-from-file=dashboard.json
gcloud monitoring policies create --policy-from-file=policy.yaml
gcloud monitoring channels list
gcloud monitoring dashboards list

# Useful built-in metric types
#   aiplatform.googleapis.com/prediction/online/prediction_count
#   aiplatform.googleapis.com/prediction/online/prediction_latencies
#   aiplatform.googleapis.com/prediction/online/error_count
#   aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle
```

## Try it

Instrument the fraud-scoring endpoint end to end:

1. Add `google-cloud-logging` to the serving container and emit a structured `prediction` log per request with `latency_ms`, `model_version`, and `score` fields; deploy and confirm the fields appear as `jsonPayload` via `gcloud logging read`.
2. Create a distribution log-based metric over `jsonPayload.latency_ms` and a counter metric over `severity>=ERROR`, then build a dashboard from a config file charting p95 latency, error count, and the endpoint's `accelerator/duty_cycle`.
3. Create an email notification channel, then an alerting policy that fires when the error-count metric exceeds a threshold for five minutes — verify the policy references the channel id you just created.
4. Create a logging sink routing your `event="prediction"` logs to a BigQuery dataset, grant the sink's writer identity `bigquery.dataEditor`, and confirm rows land — this is the table module 13's Model Monitoring will read for drift.
5. Add an exclusion filter dropping `/healthz` logs, then open Error Reporting and confirm a deliberately-thrown exception in the container shows up as a single aggregated issue.
