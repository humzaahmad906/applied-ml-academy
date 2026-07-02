# 16 — Observability for ML: CloudWatch, Logs, X-Ray

An ML system in production fails in ways a demo never shows: a GPU endpoint that quietly drops to 5% utilization while you pay for the whole instance, a Bedrock feature whose token bill triples after a prompt change, an inference request that mysteriously takes two seconds because one hop in the chain stalled. Observability is how you see these before your users — or your finance team — do. Module 13 gave a tour of CloudWatch as one piece of operational maturity; this module is the deep dive: how metrics, logs, alarms, dashboards, and distributed tracing actually work, and how to wire them around a real multi-hop inference stack.

## CloudWatch Metrics: the vocabulary

A CloudWatch **metric** is a time-ordered set of data points, identified by a **namespace** (a container like `AWS/SageMaker` or `AWS/Lambda`, or your own `MyCompany/Inference`) plus a set of **dimensions** (name/value pairs like `EndpointName=fraud-scorer`, `VariantName=AllTraffic`). The combination of namespace, metric name, and a specific set of dimensions defines a distinct metric — which is why emitting the same metric with a new dimension value silently creates a new, separately-billed time series. This is the single most important thing to understand about metric cost: cardinality is per-dimension-combination.

When you retrieve a metric you choose a **statistic** — `Average`, `Sum`, `Minimum`, `Maximum`, `SampleCount`, or a **percentile** like `p90`, `p99` — computed over a **period** (the aggregation window). For latency, percentiles are the honest measure: an average `ModelLatency` of 80 ms can hide a `p99` of 900 ms that is timing out real users. **Resolution** matters too: standard metrics store at 1-minute granularity, while **high-resolution** metrics store down to 1 second. High resolution is worth the extra cost for autoscaling triggers and latency SLOs where a one-minute average smooths over the spike you needed to catch.

```bash
# List what an endpoint is emitting, then pull p99 latency over a window
aws cloudwatch list-metrics --namespace AWS/SageMaker \
  --dimensions Name=EndpointName,Value=fraud-scorer

aws cloudwatch get-metric-data \
  --start-time 2026-07-02T00:00:00Z --end-time 2026-07-02T01:00:00Z \
  --metric-data-queries '[{
    "Id": "p99lat",
    "MetricStat": {
      "Metric": {
        "Namespace": "AWS/SageMaker",
        "MetricName": "ModelLatency",
        "Dimensions": [
          {"Name": "EndpointName", "Value": "fraud-scorer"},
          {"Name": "VariantName", "Value": "AllTraffic"}
        ]
      },
      "Period": 60, "Stat": "p99"
    }
  }]'
```

## The metrics ML services emit for free

Before you write a single custom metric, know what AWS already publishes. The managed services instrument themselves, and most ML monitoring starts here.

**SageMaker real-time endpoints** publish to the `AWS/SageMaker` namespace, dimensioned by `EndpointName` and `VariantName`: `Invocations` (request count), `ModelLatency` (time the model container spent), `OverheadLatency` (time SageMaker itself added — the gap between these two is where you diagnose "is it my model or the platform?"), and `Invocation4XXErrors` / `Invocation5XXErrors`. Crucially, the instance-level metrics — `CPUUtilization`, `MemoryUtilization`, `GPUUtilization`, `GPUMemoryUtilization`, `DiskUtilization` — publish to a *separate* namespace, `/aws/sagemaker/Endpoints`, dimensioned the same way. Watching `GPUUtilization` here is how you catch the classic ML cost leak: an endpoint on an `ml.g5.xlarge` running at 8% GPU is telling you to move to inference components or a smaller instance.

**Lambda** (the usual front for a lightweight inference or pre/post-processing step) publishes to `AWS/Lambda`, dimensioned by `FunctionName`: `Invocations`, `Duration`, `Errors`, `Throttles`, and `ConcurrentExecutions`. `Throttles` climbing means you hit a concurrency limit — a real failure mode when a burst of inference requests arrives faster than reserved concurrency allows.

**Bedrock** publishes to `AWS/Bedrock`, dimensioned by `ModelId`: `Invocations`, `InvocationLatency`, `InputTokenCount`, and `OutputTokenCount`. Those token metrics are your cost early-warning system — Bedrock bills per token, so a rising `OutputTokenCount` average after a prompt change is a bill increase you can alarm on before the invoice arrives.

```bash
# Endpoint GPU utilization lives in a DIFFERENT namespace than invocations
aws cloudwatch get-metric-statistics \
  --namespace /aws/sagemaker/Endpoints --metric-name GPUUtilization \
  --dimensions Name=EndpointName,Value=fraud-scorer Name=VariantName,Value=AllTraffic \
  --start-time 2026-07-02T00:00:00Z --end-time 2026-07-02T01:00:00Z \
  --period 300 --statistics Average Maximum
```

## Custom metrics and the Embedded Metric Format

The metrics AWS emits describe infrastructure; the metrics that describe your *model* — prediction confidence, class distribution, feature staleness, tokens consumed per request — you emit yourself. There are two ways, and the choice matters at scale.

The obvious way is `put-metric-data`, a synchronous API call per metric. It is fine for low-volume, out-of-band publishing (a nightly job reporting yesterday's mean confidence), but on a hot inference path it adds latency and API cost to every request, and `PutMetricData` is throttled.

```bash
# Simple, out-of-band custom metric
aws cloudwatch put-metric-data \
  --namespace MyCompany/Inference --metric-name PredictionConfidence \
  --value 0.87 --unit None \
  --dimensions ModelName=fraud-scorer,Stage=prod
```

The efficient way — and the right default for per-request ML metrics — is the **Embedded Metric Format (EMF)**. Instead of calling an API, you write a specially-structured JSON *log line*; CloudWatch Logs recognizes the `_aws` block and automatically extracts the named fields into metrics, at no extra API call and no request-path latency. You already write logs, so EMF makes metrics a free side effect of logging, while keeping the full high-cardinality context (request ID, model version) queryable in the log itself. From a Lambda or endpoint container, print this to stdout:

```json
{
  "_aws": {
    "Timestamp": 1751414400000,
    "CloudWatchMetrics": [
      {
        "Namespace": "MyCompany/Inference",
        "Dimensions": [["ModelName", "Stage"]],
        "Metrics": [
          {"Name": "PredictionConfidence", "Unit": "None"},
          {"Name": "InputTokens", "Unit": "Count"},
          {"Name": "FeatureStalenessSeconds", "Unit": "Seconds", "StorageResolution": 60}
        ]
      }
    ]
  },
  "ModelName": "fraud-scorer",
  "Stage": "prod",
  "PredictionConfidence": 0.87,
  "InputTokens": 512,
  "FeatureStalenessSeconds": 42,
  "requestId": "989ffbf8-9ace-4817-a57c-e4dd734019ee"
}
```

CloudWatch reads the `Dimensions` array to know which root-level fields become dimensions, and each entry in `Metrics` names a root-level field to extract as a metric. Note the cardinality warning from the spec: only put low-cardinality fields (`ModelName`, `Stage`) in `Dimensions`. `requestId` stays a plain log field — never a dimension — or you create one metric per request and the bill explodes.

## CloudWatch Logs: retention, Insights, and turning logs into signal

Every training job, Lambda, and endpoint container writes to CloudWatch Logs, organized into **log groups** (one per resource, e.g. `/aws/lambda/inference-fn`) containing **log streams** (one per instance or execution). The first discipline is **retention**: log groups default to *never expire*, which is a slow-growing bill. Set a retention policy on every group.

```bash
aws logs create-log-group --log-group-name /ml/fraud-scorer
aws logs put-retention-policy --log-group-name /ml/fraud-scorer --retention-in-days 90
aws logs tail /ml/fraud-scorer --follow --since 10m   # live debugging
```

**Logs Insights** is a purpose-built query language over your logs — the tool you reach for when an endpoint starts erroring at 3 a.m. Because EMF metrics are still just log lines, you can query the raw fields the metrics were extracted from, which is exactly the drill-down a dashboard alarm can't give you. Here is a real ML query: find the low-confidence predictions in the last window and see how latency correlates.

```bash
QID=$(aws logs start-query \
  --log-group-name /ml/fraud-scorer \
  --start-time $(($(date +%s) - 3600)) --end-time $(date +%s) \
  --query-string 'fields @timestamp, ModelName, PredictionConfidence, @duration
    | filter PredictionConfidence < 0.6
    | stats count(*) as low_conf_calls, avg(@duration) as avg_ms by bin(5m)
    | sort @timestamp desc' \
  --query queryId --output text)

aws logs get-query-results --query-id "$QID"
```

Two ways to turn logs into ongoing signal. A **metric filter** watches a log group for a pattern and increments a metric each time it matches — the way to alarm on a log message that isn't already a metric, like a specific stack trace or an OOM kill in a training container. A **subscription filter** streams matching log events in near-real-time to a destination (Kinesis, Firehose, or a Lambda) — the way to fan logs out to a data lake, a SIEM, or a custom drift detector.

```bash
# Count OOM kills in the training log and expose them as a metric to alarm on
aws logs put-metric-filter \
  --log-group-name /ml/training \
  --filter-name oom-kills \
  --filter-pattern '"CUDA out of memory"' \
  --metric-transformations \
      metricName=OOMKills,metricNamespace=MyCompany/Training,metricValue=1,defaultValue=0

# Stream error-level lines to a Lambda for custom handling
aws logs put-subscription-filter \
  --log-group-name /ml/fraud-scorer \
  --filter-name errors-to-handler \
  --filter-pattern '{ $.level = "ERROR" }' \
  --destination-arn arn:aws:lambda:us-east-1:123456789012:function:log-handler
```

## Alarms: static, anomaly-detection, and composite

An alarm evaluates a metric against a rule over time and transitions between `OK`, `ALARM`, and `INSUFFICIENT_DATA`, firing **actions** on transition. A **static** alarm compares to a fixed threshold — right when you have a hard SLO like "p99 latency under 500 ms."

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name fraud-scorer-p99-latency \
  --namespace AWS/SageMaker --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=fraud-scorer Name=VariantName,Value=AllTraffic \
  --extended-statistic p99 --period 60 \
  --evaluation-periods 5 --datapoints-to-alarm 3 \
  --threshold 500000 --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:ml-oncall
```

Two knobs above earn their keep. `--evaluation-periods 5 --datapoints-to-alarm 3` is **M-of-N** evaluation: alarm only if 3 of the last 5 periods breach, which suppresses the single-spike false alarm that pages on-call for nothing. `--treat-missing-data` decides what a gap means — `notBreaching` is right for a bursty endpoint that legitimately goes idle, whereas `breaching` is right for a heartbeat metric whose absence *is* the failure. (Note `ModelLatency` is reported in microseconds, hence `500000` for 500 ms.)

When there is no fixed "good" number — token counts, request volume, and confidence all have daily and weekly rhythms — use an **anomaly-detection** alarm. You train a band of expected values with `put-anomaly-detector`, then alarm when the metric leaves the band. This is the natural fit for catching Bedrock token-cost drift: you don't know the "right" `OutputTokenCount`, only that a sudden departure from the learned pattern is worth a look.

```bash
aws cloudwatch put-anomaly-detector \
  --namespace AWS/Bedrock --metric-name OutputTokenCount \
  --dimensions Name=ModelId,Value=anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --stat Average

aws cloudwatch put-metric-alarm \
  --alarm-name bedrock-token-drift \
  --comparison-operator LessThanLowerOrGreaterThanUpperThreshold \
  --evaluation-periods 3 --threshold-metric-id ad1 \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:ml-oncall \
  --metrics '[
    {"Id": "m1", "MetricStat": {
        "Metric": {"Namespace": "AWS/Bedrock", "MetricName": "OutputTokenCount",
          "Dimensions": [{"Name": "ModelId", "Value": "anthropic.claude-3-5-sonnet-20241022-v2:0"}]},
        "Period": 300, "Stat": "Average"}, "ReturnData": true},
    {"Id": "ad1", "Expression": "ANOMALY_DETECTION_BAND(m1, 2)", "ReturnData": true}
  ]'
```

A **composite alarm** combines other alarms with boolean logic (`ALARM("latency") AND ALARM("errors")`), so you can page only when *several* signals agree — the cure for alert fatigue when a single upstream blip trips every child alarm at once. Alarm actions aren't limited to SNS: they can trigger **application autoscaling** on an endpoint, or fire an **EventBridge** rule that kicks off a retraining pipeline when drift is detected.

## Dashboards

A dashboard is the single pane that answers "is the whole pipeline healthy?" — invocations, p99 latency, error rate, GPU utilization, and token counts side by side. It is defined as JSON and created with `put-dashboard`, which makes it version-controllable alongside the rest of your infrastructure.

```bash
aws cloudwatch put-dashboard --dashboard-name ml-inference --dashboard-body '{
  "widgets": [
    {"type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
     "properties": {
       "title": "Endpoint p99 latency (ms)", "region": "us-east-1",
       "metrics": [["AWS/SageMaker", "ModelLatency",
         "EndpointName", "fraud-scorer", "VariantName", "AllTraffic",
         {"stat": "p99", "period": 60}]]
     }},
    {"type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
     "properties": {
       "title": "GPU utilization (%)", "region": "us-east-1",
       "metrics": [["/aws/sagemaker/Endpoints", "GPUUtilization",
         "EndpointName", "fraud-scorer", "VariantName", "AllTraffic",
         {"stat": "Average", "period": 300}]]
     }}
  ]
}'
```

## AWS X-Ray: tracing the multi-hop request

Metrics and logs tell you a component is slow; they don't tell you *which hop* in a chained request ate the time. A real inference call is rarely one service — it is API Gateway → Lambda → a SageMaker endpoint or Bedrock, maybe with a feature-store read and a vector search in between. When p99 latency creeps up, **X-Ray** is how you find the culprit without guessing.

X-Ray works by propagating a trace ID through the request. Each service records a **segment** (its slice of the total time), and within a segment your code can record **subsegments** for individual operations — the DynamoDB feature read, the model invocation, the post-processing. X-Ray stitches these into a **service map** (a visual graph of every hop with latency and error rates on each edge) and per-request **traces** (the waterfall showing exactly where the 1.8 seconds went). Because tracing every request at high volume is expensive, X-Ray uses **sampling** — a configurable rule captures a small representative fraction (e.g. 1 request/second plus 5% of the rest), enough to see the distribution without instrumenting everything. You retrieve traces from the CLI:

```bash
# Service map for the last 10 minutes: every hop, its latency and error rate
EPOCH=$(date +%s)
aws xray get-service-graph --start-time $((EPOCH-600)) --end-time $EPOCH

# Find recent slow/errored traces, then pull the full waterfall for them
TRACEIDS=$(aws xray get-trace-summaries \
  --start-time $((EPOCH-600)) --end-time $EPOCH \
  --query 'TraceSummaries[?ResponseTime > `1.5`].Id' --output text)
aws xray batch-get-traces --trace-ids $TRACEIDS
```

Instrumented services call `put-trace-segments` under the hood (via the SDK or the X-Ray daemon); you rarely call it by hand. For a modern ML stack the payoff is direct: when a chained RAG request is slow, the service map instantly shows whether the time is in the vector search, the Bedrock call, or your own glue code. CloudWatch has been folding X-Ray into a unified **Application Signals** experience that correlates traces, metrics, and logs against service-level objectives on one screen — worth enabling so a latency SLO breach links straight to the offending trace.

## Tying it back to ML

The through-line is that observability for ML watches two layers at once. The *infrastructure* layer — GPU utilization, latency, throttles — is what CloudWatch metrics and X-Ray cover, and it catches the idle-endpoint cost leak and the multi-hop latency stall. The *cost* layer for generative workloads shows up as token metrics, where an anomaly alarm on `OutputTokenCount` catches a prompt regression before the bill does. But neither layer sees the thing that most often breaks an ML system silently: the model still runs fast and cheap while its predictions drift away from reality. That is **data and prediction drift**, and it needs **SageMaker Model Monitor**, which emits its own CloudWatch metrics so drift shows up right next to latency on the same dashboard and can fire the same EventBridge-triggered retraining. Model Monitor is deep enough to get its own treatment in module 18; here, just know it is the third leg — infrastructure, cost, and model quality — and that CloudWatch is the common surface all three report to.

## How this fits the whole ML solution

Observability is the nervous system of everything the rest of the course built: it is how the reference architecture reports its own health back to you. The endpoint from the inference module emits latency and GPU metrics; the Lambda front door emits throttles; the Bedrock calls emit token counts; the pipeline emits job successes; Model Monitor emits drift — and CloudWatch is the single place they converge into dashboards, alarms, and traces. Without it you are flying an expensive, always-on system blind; with it, a latency SLO breach links to the exact slow hop, a cost spike alarms before the invoice, and a drift signal auto-triggers retraining. This is what turns a deployed model into an operated one.

## Key takeaways

- A metric is namespace + name + dimensions; each unique dimension combination is a separately-billed time series, so keep dimension cardinality low. Use percentiles (p99) for latency and high-resolution (1s) metrics for autoscaling and tight SLOs.
- ML services self-instrument: SageMaker emits `Invocations`/`ModelLatency`/`OverheadLatency`/`Invocation4XX-5XXErrors` in `AWS/SageMaker` and GPU/CPU/memory utilization in `/aws/sagemaker/Endpoints`; Lambda emits `Duration`/`Errors`/`Throttles`/`ConcurrentExecutions`; Bedrock emits `InvocationLatency`/`InputTokenCount`/`OutputTokenCount`.
- Emit custom ML metrics (confidence, feature staleness, tokens) with the Embedded Metric Format — a JSON log line with an `_aws` block — to avoid per-request API calls and latency; use `put-metric-data` only for low-volume out-of-band cases.
- Always set log retention; use Logs Insights to drill into raw fields, metric filters to alarm on log patterns, and subscription filters to stream logs to a drift detector or data lake.
- Use static alarms for hard SLOs and anomaly-detection alarms for rhythmic metrics like token counts; tune M-of-N (`datapoints-to-alarm`) and missing-data treatment to kill false pages, and use composite alarms to page only when signals agree.
- X-Ray traces a request across API Gateway → Lambda → endpoint/Bedrock, and its service map pinpoints which hop is slow; pair infrastructure and cost observability with SageMaker Model Monitor (module 18) for model-quality drift, all surfacing in CloudWatch.

## CLI cheat-sheet

```bash
# Metrics
aws cloudwatch list-metrics --namespace AWS/SageMaker
aws cloudwatch get-metric-data --start-time ... --end-time ... --metric-data-queries '[...]'
aws cloudwatch get-metric-statistics --namespace /aws/sagemaker/Endpoints \
  --metric-name GPUUtilization --dimensions ... --period 300 --statistics Average
aws cloudwatch put-metric-data --namespace MyCompany/Inference \
  --metric-name PredictionConfidence --value 0.87 --dimensions ModelName=fraud-scorer

# Alarms
aws cloudwatch put-metric-alarm --alarm-name ... --namespace ... --metric-name ... \
  --extended-statistic p99 --period 60 --evaluation-periods 5 --datapoints-to-alarm 3 \
  --threshold 500000 --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching --alarm-actions <sns-arn>
aws cloudwatch put-anomaly-detector --namespace AWS/Bedrock \
  --metric-name OutputTokenCount --dimensions Name=ModelId,Value=... --stat Average

# Dashboards
aws cloudwatch put-dashboard --dashboard-name ml-inference --dashboard-body '{...}'

# Logs
aws logs create-log-group --log-group-name /ml/fraud-scorer
aws logs put-retention-policy --log-group-name /ml/fraud-scorer --retention-in-days 90
aws logs tail /ml/fraud-scorer --follow --since 10m
aws logs start-query --log-group-name /ml/fraud-scorer \
  --start-time <epoch> --end-time <epoch> --query-string 'fields ... | filter ... | stats ...'
aws logs get-query-results --query-id <id>
aws logs put-metric-filter --log-group-name ... --filter-name ... \
  --filter-pattern '...' --metric-transformations metricName=...,metricNamespace=...,metricValue=1
aws logs put-subscription-filter --log-group-name ... --filter-name ... \
  --filter-pattern '...' --destination-arn <arn>

# X-Ray
aws xray get-service-graph --start-time <epoch-sec> --end-time <epoch-sec>
aws xray get-trace-summaries --start-time <epoch-sec> --end-time <epoch-sec>
aws xray batch-get-traces --trace-ids <id> <id> ...
```

## Try it

Deploy a small model behind a Lambda-fronted SageMaker endpoint and instrument the whole path. From the Lambda, emit an EMF log line on every request carrying `PredictionConfidence` and `InputTokens` in a custom namespace, and confirm the metrics appear in CloudWatch without any `put-metric-data` call. Set 90-day retention on the log group, then write a Logs Insights query that bins average latency and low-confidence call counts over 5-minute windows. Build a `put-dashboard` panel showing endpoint p99 `ModelLatency` next to `GPUUtilization` from the `/aws/sagemaker/Endpoints` namespace, and put a p99-latency static alarm and an anomaly-detection alarm on `OutputTokenCount` (or your confidence metric) with `datapoints-to-alarm` tuned to avoid single-spike pages. Finally, enable X-Ray on the Lambda and endpoint, send a burst of requests, and read `get-service-graph` to confirm you can see the latency split between the Lambda hop and the model hop — that split is the whole point.
