# 15 — Observability: Azure Monitor, Log Analytics, Application Insights

An ML service that returns predictions is not the same as an ML service you can *operate*. Operating one means seeing its latency and error rate in real time, tracing a slow request through every hop, querying weeks of logs to explain an incident, and getting paged before users notice. Azure's observability stack — **Azure Monitor** as the umbrella, **Log Analytics** for logs and queries, **Application Insights** for application tracing, and **alerts** on top of both — is what turns a deployed model into an operable one. The cost/governance module introduced monitoring as a cross-cutting concern and covered ML-specific *drift*; this module goes deep on the infrastructure and application observability underneath it: where telemetry comes from, how to store and query it, and how to alert on it. In the end-to-end solution, this is the nervous system that drives autoscale, catches regressions, and closes the loop back to retraining.

## The pillars: metrics, logs, and traces

Azure Monitor collects three kinds of telemetry, and knowing which is which tells you where to look:

- **Metrics** — numeric time series, pre-aggregated, cheap, near-real-time (CPU/GPU utilization, request latency percentiles, requests per second, queue depth). Metrics drive **autoscale** and fast alerts.
- **Logs** — structured events with a schema, stored in a **Log Analytics workspace** and queried with the **Kusto Query Language (KQL)**. Logs answer "what happened and why" — every request, every error stack, every diagnostic line.
- **Traces** — distributed traces that follow one request across services (the front-door Function → the online endpoint → the feature store), collected by **Application Insights** so you can see where a slow request spent its time.

Everything routes into a **Log Analytics workspace**, which is the query engine and retention store at the center of the stack. Create it first; most other resources point their **diagnostic settings** at it.

```bash
# The central log store, and a workspace-based Application Insights on top of it
az monitor log-analytics workspace create -g rg-mlx-dev -n law-mlx --location eastus2 --retention-time 90
LAW_ID=$(az monitor log-analytics workspace show -g rg-mlx-dev -n law-mlx --query id -o tsv)
az monitor app-insights component create --app appi-mlx -g rg-mlx-dev --location eastus2 \
  --kind web --application-type web --workspace "$LAW_ID"
```

Modern Application Insights is **workspace-based** — it stores its data in Log Analytics rather than a separate store, so you query app traces and infra logs together in one place. The Azure ML **workspace** is created with an Application Insights component attached; endpoint and job telemetry flow there automatically.

## Diagnostic settings: getting telemetry into the workspace

A resource emits platform metrics automatically, but its **resource logs** (detailed diagnostics) only flow when you turn on a **diagnostic setting** routing them to your Log Analytics workspace. This is the step people forget, and its absence looks like "the logs just aren't there." Wire it for every resource whose logs you want to keep and query — endpoints, storage, Key Vault, the workspace itself:

```bash
# Send an online endpoint's logs + metrics to the central workspace
EP_ID=$(az ml online-endpoint show -n fraud-endpoint -g rg-mlx-dev -w mlw-mlx-dev --query id -o tsv)
az monitor diagnostic-settings create -n to-law --resource "$EP_ID" \
  --workspace "$LAW_ID" \
  --logs    '[{"categoryGroup":"allLogs","enabled":true}]' \
  --metrics '[{"category":"AllMetrics","enabled":true}]'

az monitor diagnostic-settings list --resource "$EP_ID" -o table
```

## Querying with KQL

KQL is the language you live in during an incident. It reads left-to-right through a pipe of operators — filter, aggregate, project — over tables that Azure populates (`AmlOnlineEndpointConsoleLog`, `AmlOnlineEndpointTrafficLog`, `AppRequests`, `AppExceptions`, `AzureMetrics`). A few patterns cover most ML operations:

```kusto
// Error rate by deployment over the last hour (which deployment is misbehaving?)
AmlOnlineEndpointConsoleLog
| where TimeGenerated > ago(1h)
| summarize errors = countif(Level == "ERROR"), total = count() by DeploymentName
| extend error_rate = round(100.0 * errors / total, 2)
| order by error_rate desc
```

```kusto
// p50/p95/p99 request latency in 5-minute buckets (is latency creeping up?)
AmlOnlineEndpointTrafficLog
| where TimeGenerated > ago(6h)
| summarize p50=percentile(ResponseTimeMs,50), p95=percentile(ResponseTimeMs,95),
            p99=percentile(ResponseTimeMs,99) by bin(TimeGenerated, 5m)
| render timechart
```

```kusto
// Top exceptions from the front-door Function (App Insights)
AppExceptions
| where TimeGenerated > ago(24h)
| summarize count() by ProblemId, OuterMessage
| top 10 by count_
```

Run queries from the CLI for scripting and CI checks, not just the portal:

```bash
az monitor log-analytics query -w "$LAW_ID" \
  --analytics-query "AmlOnlineEndpointConsoleLog | where TimeGenerated > ago(1h) | summarize count() by Level" \
  -o table
```

## Alerts: metric, log, and smart

Alerts are how the system tells *you* instead of you polling dashboards. Three kinds matter for ML:

- **Metric alerts** — evaluate a metric against a threshold on a schedule (p95 latency > 500 ms, GPU utilization > 90%, error rate > 1%). Fast and cheap; the workhorse.
- **Log (scheduled query) alerts** — run a KQL query on a cadence and alert on the result (e.g. more than N `ERROR` lines in 5 minutes, or a specific exception appearing). Use when the condition needs log context a raw metric cannot express.
- **Alert processing rules and action groups** — an **action group** is the reusable "who to notify and how" (email, SMS, webhook, a Function, PagerDuty). Attach it to alerts so on-call routing lives in one place.

```bash
# Action group (once), then a metric alert and a log alert that use it
az monitor action-group create -g rg-mlx-dev -n ag-oncall \
  --short-name oncall --action email sre sre@example.com

az monitor metrics alert create -g rg-mlx-dev -n high-latency \
  --scopes "$EP_ID" --condition "avg RequestLatency > 500" --window-size 5m --evaluation-frequency 1m \
  --action ag-oncall --description "Fraud endpoint p95 latency high"

az monitor scheduled-query create -g rg-mlx-dev -n error-spike \
  --scopes "$LAW_ID" --action-groups ag-oncall \
  --condition "count 'errs' > 20" --condition-query errs="AmlOnlineEndpointConsoleLog | where Level == 'ERROR'" \
  --evaluation-frequency 5m --window-size 5m
```

Alert **fatigue** is a real failure mode: too many low-value alerts and on-call ignores the one that matters. Alert on symptoms users feel (latency, error rate, availability), keep thresholds meaningful, and route noisy diagnostics to a dashboard rather than a page.

## Workbooks and dashboards

For the at-a-glance view — an SRE dashboard of endpoint latency, throughput, error rate, and GPU utilization — **Azure Monitor Workbooks** compose metrics and KQL into shareable, parameterized reports. They are defined as JSON (so they live in source control and deploy with the rest of the platform) and are the right home for the standing views you do not want cluttering your alert rules.

## The three ML monitoring layers, unified

Recall the three-layer framing from the cost/governance module, now grounded in this stack:

1. **Infra and endpoint health** — platform metrics + resource logs in Log Analytics, driving metric alerts and autoscale. (This module.)
2. **Application tracing** — Application Insights distributed traces across the front door, endpoint, and dependencies, to find *where* latency or errors originate. (This module.)
3. **Model quality and drift** — Azure ML model monitoring comparing live inputs/outputs to the frozen training baseline. (Covered in the cost/governance module; its alerts flow through the same action groups.)

Keeping all three in one Log Analytics workspace means an incident query can join an endpoint latency spike (layer 1) to the slow dependency causing it (layer 2) to a drift event that shifted the input distribution (layer 3) — one place, one query language.

## How observability fits the whole solution

Observability is the platform's feedback loop. Every **online endpoint** and the front-door **Function** route logs and metrics into the central **Log Analytics workspace** via diagnostic settings; **Application Insights** traces requests across them. **Metric alerts** on latency and utilization both drive **autoscale** and page on-call through a shared **action group**. **KQL** is the incident-response and CI language. **Workbooks** give the standing dashboards. Model **drift** monitoring reports into the same workspace and action groups, so when the world shifts the same on-call pipeline that catches a latency spike catches a quality regression — and that alert is the trigger to retrain, closing the loop back to the training module.

## Key takeaways

- Azure Monitor collects **metrics** (numeric, fast, drive autoscale), **logs** (structured, queried with **KQL** in a **Log Analytics workspace**), and **traces** (**Application Insights**, request-level). Create the **workspace-based** App Insights on top of one Log Analytics workspace.
- Resource logs only flow when you turn on a **diagnostic setting** pointing at the workspace — the most-forgotten step; wire it for endpoints, storage, vault, and the ML workspace.
- **KQL** is the incident language; know the endpoint tables (`AmlOnlineEndpointConsoleLog`, `...TrafficLog`) and App Insights tables (`AppRequests`, `AppExceptions`), and query from the CLI for automation.
- Alert with **metric alerts** (fast thresholds), **log/scheduled-query alerts** (need log context), and route via reusable **action groups**; guard against alert fatigue by paging on user-facing symptoms only.
- Keep infra health, app traces, and **model drift** in **one workspace** so incident queries can join across all three ML monitoring layers; **Workbooks** hold the standing dashboards.

## CLI cheat-sheet

```bash
# --- central log store & app insights ---
az monitor log-analytics workspace create -g rg-mlx-dev -n law-mlx --location eastus2 --retention-time 90
az monitor log-analytics workspace show   -g rg-mlx-dev -n law-mlx --query id -o tsv
az monitor app-insights component create --app appi-mlx -g rg-mlx-dev --location eastus2 \
  --kind web --application-type web --workspace "$LAW_ID"
az monitor app-insights component show   --app appi-mlx -g rg-mlx-dev --query connectionString -o tsv

# --- diagnostic settings (route resource logs to the workspace) ---
az monitor diagnostic-settings create -n to-law --resource "$EP_ID" --workspace "$LAW_ID" \
  --logs '[{"categoryGroup":"allLogs","enabled":true}]' --metrics '[{"category":"AllMetrics","enabled":true}]'
az monitor diagnostic-settings list  --resource "$EP_ID" -o table

# --- query logs (KQL) ---
az monitor log-analytics query -w "$LAW_ID" \
  --analytics-query "AmlOnlineEndpointConsoleLog | where TimeGenerated > ago(1h) | summarize count() by Level" -o table

# --- list available metrics for a resource ---
az monitor metrics list-definitions --resource "$EP_ID" -o table
az monitor metrics list --resource "$EP_ID" --metric RequestLatency --interval PT1M -o table

# --- action group (who/how to notify) ---
az monitor action-group create -g rg-mlx-dev -n ag-oncall --short-name oncall \
  --action email sre sre@example.com --action webhook slack https://hooks.example/xyz

# --- metric alert ---
az monitor metrics alert create -g rg-mlx-dev -n high-latency --scopes "$EP_ID" \
  --condition "avg RequestLatency > 500" --window-size 5m --evaluation-frequency 1m --action ag-oncall

# --- log (scheduled query) alert ---
az monitor scheduled-query create -g rg-mlx-dev -n error-spike --scopes "$LAW_ID" --action-groups ag-oncall \
  --condition "count 'errs' > 20" --condition-query errs="AmlOnlineEndpointConsoleLog | where Level=='ERROR'" \
  --evaluation-frequency 5m --window-size 5m

# --- manage alerts ---
az monitor metrics alert list -g rg-mlx-dev -o table
az monitor activity-log alert list -g rg-mlx-dev -o table
```

## Try it

Create a Log Analytics workspace and a workspace-based Application Insights. Add a diagnostic setting routing your online endpoint's logs and metrics into the workspace, send it some traffic, then write a KQL query that reports p95 latency in 5-minute buckets and another that counts errors by deployment. Create an action group with your email, a metric alert on latency > 500 ms, and a scheduled-query alert on an error spike — then trigger each and confirm you get notified. Finally, sketch how a single incident (latency spike) would be diagnosed by joining infra metrics, an App Insights trace, and a drift signal in the one workspace.
