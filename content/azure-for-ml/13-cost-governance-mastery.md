# 13 — Cost, Monitoring, and Governance

A system that trains and serves models is not done when it works — it is done when it works *reliably, affordably, and within the rules*. GPU compute is the most expensive line item most ML teams have, and an unmonitored endpoint or an ungoverned subscription is how a project quietly bleeds budget or fails an audit. This final section covers the operational discipline that turns a working ML platform into a mastered one: **monitoring and observability** (Azure Monitor and model monitoring), **cost management** (visibility, budgets, optimization), and **governance** (Azure Policy, tagging, security posture). These are the cross-cutting concerns from the reference architecture, made concrete.

## Monitoring and observability

**Azure Monitor** is the umbrella observability service; its two pillars are **metrics** (numeric time series — CPU, GPU utilization, request latency, requests per second) and **logs** (structured events queried in **Log Analytics** with the Kusto Query Language, KQL). **Application Insights** is the application-level slice for tracing requests through your services.

For ML, three layers of monitoring matter:

**Infrastructure and endpoint health.** Every managed online endpoint emits metrics — request rate, latency percentiles, error rate, and instance utilization — into Azure Monitor. These drive the **autoscale** rules from the deployment section and feed **alerts** so you know before users do.

Metrics and logs go to different sinks, and both must be wired up deliberately — this is the step most people miss, because an endpoint emits *nothing* to Log Analytics until you attach a **diagnostic setting** routing its categories there. So the foundation is a **Log Analytics workspace**, and a diagnostic setting on each resource forwarding logs and platform metrics into it:

```bash
# 1. The workspace that stores logs and powers KQL
az monitor log-analytics workspace create -g rg-mlx-dev \
  --workspace-name log-mlx-dev --location eastus2 --retention-time 90

# 2. Route the endpoint's logs + metrics into it (nothing arrives without this)
WS_ID=$(az monitor log-analytics workspace show -g rg-mlx-dev \
  --workspace-name log-mlx-dev --query id -o tsv)
EP_ID=$(az ml online-endpoint show -n fraud-endpoint -g rg-mlx-dev -w mlw-mlx-dev --query id -o tsv)
az monitor diagnostic-settings create --name to-log-analytics \
  --resource "$EP_ID" --workspace "$WS_ID" \
  --logs    '[{"categoryGroup":"allLogs","enabled":true}]' \
  --metrics '[{"category":"AllMetrics","enabled":true}]'
```

With telemetry flowing, you can browse what a resource emits and read raw metric values from the CLI before wiring an alert — `metrics list-definitions` shows the available metric names, `metrics list` pulls the time series:

```bash
# Discover which metrics the endpoint exposes, then read one
az monitor metrics list-definitions --resource "$EP_ID" -o table
az monitor metrics list --resource "$EP_ID" \
  --metric RequestLatency_P95 --interval PT5M --aggregation Average -o table

# Alert when p95 endpoint latency exceeds 500ms over 5 minutes
az monitor metrics alert create -g rg-mlx-dev --name high-latency \
  --scopes "$EP_ID" \
  --condition "avg RequestLatency > 500" --window-size 5m --evaluation-frequency 1m \
  --severity 2 --action "$(az monitor action-group show -n oncall -g rg-mlx-dev --query id -o tsv)" \
  --description "Fraud endpoint p95 latency high"

# The activity log is the control-plane audit trail (who changed/deleted what)
az monitor activity-log list -g rg-mlx-dev --offset 24h \
  --query "[?operationName.value=='Microsoft.MachineLearningServices/workspaces/onlineEndpoints/delete']"
```

The `--action` on the alert points at an **action group** — the fan-out target (email, SMS, webhook, a Function, a PagerDuty/Teams hook) that turns a fired alert into a page. An alert with no action group fires silently into the portal; wire the action group first.

```kusto
// Log Analytics (KQL): error rate by deployment over the last hour
AmlOnlineEndpointConsoleLog
| where TimeGenerated > ago(1h)
| summarize errors = countif(Level == "ERROR"), total = count() by DeploymentName
| extend error_rate = round(100.0 * errors / total, 2)
```

**Model quality and drift.** The failure mode unique to ML is that infrastructure stays green while the model quietly degrades because the world changed. **Azure ML model monitoring** addresses this: it captures production inference inputs and outputs and compares them against the **training baseline** on a schedule, computing **data drift** (input distribution shift), **prediction drift** (output distribution shift), and **data-quality** signals (nulls, out-of-range, schema violations). When a metric crosses a threshold, it alerts — and that alert is the trigger to retrain, closing the loop in the reference architecture. Remember the discipline from good ML practice: the **eval set is frozen**, so drift is measured against a stable reference; rebalancing or replacing the baseline invalidates the comparison.

```python
from azure.ai.ml.entities import (
    MonitorSchedule, MonitorDefinition, DataDriftSignal, MonitoringTarget)

monitor = MonitorSchedule(
    name="fraud-drift-monitor",
    create_monitor=MonitorDefinition(
        monitoring_target=MonitoringTarget(endpoint_deployment_id=
            "azureml:fraud-endpoint:blue"),
        monitoring_signals={"drift": DataDriftSignal()},   # vs. training baseline
    ),
)
ml_client.schedules.begin_create_or_update(monitor)
```

The monitor runs on a recurrence and is itself an `az ml schedule` object, so CI/CD manages it declaratively and you operate it from the CLI — list what monitors exist, disable one during a planned data migration that would otherwise trip false drift alarms, and re-enable it after:

```bash
# Define the monitor in YAML (trigger + drift signals + baseline) and apply it
az ml schedule create --file drift-monitor.yml -g rg-mlx-dev -w mlw-mlx-dev
az ml schedule list -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml schedule show -n fraud-drift-monitor -g rg-mlx-dev -w mlw-mlx-dev
# Pause during a known distribution shift (a migration, a promo) to suppress false alarms
az ml schedule disable -n fraud-drift-monitor -g rg-mlx-dev -w mlw-mlx-dev
az ml schedule enable  -n fraud-drift-monitor -g rg-mlx-dev -w mlw-mlx-dev
```

**Cost as a signal.** Spend is telemetry too — a training run that costs 5x its usual is often a bug (a job stuck retrying, a cluster that failed to scale down). Watch cost with the same seriousness as latency.

## Cost management

**Microsoft Cost Management** is the built-in FinOps toolset. The practice runs in a loop: **visibility** (see where money goes) → **allocation** (attribute it to teams/projects) → **optimization** (cut waste) → **forecasting** → **governance** (keep waste from returning).

**Visibility and allocation** depend entirely on **tags**. This is why every earlier section insisted on tagging resources with `project`, `env`, and `owner` from day one — Cost Analysis groups and filters by tag, so a consistent tagging scheme is the difference between "we spent $80k" and "the fraud project's dev endpoints spent $12k, mostly on idle GPU." Enforce tags with Azure Policy (below) so they cannot be omitted.

**Budgets and alerts** put guardrails on spend: set a budget per subscription or resource group with alert thresholds so you are notified at 50/80/100% of budget, and use anomaly alerts to catch spikes.

```bash
az consumption budget create --budget-name fraud-dev-monthly \
  --amount 3000 --time-grain Monthly \
  --resource-group rg-mlx-dev \
  --category Cost --start-date 2026-07-01 --end-date 2027-07-01

# Enumerate and manage budgets
az consumption budget list -g rg-mlx-dev -o table
az consumption budget show --budget-name fraud-dev-monthly -g rg-mlx-dev
az consumption budget delete --budget-name fraud-dev-monthly -g rg-mlx-dev
```

For actual spend analysis rather than guardrails, **Cost Management** exposes two commands (in the `costmanagement` extension). `costmanagement query` runs an ad-hoc, on-demand aggregation — the CLI equivalent of the Cost Analysis blade, and the fastest way to answer "what did the fraud project's dev resources cost this month, grouped by service?" `costmanagement export` schedules a recurring dump of the full cost dataset to a storage account for a FinOps dashboard or chargeback pipeline:

```bash
# Ad-hoc: month-to-date actual cost for this RG, grouped by service
az costmanagement query --type ActualCost --timeframe MonthToDate \
  --scope "/subscriptions/<sub-id>/resourceGroups/rg-mlx-dev" \
  --dataset-aggregation '{"totalCost":{"name":"Cost","function":"Sum"}}' \
  --dataset-grouping name=ServiceName type=Dimension

# Recurring: export the full cost dataset daily to storage for dashboards
az costmanagement export create --name daily-cost-export \
  --scope "/subscriptions/<sub-id>/resourceGroups/rg-mlx-dev" \
  --storage-account-id "$(az storage account show -n stmlxdata -g rg-mlx-dev --query id -o tsv)" \
  --storage-container costexports --storage-directory mlx \
  --timeframe MonthToDate --recurrence Daily --schedule-status Active
az costmanagement export list --scope "/subscriptions/<sub-id>/resourceGroups/rg-mlx-dev" -o table
```

**Optimization** is where ML teams save the most, because GPU is the biggest cost:

- **Scale to zero.** Compute clusters and batch endpoints with `min_instances=0` cost nothing idle — the single highest-impact habit. Verify clusters actually deallocate.
- **Spot / LowPriority** compute for fault-tolerant training cuts 60–90% off GPU cost when you checkpoint.
- **Autoscale online endpoints** with a tight warm minimum and a capped maximum; do not run 10 GPU instances at 3 a.m.
- **Reservations and savings plans** for steady, predictable workloads (a always-on prod endpoint) trade commitment for a large discount versus pay-as-you-go.
- **Provisioned throughput (PTU)** for high, steady GenAI volume; pay-per-token for spiky/dev traffic.
- **Storage tiering** (Hot→Cool→Cold→Archive via lifecycle policies) for datasets and artifacts that age out.
- **Right-size** — the biggest GPU is not always fastest per dollar; measure throughput-per-dollar, not raw speed.

## Governance

Governance keeps a growing platform consistent and compliant. The core tool is **Azure Policy**: you define rules and Azure evaluates and enforces them across subscriptions — denying non-compliant deployments or auto-remediating. For an ML platform, high-value policies include:

- **Require tags** (`project`, `env`, `owner`) on all resources — the foundation of cost allocation.
- **Restrict regions** to those approved for data residency and where you have GPU quota.
- **Restrict VM SKUs** to prevent someone spinning up an unapproved (or oversized) GPU family by mistake.
- **Enforce private endpoints / deny public network access** on storage, workspaces, and endpoints — network posture as code.
- **Require managed identity / RBAC** and forbid legacy access policies on Key Vault.

Azure ships hundreds of **built-in** policy definitions (referenced by GUID, like the require-tag one above), and you author **custom** ones with `az policy definition create` when nothing built-in fits — the rule body is a JSON `if`/`then` where the `then.effect` is `deny`, `audit`, `modify`, or `deployIfNotExists`. You then group definitions into an **initiative** (a policy *set*) so a whole compliance standard assigns as one unit:

```bash
# Author a custom definition (rule + effect in a JSON file)
az policy definition create --name deny-public-storage \
  --rules @deny-public-storage.rules.json --mode All \
  --display-name "Storage accounts must disable public network access"

# Bundle definitions into an initiative and assign the bundle
az policy set-definition create --name ml-platform-baseline \
  --definitions @initiative-definitions.json \
  --display-name "ML platform governance baseline"

# Enforce a required 'project' tag on all resources in a scope
az policy assignment create --name require-project-tag \
  --policy "871b6d14-10aa-478d-b590-94f262ecfa99" \
  --params '{"tagName": {"value": "project"}}' \
  --scope "/subscriptions/<sub-id>/resourceGroups/rg-mlx-dev"

# Assign the whole initiative at management-group scope; grant it a MI for modify/DINE effects
az policy assignment create --name ml-baseline \
  --policy-set-definition ml-platform-baseline \
  --scope "/providers/Microsoft.Management/managementGroups/mg-mlx" \
  --mi-system-assigned --identity-scope "/subscriptions/<sub-id>" --location eastus2
```

Assignment is only half the job — you have to *watch* compliance and *fix* existing drift. `az policy state` reports which resources are compliant or not, and for `deployIfNotExists`/`modify` policies, `az policy remediation` retroactively brings already-deployed resources into line (a fresh assignment only governs *new* deployments until you remediate):

```bash
# Which resources are non-compliant, and why
az policy state list --resource-group rg-mlx-dev \
  --filter "complianceState eq 'NonCompliant'" -o table
az policy state summarize --resource-group rg-mlx-dev

# Remediate existing non-compliant resources for a modify/DINE assignment
az policy remediation create --name fix-tags \
  --policy-assignment ml-baseline \
  --resource-discovery-mode ExistingNonCompliant
az policy remediation list --resource-group rg-mlx-dev -o table
```

Tags themselves are managed with `az tag` (and enforced by the policies above), so you can bulk-apply the `project`/`env`/`owner` scheme or audit what is missing:

```bash
az tag create --resource-id "$(az group show -n rg-mlx-dev --query id -o tsv)" \
  --tags project=fraud env=dev owner=humza
az tag list --resource-id "$(az group show -n rg-mlx-dev --query id -o tsv)"
```

Policies group into **initiatives**, and apply at **management-group** scope so every subscription inherits them — recall the management-group hierarchy from the overview section, which exists precisely so governance applies broadly and consistently. Pair policy with **Microsoft Defender for Cloud** for continuous security-posture assessment and recommendations across the platform — enable its plans and read its findings from the CLI:

```bash
# Turn on Defender for Cloud plans and review the security posture
az security pricing create -n VirtualMachines --tier Standard
az security pricing list -o table
az security assessment list -o table          # posture findings to act on
az security alert list -o table               # active security alerts
```

## The mastery checklist

You have mastered Azure for ML when your platform satisfies all of the following:

- **Identity:** no API keys or storage keys in code; everything uses managed identity + `DefaultAzureCredential`; least-privilege RBAC scoped tightly; Key Vault in RBAC mode for the rare secret.
- **Network:** storage, registry, vault, workspace, endpoints, AI Search, and Foundry are on private endpoints with public access disabled; the ML workspace uses managed-VNet isolation.
- **Compute:** training on scale-to-zero clusters with LowPriority/spot; real-time endpoints autoscaled with a warm minimum; the right SKU (ND for distributed, NC for single-node) chosen deliberately with quota confirmed.
- **Data:** medallion lake in Parquet/Delta with lifecycle tiering; versioned data assets; an online feature store for serving.
- **ML lifecycle:** training as reproducible pipelines with MLflow tracking; versioned models in a registry with lineage; dev→prod promotion.
- **Delivery:** CI/CD builds images and runs the training pipeline; deployments roll out via canary with instant rollback.
- **Observability:** endpoint metrics/alerts drive autoscale; model monitoring detects drift against a frozen baseline and triggers retraining.
- **Cost:** consistent tagging; budgets with alerts; reservations/spot/tiering applied; idle GPU eliminated.
- **Governance:** Azure Policy enforces tags, regions, SKUs, and network posture at management-group scope; Defender watches security posture.
- **Everything reproducible** in Bicep/Terraform — nothing production-critical born from a portal click.

## Key takeaways

- **Azure Monitor** (metrics + Log Analytics/KQL + Application Insights) covers infra and endpoint health and drives **autoscale and alerts**; **Azure ML model monitoring** covers the ML-specific failure — **drift** against a frozen baseline — and triggers retraining.
- **Cost Management** runs a loop of visibility → allocation → optimization → forecasting → governance; **consistent tagging** is the prerequisite, and **eliminating idle GPU** (scale-to-zero, spot, autoscale, reservations, tiering) is the biggest lever.
- **Azure Policy** enforces tags, regions, SKUs, and network posture as code at **management-group** scope; **Defender for Cloud** watches security posture.
- **Mastery** = keyless identity, private networking, cost-disciplined compute, versioned reproducible ML lifecycle, canary delivery, drift-aware monitoring, enforced governance, all provisioned as **IaC**.

## CLI cheat-sheet

```bash
# --- Monitoring: Log Analytics, diagnostics, metrics, alerts ---
az monitor log-analytics workspace create -g rg-mlx-dev --workspace-name log-mlx-dev --retention-time 90
az monitor diagnostic-settings create --name to-log-analytics --resource "$EP_ID" --workspace "$WS_ID" \
  --logs '[{"categoryGroup":"allLogs","enabled":true}]' --metrics '[{"category":"AllMetrics","enabled":true}]'
az monitor metrics list-definitions --resource "$EP_ID" -o table
az monitor metrics list --resource "$EP_ID" --metric RequestLatency_P95 --interval PT5M --aggregation Average
az monitor metrics alert create -g rg-mlx-dev --name high-latency --scopes "$EP_ID" \
  --condition "avg RequestLatency > 500" --window-size 5m --severity 2 --action "$AG_ID"
az monitor action-group create -g rg-mlx-dev -n oncall --short-name oncall --action email me me@x.com
az monitor activity-log list -g rg-mlx-dev --offset 24h -o table
# Run a KQL query against Log Analytics
az monitor log-analytics query -w "$WS_ID" --analytics-query "AmlOnlineEndpointConsoleLog | take 50"

# --- Model monitoring (drift) ---
az ml schedule create --file drift-monitor.yml -g rg-mlx-dev -w mlw-mlx-dev
az ml schedule list -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml schedule disable -n fraud-drift-monitor -g rg-mlx-dev -w mlw-mlx-dev
az ml schedule enable  -n fraud-drift-monitor -g rg-mlx-dev -w mlw-mlx-dev

# --- Cost management ---
az consumption budget create --budget-name fraud-dev-monthly --amount 3000 --time-grain Monthly \
  --resource-group rg-mlx-dev --category Cost --start-date 2026-07-01 --end-date 2027-07-01
az consumption budget list -g rg-mlx-dev -o table
az costmanagement query --type ActualCost --timeframe MonthToDate --scope "$RG_SCOPE" \
  --dataset-aggregation '{"totalCost":{"name":"Cost","function":"Sum"}}' \
  --dataset-grouping name=ServiceName type=Dimension
az costmanagement export create --name daily-cost-export --scope "$RG_SCOPE" \
  --storage-account-id "$SA_ID" --storage-container costexports --storage-directory mlx \
  --timeframe MonthToDate --recurrence Daily --schedule-status Active

# --- Governance: policy, tags, Defender ---
az policy definition create --name deny-public-storage --rules @rules.json --mode All
az policy set-definition create --name ml-platform-baseline --definitions @initiative.json
az policy assignment create --name require-project-tag --policy <builtin-guid> \
  --params '{"tagName":{"value":"project"}}' --scope "$RG_SCOPE"
az policy assignment create --name ml-baseline --policy-set-definition ml-platform-baseline \
  --scope "$MG_SCOPE" --mi-system-assigned --identity-scope "$SUB_SCOPE" --location eastus2
az policy state list -g rg-mlx-dev --filter "complianceState eq 'NonCompliant'" -o table
az policy state summarize -g rg-mlx-dev
az policy remediation create --name fix-tags --policy-assignment ml-baseline \
  --resource-discovery-mode ExistingNonCompliant
az tag create --resource-id "$RG_ID" --tags project=fraud env=dev owner=humza
az tag list --resource-id "$RG_ID"
az security pricing create -n VirtualMachines --tier Standard      # enable Defender plan
az security assessment list -o table
az security alert list -o table
```

## Try it

Instrument the platform you have been building. Create a metric alert on your online endpoint's latency and a monthly budget on its resource group with alerts at 80% and 100%. Set up an Azure ML model-monitoring schedule that compares live inputs to the training baseline and alerts on data drift. Write and assign an Azure Policy requiring a `project` tag on all resources, then try to create an untagged resource and watch it be denied. Finally, run through the mastery checklist against your own platform and write down every box you cannot yet tick — that list is your roadmap from working to mastered.
