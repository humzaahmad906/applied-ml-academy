# 01 — GCP Overview, Console, and gcloud

Google Cloud is the foundation you will build every machine learning system on in this course. Before you touch Vertex AI, BigQuery, or a single GPU, you need a mental model of how the platform is organized and fluency with the two interfaces you will use daily: the Cloud Console (the web UI) and the `gcloud` command-line interface. This module gives you that foundation and the vocabulary the rest of the course assumes.

## The resource hierarchy

Everything in Google Cloud lives inside a hierarchy, and understanding it early saves you from painful reorganizations later.

- **Organization** — the root node, tied to a Cloud Identity or Google Workspace domain. It represents your company. Policies set here cascade downward.
- **Folders** — optional grouping under the organization, typically mapping to departments, teams, or environments (for example, a `data-science` folder or separate `prod` and `dev` folders).
- **Projects** — the fundamental unit of everything. A project is a billing, IAM, and API boundary. Resources (a bucket, a training job, an endpoint) belong to exactly one project. You enable APIs per project, grant access per project, and get billed per project.
- **Resources** — the actual things: Compute Engine VMs, Cloud Storage buckets, Vertex AI models, BigQuery datasets.

For an ML team, a common pattern is one project per environment per workload: `myco-fraud-dev`, `myco-fraud-prod`. This isolates experiments from production, keeps billing legible, and lets you grant a data scientist broad rights in `dev` while locking down `prod`. A project has both a human-friendly name and a globally unique, immutable **project ID** — the ID is what every command and API call references.

You can inspect and create the hierarchy from the CLI. Reading the org and folder structure requires `roles/resourcemanager.organizationViewer` / `folderViewer` at the appropriate level, so these often only work for platform admins:

```bash
# Walk the hierarchy top-down
gcloud organizations list                        # your org(s) and their numeric IDs
gcloud resource-manager folders list \
  --organization=123456789012                    # folders directly under the org
gcloud resource-manager folders list \
  --folder=456789012345                           # nested folders under a folder

# Projects: create, describe, list, filter
gcloud projects create myco-fraud-dev \
  --name="Fraud (dev)" --folder=456789012345
gcloud projects describe myco-fraud-dev
gcloud projects list --filter="parent.id=456789012345"
gcloud projects list \
  --filter="labels.env=dev AND lifecycleState=ACTIVE" \
  --format="table(projectId, name, projectNumber)"
```

**Gotcha — project IDs are permanent.** The project ID is chosen once at creation and can never be changed or reused (even after deletion, the ID is reserved). Pick a naming convention you can live with. The display *name* is mutable; the ID is not. Every project also has an auto-assigned numeric **project number** — some APIs and service-agent identities reference that instead of the ID.

## Regions and zones

Google Cloud runs in **regions** (independent geographic areas like `us-central1`, `europe-west4`, `asia-northeast1`) each containing multiple **zones** (`us-central1-a`, `us-central1-b`, and so on). A zone is a deployment target inside a region; regions are the unit of geographic redundancy.

This matters enormously for ML:

- **Accelerator availability is regional.** The newest GPUs (NVIDIA H200 in A3 Ultra VMs, B200 in A4, GB200 in A4X) and TPUs (Trillium/v6e, Ironwood/tpu7x) are only in specific regions. You cannot assume every region has every chip. Always check accelerator availability before committing a pipeline to a region.
- **Data gravity and egress.** Keep compute in the same region as your data. Reading a multi-terabyte training set from a bucket in another region is slow and incurs network egress charges. Co-locate your Cloud Storage bucket, BigQuery dataset, and training job.
- **Latency.** Serving endpoints should sit near your users.
- **Compliance.** Data residency requirements dictate region choice.

A few services are **global** (IAM, Cloud DNS), some are **multi-regional** (Cloud Storage can use a multi-region like `US` or `EU`), and most compute is **zonal** or **regional**.

Regional capacity is governed by **quotas** — per-project, per-region ceilings on how many resources you can allocate (vCPUs of a given family, GPUs of a given type, IP addresses, and so on). Quotas are the reason a `create` call fails with `QUOTA_EXCEEDED` even though the region physically has capacity. The two you will hit as an ML engineer are the **GPU quotas** (for example `NVIDIA_A100_GPUS` in `us-central1`), which default to **zero** and must be explicitly requested, and total regional vCPU limits.

```bash
# What a region offers and your current quota headroom there
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric, quotas.usage, quotas.limit)"

# Region-level metadata (available zones, status)
gcloud compute regions list --filter="name~us-"
```

Requesting an increase is done in the Console under **IAM & Admin → Quotas & System Limits** (filter by the metric and region, select it, and submit an increase); GPU requests typically approve within minutes to a day. Do this *before* you plan a training run, not when the job fails to launch.

## The Cloud Console

The Console (console.cloud.google.com) is the web UI. It is where you will explore new services, read dashboards, inspect a failed training job's logs, and click through Vertex AI's model registry. Key habits:

- The **project selector** at the top is the single most important control — almost everything you see is scoped to the currently selected project. A huge fraction of "why can't I see my resource?" confusion is simply the wrong project selected.
- **APIs & Services** is where you enable the APIs a project can use. New projects start with almost nothing enabled; you must turn on Compute Engine, Vertex AI, BigQuery, and so on before their resources become available.
- **Cloud Shell** (the terminal icon, top right) is a free, ephemeral Linux VM with `gcloud`, `python`, Docker, and the Cloud SDK preinstalled and pre-authenticated. It is the fastest way to run a command without configuring anything locally.

The Console is excellent for exploration and observability. It is a poor fit for anything you need to reproduce — for that, you use `gcloud` and, eventually, infrastructure-as-code.

## The gcloud CLI

`gcloud` is the command-line interface to Google Cloud and the backbone of every reproducible workflow. It ships in the Google Cloud SDK alongside `bq` (BigQuery) and `gcloud storage` (Cloud Storage). Installing and authenticating:

```bash
# Authenticate your user account (opens a browser)
gcloud auth login

# Set up application-default credentials, which client libraries
# (including the Vertex AI Python SDK) pick up automatically
gcloud auth application-default login

# Point the SDK at your project and a default region/zone
gcloud config set project myco-fraud-dev
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# See everything that's currently configured
gcloud config list

# Read or clear individual properties
gcloud config get compute/region          # print one value
gcloud config unset compute/zone          # remove a property
gcloud config list --format="value(core.project)"

# Environment sanity checks
gcloud info                               # SDK paths, active config, account, python
gcloud version                            # component versions
gcloud components list                    # what's installed / updatable
gcloud components update                  # keep the SDK current
```

Configurations are named profiles — a bundle of `project`, `region`, `account`, etc. If you work across `dev` and `prod`, keep two and switch between them instead of re-typing `config set`:

```bash
gcloud config configurations create prod
gcloud config set project myco-fraud-prod
gcloud config set compute/region us-central1

gcloud config configurations list                 # all profiles, * marks active
gcloud config configurations activate dev          # switch back
gcloud config configurations describe prod         # dump one profile's properties
gcloud config configurations delete old-sandbox    # remove one you no longer need
```

**Gotcha — the default-region surprise.** If you never set `compute/region` (or `compute/zone`), every zonal/regional command either prompts you interactively or fails, and scripts that assume a default silently target the wrong place. Set them per configuration on day one. Conversely, an *inherited* default from the wrong active configuration is a classic "why did this VM land in `us-east1`?" bug — check `gcloud config list` before you blame the command.

The command grammar is consistent: `gcloud <group> <subgroup> <verb> [flags]`. Once you internalize it, you can guess commands.

```bash
# Enable the APIs an ML project needs, all at once
gcloud services enable \
  compute.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com

# List enabled services, and discover what's available to enable
gcloud services list --enabled
gcloud services list --available --filter="name:aiplatform"
gcloud services disable run.googleapis.com   # turn one off (fails if others depend on it)

# Describe a resource as structured data
gcloud compute instances list --format=json
gcloud ai endpoints list --region=us-central1 --format="table(displayName,name)"
```

**Gotcha — API enablement is not instant.** After `gcloud services enable`, there is a short propagation delay (seconds to a couple of minutes) before the API's resources become usable. A `create` call fired immediately after enabling can fail with `SERVICE_DISABLED` or a `403`; if that happens, wait and retry rather than assuming the enable failed.

Two flags you will use constantly: `--format` and `--filter`. Learning their grammar turns `gcloud` into a query tool:

```bash
# --format: projections pull specific fields; table/csv/value/json/yaml choose shape
gcloud compute instances list \
  --format="table(name, zone.basename(), machineType.basename(), status)"
gcloud projects list --format="value(projectId)"          # bare values for scripts
gcloud compute instances list --format="json(name,status)" | jq '.[].name'

# --filter: field operators are = (exact), : (substring/has), ~ (regex),
#           combined with AND / OR / NOT
gcloud compute instances list --filter="status=RUNNING AND zone~us-central1"
gcloud compute machine-types list --filter="guestCpus>=8 AND memoryMb<32768"
gcloud compute images list --filter="family~pytorch AND NOT deprecated:*"
```

Piping `--format=json` into `jq` is a staple of scripting against Google Cloud, but `--format="value(...)"` avoids the extra dependency for simple field extraction.

Authentication has its own small toolbox you will reach for constantly — checking which identity is active is the first debugging step for any "permission denied":

```bash
gcloud auth list                          # all credentialed accounts, * marks active
gcloud auth print-access-token            # a short-lived OAuth token (e.g. for curl to a REST API)
gcloud auth revoke alice@example.com      # forget a stored credential
```

## Billing

Billing is attached at the project level. A **billing account** (a payment method) is linked to one or more projects; every chargeable resource in a project bills to its linked account. Two things every ML engineer must set up on day one, because ML workloads can burn money fast (a single multi-GPU node runs into the tens of dollars per hour):

- **Budgets and alerts.** Create a budget on the billing account with threshold alerts (for example, notify at 50%, 90%, 100% of a monthly amount). Alerts are informational — they do not cap spend — but they are your early warning system.
- **Labels.** Attach key-value **labels** to resources (`team=fraud`, `env=dev`, `component=training`). Labels flow into billing export, letting you attribute cost by team, model, or pipeline stage. Establishing a labeling convention now is one of the highest-leverage habits for the cost-mastery work later in the course.

```bash
# Find the billing account ID to use
gcloud billing accounts list

# Link a project to a billing account
gcloud billing projects link myco-fraud-dev \
  --billing-account=0X0X0X-0X0X0X-0X0X0X

# See what's linked
gcloud billing projects describe myco-fraud-dev

# Budgets live in their own command group (requires the Billing Budgets API)
gcloud billing budgets list --billing-account=0X0X0X-0X0X0X-0X0X0X
gcloud billing budgets create \
  --billing-account=0X0X0X-0X0X0X-0X0X0X \
  --display-name="fraud-dev monthly" \
  --budget-amount=500USD \
  --filter-projects=projects/myco-fraud-dev \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0
```

A budget alert only *notifies* — it never stops spend. Wiring an alert to a Pub/Sub topic so an automation can react (throttle jobs, page a human) is the pattern mature teams use, and it foreshadows the cost-mastery module later in the course.

For deep analysis, enable **billing export to BigQuery** — every line item lands in a dataset you can query with SQL, which is how mature teams answer "what did last month's training runs actually cost?"

## How this fits the whole solution

Everything downstream in this course — streaming ingestion with Pub/Sub, a data warehouse in BigQuery, training on Vertex AI, serving on Cloud Run or Vertex endpoints, monitoring, and cost control — lives inside the project and region structure you set up here. The `gcloud` fluency you build now is what lets you script, reproduce, and eventually codify that whole system as infrastructure-as-code. Get the foundation right and every later module is additive.

## Key takeaways

- The hierarchy is Organization → Folders → Projects → Resources; the **project** is the billing, IAM, and API boundary and the unit you scope almost everything to.
- **Regions and zones** govern accelerator availability, data locality, latency, and compliance — co-locate compute with data and verify GPU/TPU availability per region.
- The **Console** is for exploration and observability; **`gcloud`** is for anything reproducible. Learn the `gcloud <group> <verb>` grammar and the `--format`/`--filter` flags.
- Set up **`gcloud config`** (project + region), **application-default credentials** for the SDKs, and, on day one, a **budget with alerts** and a **labeling convention**.

## CLI cheat-sheet

```bash
# --- Auth ---
gcloud auth login                                   # user login (browser)
gcloud auth application-default login               # ADC for client libraries
gcloud auth list                                    # active + all accounts
gcloud auth print-access-token                      # short-lived OAuth token
gcloud auth revoke ACCOUNT                          # forget a credential

# --- Config & configurations ---
gcloud config set project myco-fraud-dev
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
gcloud config get compute/region
gcloud config unset compute/zone
gcloud config list
gcloud config configurations create prod
gcloud config configurations list
gcloud config configurations activate dev
gcloud config configurations describe prod
gcloud config configurations delete NAME

# --- Environment ---
gcloud info
gcloud version
gcloud components list
gcloud components update

# --- Hierarchy ---
gcloud organizations list
gcloud resource-manager folders list --organization=ORG_ID
gcloud resource-manager folders list --folder=FOLDER_ID

# --- Projects ---
gcloud projects create myco-fraud-dev --name="Fraud (dev)" --folder=FOLDER_ID
gcloud projects describe myco-fraud-dev
gcloud projects list --filter="labels.env=dev" --format="table(projectId,name)"

# --- Services / APIs ---
gcloud services enable compute.googleapis.com aiplatform.googleapis.com
gcloud services list --enabled
gcloud services list --available --filter="name:aiplatform"
gcloud services disable SERVICE

# --- Quotas & regions ---
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric,quotas.usage,quotas.limit)"
gcloud compute regions list

# --- Billing ---
gcloud billing accounts list
gcloud billing projects link myco-fraud-dev --billing-account=ACCT_ID
gcloud billing projects describe myco-fraud-dev
gcloud billing budgets list --billing-account=ACCT_ID
gcloud billing budgets create --billing-account=ACCT_ID \
  --display-name=NAME --budget-amount=500USD \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9

# --- Output shaping ---
gcloud ... --format="table(name,status)"            # table projection
gcloud ... --format="value(projectId)"              # bare values for scripts
gcloud ... --format=json | jq ...                   # pipe to jq
gcloud ... --filter="status=RUNNING AND zone~us-"   # =, :, ~, AND/OR/NOT
```

## Try it

Create (or select) a project, then run this end-to-end setup and prove it worked:

1. `gcloud auth login` and `gcloud auth application-default login`.
2. `gcloud config set project <your-project>` and set a default region/zone.
3. Enable the ML-relevant APIs with a single `gcloud services enable ...` command.
4. Create a budget with an alert at 50% and 90% of a small monthly amount (via the Console's Billing → Budgets, or the `gcloud billing budgets` group).
5. Run `gcloud config list` and `gcloud services list --enabled --format="value(config.name)"` and confirm your project, region, and the enabled APIs are exactly what you expect. Then open Cloud Shell and re-run `gcloud config list` there to see how a pre-authenticated environment differs from your local one.
