# 01 — Azure Overview, Portal, and CLI

Azure is Microsoft's public cloud: a global fleet of data centers you rent compute, storage, networking, and managed services from, billed by consumption. For an ML engineer, Azure is the substrate under everything you will build — the GPU that trains your model, the object store that holds your dataset, the endpoint that serves predictions, and the identity system that decides who is allowed to touch any of it. Before you can be productive with Azure Machine Learning or Azure AI Foundry, you need a firm grip on the primitives that every other service sits on: the global hierarchy, resource organization, and the two ways you interact with the platform — the portal and the command line.

## The global footprint: geographies, regions, and availability zones

Azure is organized geographically. A **geography** is a discrete market (for example, United States, Europe, or India) that typically respects data-residency boundaries. Inside a geography are **regions** — a region is a set of data centers deployed within a latency-defined perimeter, connected by a low-latency network. Examples you will use constantly: `eastus`, `eastus2`, `westus3`, `westeurope`, `northeurope`, `swedencentral`, and `australiaeast`.

Region choice matters more for ML than for most workloads, for three reasons. First, **GPU availability is uneven** — the newest accelerators (NVIDIA H100, H200, GB200) land in a handful of regions first, and quota for them is region-scoped and often constrained. Second, **data gravity** — you want compute in the same region as your training data to avoid cross-region egress charges and latency. Third, **AI service availability** — specific foundation-model deployments in Azure AI Foundry are only offered in certain regions.

Within a region, **availability zones** are physically separate locations, each with independent power, cooling, and networking. Zone-redundant services survive the loss of a single data center. For production inference endpoints you care about zones; for a one-off training job you usually do not.

## Subscriptions, management groups, and tenants

Your account lives in a **tenant** — a dedicated instance of Microsoft Entra ID (the identity service, covered later) that represents your organization. Under a tenant you have one or more **subscriptions**. A subscription is a billing and quota boundary: costs roll up to it, quotas (like "how many H100 cores can I spin up") are enforced against it, and it is the unit most policies attach to.

**Management groups** sit above subscriptions and let large organizations apply governance (policy, access control) to many subscriptions at once. A common pattern is a management group hierarchy like `Root → Platform / Landing Zones → Prod / Dev`, with a separate subscription per environment so that a runaway training experiment in Dev can never blow the Prod budget.

## Resource groups and resources

Everything you deploy is a **resource** — a VM, a storage account, an Azure Machine Learning workspace, an endpoint. Every resource lives in exactly one **resource group**, which is a logical container scoped to a single subscription. Resource groups are the natural unit of lifecycle: you deploy, tag, apply access control to, and delete resources as a group. A clean habit is one resource group per project-environment pair, for example `rg-fraud-detection-dev` and `rg-fraud-detection-prod`, so tearing down an experiment is a single `az group delete`.

A few rules worth internalizing: a resource group has a location (used for storing metadata) but its resources can live in other regions; deleting a resource group deletes everything in it; and resources have globally or regionally unique names depending on type (a storage account name must be globally unique and DNS-safe, an ML workspace name only unique within its resource group).

Day to day you spend more time listing, inspecting, and tagging resource groups than creating them. The `az group` command group covers the whole lifecycle, and every one of these commands respects the `--query` and `--output` conventions you will lean on constantly:

```bash
# List all resource groups, projected to name/location/state as a table
az group list --query "[].{Name:name, Location:location, State:properties.provisioningState}" -o table

# Show one group, and check whether it exists (returns true/false, no error)
az group show --name rg-mlx-dev -o jsonc
az group exists --name rg-mlx-dev

# Re-tag an existing group in place (merges by default; --tags "" clears them)
az group update --name rg-mlx-dev --tags project=mlx env=dev owner=humza costcenter=ml

# Everything inside a group, or filtered to a resource type across the subscription
az resource list --resource-group rg-mlx-dev -o table
az resource list --resource-type Microsoft.MachineLearningServices/workspaces -o table

# Delete the whole group without blocking your shell (--no-wait) and without a prompt (--yes)
az group delete --name rg-mlx-dev --yes --no-wait
```

## The Azure portal

The **portal** (portal.azure.com) is the web UI. It is where you learn what a service looks like, inspect a misbehaving deployment, read metrics, and click through wizards when you are still forming a mental model. It is excellent for discovery and terrible for reproducibility — anything you build by clicking is hard to recreate, review, or hand to a teammate. Use it to explore and to debug; do not use it as your source of truth for infrastructure.

The single most useful portal feature for a CLI-oriented engineer is the **"Download a template for automation"** link that appears on the review step of most creation wizards. Fill in a form, and the portal hands you the equivalent Azure Resource Manager (ARM) template, so you can learn the declarative shape of any resource.

## The Azure CLI (`az`)

The `az` CLI is your daily driver. It is cross-platform, scriptable, and maps cleanly onto the underlying REST APIs. Install it, then authenticate:

```bash
# Sign in (opens a browser); use --use-device-code on headless machines
az login

# List subscriptions you can see, and pin the one you want as default
az account list --output table
az account set --subscription "My ML Subscription"

# Confirm who you are and where you're pointed
az account show --output table
```

`az login` opens a browser by default, but real ML work rarely happens at an interactive desktop. On a headless training box use device-code flow; in CI/CD authenticate as a service principal or, better, a managed identity; and from an Azure VM or compute cluster the local managed identity needs no secret at all:

```bash
# Headless / SSH sessions: prints a code to enter at microsoft.com/devicelogin
az login --use-device-code

# CI/CD as a service principal (secret or cert) — pin the tenant explicitly
az login --service-principal -u "$AZURE_CLIENT_ID" -p "$AZURE_CLIENT_SECRET" --tenant "$AZURE_TENANT_ID"

# Running on an Azure resource: authenticate as its managed identity, no secret
az login --identity                              # system-assigned
az login --identity --username "$UAMI_CLIENT_ID" # a specific user-assigned identity
```

Your Azure CLI is itself versioned software, and the `az ml` and other extensions ship on their own cadence. Keep both current, because Microsoft deprecates old CLI behaviors and new service features often require a recent build:

```bash
# Show core CLI + every installed extension version
az version -o jsonc

# Upgrade the CLI (and, with the flag, all extensions) in place
az upgrade --all --yes

# Manage extensions explicitly
az extension list -o table
az extension update --name ml
az extension remove --name ml
```

Set defaults so you stop repeating yourself, then create a resource group:

```bash
# Persist a default location and resource group for this shell/config
az configure --defaults location=eastus2 group=rg-mlx-dev

# Set the default output format once so you stop typing --output everywhere
az configure --defaults output=table

# Inspect or clear a default later
az configure --list-defaults -o table
az configure --defaults group=""   # empty value removes a default

# Create the resource group (tags are free and pay off later for cost tracking)
az group create \
  --name rg-mlx-dev \
  --location eastus2 \
  --tags project=mlx env=dev owner=humza
```

Discoverability is the CLI's superpower. Every command is self-documenting, and `--output table` or `--query` (a JMESPath expression) lets you slice the JSON:

```bash
# Explore any command tree
az ml --help
az vm --help

# List GPU-capable VM sizes available in your region, projected to useful columns
az vm list-sizes --location eastus2 \
  --query "[?contains(name,'Standard_NC')].{Name:name, vCPUs:numberOfCores, MemMB:memoryInMb}" \
  --output table

# Check your regional core quota for a GPU family before you try to deploy
az vm list-usage --location eastus2 \
  --query "[?contains(localName,'NC')].{Family:localName, Used:currentValue, Limit:limit}" \
  --output table
```

The `--query` flag is worth investing an afternoon in — it is [JMESPath](https://jmespath.org), a full query language over the JSON the API returns, and it turns the CLI from a printer into a filter. A few patterns you will reuse everywhere: `[?field=='x']` filters an array, `{Alias:path}` reshapes and renames fields, `[0]` and `| [0]` pick elements, and `-o tsv` strips quotes so a single value drops straight into a shell variable:

```bash
# Pull one scalar into a variable (the idiom behind every $ID capture in this course)
WS_ID=$(az ml workspace show -n mlw-mlx-dev -g rg-mlx-dev --query id -o tsv)

# Filter + reshape: workspaces missing an 'env' tag, across the subscription
az resource list --resource-type Microsoft.MachineLearningServices/workspaces \
  --query "[?tags.env==null].{Name:name, RG:resourceGroup}" -o table

# Output formats: table (humans), json/jsonc (default/colorized), tsv (scripts), yaml
az account show --output yaml
```

Some Azure capabilities are gated behind **resource provider** and **feature** registration on your subscription — a provider must be registered before you can create its resource type, and preview features are opt-in. If a deployment fails with `MissingSubscriptionRegistration`, this is why:

```bash
# Register the providers an ML platform needs (idempotent; safe to re-run)
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.ContainerRegistry
az provider show --namespace Microsoft.MachineLearningServices --query registrationState -o tsv

# Opt into (and check) a preview feature, then re-register the provider to apply it
az feature register --namespace Microsoft.MachineLearningServices --name <feature-name>
az feature show --namespace Microsoft.MachineLearningServices --name <feature-name> --query properties.state -o tsv
```

For Azure Machine Learning specifically, the `az ml` command group ships as an **extension**. Install it once:

```bash
# Install the ML extension (v2), then verify
az extension add --name ml
az ml -h
```

When your automation grows beyond a handful of commands, you graduate from imperative `az` scripts to declarative infrastructure-as-code — **Bicep** (Azure's native, terse ARM language) or **Terraform**. The mental model is the same primitives; the difference is that Bicep and Terraform describe desired state and reconcile it, which is what you want for anything that must be reproducible across environments. The CLI deploys Bicep and ARM templates directly, scoped to a resource group, subscription, or management group. Always run a `what-if` first — it prints the diff between your template and reality before anything changes:

```bash
# Preview the changes a Bicep deployment would make (no changes applied)
az deployment group what-if \
  --resource-group rg-mlx-dev --template-file main.bicep --parameters env=dev

# Apply it; parameters can come inline or from a .bicepparam / JSON file
az deployment group create \
  --resource-group rg-mlx-dev --name mlx-platform \
  --template-file main.bicep --parameters @main.parameters.json

# Inspect and clean up deployment history (deployments accumulate on the RG)
az deployment group list --resource-group rg-mlx-dev -o table
```

## How the pieces fit for an ML project

A minimal ML footprint on Azure looks like this: a subscription holds a resource group; the resource group holds an Azure Machine Learning workspace, a storage account (for datasets and model artifacts), a container registry (for environment images), and a Key Vault (for secrets). The workspace ties them together. You provision this once with the CLI or Bicep, then spend your time inside the workspace training and deploying. Every later section builds on exactly these primitives.

## Key takeaways

- Azure's hierarchy is **tenant → management group → subscription → resource group → resource**; subscriptions are the billing and quota boundary, resource groups are the lifecycle unit.
- **Region choice is an ML decision**, driven by GPU quota, data locality, and AI-service availability — not an afterthought.
- The **portal is for exploring and debugging**; the **`az` CLI (and Bicep/Terraform)** are for anything that must be reproducible.
- Tag everything (`project`, `env`, `owner`) from day one — it is the foundation of later cost and governance work.
- Check **quota** (`az vm list-usage`) before you design a training run around a GPU family you may not be able to get.

## CLI cheat-sheet

```bash
# --- auth & context ---
az login                                        # interactive (browser)
az login --use-device-code                      # headless / SSH
az login --service-principal -u "$ID" -p "$SECRET" --tenant "$TENANT"
az login --identity                             # managed identity on an Azure resource
az account list -o table                        # subscriptions you can see
az account set --subscription "My ML Subscription"
az account show -o table                         # current context
az logout

# --- CLI & extensions ---
az version -o jsonc                              # core + extension versions
az upgrade --all --yes                           # upgrade CLI + extensions
az extension add --name ml                       # install az ml (CLI v2)
az extension list -o table
az extension update --name ml
az configure --defaults location=eastus2 group=rg-mlx-dev output=table
az configure --list-defaults -o table

# --- resource groups ---
az group create -n rg-mlx-dev -l eastus2 --tags project=mlx env=dev owner=humza
az group list --query "[].{Name:name, Location:location}" -o table
az group show -n rg-mlx-dev -o jsonc
az group exists -n rg-mlx-dev
az group update -n rg-mlx-dev --tags project=mlx env=dev
az group delete -n rg-mlx-dev --yes --no-wait

# --- resources & queries ---
az resource list -g rg-mlx-dev -o table
az resource list --resource-type Microsoft.MachineLearningServices/workspaces -o table
az vm list-sizes -l eastus2 --query "[?contains(name,'Standard_NC')].name" -o tsv
az vm list-usage -l eastus2 --query "[?contains(localName,'NC')].{F:localName,Used:currentValue,Max:limit}" -o table
WS_ID=$(az ml workspace show -n mlw-mlx-dev -g rg-mlx-dev --query id -o tsv)

# --- providers & features ---
az provider register --namespace Microsoft.MachineLearningServices
az provider show --namespace Microsoft.MachineLearningServices --query registrationState -o tsv
az feature register --namespace Microsoft.MachineLearningServices --name <feature>

# --- infrastructure as code (Bicep / ARM) ---
az deployment group what-if -g rg-mlx-dev --template-file main.bicep
az deployment group create -g rg-mlx-dev -n mlx --template-file main.bicep --parameters @main.parameters.json
az deployment group list -g rg-mlx-dev -o table
```

## Try it

Install the `az` CLI, run `az login`, and set your default subscription. Then, in one shell session: create a resource group named `rg-azureml-101` in `eastus2` with tags `project=learning` and `env=sandbox`; list the `Standard_NC` VM sizes available in that region and your current GPU core quota; and finally delete the resource group with `az group delete --name rg-azureml-101 --yes --no-wait`. You have now provisioned and torn down real Azure infrastructure entirely from the command line — the loop you will repeat thousands of times.
