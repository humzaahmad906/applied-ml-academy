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

Set defaults so you stop repeating yourself, then create a resource group:

```bash
# Persist a default location and resource group for this shell/config
az configure --defaults location=eastus2 group=rg-mlx-dev

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

For Azure Machine Learning specifically, the `az ml` command group ships as an **extension**. Install it once:

```bash
# Install the ML extension (v2), then verify
az extension add --name ml
az ml -h
```

When your automation grows beyond a handful of commands, you graduate from imperative `az` scripts to declarative infrastructure-as-code — **Bicep** (Azure's native, terse ARM language) or **Terraform**. The mental model is the same primitives; the difference is that Bicep and Terraform describe desired state and reconcile it, which is what you want for anything that must be reproducible across environments.

## How the pieces fit for an ML project

A minimal ML footprint on Azure looks like this: a subscription holds a resource group; the resource group holds an Azure Machine Learning workspace, a storage account (for datasets and model artifacts), a container registry (for environment images), and a Key Vault (for secrets). The workspace ties them together. You provision this once with the CLI or Bicep, then spend your time inside the workspace training and deploying. Every later section builds on exactly these primitives.

## Key takeaways

- Azure's hierarchy is **tenant → management group → subscription → resource group → resource**; subscriptions are the billing and quota boundary, resource groups are the lifecycle unit.
- **Region choice is an ML decision**, driven by GPU quota, data locality, and AI-service availability — not an afterthought.
- The **portal is for exploring and debugging**; the **`az` CLI (and Bicep/Terraform)** are for anything that must be reproducible.
- Tag everything (`project`, `env`, `owner`) from day one — it is the foundation of later cost and governance work.
- Check **quota** (`az vm list-usage`) before you design a training run around a GPU family you may not be able to get.

## Try it

Install the `az` CLI, run `az login`, and set your default subscription. Then, in one shell session: create a resource group named `rg-azureml-101` in `eastus2` with tags `project=learning` and `env=sandbox`; list the `Standard_NC` VM sizes available in that region and your current GPU core quota; and finally delete the resource group with `az group delete --name rg-azureml-101 --yes --no-wait`. You have now provisioned and torn down real Azure infrastructure entirely from the command line — the loop you will repeat thousands of times.
