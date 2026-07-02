# 12 — Building an End-to-End ML System on Azure

Everything so far has been a component. This section wires them into one production ML system on Azure — the whole toolkit composing into a single, governed, observable, secure platform that ingests data, engineers features, trains and registers models, deploys them for real-time and batch inference (with a GenAI option), and monitors them in production. The point is to see how a dozen services fit together, where the boundaries are, and how infrastructure-as-code makes the whole thing reproducible. No single service is the "ML system"; the architecture is.

## The reference architecture

Read the flow left to right — data in, predictions out — with cross-cutting concerns (identity, network, IaC, cost) wrapping everything.

**1. Ingestion.** Two front doors feed the lake. **Azure Event Hubs** ingests streaming data (transactions, clickstream, telemetry) at high throughput, speaking the Kafka protocol so existing producers connect unchanged. **Azure Data Factory** handles scheduled batch ingestion from operational databases, SaaS APIs, and other clouds. Both land raw, immutable data in the **bronze** layer of the lake.

**2. Data lake and warehouse.** **ADLS Gen2** (Blob with hierarchical namespace) is the storage backbone, organized medallion-style: **bronze** (raw) → **silver** (cleaned, Parquet/Delta) → **gold** (curated features and datasets). A **Microsoft Fabric** lakehouse does the transformation on OneLake in open Delta format, and **mirroring** brings operational data from **Azure SQL** and **Cosmos DB** into the lake analytics-ready without brittle ETL.

**3. Feature pipeline and feature store.** Fabric/Spark jobs compute features from silver into **gold** feature tables. The offline features live in the lake for training; the low-latency subset needed at serving time is materialized into **Cosmos DB** as the **online feature store**, so an endpoint can fetch a user's features in single-digit milliseconds during a request.

**4. Training and registry.** An **Azure Machine Learning pipeline** of components — validate → feature-select → train → evaluate → register — runs on an autoscaling **compute cluster** (LowPriority, scale-to-zero; ND nodes with InfiniBand for distributed jobs), tracked with **MLflow**. The output is a **versioned model** in the **model registry** with full lineage back to the data and code. A shared **registry** promotes the winning model from the dev workspace to prod.

**5. CI/CD.** **Azure DevOps or GitHub Actions** is the automation spine. On a pull request it runs tests and evals; on merge to main it builds environment/serving images into **ACR**, triggers the Azure ML training pipeline, and — on approval — promotes the model and rolls out a new endpoint deployment. Infrastructure changes go through the same pipeline as Bicep/Terraform.

**6. Orchestration.** Responsibilities divide cleanly: **Data Factory** owns the data-movement DAG; **Azure ML pipelines** own the training/eval/register DAG; **Durable Functions** handle event-driven reactions; **Airflow** (managed) is the option for teams wanting one code-first scheduler across both. Keep each step owned by exactly one orchestrator.

**7. Deployment.** The registered model serves two ways. A **managed online endpoint** (autoscaled, blue-green/canary, inside the managed VNet) handles real-time requests, reading online features from Cosmos in the request path. A **batch endpoint** scores large datasets nightly on scale-to-zero clusters, writing results back to Azure SQL. A thin **HTTP Azure Function** (or API gateway) fronts both, adding auth, rate limiting, and logging.

**8. GenAI option.** In parallel, a **Microsoft Foundry** deployment serves foundation models. Documents from the lake are extracted with **Document Intelligence**, embedded, and indexed in **Azure AI Search** for **RAG**; a chat/agent deployment answers grounded questions through the same front door. The GenAI branch shares the identity, network, and monitoring backbone with the custom-ML branch.

**9. Monitoring.** **Azure Monitor** (with **Application Insights** and **Log Analytics**) collects endpoint latency, throughput, and errors and drives autoscale. **Azure ML model monitoring** compares live inference inputs against the training baseline to detect **data drift** and **prediction drift**, alerting when the world has shifted from what the model learned — the trigger to retrain, closing the loop back to step 4.

**Cross-cutting.** **Microsoft Entra ID** and a shared **user-assigned managed identity** authenticate every hop; **Key Vault** holds the rare human-set secret. A **VNet** with **private endpoints** keeps storage, registry, vault, workspace, endpoints, AI Search, and Foundry off the public internet. **Bicep/Terraform** provisions all of it; **Cost Management** tracks the spend.

## Infrastructure-as-code: provisioning the foundation

Everything above must be reproducible, so it is defined declaratively — **Bicep** (Azure-native) or **Terraform**. The pattern is one parameterized template per environment (dev/prod), deployed by CI/CD, so prod is a byte-for-byte redeploy of dev with different parameters. Here is the foundation of the platform in Bicep — the storage, identity, registry, vault, and ML workspace that everything else attaches to:

```bicep
// main.bicep — core ML platform foundation
param location string = resourceGroup().location
param env string = 'dev'
param prefix string = 'mlx'

// Shared managed identity — every compute/endpoint/function uses this
resource id 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${prefix}-${env}'
  location: location
}

// Data lake (Blob + hierarchical namespace = ADLS Gen2)
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${prefix}${env}'
  location: location
  sku: { name: 'Standard_ZRS' }
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true
    publicNetworkAccess: 'Disabled'         // reach only via private endpoint
    minimumTlsVersion: 'TLS1_2'
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: 'acr${prefix}${env}'
  location: location
  sku: { name: 'Premium' }                  // Premium enables private endpoints
  properties: { adminUserEnabled: false }
}

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${prefix}-${env}'
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true           // RBAC, not legacy access policies
  }
}

// Azure ML workspace with managed VNet isolation
resource ws 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: 'mlw-${prefix}-${env}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${id.id}': {} }
  }
  properties: {
    storageAccount: storage.id
    keyVault: kv.id
    containerRegistry: acr.id
    managedNetwork: { isolationMode: 'AllowOnlyApprovedOutbound' }
    publicNetworkAccess: 'Disabled'
  }
}

// Grant the shared identity data-plane access it needs (blob read/write)
resource blobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, id.id, 'blobcontrib')
  scope: storage
  properties: {
    // Storage Blob Data Contributor
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions',
      'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
    principalId: id.properties.principalId
    principalType: 'ServicePrincipal'
  }
}
```

The deploy itself is one command, but the practitioner sequence around it is what keeps prod safe. Before applying anything you run **what-if**, which diffs the template against live infrastructure and prints exactly what will be created, modified, or deleted — the single most valuable habit for avoiding an accidental teardown. Compile the Bicep to ARM JSON to catch template errors early, and scope the deployment to the right level: `deployment group` for resources inside a resource group, `deployment sub` for subscription-level things (the resource groups themselves, policy assignments, role assignments at sub scope):

```bash
# Install / update the Bicep CLI (bundled with recent az, but pin it explicitly in CI)
az bicep install
az bicep upgrade
az bicep version

# Lint + compile to ARM JSON — fails fast on template errors before any deploy
az bicep build --file main.bicep

# Preview the change set against live infra BEFORE deploying (read-only)
az deployment group what-if -g rg-mlx-dev \
  --template-file main.bicep --parameters env=dev prefix=mlx

# Deploy per environment; prod is the same template with different params
az deployment group create -g rg-mlx-dev \
  --template-file main.bicep --parameters env=dev prefix=mlx
# --what-if on create previews inline; --confirm-with-what-if forces a review gate
az deployment group create -g rg-mlx-dev \
  --template-file main.bicep --parameters env=prod prefix=mlx --confirm-with-what-if

# Subscription scope: create the resource group + sub-level policy/role assignments
az deployment sub create --location eastus2 \
  --template-file platform.bicep --parameters env=prod

# Inspect and validate deployments
az deployment group validate -g rg-mlx-dev --template-file main.bicep --parameters env=dev
az deployment group list -g rg-mlx-dev -o table
az deployment group show -g rg-mlx-dev -n main --query properties.outputs
```

Teams standardizing on **Terraform** get the same reproducibility with the `azurerm` provider — `terraform plan` is the what-if analog, `terraform apply` the deploy, and remote state in a locked storage account replaces ARM's deployment history. The choice is organizational (Bicep is Azure-native and needs no state file; Terraform is multi-cloud and has a mature module ecosystem); the discipline is identical. Streaming (Event Hubs), the online store (Cosmos DB), AI Search, the Foundry resource, and the private endpoints follow the same pattern as additional resources and modules. The discipline that matters: nothing production-critical is created by clicking in the portal — if it is not in the template, it does not exist.

Once the platform is live, **Azure Resource Graph** is how you query it at scale — a KQL-over-your-resources engine that answers "which storage accounts still allow public access?" or "every GPU VM across all subscriptions" in one call, invaluable for governance audits and drift detection:

```bash
# Find any storage account with public network access still enabled
az graph query -q "Resources
  | where type =~ 'microsoft.storage/storageaccounts'
  | where properties.publicNetworkAccess != 'Disabled'
  | project name, resourceGroup, subscriptionId"
```

## CI/CD: the automation spine in practice

The pipeline that ties IaC, training, and deployment together should authenticate to Azure with **no stored secrets**. The current standard is **OIDC federation**: GitHub Actions (or Azure DevOps) presents a short-lived workflow token to Entra ID, which exchanges it for an Azure access token against a **federated credential** you registered on an app/managed identity — so there is no client secret to rotate or leak. The `azure/login` action wires this up, and every subsequent `az` / `az ml` step runs as that identity:

```yaml
# .github/workflows/deploy.yml — keyless deploy via OIDC
permissions:
  id-token: write        # required for the workflow to request the OIDC token
  contents: read
jobs:
  ship:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: azure/login@v2                    # OIDC — no client secret
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
      - name: Provision infra
        run: az deployment group create -g rg-mlx-dev --template-file main.bicep --parameters env=dev
      - name: Train + register (Azure ML job)
        run: |
          az ml job create --file pipeline-job.yml -g rg-mlx-dev -w mlw-mlx-dev
      - name: Roll out canary deployment
        run: |
          az ml online-deployment create --file green-deployment.yml -g rg-mlx-dev -w mlw-mlx-dev
          az ml online-endpoint update -n fraud-endpoint --traffic "blue=90 green=10" -g rg-mlx-dev -w mlw-mlx-dev
```

The `az ml` verbs are the orchestration surface CI actually drives: `az ml job create --file pipeline-job.yml` kicks off the train→evaluate→register pipeline, `az ml model list`/`az ml model show` gate promotion on the new version's metrics, and the online-deployment commands from the deployment section perform the canary rollout — all the same YAML specs a data scientist ran locally, now executed by a keyless identity. The federated credential is itself declared in Bicep, so even the CI trust relationship is reproducible infrastructure.

## The lifecycle, end to end

Trace one change through the whole system. A data scientist opens a PR improving the feature pipeline. CI runs tests and evals against a frozen eval set. On merge, the pipeline builds images to ACR and triggers the Azure ML training pipeline, which reads versioned gold data, trains distributed on the GPU cluster, evaluates, and registers a new model version if it beats the baseline. A release approval promotes the model to the prod registry and creates a green online-endpoint deployment; traffic canaries from 10% to 100% while Azure Monitor watches latency and error rate, with instant rollback available via the traffic split. In production, model monitoring compares live inputs to the training baseline; when drift crosses a threshold, it alerts and can trigger retraining — back to the top. Every hop authenticates with the managed identity, stays on private endpoints, and reports cost to Cost Management. That closed loop, not any one service, is the ML system.

## Key takeaways

- A production ML system is a **composition** of a dozen Azure services — ingestion, lake/warehouse, feature store, training, registry, CI/CD, orchestration, deployment, GenAI, monitoring — not Azure ML alone.
- Data flows **bronze → silver → gold** in the lake; features split into an **offline** table (training) and an **online** Cosmos DB store (serving); models flow **train → register → promote → deploy**.
- **CI/CD (Azure DevOps / GitHub Actions)** is the spine that builds images, runs the training pipeline, and rolls out canary deployments; **orchestration** duties split cleanly across Data Factory, ML pipelines, and Durable Functions.
- The **GenAI branch** (Foundry + AI Search RAG) runs in parallel, sharing the identity/network/monitoring backbone with the custom-ML branch.
- Provision everything with **Bicep/Terraform** so environments are reproducible; wrap it all in **Entra ID identity, private-endpoint networking, and Cost Management** — the cross-cutting concerns that make it production-grade.

## CLI cheat-sheet

```bash
# --- Bicep tooling ---
az bicep install                    # install the Bicep CLI (pin in CI)
az bicep upgrade                    # update to latest
az bicep version
az bicep build --file main.bicep    # lint + compile Bicep -> ARM JSON

# --- Deploy (resource-group scope) ---
az deployment group what-if -g rg-mlx-dev --template-file main.bicep --parameters env=dev   # preview change set
az deployment group validate -g rg-mlx-dev --template-file main.bicep --parameters env=dev  # template check
az deployment group create -g rg-mlx-dev --template-file main.bicep --parameters env=dev prefix=mlx
az deployment group create -g rg-mlx-dev --template-file main.bicep --parameters env=prod --confirm-with-what-if
az deployment group list -g rg-mlx-dev -o table
az deployment group show -g rg-mlx-dev -n main --query properties.outputs

# --- Deploy (subscription scope: resource groups, policy/role assignments) ---
az deployment sub what-if --location eastus2 --template-file platform.bicep --parameters env=prod
az deployment sub create  --location eastus2 --template-file platform.bicep --parameters env=prod

# --- Resource Graph (query the live estate with KQL) ---
az graph query -q "Resources | where type =~ 'microsoft.machinelearningservices/workspaces' | project name, resourceGroup"
az graph query -q "Resources | summarize count() by type | order by count_ desc"

# --- Azure ML orchestration (what CI/CD drives) ---
az ml job create --file pipeline-job.yml -g rg-mlx-dev -w mlw-mlx-dev    # train -> eval -> register
az ml model list -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml model show -n fraud-detector --version 2 -g rg-mlx-dev -w mlw-mlx-dev
az ml online-deployment create --file green-deployment.yml -g rg-mlx-dev -w mlw-mlx-dev
az ml online-endpoint update -n fraud-endpoint --traffic "blue=90 green=10" -g rg-mlx-dev -w mlw-mlx-dev

# --- Keyless CI login (GitHub Actions / Azure DevOps via OIDC) ---
# azure/login@v2 with client-id/tenant-id/subscription-id + a federated credential — no client secret
```

## Try it

Write a Bicep (or Terraform) template that provisions the platform foundation: a shared user-assigned managed identity, an HNS-enabled storage account with public access disabled, a Premium ACR, an RBAC-mode Key Vault, and an Azure ML workspace with managed-VNet isolation, plus a role assignment granting the identity Storage Blob Data Contributor. Deploy it to a `dev` resource group, then deploy the *same* template with `env=prod` parameters to a second group — proving reproducibility. Finally, draw the full reference architecture on one page and annotate each arrow with the service and the auth method (managed identity vs. Key Vault secret) it uses. If you can build and explain that diagram, you can build the system.
