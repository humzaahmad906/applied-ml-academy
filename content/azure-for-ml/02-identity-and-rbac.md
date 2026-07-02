# 02 — Identity and RBAC (Entra ID)

Every action in Azure is an authenticated, authorized API call. Before a training job can read a dataset, before an endpoint can pull a model, before a pipeline can write to the data lake, the platform asks two questions: *who are you* (authentication) and *are you allowed to do this* (authorization). Get identity right and the rest of your system composes cleanly and securely; get it wrong and you end up with hard-coded keys leaking through code, over-privileged jobs, and a security review that blocks your launch. In an end-to-end ML solution — where a dozen services (storage, registry, endpoints, Key Vault, Functions, Fabric) all talk to each other — identity is the connective tissue that lets those services trust one another without secrets.

## Microsoft Entra ID: the identity plane

**Microsoft Entra ID** is Azure's cloud identity and access management service. It was renamed from Azure Active Directory (Azure AD); you will still see "Azure AD" in old blog posts and some SDK class names, but the service, the portal blade, and current documentation all say Entra ID. It holds the **identities** in your tenant: human users, groups, and — critically for ML automation — non-human principals.

The identity types you will actually work with:

- **Users** — humans, backed by the org's directory, ideally protected by multi-factor authentication and conditional access.
- **Groups** — collections of users (and other groups). Assign access to groups, not individuals, so onboarding and offboarding is a membership change rather than an access-control rewrite.
- **Service principals** — an identity for an application or automation. When you register an app or a CI/CD pipeline needs to deploy, it acts as a service principal. Historically these authenticated with a client secret or certificate.
- **Managed identities** — a service principal that Azure creates and rotates for you, with no secret you ever see or store. This is the single most important identity concept for ML on Azure.

## Managed identities: the end of hard-coded credentials

A **managed identity** gives an Azure resource — a VM, an Azure Machine Learning compute cluster, a Function app, an endpoint — its own identity in Entra ID, with credentials that Azure provisions and rotates automatically. The resource requests a token from a local endpoint at runtime and uses it to call other Azure services. You never see, store, or rotate a secret. This is how you eliminate the classic anti-pattern of a storage account key pasted into a training script or a `.env` file.

There are two flavors:

- **System-assigned** — tied one-to-one to a single resource; created and deleted with it. Good for a resource that only it needs to authenticate as.
- **User-assigned** — a standalone resource you create once and attach to *many* other resources. This is the right choice for an ML platform, because your training compute, batch endpoints, and Functions can all share one identity with a consistent, auditable set of permissions.

```bash
# Create a user-assigned managed identity for the whole ML platform
az identity create \
  --name id-mlplatform \
  --resource-group rg-mlx-dev

# Capture its principal ID and client ID for later role assignments and code
PRINCIPAL_ID=$(az identity show -n id-mlplatform -g rg-mlx-dev --query principalId -o tsv)
CLIENT_ID=$(az identity show -n id-mlplatform -g rg-mlx-dev --query clientId -o tsv)
```

The `principalId` is the object the RBAC system grants roles to; the `clientId` is what your code passes to `DefaultAzureCredential` when a resource has *more than one* user-assigned identity attached and you must disambiguate. Managing user-assigned identities is the usual list/show/delete lifecycle:

```bash
# List every user-assigned identity in a group, and delete one you no longer need
az identity list -g rg-mlx-dev -o table
az identity delete -n id-mlplatform -g rg-mlx-dev
```

A **system-assigned** identity is never created on its own — you enable it *on* a resource, so it lives and dies with that resource. You will meet these flags again in later modules (`--assign-identity` on a VM, `--identity-type SystemAssigned` on a workspace); the point here is that system-assigned suits a single-purpose resource, while the standalone user-assigned identity above is what an ML *platform* shares across many.

### Federated credentials: passwordless CI/CD

The one place automation still tends to smuggle in a secret is CI/CD — a GitHub Actions pipeline that deploys your workspace. The modern fix is a **federated identity credential**: you tell a user-assigned managed identity to trust tokens issued by GitHub's OIDC provider for a specific repo and branch, so the pipeline exchanges a short-lived GitHub token for an Azure token with no stored client secret at all.

```bash
# Trust GitHub Actions on the main branch of one repo — no secret is created
az identity federated-credential create \
  --name gha-main --identity-name id-mlplatform --resource-group rg-mlx-dev \
  --issuer "https://token.actions.githubusercontent.com" \
  --subject "repo:my-org/ml-platform:ref:refs/heads/main" \
  --audiences "api://AzureADTokenExchange"

az identity federated-credential list --identity-name id-mlplatform -g rg-mlx-dev -o table
```

The `--subject` string must match exactly what GitHub puts in the token — swap `ref:refs/heads/main` for `environment:production` or `pull_request` to scope the trust differently.

In Python, the SDK "just works" because it uses a credential that tries managed identity, then developer login, transparently:

```python
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# Locally this uses your `az login`; on an Azure compute it uses the
# managed identity. No keys, no connection strings, no code change.
credential = DefaultAzureCredential()
blob = BlobServiceClient(
    account_url="https://stmlxdata.blob.core.windows.net",
    credential=credential,
)
```

`DefaultAzureCredential` is the pattern to standardize on across every service in your solution: the same three lines authenticate your laptop during development and your production endpoint at runtime.

## Azure RBAC: roles, scopes, and assignments

Authorization is **role-based access control (RBAC)**. An **assignment** is a triple: a **security principal** (who) gets a **role definition** (what actions) at a **scope** (where). Scope is hierarchical — management group, subscription, resource group, or a single resource — and permissions inherit downward. An assignment at the resource-group level applies to every resource inside it.

A role definition is a set of allowed `Actions` (control-plane operations, like "create a VM") and `DataActions` (data-plane operations, like "read a blob"). This control-plane / data-plane split trips people up constantly: the built-in **Owner** and **Contributor** roles let you *manage* a storage account but do **not** grant permission to read the blobs inside it. To read data you need a data-plane role such as **Storage Blob Data Reader**.

Roles you will use in an ML solution:

- **Storage Blob Data Reader / Contributor** — read or read-write datasets and artifacts in Blob and Data Lake Gen2.
- **AzureML Data Scientist** — full authoring inside an Azure Machine Learning workspace (jobs, endpoints, models) without control over the workspace resource itself.
- **AzureML Compute Operator** — manage compute in a workspace.
- **Key Vault Secrets User / Officer** — read secrets, or manage them.
- **AcrPull / AcrPush** — pull or push container images (your endpoints need AcrPull).
- **Cognitive Services OpenAI User** — call model deployments in Azure AI Foundry / Azure OpenAI.

```bash
# Grant the platform identity data-plane read on a storage account, scoped tightly
STORAGE_ID=$(az storage account show -n stmlxdata -g rg-mlx-dev --query id -o tsv)

az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Reader" \
  --scope "$STORAGE_ID"

# Let the same identity pull container images for serving
ACR_ID=$(az acr show -n acrmlx -g rg-mlx-dev --query id -o tsv)
az role assignment create --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "AcrPull" --scope "$ACR_ID"
```

A word on the `--assignee` flags, because this is a common source of flaky scripts. The convenient `--assignee <name-or-appId>` form makes the CLI do a directory lookup to resolve the object ID, which fails intermittently for brand-new principals that have not replicated yet, and can be ambiguous. Prefer the explicit pair `--assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal` (as above) for managed identities and service principals — it skips the lookup and is deterministic. Use plain `--assignee user@contoso.com` for human users where the lookup is reliable.

Auditing and cleanup use the same command group. `--scope` narrows the query, `--all` walks the whole subscription, and `--include-inherited` shows assignments a resource picks up from its parent scopes:

```bash
# Who has what on this storage account (including inherited RG/subscription grants)?
az role assignment list --scope "$STORAGE_ID" --include-inherited -o table

# Everything one identity can do, subscription-wide
az role assignment list --assignee "$PRINCIPAL_ID" --all -o table

# Remove a specific grant (same triple you created it with)
az role assignment delete --assignee-object-id "$PRINCIPAL_ID" \
  --role "Storage Blob Data Reader" --scope "$STORAGE_ID"
```

You do not have to memorize role names. Inspect built-in definitions to find the least-privileged one that covers your `Actions`/`DataActions`, and when nothing fits, author a **custom role** from a JSON definition:

```bash
# Find candidate roles and read exactly what one grants
az role definition list --query "[?contains(roleName,'AzureML')].roleName" -o tsv
az role definition list --name "Storage Blob Data Reader" --query "[0].permissions" -o jsonc

# Create a narrow custom role from a JSON file (AssignableScopes must list your subscription)
az role definition create --role-definition @ml-dataset-reader.json
az role definition update --role-definition @ml-dataset-reader.json
```

When you genuinely cannot use a managed identity — say a tool running entirely outside Azure — `az ad sp create-for-rbac` mints a service principal *with a secret*. Reach for it last: it hands you a password you now own, must store securely, and must rotate. A federated credential (above) is almost always the better answer.

```bash
# Last resort: a service principal WITH a client secret, scoped to one RG
az ad sp create-for-rbac --name sp-legacy-tool \
  --role Contributor --scopes "/subscriptions/$SUB_ID/resourceGroups/rg-mlx-dev"
```

## Least privilege as a design principle

A newly created managed identity starts with **zero** permissions and inherits nothing until you assign a role. Keep it that way as long as possible and grant the narrowest role at the tightest scope that lets the job succeed. Concretely, for ML work: give a training identity read on the *dataset* container, not Contributor on the *subscription*; give an inference endpoint AcrPull on the *one* registry it serves from, not on all of them; give a scoring Function `Cognitive Services OpenAI User`, not `Contributor`, on the AI resource.

Assign roles to **groups** for humans and to **user-assigned managed identities** for automation, both at **resource-group scope** where practical, so the number of assignments stays small and auditable. When someone needs elevated rights briefly (say, an on-call engineer debugging prod), prefer **Privileged Identity Management (PIM)** for just-in-time, time-boxed elevation over a standing assignment — the engineer *activates* the role for a few hours with justification and approval, and it lapses automatically, so there is no permanent Owner sitting on production. PIM is an Entra ID P2 feature managed through the portal and the Microsoft Graph / `az rest` APIs rather than a first-class `az role` verb, but it consumes the same role definitions you list above.

To audit what actually accumulated, list assignments per identity or per scope periodically and prune anything a job no longer uses — stale grants are the quiet way least privilege erodes.

## Key Vault: the secret and key store

Some secrets are unavoidable — a third-party API token, a database password for a legacy system, a signing key. **Azure Key Vault** is the managed store for secrets, keys, and certificates. It integrates with managed identities so an application fetches a secret at runtime with its own identity and no bootstrapping credential.

Key Vault has the same control-plane/data-plane split. Modern vaults use **Azure RBAC** for data-plane access, and as of the current API version RBAC is the default for newly created vaults; the legacy per-vault "access policies" model is discouraged because it lacks PIM support and fine-grained auditing. Grant **Key Vault Secrets User** to the identities that read secrets and **Key Vault Secrets Officer** only to the small set that manages them.

```bash
# Create an RBAC-mode vault and store a secret
az keyvault create -n kv-mlx-dev -g rg-mlx-dev --enable-rbac-authorization true
az keyvault secret set --vault-name kv-mlx-dev --name external-api-token --value "s3cr3t"

# Let the platform identity read (not manage) secrets
KV_ID=$(az keyvault show -n kv-mlx-dev -g rg-mlx-dev --query id -o tsv)
az role assignment create --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" --scope "$KV_ID"

# List / read / delete secrets — data-plane, so --auth-mode login uses your RBAC
az keyvault secret list --vault-name kv-mlx-dev -o table
az keyvault secret show --vault-name kv-mlx-dev --name external-api-token --query value -o tsv
az keyvault secret delete --vault-name kv-mlx-dev --name external-api-token
```

The Key Vault RBAC roles map to what each principal actually does: **Key Vault Secrets User** (read secret values) for the workloads that consume them, **Key Vault Secrets Officer** (full secret management) for the small operator group, and the analogous **Key Vault Crypto User / Officer** and **Certificates Officer** for key and certificate operations. Grant these at the vault scope, or — for the tightest control — scope a Secrets User role to an *individual secret* so an identity can read exactly the one credential it needs and nothing else.

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

client = SecretClient("https://kv-mlx-dev.vault.azure.net", DefaultAzureCredential())
token = client.get_secret("external-api-token").value  # no secret bootstrapping needed
```

An Azure Machine Learning workspace can be attached to a Key Vault so that connection strings and datastore credentials it manages live there rather than in your code. In the end-to-end solution, Key Vault is the one place a human-set secret enters the system; everything internal uses managed identity.

## Key takeaways

- **Entra ID** (formerly Azure AD) is the identity plane; the identities that matter for ML automation are **managed identities** — service principals with no secret you ever handle.
- Prefer a **user-assigned managed identity** shared across your ML platform's compute, endpoints, and Functions, and authenticate everywhere with `DefaultAzureCredential`.
- **RBAC** is (principal, role, scope). Remember the **control-plane vs data-plane** split: Owner/Contributor does *not* grant blob reads — you need a `Data` role.
- Practice **least privilege**: narrowest role, tightest scope, assign to groups and shared identities, use **PIM** for temporary elevation.
- **Key Vault** (RBAC mode) holds the rare human-set secrets; managed identities read them at runtime so nothing sensitive lands in code.

## CLI cheat-sheet

```bash
# --- managed identities ---
az identity create -n id-mlplatform -g rg-mlx-dev
az identity show -n id-mlplatform -g rg-mlx-dev --query principalId -o tsv
az identity show -n id-mlplatform -g rg-mlx-dev --query clientId -o tsv
az identity list -g rg-mlx-dev -o table
az identity delete -n id-mlplatform -g rg-mlx-dev

# --- federated credentials (passwordless CI/CD) ---
az identity federated-credential create --name gha-main --identity-name id-mlplatform -g rg-mlx-dev \
  --issuer "https://token.actions.githubusercontent.com" \
  --subject "repo:my-org/ml-platform:ref:refs/heads/main" \
  --audiences "api://AzureADTokenExchange"
az identity federated-credential list --identity-name id-mlplatform -g rg-mlx-dev -o table

# --- role assignments (prefer object-id form for MIs/SPs) ---
az role assignment create --assignee-object-id "$PID" --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Reader" --scope "$STORAGE_ID"
az role assignment list --assignee "$PID" --all -o table
az role assignment list --scope "$STORAGE_ID" --include-inherited -o table
az role assignment delete --assignee-object-id "$PID" --role "AcrPull" --scope "$ACR_ID"

# --- role definitions & custom roles ---
az role definition list --query "[?contains(roleName,'AzureML')].roleName" -o tsv
az role definition list --name "Storage Blob Data Reader" --query "[0].permissions" -o jsonc
az role definition create --role-definition @ml-dataset-reader.json
az role definition update --role-definition @ml-dataset-reader.json

# --- service principal with secret (last resort) ---
az ad sp create-for-rbac --name sp-legacy-tool --role Contributor \
  --scopes "/subscriptions/$SUB_ID/resourceGroups/rg-mlx-dev"

# --- Key Vault (RBAC mode) ---
az keyvault create -n kv-mlx-dev -g rg-mlx-dev --enable-rbac-authorization true
az keyvault secret set --vault-name kv-mlx-dev --name external-api-token --value "s3cr3t"
az keyvault secret show --vault-name kv-mlx-dev --name external-api-token --query value -o tsv
az keyvault secret list --vault-name kv-mlx-dev -o table
# grant Key Vault Secrets User / Officer via az role assignment create against $KV_ID
```

## Try it

Create a user-assigned managed identity `id-mlplatform` and a storage account. Assign the identity **Storage Blob Data Reader** scoped to that storage account, then try to assign it **only** Reader on the resource group and observe that it can *see* the account but not *read* the blobs — proving the control-plane/data-plane distinction to yourself. Finally, create an RBAC-mode Key Vault, store a secret, grant the identity `Key Vault Secrets User`, and confirm you did all of this without ever copying an access key into a file.
