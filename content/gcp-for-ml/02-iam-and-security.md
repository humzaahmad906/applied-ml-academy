# 02 — IAM and Security

Identity and Access Management (IAM) is the layer that decides *who* can do *what* to *which* resource in Google Cloud. For an ML engineer this is not bureaucratic overhead — it is the difference between a training job that can read your dataset and write your model, and one that silently fails with a permission error, or worse, a pipeline running with god-mode credentials that a security review will reject. Getting IAM right early makes every later module smoother.

## The three-part model: who, what, which

Every access decision in Google Cloud is an **IAM policy** attached to a **resource** (a project, a bucket, a Vertex AI endpoint). A policy is a list of **bindings**, and each binding ties:

- a **principal** (the "who") — a Google account, a group, a **service account**, a domain, or a federated identity,
- to a **role** (the "what") — a named bundle of granular permissions,
- optionally gated by an **IAM condition** (the "when/which") — an expression like "only for resources whose name starts with `dev-`" or "only until a certain date."

Principals used to be called "members." A binding reads: *principal X has role Y on resource Z*. Policies are inherited down the hierarchy — a role granted at the project level applies to every resource in that project.

## Roles: basic, predefined, and custom

There are three tiers of roles, and choosing the right tier is the core skill:

- **Basic roles** — `roles/owner`, `roles/editor`, `roles/viewer`. These are coarse, legacy, and dangerous: `editor` can modify almost everything in a project. Avoid them for anything beyond a personal sandbox.
- **Predefined roles** — hundreds of curated, service-specific roles that follow least privilege. Examples an ML engineer uses constantly: `roles/aiplatform.user` (use Vertex AI), `roles/storage.objectViewer` and `roles/storage.objectAdmin` (read vs read-write on bucket objects), `roles/bigquery.dataViewer` and `roles/bigquery.jobUser` (read data vs run queries), `roles/artifactregistry.writer` (push images). Prefer these.
- **Custom roles** — you assemble an exact permission set when no predefined role fits. Powerful but higher maintenance; reach for them only when a predefined role is genuinely too broad or too narrow.

```bash
# Grant a data scientist the ability to use Vertex AI in a project
gcloud projects add-iam-policy-binding myco-fraud-dev \
  --member="user:alice@example.com" \
  --role="roles/aiplatform.user"

# Prefer groups over individual users for maintainability
gcloud projects add-iam-policy-binding myco-fraud-dev \
  --member="group:ml-team@example.com" \
  --role="roles/bigquery.dataViewer"

# Inspect who has access to a project
gcloud projects get-iam-policy myco-fraud-dev
```

**Custom roles** are managed with their own verbs. Define one from a YAML file (or inline flags), then iterate on it as your permission needs shift:

```bash
# Create a custom role scoped to a project (or use --organization for org-wide)
gcloud iam roles create fraudModelDeployer \
  --project=myco-fraud-prod \
  --title="Fraud model deployer" \
  --permissions=aiplatform.endpoints.deploy,aiplatform.models.get \
  --stage=GA

# Inspect and enumerate roles
gcloud iam roles describe fraudModelDeployer --project=myco-fraud-prod
gcloud iam roles describe roles/aiplatform.user            # inspect a predefined role
gcloud iam roles list --project=myco-fraud-prod            # your custom roles
gcloud iam roles list --filter="name:aiplatform" --show-details  # predefined, with permissions

# Update by adding/removing permissions
gcloud iam roles update fraudModelDeployer --project=myco-fraud-prod \
  --add-permissions=aiplatform.endpoints.predict
```

`--show-details` on `list` is how you discover exactly which permissions a predefined role bundles before deciding whether it fits — the fast way to answer "is `roles/aiplatform.user` too broad for this?"

**Gotcha — IAM changes propagate, they don't apply instantly.** After a `add-iam-policy-binding`, allow up to a couple of minutes for the grant (or revoke) to take effect across all systems. A workload that fails with a permission error *immediately* after you fixed the binding is usually just early — retry before re-debugging. The same delay applies to removing access, which matters for security response.

## Service accounts: the identity of your workloads

A **service account** is a non-human identity that your code, VMs, training jobs, and pipelines run as. This is the single most important IAM concept for ML engineering, because your Vertex AI training job, your Cloud Run inference service, and your Dataflow pipeline each execute as a service account and can only touch what that account is granted.

Two things to internalize:

- **Grant roles *to* the service account** so the workload can act (for example, a training job's service account needs `roles/storage.objectAdmin` on the bucket it writes checkpoints to and `roles/bigquery.dataViewer` on the dataset it reads).
- **Grant users the right to *use* the service account** (`roles/iam.serviceAccountUser`) — a data scientist who submits a training job that runs as `training-sa@...` needs permission to act as that account.

```bash
# Create a dedicated service account for training jobs
gcloud iam service-accounts create training-sa \
  --display-name="Vertex AI training jobs"

gcloud iam service-accounts list                              # all SAs in the project
gcloud iam service-accounts describe \
  training-sa@myco-fraud-dev.iam.gserviceaccount.com

# Give it exactly the data access it needs (grant the ROLE to the SA)
gcloud projects add-iam-policy-binding myco-fraud-dev \
  --member="serviceAccount:training-sa@myco-fraud-dev.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

Two grants live *on the service account itself* (managed with `gcloud iam service-accounts add-iam-policy-binding`, not `projects add-iam-policy-binding`), because they answer "who may act as this SA":

```bash
# Let a data scientist submit jobs that RUN AS training-sa
gcloud iam service-accounts add-iam-policy-binding \
  training-sa@myco-fraud-dev.iam.gserviceaccount.com \
  --member="user:alice@example.com" \
  --role="roles/iam.serviceAccountUser"

# Let a CI principal MINT short-lived tokens for training-sa (enables impersonation)
gcloud iam service-accounts add-iam-policy-binding \
  training-sa@myco-fraud-dev.iam.gserviceaccount.com \
  --member="group:ci-deployers@example.com" \
  --role="roles/iam.serviceAccountTokenCreator"
```

The distinction matters: `serviceAccountUser` lets a principal attach the SA to a resource (a VM, a Vertex job); `serviceAccountTokenCreator` lets a principal directly obtain the SA's credentials (the mechanism behind `--impersonate-service-account`).

**Gotcha — the default Compute Engine service account.** The classic mistake is running everything as the **default Compute Engine service account** (`PROJECT_NUMBER-compute@developer.gserviceaccount.com`), which historically starts with the broad `editor` role — a `create-with-container` VM or a quick training box silently inherits near-god-mode. For production, create purpose-built service accounts with narrow roles per workload — one for training, one for serving, one for the data pipeline — so a compromise of one is contained. Relatedly, granting anyone `roles/editor` "to unblock them" is the most common privilege-creep mistake; `editor` can modify almost every resource in the project, so scope to predefined roles instead.

You can audit exactly which keys exist on a service account, which is how you catch a stray downloaded key:

```bash
gcloud iam service-accounts keys list \
  --iam-account=training-sa@myco-fraud-dev.iam.gserviceaccount.com
```

Notice there is no `keys create` in the workflows above — that is deliberate. The next section explains why you avoid minting keys at all.

## Avoid service account keys — use better auth

Downloading a service account **key file** (a JSON private key) is a liability: it is a long-lived credential that leaks in git repos, laptops, and CI logs. Modern Google Cloud gives you two far safer options:

- **Attached service accounts.** A VM, Cloud Run service, Vertex AI job, or GKE workload runs *as* a service account with no key file — credentials are fetched automatically from the metadata server. On GKE this is **Workload Identity Federation for GKE**, which maps Kubernetes service accounts to Google service accounts.
- **Service account impersonation.** Instead of holding a key, an authorized principal *impersonates* a service account to obtain short-lived tokens. Client libraries and `gcloud` support this directly:

```bash
# Any gcloud command accepts --impersonate-service-account (needs TokenCreator on the SA)
gcloud storage ls gs://myco-fraud-data \
  --impersonate-service-account=training-sa@myco-fraud-dev.iam.gserviceaccount.com

# Or set it for a whole session/config so every command runs as the SA
gcloud config set auth/impersonate_service_account \
  training-sa@myco-fraud-dev.iam.gserviceaccount.com
```

- **Workload Identity Federation.** Lets external identities (a GitHub Actions runner, an on-prem system, another cloud) exchange their native tokens for short-lived Google credentials — no keys crossing a boundary. This is how CI/CD (covered later in the course) authenticates to deploy models without a stored key. You create a **pool** to hold external identities and a **provider** that trusts a specific issuer (here, GitHub Actions' OIDC), then let those identities impersonate a deploy service account:

```bash
# 1. Create the pool
gcloud iam workload-identity-pools create github-pool \
  --location="global" --display-name="GitHub Actions"

# 2. Add an OIDC provider that trusts GitHub's token issuer,
#    mapping claims and restricting to one repo
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location="global" --workload-identity-pool="github-pool" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='myco/fraud'"

# 3. Let identities from that repo impersonate the deploy SA (no key involved)
gcloud iam service-accounts add-iam-policy-binding \
  deploy-sa@myco-fraud-prod.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/myco/fraud"
```

The `--attribute-condition` is the security-critical line — without a repo (or branch) restriction, *any* GitHub Actions workflow anywhere could exchange a token for your credentials.

## Least privilege in practice

The organizing principle is **least privilege**: grant the minimum access needed, at the narrowest scope, for the shortest time. Concretely for ML:

- Grant roles on the **specific resource** (a single bucket or dataset) rather than the whole project when you can. A bucket has its own IAM policy.
- Use **groups** for people and **dedicated service accounts** for workloads.
- Use **IAM conditions** to scope grants (by resource name prefix, or with an expiry).
- Separate duties: the person who trains a model need not have rights to deploy to the production serving project.

**IAM conditions** attach a CEL expression to a binding via `--condition`, so a grant only applies when the expression is true. Two patterns cover most ML needs — a time-boxed grant and a resource-name restriction:

```bash
# Temporary access that auto-expires (great for on-call or a debugging session)
gcloud projects add-iam-policy-binding myco-fraud-prod \
  --member="user:oncall@example.com" \
  --role="roles/aiplatform.user" \
  --condition="expression=request.time < timestamp('2026-08-01T00:00:00Z'),title=expires-aug,description=temporary"

# Scope a bucket grant to objects under a prefix only
gcloud storage buckets add-iam-policy-binding gs://myco-fraud-data \
  --member="serviceAccount:training-sa@myco-fraud-dev.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer" \
  --condition="expression=resource.name.startsWith('projects/_/buckets/myco-fraud-data/objects/training/'),title=training-prefix"
```

## Auditing, deny policies, and org-wide guardrails

Least privilege is only real if you can *see* who has what and *prevent* the dangerous grants from happening. Google Cloud gives you a layered set of tools for this, from read-only auditing up to hard organizational guardrails.

For **auditing at scale**, Cloud Asset Inventory searches every IAM policy across a project, folder, or org in one query — far faster than reading each resource's policy by hand:

```bash
# Everywhere alice has any grant, org-wide
gcloud asset search-all-iam-policies \
  --scope=organizations/123456789012 \
  --query="policy:alice@example.com"

# Everyone who holds the dangerous editor role in a project
gcloud asset search-all-iam-policies \
  --scope=projects/myco-fraud-prod \
  --query="policy.role.permissions:roles/editor"
```

When someone asks "why can (or can't) this identity do X," the **policy troubleshooter** evaluates the effective access — allow bindings, conditions, and deny rules together — instead of you tracing inheritance manually:

```bash
gcloud policy-intelligence troubleshoot-policy iam \
  --principal-email=training-sa@myco-fraud-dev.iam.gserviceaccount.com \
  --resource=//storage.googleapis.com/projects/_/buckets/myco-fraud-data \
  --permission=storage.objects.get
```

**Deny policies** are a distinct layer that *overrides* allow grants: even a project owner is blocked from the denied permissions. Use them to fence off a permission no matter what roles someone accumulates. They are defined in a JSON rule file and attached to a resource:

```bash
gcloud iam policies create no-key-creation \
  --attachment-point="cloudresourcemanager.googleapis.com/projects/myco-fraud-prod" \
  --kind=denypolicies --policy-file=deny.json
gcloud iam policies list \
  --attachment-point="cloudresourcemanager.googleapis.com/projects/myco-fraud-prod" \
  --kind=denypolicies
```

**Organization policies** (constraints) are the broadest guardrail — they restrict what *can* be configured at all, independent of IAM. The one every ML org should know is disabling service-account key creation, which structurally enforces the keyless posture this module argues for (and is enforced by default for orgs created since May 2024):

```bash
gcloud org-policies set-policy policy.yaml    # policy.yaml enforces the constraint
# where policy.yaml names constraints/iam.disableServiceAccountKeyCreation
gcloud org-policies describe \
  iam.disableServiceAccountKeyCreation --project=myco-fraud-prod
```

## Secret Manager

ML systems are full of secrets — API keys for third-party model providers, database passwords, tokens for external data sources. Never bake these into code, container images, or environment variables committed to source control. **Secret Manager** stores them encrypted, versioned, and access-controlled by IAM, and it integrates with everything. (Module 14 covers Secret Manager and Cloud KMS in depth — rotation, CMEK, and mounting patterns; the essentials below are enough for the IAM story.)

```bash
# Create a secret and add a version
echo -n "sk-my-provider-key" | \
  gcloud secrets create external-model-api-key --data-file=-

# Grant a serving service account read access to just this secret
gcloud secrets add-iam-policy-binding external-model-api-key \
  --member="serviceAccount:serving-sa@myco-fraud-dev.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

Accessing a secret from Python at runtime:

```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = "projects/myco-fraud-dev/secrets/external-model-api-key/versions/latest"
response = client.access_secret_version(request={"name": name})
api_key = response.payload.data.decode("UTF-8")
```

Secrets are versioned, so you rotate by adding a new version and pointing consumers at `latest` (or pinning a version for reproducibility). Cloud Run, Vertex AI, and GKE can also mount secrets directly, so your inference service reads a mounted file or injected env var without any client code.

## How this fits the whole solution

Every service in the end-to-end system you build later runs as *some* identity and touches *some* resource — and IAM governs each edge. The ingestion pipeline's service account reads Pub/Sub and writes BigQuery; the training job's account reads the dataset and writes the model registry; the serving account pulls the model and reads secrets; CI/CD authenticates via Workload Identity Federation to deploy. Designing these identities and grants with least privilege, keyless auth, and Secret Manager is what turns a pile of services into a system a security team will approve.

## Key takeaways

- IAM binds a **principal** to a **role** on a **resource**, optionally gated by a **condition**, and policies **inherit** down the hierarchy.
- Prefer **predefined roles** over basic roles; use **custom roles** only when nothing fits. Grant on the **narrowest resource** and use **groups** for people.
- **Service accounts** are the identity of your workloads — create dedicated ones per workload with minimal roles, and grant users `serviceAccountUser` to run jobs as them.
- **Avoid key files**: use attached service accounts, **impersonation**, and **Workload Identity Federation** for keyless, short-lived credentials. Store all secrets in **Secret Manager**.

## CLI cheat-sheet

```bash
# --- Grant / inspect / revoke on a project ---
gcloud projects add-iam-policy-binding PROJECT \
  --member="user:alice@example.com" --role="roles/aiplatform.user"
gcloud projects get-iam-policy PROJECT
gcloud projects remove-iam-policy-binding PROJECT \
  --member="user:alice@example.com" --role="roles/aiplatform.user"

# --- Conditions (CEL) ---
gcloud projects add-iam-policy-binding PROJECT \
  --member="user:x@example.com" --role="roles/aiplatform.user" \
  --condition="expression=request.time < timestamp('2026-08-01T00:00:00Z'),title=expires"

# --- Custom roles ---
gcloud iam roles create myRole --project=PROJECT \
  --permissions=aiplatform.endpoints.deploy --stage=GA
gcloud iam roles describe roles/aiplatform.user
gcloud iam roles list --filter="name:aiplatform" --show-details
gcloud iam roles update myRole --project=PROJECT --add-permissions=PERM

# --- Service accounts ---
gcloud iam service-accounts create training-sa --display-name="Training"
gcloud iam service-accounts list
gcloud iam service-accounts describe SA_EMAIL
gcloud iam service-accounts keys list --iam-account=SA_EMAIL   # audit (avoid create)
gcloud iam service-accounts add-iam-policy-binding SA_EMAIL \
  --member="user:x@example.com" --role="roles/iam.serviceAccountUser"
gcloud iam service-accounts add-iam-policy-binding SA_EMAIL \
  --member="group:ci@example.com" --role="roles/iam.serviceAccountTokenCreator"

# --- Keyless auth ---
gcloud storage ls gs://b --impersonate-service-account=SA_EMAIL
gcloud config set auth/impersonate_service_account SA_EMAIL

# --- Workload Identity Federation ---
gcloud iam workload-identity-pools create POOL --location=global
gcloud iam workload-identity-pools providers create-oidc PROVIDER \
  --location=global --workload-identity-pool=POOL \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='myco/fraud'"

# --- Audit & troubleshoot ---
gcloud asset search-all-iam-policies --scope=projects/PROJECT --query="policy:alice@example.com"
gcloud policy-intelligence troubleshoot-policy iam \
  --principal-email=SA_EMAIL --resource=//RESOURCE --permission=PERM

# --- Guardrails ---
gcloud iam policies create NAME --kind=denypolicies \
  --attachment-point="cloudresourcemanager.googleapis.com/projects/PROJECT" \
  --policy-file=deny.json
gcloud org-policies describe iam.disableServiceAccountKeyCreation --project=PROJECT
gcloud org-policies set-policy policy.yaml

# --- Secret Manager (see module 14 for depth) ---
echo -n "value" | gcloud secrets create NAME --data-file=-
gcloud secrets add-iam-policy-binding NAME \
  --member="serviceAccount:SA_EMAIL" --role="roles/secretmanager.secretAccessor"
```

## Try it

Set up production-grade identity for a training workflow with zero key files:

1. Create a dedicated `training-sa` service account.
2. Create a bucket and grant `training-sa` `roles/storage.objectAdmin` **on that bucket only** (not project-wide).
3. Store a dummy API key in Secret Manager and grant `training-sa` `secretAccessor` on just that secret.
4. Grant your own user `roles/iam.serviceAccountUser` on `training-sa`, then run a `gcloud storage` command with `--impersonate-service-account=training-sa@...` and confirm it works — proving you can act as the workload identity without ever downloading a key.
5. Try the same command against a *different* bucket the service account has no access to, and confirm it is denied — that denial is least privilege working as intended.
