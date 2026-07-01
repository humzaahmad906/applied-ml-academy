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

## Service accounts: the identity of your workloads

A **service account** is a non-human identity that your code, VMs, training jobs, and pipelines run as. This is the single most important IAM concept for ML engineering, because your Vertex AI training job, your Cloud Run inference service, and your Dataflow pipeline each execute as a service account and can only touch what that account is granted.

Two things to internalize:

- **Grant roles *to* the service account** so the workload can act (for example, a training job's service account needs `roles/storage.objectAdmin` on the bucket it writes checkpoints to and `roles/bigquery.dataViewer` on the dataset it reads).
- **Grant users the right to *use* the service account** (`roles/iam.serviceAccountUser`) — a data scientist who submits a training job that runs as `training-sa@...` needs permission to act as that account.

```bash
# Create a dedicated service account for training jobs
gcloud iam service-accounts create training-sa \
  --display-name="Vertex AI training jobs"

# Give it exactly the data access it needs
gcloud projects add-iam-policy-binding myco-fraud-dev \
  --member="serviceAccount:training-sa@myco-fraud-dev.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

The classic mistake is running everything as the **default Compute Engine service account**, which starts with the broad `editor` role. For production, create purpose-built service accounts with narrow roles per workload — one for training, one for serving, one for the data pipeline — so a compromise of one is contained.

## Avoid service account keys — use better auth

Downloading a service account **key file** (a JSON private key) is a liability: it is a long-lived credential that leaks in git repos, laptops, and CI logs. Modern Google Cloud gives you two far safer options:

- **Attached service accounts.** A VM, Cloud Run service, Vertex AI job, or GKE workload runs *as* a service account with no key file — credentials are fetched automatically from the metadata server. On GKE this is **Workload Identity Federation for GKE**, which maps Kubernetes service accounts to Google service accounts.
- **Service account impersonation.** Instead of holding a key, an authorized principal *impersonates* a service account to obtain short-lived tokens. Client libraries and `gcloud` support this directly:

```bash
gcloud storage ls gs://my-bucket \
  --impersonate-service-account=training-sa@myco-fraud-dev.iam.gserviceaccount.com
```

- **Workload Identity Federation.** Lets external identities (a GitHub Actions runner, an on-prem system, another cloud) exchange their native tokens for short-lived Google credentials — no keys crossing a boundary. This is how CI/CD (covered later in the course) authenticates to deploy models without a stored key.

## Least privilege in practice

The organizing principle is **least privilege**: grant the minimum access needed, at the narrowest scope, for the shortest time. Concretely for ML:

- Grant roles on the **specific resource** (a single bucket or dataset) rather than the whole project when you can. A bucket has its own IAM policy.
- Use **groups** for people and **dedicated service accounts** for workloads.
- Use **IAM conditions** to scope grants (by resource name prefix, or with an expiry).
- Separate duties: the person who trains a model need not have rights to deploy to the production serving project.

## Secret Manager

ML systems are full of secrets — API keys for third-party model providers, database passwords, tokens for external data sources. Never bake these into code, container images, or environment variables committed to source control. **Secret Manager** stores them encrypted, versioned, and access-controlled by IAM, and it integrates with everything.

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

## Try it

Set up production-grade identity for a training workflow with zero key files:

1. Create a dedicated `training-sa` service account.
2. Create a bucket and grant `training-sa` `roles/storage.objectAdmin` **on that bucket only** (not project-wide).
3. Store a dummy API key in Secret Manager and grant `training-sa` `secretAccessor` on just that secret.
4. Grant your own user `roles/iam.serviceAccountUser` on `training-sa`, then run a `gcloud storage` command with `--impersonate-service-account=training-sa@...` and confirm it works — proving you can act as the workload identity without ever downloading a key.
5. Try the same command against a *different* bucket the service account has no access to, and confirm it is denied — that denial is least privilege working as intended.
