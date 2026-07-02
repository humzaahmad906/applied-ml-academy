# 14 — Secrets and Keys: Secret Manager and Cloud KMS

Module 02 introduced Secret Manager as one piece of a least-privilege story; this module goes deep on it and adds the layer beneath it, Cloud Key Management Service (Cloud KMS). Every serious ML system on Google Cloud eventually has to answer two questions a security review will ask: *where do your credentials live* and *who holds the keys that encrypt your data*. Secret Manager answers the first — it stores API keys, database passwords, and provider tokens encrypted, versioned, and IAM-gated. Cloud KMS answers the second — it holds the cryptographic keys used to encrypt buckets, datasets, and Vertex AI resources, so that a regulated fraud-scoring pipeline can prove *the customer*, not just Google, controls the key material. For IAM basics — principals, roles, service accounts, keyless auth — refer back to module 02; this module assumes them and builds on top.

## Secret Manager: secrets versus versions

A **secret** in Secret Manager is a named container; the actual sensitive bytes live in **versions** underneath it. This split is the whole point: you never overwrite a secret's value, you *add a new version*, and every version has an integer id and a state (`enabled`, `disabled`, `destroyed`). Consumers reference either a pinned version (`.../versions/5`) or the alias `latest`, which resolves to the highest enabled version. Rotating a credential is therefore just "add version, flip consumers, disable the old one" — no downtime, and a full audit trail of what value was live when.

```bash
# Create the secret container, then add the first version from stdin
gcloud secrets create fraud-db-password --replication-policy=automatic
echo -n "s3cr3t-pw" | gcloud secrets versions add fraud-db-password --data-file=-

# Add a rotated value; this becomes version 2 and the new `latest`
echo -n "n3w-pw" | gcloud secrets versions add fraud-db-password --data-file=-

# List versions, read the current value, inspect one version's metadata
gcloud secrets versions list fraud-db-password
gcloud secrets versions access latest --secret=fraud-db-password
gcloud secrets versions access 1 --secret=fraud-db-password
```

## Replication: automatic, user-managed, and regional

When you create a secret you choose how its payload is replicated, and **this choice is permanent** — you cannot change a secret's replication policy after creation, only recreate it. There are two families:

- **Automatic replication** (`--replication-policy=automatic`) — Google stores the secret redundantly across the globe and picks the regions. Simplest, and the right default for most workloads.
- **User-managed replication** (`--replication-policy=user-managed --locations=...`) — you pin the exact regions the payload is stored in, for data-residency requirements. All listed regions must be *available* at the moment you add a version, or the write fails.

Separately, Secret Manager offers **regional secrets**: instead of the global endpoint, the secret lives in a single region and is accessed through a regional endpoint, which is what strict compliance regimes (data must never leave `us-central1`) require. You select this by passing `--location` on every command; the secret then has no global replica at all.

```bash
# User-managed replication pinned to two regions
gcloud secrets create fraud-provider-key \
  --replication-policy=user-managed \
  --locations=us-central1,us-east1
echo -n "sk-provider-abc" | gcloud secrets versions add fraud-provider-key --data-file=-

# A regional secret — single-region endpoint, addressed by --location everywhere
gcloud secrets create fraud-residency-token --location=us-central1
echo -n "tok-xyz" | \
  gcloud secrets versions add fraud-residency-token --location=us-central1 --data-file=-
gcloud secrets versions access latest --secret=fraud-residency-token --location=us-central1
```

## `latest` versus pinned versions and reproducibility

Pointing your serving service at `latest` means rotation is transparent — you add a new version and the service picks it up on its next read. That is what you want for a credential like a database password. But for anything that must be *reproducible* — say a signing key or a config token baked into a specific model release — pin the integer version (`.../versions/7`) so a re-run of that pipeline reads exactly the bytes it read the first time. The tradeoff is the mirror image of module 04's storage-class decision: `latest` optimizes for operational freshness, pinning optimizes for reproducibility and auditability. Serving code reads `latest`; a released, versioned artifact pins.

## Enabling, disabling, and destroying versions

Managing a version's lifecycle is how you rotate safely. The best practice is to **disable before you destroy**: disabling makes a version unreadable but reversible, so you can prove nothing broke before you make the destruction permanent. Destroying wipes the payload irreversibly (subject to a TTL, below).

```bash
gcloud secrets versions disable 1 --secret=fraud-db-password   # reversible "off"
gcloud secrets versions enable  1 --secret=fraud-db-password   # undo
gcloud secrets versions destroy 1 --secret=fraud-db-password   # irreversible

# Optional safety net: a version-destroy TTL delays actual destruction,
# parking a destroyed version in a disabled state for the TTL window first
gcloud secrets create fraud-signing-key --version-destroy-ttl=86400s
```

## IAM scoped to a single secret

Grant `roles/secretmanager.secretAccessor` **on the individual secret**, never project-wide — a serving service account should be able to read the one provider key it needs and nothing else. This is module 02's least-privilege principle applied at resource granularity: a secret has its own IAM policy.

```bash
gcloud secrets add-iam-policy-binding fraud-provider-key \
  --member="serviceAccount:serving-sa@myco-fraud-prod.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Rotation schedules and CMEK on secrets

Secret Manager can *remind* you to rotate. A **rotation schedule** does not change the value for you — it publishes a message to a Pub/Sub topic on a cadence, which a Cloud Function or pipeline consumes to fetch a fresh credential and add a new version. You wire it up with `--next-rotation-time` and `--rotation-period`, plus a topic the Secret Manager service agent is allowed to publish to. Separately, you can encrypt the secret payload itself with your own KMS key (**CMEK**) via `--kms-key-name`; note that CMEK on secrets requires user-managed or regional replication, because the key has a fixed location. You can also give any secret an **expiration** with `--expire-time`, after which the whole secret is deleted — useful for genuinely short-lived tokens.

```bash
gcloud secrets create fraud-rotating-key \
  --replication-policy=user-managed --locations=us-central1 \
  --topics=projects/myco-fraud-prod/topics/secret-rotation \
  --next-rotation-time="2026-08-01T00:00:00Z" \
  --rotation-period="2592000s" \
  --kms-key-name="projects/myco-fraud-prod/locations/us-central1/keyRings/fraud-ring/cryptoKeys/secret-cmek" \
  --expire-time="2027-01-01T00:00:00Z"
```

## Accessing secrets: code versus native mounts

The single most important rule: **your code should never embed a key**. There are two clean ways to get a secret to a workload.

First, the Python client library, for programmatic reads at runtime:

```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = "projects/myco-fraud-prod/secrets/fraud-provider-key/versions/latest"
response = client.access_secret_version(request={"name": name})
api_key = response.payload.data.decode("UTF-8")
```

Second — and preferred for serving — let the platform **mount** the secret so the code never even calls the API. Cloud Run, GKE, and Vertex AI can inject a secret as an environment variable or mount it as a file; your container reads a path or an env var and stays oblivious to Secret Manager entirely. On Cloud Run:

```bash
# Mount as a file at /secrets/api-key, and inject one as an env var
gcloud run deploy fraud-scorer \
  --image=us-central1-docker.pkg.dev/myco-fraud-prod/serving/scorer:latest \
  --set-secrets=/secrets/api-key=fraud-provider-key:latest \
  --set-secrets=DB_PASSWORD=fraud-db-password:latest \
  --service-account=serving-sa@myco-fraud-prod.iam.gserviceaccount.com
```

The serving code then reads `open("/secrets/api-key").read()` or `os.environ["DB_PASSWORD"]` — no SDK, no embedded credential. On GKE, the Secret Manager CSI driver mounts secrets as volumes; on Vertex custom training/serving you read them via the client library using the job's service account.

## Cloud KMS: key rings, keys, and versions

Cloud KMS is the vault for the *keys*, not the secrets. Its hierarchy is: a **key ring** (a grouping bound to one location — the location is immutable and a key ring can never be deleted, only emptied) contains **crypto keys**, and each key contains **key versions** that hold the actual cryptographic material. One version is the **primary version**, the one used when you encrypt; rotation creates a new version and promotes it to primary while old versions stay available to decrypt data they previously encrypted.

Keys have a **purpose** that fixes what they can do:

- `encryption` — symmetric encrypt/decrypt (the common case for CMEK).
- `asymmetric-signing` — sign and verify with a public/private key pair.
- `asymmetric-encryption` — encrypt with a public key, decrypt with the private key.
- `mac` — message authentication codes.

```bash
# A regional key ring, then a symmetric key that auto-rotates every 90 days
gcloud kms keyrings create fraud-ring --location=us-central1

gcloud kms keys create data-cmek \
  --keyring=fraud-ring --location=us-central1 \
  --purpose=encryption \
  --rotation-period=90d \
  --next-rotation-time="2026-08-01T00:00:00Z"

gcloud kms keys versions list --key=data-cmek --keyring=fraud-ring --location=us-central1
```

## Protection levels and direct encrypt/decrypt

Each key has a **protection level** describing where the key material lives and how it is guarded:

- `SOFTWARE` — Google-managed software backend; the default.
- `HSM` — Cloud HSM, FIPS 140-2 Level 3 hardware security modules; use for regulated workloads.
- `EXTERNAL` / `EXTERNAL_VPC` — the key lives in an external key manager (over the internet, or reachable through your VPC), so key material never resides in Google Cloud at all.

You can encrypt and decrypt small payloads directly with a symmetric key, which is handy for wrapping a config blob or a one-off secret without standing up Secret Manager:

```bash
gcloud kms keys create hsm-signer \
  --keyring=fraud-ring --location=us-central1 \
  --purpose=encryption --protection-level=hsm

gcloud kms encrypt --location=us-central1 --keyring=fraud-ring --key=data-cmek \
  --plaintext-file=config.json --ciphertext-file=config.json.enc
gcloud kms decrypt --location=us-central1 --keyring=fraud-ring --key=data-cmek \
  --ciphertext-file=config.json.enc --plaintext-file=config.json.dec
```

Grant a workload the ability to use a key with `roles/cloudkms.cryptoKeyEncrypterDecrypter` on that specific key:

```bash
gcloud kms keys add-iam-policy-binding data-cmek \
  --keyring=fraud-ring --location=us-central1 \
  --member="serviceAccount:training-sa@myco-fraud-prod.iam.gserviceaccount.com" \
  --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"
```

## Destroying and restoring key versions

You cannot delete a KMS key or key ring — they are permanent by design, so that ciphertext can always be traced to a key. What you *can* do is **schedule destruction** of a key *version*: it enters a `DESTROY_SCHEDULED` state for a configurable window (24 hours by default), during which you can still **restore** it. Once the window elapses the material is gone irreversibly, and any data still encrypted under only that version becomes permanently unrecoverable — which is exactly why the scheduled window and the restore command exist.

```bash
gcloud kms keys versions destroy 1 --key=data-cmek \
  --keyring=fraud-ring --location=us-central1
# ...changed your mind within the window:
gcloud kms keys versions restore 1 --key=data-cmek \
  --keyring=fraud-ring --location=us-central1
```

## CMEK for ML data: buckets, BigQuery, and Vertex AI

By default Google encrypts everything at rest with Google-managed keys. **Customer-managed encryption keys (CMEK)** replace that key with one *you* control in Cloud KMS, so you can rotate it, audit its use, and disable it to cut off access to the data instantly. Regulated fraud data typically mandates CMEK precisely for that last property: the ability to prove you can revoke access unilaterally.

The universal prerequisite — and the number-one thing people miss — is that each service acts through its own **service agent**, and that service agent must hold `roles/cloudkms.cryptoKeyEncrypterDecrypter` on the key before the service can use it. If you skip this grant, resource creation or object writes fail with a permission error that does not mention KMS clearly.

```bash
# Grant the Cloud Storage service agent use of the key, then create a CMEK bucket
gcloud storage service-agent --project=myco-fraud-prod   # prints the SA email to grant
gcloud kms keys add-iam-policy-binding data-cmek \
  --keyring=fraud-ring --location=us-central1 \
  --member="serviceAccount:service-<PROJECT_NUMBER>@gs-project-accounts.iam.gserviceaccount.com" \
  --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"

gcloud storage buckets create gs://myco-fraud-data \
  --location=us-central1 \
  --default-encryption-key="projects/myco-fraud-prod/locations/us-central1/keyRings/fraud-ring/cryptoKeys/data-cmek"

# BigQuery: set a dataset's default KMS key (grant the BQ service agent first)
bq update --default_kms_key="projects/myco-fraud-prod/locations/us-central1/keyRings/fraud-ring/cryptoKeys/data-cmek" \
  myco-fraud-prod:fraud

# Vertex AI resources (training jobs, endpoints, metadata store) take an
# --encryption-spec-key-name pointing at the same key.
```

Vertex AI resources accept an `--encryption-spec-key-name` flag (or `encryption_spec` in the SDK) so training jobs, endpoints, and the metadata store all encrypt under your key. The KMS key's **location must match the resource's location** — a `us-central1` key cannot encrypt a `us-east1` bucket. For teams that want CMEK without hand-managing key rings, **Cloud KMS Autokey** provisions and rotates purpose-scoped keys on demand as you create resources, which reduces the operational surface at scale.

## How this fits the whole solution

Secrets and keys are the trust layer that runs under everything else in the course. The ingestion and training identities from module 02 read their provider credentials and database passwords out of Secret Manager rather than from baked-in config; the Cloud Storage buckets of module 04, the BigQuery datasets of module 08, and the Vertex AI training (module 09), registry, and prediction (module 10) resources all encrypt their data under a Cloud KMS CMEK you rotate and can revoke. The serving service of modules 06 and 10 mounts its API key with `--set-secrets` so no key ever touches the image. And the cost and observability work in modules 13 and 15 watches for secret-access and key-use anomalies. Getting this layer right is what lets a security and compliance review sign off on the whole fraud-scoring system.

## Key takeaways

- **Secret Manager separates secrets from versions**: rotate by adding a version and flipping consumers between `latest` (operational freshness) and a pinned version (reproducibility); disable before you destroy.
- **Replication is permanent**: choose `automatic`, `user-managed --locations=...`, or a `--location` regional secret at creation for data residency — you cannot change it later, and user-managed regions must all be up to add a version.
- **Cloud KMS holds the keys**: key rings are location-bound and undeletable, keys have a purpose (`encryption`/asymmetric/`mac`) and a protection level (`SOFTWARE`/`HSM`/`EXTERNAL`), and destroying a key version is irreversible after its scheduled window.
- **CMEK puts regulated ML data under your key** for Cloud Storage, BigQuery, and Vertex AI — but each service's **service agent must first hold `cryptoKeyEncrypterDecrypter` on the key**, and the key's location must match the resource's.

## CLI cheat-sheet

```bash
# --- Secret Manager: secrets & versions ---
gcloud secrets create SECRET --replication-policy=automatic
gcloud secrets create SECRET --replication-policy=user-managed --locations=us-central1,us-east1
gcloud secrets create SECRET --location=us-central1                # regional secret
echo -n "VALUE" | gcloud secrets versions add SECRET --data-file=-
gcloud secrets versions list SECRET
gcloud secrets versions access latest --secret=SECRET
gcloud secrets versions access 3 --secret=SECRET
gcloud secrets versions disable|enable|destroy N --secret=SECRET

# Secret IAM (scoped to one secret), rotation, CMEK, expiry
gcloud secrets add-iam-policy-binding SECRET \
  --member="serviceAccount:serving-sa@myco-fraud-prod.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
gcloud secrets create SECRET --topics=projects/PROJ/topics/TOPIC \
  --next-rotation-time=TS --rotation-period=2592000s
gcloud secrets create SECRET --kms-key-name=KMS_KEY --replication-policy=user-managed --locations=us-central1
gcloud secrets create SECRET --expire-time=TS

# Native mount into Cloud Run (no SDK, no embedded key)
gcloud run deploy SVC --set-secrets=/secrets/api-key=SECRET:latest --set-secrets=ENV_VAR=SECRET2:latest

# --- Cloud KMS ---
gcloud kms keyrings create RING --location=us-central1
gcloud kms keys create KEY --keyring=RING --location=us-central1 \
  --purpose=encryption --rotation-period=90d --next-rotation-time=TS
gcloud kms keys create KEY --keyring=RING --location=us-central1 \
  --purpose=encryption --protection-level=hsm
gcloud kms keys versions list --key=KEY --keyring=RING --location=us-central1
gcloud kms encrypt --location=us-central1 --keyring=RING --key=KEY \
  --plaintext-file=in --ciphertext-file=out.enc
gcloud kms decrypt --location=us-central1 --keyring=RING --key=KEY \
  --ciphertext-file=out.enc --plaintext-file=out.dec
gcloud kms keys versions destroy|restore N --key=KEY --keyring=RING --location=us-central1
gcloud kms keys add-iam-policy-binding KEY --keyring=RING --location=us-central1 \
  --member="serviceAccount:SA" --role="roles/cloudkms.cryptoKeyEncrypterDecrypter"

# --- CMEK on ML data (grant the service agent on the key FIRST) ---
gcloud storage service-agent --project=myco-fraud-prod
gcloud storage buckets create gs://myco-fraud-data --location=us-central1 --default-encryption-key=KMS_KEY
bq update --default_kms_key=KMS_KEY myco-fraud-prod:fraud
# Vertex: pass --encryption-spec-key-name=KMS_KEY on the resource
```

## Try it

Stand up the trust layer for the fraud-scoring serving path, keys and secrets both under your control:

1. Create a key ring `fraud-ring` in `us-central1`, then a symmetric `encryption` key `data-cmek` with a 90-day `--rotation-period`.
2. Grant the Cloud Storage service agent `roles/cloudkms.cryptoKeyEncrypterDecrypter` on `data-cmek`, then create a bucket `gs://myco-fraud-data` with `--default-encryption-key` pointing at it. Upload a file and confirm the object reports CMEK encryption.
3. Create a secret `fraud-provider-key` with `--replication-policy=user-managed --locations=us-central1`, add a value as version 1, then rotate by adding version 2. List versions and `access latest` to confirm it returns the new value; then `disable` version 1.
4. Grant `serving-sa` `roles/secretmanager.secretAccessor` on `fraud-provider-key` only, then deploy a Cloud Run service with `--set-secrets=/secrets/api-key=fraud-provider-key:latest` and confirm the container can read `/secrets/api-key` with no SDK call.
5. Schedule destruction of KMS key version 1, observe it enter `DESTROY_SCHEDULED`, then `restore` it within the window — proving the irreversibility only kicks in after the scheduled delay.
