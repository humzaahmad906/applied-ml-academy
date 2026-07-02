# 14 — Secrets: Key Vault and Managed Identities

The identity module established the golden rule: internal services authenticate with **managed identities**, so nothing sensitive lands in code. But real ML systems still have a handful of genuine secrets — a partner API token, a database password for a legacy system, a webhook signing key, a TLS certificate for an ingress — and they also need **cryptographic keys** for envelope encryption and **certificates** for mutual TLS. **Azure Key Vault** is the managed store for all three: secrets, keys, and certificates. This module goes past the "store a token" basics from the identity section and covers the vault's data-plane object types, the two ways it authorizes access, soft-delete and rotation, and how managed identities read from it at runtime with no bootstrapping credential. In the end-to-end solution, Key Vault is the single, audited place a human-set secret enters the platform — everything else is keyless.

## The three object types: secrets, keys, certificates

A vault holds three kinds of objects, and the distinction matters because they have different data-plane roles and different SDK clients:

- **Secrets** — arbitrary strings up to 25 KB: API tokens, connection strings, passwords. The vault stores and returns the value; your code reads it. This is what you use most in ML glue.
- **Keys** — asymmetric or symmetric cryptographic keys (RSA, EC, or AES) that **never leave the vault**. You do not read a key's material; you ask the vault to sign, verify, encrypt, decrypt, or wrap/unwrap *with* it. This is the basis of **customer-managed keys (CMK)** — encrypting your storage account, ML workspace, or disks with a key you control and can revoke. Premium vaults back keys with a **FIPS 140-2 Level 3 HSM**.
- **Certificates** — X.509 certificates with their private keys, plus lifecycle policy (issuer, validity, auto-renewal). Used for TLS on an ingress or mutual-TLS between services.

```bash
# A secret (a value you read back)
az keyvault secret set --vault-name kv-mlx-dev -n db-password --value "$(openssl rand -base64 24)"
az keyvault secret show --vault-name kv-mlx-dev -n db-password --query value -o tsv

# A key (material stays in the vault; you use it, never read it)
az keyvault key create --vault-name kv-mlx-dev -n cmk-storage --kty RSA --size 3072
az keyvault key list --vault-name kv-mlx-dev -o table

# A certificate with an auto-renewing self-signed policy (swap the issuer for a real CA)
az keyvault certificate create --vault-name kv-mlx-dev -n tls-ingress \
  -p "$(az keyvault certificate get-default-policy)"
```

Use a **Premium** vault (`--sku premium`) when you need HSM-backed keys for CMK or compliance; **Standard** is fine for secrets and software-protected keys.

## Two authorization models — pick RBAC

Key Vault has two mutually exclusive ways to authorize *data-plane* access (reading a secret, using a key), and this is the single most common source of "I get 403 and I have Owner" confusion:

- **Azure RBAC** (recommended, and the default for new vaults) — data-plane permissions are ordinary role assignments, so they integrate with PIM, custom roles, and the same `az role assignment` tooling as everything else.
- **Access policies** (legacy) — a per-vault list of principal → allowed operations, with no PIM and coarse auditing. Avoid for new work; you will inherit it on older vaults.

Remember the control-plane/data-plane split from the identity module: **Owner/Contributor lets you manage the vault but does not let you read a secret**. Grant the right *data* role at vault scope (or, for tighter control, at an individual secret's scope):

```bash
az keyvault create -n kv-mlx-dev -g rg-mlx-dev --enable-rbac-authorization true --sku standard
KV_ID=$(az keyvault show -n kv-mlx-dev -g rg-mlx-dev --query id -o tsv)

# Readers get "Secrets User"; the few who manage secrets get "Secrets Officer"
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" --scope "$KV_ID"
# Key users (for CMK crypto operations) get "Crypto User"; cert managers get "Certificates Officer"
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Key Vault Crypto User" --scope "$KV_ID"
```

The relevant data-plane roles: **Key Vault Secrets User** (read secrets), **Secrets Officer** (manage), **Crypto User** (use keys), **Crypto Officer** (manage keys), **Certificates Officer** (manage certs), and **Reader** (list metadata only, no values).

## Reading secrets at runtime with a managed identity

The whole point is that an ML workload fetches a secret at runtime using its **own managed identity** — no secret is needed to get the secret, so there is no bootstrapping problem. This is the identical `DefaultAzureCredential` pattern from every other module:

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

client = SecretClient("https://kv-mlx-dev.vault.azure.net", DefaultAzureCredential())
db_password = client.get_secret("db-password").value   # locally uses az login; in Azure uses the MI
```

For the **key** case (envelope encryption, signing) use the `azure-keyvault-keys` `CryptographyClient`, which performs the operation inside the vault so the key material never reaches your process. For certificates, `azure-keyvault-certificates` retrieves the cert and (with the right role) its private key for TLS.

An **Azure ML workspace** is attached to a vault at creation; the datastore connection strings and any credentials the workspace manages live there, not in your code. When a job needs a secret, reference the workspace's vault rather than shipping the value in an environment variable or the code snapshot.

## Soft-delete, purge protection, and rotation

Two data-loss safeguards are on by default on modern vaults and you should never disable them:

- **Soft-delete** — a deleted vault or object is recoverable for a retention window (7–90 days), so a fat-fingered delete or a compromised principal cannot permanently destroy a signing key.
- **Purge protection** — once enabled, even a soft-deleted object cannot be *permanently* purged before its retention expires, defeating a "delete then purge" attack. Required for CMK scenarios.

```bash
az keyvault update -n kv-mlx-dev -g rg-mlx-dev --enable-purge-protection true --retention-days 90
az keyvault secret list-deleted --vault-name kv-mlx-dev -o table   # what's recoverable
az keyvault secret recover --vault-name kv-mlx-dev -n db-password  # undo a delete
```

**Rotation** is the operational discipline that turns a secret from a liability into a managed asset. Secrets and keys are **versioned** — setting a new value creates a new version while old versions remain resolvable, so a rotation does not break in-flight readers. For supported secret types (like storage keys), Key Vault can **auto-rotate** on a policy; for the rest, rotate on a schedule and let readers pick up the current version by omitting the version in the URI. Wire an **Event Grid** notification on `SecretNearExpiry` (see the messaging module) so a Function rotates before anything expires.

```bash
# Version history, and an auto-rotation policy (e.g. rotate 30 days before expiry)
az keyvault secret list-versions --vault-name kv-mlx-dev -n db-password -o table
az keyvault secret set-attributes  --vault-name kv-mlx-dev -n db-password --expires "2027-01-01T00:00:00Z"
az keyvault key rotation-policy update --vault-name kv-mlx-dev -n cmk-storage --value @rotation-policy.json
```

## Network isolation

Like storage, a vault has a public endpoint by default. For a production ML platform, give it a **private endpoint** and disable public access so it is reachable only from inside the VNet (the discipline from the networking module). A firewall exception for trusted Azure services and your CI/CD egress IPs is the common middle ground during migration.

```bash
az keyvault update -n kv-mlx-dev -g rg-mlx-dev --public-network-access Disabled --default-action Deny
az network private-endpoint create -g rg-mlx-dev -n pe-kv --vnet-name vnet-mlx --subnet snet-pe \
  --private-connection-resource-id "$KV_ID" --group-id vault --connection-name conn-kv
```

## How Key Vault fits the whole solution

Key Vault is the platform's secret and key backbone, sitting behind a private endpoint on the VNet. The **workspace** attaches to it for datastore credentials. **Managed identities** on training compute, endpoints, and Functions read the rare human-set secret (a partner token, a legacy DB password) at runtime via `DefaultAzureCredential` — no secret bootstraps another secret. **Customer-managed keys** in the vault encrypt the storage account, workspace, and disks, so revoking a key instantly cuts access to the data it protects. **Certificates** terminate TLS at the ingress in front of the endpoints. **Event Grid** fires on near-expiry so rotation is automated. And because the vault is RBAC-mode, every access is a role assignment you can audit, PIM-gate, and scope — the same governance the rest of the platform uses.

## Key takeaways

- A vault holds three object types: **secrets** (values you read), **keys** (crypto ops that stay in the vault, the basis of **CMK**), and **certificates** (TLS/mTLS with lifecycle); use **Premium** for HSM-backed keys.
- Choose **Azure RBAC** authorization (default for new vaults), not legacy access policies; remember Owner/Contributor does **not** grant data-plane reads — you need **Secrets User / Crypto User / Certificates Officer** at the right scope.
- Read secrets at runtime with a **managed identity + `DefaultAzureCredential`** — no secret is needed to fetch a secret; the ML **workspace** attaches to a vault for its datastore credentials.
- Keep **soft-delete** and **purge protection** on, **version** and **rotate** secrets/keys, and use Event Grid `SecretNearExpiry` to automate rotation.
- Put the vault behind a **private endpoint** with public access disabled — same network isolation as storage.

## CLI cheat-sheet

```bash
# --- vault lifecycle (RBAC mode, protected) ---
az keyvault create -n kv-mlx-dev -g rg-mlx-dev --enable-rbac-authorization true --sku standard
az keyvault update -n kv-mlx-dev -g rg-mlx-dev --enable-purge-protection true --retention-days 90
az keyvault show   -n kv-mlx-dev -g rg-mlx-dev -o jsonc
az keyvault list-deleted -o table
az keyvault purge  -n kv-mlx-dev   # only after retention; blocked by purge protection

# --- secrets ---
az keyvault secret set          --vault-name kv-mlx-dev -n db-password --value "s3cr3t"
az keyvault secret show         --vault-name kv-mlx-dev -n db-password --query value -o tsv
az keyvault secret list         --vault-name kv-mlx-dev -o table
az keyvault secret list-versions --vault-name kv-mlx-dev -n db-password -o table
az keyvault secret set-attributes --vault-name kv-mlx-dev -n db-password --expires "2027-01-01T00:00:00Z"
az keyvault secret list-deleted --vault-name kv-mlx-dev -o table
az keyvault secret recover      --vault-name kv-mlx-dev -n db-password

# --- keys (CMK / crypto) ---
az keyvault key create   --vault-name kv-mlx-dev -n cmk-storage --kty RSA --size 3072
az keyvault key create   --vault-name kv-mlx-dev -n cmk-hsm     --kty RSA-HSM --size 3072   # Premium/HSM
az keyvault key list     --vault-name kv-mlx-dev -o table
az keyvault key rotation-policy update --vault-name kv-mlx-dev -n cmk-storage --value @rotation-policy.json

# --- certificates ---
az keyvault certificate create --vault-name kv-mlx-dev -n tls-ingress -p "$(az keyvault certificate get-default-policy)"
az keyvault certificate show   --vault-name kv-mlx-dev -n tls-ingress -o jsonc
az keyvault certificate list   --vault-name kv-mlx-dev -o table

# --- data-plane RBAC (grant the right role, not Owner) ---
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" --scope "$KV_ID"
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Key Vault Crypto User"  --scope "$KV_ID"

# --- network isolation ---
az keyvault update -n kv-mlx-dev -g rg-mlx-dev --public-network-access Disabled --default-action Deny
az network private-endpoint create -g rg-mlx-dev -n pe-kv --vnet-name vnet-mlx --subnet snet-pe \
  --private-connection-resource-id "$KV_ID" --group-id vault --connection-name conn-kv
```

## Try it

Create an RBAC-mode vault with purge protection enabled. Store a secret, a key, and a self-signed certificate. Grant your managed identity **Key Vault Secrets User** (only), then read the secret back with `DefaultAzureCredential` and confirm you *cannot* read it after removing the role — proving data-plane RBAC works independently of any Owner/Contributor grant you hold. Delete the secret, then `recover` it from soft-delete. Finally, add a private endpoint and set public access to Disabled, and confirm a reader inside the VNet still works while an outside caller now fails.
