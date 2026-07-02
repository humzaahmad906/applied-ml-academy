# 15 — Secrets and Keys: Secrets Manager, KMS, Parameter Store

Module 02 established that credentials never belong in code and data should be encrypted at rest. This module makes that concrete with the three services that do the work: **AWS KMS** manages the encryption keys that protect your S3 data, EBS volumes, model artifacts, and everything else; **AWS Secrets Manager** stores and rotates the passwords, database credentials, and third-party API keys your ML system needs at runtime; and **SSM Parameter Store** holds configuration and lighter secrets cheaply. Getting these right is what lets an ML platform pass a security review — an auditor can see who can decrypt training data, and a leaked config file contains no live secrets.

## KMS: the key management foundation

**AWS Key Management Service** manages cryptographic keys and performs encryption/decryption on your behalf without ever exposing the raw key material. Almost every "encrypt this" checkbox in AWS — S3 SSE-KMS, EBS volume encryption, Secrets Manager, SageMaker volumes and outputs — resolves to a KMS key. There are three flavors you will meet: **AWS-owned keys** (invisible, managed entirely by AWS), **AWS-managed keys** (named like `aws/s3`, created per service, free but not configurable), and **customer-managed keys (CMKs)** — the ones you create, whose key policy, rotation, and access you control. For ML data you want CMKs, because a CMK lets you audit every decrypt in CloudTrail and *revoke* access to encrypted data independently of the storage service.

```bash
# Create a customer-managed key and give it a friendly alias
aws kms create-key --description "ml-data encryption key" --tags TagKey=team,TagValue=ml
aws kms create-alias --alias-name alias/ml-data --target-key-id <key-id>

# Turn on automatic yearly rotation (KMS keeps old key versions to decrypt old data)
aws kms enable-key-rotation --key-id <key-id>
```

The mechanism that makes KMS scale is **envelope encryption**: KMS does not encrypt your 50 GB dataset directly. Instead it generates a **data key**, returns it to the service in both plaintext (to encrypt the data locally) and encrypted forms (to store alongside the ciphertext); the plaintext data key is discarded from memory after use. This is why S3 can encrypt petabytes with one KMS key without every byte round-tripping through KMS.

```bash
# Envelope encryption primitives (what services call under the hood)
aws kms generate-data-key --key-id alias/ml-data --key-spec AES_256
aws kms encrypt --key-id alias/ml-data --plaintext fileb://token.bin --output text --query CiphertextBlob
aws kms decrypt --ciphertext-blob fileb://cipher.bin --query Plaintext --output text | base64 -d
```

Two access-control concepts matter. The **key policy** is the resource-based policy on the key itself and is the root of trust — an IAM policy granting `kms:Decrypt` does nothing unless the key policy also allows that principal. **Grants** are a lighter, temporary, programmatic way to delegate specific key operations to a service (SageMaker, Lambda) for the life of a workload without editing the key policy. If a training job fails with `AccessDenied` decrypting its input, the cause is almost always the key policy, not the S3 bucket policy.

```bash
# Grant a SageMaker execution role permission to use the key for a job
aws kms create-grant --key-id alias/ml-data \
  --grantee-principal arn:aws:iam::<acct>:role/ml-sagemaker-exec \
  --operations Decrypt GenerateDataKey
```

## Secrets Manager: runtime credentials with rotation

**AWS Secrets Manager** stores secrets encrypted with KMS and hands them to your code at runtime by name or ARN. Its differentiator over a plain encrypted parameter is **automatic rotation**: it can invoke a Lambda function on a schedule to generate a new credential, update the source system (an RDS database, an API provider), and atomically swap the stored value — using staging labels (`AWSCURRENT`, `AWSPENDING`, `AWSPREVIOUS`) so in-flight callers never see a broken state. For RDS/Aurora, Redshift, and DocumentDB, AWS provides the rotation Lambda for you.

```bash
# Store a secret (JSON is the convention so one secret holds related fields)
aws secretsmanager create-secret --name prod/model-provider \
  --secret-string '{"api_key":"sk-...","org":"acme"}' \
  --kms-key-id alias/ml-data

# Fetch it at runtime (this is what your ML code calls)
aws secretsmanager get-secret-value --secret-id prod/model-provider \
  --query SecretString --output text

# Turn on automatic rotation every 30 days via a rotation Lambda
aws secretsmanager rotate-secret --secret-id prod/model-provider \
  --rotation-lambda-arn arn:aws:lambda:us-east-1:<acct>:function:rotate-provider-key \
  --rotation-rules '{"AutomaticallyAfterDays":30}'
```

In boto3, the pattern is fetch-once-and-cache — Secrets Manager charges per 10,000 API calls, so pulling the secret on every inference request is both slow and needlessly expensive:

```python
import boto3, json
sm = boto3.client("secretsmanager")
_secret = json.loads(sm.get_secret_value(SecretId="prod/model-provider")["SecretString"])
# reuse _secret across warm invocations; refresh only on rotation
```

The common ML gotcha: a Lambda or SageMaker job that can *read* the secret's metadata but fails to retrieve the value because its execution role lacks `kms:Decrypt` on the CMK the secret is encrypted with. Secrets Manager access is two permissions — `secretsmanager:GetSecretValue` **and** KMS decrypt.

## Parameter Store: cheap config and light secrets

**SSM Parameter Store** stores configuration as a hierarchical tree of parameters. A `String` or `StringList` parameter is plaintext config (a model name, an endpoint URL, a feature-flag value); a **`SecureString`** is encrypted with KMS, giving you a no-extra-cost secret store for values that do not need automatic rotation. The naming hierarchy (`/ml/prod/endpoint-name`) lets you fetch a whole subtree at once, which is how services load their config at startup.

```bash
# Config value and an encrypted secret, both in one hierarchy
aws ssm put-parameter --name /ml/prod/endpoint-name --value fraud-scorer --type String
aws ssm put-parameter --name /ml/prod/db-password --value 's3cr3t' \
  --type SecureString --key-id alias/ml-data

# Read a whole subtree at container startup, decrypting SecureStrings
aws ssm get-parameters-by-path --path /ml/prod --recursive --with-decryption
```

Choosing between them: **Secrets Manager** when you need automatic rotation, cross-account sharing, or managed RDS credential integration; **Parameter Store SecureString** when you just need an encrypted value read cheaply and often. Parameter Store's standard tier is free and handles thousands of reads; Secrets Manager charges per secret per month plus per API call, which is the price of rotation and lifecycle features. Many teams use both: rotating database/provider credentials in Secrets Manager, static config and non-rotating tokens in Parameter Store.

## Encrypting the ML pipeline end to end

Tie it together: create one CMK per data classification, encrypt the S3 data lake with SSE-KMS pointing at it, encrypt SageMaker training volumes and outputs with the same key (`VolumeKmsKeyId` / `OutputDataConfig.KmsKeyId`), store the model-provider API key and database password in Secrets Manager (encrypted with that CMK), and put non-secret config in Parameter Store. Because every decrypt lands in CloudTrail, you get one audit trail answering "who accessed this model's data," and revoking the grant or disabling the key instantly locks everyone out — the property compliance reviews demand.

```bash
# Encrypt a bucket with your CMK; force all writes to use it
aws s3api put-bucket-encryption --bucket my-ml-data \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms","KMSMasterKeyID":"alias/ml-data"},"BucketKeyEnabled":true}]}'
```

Note `BucketKeyEnabled` above: it caches a bucket-level data key so S3 does not call KMS on every object, cutting KMS request cost dramatically on high-volume ML buckets — a real bill saver people miss.

## How this fits the whole ML solution

Secrets and keys are the enforcement layer beneath the IAM trust graph from module 02. Every arrow in the reference architecture that carries data — ingestion to lake, lake to features, training reading data and writing artifacts, an endpoint fetching a provider key — crosses a KMS boundary and often a Secrets Manager lookup. The execution roles you designed decide *who* can act; KMS key policies and secret permissions decide *what they can decrypt and retrieve*. Get this layer right once and encryption becomes invisible plumbing; get it wrong and it surfaces as the `AccessDenied` that stalls a training job at 2 a.m.

## Key takeaways

- KMS manages keys and does encryption via **envelope encryption** (data keys), so one key protects petabytes; use customer-managed keys (CMKs) for auditable, revocable control over ML data.
- A KMS **key policy** is the root of trust — an IAM `kms:Decrypt` grant is useless unless the key policy allows the principal; **grants** delegate key use to a workload temporarily.
- Secrets Manager stores and **automatically rotates** credentials (managed rotation for RDS/Aurora); reading a secret needs both `GetSecretValue` and `kms:Decrypt`.
- Parameter Store `SecureString` is the cheap encrypted store for config and non-rotating secrets; use hierarchical paths and fetch subtrees at startup.
- Cache fetched secrets across warm invocations; enable `BucketKeyEnabled` on KMS-encrypted S3 to slash KMS request cost.

## CLI cheat-sheet

```bash
# --- KMS: keys & aliases ---
aws kms create-key --description "ml key"
aws kms create-alias --alias-name alias/ml-data --target-key-id <key-id>
aws kms enable-key-rotation --key-id alias/ml-data
aws kms list-aliases
aws kms describe-key --key-id alias/ml-data
aws kms schedule-key-deletion --key-id <key-id> --pending-window-in-days 30

# --- KMS: crypto & delegation ---
aws kms generate-data-key --key-id alias/ml-data --key-spec AES_256
aws kms encrypt --key-id alias/ml-data --plaintext fileb://in.bin --query CiphertextBlob --output text
aws kms decrypt --ciphertext-blob fileb://cipher.bin --query Plaintext --output text | base64 -d
aws kms create-grant --key-id alias/ml-data --grantee-principal <role-arn> --operations Decrypt GenerateDataKey
aws kms put-key-policy --key-id <key-id> --policy-name default --policy file://key-policy.json

# --- Secrets Manager ---
aws secretsmanager create-secret --name prod/model-provider --secret-string '{"api_key":"..."}' --kms-key-id alias/ml-data
aws secretsmanager get-secret-value --secret-id prod/model-provider --query SecretString --output text
aws secretsmanager put-secret-value --secret-id prod/model-provider --secret-string '{"api_key":"new"}'
aws secretsmanager rotate-secret --secret-id prod/model-provider \
  --rotation-lambda-arn <arn> --rotation-rules '{"AutomaticallyAfterDays":30}'
aws secretsmanager list-secrets
aws secretsmanager delete-secret --secret-id prod/model-provider --recovery-window-in-days 7

# --- Parameter Store (SSM) ---
aws ssm put-parameter --name /ml/prod/endpoint-name --value fraud-scorer --type String
aws ssm put-parameter --name /ml/prod/db-password --value 's3cr3t' --type SecureString --key-id alias/ml-data
aws ssm get-parameter --name /ml/prod/db-password --with-decryption
aws ssm get-parameters-by-path --path /ml/prod --recursive --with-decryption
aws ssm delete-parameter --name /ml/prod/db-password
```

## Try it

Create a customer-managed KMS key with an alias and enable rotation. Encrypt an S3 bucket with it (SSE-KMS, `BucketKeyEnabled`) and confirm objects written there are encrypted with your key. Store a fake model-provider API key in Secrets Manager encrypted with the same CMK, then write a boto3 snippet that fetches it once and caches it. Now remove `kms:Decrypt` from your role's access to the key (or revoke the grant) and confirm the fetch fails with `AccessDenied` — proving that KMS, not just the secret's own permissions, gates access. Finally, put a non-secret config value in Parameter Store and read a whole `/ml/prod` subtree with `--with-decryption`, seeing config and SecureString side by side.
