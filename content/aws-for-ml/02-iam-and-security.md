# 02 — IAM and Security Foundations

Identity and Access Management is the service that decides who can do what in your account, and it is the single most common cause of "it works on my laptop but fails in the cloud." Every training job, every endpoint, every Lambda function acts *as some identity* with *some set of permissions*. Get IAM wrong and either nothing works or everything is exposed. This module builds a working model of IAM and the security services around it, framed for the way ML workloads actually consume them.

## The four IAM building blocks

**Users** are long-lived identities for humans (or, historically, for scripts). A user can have a console password and/or access keys. In a mature setup you have very few IAM users, because human access is better handled through IAM Identity Center (below).

**Groups** are collections of users that share policies — "data-scientists," "ml-admins." You attach permissions to the group, not to each user.

**Roles** are the identity type that matters most for ML. A role is a set of permissions with no long-lived credentials, meant to be *assumed* temporarily. Services assume roles: a SageMaker training job runs as a **SageMaker execution role**, an EC2 instance runs as an **instance profile**, a Lambda function runs as its **execution role**. When code inside these services calls AWS, it receives short-lived credentials from the role automatically — you never put keys in the code.

**Policies** are JSON documents listing permissions. An **identity-based policy** attaches to a user, group, or role and says what that identity may do. A **resource-based policy** attaches to a resource (an S3 bucket, a KMS key) and says who may touch it. Policies can be AWS-**managed** (maintained by AWS, e.g. `AmazonS3ReadOnlyAccess`) or **customer-managed** (yours, reusable) or **inline** (embedded in one identity, avoid these — they do not reuse).

A policy statement is `Effect` (Allow/Deny), `Action` (like `s3:GetObject`), `Resource` (an ARN), and optional `Condition`:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::my-ml-data",
      "arn:aws:s3:::my-ml-data/*"
    ]
  }]
}
```

## Roles and trust: how a training job gets its permissions

A role has two policies of interest. Its **permissions policy** says what it can do. Its **trust policy** says who can assume it. For a SageMaker execution role, the trust policy names the SageMaker service as the principal allowed to assume it:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "sagemaker.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
```

When you launch a training job and pass this role's ARN, SageMaker calls `sts:AssumeRole`, receives temporary credentials, and your container uses them to read training data from S3 and write the model artifact back. If the job fails with `AccessDenied` reading S3, the fix is almost always in this role's permissions policy, not your code.

```bash
# Create the role from a trust policy, then attach permissions
aws iam create-role --role-name ml-sagemaker-exec \
  --assume-role-policy-document file://trust.json
aws iam attach-role-policy --role-name ml-sagemaker-exec \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

The full lifecycle of a role runs through a handful of commands you will use constantly. You inspect what is attached, create your own reusable customer-managed policy, and — the step people forget — detach every policy before you can delete the role:

```bash
# What can this role do right now?
aws iam list-attached-role-policies --role-name ml-sagemaker-exec   # managed policies
aws iam list-role-policies --role-name ml-sagemaker-exec            # inline policies
aws iam get-role --role-name ml-sagemaker-exec                      # includes the trust policy

# Create a scoped, reusable customer-managed policy and attach it
aws iam create-policy --policy-name ml-s3-scoped \
  --policy-document file://s3-scoped.json
aws iam attach-role-policy --role-name ml-sagemaker-exec \
  --policy-arn arn:aws:iam::111111111111:policy/ml-s3-scoped

# Update the trust policy later (e.g. add a second principal)
aws iam update-assume-role-policy --role-name ml-sagemaker-exec \
  --policy-document file://trust.json

# Tear down: detach first, then delete — deleting a role with attachments fails
aws iam detach-role-policy --role-name ml-sagemaker-exec \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam delete-role --role-name ml-sagemaker-exec
```

Many AWS services need a role they own the wiring for — a **service-linked role** — which you create by naming the service rather than writing a trust policy yourself:

```bash
aws iam create-service-linked-role --aws-service-name sagemaker.amazonaws.com
```

## Least privilege and permission boundaries

**Least privilege** means granting only the actions and resources a workload genuinely needs. `AmazonSageMakerFullAccess` is fine for a sandbox but too broad for production; a real execution role should scope S3 access to specific buckets and prefixes. Start broad while iterating, then tighten by reading CloudTrail to see what was actually used.

You do not have to guess whether a policy grants what you intend — you can **test it before shipping** with the IAM policy simulator, which evaluates a principal's effective permissions (including boundaries and SCPs) against specific actions and resources. This is the single most useful command for verifying least privilege:

```bash
# Would this role be allowed to read one object and denied another?
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::111111111111:role/ml-sagemaker-exec \
  --action-names s3:GetObject s3:PutObject \
  --resource-arns arn:aws:s3:::my-ml-data/train/part-0.parquet

# Dry-run a proposed policy document without attaching it anywhere
aws iam simulate-custom-policy \
  --policy-input-list file://s3-scoped.json \
  --action-names s3:DeleteObject \
  --resource-arns 'arn:aws:s3:::my-ml-data/*'
```

Each result comes back as `allowed` or `implicitDeny`/`explicitDeny`, so an unexpected `implicitDeny` tells you a needed permission is missing before a training job ever fails at 2 a.m.

A **permission boundary** is a guardrail policy that caps the maximum permissions an identity can have, even if someone attaches a broader policy later. Platform teams use boundaries so data scientists can create their own roles without being able to escalate beyond an approved ceiling. You set the boundary at creation time (or later) with `--permissions-boundary`:

```bash
# Create a role whose effective permissions can never exceed the boundary policy
aws iam create-role --role-name ds-self-service \
  --assume-role-policy-document file://trust.json \
  --permissions-boundary arn:aws:iam::111111111111:policy/ds-boundary

# Add or replace a boundary on an existing role
aws iam put-role-permissions-boundary --role-name ds-self-service \
  --permissions-boundary arn:aws:iam::111111111111:policy/ds-boundary
```

To catch resources that are exposed *outside* your account — a bucket or role that trusts an external principal — use **IAM Access Analyzer**. It continuously scans resource-based policies and flags anything reachable from outside a defined trust zone, which is how you find the accidentally-public training bucket before an auditor does:

```bash
aws accessanalyzer create-analyzer --analyzer-name ml-account --type ACCOUNT
aws accessanalyzer list-findings \
  --analyzer-arn arn:aws:access-analyzer:us-east-1:111111111111:analyzer/ml-account
```

## Credentials: prefer temporary, avoid long-lived

Long-lived **access keys** (an ID and secret) are the classic leak vector — committed to Git, baked into a Docker image, pasted in Slack. Inside AWS you almost never need them because roles supply temporary credentials via **STS** (Security Token Service) automatically. For humans, **IAM Identity Center** (the successor to AWS SSO) issues short-lived credentials through a login portal and integrates with the CLI:

```bash
aws configure sso           # one-time setup against your Identity Center
aws sso login --profile ml  # refreshes temporary creds; no static keys stored
```

Under the hood, both SSO and role assumption hand out short-lived credentials from **STS**. When you assume a role directly you get back an access key, secret, and — crucially — a **session token** that must accompany them; the default session lasts one hour but can be extended up to the role's `MaxSessionDuration` (up to 12 hours) via `--duration-seconds`. For cross-account access from the CLI:

```bash
aws sts assume-role \
  --role-arn arn:aws:iam::222222222222:role/ml-cross-account \
  --role-session-name humza --duration-seconds 3600
```

Always enable **MFA** on any identity that can log in, and on the root user unconditionally. When a policy requires MFA (via an `aws:MultiFactorAuthPresent` condition), you exchange your MFA code for a temporary token before the CLI will let you act:

```bash
# Register a virtual MFA device, then trade a code for a short-lived session
aws iam create-virtual-mfa-device --virtual-mfa-device-name humza-phone \
  --outfile qr.png --bootstrap-method QRCodePNG
aws sts get-session-token \
  --serial-number arn:aws:iam::111111111111:mfa/humza-phone \
  --token-code 123456
```

If you must use long-lived access keys (for an external CI system, say), rotate them on a schedule and scope them tightly. The rotation dance is: create a second key, deploy it, then deactivate and delete the old one:

```bash
aws iam list-access-keys --user-name ci-bot
aws iam create-access-key --user-name ci-bot            # deploy this, then...
aws iam update-access-key --user-name ci-bot \
  --access-key-id AKIAOLDKEY --status Inactive          # verify nothing breaks
aws iam delete-access-key --user-name ci-bot --access-key-id AKIAOLDKEY
```

## Secrets and encryption

Application secrets — database passwords, third-party API keys, model-provider tokens — do not belong in environment variables checked into a repo. **AWS Secrets Manager** stores them encrypted, supports automatic rotation, and is fetched at runtime by ARN. **SSM Parameter Store** is a lighter, cheaper option for configuration and non-rotating secrets. Both integrate with **KMS** (Key Management Service), which manages the encryption keys that protect S3 objects, EBS volumes, and secrets. Using a customer-managed KMS key lets you audit and revoke access to encrypted ML data independently of the storage service itself.

```python
import boto3, json
sm = boto3.client("secretsmanager")
secret = json.loads(sm.get_secret_value(SecretId="prod/model-provider")["SecretString"])
```

From the CLI the basic lifecycle is create, read, and (optionally) enable rotation — deeper KMS and rotation-Lambda mechanics live in module 15, but these three commands cover the common case:

```bash
# Create a secret (a JSON blob is the convention for multi-field secrets)
aws secretsmanager create-secret --name prod/model-provider \
  --secret-string '{"api_key":"sk-...","org":"acme"}'

# Read it back at runtime by name or ARN
aws secretsmanager get-secret-value --secret-id prod/model-provider \
  --query SecretString --output text

# Turn on managed rotation against a rotation Lambda
aws secretsmanager rotate-secret --secret-id prod/model-provider \
  --rotation-lambda-arn arn:aws:lambda:us-east-1:111111111111:function:rotate-provider \
  --rotation-rules AutomaticallyAfterDays=30
```

For plain configuration and non-rotating values, **SSM Parameter Store** is the cheaper path; use `SecureString` to have KMS encrypt the value at rest:

```bash
aws ssm put-parameter --name /ml/model-endpoint --type String --value my-endpoint
aws ssm put-parameter --name /ml/db-password --type SecureString --value 's3cr3t'
aws ssm get-parameter --name /ml/db-password --with-decryption \
  --query Parameter.Value --output text
```

## How this fits the whole ML solution

Security is not a module you bolt on at the end — it is the connective tissue of the whole system. The training job's execution role, the endpoint's role, the Lambda's role, the CI pipeline's role, and the humans' Identity Center permissions together define the trust graph of your entire ML platform. Every VPC decision, every S3 bucket policy, and every KMS key in later modules is really an IAM decision in disguise. Design roles per workload with least privilege from the start and the rest of the architecture stays auditable.

## Key takeaways

- Users/groups are for humans; **roles** are how services (SageMaker, EC2, Lambda) get permissions — no keys in code.
- A role's **trust policy** says who can assume it; its **permissions policy** says what it can do.
- Prefer temporary credentials via STS and IAM Identity Center over long-lived access keys; always enable MFA.
- Apply least privilege and use permission boundaries to cap escalation.
- Store secrets in Secrets Manager or Parameter Store, encrypt data with KMS, and audit access with CloudTrail.

## CLI cheat-sheet

```bash
# --- Roles: create, inspect, tear down ---
aws iam create-role --role-name ml-exec --assume-role-policy-document file://trust.json
aws iam get-role --role-name ml-exec                       # includes trust policy
aws iam list-attached-role-policies --role-name ml-exec    # managed policies
aws iam list-role-policies --role-name ml-exec             # inline policies
aws iam update-assume-role-policy --role-name ml-exec --policy-document file://trust.json
aws iam create-service-linked-role --aws-service-name sagemaker.amazonaws.com
aws iam detach-role-policy --role-name ml-exec --policy-arn <arn>   # before delete
aws iam delete-role --role-name ml-exec

# --- Policies: managed + attach ---
aws iam create-policy --policy-name ml-s3-scoped --policy-document file://s3-scoped.json
aws iam attach-role-policy --role-name ml-exec --policy-arn <arn>
aws iam list-policies --scope Local --output table          # your customer-managed policies

# --- Testing least privilege (do this before shipping) ---
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::111111111111:role/ml-exec \
  --action-names s3:GetObject s3:PutObject \
  --resource-arns 'arn:aws:s3:::my-ml-data/*'
aws iam simulate-custom-policy --policy-input-list file://s3-scoped.json \
  --action-names s3:DeleteObject --resource-arns 'arn:aws:s3:::my-ml-data/*'

# --- Permission boundaries ---
aws iam create-role --role-name ds-self-service \
  --assume-role-policy-document file://trust.json \
  --permissions-boundary arn:aws:iam::111111111111:policy/ds-boundary
aws iam put-role-permissions-boundary --role-name ds-self-service \
  --permissions-boundary arn:aws:iam::111111111111:policy/ds-boundary

# --- Temporary credentials & MFA ---
aws sts assume-role --role-arn <arn> --role-session-name s --duration-seconds 3600
aws sts get-session-token --serial-number <mfa-arn> --token-code 123456
aws iam create-virtual-mfa-device --virtual-mfa-device-name phone \
  --outfile qr.png --bootstrap-method QRCodePNG

# --- IAM Identity Center (SSO) ---
aws configure sso
aws sso login --profile ml

# --- Access key rotation (avoid long-lived keys where possible) ---
aws iam list-access-keys --user-name ci-bot
aws iam create-access-key --user-name ci-bot
aws iam update-access-key --user-name ci-bot --access-key-id <id> --status Inactive
aws iam delete-access-key --user-name ci-bot --access-key-id <id>

# --- Access Analyzer (find externally-exposed resources) ---
aws accessanalyzer create-analyzer --analyzer-name ml-account --type ACCOUNT
aws accessanalyzer list-findings --analyzer-arn <analyzer-arn>

# --- Secrets & parameters (overview; deep dive in module 15) ---
aws secretsmanager create-secret --name prod/key --secret-string '{"api_key":"..."}'
aws secretsmanager get-secret-value --secret-id prod/key --query SecretString --output text
aws secretsmanager rotate-secret --secret-id prod/key \
  --rotation-lambda-arn <arn> --rotation-rules AutomaticallyAfterDays=30
aws ssm put-parameter --name /ml/db-password --type SecureString --value 's3cr3t'
aws ssm get-parameter --name /ml/db-password --with-decryption --query Parameter.Value --output text
```

## Try it

Create a SageMaker execution role with a correct trust policy and a *scoped* permissions policy that allows read/write only to a single S3 bucket and prefix (not `AmazonSageMakerFullAccess`). Store a fake "model provider API key" in Secrets Manager. Then write a short boto3 snippet that assumes the role's perspective conceptually by fetching the secret and listing only the allowed prefix — and confirm that listing a *different* bucket returns `AccessDenied`. That denial is least privilege working as intended.
