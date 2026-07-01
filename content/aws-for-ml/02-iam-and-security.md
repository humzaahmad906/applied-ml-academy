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

## Least privilege and permission boundaries

**Least privilege** means granting only the actions and resources a workload genuinely needs. `AmazonSageMakerFullAccess` is fine for a sandbox but too broad for production; a real execution role should scope S3 access to specific buckets and prefixes. Start broad while iterating, then tighten by reading CloudTrail to see what was actually used.

A **permission boundary** is a guardrail policy that caps the maximum permissions an identity can have, even if someone attaches a broader policy later. Platform teams use boundaries so data scientists can create their own roles without being able to escalate beyond an approved ceiling.

## Credentials: prefer temporary, avoid long-lived

Long-lived **access keys** (an ID and secret) are the classic leak vector — committed to Git, baked into a Docker image, pasted in Slack. Inside AWS you almost never need them because roles supply temporary credentials via **STS** (Security Token Service) automatically. For humans, **IAM Identity Center** (the successor to AWS SSO) issues short-lived credentials through a login portal and integrates with the CLI:

```bash
aws configure sso           # one-time setup against your Identity Center
aws sso login --profile ml  # refreshes temporary creds; no static keys stored
```

Always enable **MFA** on any identity that can log in, and on the root user unconditionally. If you must use access keys (for an external CI system, say), rotate them on a schedule and scope them tightly.

## Secrets and encryption

Application secrets — database passwords, third-party API keys, model-provider tokens — do not belong in environment variables checked into a repo. **AWS Secrets Manager** stores them encrypted, supports automatic rotation, and is fetched at runtime by ARN. **SSM Parameter Store** is a lighter, cheaper option for configuration and non-rotating secrets. Both integrate with **KMS** (Key Management Service), which manages the encryption keys that protect S3 objects, EBS volumes, and secrets. Using a customer-managed KMS key lets you audit and revoke access to encrypted ML data independently of the storage service itself.

```python
import boto3, json
sm = boto3.client("secretsmanager")
secret = json.loads(sm.get_secret_value(SecretId="prod/model-provider")["SecretString"])
```

## How this fits the whole ML solution

Security is not a module you bolt on at the end — it is the connective tissue of the whole system. The training job's execution role, the endpoint's role, the Lambda's role, the CI pipeline's role, and the humans' Identity Center permissions together define the trust graph of your entire ML platform. Every VPC decision, every S3 bucket policy, and every KMS key in later modules is really an IAM decision in disguise. Design roles per workload with least privilege from the start and the rest of the architecture stays auditable.

## Key takeaways

- Users/groups are for humans; **roles** are how services (SageMaker, EC2, Lambda) get permissions — no keys in code.
- A role's **trust policy** says who can assume it; its **permissions policy** says what it can do.
- Prefer temporary credentials via STS and IAM Identity Center over long-lived access keys; always enable MFA.
- Apply least privilege and use permission boundaries to cap escalation.
- Store secrets in Secrets Manager or Parameter Store, encrypt data with KMS, and audit access with CloudTrail.

## Try it

Create a SageMaker execution role with a correct trust policy and a *scoped* permissions policy that allows read/write only to a single S3 bucket and prefix (not `AmazonSageMakerFullAccess`). Store a fake "model provider API key" in Secrets Manager. Then write a short boto3 snippet that assumes the role's perspective conceptually by fetching the secret and listing only the allowed prefix — and confirm that listing a *different* bucket returns `AccessDenied`. That denial is least privilege working as intended.
