# 12b — Terraform for AWS

Lesson 12 built the whole ML platform as code with the **CDK** and **CloudFormation** — the AWS-native path. But walk into most ML infrastructure teams and the tool you will actually find checked into the repo is **Terraform**. It is the platform-engineering default: multi-cloud, provider-driven, with the largest module ecosystem in infrastructure-as-code. This lesson gives Terraform the same treatment lesson 12 gave the CDK — the core loop, state, modules, and a realistic ML stack — and maps every piece back to the architecture you already drew. If you can read a CloudFormation template, HCL will feel familiar within an hour.

## Why Terraform, and how it differs from CDK/CloudFormation

CloudFormation and the CDK only speak AWS. Terraform speaks to **AWS, Azure, GCP, and 4,000+ other providers** — Datadog, GitHub, Snowflake, Kubernetes, Cloudflare — through one declarative language and one workflow. For an ML team whose stack is rarely pure AWS (a model on SageMaker, secrets in Vault, dashboards in Datadog, a DNS record at Cloudflare), that single pane is the whole pitch. It is also why Terraform dominates the platform-engineering job market: the skill transfers across clouds.

Three concrete differences from lesson 12's tools matter:

- **Declarative HCL, not a general-purpose language.** The CDK is Python/TypeScript that *synthesizes* a CloudFormation template — you get loops, classes, and `if` statements. Terraform's **HCL** is a purpose-built configuration language: you declare the resources you want and Terraform figures out the order from their dependencies. There is less programming and more describing. HCL has expressions, `for_each`, and conditionals, but you are configuring, not coding.
- **The provider model, and direct API calls.** A CDK/CloudFormation `deploy` hands a template to AWS's CloudFormation *engine*, which provisions on your behalf. Terraform calls the **AWS APIs directly** through the `aws` provider — a plugin it downloads at `init`. That makes provisioning noticeably faster (a 10–20 resource stack lands in ~1–3 minutes versus ~3–5 for CloudFormation) and puts error handling in Terraform's hands rather than CloudFormation's rollback machinery. The tradeoff: you debug provider errors, not stack events.
- **State is yours to manage.** This is the big one. CloudFormation tracks what it deployed inside AWS, invisibly. Terraform records reality in a **state file** that *you* store and protect. It is more moving parts, but it is also why `terraform plan` can show you an exact, resource-level diff before you touch anything.

## The core loop: providers, resources, plan, apply

Every Terraform config starts by pinning the tool and declaring a provider. Pin versions tightly — an unpinned provider upgrade reaching prod is a classic self-inflicted outage.

```hcl
terraform {
  required_version = ">= 1.11"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"          # allow 6.x patches, block a surprise 7.0
    }
  }
}

provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = {                      # cost + ownership tags on every resource
      Project   = "ml-platform"
      ManagedBy = "terraform"
      Env       = "dev"
    }
  }
}
```

> Version note: the AWS provider was on the v6.x line in 2026 — confirm the current major with `terraform init` output and pin to what you test against. The `default_tags` block is the Terraform way to tag *everything* at once, matching lesson 12's "cost tags on everything" rule.

A **resource** block is the unit of infrastructure. Here is a real one — the data bucket and SageMaker execution role from lesson 12's stack, in HCL:

```hcl
resource "aws_s3_bucket" "ml_data" {
  bucket = "ml-platform-data-dev"
}

resource "aws_s3_bucket_versioning" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id           # reference = implicit dependency
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "ml_data" {
  bucket                  = aws_s3_bucket.ml_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_role" "sm_exec" {
  name = "sagemaker-exec-dev"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Least-privilege: read/write scoped to THIS bucket, not s3:* on *
resource "aws_iam_role_policy" "sm_data_access" {
  name = "sm-data-access"
  role = aws_iam_role.sm_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = [aws_s3_bucket.ml_data.arn, "${aws_s3_bucket.ml_data.arn}/*"]
    }]
  })
}
```

Notice there is no explicit "create the bucket before the role" ordering. When the IAM policy references `aws_s3_bucket.ml_data.arn`, Terraform infers the dependency and builds a graph. That graph is what the four core commands drive:

```bash
terraform init      # download the aws provider + configure the backend (once per clone)
terraform plan      # dry run: show exactly what will add / change / destroy
terraform apply     # execute the plan after you confirm (type "yes")
terraform destroy   # tear the whole environment down
```

`plan` is the safety rail lesson 12 got from `cdk diff` — read it every time, and treat a `-/+ destroy and then create` on a stateful resource (a bucket, an endpoint, an RDS instance) as a stop sign. `destroy` is the equivalent of `cdk destroy`: one command tears an entire environment down, which is exactly what you want for ephemeral dev/PR stacks.

## State: the file that matters most

The **state file** (`terraform.tfstate`) is Terraform's record of which real AWS resources map to which blocks in your config. `plan` works by diffing your config against state against the live API. Two rules follow, and both bite teams that ignore them:

**Never commit state to git, and never edit it by hand.** State contains resource IDs and, for some resources, **secrets in plaintext** — a database password, a generated access key. A committed `tfstate` is a credential leak. Add `*.tfstate*` to `.gitignore` on day one.

**Use remote state for any team.** A local state file means only one laptop knows the truth, and two engineers applying at once corrupt it. The standard fix on AWS is the **S3 backend**: state lives in a versioned, encrypted S3 bucket, and a **lock** prevents concurrent `apply`s.

```hcl
terraform {
  backend "s3" {
    bucket       = "ml-platform-tfstate"
    key          = "platform/dev/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true          # SSE on the state object at rest
    use_lockfile = true          # native S3 state locking (see note)
  }
}
```

Locking guidance changed recently and is worth getting right. For years the pattern was **S3 + a DynamoDB table** for the lock. As of Terraform **1.11**, S3 gained **native state locking** via `use_lockfile = true` — Terraform writes a small `.tflock` object next to your state using a conditional (atomic) put, so no DynamoDB table is needed. The old `dynamodb_table` argument still works but is **deprecated** and slated for removal in a future minor version. For new stacks on Terraform ≥ 1.11 (or OpenTofu ≥ 1.10), prefer `use_lockfile = true`; you can set both during a migration. Exact version behavior evolves — **check the current [S3 backend docs](https://developer.hashicorp.com/terraform/language/backend/s3) before relying on the DynamoDB-removal timeline.**

> Bootstrapping note: the state bucket itself has to exist before the backend can use it. Teams either create it once by hand (or a tiny separate Terraform config with local state) and never touch it again.

## Variables, outputs, and modules

Hard-coded values do not survive contact with dev/staging/prod. **Variables** parameterize a config; **outputs** expose values for humans or other stacks; **modules** package a set of resources for reuse — the equivalent of a CDK construct.

```hcl
variable "env" {
  type    = string
  default = "dev"
}

variable "instance_type" {
  type    = string
  default = "ml.m5.large"
}

output "data_bucket_arn" {
  value = aws_s3_bucket.ml_data.arn
}
```

A **module** is just a directory of `.tf` files you call from elsewhere, passing variables in and reading outputs back:

```hcl
module "training_bucket" {
  source        = "./modules/ml-bucket"   # local, a registry, or a git URL
  env           = var.env
  bucket_prefix = "ml-platform-train"
}
```

The **Terraform Registry** hosts thousands of published modules; for AWS, prefer ones from verified authors (AWS, HashiCorp partners) and pin their versions the same way you pin the provider. The same config, driven by different `.tfvars` files (`dev.tfvars`, `prod.tfvars`), stamps out three environments from one codebase — Terraform's answer to CloudFormation's `--parameter-overrides`.

## A realistic ML slice: ECR + role + endpoint

Lesson 12's principle "one artifact, many stages" starts with a container in **ECR** that both training and serving pull. Here is that slice in Terraform — the repo, a serving-scoped permission, and a SageMaker model/endpoint — showing how the request path's backend is provisioned:

```hcl
resource "aws_ecr_repository" "model" {
  name                 = "fraud-scorer"
  image_tag_mutability = "IMMUTABLE"                # pinned tags, no silent overwrite
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_iam_role_policy" "sm_ecr_pull" {
  name = "sm-ecr-pull"
  role = aws_iam_role.sm_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
                  "ecr:GetAuthorizationToken"]
      Resource = "*"                                # GetAuthorizationToken requires *
    }]
  })
}

resource "aws_sagemaker_model" "fraud" {
  name               = "fraud-scorer-${var.env}"
  execution_role_arn = aws_iam_role.sm_exec.arn
  primary_container {
    image = "${aws_ecr_repository.model.repository_url}:2026-07-01"
  }
}

resource "aws_sagemaker_endpoint_configuration" "fraud" {
  name = "fraud-scorer-${var.env}"
  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.fraud.name
    initial_instance_count = 1
    instance_type          = var.instance_type
  }
}

resource "aws_sagemaker_endpoint" "fraud" {
  name                 = "fraud-scorer-${var.env}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.fraud.name
}
```

> Cost note: a real-time endpoint bills for the instance **for as long as it exists** — an `ml.m5.large` runs around the clock whether or not it serves traffic. In dev, `terraform destroy` between sessions is the cheapest habit you can build; in prod, right-size the variant and consider serverless (matching lesson 12's "right inference option per workload").

## OpenTofu: the fork in the road

You will hit this the moment you `terraform --version` on a new team. In 2023, HashiCorp relicensed Terraform from the open-source **MPL 2.0** to the source-available **Business Source License (BSL)**; the community forked the last MPL release into **OpenTofu**, now governed by the Linux Foundation. IBM acquired HashiCorp in 2025, which sharpened questions about stewardship but did not change the license.

For a *reader of this course*, the day-to-day is identical: OpenTofu uses the **same HCL and the same state format**, the CLI is `tofu` instead of `terraform`, and it reads existing Terraform state, so migration is a drop-in. The forks have since diverged — OpenTofu shipped **client-side state encryption** and provider-defined functions ahead of (or without) Terraform's open CLI. A pragmatic rule teams use in 2026: **choose OpenTofu for greenfield, self-managed setups** where open licensing, no telemetry, and state encryption matter; **stay on Terraform** if you depend on HCP Terraform, Stacks, Sentinel policy-as-code, or an enterprise SLA. The concepts in this lesson apply verbatim to either.

## How this maps to the lesson-12 architecture

Nothing about the reference architecture changes — only the authoring tool. The **S3 data lake**, **SageMaker Feature Store**, the **pipeline → registry → endpoint** chain, the **API Gateway + Lambda** front door, the **VPC endpoints, KMS, IAM roles, and tags** are all provisioned by `aws_*` resources and, at scale, by modules (`module "vpc"`, `module "feature_store"`, `module "serving"`). What Terraform adds over the CDK path is the state file — the artifact that makes `plan` a precise, resource-level preview — and the provider model, which lets the *same* config reach past AWS to the Datadog monitor, the Cloudflare DNS record, or the GitHub repo that a real ML platform also depends on. Draw the five planes once; provision them with whichever IaC tool your team standardized on.

## Key takeaways

- Terraform is the platform-team default: multi-cloud, 4,000+ providers, declarative **HCL**, and a provider model that calls AWS APIs directly (faster than CloudFormation's engine).
- The core loop is `init` → `plan` → `apply` → `destroy`; always read `plan` and stop on a destroy/recreate of a stateful resource.
- **State** is yours to manage — never commit `tfstate` (it holds secrets), and use the encrypted **S3 backend** with **native `use_lockfile` locking** (Terraform ≥ 1.11) instead of the now-deprecated DynamoDB table; verify version specifics against current docs.
- **Variables, outputs, and modules** parameterize and package infra for dev/staging/prod reuse — modules are Terraform's constructs; pin their versions like the provider.
- **OpenTofu** is the MPL-licensed fork with identical HCL/state and extra features (state encryption) — pick it for greenfield self-managed work, stay on Terraform for HCP/Sentinel/enterprise; the concepts transfer either way.

## Try it

Provision lesson 12's thin vertical slice in Terraform instead of the CDK. Write a config with the `aws` provider (pinned), an **S3 backend** using `use_lockfile = true`, an encrypted versioned data bucket, a least-privilege SageMaker execution role scoped to that bucket, an ECR repo, and a SageMaker model + endpoint config + endpoint. Parameterize the environment with a `var.env` and a `dev.tfvars` file, and add an `output` for the endpoint name. Run `terraform init`, read the `terraform plan` output line by line, `apply`, invoke the endpoint, then `terraform destroy` to prove you can stand the environment up and tear it down as code. For extra credit, `tofu init` the same directory and confirm OpenTofu reads the identical state.
