# 01 — AWS Overview, Console, and CLI

Amazon Web Services is the substrate almost every production ML system now runs on. As an ML engineer you rarely get to stay inside a notebook: your model needs data that lives in object storage, GPUs that must be requested and paid for by the second, an endpoint that other services call, and a security boundary that keeps all of it from leaking. This module builds the mental model of AWS you need before any of the ML-specific services make sense, and it gets you productive with the two interfaces you will live in — the Console and the CLI.

## The shape of AWS: Regions and Availability Zones

AWS is a global collection of data centers organized into **Regions**. A Region is a named geographic area such as `us-east-1` (N. Virginia), `us-west-2` (Oregon), or `eu-west-1` (Ireland). Regions are isolated from one another by design: an S3 bucket, a training job, or an endpoint lives in exactly one Region, and data does not silently cross Region boundaries. This isolation is what lets you reason about data residency and blast radius.

Inside each Region are multiple **Availability Zones** (AZs), typically three or more. An AZ is one or more physically separate data centers with independent power and networking, connected to the other AZs in the Region by low-latency links. The pattern you will use constantly: spread stateless components across AZs for resilience, and keep latency-sensitive chatter (for example, a distributed training cluster) inside a single AZ or a placement group.

Region choice matters more for ML than for typical web apps because **GPU and accelerator capacity is not uniform**. The newest instances — the NVIDIA Blackwell-based P6-B200 and the P6e-GB200 UltraServers, along with Trainium2 — land first in a handful of Regions and are often gated behind capacity reservations. Before you design a training strategy, confirm the instance family you want actually exists in your Region.

## The AWS account and the shared responsibility model

Everything you do sits inside an **AWS account**, a billing and isolation boundary identified by a 12-digit account ID. Serious organizations run many accounts (dev, staging, prod, a sandbox per team) under **AWS Organizations**, which gives consolidated billing and org-wide guardrails called Service Control Policies. Even as a solo learner, treat the root user of your account as radioactive: use it only to create your first administrative identity, enable MFA on it, and then never log in as root again.

The **shared responsibility model** draws the line between what AWS secures and what you secure. AWS is responsible for security *of* the cloud — the physical facilities, the hypervisor, the managed service infrastructure. You are responsible for security *in* the cloud — your data, who can access it, how you configure networking, and whether your S3 bucket is public. For ML this line is sharp: AWS keeps the SageMaker training fleet patched, but the IAM role your job assumes, the encryption on your training data, and the VPC it runs in are entirely yours.

## The Console

The **AWS Management Console** is the web UI at the heart of exploration. It is organized by service, with a global Region selector in the top-right that scopes almost everything you see — a common early confusion is "my bucket disappeared" when really you switched Regions. The Console is excellent for learning a service, inspecting state, and one-off actions. It is a poor fit for anything you need to reproduce, because clicks are not version-controlled. The professional pattern is: explore in the Console, then codify in the CLI or infrastructure-as-code once you understand what you are building.

## The AWS CLI

The **AWS CLI** (version 2 is current) is how you script AWS from a terminal. Install it, then configure a profile:

```bash
aws configure --profile ml
# AWS Access Key ID: ...
# AWS Secret Access Key: ...
# Default region name: us-east-1
# Default output format: json

# Verify who you are — the fastest sanity check in all of AWS
aws sts get-caller-identity --profile ml
```

`sts get-caller-identity` returns the account ID and the ARN of the identity you are acting as. Run it whenever a command fails with an access error; nine times out of ten you are in the wrong account or profile. Every service has a CLI namespace:

```bash
# List your buckets
aws s3 ls --profile ml

# List available GPU instance types offered in the Region
aws ec2 describe-instance-type-offerings \
  --filters Name=instance-type,Values='p5.*','g6.*','trn2.*' \
  --region us-east-1 --profile ml --output table
```

## boto3: the CLI's Python twin

Everything the CLI can do, the **boto3** SDK does from Python — and boto3 is what your ML code will actually call in production. The two share the same authentication and the same API surface.

```python
import boto3

session = boto3.Session(profile_name="ml", region_name="us-east-1")
sts = session.client("sts")
print(sts.get_caller_identity()["Arn"])

# Every AWS service is a client (low-level) or resource (high-level)
s3 = session.client("s3")
for b in s3.list_buckets()["Buckets"]:
    print(b["Name"])
```

Keep a single `Session` per process and hand its clients to your components. This makes credentials, Region, and retries consistent across your whole application.

## How this fits the whole ML solution

A complete ML system on AWS is not one service — it is a graph of many: object storage feeding a training fleet, a model registry gating a deployment, an endpoint behind an API, and monitoring watching all of it. Regions and accounts define *where* that graph lives and *who can touch it*; the Console and CLI/SDK are how you build and operate it. Every later module snaps another node into this graph. Getting the account, Region discipline, and CLI muscle memory right now is what keeps the whole thing coherent later.

## Key takeaways

- AWS is Regions (isolated geographies) made of Availability Zones (isolated data centers); ML capacity, especially GPUs, varies by Region.
- The account is your billing and isolation boundary; secure the root user, then work through named identities.
- Under the shared responsibility model, AWS secures the cloud; you secure your data, access, and configuration.
- The Console is for exploring and inspecting; the CLI and boto3 are for anything reproducible or programmatic.
- `aws sts get-caller-identity` is your first debugging step for any access problem.

## Try it

Create a fresh AWS account, enable MFA on the root user, and create one IAM administrator identity for yourself. Install the AWS CLI v2, configure a named profile, and run `aws sts get-caller-identity`. Then, in three different Regions, run `aws ec2 describe-instance-type-offerings` filtered to GPU families (`p5.*`, `g6.*`, `trn2.*`) and note which accelerator instances are actually available where you are — this is the first real constraint you will design around.
