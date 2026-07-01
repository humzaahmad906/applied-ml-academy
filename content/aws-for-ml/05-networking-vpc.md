# 05 — Networking: VPC Basics

Networking is the module ML engineers most want to skip and most often get burned by. Your training job fails to reach S3, your endpoint cannot be called from the API, or your inference traffic silently routes through a NAT Gateway that adds a surprise line to the bill — all networking. A Virtual Private Cloud is your own isolated network inside AWS, and understanding its handful of core pieces is enough to make ML workloads both reachable and private. This module keeps VPC concepts practical and tied to how ML systems actually communicate.

## What a VPC is

A **VPC** is a logically isolated virtual network in one Region, defined by an IP address range in CIDR notation (for example `10.0.0.0/16`, giving you 65,536 addresses). Everything you launch — EC2 instances, SageMaker training jobs in VPC mode, endpoints, databases — gets an IP inside a VPC. By default AWS gives you a default VPC so things "just work," but production systems use purpose-built VPCs with deliberate subnet layout.

## Subnets, route tables, gateways

A **subnet** is a slice of the VPC's address range that lives in a single Availability Zone. The public/private distinction is the one that matters:

- A **public subnet** has a route to an **Internet Gateway (IGW)**, so resources in it can have public IPs and talk to the internet directly. Load balancers and bastion hosts live here.
- A **private subnet** has no direct internet route. Resources in it reach the internet only through a **NAT Gateway** sitting in a public subnet. This is where training jobs, endpoints, and databases belong — reachable by your own services but not directly exposed.

A **route table** attached to each subnet decides where traffic goes. The presence or absence of a `0.0.0.0/0` route to an IGW or NAT Gateway is literally what makes a subnet public or private.

The NAT Gateway detail bites ML teams specifically: it is billed per hour *and per gigabyte processed*. A large training job pulling terabytes from S3 through a NAT Gateway can rack up meaningful data-processing charges — which is exactly the problem VPC endpoints solve.

## Security groups vs NACLs

Two layers control what traffic is allowed:

- A **security group** is a stateful firewall attached to a resource (an instance, an endpoint). You write *allow* rules; return traffic for an allowed connection is automatically permitted. This is the layer you tune constantly: "allow the API's security group to reach the model endpoint on port 8080."
- A **Network ACL (NACL)** is a stateless firewall at the subnet level, with explicit allow/deny rules for both directions. Most teams leave NACLs permissive and do their real work in security groups.

Reference security groups by ID in each other's rules rather than by IP range — "allow from the app-tier security group" stays correct as instances come and go.

## VPC endpoints: keeping ML traffic private and cheap

This is the single most important networking feature for ML on AWS. A **VPC endpoint** lets resources in a private subnet reach AWS services *without* traversing the internet or a NAT Gateway. Two kinds:

- **Gateway endpoints** for **S3** and **DynamoDB**. They add a route so S3/DynamoDB traffic stays on the AWS network. They are free and eliminate NAT data-processing charges for S3 access — essential for training jobs that read large datasets.
- **Interface endpoints** (powered by **AWS PrivateLink**) for most other services — SageMaker runtime, ECR, Secrets Manager, CloudWatch, and many more. They create a private IP inside your subnet that resolves to the service, so, for example, calls to a SageMaker endpoint never leave your VPC.

For a security- or compliance-sensitive ML platform, the pattern is: private subnets, no internet route for the workload, an S3 gateway endpoint for data, and interface endpoints for the AWS APIs the workload calls. Data reads never touch the public internet, and NAT costs disappear.

```bash
# S3 gateway endpoint — training data reads bypass NAT and stay on the AWS backbone
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-0abc123 \
  --service-name com.amazonaws.us-east-1.s3 \
  --route-table-ids rtb-0def456 \
  --vpc-endpoint-type Gateway

# Interface endpoint so private-subnet code can call the SageMaker runtime privately
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-0abc123 \
  --service-name com.amazonaws.us-east-1.sagemaker.runtime \
  --vpc-endpoint-type Interface \
  --subnet-ids subnet-0aaa subnet-0bbb \
  --security-group-ids sg-0endpoint
```

## VPC mode for SageMaker

By default SageMaker training jobs and endpoints run in AWS-managed networking. In **VPC mode** you attach them to your own private subnets and security groups, which is required when the job must reach private resources (an RDS database, a self-hosted feature store) or when policy demands that no traffic touch the internet. When you enable VPC mode you must also provide the endpoints (S3 gateway, ECR interface, etc.) the job needs, or it will hang trying to pull its container image or read data.

```python
from sagemaker.estimator import Estimator
est = Estimator(
    image_uri=training_image,
    role=role,
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    subnets=["subnet-0aaa", "subnet-0bbb"],
    security_group_ids=["sg-0training"],
    encrypt_inter_container_traffic=True,
)
```

## How this fits the whole ML solution

The VPC is the private backbone every component communicates over. Data ingestion writes across it, training pulls data across it, the endpoint receives requests across it, and monitoring ships metrics across it. Good VPC design — private subnets for workloads, gateway endpoints for S3, interface endpoints for AWS APIs — is what makes the whole system both secure (nothing exposed by accident) and economical (no NAT tax on your training data). Networking decisions made once here ripple through cost and compliance for the life of the platform.

## Key takeaways

- A VPC is your isolated network; subnets are per-AZ slices, made public or private by their route to an Internet Gateway or NAT Gateway.
- Security groups are stateful, resource-level allow-lists (your main tuning knob); NACLs are stateless subnet-level rules.
- NAT Gateways charge per GB — large training reads through them get expensive.
- **VPC endpoints** are the key ML pattern: free S3/DynamoDB gateway endpoints and PrivateLink interface endpoints keep traffic private and cut NAT costs.
- SageMaker VPC mode attaches jobs/endpoints to your subnets but requires you to supply the endpoints they depend on.

## Try it

Build a VPC with one public and two private subnets across two AZs. Launch an EC2 instance in a private subnet and confirm it *cannot* reach the internet. Add an S3 gateway endpoint and verify the instance can now `aws s3 ls` your bucket with no NAT Gateway in the path. Then add an interface endpoint for a service like Secrets Manager and confirm a boto3 call succeeds from the private subnet. You have just built the private data plane that production ML platforms run on.
