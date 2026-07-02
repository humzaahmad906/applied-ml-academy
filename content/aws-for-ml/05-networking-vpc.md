# 05 — Networking: VPC Basics

Networking is the module ML engineers most want to skip and most often get burned by. Your training job fails to reach S3, your endpoint cannot be called from the API, or your inference traffic silently routes through a NAT Gateway that adds a surprise line to the bill — all networking. A Virtual Private Cloud is your own isolated network inside AWS, and understanding its handful of core pieces is enough to make ML workloads both reachable and private. This module keeps VPC concepts practical and tied to how ML systems actually communicate.

## What a VPC is

A **VPC** is a logically isolated virtual network in one Region, defined by an IP address range in CIDR notation (for example `10.0.0.0/16`, giving you 65,536 addresses). Everything you launch — EC2 instances, SageMaker training jobs in VPC mode, endpoints, databases — gets an IP inside a VPC. By default AWS gives you a default VPC so things "just work," but production systems use purpose-built VPCs with deliberate subnet layout.

Pick the CIDR carefully — you cannot shrink it later (you can only add secondary blocks), and it must not overlap with other VPCs you intend to peer with or connect over a transit gateway. A `/16` is the maximum size; `/28` is the minimum. Enabling DNS hostnames and resolution up front avoids a common failure mode where interface endpoints and private hostnames do not resolve:

```bash
aws ec2 create-vpc --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=ml-vpc}]'
aws ec2 modify-vpc-attribute --vpc-id vpc-0abc123 --enable-dns-support '{"Value":true}'
aws ec2 modify-vpc-attribute --vpc-id vpc-0abc123 --enable-dns-hostnames '{"Value":true}'
aws ec2 describe-vpcs --vpc-ids vpc-0abc123 --query 'Vpcs[0].CidrBlock'
```

AWS reserves five addresses in every subnet (the first four and the last), so a `/24` gives you 251 usable IPs, not 256 — a detail that bites when you size subnets tightly for a large training cluster and run out of addresses mid-scale-up.

## Subnets, route tables, gateways

A **subnet** is a slice of the VPC's address range that lives in a single Availability Zone. The public/private distinction is the one that matters:

- A **public subnet** has a route to an **Internet Gateway (IGW)**, so resources in it can have public IPs and talk to the internet directly. Load balancers and bastion hosts live here.
- A **private subnet** has no direct internet route. Resources in it reach the internet only through a **NAT Gateway** sitting in a public subnet. This is where training jobs, endpoints, and databases belong — reachable by your own services but not directly exposed.

A **route table** attached to each subnet decides where traffic goes. The presence or absence of a `0.0.0.0/0` route to an IGW or NAT Gateway is literally what makes a subnet public or private.

Building this by hand is worth doing once so the pieces are concrete. The sequence is: create the VPC, carve subnets in distinct AZs, attach an IGW, and give the public subnet a default route to it. Note that a subnet is only "public" once you both add the IGW route *and* enable auto-assign public IPs on it (`map-public-ip-on-launch`) — either alone is a silent misconfiguration:

```bash
# subnets in two AZs (public + private per AZ is the standard layout)
aws ec2 create-subnet --vpc-id vpc-0abc123 --cidr-block 10.0.1.0/24 --availability-zone us-east-1a
aws ec2 create-subnet --vpc-id vpc-0abc123 --cidr-block 10.0.2.0/24 --availability-zone us-east-1b
aws ec2 modify-subnet-attribute --subnet-id subnet-0pub --map-public-ip-on-launch

# internet gateway → attach → public route table → default route → associate
aws ec2 create-internet-gateway
aws ec2 attach-internet-gateway --internet-gateway-id igw-0abc --vpc-id vpc-0abc123
aws ec2 create-route-table --vpc-id vpc-0abc123
aws ec2 create-route --route-table-id rtb-0pub --destination-cidr-block 0.0.0.0/0 \
  --gateway-id igw-0abc
aws ec2 associate-route-table --route-table-id rtb-0pub --subnet-id subnet-0pub
```

A **NAT Gateway** lets private-subnet resources reach the internet outbound (to pull packages, call external APIs) without being reachable inbound. It lives in a *public* subnet and needs an Elastic IP; the private route table then points its default route at the NAT, not the IGW:

```bash
aws ec2 allocate-address --domain vpc                       # Elastic IP for the NAT
aws ec2 create-nat-gateway --subnet-id subnet-0pub --allocation-id eipalloc-0abc
aws ec2 create-route --route-table-id rtb-0priv --destination-cidr-block 0.0.0.0/0 \
  --nat-gateway-id nat-0abc
aws ec2 associate-route-table --route-table-id rtb-0priv --subnet-id subnet-0priv
```

The NAT Gateway detail bites ML teams specifically: it is billed per hour *and per gigabyte processed*. A large training job pulling terabytes from S3 through a NAT Gateway can rack up meaningful data-processing charges — which is exactly the problem VPC endpoints solve. Two gotchas here: a NAT Gateway is single-AZ, so for real HA you deploy one per AZ and give each private subnet a route to the NAT in *its own* AZ; and the Elastic IP keeps costing money if you delete the NAT but forget to release the address.

## Security groups vs NACLs

Two layers control what traffic is allowed:

- A **security group** is a stateful firewall attached to a resource (an instance, an endpoint). You write *allow* rules; return traffic for an allowed connection is automatically permitted. This is the layer you tune constantly: "allow the API's security group to reach the model endpoint on port 8080."
- A **Network ACL (NACL)** is a stateless firewall at the subnet level, with explicit allow/deny rules for both directions. Most teams leave NACLs permissive and do their real work in security groups.

Reference security groups by ID in each other's rules rather than by IP range — "allow from the app-tier security group" stays correct as instances come and go. In the CLI this means the ingress source is a source-group reference, not a CIDR. Create the group, then author rules; a new security group starts with *no* inbound rules and an allow-all outbound rule:

```bash
aws ec2 create-security-group --group-name model-endpoint --description "model server" \
  --vpc-id vpc-0abc123
# allow the app tier's SG to reach this endpoint on 8080 (SG-to-SG, not CIDR)
aws ec2 authorize-security-group-ingress --group-id sg-0endpoint \
  --protocol tcp --port 8080 --source-group sg-0apptier
# allow HTTPS from a specific CIDR (e.g. a corporate range) — richer form
aws ec2 authorize-security-group-ingress --group-id sg-0endpoint \
  --ip-permissions 'IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=10.0.0.0/16,Description=internal-https}]'
aws ec2 revoke-security-group-ingress --group-id sg-0endpoint \
  --protocol tcp --port 8080 --source-group sg-0apptier   # remove a rule
aws ec2 describe-security-groups --group-ids sg-0endpoint  # audit the ruleset
```

There is a hard default limit of 60 inbound and 60 outbound rules per security group, and 5 security groups per network interface (both adjustable) — worth knowing before you try to encode a large allow-list as individual rules. NACLs, when you do touch them, are numbered ordered rules evaluated low-to-high, and because they are stateless you must open the ephemeral return port range (1024–65535) explicitly:

```bash
aws ec2 create-network-acl --vpc-id vpc-0abc123
aws ec2 create-network-acl-entry --network-acl-id acl-0abc --rule-number 100 \
  --protocol tcp --port-range From=443,To=443 --cidr-block 0.0.0.0/0 \
  --rule-action allow --ingress
```

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
  --security-group-ids sg-0endpoint \
  --private-dns-enabled
```

The two types differ in mechanism and, critically, in cost. A **gateway endpoint** is just a route-table entry — it is free, and it is added to *route tables* (`--route-table-ids`), so a private subnet uses it only if that subnet's route table is in the list. An **interface endpoint** is a real ENI with a private IP in each subnet you name (`--subnet-ids`), guarded by a security group, and it is billed *per hour per AZ plus per GB processed*. That per-GB charge is the reason you route bulk S3 traffic through the free gateway endpoint and reserve interface endpoints for the AWS *control-plane* APIs (SageMaker, ECR, STS, Secrets Manager) that move little data.

`--private-dns-enabled` is what makes an interface endpoint transparent: it overrides the public service DNS name so existing boto3 code resolves the service to the private ENI with no code change. If you forget it, calls still go out over the internet and you have quietly paid for an endpoint that does nothing. The security group on the endpoint must allow inbound 443 from your workload's SG, or connections hang with no obvious error.

To wire a gateway endpoint into more route tables, or to add subnets/SGs to an interface endpoint, modify it in place rather than recreating:

```bash
aws ec2 modify-vpc-endpoint --vpc-endpoint-id vpce-0abc \
  --add-route-table-ids rtb-0priv2          # extend an S3 gateway endpoint
aws ec2 modify-vpc-endpoint --vpc-endpoint-id vpce-0def \
  --add-subnet-ids subnet-0ccc              # add an AZ to an interface endpoint
```

When a private-subnet call to an AWS service hangs or fails, these three `describe` commands localize the fault fast — is the endpoint present and available, does the route exist, does the security group allow 443:

```bash
aws ec2 describe-vpc-endpoints --filters Name=vpc-id,Values=vpc-0abc123 \
  --query 'VpcEndpoints[].{Svc:ServiceName,State:State,Type:VpcEndpointType}'
aws ec2 describe-route-tables --route-table-ids rtb-0priv \
  --query 'RouteTables[0].Routes'
aws ec2 describe-security-groups --group-ids sg-0endpoint --query 'SecurityGroups[0].IpPermissions'
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

## Flow logs and debugging connectivity

When a training job "can't reach S3" or an endpoint call times out, you need to see whether packets are being accepted or rejected, and where. **VPC Flow Logs** capture accepted and rejected traffic at the VPC, subnet, or ENI level and ship it to CloudWatch Logs or S3 — the first thing to enable when connectivity is mysterious. Log `REJECT` traffic to spot a blocking security group or NACL, or `ALL` for full visibility:

```bash
aws ec2 create-flow-logs \
  --resource-type VPC --resource-ids vpc-0abc123 \
  --traffic-type REJECT \
  --log-destination-type cloud-watch-logs \
  --log-destination arn:aws:logs:us-east-1:123456789012:log-group:vpc-flow \
  --deliver-logs-permission-arn arn:aws:iam::123456789012:role/flowlogsRole
```

Flow logs tell you *that* traffic was rejected; **Reachability Analyzer** and a handful of `describe` calls tell you *why*. The debugging loop for a private-subnet ML workload is almost always the same: confirm the route table has the endpoint/NAT route, confirm the security groups on both ends allow the port, and confirm the endpoint is `available`:

```bash
aws ec2 describe-subnets --subnet-ids subnet-0priv \
  --query 'Subnets[0].{AZ:AvailabilityZone,Public:MapPublicIpOnLaunch}'
aws ec2 describe-nat-gateways --filter Name=vpc-id,Values=vpc-0abc123 \
  --query 'NatGateways[].{Id:NatGatewayId,State:State}'
aws ec2 describe-network-interfaces --filters Name=vpc-id,Values=vpc-0abc123 \
  --query 'NetworkInterfaces[].{Id:NetworkInterfaceId,IP:PrivateIpAddress,Desc:Description}'
```

## How this fits the whole ML solution

The VPC is the private backbone every component communicates over. Data ingestion writes across it, training pulls data across it, the endpoint receives requests across it, and monitoring ships metrics across it. Good VPC design — private subnets for workloads, gateway endpoints for S3, interface endpoints for AWS APIs — is what makes the whole system both secure (nothing exposed by accident) and economical (no NAT tax on your training data). Networking decisions made once here ripple through cost and compliance for the life of the platform.

## Key takeaways

- A VPC is your isolated network; subnets are per-AZ slices, made public or private by their route to an Internet Gateway or NAT Gateway.
- Security groups are stateful, resource-level allow-lists (your main tuning knob); NACLs are stateless subnet-level rules.
- NAT Gateways charge per GB — large training reads through them get expensive.
- **VPC endpoints** are the key ML pattern: free S3/DynamoDB gateway endpoints and PrivateLink interface endpoints keep traffic private and cut NAT costs.
- SageMaker VPC mode attaches jobs/endpoints to your subnets but requires you to supply the endpoints they depend on.

## CLI cheat-sheet

```bash
# --- VPC + DNS ---
aws ec2 create-vpc --cidr-block 10.0.0.0/16
aws ec2 modify-vpc-attribute --vpc-id vpc-x --enable-dns-hostnames '{"Value":true}'
aws ec2 modify-vpc-attribute --vpc-id vpc-x --enable-dns-support '{"Value":true}'
aws ec2 describe-vpcs --vpc-ids vpc-x

# --- Subnets ---
aws ec2 create-subnet --vpc-id vpc-x --cidr-block 10.0.1.0/24 --availability-zone us-east-1a
aws ec2 modify-subnet-attribute --subnet-id subnet-x --map-public-ip-on-launch
aws ec2 describe-subnets --filters Name=vpc-id,Values=vpc-x

# --- Internet gateway (public egress) ---
aws ec2 create-internet-gateway
aws ec2 attach-internet-gateway --internet-gateway-id igw-x --vpc-id vpc-x

# --- Route tables ---
aws ec2 create-route-table --vpc-id vpc-x
aws ec2 create-route --route-table-id rtb-x --destination-cidr-block 0.0.0.0/0 --gateway-id igw-x
aws ec2 create-route --route-table-id rtb-x --destination-cidr-block 0.0.0.0/0 --nat-gateway-id nat-x
aws ec2 associate-route-table --route-table-id rtb-x --subnet-id subnet-x
aws ec2 describe-route-tables --route-table-ids rtb-x

# --- NAT gateway (private egress) ---
aws ec2 allocate-address --domain vpc
aws ec2 create-nat-gateway --subnet-id subnet-pub --allocation-id eipalloc-x
aws ec2 describe-nat-gateways --filter Name=vpc-id,Values=vpc-x

# --- Security groups ---
aws ec2 create-security-group --group-name NAME --description DESC --vpc-id vpc-x
aws ec2 authorize-security-group-ingress --group-id sg-x --protocol tcp --port 8080 --source-group sg-y
aws ec2 authorize-security-group-ingress --group-id sg-x --protocol tcp --port 443 --cidr 10.0.0.0/16
aws ec2 revoke-security-group-ingress --group-id sg-x --protocol tcp --port 8080 --source-group sg-y
aws ec2 describe-security-groups --group-ids sg-x

# --- NACLs (stateless, ordered) ---
aws ec2 create-network-acl --vpc-id vpc-x
aws ec2 create-network-acl-entry --network-acl-id acl-x --rule-number 100 \
  --protocol tcp --port-range From=443,To=443 --cidr-block 0.0.0.0/0 --rule-action allow --ingress

# --- VPC endpoints ---
# S3 gateway (free) — attaches to ROUTE TABLES
aws ec2 create-vpc-endpoint --vpc-id vpc-x --vpc-endpoint-type Gateway \
  --service-name com.amazonaws.us-east-1.s3 --route-table-ids rtb-priv
# Interface (PrivateLink) — attaches to SUBNETS + SG, enable private DNS
aws ec2 create-vpc-endpoint --vpc-id vpc-x --vpc-endpoint-type Interface \
  --service-name com.amazonaws.us-east-1.ecr.dkr \
  --subnet-ids subnet-a subnet-b --security-group-ids sg-x --private-dns-enabled
aws ec2 modify-vpc-endpoint --vpc-endpoint-id vpce-x --add-route-table-ids rtb-priv2
aws ec2 modify-vpc-endpoint --vpc-endpoint-id vpce-y --add-subnet-ids subnet-c
aws ec2 describe-vpc-endpoints --filters Name=vpc-id,Values=vpc-x

# --- Flow logs + debugging ---
aws ec2 create-flow-logs --resource-type VPC --resource-ids vpc-x --traffic-type ALL \
  --log-destination-type cloud-watch-logs \
  --log-destination arn:aws:logs:REGION:ACCT:log-group:vpc-flow \
  --deliver-logs-permission-arn arn:aws:iam::ACCT:role/flowlogsRole
aws ec2 describe-network-interfaces --filters Name=vpc-id,Values=vpc-x
```

## Try it

Build a VPC with one public and two private subnets across two AZs. Launch an EC2 instance in a private subnet and confirm it *cannot* reach the internet. Add an S3 gateway endpoint and verify the instance can now `aws s3 ls` your bucket with no NAT Gateway in the path. Then add an interface endpoint for a service like Secrets Manager and confirm a boto3 call succeeds from the private subnet. You have just built the private data plane that production ML platforms run on.
