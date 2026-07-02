# 05 — Networking: VPC Basics

Networking is the part of Google Cloud that ML engineers most love to ignore — right up until a training VM can't reach the internet to `pip install`, a private endpoint times out, or a security review blocks a launch because data can egress to anywhere. You do not need to become a network engineer, but you do need a working model of the Virtual Private Cloud (VPC) and the handful of controls that keep an ML system connected, private, and compliant.

## What a VPC is

A **VPC network** is a global, private, software-defined network inside your project. Unlike other clouds, a Google Cloud VPC is **global**: a single VPC spans all regions. Inside it you carve out **subnets**, and here is the key fact — **subnets are regional**. A subnet is an IP range (CIDR block, for example `10.0.0.0/20`) tied to one region. VMs, GKE nodes, and many managed services draw their internal IP addresses from a subnet.

Every new project comes with a **default network** that has an auto-created subnet in every region and permissive default firewall rules. It is fine for learning, but for anything real you create a **custom-mode VPC** where you define subnets explicitly and control the address plan. This matters for ML because your training cluster, feature store, database, and serving layer all need predictable, non-overlapping IP space — especially once you connect to on-prem systems or peer with other networks.

```bash
# Create a custom-mode VPC and a subnet for ML workloads
gcloud compute networks create ml-vpc --subnet-mode=custom

gcloud compute networks subnets create ml-train-us \
  --network=ml-vpc \
  --region=us-central1 \
  --range=10.10.0.0/20 \
  --enable-private-ip-google-access
```

That `--enable-private-ip-google-access` flag is important and covered below.

Day to day you list and inspect networks and subnets far more than you create them, and you occasionally need to grow a subnet or attach secondary ranges. GKE in particular needs **secondary ranges** — one for Pods and one for Services — because it assigns Pod and Service IPs from those alias ranges rather than the primary subnet range. You attach them at creation with `--secondary-range NAME=CIDR` (or later with `subnets update --add-secondary-ranges`), and you can grow a too-small primary range in place with `subnets expand-ip-range`.

```bash
# Inspect the address plan you already have
gcloud compute networks list
gcloud compute networks describe ml-vpc
gcloud compute networks subnets list --filter="network:ml-vpc"

# Subnet with secondary ranges sized for a GKE cluster (pods + services)
gcloud compute networks subnets create ml-gke-us \
  --network=ml-vpc --region=us-central1 --range=10.20.0.0/20 \
  --secondary-range=pods=10.21.0.0/16,services=10.22.0.0/20 \
  --enable-private-ip-google-access

# Grow a primary range that ran out of IPs (can only expand, never shrink)
gcloud compute networks subnets expand-ip-range ml-train-us \
  --region=us-central1 --prefix-length=19
```

## Firewall rules and firewall policies

By default a VPC denies all ingress and allows all egress. **Firewall rules** open specific paths. Each rule has a direction (ingress/egress), an action (allow/deny), a priority, protocols/ports, and source/target selectors. The cleanest way to target rules is with **network tags** or **service accounts** rather than raw IP ranges — you tag a VM and write rules against the tag.

```bash
# Allow SSH only from your corporate IP range, to VMs tagged "ssh-ok"
gcloud compute firewall-rules create allow-ssh \
  --network=ml-vpc \
  --direction=INGRESS --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=203.0.113.0/24 \
  --target-tags=ssh-ok

# Allow internal traffic between nodes in the subnet (e.g. multi-worker training)
gcloud compute firewall-rules create allow-internal \
  --network=ml-vpc \
  --direction=INGRESS --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=10.10.0.0/20
```

That second rule is exactly what distributed training needs: worker nodes in a Vertex AI or Compute Engine cluster must reach each other on arbitrary ports for the collective-communication traffic (NCCL over TCP) that DDP and reduction servers rely on.

You list, edit, and delete rules constantly, and you almost always want **logging** on the rules that matter so you can see what a rule actually allowed or denied during an incident (it writes connection records to Cloud Logging — leave it off high-volume internal rules to control log cost).

```bash
gcloud compute firewall-rules list --filter="network:ml-vpc"
gcloud compute firewall-rules update allow-ssh --enable-logging
gcloud compute firewall-rules update allow-ssh --source-ranges=203.0.113.0/24,198.51.100.0/24
gcloud compute firewall-rules delete allow-ssh
```

Beyond individual rules, Google Cloud offers **network firewall policies** — hierarchical, reusable rule sets you can attach at the organization, folder, or network level. These are the modern, scalable way to manage firewalls across many projects (a security team defines a baseline policy once and it applies everywhere), and they coexist with per-network rules. You create a policy, add rules to it, then associate it with a network:

```bash
gcloud compute network-firewall-policies create ml-baseline \
  --global --description="org-wide ML baseline"

gcloud compute network-firewall-policies rules create 1000 \
  --firewall-policy=ml-baseline --global-firewall-policy \
  --direction=INGRESS --action=allow \
  --layer4-configs=tcp:22 --src-ip-ranges=203.0.113.0/24 \
  --enable-logging

gcloud compute network-firewall-policies associations create \
  --firewall-policy=ml-baseline --network=ml-vpc \
  --name=ml-vpc-assoc --global-firewall-policy
```

## Private access: keeping ML traffic off the public internet

By default a VM with no external IP cannot reach Google APIs (Cloud Storage, BigQuery, Vertex AI) — which is a problem, because best practice for training and serving nodes is to give them **no external IP** to shrink their attack surface. Three features solve this:

- **Private Google Access** — enable it on a subnet (the flag shown earlier) and VMs without external IPs can still reach Google's public API endpoints over Google's internal network. This is the single most common thing you must turn on for a locked-down training VM to `import` data from a bucket.
- **Private Service Connect (PSC)** — creates a private endpoint inside your VPC for a Google-managed or partner service. On the ML side, PSC is how you reach a Vertex AI **online prediction endpoint** privately: prediction traffic flows over your VPC's internal IP space, never touching the public internet. It is the standard for latency-sensitive, security-conscious serving.
- **Cloud NAT** — if a private VM needs *outbound* internet access (to `pip install` from PyPI, pull a public container, or hit a third-party API) without having an external IP, Cloud NAT provides managed, egress-only network address translation.

Keep the three private-access mechanisms straight, because they solve different problems and reviewers conflate them. **Private Google Access (PGA)** is a subnet toggle that only reaches *Google's own* public API endpoints. **Private Service Connect (PSC)** creates a private endpoint in your VPC for a specific published service (a Vertex endpoint, a partner service). **Private Service Access (PSA)** is a VPC-peering-based reserved range used by managed services like Cloud SQL and older Vertex configurations. PGA needs no IP allocation; PSC and PSA both consume address space you must plan for.

```bash
# Give private VMs outbound internet without external IPs
gcloud compute routers create ml-router \
  --network=ml-vpc --region=us-central1

gcloud compute routers nats create ml-nat \
  --router=ml-router --region=us-central1 \
  --nat-all-subnet-ip-ranges \
  --auto-allocate-nat-external-ips \
  --min-ports-per-vm=128 \
  --enable-logging --log-filter=ERRORS_ONLY
```

Two Cloud NAT knobs matter at ML scale. `--min-ports-per-vm` sets how many source ports each VM gets; a node fanning out many concurrent connections (pulling shards, hitting many APIs) can exhaust the default and see connection failures, so raise it for busy training nodes. `--enable-logging` (with `--log-filter=ERRORS_ONLY` to keep volume down) surfaces exactly those port-exhaustion drops.

To reach a private service — most commonly a Vertex AI online prediction endpoint — you reserve an internal address and point a forwarding rule at the service's **service attachment**. That address becomes the private entrypoint your clients call, with traffic staying on internal IP space.

```bash
# Reserve a static internal IP for the PSC endpoint
gcloud compute addresses create psc-vertex-ip \
  --region=us-central1 --subnet=ml-train-us \
  --addresses=10.10.0.100

# Create the PSC endpoint (a forwarding rule -> the service attachment)
gcloud compute forwarding-rules create psc-vertex \
  --region=us-central1 --network=ml-vpc \
  --address=psc-vertex-ip \
  --target-service-attachment=projects/SERVICE_PROJECT/regions/us-central1/serviceAttachments/SA_NAME
```

You also reserve **static external** addresses (`gcloud compute addresses create NAME --region=...`) when a load balancer or NAT needs a stable public IP that survives instance recreation.

## Connecting networks: peering and Shared VPC

Two patterns show up constantly in real ML platforms:

- **VPC Peering** connects two VPCs so resources communicate over internal IPs. It is used both between your own networks and, under the hood, for some managed-service connectivity. Peered ranges must not overlap — another reason to plan your CIDR blocks.
- **Shared VPC** lets a central "host" project own the network while multiple "service" projects attach to it. This is the enterprise standard: a platform team owns and secures one VPC, and each ML team gets a service project that runs workloads on the shared network without managing networking themselves. It cleanly separates "who owns the network" from "who runs the models."

```bash
# VPC peering: connect two of your networks (must be created on both sides)
gcloud compute networks peerings create ml-to-data \
  --network=ml-vpc --peer-network=data-vpc \
  --peer-project=myco-fraud-prod

# Shared VPC: designate a host project, then attach a service project
gcloud compute shared-vpc enable HOST_PROJECT_ID
gcloud compute shared-vpc associated-projects add SERVICE_PROJECT_ID \
  --host-project=HOST_PROJECT_ID
```

Two supporting services round out a real network. **Cloud DNS private zones** give internal-only DNS names (so services resolve `feature-store.internal` without a public record); you create a zone with `gcloud dns managed-zones create --visibility=private --networks=ml-vpc`. And **Cloud Router** — the same resource that anchors Cloud NAT — runs **BGP** to exchange routes dynamically with on-prem over a VPN or Interconnect, which is how a training cluster reaches an on-prem data source without hand-maintained static routes.

## VPC Service Controls

For regulated data, **VPC Service Controls** draws a **service perimeter** around a set of projects and their managed services (Cloud Storage, BigQuery, Vertex AI). It prevents data from being read or copied to anything outside the perimeter, even by a caller with valid IAM credentials — it defends against exfiltration, not just unauthorized access. If your training data contains PII or is under a compliance regime, a VPC-SC perimeter around your data and ML projects is often mandatory, and it is a control you should design for early because retrofitting it is disruptive.

Because a badly scoped perimeter can break every pipeline at once, always start in **dry-run** mode: the perimeter logs what it *would* block without actually blocking, so you can find the legitimate cross-project calls before you enforce.

```bash
# Dry-run perimeter first: logs violations without enforcing
gcloud access-context-manager perimeters dry-run create ml-perimeter \
  --policy=POLICY_ID --title="ML data perimeter" \
  --resources=projects/PROJECT_NUMBER \
  --restricted-services=storage.googleapis.com,bigquery.googleapis.com,aiplatform.googleapis.com

# After the dry-run logs are clean, enforce it
gcloud access-context-manager perimeters dry-run enforce ml-perimeter --policy=POLICY_ID
```

A few network gotchas bite ML teams repeatedly. **Overlapping CIDR** ranges make two networks impossible to peer — plan non-overlapping address space up front, including GKE secondary ranges. The **default network's firewall rules are permissively open** (it ships with rules like `default-allow-internal` and SSH/RDP from anywhere), so never run production on it; use a custom-mode VPC. And to SSH into a no-external-IP VM, don't add a public IP — tunnel through **Identity-Aware Proxy** with `gcloud compute ssh VM --tunnel-through-iap`, which reaches the VM over Google's backbone with IAM-gated access.

## How this fits the whole solution

Networking is the connective tissue of the end-to-end system you build later. Streaming ingestion, the BigQuery warehouse, feature serving, distributed training clusters, and low-latency prediction endpoints all ride on the VPC. The defaults that make an ML system production-grade — no external IPs, Private Google Access for API reach, Cloud NAT for controlled egress, Private Service Connect for private endpoints, and a VPC-SC perimeter for sensitive data — are decisions you make here. A reference architecture that ignores networking is a demo, not a system.

## Key takeaways

- A **VPC is global**; **subnets are regional** IP ranges. Use **custom-mode** VPCs and plan non-overlapping CIDR blocks so training, data, and serving coexist and can peer later.
- Firewalls **deny ingress by default**; open paths with rules targeted by **tags or service accounts**, and remember distributed training needs an internal all-ports rule for collective communication.
- Lock down nodes by giving them **no external IP**, then restore reachability with **Private Google Access** (Google APIs), **Cloud NAT** (outbound internet), and **Private Service Connect** (private Vertex endpoints).
- **Shared VPC** separates network ownership from workload ownership; **VPC Service Controls** builds an anti-exfiltration perimeter around sensitive ML data and services.

## CLI cheat-sheet

```bash
# --- VPC + subnets ---
gcloud compute networks create ml-vpc --subnet-mode=custom
gcloud compute networks list
gcloud compute networks describe ml-vpc
gcloud compute networks subnets create ml-train-us --network=ml-vpc --region=us-central1 \
  --range=10.10.0.0/20 --enable-private-ip-google-access
gcloud compute networks subnets create ml-gke-us --network=ml-vpc --region=us-central1 \
  --range=10.20.0.0/20 --secondary-range=pods=10.21.0.0/16,services=10.22.0.0/20
gcloud compute networks subnets expand-ip-range ml-train-us --region=us-central1 --prefix-length=19

# --- Firewall rules + policies ---
gcloud compute firewall-rules create allow-ssh --network=ml-vpc --direction=INGRESS --action=ALLOW \
  --rules=tcp:22 --source-ranges=203.0.113.0/24 --target-tags=ssh-ok
gcloud compute firewall-rules list --filter="network:ml-vpc"
gcloud compute firewall-rules update allow-ssh --enable-logging
gcloud compute firewall-rules delete allow-ssh
gcloud compute network-firewall-policies create ml-baseline --global
gcloud compute network-firewall-policies rules create 1000 --firewall-policy=ml-baseline \
  --global-firewall-policy --direction=INGRESS --action=allow --layer4-configs=tcp:22 \
  --src-ip-ranges=203.0.113.0/24 --enable-logging
gcloud compute network-firewall-policies associations create --firewall-policy=ml-baseline \
  --network=ml-vpc --name=ml-vpc-assoc --global-firewall-policy

# --- Private access: NAT, PGA, PSC ---
gcloud compute routers create ml-router --network=ml-vpc --region=us-central1
gcloud compute routers nats create ml-nat --router=ml-router --region=us-central1 \
  --nat-all-subnet-ip-ranges --auto-allocate-nat-external-ips \
  --min-ports-per-vm=128 --enable-logging --log-filter=ERRORS_ONLY
gcloud compute addresses create psc-vertex-ip --region=us-central1 --subnet=ml-train-us
gcloud compute forwarding-rules create psc-vertex --region=us-central1 --network=ml-vpc \
  --address=psc-vertex-ip --target-service-attachment=projects/P/regions/us-central1/serviceAttachments/SA

# --- Connect networks ---
gcloud compute networks peerings create ml-to-data --network=ml-vpc --peer-network=data-vpc
gcloud compute shared-vpc enable HOST_PROJECT
gcloud compute shared-vpc associated-projects add SVC_PROJECT --host-project=HOST_PROJECT

# --- VPC Service Controls (dry-run first) ---
gcloud access-context-manager perimeters dry-run create ml-perimeter --policy=POLICY_ID \
  --resources=projects/NUM --restricted-services=storage.googleapis.com,aiplatform.googleapis.com
gcloud access-context-manager perimeters dry-run enforce ml-perimeter --policy=POLICY_ID

# --- SSH into a no-external-IP VM ---
gcloud compute ssh VM --tunnel-through-iap
```

## Try it

Build a locked-down training network and prove a private VM can still reach Cloud Storage:

1. Create a custom-mode VPC `ml-vpc` and a subnet with `--enable-private-ip-google-access`.
2. Create firewall rules: SSH from your IP only, plus an internal all-ports rule for the subnet.
3. Set up a Cloud Router and Cloud NAT on the subnet's region.
4. Create a Compute Engine VM in the subnet with `--no-address` (no external IP).
5. SSH in via Identity-Aware Proxy (`gcloud compute ssh <vm> --tunnel-through-iap`), then run `gcloud storage ls gs://<some-bucket>` and `pip install requests`. Confirm the first works via Private Google Access and the second via Cloud NAT — with the VM having no public IP at all.
