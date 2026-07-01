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

Beyond individual rules, Google Cloud offers **network firewall policies** — hierarchical, reusable rule sets you can attach at the organization, folder, or network level. These are the modern, scalable way to manage firewalls across many projects (a security team defines a baseline policy once and it applies everywhere), and they coexist with per-network rules.

## Private access: keeping ML traffic off the public internet

By default a VM with no external IP cannot reach Google APIs (Cloud Storage, BigQuery, Vertex AI) — which is a problem, because best practice for training and serving nodes is to give them **no external IP** to shrink their attack surface. Three features solve this:

- **Private Google Access** — enable it on a subnet (the flag shown earlier) and VMs without external IPs can still reach Google's public API endpoints over Google's internal network. This is the single most common thing you must turn on for a locked-down training VM to `import` data from a bucket.
- **Private Service Connect (PSC)** — creates a private endpoint inside your VPC for a Google-managed or partner service. On the ML side, PSC is how you reach a Vertex AI **online prediction endpoint** privately: prediction traffic flows over your VPC's internal IP space, never touching the public internet. It is the standard for latency-sensitive, security-conscious serving.
- **Cloud NAT** — if a private VM needs *outbound* internet access (to `pip install` from PyPI, pull a public container, or hit a third-party API) without having an external IP, Cloud NAT provides managed, egress-only network address translation.

```bash
# Give private VMs outbound internet without external IPs
gcloud compute routers create ml-router \
  --network=ml-vpc --region=us-central1

gcloud compute routers nats create ml-nat \
  --router=ml-router --region=us-central1 \
  --nat-all-subnet-ip-ranges \
  --auto-allocate-nat-external-ips
```

## Connecting networks: peering and Shared VPC

Two patterns show up constantly in real ML platforms:

- **VPC Peering** connects two VPCs so resources communicate over internal IPs. It is used both between your own networks and, under the hood, for some managed-service connectivity. Peered ranges must not overlap — another reason to plan your CIDR blocks.
- **Shared VPC** lets a central "host" project own the network while multiple "service" projects attach to it. This is the enterprise standard: a platform team owns and secures one VPC, and each ML team gets a service project that runs workloads on the shared network without managing networking themselves. It cleanly separates "who owns the network" from "who runs the models."

## VPC Service Controls

For regulated data, **VPC Service Controls** draws a **service perimeter** around a set of projects and their managed services (Cloud Storage, BigQuery, Vertex AI). It prevents data from being read or copied to anything outside the perimeter, even by a caller with valid IAM credentials — it defends against exfiltration, not just unauthorized access. If your training data contains PII or is under a compliance regime, a VPC-SC perimeter around your data and ML projects is often mandatory, and it is a control you should design for early because retrofitting it is disruptive.

## How this fits the whole solution

Networking is the connective tissue of the end-to-end system you build later. Streaming ingestion, the BigQuery warehouse, feature serving, distributed training clusters, and low-latency prediction endpoints all ride on the VPC. The defaults that make an ML system production-grade — no external IPs, Private Google Access for API reach, Cloud NAT for controlled egress, Private Service Connect for private endpoints, and a VPC-SC perimeter for sensitive data — are decisions you make here. A reference architecture that ignores networking is a demo, not a system.

## Key takeaways

- A **VPC is global**; **subnets are regional** IP ranges. Use **custom-mode** VPCs and plan non-overlapping CIDR blocks so training, data, and serving coexist and can peer later.
- Firewalls **deny ingress by default**; open paths with rules targeted by **tags or service accounts**, and remember distributed training needs an internal all-ports rule for collective communication.
- Lock down nodes by giving them **no external IP**, then restore reachability with **Private Google Access** (Google APIs), **Cloud NAT** (outbound internet), and **Private Service Connect** (private Vertex endpoints).
- **Shared VPC** separates network ownership from workload ownership; **VPC Service Controls** builds an anti-exfiltration perimeter around sensitive ML data and services.

## Try it

Build a locked-down training network and prove a private VM can still reach Cloud Storage:

1. Create a custom-mode VPC `ml-vpc` and a subnet with `--enable-private-ip-google-access`.
2. Create firewall rules: SSH from your IP only, plus an internal all-ports rule for the subnet.
3. Set up a Cloud Router and Cloud NAT on the subnet's region.
4. Create a Compute Engine VM in the subnet with `--no-address` (no external IP).
5. SSH in via Identity-Aware Proxy (`gcloud compute ssh <vm> --tunnel-through-iap`), then run `gcloud storage ls gs://<some-bucket>` and `pip install requests`. Confirm the first works via Private Google Access and the second via Cloud NAT — with the VM having no public IP at all.
