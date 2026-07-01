# 05 — Networking: VNet Basics

Networking is the part of Azure most ML engineers skip until a security review forces them to learn it fast. It is worth learning early, because in any real production ML system the difference between a demo and a shippable solution is often network isolation: keeping your training data, model artifacts, and inference endpoints off the public internet and reachable only from inside a trusted network boundary. This section covers the primitives — virtual networks, subnets, network security groups, and private endpoints — and how they wrap around an ML platform so that storage, Key Vault, the registry, and Azure Machine Learning talk to each other privately.

## Virtual networks and subnets

A **virtual network (VNet)** is your own isolated, software-defined network in a region, defined by a private IP address space in CIDR notation (for example `10.0.0.0/16`). It is the boundary inside which your resources can reach each other by private IP, and the anchor point for controlling what enters and leaves.

A VNet is divided into **subnets**, each carving out a slice of the address space (`10.0.1.0/24`, `10.0.2.0/24`, and so on). Subnets are how you segment a system into tiers with different rules — a common ML layout is one subnet for compute (training clusters, endpoints), one dedicated subnet for **private endpoints** (the private IPs that front your storage, vault, and registry), and one for any supporting services. Segmentation lets you apply different security policies per tier and reason about traffic flow.

```bash
# Create a VNet with a compute subnet and a private-endpoint subnet
az network vnet create \
  --resource-group rg-mlx-dev --name vnet-mlx \
  --address-prefix 10.0.0.0/16 \
  --subnet-name snet-compute --subnet-prefix 10.0.1.0/24

az network vnet subnet create \
  --resource-group rg-mlx-dev --vnet-name vnet-mlx \
  --name snet-pe --address-prefix 10.0.2.0/24 \
  --disable-private-endpoint-network-policies true
```

Sizing note for ML: Azure Machine Learning needs a **private IP per compute-instance, per compute-cluster node, and per private endpoint**. A large training cluster plus several endpoints can consume many addresses, so give the compute subnet a generous range (`/24` or bigger) rather than discovering mid-scale-out that you have run out of IPs.

## Network security groups

A **network security group (NSG)** is a stateful firewall of allow/deny rules you attach to a subnet (or a network interface). Each rule matches on source, destination, port, and protocol, and has a priority; lower numbers evaluate first. Because NSGs are stateful, allowing an inbound flow automatically permits the return traffic.

The default rules deny inbound from the internet and allow outbound, which is a sane starting posture. For an ML platform you tighten inbound to essentially nothing (compute reaches *out*; it does not accept unsolicited inbound), and you increasingly control **outbound** too, so a compromised training job cannot exfiltrate data to an arbitrary host.

```bash
# Attach an NSG to the compute subnet and deny inbound internet by default
az network nsg create -g rg-mlx-dev -n nsg-compute
az network nsg rule create -g rg-mlx-dev --nsg-name nsg-compute \
  --name deny-inbound-internet --priority 4096 \
  --direction Inbound --access Deny --protocol '*' \
  --source-address-prefixes Internet --destination-port-ranges '*'
az network vnet subnet update -g rg-mlx-dev --vnet-name vnet-mlx \
  --name snet-compute --network-security-group nsg-compute
```

## Private endpoints and Private Link

By default, an Azure storage account, Key Vault, or ML workspace has a **public endpoint** — a resource on the internet, protected by identity and RBAC but still network-reachable from anywhere. For sensitive ML data that is often unacceptable. **Azure Private Link** solves this: a **private endpoint** projects a specific Azure resource into your VNet as a **private IP address** on a subnet you choose. Traffic to that resource then flows entirely over the Azure backbone and never traverses the public internet. You typically pair this with disabling the resource's public network access, so the *only* way to reach it is from inside the VNet.

```bash
# Give a storage account a private endpoint in the PE subnet
STORAGE_ID=$(az storage account show -n stmlxdata -g rg-mlx-dev --query id -o tsv)
az network private-endpoint create \
  --resource-group rg-mlx-dev --name pe-stmlxdata \
  --vnet-name vnet-mlx --subnet snet-pe \
  --private-connection-resource-id "$STORAGE_ID" \
  --group-id blob \
  --connection-name conn-stmlxdata

# Lock the account down to VNet-only
az storage account update -n stmlxdata -g rg-mlx-dev \
  --public-network-access Disabled
```

Private endpoints depend on **private DNS**: inside the VNet, `stmlxdata.blob.core.windows.net` must resolve to the private IP, not the public one. You wire this with a **Private DNS Zone** (for example `privatelink.blob.core.windows.net`) linked to the VNet, so name resolution transparently routes callers to the private endpoint. Getting DNS right is the step most people miss — the private endpoint exists, but code still resolves the public name and fails, or silently goes over the internet.

## Managed VNet: network isolation without the plumbing

Wiring VNets, subnets, NSGs, route tables, private endpoints, and DNS by hand for an entire ML platform is a lot of surface area to get right. Azure Machine Learning offers a **managed virtual network** that does most of this for you. When you enable it, the workspace and its **managed compute** run inside a VNet that Azure provisions and secures automatically, and it can create the private endpoints to the workspace's dependent resources — storage, Key Vault, container registry — without you defining each NSG and route.

You choose an isolation mode:

- **Allow internet outbound** — compute is isolated inbound but can reach the internet for package installs and public data.
- **Allow only approved outbound** — the tightest common mode; compute can only reach destinations you explicitly allow (specific FQDNs, private endpoints, service tags), which is what regulated environments want.

Enabling it via the SDK is a workspace property:

```python
from azure.ai.ml.entities import Workspace, ManagedNetwork
from azure.ai.ml.constants import IsolationMode

ws = Workspace(
    name="mlw-mlx-prod",
    location="eastus2",
    managed_network=ManagedNetwork(isolation_mode=IsolationMode.ALLOW_ONLY_APPROVED_OUTBOUND),
)
ml_client.workspaces.begin_create(ws).result()
```

The managed VNet is the recommended path for most teams: you get network isolation of the workspace and its data-plane dependencies with a fraction of the manual configuration. Reserve fully custom VNets (where you own every NSG, UDR, and firewall rule) for cases with strict, pre-existing network topology you must integrate into.

## How networking fits the whole solution

Network isolation is the outer ring of the end-to-end architecture. Picture the medallion data lake, the Key Vault, the container registry, and the Azure Machine Learning workspace all reachable *only* through private endpoints inside one VNet, with public access disabled on each. Training compute and inference endpoints run inside that VNet's subnets, authenticate with managed identity, and pull data and images over the private backbone. Ingestion services (Event Hubs) and the GenAI layer (Azure AI Foundry / Azure OpenAI) get their own private endpoints so prompts and documents never leave the network. Exposure to the outside world narrows to a single, deliberately chosen front door — an application gateway or API layer — while everything behind it is dark to the internet. Identity (previous section) answers *who* can call a resource; networking answers *from where*, and defense in depth means using both.

## Key takeaways

- A **VNet** is your private IP space; **subnets** segment it into tiers (compute, private endpoints, services). Give the compute subnet plenty of IPs — Azure ML needs one per node and endpoint.
- **NSGs** are stateful allow/deny firewalls on subnets; lock down inbound and increasingly control outbound to prevent exfiltration.
- **Private endpoints (Private Link)** project a specific resource into your VNet as a private IP so traffic avoids the public internet; pair with disabling public access — and don't forget **private DNS**.
- Azure Machine Learning's **managed VNet** automates most isolation (workspace, managed compute, dependent private endpoints); use **allow-only-approved-outbound** for the tightest posture.
- Networking (from where) and identity (who) are complementary layers of **defense in depth** — a production ML system uses both.

## Try it

Create a VNet with a `snet-compute` and a `snet-pe` subnet. Add a private endpoint for an existing storage account into `snet-pe` with `--group-id blob`, link a `privatelink.blob.core.windows.net` Private DNS Zone to the VNet, then set the account's public network access to Disabled. From a VM inside the VNet, resolve the blob endpoint name and confirm it returns the *private* IP and that data access still works — then try from outside the VNet and confirm it now fails. That contrast is the whole point of network isolation.
