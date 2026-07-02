# 03 — Compute: VMs and Scale Sets

Compute is where the model actually trains and, sometimes, where it serves. Azure offers many ways to run code — managed clusters inside Azure Machine Learning, containers on AKS, serverless Functions — but underneath almost all of them are **virtual machines**. Understanding the VM layer directly pays off even when you mostly use higher-level services, because the VM **SKU** (the size string like `Standard_ND96isr_H100_v5`) determines your GPU, your memory, your interconnect, and most of your bill. In an end-to-end solution, VMs show up in three places: as the nodes behind Azure Machine Learning compute clusters that run training, as the nodes in AKS GPU pools that serve models, and occasionally as a standalone box for interactive development or a one-off experiment.

## How to read a VM SKU

Azure VM sizes follow a decodable convention. Take `Standard_ND96isr_H100_v5`:

- **Family letter(s)** — the workload class. `D` general-purpose, `E` memory-optimized, `F` compute-optimized, and for ML the accelerated families: **`N`** for GPU. Within N: **`NC`** (compute/applied AI and inference), **`ND`** (deep-learning training, network-dense with high-bandwidth interconnect), **`NV`** (visualization/graphics, also used for lighter inference).
- **Number** — vCPU count (`96` here).
- **Additional letters** — capabilities: `i` (InfiniBand for multi-node), `s` (premium SSD support), `r` (RDMA / remote direct memory access for tight multi-node communication), `a` (AMD CPU), `d` (local temp disk).
- **GPU + version** — `H100 v5` names the accelerator and generation.

Learning to read the SKU means you can look at a size string and immediately know whether it will scale across nodes efficiently (does it have `i`/`r`?) and what class of GPU you are paying for.

## The GPU families that matter for ML

**ND family — distributed training.** These are the big multi-GPU boxes with fast interconnect for scaling across nodes. Current and common members:

- **ND H100 v5** — 8× NVIDIA H100 (80 GB) per VM, connected by NVLink inside the box and NDR InfiniBand across boxes for thousands-of-GPU clusters. The top SKU `Standard_ND96isr_H100_v5` is the workhorse for large-scale training and LLM fine-tuning.
- **ND H200 v5** — 8× NVIDIA H200 with substantially more high-bandwidth memory than H100 (roughly a 76% HBM increase), which lets larger models and longer context fit per GPU.
- **ND A100 v4** — 8× A100 (40 or 80 GB), the prior generation, still widely available and cheaper; a sensible default when H100 quota is scarce.
- **ND MI300X v5** — 8× AMD Instinct MI300X, an alternative accelerator with very large memory, useful for memory-bound inference and training when you can run on ROCm.
- Newer **GB200 / Blackwell-class** systems are rolling into select regions for frontier-scale training; availability and quota are tightly constrained, so plan region and capacity early.

**NC family — applied AI, fine-tuning, and inference.** Smaller GPU counts, PCIe rather than the densest interconnect, better price-per-GPU for single-node work:

- **NCads H100 v5** — up to 2× NVIDIA H100 NVL (94 GB each) on AMD EPYC Genoa CPUs. A strong single-node fine-tuning and high-end inference box.
- **NC A100 v4** — up to 4× A100 (80 GB) PCIe. A cost-effective default for single-node training and batch inference.

**NV family — visualization and light inference.** GPU-accelerated but tuned for graphics/remote-workstation use; occasionally used for small models or dev boxes.

Rule of thumb: **ND for multi-node distributed training** (you need the InfiniBand/RDMA), **NC for single-node fine-tuning and inference**, **NV for dev/visualization**. Always confirm the exact SKU is offered *and that you have quota* in your target region before designing around it.

```bash
# See which GPU sizes exist in a region
az vm list-sizes --location eastus2 \
  --query "[?contains(name,'Standard_N')].name" -o tsv | sort

# Check GPU-family quota (deployments fail silently-late without it)
az vm list-usage --location eastus2 \
  --query "[?contains(localName,'ND') || contains(localName,'NC')].{Family:localName, Used:currentValue, Limit:limit}" \
  -o table

# Inspect one SKU's full spec (vCPUs, memory, max data disks, accelerated networking)
az vm list-sizes --location eastus2 \
  --query "[?name=='Standard_NC24ads_A100_v4']" -o jsonc
```

If your quota is zero (the default for many GPU families), you request an increase through the portal's **Quotas** blade or a support request — do this days ahead of a deadline, because approval is not instant.

## Images: what the OS starts as

A VM boots from an **image**. For ML you want the **Data Science Virtual Machine (DSVM)** or an Azure Machine Learning curated environment image, which ships with CUDA, cuDNN, the NVIDIA driver, Python, and common frameworks preinstalled — this saves you the fragile dance of matching driver, CUDA, and framework versions. For reproducibility, build your own image with **Azure VM Image Builder** or, more commonly for ML, bake your dependencies into a **container image** in Azure Container Registry and run that on the VM. The container route is preferable because the same image runs on your laptop, on the training cluster, and behind the serving endpoint — identical environment end to end.

```bash
# Create a single GPU dev box from the Ubuntu-based DSVM image
az vm create \
  --resource-group rg-mlx-dev \
  --name vm-gpu-dev \
  --size Standard_NC24ads_A100_v4 \
  --image microsoft-dsvm:ubuntu-hpc:2204:latest \
  --admin-username azureuser \
  --generate-ssh-keys \
  --assign-identity id-mlplatform   # attach the platform managed identity
```

To find a concrete image URN rather than guessing, list what a publisher offers. Image references are `publisher:offer:sku:version`, and `latest` is convenient for dev but you should pin an exact version for anything reproducible:

```bash
# Browse the Data Science VM and HPC (CUDA-preinstalled) images
az vm image list --publisher microsoft-dsvm --all -o table
az vm image list --publisher microsoft-dsvm --offer ubuntu-hpc --all -o table
az vm image list-skus --location eastus2 --publisher microsoft-dsvm --offer ubuntu-hpc -o table
```

Once a VM exists you spend most of your time in its power lifecycle, and the distinction between **stop** and **deallocate** is the one that costs people money. A merely *stopped* (`az vm stop`) VM still holds its compute reservation and keeps billing for cores; a **deallocated** VM releases the hardware and stops compute charges (you still pay for the disks). Always `deallocate` a GPU box you are not using:

```bash
az vm list -g rg-mlx-dev -o table                # all VMs and their power state
az vm show -g rg-mlx-dev -n vm-gpu-dev -d --query powerState -o tsv
az vm start -g rg-mlx-dev -n vm-gpu-dev
az vm deallocate -g rg-mlx-dev -n vm-gpu-dev     # stops COMPUTE billing (do this)
az vm stop -g rg-mlx-dev -n vm-gpu-dev           # stops the OS but KEEPS billing cores
az vm resize -g rg-mlx-dev -n vm-gpu-dev --size Standard_NC48ads_A100_v4  # requires deallocate for a different family
az vm delete -g rg-mlx-dev -n vm-gpu-dev --yes
```

## Spot VMs: cheap, interruptible compute

**Spot VMs** run on Azure's spare capacity at a steep discount (often 60–90% off pay-as-you-go), with the catch that Azure can **evict** them with 30 seconds' notice when it needs the capacity back. For ML this is a superb fit for **fault-tolerant training** — if you checkpoint frequently, an eviction costs you a few minutes of recompute, not the run. Set a max price and an eviction policy:

```bash
az vm create \
  --resource-group rg-mlx-dev --name vm-spot-train \
  --size Standard_NC24ads_A100_v4 \
  --image microsoft-dsvm:ubuntu-hpc:2204:latest \
  --priority Spot --max-price -1 \
  --eviction-policy Deallocate \
  --generate-ssh-keys
```

`--max-price -1` means "pay up to the on-demand price," maximizing the chance you keep the VM. Azure Machine Learning compute clusters expose the same capability via a `low_priority` tier, which is where you will use spot most often — the cluster handles eviction and resubmission for you. Do **not** put spot behind a latency-sensitive production endpoint; use it for training and batch, keep dedicated capacity for real-time serving.

## Scale sets: horizontal fleets

A **Virtual Machine Scale Set (VMSS)** manages an identical group of VMs as one unit, with autoscaling based on metrics (CPU, GPU, queue depth) or schedule. You rarely create raw VMSS for ML because the managed services build on top of it — an **Azure Machine Learning compute cluster** *is* a scale set with min/max node counts, and an **AKS node pool** is a scale set of Kubernetes workers. Understanding VMSS explains their behavior: scale-to-zero when idle (so you pay nothing between jobs), scale-out under load, and per-node health management.

If you ever do touch raw VMSS, the shape is familiar — a scale set has a capacity you set directly or hand to an autoscale rule, and you can list its member instances:

```bash
az vmss create -g rg-mlx-dev -n vmss-workers --image Ubuntu2204 \
  --vm-sku Standard_NC24ads_A100_v4 --instance-count 2 --generate-ssh-keys
az vmss scale -g rg-mlx-dev -n vmss-workers --new-capacity 4
az vmss list-instances -g rg-mlx-dev -n vmss-workers -o table
az vmss deallocate -g rg-mlx-dev -n vmss-workers
```

```bash
# The managed equivalent you'll actually use: an autoscaling ML compute cluster
az ml compute create \
  --name gpu-cluster \
  --type AmlCompute \
  --size Standard_NC24ads_A100_v4 \
  --min-instances 0 --max-instances 4 \
  --tier LowPriority \
  --resource-group rg-mlx-dev --workspace-name mlw-mlx-dev
```

Min-instances of 0 is the money-saver: the cluster deallocates all nodes when no job is queued, and spins them up on demand. Combined with the LowPriority (spot) tier, this is the cheapest way to run training on Azure.

The `az ml compute` command group (CLI v2, the `ml` extension) is where you manage both cluster types you actually use: **AmlCompute** (the scale-to-zero training cluster above) and a **ComputeInstance** (a single-user managed dev box, the cloud equivalent of the DSVM VM but wired into the workspace). Both take a managed identity via `--identity-type` so jobs authenticate without keys, and both support the full lifecycle:

```bash
# A single-user managed dev box tied to the workspace, with the platform MI attached
az ml compute create --name ci-dev --type ComputeInstance \
  --size Standard_NC24ads_A100_v4 \
  --identity-type UserAssigned --user-assigned-identities "$UAMI_ID" \
  -g rg-mlx-dev -w mlw-mlx-dev

# List / inspect / retune / tear down clusters and instances
az ml compute list -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml compute show --name gpu-cluster -g rg-mlx-dev -w mlw-mlx-dev
az ml compute update --name gpu-cluster --min-instances 0 --max-instances 8 \
  -g rg-mlx-dev -w mlw-mlx-dev
az ml compute list-nodes --name gpu-cluster -g rg-mlx-dev -w mlw-mlx-dev -o table  # what's running now
az ml compute stop --name ci-dev -g rg-mlx-dev -w mlw-mlx-dev   # ComputeInstance only
az ml compute delete --name gpu-cluster -g rg-mlx-dev -w mlw-mlx-dev --yes
```

Two practitioner notes: a ComputeInstance keeps billing while *running*, so `az ml compute stop` it at the end of the day (an idle-shutdown schedule in the YAML spec automates this); and `az ml compute update` can grow or shrink an AmlCompute cluster's node bounds live, but you cannot change its VM `--size` after creation — that is fixed at create time, so pick the SKU deliberately.

## Choosing compute for the whole solution

In the reference architecture you will build later, compute divides by job type. **Training** runs on Azure Machine Learning compute clusters — ND SKUs for multi-node jobs, NC for single-node, LowPriority tier and scale-to-zero to control cost. **Batch inference** runs on the same clusters through batch endpoints. **Real-time inference** runs on dedicated (non-spot) managed online endpoints or AKS GPU pools sized on NC. **Interactive development** happens on a small compute instance or a single NC dev box. You almost never manage raw VMs by hand; you let the managed layer manage scale sets for you and reserve direct VM control for special cases.

## Key takeaways

- **Decode the SKU**: `N` = GPU; `ND` = multi-node training (has InfiniBand/RDMA), `NC` = single-node fine-tuning/inference, `NV` = visualization.
- Current GPU workhorses: **ND H100 v5 / ND H200 v5** for large distributed training, **NC A100 v4 / NCads H100 v5** for single-node work; verify SKU **and quota** in your region first.
- Use **container images** (via ACR) for reproducible environments across dev, training, and serving.
- **Spot / LowPriority** compute cuts training cost dramatically if you checkpoint; never put it behind latency-sensitive endpoints.
- You rarely touch raw VMs or VMSS — Azure Machine Learning compute clusters and AKS node pools *are* managed scale sets; use `min-instances 0` to pay nothing when idle.

## CLI cheat-sheet

```bash
# --- discover SKUs, quota, images ---
az vm list-sizes -l eastus2 --query "[?contains(name,'Standard_N')].name" -o tsv | sort
az vm list-usage -l eastus2 --query "[?contains(localName,'ND')||contains(localName,'NC')].{F:localName,Used:currentValue,Max:limit}" -o table
az vm image list --publisher microsoft-dsvm --offer ubuntu-hpc --all -o table

# --- raw VM lifecycle ---
az vm create -g rg-mlx-dev -n vm-gpu-dev --size Standard_NC24ads_A100_v4 \
  --image microsoft-dsvm:ubuntu-hpc:2204:latest --generate-ssh-keys --assign-identity id-mlplatform
az vm create -g rg-mlx-dev -n vm-spot --size Standard_NC24ads_A100_v4 \
  --priority Spot --max-price -1 --eviction-policy Deallocate --generate-ssh-keys
az vm list -g rg-mlx-dev -o table
az vm show -g rg-mlx-dev -n vm-gpu-dev -d --query powerState -o tsv
az vm start -g rg-mlx-dev -n vm-gpu-dev
az vm deallocate -g rg-mlx-dev -n vm-gpu-dev      # stops compute billing
az vm resize -g rg-mlx-dev -n vm-gpu-dev --size Standard_NC48ads_A100_v4
az vm delete -g rg-mlx-dev -n vm-gpu-dev --yes

# --- scale sets (rarely by hand) ---
az vmss create -g rg-mlx-dev -n vmss-workers --image Ubuntu2204 --vm-sku Standard_NC24ads_A100_v4 --instance-count 2 --generate-ssh-keys
az vmss scale -g rg-mlx-dev -n vmss-workers --new-capacity 4
az vmss list-instances -g rg-mlx-dev -n vmss-workers -o table

# --- Azure ML compute (CLI v2, what you actually use) ---
az ml compute create --name gpu-cluster --type AmlCompute --size Standard_NC24ads_A100_v4 \
  --min-instances 0 --max-instances 4 --tier LowPriority -g rg-mlx-dev -w mlw-mlx-dev
az ml compute create --name ci-dev --type ComputeInstance --size Standard_NC24ads_A100_v4 \
  --identity-type UserAssigned --user-assigned-identities "$UAMI_ID" -g rg-mlx-dev -w mlw-mlx-dev
az ml compute list -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml compute show --name gpu-cluster -g rg-mlx-dev -w mlw-mlx-dev
az ml compute update --name gpu-cluster --min-instances 0 --max-instances 8 -g rg-mlx-dev -w mlw-mlx-dev
az ml compute list-nodes --name gpu-cluster -g rg-mlx-dev -w mlw-mlx-dev -o table
az ml compute stop --name ci-dev -g rg-mlx-dev -w mlw-mlx-dev
az ml compute delete --name gpu-cluster -g rg-mlx-dev -w mlw-mlx-dev --yes
```

## Try it

Check GPU quota in two regions with `az vm list-usage` and note where you actually have capacity. Then create an Azure Machine Learning compute cluster on an NC A100 SKU with `--min-instances 0 --max-instances 2 --tier LowPriority`, list it, and confirm it reports zero running nodes (and therefore near-zero cost) while idle. Bonus: read three real SKU strings such as `Standard_ND96isr_H100_v5`, `Standard_NC24ads_A100_v4`, and `Standard_NV36ads_A10_v5`, and write down what each letter and number means before checking your answer.
