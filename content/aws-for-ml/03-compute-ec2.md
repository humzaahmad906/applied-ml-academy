# 03 — Compute: EC2 for ML

Elastic Compute Cloud is raw virtual servers you rent by the second. Higher-level ML services like SageMaker run *on top of* EC2, so understanding the instance zoo, how AMIs and pricing work, and when to drop down to bare EC2 is foundational even if you spend most of your time in managed services. This module maps the compute landscape an ML engineer actually cares about: which accelerators exist, how to pay less for them, and when EC2 is the right tool versus a managed alternative.

## Instance families and the naming convention

An instance type name encodes a lot: `p5.48xlarge` is family `p`, generation `5`, size `48xlarge`. General patterns you will meet:

- **General purpose** (`m`, `t`): balanced CPU/memory, fine for data prep, orchestration, small inference. `t` instances are burstable and cheap for intermittent work.
- **Compute optimized** (`c`): high CPU-to-memory ratio, good for CPU inference and feature engineering.
- **Memory optimized** (`r`, `x`): large RAM for big data frames, in-memory joins, embedding stores.
- **Storage optimized** (`i`): fast local NVMe for shuffling large datasets.

For ML the families that matter most are the accelerated ones.

Before you commit to a type, interrogate its actual specs rather than trusting the name. `describe-instance-types` returns vCPUs, memory, and — for accelerated families — GPU count, GPU memory, and network bandwidth, while `describe-instance-type-offerings` tells you *where* (which Region or even which AZ) the type is actually offered. The AZ-level check matters because a scarce accelerator can exist in `us-east-1` but only in one or two of its zones:

```bash
# Full spec sheet for a GPU type: GPU count, GPU memory, network Gbps
aws ec2 describe-instance-types --instance-types p5.48xlarge \
  --query 'InstanceTypes[0].{vCPU:VCpuInfo.DefaultVCpus,MemGiB:MemoryInfo.SizeInMiB,
    GPUs:GpuInfo.Gpus[0].Count,GpuMemMiB:GpuInfo.TotalGpuMemoryInMiB,
    Net:NetworkInfo.NetworkPerformance}'

# Which AZs in this Region actually offer g6.2xlarge?
aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters Name=instance-type,Values=g6.2xlarge \
  --query 'InstanceTypeOfferings[].Location' --output text
```

## GPU instances

The current NVIDIA GPU lineup, newest first:

- **P6-B200** — NVIDIA Blackwell B200 GPUs, the current top tier for large-scale training and inference. **P6e-GB200 UltraServers** use NVIDIA Grace Blackwell (GB200) superchips and interconnect up to 72 GPUs into a single UltraServer for trillion-parameter-scale training.
- **P5 / P5e / P5en** — NVIDIA H100 (P5) and H200 (P5e/P5en), the workhorse for foundation-model training over the last generation.
- **P4d / P4de** — NVIDIA A100, still widely available and cost-effective for many training jobs.
- **G6 / G6e** — NVIDIA L4 (G6) and L40S (G6e), tuned for cost-efficient inference, fine-tuning, and graphics-adjacent ML.
- **G5** — NVIDIA A10G, a longtime default for mid-size inference and small training.

Rough rule: **P-family for training, G-family for inference**, though large-model inference increasingly uses P-family too.

## AWS custom accelerators

AWS designs its own silicon to undercut GPU cost/performance:

- **Trainium** (`Trn1`, `Trn2`, built on Trainium2 chips) targets training. Trainium3 has been announced as the next generation on a 3nm process. Trainium instances can be linked into **UltraClusters** and **UltraServers** for very large jobs.
- **Inferentia** (`Inf1`, `Inf2`) targets high-throughput, low-cost inference.

Both are programmed through the **AWS Neuron SDK**, which plugs into PyTorch and JAX. The tradeoff is real: Trainium/Inferentia can be dramatically cheaper per unit of throughput, but your model and framework must be Neuron-compatible, which is smoother for mainstream architectures than for exotic custom kernels.

## Networking for distributed training

Multi-node GPU training is bottlenecked by inter-node bandwidth. Two features address this: **cluster placement groups**, which pack instances physically close for low latency, and the **Elastic Fabric Adapter (EFA)**, a network interface that bypasses the OS kernel for high-throughput, low-latency collective communication (the all-reduce operations that DDP and NCCL depend on). Any serious multi-node training on EC2 uses both.

You create a `cluster`-strategy placement group once, then launch every node into it. EFA is added by attaching a network interface with the EFA `InterfaceType` at launch (the node must be a type that supports EFA, and all EFA-enabled nodes must sit in the same subnet and placement group):

```bash
# One low-latency cluster placement group for the whole job
aws ec2 create-placement-group --group-name ml-cluster --strategy cluster

# Launch a node into it with an EFA interface attached
aws ec2 run-instances --instance-type p5.48xlarge --count 2 \
  --image-id ami-0123456789abcdef0 --key-name my-key \
  --placement GroupName=ml-cluster \
  --network-interfaces '[{"DeviceIndex":0,"InterfaceType":"efa","SubnetId":"subnet-abc","Groups":["sg-abc"]}]'

# Confirm a type supports EFA before you rely on it
aws ec2 describe-instance-types --instance-types p5.48xlarge \
  --query 'InstanceTypes[0].NetworkInfo.EfaSupported'
```

The two other placement strategies are worth knowing so you pick the right one: `spread` places instances on distinct hardware for maximum fault isolation (good for a few critical inference hosts), and `partition` groups instances into fault-isolated partitions (good for large distributed data stores). For training you almost always want `cluster`.

## AMIs and containers

An **Amazon Machine Image** is the disk template an instance boots from. Rather than installing CUDA and drivers yourself, use the **Deep Learning AMI (DLAMI)**, which ships with GPU drivers, CUDA, cuDNN, and preinstalled frameworks. For containerized workflows, **AWS Deep Learning Containers (DLCs)** are prebuilt, optimized images for PyTorch/TensorFlow that run on EC2, ECS, EKS, and SageMaker — using the same DLC across training and serving keeps your environment consistent end to end.

Do not hardcode an AMI ID — they change with every DLAMI release and differ per Region. AWS publishes the current DLAMI IDs as **public SSM parameters**, so you resolve the latest ID at launch time. The parameter path encodes architecture, framework, version, and OS:

```bash
# Resolve the latest DLAMI id for PyTorch GPU on Ubuntu, this Region
aws ssm get-parameter \
  --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id \
  --query Parameter.Value --output text

# Discover which framework/version parameter paths exist
aws ssm get-parameters-by-path \
  --path /aws/service/deeplearning/ami/x86_64/ --recursive \
  --query 'Parameters[].Name' --output text
```

You can even pass the SSM parameter directly to `run-instances` with the `resolve:ssm:` prefix, so the launch always picks up the newest image:

```bash
# Launch a GPU instance, resolving the DLAMI id from SSM at launch time
DLAMI=$(aws ssm get-parameter \
  --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id \
  --query Parameter.Value --output text)

aws ec2 run-instances \
  --image-id "$DLAMI" \
  --instance-type g6.2xlarge \
  --key-name my-key \
  --iam-instance-profile Name=ml-ec2-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]'
```

Once a launch configuration stabilizes, capture it in a **launch template** so every future instance — and every Auto Scaling group or Spot fleet — starts identically. Templates version cleanly, which is how you roll AMI upgrades without editing scattered scripts:

```bash
aws ec2 create-launch-template --launch-template-name ml-gpu \
  --launch-template-data '{"InstanceType":"g6.2xlarge",
    "ImageId":"resolve:ssm:/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id",
    "IamInstanceProfile":{"Name":"ml-ec2-profile"}}'
aws ec2 run-instances --launch-template LaunchTemplateName=ml-gpu --count 1
```

## Pricing: Spot, On-Demand, Savings Plans, Capacity Blocks

You pay for EC2 four main ways:

- **On-Demand**: pay per second, no commitment. Maximum flexibility, highest price.
- **Spot**: spare capacity at up to ~90% off, but AWS can reclaim it with a two-minute warning. Ideal for **fault-tolerant training** that checkpoints frequently — you resume from the last checkpoint after an interruption. Not for stateful, uninterruptible work.
- **Savings Plans / Reserved Instances**: commit to a steady spend (1 or 3 years) for a large discount. Right for baseline inference capacity that runs continuously.
- **Capacity Blocks for ML**: reserve a block of GPU capacity (for example P6e-GB200 or P5) for a defined future window. This is how you *guarantee* you will actually get scarce accelerators for a planned training run, since On-Demand GPU capacity is frequently exhausted.

The engineering consequence: architect training to checkpoint often so you can ride Spot, and reserve Capacity Blocks when a run absolutely must start on time.

**Reading the Spot market.** Before you commit to Spot for a family, look at its recent price history to gauge volatility — a GPU type whose Spot price hovers near On-Demand is one that gets reclaimed often. The cleanest way to *use* Spot today is `run-instances` with a Spot market spec (the standalone `request-spot-instances` API still works but is the older path):

```bash
# Recent Spot prices for a GPU type across AZs
aws ec2 describe-spot-price-history --instance-types g6.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u -v-1d +%Y-%m-%dT%H:%M:%SZ)" \
  --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice,Timestamp]' --output table

# Launch a Spot instance directly (persistent so it relaunches after reclaim)
aws ec2 run-instances --instance-type g6.2xlarge --count 1 \
  --image-id "$DLAMI" --key-name my-key \
  --instance-market-options '{"MarketType":"spot",
    "SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}'
```

**Guaranteeing scarce GPUs.** For a planned run you either buy a **Capacity Block for ML** (a fixed future window of GPU capacity) or create an **On-Demand Capacity Reservation** (open-ended, in one AZ). Capacity Blocks are a search-then-purchase flow:

```bash
# 1) Find an available Capacity Block offering
aws ec2 describe-capacity-block-offerings \
  --instance-type p5.48xlarge --instance-count 16 \
  --start-date-range 2026-08-01T00:00:00Z --end-date-range 2026-08-15T00:00:00Z \
  --capacity-duration-hours 48

# 2) Purchase the offering id it returns
aws ec2 purchase-capacity-block \
  --capacity-block-offering-id cb-0123456789abcdef0 \
  --instance-platform Linux/UNIX

# Or reserve On-Demand capacity in one AZ, open-ended
aws ec2 create-capacity-reservation \
  --instance-type g6.2xlarge --instance-platform Linux/UNIX \
  --availability-zone us-east-1a --instance-count 4
```

## When to use EC2 versus managed services

Reach for raw EC2 when you need full control of the environment, custom drivers or kernels, a long-lived experimentation box, or a serving stack that does not fit SageMaker or Lambda. Reach for **SageMaker** when you want managed training jobs, endpoints, and MLOps without operating servers. Reach for **containers on ECS/EKS** when you already run Kubernetes or need portable, multi-framework serving. EC2 is the floor everything else stands on; most teams use it directly for a handful of things and rely on managed layers for the rest.

Whichever route you take, the day-to-day lifecycle commands are the same handful. The gotcha that costs money: **stopping** a GPU box halts compute billing but keeps the EBS volume (and its data) around, whereas **terminating** deletes the instance and, by default, its root volume — always stop, not terminate, an experimentation box you want to come back to:

```bash
aws ec2 describe-instances \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output table
aws ec2 stop-instances --instance-ids i-0abc123    # halts compute billing, keeps EBS
aws ec2 start-instances --instance-ids i-0abc123
aws ec2 terminate-instances --instance-ids i-0abc123   # deletes the instance (and root EBS)
```

## How this fits the whole ML solution

EC2 is the compute primitive under every other node in the system. Whether a service is "serverless" or "managed," somewhere a decision was made about instance family, accelerator, and pricing model — and those decisions dominate both your training speed and your bill. Knowing the families lets you pick the right SageMaker `instance_type`, size an EKS GPU node group, or decide that a nightly Spot training job on Trainium is the cheapest path to your next model.

## Key takeaways

- Instance names encode family/generation/size; P-family for training (P6 Blackwell, P5 H100/H200, P4d A100), G-family for inference (G6 L4/L40S, G5 A10G).
- Trainium and Inferentia are AWS accelerators programmed via the Neuron SDK — cheaper throughput, at the cost of framework compatibility.
- Use cluster placement groups plus EFA for multi-node distributed training.
- Use DLAMI/DLC images instead of installing CUDA by hand; reuse the same container across train and serve.
- Match pricing to workload: Spot for checkpointed training, Savings Plans for steady inference, Capacity Blocks to guarantee scarce GPUs.

## CLI cheat-sheet

```bash
# --- Discover instance types & availability ---
aws ec2 describe-instance-types --instance-types p5.48xlarge \
  --query 'InstanceTypes[0].{GPUs:GpuInfo.Gpus[0].Count,Net:NetworkInfo.NetworkPerformance}'
aws ec2 describe-instance-type-offerings \
  --filters Name=instance-type,Values='p5.*','g6.*','trn2.*' --output table
aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters Name=instance-type,Values=g6.2xlarge --query 'InstanceTypeOfferings[].Location'

# --- Resolve Deep Learning AMI ids from SSM (never hardcode) ---
aws ssm get-parameter \
  --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id \
  --query Parameter.Value --output text
aws ssm get-parameters-by-path --path /aws/service/deeplearning/ami/x86_64/ --recursive \
  --query 'Parameters[].Name' --output text

# --- Launch, inspect, lifecycle ---
aws ec2 run-instances --instance-type g6.2xlarge --count 1 \
  --image-id "$DLAMI" --key-name my-key --iam-instance-profile Name=ml-ec2-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]'
aws ec2 describe-instances --filters Name=instance-state-name,Values=running \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output table
aws ec2 stop-instances --instance-ids i-0abc123        # keeps EBS, halts billing
aws ec2 start-instances --instance-ids i-0abc123
aws ec2 terminate-instances --instance-ids i-0abc123   # deletes instance + root EBS

# --- Launch templates ---
aws ec2 create-launch-template --launch-template-name ml-gpu \
  --launch-template-data '{"InstanceType":"g6.2xlarge","IamInstanceProfile":{"Name":"ml-ec2-profile"}}'
aws ec2 run-instances --launch-template LaunchTemplateName=ml-gpu --count 1

# --- Spot ---
aws ec2 describe-spot-price-history --instance-types g6.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --query 'SpotPriceHistory[].[AvailabilityZone,SpotPrice]' --output table
aws ec2 run-instances --instance-type g6.2xlarge --count 1 --image-id "$DLAMI" --key-name my-key \
  --instance-market-options '{"MarketType":"spot"}'

# --- Guaranteed capacity: Capacity Blocks & Reservations ---
aws ec2 describe-capacity-block-offerings --instance-type p5.48xlarge --instance-count 16 \
  --start-date-range 2026-08-01T00:00:00Z --end-date-range 2026-08-15T00:00:00Z \
  --capacity-duration-hours 48
aws ec2 purchase-capacity-block --capacity-block-offering-id cb-0123 --instance-platform Linux/UNIX
aws ec2 create-capacity-reservation --instance-type g6.2xlarge \
  --instance-platform Linux/UNIX --availability-zone us-east-1a --instance-count 4

# --- Networking for distributed training ---
aws ec2 create-placement-group --group-name ml-cluster --strategy cluster
aws ec2 describe-instance-types --instance-types p5.48xlarge \
  --query 'InstanceTypes[0].NetworkInfo.EfaSupported'
```

## Try it

Launch a `g6.2xlarge` (or `g5.2xlarge`) from a Deep Learning AMI, SSH in, and confirm the GPU with `nvidia-smi`. Run a tiny PyTorch training loop that writes a checkpoint every N steps to a local `gp3` volume. Then repeat the launch as a **Spot** request and simulate an interruption by terminating mid-run; relaunch and verify your loop resumes from the last checkpoint. You have just built the core resilience pattern that makes Spot training economical.
