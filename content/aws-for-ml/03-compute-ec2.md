# 03 — Compute: EC2 for ML

Elastic Compute Cloud is raw virtual servers you rent by the second. Higher-level ML services like SageMaker run *on top of* EC2, so understanding the instance zoo, how AMIs and pricing work, and when to drop down to bare EC2 is foundational even if you spend most of your time in managed services. This module maps the compute landscape an ML engineer actually cares about: which accelerators exist, how to pay less for them, and when EC2 is the right tool versus a managed alternative.

## Instance families and the naming convention

An instance type name encodes a lot: `p5.48xlarge` is family `p`, generation `5`, size `48xlarge`. General patterns you will meet:

- **General purpose** (`m`, `t`): balanced CPU/memory, fine for data prep, orchestration, small inference. `t` instances are burstable and cheap for intermittent work.
- **Compute optimized** (`c`): high CPU-to-memory ratio, good for CPU inference and feature engineering.
- **Memory optimized** (`r`, `x`): large RAM for big data frames, in-memory joins, embedding stores.
- **Storage optimized** (`i`): fast local NVMe for shuffling large datasets.

For ML the families that matter most are the accelerated ones.

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

## AMIs and containers

An **Amazon Machine Image** is the disk template an instance boots from. Rather than installing CUDA and drivers yourself, use the **Deep Learning AMI (DLAMI)**, which ships with GPU drivers, CUDA, cuDNN, and preinstalled frameworks. For containerized workflows, **AWS Deep Learning Containers (DLCs)** are prebuilt, optimized images for PyTorch/TensorFlow that run on EC2, ECS, EKS, and SageMaker — using the same DLC across training and serving keeps your environment consistent end to end.

```bash
# Launch a GPU instance from a Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0deeplearningxxxx \
  --instance-type g6.2xlarge \
  --key-name my-key \
  --iam-instance-profile Name=ml-ec2-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]'
```

## Pricing: Spot, On-Demand, Savings Plans, Capacity Blocks

You pay for EC2 four main ways:

- **On-Demand**: pay per second, no commitment. Maximum flexibility, highest price.
- **Spot**: spare capacity at up to ~90% off, but AWS can reclaim it with a two-minute warning. Ideal for **fault-tolerant training** that checkpoints frequently — you resume from the last checkpoint after an interruption. Not for stateful, uninterruptible work.
- **Savings Plans / Reserved Instances**: commit to a steady spend (1 or 3 years) for a large discount. Right for baseline inference capacity that runs continuously.
- **Capacity Blocks for ML**: reserve a block of GPU capacity (for example P6e-GB200 or P5) for a defined future window. This is how you *guarantee* you will actually get scarce accelerators for a planned training run, since On-Demand GPU capacity is frequently exhausted.

The engineering consequence: architect training to checkpoint often so you can ride Spot, and reserve Capacity Blocks when a run absolutely must start on time.

## When to use EC2 versus managed services

Reach for raw EC2 when you need full control of the environment, custom drivers or kernels, a long-lived experimentation box, or a serving stack that does not fit SageMaker or Lambda. Reach for **SageMaker** when you want managed training jobs, endpoints, and MLOps without operating servers. Reach for **containers on ECS/EKS** when you already run Kubernetes or need portable, multi-framework serving. EC2 is the floor everything else stands on; most teams use it directly for a handful of things and rely on managed layers for the rest.

## How this fits the whole ML solution

EC2 is the compute primitive under every other node in the system. Whether a service is "serverless" or "managed," somewhere a decision was made about instance family, accelerator, and pricing model — and those decisions dominate both your training speed and your bill. Knowing the families lets you pick the right SageMaker `instance_type`, size an EKS GPU node group, or decide that a nightly Spot training job on Trainium is the cheapest path to your next model.

## Key takeaways

- Instance names encode family/generation/size; P-family for training (P6 Blackwell, P5 H100/H200, P4d A100), G-family for inference (G6 L4/L40S, G5 A10G).
- Trainium and Inferentia are AWS accelerators programmed via the Neuron SDK — cheaper throughput, at the cost of framework compatibility.
- Use cluster placement groups plus EFA for multi-node distributed training.
- Use DLAMI/DLC images instead of installing CUDA by hand; reuse the same container across train and serve.
- Match pricing to workload: Spot for checkpointed training, Savings Plans for steady inference, Capacity Blocks to guarantee scarce GPUs.

## Try it

Launch a `g6.2xlarge` (or `g5.2xlarge`) from a Deep Learning AMI, SSH in, and confirm the GPU with `nvidia-smi`. Run a tiny PyTorch training loop that writes a checkpoint every N steps to a local `gp3` volume. Then repeat the launch as a **Spot** request and simulate an interruption by terminating mid-run; relaunch and verify your loop resumes from the last checkpoint. You have just built the core resilience pattern that makes Spot training economical.
