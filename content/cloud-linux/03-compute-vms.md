# 03 — Compute: Virtual Machines

At its heart, the cloud is a way to rent computers by the hour instead of buying them. The most fundamental thing you can rent is a **virtual machine** (VM): a full computer, with a CPU, memory, and disk, that you can log into and control. Different providers give it different names — an EC2 instance on AWS, a Compute Engine VM on Google Cloud, a Virtual Machine on Azure — but the idea is identical. This lesson explains what a VM actually is and how to choose one.

## What "virtual" means

A cloud provider owns enormous physical servers. Rather than hand you a whole machine, they use software called a **hypervisor** to slice one big physical server into many isolated virtual machines. Each VM behaves like its own independent computer, unaware of the neighbors sharing the same hardware.

This is what makes the cloud economical and flexible. You can rent a small slice for a hobby project, or a huge slice for a heavy job, and the provider bills you for exactly what you use. When you're done, you release it and stop paying.

The practical upshot: a VM feels exactly like a real Linux (or Windows) machine. You get an IP address, you log in, you install software, you run programs. Everything from the first two lessons applies directly.

## Connecting to a VM

For a Linux VM, you connect over **SSH** (Secure Shell), an encrypted remote login. Instead of a password, you typically use a **key pair**: a private key that stays on your laptop and a public key the server holds. If they match, you're let in.

```bash
ssh -i ~/.ssh/my-key.pem maya@203.0.113.42
```

Here `-i` points to your private key file, `maya` is the username on the server, and the number is the VM's public IP address. Once connected, your terminal is now controlling the remote machine.

## Anatomy of an instance type

When you create a VM, you pick an **instance type** (or "machine type"/"VM size"), which is a bundle of resources. The main dimensions:

- **vCPUs** — how many virtual processor cores. More cores means more parallel work.
- **Memory (RAM)** — working space for running programs. Data-heavy jobs need more.
- **Storage** — disk attached to the machine (covered in the next lesson).
- **Network bandwidth** — how fast data moves in and out.
- **Accelerators** — optional GPUs for machine learning, graphics, or scientific computing.

Providers group instance types into **families** tuned for different jobs:

- **General purpose** — a balanced mix of CPU and memory. A safe default for web apps and mixed workloads.
- **Compute optimized** — lots of CPU relative to memory. Good for number crunching and busy web servers.
- **Memory optimized** — lots of RAM. Good for large databases and in-memory data processing.
- **GPU / accelerated** — includes graphics cards. This is what you rent to train or serve machine learning models.

Within a family, sizes scale up in roughly doubling steps: a "large" might have 2 vCPUs and 8 GB of RAM, an "xlarge" 4 vCPUs and 16 GB, and so on. Start small; you can almost always resize later.

## Images: what the machine boots with

A VM starts from an **image** — a snapshot of an operating system and any pre-installed software. You might boot from a plain Ubuntu image, or from a specialized image that already has GPU drivers and machine learning libraries installed, so you don't have to set them up by hand.

You can also make your own image after configuring a machine the way you like, then launch identical copies from it. This is how teams keep fleets of servers consistent.

## Pricing models

How you pay for a VM matters as much as which one you pick:

- **On-demand** — pay per second or hour, no commitment. The most flexible and the most expensive. Best for short or unpredictable work.
- **Reserved / committed** — commit to a year or more in exchange for a large discount. Best for steady, always-on workloads.
- **Spot / preemptible** — rent spare capacity at a steep discount, but the provider can reclaim the machine with little warning. Excellent for fault-tolerant batch jobs (like many ML training runs that checkpoint their progress), risky for anything that must stay up.

A common mistake is leaving an on-demand GPU instance running overnight after a job finishes. The meter keeps running whether you're using it or not, so **stopping** or **terminating** idle machines is the single biggest way to control cost.

## Stop versus terminate

Two different "off" buttons cause a lot of confusion:

- **Stopping** a VM shuts it down but keeps its disk. You stop paying for the compute, keep paying a little for storage, and can start it back up later with everything intact.
- **Terminating** (or deleting) a VM throws it away entirely, usually including its disk. Cheaper, but permanent.

Stop when you'll come back tomorrow. Terminate when you're truly done and have saved anything you need elsewhere.

## Key takeaways

- A cloud VM is a full computer you rent, carved out of a big physical server by a hypervisor.
- You connect to Linux VMs over SSH, usually with a private/public key pair.
- Instance types bundle vCPUs, RAM, storage, network, and optional GPUs; families are tuned for general, compute, memory, or accelerated work.
- A VM boots from an image — a snapshot of an OS plus pre-installed software.
- Pricing ranges from flexible-but-pricey on-demand, to discounted reserved, to cheap-but-interruptible spot.
- Stopping keeps the disk and pauses compute charges; terminating deletes the machine for good. Idle machines still cost money.

## Try it

You don't need to spend real money to think this through. Pick a small project you'd like to run in the cloud — say, a personal website, or a script that processes a batch of images.

1. Write down what it needs: roughly how many CPUs, how much memory, and whether it needs a GPU.
2. Match it to an instance family (general, compute, memory, or accelerated) and justify the choice in one sentence.
3. Decide on a pricing model. Would on-demand, reserved, or spot fit best, and why?
4. If your provider offers a free tier, launch the smallest eligible Linux VM, connect to it with `ssh`, run `htop` to inspect its resources, then **stop or terminate it** so you're not billed.

Being able to reason from "what my job needs" to "which machine to rent" is the whole skill.
