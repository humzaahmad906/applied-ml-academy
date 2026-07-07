# 12 — Cloud from Basics to ML Expert (DL-Focused) — Part 8 of 8: Capstone Scenarios & Interview Prep (Part E)

This is the final part, 8 of 8, of the "Cloud from Basics to ML Expert" lesson. Parts 1–7 covered universal foundations, AWS in depth, GCP/Azure/multi-cloud, and DL-specific cloud patterns. Here we cover Part E: four capstone scenarios that integrate everything from the earlier parts, revised interview-prep guidance for the lesson now that it spans eight files, the chapter-level capability summary, the greenfield capstone project, and space for your own notes.

---

## Part E — Capstone Scenarios

### Scenario E1 — Greenfield DL platform on AWS

You're hired into a F500 that's done some MLOps but never DL at production scale. Mandate: stand up a DL platform that can support 5 model teams. Budget: $1.5 M / year. Timeline: 12 months.

Sketch the architecture, the AWS service choices, the team org, the cost model, the migration sequence, and the operational concerns. Use the question prompts above as the surface area you cover.

### Scenario E2 — Migrate from SageMaker to self-hosted EKS

Your team is currently fully on SageMaker. Cost is exploding (model serving + notebook hours). You suspect 40% can be saved by moving to EKS + KServe + vLLM. Design the migration: ADRs, parallel-run validation, decommission criteria, training the team on EKS, the new on-call rotation, the cost model proof.

### Scenario E3 — Multi-region LLM serving for global enterprise

You serve an internal LLM platform to a global F500 (users in US, EU, APAC). Requirements: P95 TTFT < 1 second from any region, data residency for EU (no traffic leaves EU), failover within 60 seconds of regional outage, per-tenant cost attribution. Walk through architecture across regions, model artifact distribution, gateway routing, observability, and the operational model.

### Scenario E4 — Train and serve a CV model fleet on a budget

You're at a smaller F500. You need to train and serve 12 CV models (image classification, object detection, semantic segmentation) across product teams. Total budget: $200K / year for cloud. Pick the cloud, the architecture, the training rhythm, the serving stack. Show the math.

---

## How to Use This Chapter for Interview Prep

This lesson is split across eight files (`12`, `12b` … `12h`) so each sitting stays to a focused session. A pacing plan across all eight:

1. **Part 1 — Part A foundations.** One session. Master each "F500 Q" before moving on; these are the universal foundations every later part assumes.
2. **Parts 2–5 — Part B, AWS in depth.** Four sessions, one per file (account topology/IAM/VPC, then EC2/S3/EKS, then SageMaker/Bedrock, then data/observability/cost). AWS depth is a long game — after each subsection, answer the F500 Q's aloud.
3. **Part 6 — Part C, GCP and Azure.** Skim it unless you're targeting a GCP-heavy or Azure-heavy F500.
4. **Part 7 — Part D, DL-specific patterns.** Hit it twice. This is the DL-cloud intersection where most F500 senior interviews score you.
5. **Part 8 (this file) — Part E capstones.** Take it like a system-design round. Time-box each scenario, write notes, sketch architecture, then rehearse the verbal answer for 5 minutes.

In total, the eight files are about 8–12 hours of careful reading + practice. Done well, they close the cloud gap for a DL engineer who already has the modeling background.

The remaining DL specializations — RAG architecture, fine-tuning factory, multi-tenant LLM platform, real-time CV serving — are covered in the practitioner and specialization chapters of this course. With the cloud vocabulary from this lesson, you can consume all of them at depth.


---

## You can now

- Design a multi-account AWS topology with OUs, SCPs, and least-privilege IAM — distinguishing trust from permission policies and wiring up IRSA, cross-account AssumeRole, and OIDC federation for CI.
- Size VPCs and subnets with CIDR math, and cut data-transfer cost by routing S3 / ECR / DynamoDB traffic through VPC endpoints instead of NAT.
- Pick the right GPU instance, storage tier, and serving path for a workload — reasoning through SageMaker vs Bedrock vs self-hosted vLLM on EKS with real cost and latency numbers.
- Diagnose distributed-training and inference bottlenecks: silent NCCL-over-TCP fallback, placement-group misconfiguration, DataLoader starvation, and GPU under-utilization.
- Run a GPU FinOps program end to end — tagging, right-sizing, spot with checkpointing, reserved capacity, and scale-to-zero — to cut spend without a quality regression.

---

## Capstone Project — F500-Grade Greenfield AWS Account Topology + EKS DL Cluster

_Anchored on the section: **Part B — AWS in Depth**. The headline build that turns this chapter's knowledge into a Fortune 500 portfolio artifact._

### What you'll build

A complete F500-style AWS bring-up: (1) AWS Control Tower with 4 OUs + 4 accounts (Security, Shared Services, ML-Platform, Workload), (2) SCPs enforcing 'no GPU instances outside approved regions' and 'no public S3,' (3) EKS cluster in ML-Platform account with Karpenter + GPU Operator + IRSA, (4) cross-account S3 read from EKS pod to training-data bucket in Workload account, (5) vLLM serving Qwen2.5-7B-AWQ at 50 RPS with VPC endpoint for ECR, (6) CloudWatch + AMP + Managed Grafana, (7) CUR → Athena query showing per-tag cost, (8) one-page architecture diagram. Tear down nightly.

### Skills demonstrated

- AWS Organizations + Control Tower
- SCPs
- EKS + Karpenter + GPU Operator + IRSA
- cross-account IAM trust
- VPC endpoints for cost
- Managed Prometheus + Grafana
- CUR + Athena
- Bedrock vs self-hosted LLM trade-offs

### Tech stack

`AWS Control Tower + Organizations` · `Terraform 1.9+ for everything` · `EKS 1.30+ with Karpenter 0.37+` · `NVIDIA GPU Operator + EFA driver` · `vLLM 0.6+ on g5.2xlarge` · `Qwen2.5-7B-Instruct-AWQ` · `AMP (Managed Prometheus) + Managed Grafana` · `CUR → Athena → QuickSight (optional)`

### Acceptance criteria

- [ ] 4 AWS accounts under Control Tower with SCPs enforced
- [ ] EKS cluster up; vLLM serving at 50 RPS with measured P95 TTFT
- [ ] Cross-account S3 read via IRSA proven
- [ ] CUR shows per-tag cost in Athena
- [ ] Tear-down script saves >80% nightly cost (kill GPU node pool)
- [ ] Architecture diagram committed (drawio or Excalidraw)

### Fortune 500 talking point

> I have run a Control-Tower-managed multi-account AWS with EKS + GPU Operator + IRSA + vLLM + CUR for a real workload. Most candidates have touched maybe half of this. AWS-heavy F500 (most of them) interview score senior on this depth.

**Estimated time:** 30

**Stretch goal:** Add SageMaker HyperPod as a comparison training environment; document the operator-experience delta vs raw EKS.



---

## 📝 Your Notes

> Take 10 minutes after this chapter. Answer below and revisit weekly.

**What surprised me in this chapter?**

_(write here)_

**Three open questions I still need to answer:**

1.
2.
3.

**Code snippets, links, or portfolio references to remember:**

-
-
-

**Concepts I should re-explain aloud to verify understanding:**

-
-

---
