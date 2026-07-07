# 08 — Use Cases and Mental Models: How Data Engineers and Data Architects Actually Think — Part 1 of 4: How to Read This Section, and Scenarios 1–2

The earlier sections teach the *tools*. This one teaches the *thinking*. Eight complex, realistic scenarios — each shown as a senior data engineer / staff IC architect would approach it, then again as a senior data solutions architect (or consulting / vendor SA) would approach it. You'll see two minds working the same problem.

By the end, when you read a job description that says "thinks systematically about data problems," you know exactly what that means and can do it on demand.

## How to Use This Section

Read each scenario actively:

1. **Read the problem statement** and pause 5 minutes. Think how you would approach it.
2. **Read the IC architect's approach.** Note the questions, the decomposition, the deliberate refusals.
3. **Read the SA approach.** Note the differences — discovery, multi-vendor honesty, customer politics.
4. **Compare to your initial take.** Add what the seniors thought of that you missed.

Each scenario uses the same structure:

- **The situation.** Real-feeling context.
- **What you're not told.** Unspoken questions.
- **IC architect's approach.** A staff/principal data engineer reasoning.
- **SA approach.** Customer-facing solutions architect reasoning.
- **Where they diverge.**
- **The proposed architecture.**
- **What they'd worry about in month 3.**
- **The interview-ready summary.**

---

## Scenario 1 — The Bank Whose Quarter-End Close Takes Five Days

### The Situation

A US regional bank's finance team takes 5 business days to close the quarter. Competitors do it in 2. The CFO has been told by an analyst that the bank is "running on a 1990s data stack" and feels exposed. The CTO calls in a senior data engineer or vendor SA:

> "We have Oracle on-prem. We have a legacy DataStage / Informatica setup running 2000+ jobs nightly. We have Tableau. We have a 30-person data team. We've tried to migrate to the cloud twice and both times it stalled. The CFO wants the close cycle cut in half this year. What do we do?"

You have 60 minutes.

### What You're Not Told

- **Why does close take 5 days?** Could be: data isn't ready, reconciliation is manual, the warehouse schema doesn't match how finance thinks, the close team's process itself is the bottleneck, regulatory adjustments slow the last day.
- **What did the prior migrations actually look like?** Two stalled migrations is a culture / political signal. Was it scope, sponsorship, vendor selection, or capability?
- **What's the close calendar?** Day 1: accruals. Day 2: reconciliation. Day 3: adjustments. Day 4: management review. Day 5: regulatory reporting. Different bottlenecks at each day.
- **Who's the actual user?** Controllers, FP&A, regulatory reporting team. Each has different needs.
- **What's the regulatory landscape?** SR 11-7 model risk for any model used in financial statements. SOX for the close process itself. CCAR / DFAST adjacency.
- **What's the team's skill?** A 30-person team trained on Informatica may not know dbt, Spark, Snowflake. Reskilling is part of the timeline.
- **What's the on-prem footprint?** Hardware refresh cycles. Existing licensing deals (Oracle, IBM).
- **Where do upstream systems live?** Core banking on a mainframe? GL on Oracle EBS? Each has its own integration story.

### IC Architect's Approach

A staff data architect at the bank reframes the conversation:

**The CFO asked for "cut close cycle in half" — what's actually causing the 5 days?** The architect's first move is not technical; it's diagnostic. Often the close cycle is gated by:

- **Day 1–2 wasted waiting for source data.** Mainframe batch finishes at 4am Day 2. Until then, downstream can't start. The fix is replication, not transformation speed.
- **Day 3 wasted on reconciliation.** Two systems disagree on the same number; finance hunts the difference manually. The fix is data quality at the source.
- **Day 4 wasted on management review iterations.** Numbers change every revision; the leadership team meets daily. The fix is self-service tooling for the controllers so they can answer questions without an engineer.
- **Day 5 wasted on the regulatory reporting team.** They build their numbers from scratch every quarter because they don't trust the warehouse. The fix is governance and lineage, not new tools.

A naive answer is "migrate to Snowflake and use dbt." A senior answer is "identify which day is the bottleneck and fix that specifically."

**The architect proposes a diagnostic week first:**

> "Before we recommend tools, give me a week with the finance team to map exactly where the 5 days go. Then I'll come back with a specific intervention. If we replace the data stack without solving the actual bottleneck, we'll spend $40M and still take 5 days."

The CFO usually agrees if the architect can credibly say "we'll have an answer next Friday."

**Likely findings (typical at regional banks):**

1. Day 1 is consumed by mainframe batch latency. The DB2 z/OS unload takes 6 hours.
2. Day 2 is consumed by reconciliation between GL and sub-ledger. Different code lists in each.
3. Day 3 is consumed by manual journal entries that should be automated.
4. Day 4 is consumed by management review iteration. Each change triggers re-runs of dependent reports.
5. Day 5 is regulatory: numbers must match GL exactly; reconciliation by hand.

**The architecture intervention (phased, not big-bang):**

```
Phase 1 (months 1–4): Source acceleration.
  - Add CDC from mainframe DB2 to a landing zone (Db2 Replication / Qlik / IBM CDC / Oracle GoldenGate).
  - Land in S3 (or ADLS / GCS); query via lakehouse engine (Iceberg + Trino, or Snowflake external tables).
  - Eliminates Day-1 latency.

Phase 2 (months 4–8): Reconciliation as code.
  - Build a unified semantic layer (dbt + a metrics layer) over GL and sub-ledger.
  - Automated reconciliation tests run continuously; differences surface as alerts, not Day-3 surprises.
  - Eliminates most of Day 2.

Phase 3 (months 8–14): Self-service controllers.
  - Build a finance-specific BI surface (Hex, ThoughtSpot, embedded analytics).
  - Pre-computed metrics that the controllers can slice without engineer help.
  - Eliminates Day-4 iteration delay.

Phase 4 (months 14–18): Regulatory reporting integration.
  - Lineage tooling so regulatory reporting team can trust warehouse numbers.
  - Audit trail per metric, per period, per change.
  - Eliminates Day-5 hand-reconciliation.

Throughout: keep Informatica running. It's the source of truth. Migration to dbt-on-Snowflake (or Iceberg-on-Trino) happens job by job, validated against the legacy output, retired only after parallel-run proves equivalence.
```

**The architect's deliberate refusals:**

- "We will not migrate everything in one cutover. Big-bang migrations at banks have a 90% failure rate."
- "We will not retire the mainframe. It's not a goal of this engagement."
- "We will not commit to halving the close cycle in 12 months. We'll commit to a Day-2 close in 18–24 months if Phase 1–3 land. The reason prior migrations stalled is they over-promised."
- "We will not lift-and-shift Informatica jobs to a new tool. We will redesign them, one domain at a time."

**Why the IC architect can say this:** They live with the consequences. An external SA promising 12 months wins the deal and leaves the customer with a broken expectation. The internal architect, accountable, calibrates.

### SA Approach (Cloud Vendor or Snowflake / Databricks)

A Snowflake or Databricks Senior SA with financial services specialty walks in:

**Discovery, with industry-specific framing:**

- "Which day of close is the painful one? Day 1, 3, 5?"
- "Walk me through your reconciliation between GL and sub-ledger. How long does that take?"
- "What's your CCAR / DFAST submission timeline? Does that pull into your close?"
- "What happened on your prior migrations? Was it sponsorship, scope, or capability?"
- "What's your stance on hybrid (Oracle stays, lakehouse for analytics) vs. full cloud?"

The SA collects:

- Specific bottleneck day (avoids the "migrate everything" trap)
- Existing reconciliation pain (their tool's headline use case)
- Regulatory pressure (creates urgency the customer didn't articulate)
- Migration history (signals what to do differently)
- Architecture philosophy (hybrid or all-in)

**The SA's typical structure for the response:**

> "Three observations before architecture. First, halving the close cycle is feasible but only if we attack the specific bottleneck day, not migrate the platform. Second, prior migrations stalled because they tried to replace everything; we've seen this 30 times. Third, the right path for you starts with a 'connect, don't migrate' move — Snowflake external tables on your existing data lake, or Iceberg tables that Snowflake reads. You don't have to lift and shift Oracle to get value. Then we extend from there."

**The SA's likely architecture recommendation:**

If Snowflake:

- Snowflake reading Iceberg tables on S3, fed by CDC from mainframe / Oracle
- dbt Core or dbt Cloud for transformations, replacing Informatica gradually
- Snowflake's Dynamic Tables for streaming-style transformations as a stepping stone
- Snowflake Time Travel for audit / reconciliation evidence
- Cortex (Snowflake's LLM service) for self-service Q&A *only if* the customer is ready (most banks aren't, yet)

If Databricks:

- Delta Lake on the existing data lake, fed by CDC
- Databricks SQL for analyst-facing queries
- dbt against Databricks
- Unity Catalog for governance
- Lineage end-to-end

**The SA explicitly compares paths:**

> "I can sell you Snowflake. But honestly, for your Oracle-heavy stack, Databricks may fit slightly more naturally because of the Spark heritage and Delta Lake integration with your existing big-data tooling. For your finance-team-driven workload, Snowflake's SQL-first experience tends to win adoption faster. Both are valid; the right answer depends on your team's skill profile. Let me ask: are your 30 people more SQL-leaning or Python/Spark-leaning?"

That honesty wins trust. The SA who only sells one product is detectable in 90 seconds.

**The SA pulls in:**

- A financial services industry SA
- A migration architect from the vendor's professional services arm
- A partner SI (Deloitte, Accenture, Slalom) for the people-side of migration
- Reference customers in similar size/sector

**The SA's distinctive value:** Migration playbooks. They've seen what works. The playbook for a regional bank migrating off DataStage typically:

1. Stand up the new platform parallel to old (3 months)
2. Pick one finance domain (e.g., loan loss reserves) as the pilot (3 months)
3. Parallel-run for 2 close cycles
4. Cut over that domain
5. Repeat for next domain
6. Decommission only after >80% of domains migrated

### Where the Two Diverge

| Concern | IC Architect | SA |
|---|---|---|
| Question of "should we migrate" | Will recommend "no" if the answer is no | Has incentive to recommend yes (deal); good SAs resist when honest |
| Vendor neutrality | Cares deeply (lock-in costs them) | Will recommend their own product; honest about fit |
| Day-100 ops | They own it | They hand off to Customer Success post-deal |
| Migration playbook | Built from scratch for their context | Brought from 30 prior engagements |

Both should converge on a phased plan with a specific bottleneck-day focus. The SA brings playbook leverage; the IC brings customization to internal politics.

### The Proposed Architecture

As above — phased, source-acceleration first, semantic layer second, self-service third, regulatory integration fourth. Both vendors and IC architects often converge here.

### What They'd Worry About in Month 3

- **The CFO has lost patience.** "When does the close get faster?" Need an early-win metric in Phase 1 — e.g., the loan-loss-reserves calculation now finishes Day 1 instead of Day 2.
- **The Informatica team feels threatened.** Reskilling plan from day 1. Promotion paths for engineers who migrate jobs successfully. Otherwise quiet quitting.
- **Reconciliation discovers actual data quality issues.** The bank has been masking them. Suddenly you're explaining to the controller why GL is $3M off from sub-ledger every month.
- **Regulatory examiners ask about the migration.** Document everything. SR 11-7-style documentation for the new analytical pipelines.
- **The mainframe team blocks CDC.** Operational risk concerns, throughput concerns. Get the mainframe team in the room from day 1 or the project stalls.

### Interview-Ready Summary

> "Don't migrate. Diagnose. Find the bottleneck day — usually Day 1 (source latency), Day 3 (reconciliation), or Day 4 (self-service iteration). Attack that day first with a phased intervention: CDC from mainframe / Oracle for Day-1 acceleration, semantic layer for Day-2 reconciliation, self-service BI for Day-4. Keep Informatica running until each domain is parallel-run validated. Commit to a Day-2 close in 18–24 months, not 12. The reason prior migrations stalled is they over-promised and tried to migrate everything. The right move is phased, with early wins and ruthless scope discipline."

---

## Scenario 2 — The Marketplace Whose Recommendation Engine Eats All the Compute

### The Situation

An online marketplace ($1.2B GMV) runs nightly batch jobs on Spark on EMR. The data engineering bill has grown 6x in 18 months — from $30K/month to $180K/month. Most of the growth is in the "recommendations training" pipeline. The CTO wants the bill to plateau or shrink, not because the company can't afford it, but because the growth rate is unsustainable.

The Director of Engineering says: "We've doubled the data and doubled the models. The bill 6x'd. Something's wrong. Can you find it?"

### What You're Not Told

- **What does "recommendations training" mean here?** Could be one feature pipeline running 200 model retrains, or 200 separate pipelines each retraining one model.
- **What's the cluster utilization?** A cluster running at 30% utilization for 8 hours costs the same as 100% for 8 hours.
- **What's the failure / retry rate?** Jobs that fail and retry burn compute twice.
- **What's the actual data volume per job vs. the cluster size?** Spark clusters are often over-provisioned because someone copied a template.
- **What's the storage cost?** Often the secret cost grower.
- **Are jobs writing intermediate data they don't need to?** Shuffles and intermediate writes can cost more than the final output.
- **Is there job-level cost attribution?** Most teams don't tag, so they can't see.
- **Who decides cluster sizes?** Usually whoever copy-pasted the template last.

### IC Architect's Approach

A staff data engineer hired to investigate doesn't start by reading code. They start by reading the bill.

**Step 1: Cost attribution.** Pull the AWS cost explorer with EMR tags. If tags don't exist, tag the clusters by job name. Build a per-job-per-day cost view. Within a week, you know which 5 jobs are eating 80% of the budget.

**Step 2: For each top-5 job, profile.** What's the cluster size? What's the actual runtime? What's the Spark UI showing?

Typical findings at marketplaces:

1. **The "user features" job runs on 200 r5.4xlarge nodes for 6 hours.** Spark UI shows skewed joins. One key (user_id 0, the "anonymous" user) has 40% of the rows. The job spends 5 of 6 hours on that one key. The fix is salt the skewed key — 30 minutes of work, eliminates 80% of the job's cost.
2. **The "session aggregation" job re-reads the entire 18-month event log every night.** It's an incremental pipeline that forgot how to be incremental. Fix is partitioned writes + only-read-new-partitions. Saves 70%.
3. **The model training job is running on GPUs but spending most of its time on data loading.** GPU utilization 15%. Move data loading to CPU pre-stage; GPUs run only for training. Saves 60%.
4. **Three different jobs compute "items viewed in last 7 days per user."** Each was built independently. Consolidate into a feature store; each job reads the precomputed feature. Saves 40%.
5. **The orchestrator retries failed jobs 3 times. The 80%-failing job is therefore costing 4x.** Find why it fails; fix the underlying issue. Often a flaky downstream API or a memory limit.

**Step 3: Pricing structure.** EMR on-demand is the most expensive option. The architect typically recommends:

- Spot instances for the worker nodes (60–70% savings; jobs handle preemption via checkpointing)
- Reserved or Savings Plan for the master nodes
- EMR Serverless for jobs with bursty patterns

**Step 4: Storage cost.** Often missed. The architect checks:

- Are intermediate files being lifecycled? S3 storage at hot tier for files no one reads is shockingly expensive.
- Are old job artifacts retained forever? Add a 30-day lifecycle to job logs.
- Are old EMR cluster logs being deleted? They accumulate.

**Step 5: The "kill it" candidates.** Some jobs run nightly because they ran nightly last year. Nobody owns them. Nobody checks the output. The architect proposes:

> "Job X has been running for 3 years. Last week I changed its output to NULL. Nobody noticed. I'm proposing we turn it off. If something breaks, we'll know by next Tuesday."

About 5–10% of nightly jobs at any marketplace can be killed this way. Real money.

**The architect's deliverable:**

- A per-job cost table (before / after)
- A prioritized fix list
- A "kill list" of jobs to retire
- An estimate: typical first-pass reduction is 40–50%

### SA Approach (Databricks, Snowflake, EMR Specialist, or FinOps Vendor)

An SA in this conversation, especially from Databricks or a FinOps tool, thinks:

**This is a Databricks Photon / Snowflake / FinOps-Vantage / Cloudability conversation.** The customer is on EMR; the SA's job is honest comparison.

**Discovery:**

- "What workload mix? Streaming, batch, ad-hoc?"
- "How elastic is your demand? Is your evening batch window the same every night?"
- "What's your team's Python vs. SQL split? Notebook-heavy or job-heavy?"
- "Are you using Iceberg/Delta yet, or raw Parquet?"

**The SA's honest framing:**

> "Several things will help here. Some you can do without changing platforms — better tagging, spot, kill the dead jobs, fix the skew. That alone is probably 30%. Beyond that, if you want to consider platform options, Databricks' Photon engine often gives another 30–50% on Spark workloads through query optimization and the C++ vectorized engine. But that's a migration project, not a quick win. Start with the free wins; then evaluate."

The SA explicitly *separates* the cheap wins (any vendor) from the platform wins (their vendor). They tell the customer to do the cheap wins first. This is what a good SA does: deliver value before selling.

**If Databricks pitches:**

- Photon engine for compiled SQL
- Job clusters with auto-scaling (replacing fixed EMR pools)
- Delta Lake with OPTIMIZE for file compaction
- Unity Catalog for governance + lineage (often the hidden value)

**If a FinOps tool pitches:**

- Per-cluster, per-job cost attribution out of the box
- Recommendations engine (Spot, right-sizing, scheduled scaling)
- Anomaly detection on cost (alert when a job's cost spikes)

**The SA's value-add:**

- They've seen which optimizations work for which workload shapes
- They have a benchmark library: "marketplaces of your size typically run X TB/day; you're at Y; here's why you're 3x over baseline"
- They can connect to similar customers

### Where the Two Diverge

The IC engineer can change the code. The SA usually can't. The SA's leverage is in framing, sizing, and reference patterns. Both arrive at the same first-pass diagnosis: skewed joins, dead jobs, non-incremental incremental jobs, retry loops, storage waste. The vendor SA adds the "and here's a platform that would help further."

### The Proposed Architecture (After Diagnosis)

```
Before:
  EMR on-demand × 12 always-on clusters
  Dead jobs running nightly
  No tagging, no cost attribution
  Skewed shuffles, full-rewrite "incremental" jobs
  Storage at hot tier indefinitely

After (Phase 1, weeks 1–8):
  Tag all jobs by team/feature/owner
  Per-job cost dashboard
  Kill 15% of jobs (proven unused)
  Move 80% of compute to spot
  Fix skew on top-5 jobs
  Make incremental jobs actually incremental
  Lifecycle policies on storage

After (Phase 2, months 3–6):
  Migrate to EMR Serverless or Databricks for the bursty workloads
  Feature consolidation: shared feature store for repeated computations
  Iceberg or Delta tables for ACID + better optimization
  Anomaly detection on cost via FinOps tool

Result: bill 40–50% lower at same data volume.
```

### What They'd Worry About in Month 3

- **The savings start to erode.** New jobs get added. Without ongoing discipline, the bill creeps back. Need recurring cost reviews.
- **The kill-the-dead-jobs strategy hits a job someone secretly relied on.** Have a 2-week "deprecation window" with logs of failed lookups before deleting.
- **Spot interruptions on critical jobs.** Move critical jobs to mixed-instance fleets, not pure spot.
- **The "I want a bigger cluster" requests.** Engineers will instinctively scale up to fix bugs. Add a review process; require a profile screenshot before approving cluster growth.

### Interview-Ready Summary

> "Don't migrate platforms first. Diagnose first. Most marketplaces' Spark bills are eaten by 5 jobs out of 200. Top causes: skewed joins, non-incremental 'incremental' jobs, redundant feature computations, low GPU utilization on data-loading-bound jobs, dead jobs nobody owns, all-on-demand pricing. First-pass fix is 40–50% reduction without any platform change. Tag everything, attribute cost per-job-per-day, kill the dead jobs, fix the skew, make incremental actually incremental, move 80% to spot, lifecycle storage. *Then* consider platform changes if needed. The senior move is to fix the cheap stuff first."


---

## You can now

- Read a vague, high-stakes problem statement (a bank's slow quarter-close, a marketplace's runaway compute bill) and generate the list of unstated questions a senior practitioner would ask before proposing anything.
- Reframe "migrate everything" asks into a diagnostic-first response, and explain why a bottleneck-day or per-job cost attribution approach beats a platform swap as the first move.
- Compare how an internal IC architect and a vendor SA approach the same problem — what each is incentivized to say, and where their advice legitimately converges.
- Recite the interview-ready summary pattern: diagnose, phase the fix, name the deliberate refusals, state a realistic (not aspirational) timeline.

## Try it

Before reading further scenarios, pick a real cost or performance complaint from your own work (or a job posting's "our data platform costs too much / is too slow" framing) and write your own version of Scenario 2's Step 1–2: what per-job or per-tenant cost attribution would you build first, and what are the top 3 candidate root causes you'd check before recommending any tool or platform change? Compare your answer to the scenario's diagnosis once you've written it down — don't peek first.
