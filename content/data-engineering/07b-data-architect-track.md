# 07 — The Data Architect Track — Part 2 of 4: Build vs Buy, Migration, Cost, and Data Strategy

This is part 2 of 4 of the Data Architect Track. Part 1 covered what architects do, the architect mindset, ADRs, reference architectures, and architecture frameworks. Here we cover the four recurring decisions architects own: build vs buy, migration strategy, capacity and cost modeling, and translating business strategy into data strategy.

## Phase 6 — Build vs Buy

The decision an architect makes more than any other.

### The Frame

Every capability you need can be:

1. **Built** in-house from scratch
2. **Bought** as a managed/SaaS service
3. **Adopted** as open-source software you operate yourself
4. **Hybrid** — buy the core, build the integrations

The decision is rarely obvious. The default assumption ("we can build it cheaper") is almost always wrong; the opposite assumption ("just buy the SaaS") is almost always wrong too.

### The Total Cost of Ownership Calculation

What junior engineers see:

> "Self-hosted Airflow: $0/month. Astronomer Cloud: $2000/month. Save $24K/year."

What architects see:

> Self-hosted Airflow:
> - Engineer time to deploy: 4 weeks * $200/hr loaded = $32K
> - Ongoing operations: 0.3 FTE * $300K loaded = $90K/year
> - Outage costs (estimated): $40K/year (downtime, debugging)
> - Upgrade cycles every 6 months: $20K/year
> - **True annual cost: ~$150K**
>
> Astronomer:
> - License: $24K/year
> - Light operations: 0.05 FTE = $15K
> - **True annual cost: ~$40K**

Self-hosted "free" software almost always has hidden cost roughly equal to a small engineer-fraction. Buying the SaaS frees that engineer for higher-leverage work.

### The Build-vs-Buy Decision Framework

Score each option on these dimensions, weighted by your context:

| Dimension | Weight Notes |
|-----------|--------------|
| Total cost of ownership (3-year) | Always include hidden costs |
| Time to value | How long until usable in production |
| Strategic fit | Does this capability differentiate your business? |
| Vendor risk | What happens if the vendor disappears or doubles prices? |
| Talent availability | Can you hire people who know this tool? |
| Operational burden | Engineer-hours per month to keep it running |
| Switching cost | If we change our mind in 3 years, what does it cost? |
| Compliance & data residency | Especially in regulated industries |
| Integration with existing stack | Cost of glue code |
| Roadmap alignment | Is the vendor building what we need next? |

### When to Build

Build when:

- The capability is **core differentiation** for your business (e.g., a search company building its own search infrastructure)
- No off-the-shelf solution fits within tolerable customization
- You have the team and time
- The capability is small and you can build it well

### When to Buy

Buy when:

- The capability is **commodity** (orchestration, observability, BI tools)
- Vendor's roadmap matches yours
- The capability is mature and well-understood
- Your team's time is better spent on differentiated work

### When to Adopt Open Source

The middle path. Best when:

- You need ownership (audit, compliance, customization)
- Cloud lock-in is a real concern
- You have the operational skill to run it
- The OSS project is well-maintained (look at commit cadence, contributor diversity, governance)

The OSS choice is *not* free. Operating Kafka or Airflow yourself is a multi-person commitment. The savings vs SaaS only materialize at scale.

### The Hidden Trap: Vendor Lock-In

Buy decisions accumulate lock-in over time. Five years in:

- You've trained 30 engineers on Vendor X's specific quirks
- You've built 200 integrations against Vendor X's APIs
- You've negotiated complex pricing tied to Vendor X
- Switching means an 18-month migration

The buy decision wasn't wrong at the time. But you didn't preserve optionality. Architects mitigate by:

- Preferring vendors with open formats (Iceberg, Parquet, SQL)
- Building abstraction layers when the cost is small (a thin client over the vendor's API)
- Periodically running the "what would migration look like?" exercise
- Tracking vendor health (financials, leadership, support quality)

---

## Phase 7 — Migration Strategy

Most architect work, at established companies, is migration work. You're rarely starting greenfield. Mastery of migration patterns is therefore a core architect skill.

### Why Migrations Fail

1. **Big bang cutovers.** Migrating everything at once. The blast radius is the entire platform; the rollback is impossible.
2. **No measurable success criteria.** Migration "completes" when someone declares victory. Six months later, half the old system is still running.
3. **Underestimating the long tail.** Migrating the easy 80% is 20% of the work. The hard 20% (legacy pipelines, undocumented dependencies, niche use cases) is 80% of the work.
4. **Not committing to deprecation.** Two systems run in parallel forever. You now have *more* complexity, not less.

### The Strangler Fig Pattern

Named for the strangler fig vine that gradually replaces the tree it grows on. The canonical migration pattern:

1. New system stands up alongside the old
2. New use cases go to the new system
3. Old use cases migrate piece by piece
4. At some point, the old system has nothing left to do; you turn it off

This works because the blast radius at each step is small. If the new system has a problem, you stop migrating and fix it before continuing.

### Parallel Run Migrations

For warehouse migrations specifically:

1. Build the new warehouse alongside the old
2. Run both pipelines in parallel — same sources, both destinations
3. Compare outputs daily; track divergence
4. When divergence is consistently <0.1%, switch BI tools to read from new
5. Keep old running for 1–2 quarters as fallback
6. Decommission old; reclaim costs

The validation step (#3) is what most teams skip and what makes migrations succeed. Without it, you don't know whether the new warehouse is "correct" until users complain.

### The Migration ADR

Every serious migration deserves its own ADR + design doc combo. The design doc should include:

- **Scope:** what's in, what's out, what's deferred
- **Phasing:** the sequence of migrations
- **Validation:** how you'll know each phase succeeded
- **Rollback:** how to abort if something goes wrong
- **Decommission criteria:** what conditions allow you to turn off the old system
- **Cost model:** during overlap and steady-state
- **Timeline:** with explicit milestones

### The "We're Going to Be Done Soon" Trap

Migrations always run long. Always. Plan for 1.5–2x your initial estimate. Resist the urge to add scope ("while we're migrating, let's also...") — that's how a 6-month migration becomes 18.

Hard discipline: anything that would extend the timeline but isn't critical to migration *waits until after migration*. Even good ideas. Especially good ideas.

---

## Phase 8 — Capacity Planning and Cost Modeling

Architects own cost. Not "in a vague way." Literally — when the CFO asks why the data platform costs $4M/year, the architect has to answer.

### The Cost Model Skeleton

Build a spreadsheet (or Python notebook, or BI dashboard) that models:

```
For each layer of the stack:
  - Current monthly cost
  - Volume drivers (TB stored, queries/day, events/second, users)
  - Cost per unit
  - Projection at 1.5x, 2x, 5x growth
  - Sensitivity: what if a driver doubles?
```

Example for a warehouse:

| Driver | Current | 2x Growth | 5x Growth |
|--------|---------|-----------|-----------|
| Storage (TB) | 200 | 400 | 1000 |
| Compute hours/month | 1500 | 3000 | 7500 |
| Storage cost ($30/TB/mo) | $6K | $12K | $30K |
| Compute cost ($4/hr) | $6K | $12K | $30K |
| **Total monthly** | **$12K** | **$24K** | **$60K** |
| **Annual** | **$144K** | **$288K** | **$720K** |

When leadership asks "what happens if we onboard 3x more customers," you have an answer in 5 minutes instead of three weeks.

### The FinOps Discipline

FinOps = Cloud Financial Operations. The growing discipline of treating cloud cost as a first-class engineering concern. Worth a deep read of the [FinOps Foundation framework](https://www.finops.org/).

Key practices:

1. **Tagging everything.** Every resource tagged by team, project, environment, cost center.
2. **Showback / chargeback.** Each team sees what they cost. Showback (info only) is the starting point; chargeback (actual billing) creates the strongest incentives.
3. **Budgets and alerts.** Per-team monthly budgets with alerts at 50%, 80%, 100%.
4. **Right-sizing reviews.** Quarterly review of cluster sizes, instance types, warehouse sizes.
5. **Commitment management.** Reserved instances, Snowflake credits, BigQuery slots — negotiate annually.

### The Hidden Costs

What architects watch for that surprises everyone else:

- **Data egress.** Cross-region or cross-cloud data transfer is shockingly expensive. A single misconfigured replication can cost $10K/month.
- **NAT Gateway** on AWS. Lambda functions hitting S3 through NAT instead of VPC endpoints. Adds up fast.
- **Cross-region queries** on Snowflake. Pricing varies. Architects standardize regions.
- **Idle clusters.** EMR clusters left running. Dataproc clusters not auto-shutting down. The classic Monday-morning surprise.
- **Long-tail storage.** "Why do we have 30TB of log files from 2019?"

A simple weekly cost report flagging top movers prevents most of these.

### The 30% Rule

Most established data platforms can have 20–30% of their cost cut without affecting outcomes. The warehouse cost-audit portfolio project was practice for this. As an architect, you'll do this every 12–18 months.

The right way: a one-month focused engagement, find the 30%, present a plan, execute. Don't try to nibble at it continuously — make it a project, ship the result, move on.

---

## Phase 9 — Data Strategy

The art of translating business strategy into data platform strategy.

### What "Data Strategy" Actually Means

Not a buzzword. A specific document/process that answers:

- What business outcomes are we trying to enable in the next 1–3 years?
- What data capabilities does that require?
- What's the current state, and what's the gap?
- What investments close the gap?
- What's the sequencing?
- What does success look like, measurably?

### The Diagnostic Phase

Before recommending strategy, architects do a diagnosis:

1. **Stakeholder interviews.** 30-min conversations with 10–20 people across the business — execs, product, sales, analysts, scientists, ICs.
2. **System inventory.** What pipelines exist? What dashboards are watched? Where does data live?
3. **Usage analysis.** Which datasets get queried most? Which dashboards have actual users? What's used vs what's just running?
4. **Cost analysis.** Where's the money going? What's the unit economics?
5. **Incident review.** What broke in the last year and why?

The diagnosis output is usually a 10–20-page document that nobody asked for but everybody desperately needs. It changes the conversation from "we should buy Snowflake" to "the reason our growth team can't self-serve is that the marketing-attribution pipeline has 87 untested transformations on top of a poorly-modeled raw layer."

### Strategy Outputs

A data strategy document typically includes:

1. **Executive summary** (1 page — the only thing most execs read)
2. **Current state** (a few pages — diagnosis findings)
3. **Future state vision** (the target architecture)
4. **Roadmap** (12–24 months, quarterly milestones)
5. **Investment plan** (people + tooling cost)
6. **Risk register** (what could go wrong)
7. **Success metrics** (how we'll know)

Read examples online. The "Modern Data Stack" archetypal strategy doc has been written 1000 times; you can find good public examples on Medium, the dbt blog, and various consulting-firm white papers.

### The Hard Part: Saying No

A strategy document is also a list of things you're *not* doing. Architects who can't say no produce strategies with 47 priorities, which is the same as 0.

The discipline: pick 3–5 themes for the year. Everything else is parked. When new requests come in, they either fit a theme (resourced) or they don't (deferred). This frustrates people in the short term and saves the platform in the long term.


---

## You can now

- Run a build-vs-buy analysis on true total cost of ownership (not just sticker price) and place a capability correctly among build, buy, adopt-open-source, or hybrid.
- Explain why big-bang migrations fail and design a phased migration using the strangler fig pattern or a parallel-run with an explicit validation gate.
- Build a capacity and cost model that projects spend at 1.5x/2x/5x growth, and name the FinOps practices (tagging, showback/chargeback, budgets, right-sizing) that keep cost visible.
- Recognize the hidden cost categories (data egress, idle clusters, cross-region queries, long-tail storage) that surprise teams who only track the obvious line items.
- Run the diagnostic phase of a data strategy engagement (stakeholder interviews, system inventory, usage analysis, cost analysis, incident review) and translate findings into a scoped roadmap with 3–5 themes.

## Try it

Pick one system your team currently operates (a database, an orchestrator, a BI tool — anything with a license or hosting cost). Build the true-TCO comparison from Phase 6: your current "build/self-host" cost including engineer time, ops burden, and outage risk, versus the "buy" alternative's all-in annual cost. Score both against the ten dimensions in the build-vs-buy framework. State which you'd recommend and why — and name the one dimension that would flip your answer if its weight changed.
