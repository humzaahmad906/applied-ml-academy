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

Most architect work at an established company is migration, not greenfield. Migrations fail for predictable reasons — big-bang cutovers with no rollback, no measurable success criteria, underestimating the hard-20% long tail (legacy pipelines, undocumented dependencies, niche use cases), and never committing to deprecating the old system. The durable patterns are the **strangler fig** (stand the new system up beside the old, migrate use cases piece by piece, turn the old one off once it has nothing left to do) and the **parallel run** (run both, compare outputs, cut over only when divergence stays under threshold), and every serious migration deserves its own ADR + design doc with explicit scope, phasing, validation, rollback, decommission criteria, cost model, and timeline. Plan for 1.5–2x your initial estimate and refuse mid-migration scope creep — even good ideas, especially good ideas, wait until after.

This is covered in depth in the MLOps course's ML Architect Track (Phase 7: Migration Strategy) — the principle is identical; what follows is the data-engineering-specific delta.

**Data-engineering delta — the warehouse parallel run.** For a warehouse migration the validation gate is concrete:

1. Build the new warehouse alongside the old
2. Run both pipelines in parallel — same sources, both destinations
3. Compare outputs daily; track divergence
4. When divergence is consistently <0.1%, switch BI tools to read from new
5. Keep old running for 1–2 quarters as fallback
6. Decommission old; reclaim costs

The daily output-comparison step (#3) is the one teams skip and the one that makes the migration succeed — without it you don't learn the new warehouse is "correct" until users complain. (The ML track's parallel-run example instead compares model predictions with a ~0.5% divergence gate.)

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

Data strategy is the process of translating business outcomes into a data-platform plan: what outcomes we want in 1–3 years, what data capabilities that requires, the current-state gap, the investments and sequencing to close it, and what measurable success looks like. It begins with a **diagnostic** — stakeholder interviews (10–20 people across the business), a system inventory, usage analysis, cost analysis, and an incident review — which produces the 10–20 page document nobody asked for and everybody needs. The output document runs executive summary, current state, future-state vision, a 12–24 month quarterly roadmap, an investment plan, a risk register, and success metrics. The hard part is **saying no**: pick 3–5 themes for the year and park everything else, because a strategy with 47 priorities has none.

This is covered in depth in the MLOps course's ML Architect Track (Phase 9: ML Strategy) — the process is identical; what follows is the data-engineering-specific delta.

**Data-engineering delta.** The diagnostic is what earns its keep — it changes the conversation from "we should buy Snowflake" to "the reason our growth team can't self-serve is that the marketing-attribution pipeline has 87 untested transformations on top of a poorly-modeled raw layer." Good public examples of the archetypal "Modern Data Stack" strategy doc are on Medium, the dbt blog, and consulting-firm white papers. (The ML track additionally enumerates the F50 ML-strategy themes for 2026; the equivalent data-platform themes — lakehouse consolidation, self-serve, governance, cost — recur throughout this track.)


---

## You can now

- Run a build-vs-buy analysis on true total cost of ownership (not just sticker price) and place a capability correctly among build, buy, adopt-open-source, or hybrid.
- Explain why big-bang migrations fail and design a phased migration using the strangler fig pattern or a parallel-run with an explicit validation gate.
- Build a capacity and cost model that projects spend at 1.5x/2x/5x growth, and name the FinOps practices (tagging, showback/chargeback, budgets, right-sizing) that keep cost visible.
- Recognize the hidden cost categories (data egress, idle clusters, cross-region queries, long-tail storage) that surprise teams who only track the obvious line items.
- Run the diagnostic phase of a data strategy engagement (stakeholder interviews, system inventory, usage analysis, cost analysis, incident review) and translate findings into a scoped roadmap with 3–5 themes.

## Try it

Pick one system your team currently operates (a database, an orchestrator, a BI tool — anything with a license or hosting cost). Build the true-TCO comparison from Phase 6: your current "build/self-host" cost including engineer time, ops burden, and outage risk, versus the "buy" alternative's all-in annual cost. Score both against the ten dimensions in the build-vs-buy framework. State which you'd recommend and why — and name the one dimension that would flip your answer if its weight changed.
