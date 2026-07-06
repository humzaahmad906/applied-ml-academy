# Module 00 — Syllabus: The Road to Principal

## Why this module matters

This course has one goal: to compress into roughly fifty hours of deliberate work the judgment that normally takes fifteen years to accumulate. "Fifteen years of experience" is not a mystical quantity — it decomposes into a pattern library of systems seen and failures survived, a set of decision frameworks applied enough times to be calibrated, and the writing and influence skills to make an organization act on them. All three are learnable faster than they are usually learned, because most engineers acquire them by accident. This syllabus is the map; read it once now and again whenever you lose the thread of why a module exists.

## What this course is

You are a solid senior ML engineer. You can design a training pipeline, debug a serving regression, and pass an L5/L6 system-design loop. The gap between you and a principal engineer (Google L7/L8, Meta E7, Amazon Principal) is not more of the same — it is a different job. The unit of work changes from "feature shipped" to "decision made correctly across a 2–5 year horizon." The output changes from code to strategy documents, standards, reviews, and unblocked teams. The accountability changes from "my service is healthy" to "the company's ML investment is pointed in the right direction."

This course teaches that job. It sits above the academy's ML System Design and MLOps courses: those teach the mechanics of building systems; this one teaches deciding which systems to build, when to kill them, how to pay for them, and how to move a hundred engineers without authority over any of them. Where a topic's mechanics are covered elsewhere, we say so and stay at the decision level.

## Who it's for

- Senior ML engineers (5–8 years) targeting Staff/Principal promotion or an external L7+ loop.
- Staff engineers who got the title but are still operating like seniors with more meetings.
- Engineering managers moving back to a senior-IC track who need the IC-leadership toolkit.
- Not for: engineers still building their first production systems. Do the ML System Design course first; this course assumes you already know what a feature store is and why offline metrics lie.

### Prerequisites self-check

You should be able to answer all of these without looking anything up. If more than two are shaky, do the ML System Design and MLOps courses first:

- Explain training-serving skew and name three mitigations.
- Compute a $/request estimate for an LLM feature from token counts and published rates, and state where the API-vs-self-host crossover lives.
- Describe what a point-in-time-correct feature join is and why it matters.
- Sketch the offline/online plane decomposition of a production ML system from memory.
- Explain why offline metric lifts sometimes fail to move online metrics, with two concrete mechanisms.
- Describe a shadow deployment and a progressive rollout, and when each is insufficient.
- Say what an eval golden set is and why judge-model scores need bias auditing.
- Read a distributed training job's cost from GPU count, hourly rate, and wall-clock time.

## The module map

**Module 01 — The Principal Delta.** What actually separates senior from principal, stripped of mythology. The level-ladder delta across scope, horizon, output, coding ratio, ambiguity, and business accountability; Larson's archetypes as situational modes rather than identities; the concrete weekly operating cadence of a working principal; and the four anti-patterns (astronaut, gatekeeper, ghost, hero) that kill people at this level. Ends with a gap analysis of your own last month against the ladder.

**Module 02 — Technical Strategy.** Strategy is a diagnosis, a set of guiding policies, and coherent actions — not a roadmap with quarters on it. How to diagnose an ML org (inventory the models, pipelines, teams, and spend; find the constraint), write policies that actually forbid things, and produce the one-page document that moves the org. Includes a complete strategy-doc template and a full worked strategy for a fictional org drowning in serving stacks.

**Module 03 — ML Platform Architecture.** The org-level platform decision: when a company needs an ML platform team at all, the build-order of platform capabilities (registry, features, training orchestration, serving, evals, observability), platform-as-product discipline, and how to avoid the platform team becoming a bottleneck the product teams route around. Golden paths versus mandates.

**Module 04 — Training Infrastructure.** Capacity planning and economics of training at org scale: GPU fleet sizing, scheduler and quota design, the reserved-versus-spot-versus-cloud-committed portfolio, distributed-training failure economics (a 4% failure rate on 2 000-GPU jobs is a budget line), and when fine-tuning infrastructure deserves dedicated headcount.

**Module 05 — Inference & Serving.** The serving estate as a portfolio: consolidating stacks, latency-tier taxonomy, the API-versus-self-host crossover computed honestly with utilization, multi-model routing and cascades as economic instruments, and capacity/failover design that survives a region loss. The principal's job is the estate, not the endpoint.

**Module 06 — Data Strategy & Flywheels.** Proprietary interaction data as the only durable ML moat. Designing feedback loops as first-class architecture, data contracts and ownership boundaries, labeling economics (when to build a labeling org versus buy), and the multi-year sequencing that turns exhaust into training sets competitors cannot buy.

**Module 07 — Evaluation Discipline.** Evals as the org's immune system: the eval hierarchy from unit-level golden sets to A/B experiments to long-term holdbacks, eval governance ("no new model without an eval" as enforceable policy), judge-model bias management, and the economics of eval infrastructure — why underfunding evals is the most expensive saving an ML org makes.

**Module 08 — Migrations.** Principals inherit migrations the way surgeons inherit trauma cases. Strangler patterns for ML systems, shadow-mode and progressive-traffic playbooks, the parity-metric trap (bug-for-bug compatibility versus intended behavior), migration cost estimation with real multipliers, and how to kill a migration that is failing without killing the team.

**Module 09 — Build-vs-Buy & Unit Economics.** The full TCO discipline: modeling vendor costs against loaded-headcount costs, the hidden line items (integration, lock-in exit cost, vendor roadmap risk), unit-economics dashboards ($/prediction, $/1M tokens, margin per feature), and how to write a build-vs-buy recommendation an exec can approve in one read.

**Module 10 — Reliability Engineering.** SLOs for ML systems where "correct" is statistical, degradation ladders (full model → smaller model → cached → heuristic → fail closed), error budgets shared between infra and model quality, on-call design for ML teams, and incident review practice that improves the system rather than assigning blame.

**Module 11 — Production Failure Pattern Library.** The war-story module. Zillow's iBuyer collapse, Watson Oncology, Uber ATG, Tay, phantom braking, hallucinated citations in court, and the rest — each reduced to its transferable failure pattern (regime change, proxy labels, OOD inputs, adversarial feedback, premature scaling) so you can recognize the shape before it costs your company nine figures.

**Module 12 — Governance & Model Risk.** The second dedicated module for material that must not sprawl: model risk management frameworks, the EU AI Act and sector regulators as engineering constraints, model cards and audit trails as artifacts, fairness impossibility results (COMPAS/Chouldechova) as decision inputs, and how a principal builds governance that protects the company without strangling velocity.

**Module 13 — Influence & Org Design.** Influence without authority as a learnable mechanism: sponsorship, pre-wiring, review culture, writing as the primary lever. Plus the org-design questions principals get pulled into — centralized versus embedded ML teams, platform funding models, and how team topology determines architecture (Conway is not optional).

**Module 14 — Technology Bets.** How to make 2–5 year bets under uncertainty: separating secular trends from hype cycles, cheap-optionality experiments, bet sizing and kill criteria, the meta-skill of updating publicly when wrong, and a framework for the recurring question "do we adopt this now, in a year, or never?"

**Module 15 — Interview Bank.** The L7/E7/Principal interview loop deconstructed: what changes versus senior loops (strategy questions, org-scale system design, "tell me about a time you moved an org"), question banks with graded answers, and how to present a principal-shaped narrative from a senior-shaped career.

**Module 16 — Capstone.** You produce the portfolio artifact the whole course has been accumulating: a full technical strategy for a realistic ML org — diagnosis with numbers, policies, action plan, platform architecture, unit-economics model, eval and governance posture, and a migration plan — defended against a written red-team critique.

## How the modules connect

The course has a deliberate arc, and knowing it changes how you read each module:

```text
01–02   The operating model: what the job is, and its highest-leverage
        artifact (strategy). Every later module produces inputs to
        these two.
03–05   The compute estate: platform, training, serving. The build
        arc — what the org runs on.
06–07   The compounding assets: data flywheels and evals. The moat
        arc — what makes the estate worth owning.
08–10   The cost of time: migrations, unit economics, reliability.
        The operate arc — what keeps the estate alive and paid for.
11–12   The bounded pair: failure patterns and governance. Deliberately
        quarantined into two modules so war stories inform judgment
        without flavoring every chapter with fear.
13–14   The human layer: influence, org design, and long-horizon bets.
15–16   The conversion layer: interviews and the capstone portfolio.
```

Cross-references throughout use "Module NN" — when Module 02's worked example defers a migration plan to Module 08, that is the arc working as designed.

## How to work through it

In order. Modules 01–02 are the foundation everything else references; 03–10 are the technical-leadership core and follow a build → run → pay-for-it arc; 11–12 are the bounded failure/governance pair; 13–14 are the org and futures layer; 15–16 convert it into interview performance and a portfolio.

Every teaching module ends with an exercise that produces a real artifact — a gap analysis, a strategy doc, a cost model, a migration plan. Do them. The exercises are not homework; they are the course. Each artifact feeds the Module 16 capstone, and collectively they become a portfolio you can put in front of a promotion committee or carry into an interview loop. Reading without producing the artifacts gets you vocabulary; producing them gets you calibration, which is the thing this course exists to compress.

Budget roughly **50 hours**: 2–3 hours per teaching module (reading plus exercise) and 8–10 for the capstone. Spread over 10–12 weeks at a sustainable pace; cramming defeats the purpose, because several exercises require observing your own org for a week or two.

### The portfolio you will accumulate

Each exercise produces a named artifact. By the capstone you will have, in one folder:

- **01** — a gap analysis of your own operating model against the principal ladder, plus a 90-day plan.
- **02** — a complete technical strategy document for a real or fictional ML org, with an explicit "what we will NOT do" section.
- **03** — a platform capability assessment and build-order recommendation.
- **04** — a GPU capacity plan and training-fleet cost model with stated assumptions.
- **05** — a serving-estate consolidation analysis with an API/self-host crossover computation on realistic traffic.
- **06** — a data-flywheel design doc: feedback loops, contracts, and a labeling-economics model.
- **07** — an eval governance policy and the eval hierarchy for one product surface.
- **08** — a full migration plan with staged traffic shifts, parity metrics, and kill criteria.
- **09** — a build-vs-buy recommendation memo with a TCO spreadsheet an exec could approve.
- **10** — SLOs, a degradation ladder, and an on-call design for an ML system.
- **11–12** — a failure-pattern casebook entry and a model-risk assessment, in the formats those modules define.
- **13–14** — an influence campaign retrospective and a written technology bet with kill criteria.
- **15** — graded answers to the principal-loop question bank, in your own voice.
- **16** — the capstone: a full org strategy defended against a red-team critique.

This is not busywork accounting. A promotion packet and a principal interview loop both run on *evidence of judgment exercised* — and this folder is that evidence, produced in a compressed, reviewable form.

## Planning assumptions used throughout

Numbers in this course are planning-grade, stated so you can adjust them: H100 cloud pricing $2–4/GPU-hr, loaded senior-engineer cost $250–350k/year, frontier API pricing in the $1–15 per 1M token band depending on tier, and org sizes described in engineer headcount. Where a number matters to a decision, the module shows the sensitivity, not just the point estimate.

## What this course will not do

It will not re-teach system-design mechanics (that course exists), give you a governance-checklist substitute for judgment, or pretend the principal role is the same at a 50-person startup and a 5 000-engineer company — where the difference matters, modules call it out. And it cannot promote you; it can only make you someone who is already doing the job, which is, at every company with a functioning ladder, the actual promotion criterion.
