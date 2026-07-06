# Module 16 — Capstone: The Principal Portfolio

## Why this module matters

You cannot put "operates at principal level" on a résumé and expect it to land. Principal is a level you demonstrate, not one you claim, and the demonstration is a body of *decisions* — strategy documents, architecture diagrams, cost models, RFCs — that a hiring committee or a promotion packet can read and calibrate against. This capstone gives you one rich, realistic scenario and asks you to produce the full artifact set a principal actually ships in their first two quarters at a struggling org. Do it well and you walk out with a portfolio that answers most of the Module 15 interview questions before they are asked, plus proof — to a promotion committee or an external loop — that you have already done the job. The scenario is deliberately messy, underspecified, and full of competing interests, because that is the medium principals work in. Your job is to impose clarity on it.

## The scenario: Northwind Commerce — the ML platform turnaround

Read this once end to end before touching a deliverable. Everything you need is here; the ambiguity that remains is intentional and you resolve it by stating assumptions, exactly as you would in a real interview or a real job.

### The company

**Northwind Commerce** is a 12-year-old online marketplace (think mid-market e-commerce) that added a fintech arm four years ago: buy-now-pay-later (BNPL), a branded credit product, and seller cash-advance lending. Roughly $3.1B annual GMV, 14M monthly active buyers, 220k active sellers. **~350 engineers total**, of whom **~40 work on ML** across six teams. Profitable but margins are thin and the CFO watches infrastructure spend closely. A new CTO started five months ago and has asked you — a newly hired Principal ML Engineer reporting to the VP of Engineering — to "figure out what's wrong with how we do ML and fix it."

### The org (ML-relevant)

```
VP Engineering
├── Search & Discovery (8 eng)      — query understanding, ranking, autocomplete
├── Recommendations (7 eng)         — homepage, PDP, cart, email recs
├── Risk & Fraud (6 eng)            — payment fraud, account takeover
├── Credit & Lending (5 eng)        — BNPL underwriting, credit-line decisions
├── Marketing/Growth ML (4 eng)     — LTV, churn, bidding
├── Trust & Safety (3 eng)          — listing moderation, seller risk
└── (no platform/infra ML team exists — each team runs its own stack)
Data Engineering (11 eng, separate org under a different director)
```

### The current state (what you found in weeks 1–3)

- **28 models in production.** Nobody had a list; you built the first inventory yourself by interviewing teams. Of the 28: 9 are gradient-boosted trees (fraud, credit, LTV, ranking features), 11 are various neural nets (recs embeddings, query understanding, image moderation), 5 wrap third-party LLM APIs (support triage, listing-quality copy, review summarization, seller-message classification, a nascent shopping assistant), 3 are "models" that are actually SQL heuristics someone never replaced.
- **5 distinct serving stacks:** (1) sklearn models pickled and served from a Flask app on VMs; (2) a custom Go gRPC server for the fraud trees; (3) TorchServe for recs; (4) a Ray Serve deployment one team stood up last year; (5) direct calls to a frontier LLM API from application code, no gateway. On-call is per-team and uneven; the recs team has a real rotation, Trust & Safety has one engineer who gets paged at night.
- **GPU utilization ~38%.** Two on-prem A100 nodes (16 GPUs) bought two years ago plus ad-hoc cloud H100 rental. Training and a little inference share the same pool with no scheduler — teams coordinate in a Slack channel. The finance-visible ML infra spend is **~$2.9M/year** (cloud GPU rental $1.4M, on-prem amortized + colo $0.6M, LLM API spend $0.7M and rising ~15%/quarter, data infra allocation $0.2M).
- **No evaluation standard.** Each team defines "better" its own way. Two teams do proper A/B testing; the rest ship on offline metric improvements and vibes. There is no frozen test set discipline; the credit team rebalanced their eval set last quarter and can no longer compare to historical numbers.
- **No model governance.** The credit and BNPL models make lending decisions with zero documented fairness analysis, no model cards, and no audit trail of what data trained which model version. Legal flagged this after a competitor got a regulatory inquiry. Northwind operates in the US (ECOA/Reg B applies to credit) and is expanding to the EU next year (AI Act high-risk obligations will apply to the credit models).

### The recent incident (the thing that got you hired)

**Nine weeks ago**, the fraud model's false-negative rate silently tripled over 11 days. Root cause: a payments vendor changed a `currency_code` field from ISO numeric ("840") to alpha ("USD") in an upstream event; the feature pipeline mapped the unrecognized value to a default that collapsed a discriminative feature. Dashboards stayed green (infra healthy, latency fine, model "up"). It was caught only when the finance team noticed a chargeback spike in the weekly review. **Estimated loss: ~$2.0M** in fraudulent transactions plus remediation. The postmortem, such as it was, said "add an alert on currency_code" and closed. No systemic fix.

### The board mandate

The board, having read the same headlines as everyone, has asked for a **"GenAI strategy"** and specifically wants the shopping assistant (currently a leaky prototype calling a frontier API directly from the web app) turned into a real product within the year. There is enthusiasm and a budget line but no plan, no eval, and no cost model. The current prototype spends ~$0.7M/year in API calls at low traffic; nobody has modeled what it costs at full rollout.

### The three vendor proposals on the table

Sitting in your inbox when you started:

1. **"FeatureForge" — managed feature store.** $420k/year (platform) + usage; vendor estimates $180k/year infra passthrough at your scale. Slick demo. Would replace the ad-hoc feature logic currently duplicated across the fraud, credit, and recs teams. 2 competitors exist at similar price.
2. **"ServeMax" — managed GPU inference platform.** Per-GPU-hour pricing that pencils to ~$1.1M/year at your current inference volume, claims to consolidate serving and improve utilization. Lock-in risk: proprietary model packaging format.
3. **A frontier lab's enterprise LLM agreement.** Committed spend of $1.2M/year for a 30% token discount and data-residency guarantees, versus current pay-as-you-go. Sales pressure to sign this quarter.

Your predecessors deferred all three. The CTO wants your recommendation on each.

---

You have the scenario. Now produce the artifacts. Treat each as something you would actually hand to the CTO, the CFO, or a promotion committee — not a homework answer.

## The deliverables

### 1. Technical strategy document (the anchor artifact)

Using the diagnosis → guiding policies → coherent actions structure from Module 02, write the ML technical strategy for Northwind. One page of body plus appendix. It must:

- **Diagnose** the real constraint (it is not "we need more GPUs" — it is the absence of a platform, standards, and ownership, which produces the incident, the wasted spend, and the governance gap as *symptoms*). Name the constraint in one sentence.
- State **3–5 guiding policies** (e.g., "one paved-road serving stack, escape hatches for the 20%"; "no model ships without an eval card and a rollback plan"; "API-first for GenAI, distill and self-host only proven high-volume paths").
- List **coherent actions** that follow, sequenced.
- Contain an explicit **"What we will NOT do"** section. If it does not forbid anything, it is a wish list, not a strategy.

**Acceptance:** a reader who knows nothing about Northwind understands the constraint, the bets, and the tradeoffs in under five minutes, and can tell you what the strategy *refuses*.

### 2. Platform target architecture + migration sequencing

- A one-page target architecture: the five platform planes (data/feature, training, registry + model CI/CD, serving, eval/observability), the interfaces between them, and concrete component choices at Northwind's scale (300-ish eng reference point from Module 03).
- A **sequencing plan**: what you platformize first and *why* (hint: the thing whose absence caused the $2M incident and the thing that unlocks the GenAI mandate are strong candidates; the feature store is a build-vs-buy question, deliverable 3).
- An **interface contract table**: for each plane boundary, who owns it, what the contract guarantees, and the backward-compat policy.

**Acceptance:** the sequencing has explicit entry/exit criteria per phase and names what runs in parallel vs what blocks what.

### 3. Build-vs-buy TCO for one component

Pick **one** of the three vendor proposals (the feature store is the richest; the GPU platform and the LLM enterprise agreement are also valid) and produce the full analysis from Module 09:

- 3-year TCO both paths (build path must include fully-loaded headcount — assume $380k/eng/year loaded at Northwind — the roadmap tax, and the team you staff forever; buy path must include integration glue, typically 30–50% of build cost, plus per-usage scaling).
- Sensitivity analysis (what volume or price change flips the decision).
- A **one-page CFO memo** framed in $/outcome, time-to-market, and risk — not model accuracy.
- A recorded **revisit trigger** ("reconsider if X or Y").

**Acceptance:** the recommendation is defensible to a hostile CFO, and you can state the exact assumption that, if wrong, reverses it.

### 4. Org-wide evaluation standard + one eval card

- The eval standard (from Module 07): the launch-gate policy every model must pass (eval card present, regression suite green, guardrail metrics defined, rollback criteria stated), the frozen-test-set governance rule, and the offline→online correlation expectation.
- The **eval-card template** as a reusable block.
- **One filled-in eval card** for a specific Northwind model — the fraud model is the sharpest choice given the incident.

**Acceptance:** the standard would have blocked or rapidly caught the currency_code incident, and you can point to the specific line that does so.

### 5. Governance one-pager with risk tiering

From Module 12:

- Risk-tier all 28 models by blast radius (a recs tweak vs a credit denial are not the same tier). You do not need to name all 28 — define the tiers and place each *category*.
- The governance policy: review depth per tier, what the credit/BNPL models specifically need (ECOA adverse-action explanations, a chosen-and-defended fairness definition, model cards, audit trail), and the EU AI Act high-risk obligations coming with expansion.
- How governance is enforced **without killing velocity** (self-serve checklist for low tiers, review board only for high tiers).

**Acceptance:** a low-tier model ships same-day with a checklist; a credit model cannot ship without the high-tier artifacts, and the policy says exactly which.

### 6. RFC + pre-wiring plan for the most contentious change

The most contentious change is almost certainly **serving consolidation** — six teams each own a stack they like and will resist. Using Module 13:

- Write the RFC for consolidating onto the paved-road serving platform.
- A **stakeholder map**: each team, their likely objection, your response, and who your lighthouse team is (the one you convert first to prove the road).
- The **pre-wiring plan**: who you talk to before the review, in what order, and how you fold their objections into the RFC so the meeting ratifies rather than debates.
- A disagree-and-commit provision with a revisit trigger for holdouts.

**Acceptance:** the plan wins adoption without a mandate; it names the lighthouse team and the forcing function.

### 7. 90-day and 3-year roadmap

- **90 days:** what you personally do and what the org does — inventory (done), the eval standard, the first platform plane, the incident-class fix, the vendor decisions. Concrete, dated, owned.
- **3-year:** the platform maturity arc, the GenAI product trajectory (prototype → gated product → possible distillation of the high-volume path), the governance buildout ahead of EU expansion, and the utilization/cost target ($2.9M spend at 38% util → what, by when).

**Acceptance:** the 90-day plan is executable by a real person in a real quarter; the 3-year plan sequences the bets and states what evidence advances each.

### 8. 15-minute executive presentation outline

Slides as markdown (title + 3–5 bullets each), ~10–12 slides, aimed at the CTO + CFO + the board sponsor. It tells the story: here is the constraint, here is what it is costing you (the $2M incident and the 38% utilization are your evidence), here is the strategy, here are the three vendor decisions, here is the 90-day plan and the 3-year arc, here is what I need from you. Frame everything in business terms; the ML is implied, not lectured.

**Acceptance:** an exec who sees only these slides can approve the plan and the budget asks.

---

## Grading rubric

Grade each deliverable against this. The gap between columns *is* the senior→principal delta this whole course teaches.

| Dimension | Senior-level answer | Staff-level answer | Principal-level answer |
|---|---|---|---|
| **Problem framing** | Fixes the stated problem (add the currency_code alert) | Fixes the class of problem for the team | Names the *constraint* behind all the symptoms; reframes what leadership even asked |
| **Scope** | One model / one stack | One team's systems | The org: 28 models, 6 teams, the platform none of them own |
| **Tradeoffs** | Picks the technically best option | States tradeoffs, picks one | Forbids things explicitly; the strategy has a "will NOT do" and it costs something |
| **Economics** | "GPUs are expensive" | Rough cost estimate | 3-year TCO with fully-loaded headcount, sensitivity, the assumption that flips it, framed for the CFO |
| **Influence** | Would file a ticket | Would write a design doc | Pre-wires the decision, names the lighthouse team, wins adoption with no mandate |
| **Risk & governance** | "We should be fair" | Adds a fairness metric | Tiers by blast radius, chooses and *defends* a fairness definition, maps to ECOA + AI Act |
| **Evaluation** | Improves offline metric | A/B tests the change | Org-wide launch gate that would have caught the incident; frozen-set governance |
| **Communication** | Technical write-up | Design doc + diagram | One-pager the CFO reads; a 15-min deck an exec approves from |
| **Time horizon** | This sprint | This quarter's roadmap | 90 days *and* a 3-year arc with evidence-gated bets |

If a deliverable sits in the "senior" or "staff" column, redo it. The point of the capstone is to force every artifact into the right-hand column.

## How to present it as portfolio / interview evidence

Structure it as a repo so a hiring committee or interviewer can navigate it:

```
principal-portfolio-northwind/
├── README.md                      # the scenario + a 1-paragraph "what this demonstrates"
├── 01-technical-strategy.md
├── 02-platform-architecture.md    # + a diagram (draw it, even in ASCII/mermaid)
├── 03-build-vs-buy-featurestore.md
│   └── tco-model.csv              # the actual numbers, not prose
├── 04-eval-standard.md + eval-card-fraud.md
├── 05-governance-policy.md
├── 06-serving-consolidation-rfc.md
├── 07-roadmap.md
└── 08-exec-deck.md
```

**In interviews**, each artifact is a pre-built answer:
- "Design the ML platform for a large company" → deliverable 2.
- "You inherit 40 models and 9 stacks — first 90 days" → deliverables 1 + 7.
- "Build-vs-buy a feature store, defend it to the CFO" → deliverable 3, verbatim.
- "A critical model degraded and nobody noticed — fix the org, not the model" → the incident + deliverables 4 + the reliability fix in 7.
- "Convince N teams onto one platform without authority" → deliverable 6.
- "Design governance for ML at a lender without killing velocity" → deliverable 5.

Do not read the artifact aloud. Use it as the backbone and *talk* the tradeoffs — the committee is calibrating your judgment, and the judgment lives in the choices you can defend, not the document you produced.

**What to say about it:** "I don't have a real 350-person org's confidential docs to show, so I built the full artifact set for a realistic turnaround scenario. Here's the constraint I diagnosed, here's the one decision I'd defend hardest, and here's the assumption that, if I'm wrong about it, changes my recommendation." That framing — owning the assumption that could sink you — is itself a principal signal.

## Suggested time budget (~30–40 hours)

| Deliverable | Hours |
|---|---|
| 1. Strategy doc | 5–6 |
| 2. Platform architecture + sequencing | 5–6 |
| 3. Build-vs-buy TCO + CFO memo | 5–6 |
| 4. Eval standard + card | 3–4 |
| 5. Governance one-pager | 3–4 |
| 6. RFC + pre-wiring | 4–5 |
| 7. Roadmaps | 2–3 |
| 8. Exec deck | 3–4 |

Do them in order — the strategy anchors everything, and later artifacts reference its policies. Reuse the templates from Modules 02, 03, 07, 09, 12, and 13; the capstone is where they compound into one coherent body of work rather than eight isolated exercises.

## Self-assessment

You are done when you can answer yes to every line:

- [ ] My strategy names the constraint in one sentence, and a stranger can restate it after one read.
- [ ] My strategy forbids something real, and I can say what capacity that frees.
- [ ] My platform sequencing fixes the incident class *before* it chases the GenAI mandate — or I can defend why not.
- [ ] My TCO includes fully-loaded headcount and I can state the single assumption that reverses the recommendation.
- [ ] My eval standard has a specific line that would have caught the currency_code incident, and I can point to it.
- [ ] My governance policy lets a low-tier model ship same-day and blocks a credit model until the high-tier artifacts exist.
- [ ] My RFC names a lighthouse team and a forcing function, and wins adoption without a mandate.
- [ ] My exec deck is approvable by someone who reads *only* the deck.
- [ ] Every deliverable sits in the principal column of the rubric — not the staff column.
- [ ] I can walk into a Module 15 interview question and answer it from this portfolio without inventing anything on the spot.

When all ten are checked, you are not preparing to be a principal. You have produced the evidence that you already operate as one — which, at every company with a functioning ladder, is the only thing that gets you the title.
