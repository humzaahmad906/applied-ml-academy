# 07 — The Data Architect Track — Part 3 of 4: Stakeholders, Documentation, Org Design, and Vendors

This is part 3 of 4 of the Data Architect Track. Parts 1–2 covered the architect's mindset and the technical/financial decisions (build vs buy, migration, cost, strategy). Here we cover the people-and-process side: managing stakeholders, writing documentation that holds up, shaping org design, and running vendor selection.

## Phase 10 — Stakeholder Management

The skill set most absent from technical training.

### The Stakeholder Map

Every architect's job involves at least these groups:

- **Executive sponsors** (VP of Data, CTO, sometimes CEO) — care about outcomes, cost, risk
- **Peer engineering leaders** (eng managers, other architects) — care about coordination, dependencies
- **Direct technical teams** (DEs, analysts, data scientists) — care about being unblocked
- **Business stakeholders** (product, marketing, finance) — care about getting answers
- **Compliance / security / legal** — care about risk
- **Vendors** — care about your contract size
- **External (auditors, regulators, sometimes customers)** — care about your governance posture

Different groups want different things from you. Failing to adapt your communication to each is the most common architect failure mode after going-too-deep-on-code.

### Communicating Up

When talking to executives:

- **Lead with the conclusion.** They have 90 seconds.
- **Use their vocabulary.** "Revenue impact," "compliance risk," "competitive advantage" — not "Iceberg" or "DAG."
- **Quantify trade-offs.** "Option A: $500K, 6 months, low risk. Option B: $200K, 3 months, moderate vendor risk."
- **Be honest about uncertainty.** "We don't know" is fine if followed by "and here's how we'll find out."
- **Never surprise them.** If a problem is coming, they hear about it from you first.

The art of the executive summary deserves its own study. Read the Amazon "narrative" memo culture writing if you can find examples.

### Communicating Down

When talking to engineers on your team:

- **Be specific.** Architects who give vague guidance produce confused implementations.
- **Explain the why.** Engineers who understand the why make better local decisions than engineers following orders.
- **Be available for technical depth.** Don't disappear into meetings. Office hours, code reviews, design reviews.
- **Defend them up.** When leadership pushes for the unrealistic, your job is partly to absorb that.

### Communicating Across

Peer engineering leaders:

- **Make commitments and keep them.** Reliability is the currency.
- **Surface dependencies early.** "I'll need X from your team by Q3" said in Q1 is fine; in Q2 is bad.
- **Don't undermine.** Disagree privately, support publicly.

Business stakeholders:

- **Translate.** Their requirements arrive in business language; your team needs them in technical language. You're the translator.
- **Set expectations on timelines.** Most business stakeholders think pipelines take an hour. Educate gently.
- **Be the data ambassador.** When their dashboard breaks, you're the face of the platform.

### The Hardest Skill: Pushback

Sometimes a stakeholder is wrong. They want something that would damage the platform, or is technically infeasible, or is the wrong solution to a real problem.

The architect's job is to disagree productively. Patterns that work:

- "Help me understand the underlying problem you're trying to solve" (often reveals a different solution)
- "Here's what I'd recommend instead, and why" (always have an alternative)
- "If we do that, here's what we'll have to give up" (make the trade-off visible)
- "Let's run a small pilot before committing" (de-risk irreversible decisions)

What doesn't work: just saying no. Or just doing it anyway and resenting it.

---

## Phase 11 — Documentation

A separate skill from writing code or writing strategy. Architects produce a lot of it.

### The Documentation Hierarchy

| Type | Audience | Frequency | Length |
|------|----------|-----------|--------|
| One-pager | Execs | Monthly | 1 page |
| ADR | Team + future architects | Per decision | 1–3 pages |
| Design doc | Engineers implementing | Per project | 5–30 pages |
| Strategy doc | Leadership | Yearly | 10–30 pages |
| Runbook | On-call engineers | Per system | 2–10 pages |
| Architecture overview | Everyone | Quarterly refresh | 5–15 pages |
| RFC | Engineers debating | Per proposal | 3–15 pages |

### What Makes Architectural Documents Good

1. **Lead with the conclusion.** Most readers stop after the first paragraph.
2. **State the decision and its rationale clearly.** Not "we considered many options" — *which* options, *why* this one.
3. **Be explicit about trade-offs.** The negative consequences. The things you're giving up.
4. **Date everything.** Future-you needs to know whether this doc reflects current reality.
5. **Diagrams that match the words.** Don't say "the pipeline goes A to B" if your diagram shows A to C.
6. **Specific over abstract.** "The customer master pipeline" not "the integration layer."
7. **Maintained, not abandoned.** Mark deprecated docs as such; don't let stale documents quietly mislead.

### Diagramming Tools Worth Using

- **Mermaid** — text-based, version-controlled, renders in GitHub. The default for diagrams in markdown docs.
- **Excalidraw** — quick sketches that look hand-drawn. Great for whiteboarding.
- **Lucidchart / draw.io** — for complex diagrams with detailed shapes.
- **Structurizr** — C4 model-native, code-generated diagrams.
- **PlantUML / D2** — text-based, for engineers who want diagram-as-code.

Avoid: anything that requires a paid license you'll lose access to when you change jobs. Avoid: tools that produce screenshots no one else can edit.

### Writing Practice

Most engineers under-invest in writing. The fix is simple: write more, frequently, in low-stakes contexts. A weekly Slack post recapping platform changes. A monthly blog post. A quarterly retrospective. The practice compounds.

Books worth reading: *On Writing Well* (Zinsser), *The Elements of Style* (Strunk & White), and for engineers specifically *Engineers Survival Guide* (Merih Taze) — short, blunt, practical.

---

## Phase 12 — Organizational Design

Architecture and organizational structure are joined at the hip. Architects who don't pay attention to org design produce architectures that fight the organization.

### Conway's Law

> "Any organization that designs a system will produce a design whose structure is a copy of the organization's communication structure."

If your organization has three siloed teams (engineering, analytics, data science) and one platform team, you'll get four data systems with painful integration points. If your teams are organized by domain (commerce, marketing, finance), you'll get domain-aligned data products.

The architect's leverage: shape the organization to produce the architecture you want, not just react to the organization you have. This is "the inverse Conway maneuver."

### Team Topologies (Skelton & Pais)

A useful framework. Four team types:

1. **Stream-aligned teams** — own a value stream end-to-end (domain teams)
2. **Platform teams** — provide self-serve internal platforms (data platform team)
3. **Enabling teams** — short-lived, spread expertise (e.g., dbt rollout team)
4. **Complicated subsystem teams** — own deeply specialized components (e.g., the ML feature store team)

For data:

- A central **data platform team** (warehouse, lake, orchestration, observability)
- **Stream-aligned data teams** within business domains (commerce data team, marketing data team)
- **Enabling teams** for major capability rollouts (a 6-month "embedded analytics enablement team")
- **Complicated subsystem teams** for highly specialized work (real-time ML serving)

The architect's job is to advise on which capability sits where. Putting too much in the central team creates a bottleneck. Decentralizing too much produces inconsistency. The right balance is contextual.

### The Centralization vs Federation Spectrum

```
Fully Centralized                                     Fully Federated
─────────────────                                     ───────────────
One team owns                                         Every domain team
all data work                                         owns their data
        │                                                       │
        ▼                                                       ▼
Pros: consistency,                                   Pros: speed, autonomy,
quality, lower cost                                  domain expertise
        │                                                       │
        ▼                                                       ▼
Cons: bottleneck,                                    Cons: inconsistency,
slow, distant from                                   duplication, governance
domain context                                       overhead
```

Most healthy orgs land somewhere in the middle. Central platform team for infrastructure, federated domain teams for transformation and modeling, central governance for standards. The exact split depends on the company's size and stage.

### The Path Through Org Maturity

A common evolution:

1. **Stage 1: Solo DE** — one engineer doing everything, embedded in product engineering
2. **Stage 2: Small data team** — 2–5 engineers, central team, owns everything
3. **Stage 3: Platform + analytics split** — platform team for infra, analytics engineers within domains
4. **Stage 4: Full data org** — platform team, governance team, ML platform team, domain data teams, central analytics team

Architects who try to apply Stage 4 patterns to Stage 2 companies produce overengineering. Architects who try to apply Stage 2 patterns to Stage 4 companies produce chaos. Recognize the stage; design for it.

---

## Phase 13 — Vendor Selection and Contracts

A skill no one tells you about until you're doing it.

### The Vendor Selection Process

When evaluating a major data tool (warehouse, observability platform, catalog):

1. **Requirements doc.** Write what you need *before* talking to vendors. Otherwise their salespeople tell you what you need.
2. **Long list → short list.** Often 8–10 vendors initially. Cut to 3 based on basic fit.
3. **Demos with structured scoring.** Don't let demos be unstructured. Score each vendor on the same dimensions.
4. **Proof of concept.** Run a real workload through the top 2 vendors. 2–4 weeks each. Don't skip this — vendor demos lie.
5. **Reference calls.** Talk to 3 current customers — *not* the ones the vendor recommends. Find them yourself via LinkedIn or community.
6. **Contract negotiation.** Real procurement work. Discount of 20–50% is normal at F100 scale.

### Reading Vendor Contracts

You don't need to be a lawyer, but you should know to look for:

- **Term length and renewal.** Auto-renew clauses can be expensive. Multi-year discounts can be a trap.
- **Price escalation.** Year-over-year increases capped at what?
- **Termination.** Can you exit cleanly? Data export rights?
- **Data ownership.** Is your data yours? In what format on exit?
- **SLAs.** What's promised? What's the penalty if missed?
- **Liability caps.** If they leak your data, what do you get?
- **Most-favored-customer clauses.** Rare but valuable.

Legal and procurement own the legal review, but you should be educated enough to raise concerns when something looks off.

### Negotiating Position

Architects often have more leverage than they realize:

- **Multi-year commitments unlock discounts.** Negotiate them aggressively.
- **Volume tiers.** Get the next tier's price even if you're not quite there.
- **Reference customer status.** Worth real discount; you'll be on their website.
- **Competing offers.** Even bluffs work. Often the threat is enough.
- **End-of-quarter timing.** Salespeople have quotas.
- **Bundling.** "Storage + compute + support all together."

### The Hidden Vendor Hazard: Lock-In Accumulating

Watch for:

- Vendor-specific SQL extensions (you can't run elsewhere)
- Proprietary file formats
- Custom UI tools that bake business logic outside version control
- Engineering teams who only know this vendor

The architect's defense: explicit periodic exercise of "what would migration look like?" Even if you never migrate, knowing it's possible disciplines your buy decisions.


---

## You can now

- Map your stakeholders (executive sponsors, peers, direct teams, business stakeholders, compliance, vendors) and adapt how you communicate up, down, and across to each.
- Push back productively when a stakeholder wants something that would damage the platform, using a concrete alternative rather than a flat no.
- Choose the right documentation type for an audience (one-pager, ADR, design doc, strategy doc, runbook, RFC) and apply the traits that make architectural documents actually get read.
- Apply Conway's Law and the Team Topologies framework to diagnose whether your org structure is fighting the architecture you want, and recognize which org-maturity stage your company is in.
- Run a structured vendor selection process (requirements doc, long-list to short-list, scored demos, proof of concept, reference calls, negotiation) and read a vendor contract for the clauses that create lock-in.

## Try it

Pick a real (or recent) stakeholder disagreement — a request you had to push back on, or a decision that needed executive buy-in. Write the one-page executive summary you would have sent, following the "communicating up" guidance in Phase 10: lead with the conclusion, use their vocabulary, quantify the trade-off in dollars or time, and state what you don't yet know. Then write the one paragraph of honest pushback you'd give the requester directly, using one of the four productive-disagreement patterns from Phase 10.
