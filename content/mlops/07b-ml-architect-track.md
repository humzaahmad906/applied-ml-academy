# 07 — The ML / AI Platform Architect Track — Part 2 of 2: Stakeholders, Documentation, Org Design, Vendors, Career, and Interviews

This is part 2 of the ML / AI Platform Architect Track lesson. Here we cover stakeholder management, documentation, organizational design, vendor selection, the architect's toolkit, the senior-to-architect career path, common failure modes, and interview preparation.

## Phase 10 — Stakeholder Management

The skill set most absent from technical training.

### The Stakeholder Map

- **Executive sponsors** (CTO, CIO, VP Data/AI, sometimes CEO) — care about outcomes, cost, risk, narrative
- **Peer engineering leaders** — coordination, dependencies
- **ML teams** (data scientists, applied ML engineers) — being unblocked
- **Domain teams** (product, marketing, finance) — getting answers
- **Compliance / security / legal** — risk, audit
- **Vendors** — your contract size
- **External** (auditors, regulators, sometimes customers) — governance posture

Different groups want different things. Failing to adapt your communication is the most common architect failure after going-too-deep-on-code.

### Communicating Up

To executives:

- **Lead with conclusion.** They have 90 seconds.
- **Their vocabulary.** "Revenue impact," "compliance risk," "competitive advantage" — not "GPU memory" or "KV cache."
- **Quantify trade-offs.** "Option A: $500K, 6 months, low risk. Option B: $200K, 3 months, moderate vendor risk."
- **Honest about uncertainty.** "We don't know, here's how we'll find out" is fine.
- **Never surprise them.** Problems come from you first.

### Communicating Down

To engineers:

- **Specific.** Vague architects produce confused implementations.
- **Explain the why.** Engineers who understand why make better local decisions.
- **Available for depth.** Don't disappear into meetings. Office hours, code review, design review.
- **Defend them up.** When leadership pushes for the unrealistic, absorb it.

### Communicating Across

To peer leaders:

- **Commitments kept.** Reliability is the currency.
- **Dependencies surfaced early.** "I need X by Q3" said in Q1 is fine; in Q2 is bad.
- **Don't undermine.** Disagree privately, support publicly.

To business stakeholders:

- **Translate.** Their requirements arrive in business; your team needs them in technical. You're the translator.
- **Set timeline expectations.** Most stakeholders think pipelines take an hour.
- **Be the platform ambassador.** When their dashboard breaks, you're the face.

### Pushback

Sometimes a stakeholder is wrong. They want something that would damage the platform, is infeasible, or is the wrong solution to a real problem.

The architect disagrees productively:

- "Help me understand the underlying problem" (often reveals a different solution)
- "Here's what I'd recommend instead, and why" (always have an alternative)
- "If we do that, here's what we give up" (make trade-offs visible)
- "Let's pilot before committing" (de-risk irreversible)

Doesn't work: just saying no. Or doing it anyway and resenting it.

### LLM-Era Stakeholder Patterns

A specific 2026 pattern: every executive wants their team to "use AI." Some asks are real; many are trend-following. The architect's job: identify which is which, and protect the platform from a thousand half-thought-out asks. Concrete frameworks:

- A short "AI use case intake form" — what problem, what data, what success looks like, what budget
- A staged approach: 2-week prototype → 2-month pilot → production only after success metrics met
- A clear "we don't fund this" signal for asks that don't pass the intake bar

---

## Phase 11 — Documentation

A separate skill. Architects produce a lot of it.

### The Documentation Hierarchy

| Type | Audience | Frequency | Length |
|---|---|---|---|
| One-pager | Execs | Monthly | 1 page |
| ADR | Team + future architects | Per decision | 1–3 pages |
| Design doc | Engineers implementing | Per project | 5–30 pages |
| Strategy doc | Leadership | Yearly | 10–30 pages |
| Runbook | On-call | Per system | 2–10 pages |
| Architecture overview | Everyone | Quarterly refresh | 5–15 pages |
| RFC | Engineers debating | Per proposal | 3–15 pages |
| Model card | Compliance, users | Per model | 1–3 pages |

### What Makes ML Documents Good

1. **Lead with conclusion.** Most readers stop after the first paragraph.
2. **State decision and rationale.** Not "we considered many options" — *which*, *why*.
3. **Explicit trade-offs.** Negatives. What you give up.
4. **Date everything.**
5. **Diagrams match the words.**
6. **Specific over abstract.** "The fraud feature pipeline" not "the data integration layer."
7. **Maintained, not abandoned.** Mark deprecated; don't let stale docs mislead.

### Diagram Tools

- **Mermaid** — text-based, version-controlled, renders in GitHub. Default for markdown docs.
- **Excalidraw** — quick sketches that look hand-drawn.
- **Lucidchart / draw.io** — complex diagrams.
- **Structurizr** — C4-native, code-generated.
- **D2 / PlantUML** — text-based diagrams-as-code.

Avoid: paid-license tools you'll lose when changing jobs. Tools producing screenshots no one else can edit.

### Writing Practice

Engineers under-invest in writing. Fix: write more, frequently, low-stakes. A weekly Slack post recapping platform changes. A monthly blog post. A quarterly retrospective. Compounds.

Worth reading: *On Writing Well* (Zinsser), *The Elements of Style* (Strunk & White), *Engineers Survival Guide* (Taze).

---

## Phase 12 — Organizational Design

Architecture and org structure are joined at the hip.

### Conway's Law

> "Any organization that designs a system will produce a design whose structure copies the organization's communication structure."

If your org has separate ML research, applied ML, and ML platform teams plus a separate data team, you'll get four ML systems with painful integration. If teams are organized by domain (commerce, marketing, finance), you'll get domain-aligned data products.

The architect's leverage: shape org to produce the architecture you want. "Inverse Conway maneuver."

### Team Topologies for ML

Four team types:

1. **Stream-aligned** — own a value stream end-to-end (domain ML teams)
2. **Platform** — provide self-serve internal platforms (ML platform team)
3. **Enabling** — short-lived, spread expertise (a 6-month "LLM rollout" team)
4. **Complicated subsystem** — deeply specialized (real-time feature store team, LLM gateway team)

For ML:

- A central **ML platform team** (compute, registry, feature platform, LLM gateway, monitoring)
- **Stream-aligned ML teams** in business domains (fraud, recs, pricing)
- **Enabling teams** for major rollouts ("Iceberg migration team")
- **Complicated subsystem teams** for highly specialized work (the LLM serving cluster team)

Putting too much central creates a bottleneck. Decentralizing too much produces inconsistency. The right balance is contextual.

### Centralization vs Federation Spectrum

```
Fully Centralized                                       Fully Federated
─────────────────                                       ───────────────
One team owns                                           Every domain team
all ML work                                             owns its ML
        │                                                       │
        ▼                                                       ▼
Pros: consistency,                                     Pros: speed, autonomy,
quality, lower cost                                    domain expertise
        │                                                       │
        ▼                                                       ▼
Cons: bottleneck, slow,                                Cons: inconsistency,
distant from domain                                    duplication, governance overhead
```

Most healthy orgs sit in the middle. Central platform team for infrastructure, federated domain teams for modeling, central governance for standards. Exact split depends on size and stage.

### Org Maturity Path

1. **Stage 1: Solo ML engineer** — embedded in product
2. **Stage 2: Small ML team** — 2–5 people, central, owns everything
3. **Stage 3: Platform + applied split** — platform team for infra, applied ML in domains
4. **Stage 4: Full ML org** — platform, governance, LLM platform, applied teams, central research

Architects who apply Stage 4 patterns to Stage 2 companies produce over-engineering. Stage 2 patterns at Stage 4 produces chaos. Recognize the stage; design for it.

### The "Embedded vs Central" Loop

A persistent argument: should data scientists be embedded in product teams, or central in an "AI" org?

- **Embedded** → faster iteration, closer to business, but inconsistency across teams and harder ML career path
- **Central** → consistency, career path, but disconnect from product

Most F50s end up hybrid: applied ML embedded in domains; platform/research central. Architect's job is to make the seams work — shared platform, shared standards, clear interfaces.

---

## Phase 13 — Vendor Selection and Contracts

A skill no one tells you about until you're doing it.

### Process

When evaluating a major ML tool (model platform, feature store, observability, LLM provider):

1. **Requirements doc.** Write needs *before* talking to vendors. Otherwise their salespeople write them for you.
2. **Long list → short list.** Often 8–10 initially. Cut to 3 on basic fit.
3. **Structured demos.** Score on the same dimensions. Don't let demos be unstructured.
4. **POC.** Real workload through top 2. 2–4 weeks each. Don't skip — demos lie.
5. **Reference calls.** Talk to 3 customers — *not* the ones the vendor recommends. Find them via LinkedIn or community.
6. **Contract negotiation.** Real procurement work. 20–50% discount is normal at F50 scale.

### Reading ML Vendor Contracts

What to look for:

- **Term and renewal.** Auto-renew clauses can be expensive.
- **Price escalation.** Year-over-year caps.
- **Termination.** Clean exit? Data export rights?
- **Data ownership.** Your data is yours? In what format on exit?
- **Model ownership.** If they hosted your training or fine-tuning, who owns the weights?
- **Usage rights.** Can the vendor train on your prompts/data? (Critical for LLM vendors.)
- **SLAs.** What's promised? Penalty if missed?
- **Liability caps.** If they leak your data, what do you get?

Legal owns the legal review; you raise concerns when something looks off.

### Negotiating Position

- **Multi-year commits unlock discounts.** Negotiate aggressively.
- **Volume tiers.** Get the next tier's price even if you're not quite there.
- **Reference customer status.** Real discount; you're on their website.
- **Competing offers.** Even bluffs work.
- **End-of-quarter timing.** Salespeople have quotas.
- **Bundling.** "Training + serving + observability all together."

### LLM-Specific Vendor Hazards

A unique 2026 ML challenge: rapid model deprecation. The model you bake into your product gets deprecated in 12 months. Plan for it:

- **Prefer providers with clear deprecation policies.** Anthropic publishes deprecation timelines; OpenAI's used to be murky and has improved.
- **Abstract your provider layer.** A thin shim (LiteLLM, Portkey, your own) means swapping providers is config, not code.
- **Re-eval on every major model release.** Set quarterly cycles.
- **Watch the "data training" clauses.** Most enterprise tiers exclude your data from training; consumer tiers may not. Make sure you're on enterprise.

### Hidden Vendor Hazard: Lock-In Accumulating

Watch for:

- Vendor-specific prompt formats
- Proprietary model evaluation
- Custom UIs baking logic outside version control
- Engineers who only know this vendor

The architect's defense: periodic "what would migration look like?" exercise. Even if you never migrate, knowing it's possible disciplines your buy decisions.

---

## Phase 14 — The Architect's Toolkit

### Thinking

- **Second monitor with a notes app open** — most architecting happens by writing
- **Obsidian or Notion** — PKM; track decisions across years
- **A spreadsheet** — cost modeling, comparison matrices, capacity planning
- **A diagramming tool** — Mermaid, Excalidraw, or whatever clicks

### Communicating

- **Google Docs / Notion** — collaborative writing
- **Presentation tool** — for execs
- **Whiteboard** — physical or virtual (Miro, FigJam, Excalidraw multiplayer)
- **Slack / Teams** — proficient async writing matters more than you'd think

### Staying Current

- **Curated newsletters** (a few, not 30)
- **One or two podcasts** during commute (Latent Space, MLOps Community, Practical AI)
- **Conference talks** at 1.5x
- **Small reading habit** — 20 pages/day of a real book

### Staying Technical

- **Working laptop dev env** (Docker, the major SDKs, a working MLOps project)
- **Personal lab** — a side project where your hands stay in code
- **Code review activity** — commit to 5+ PRs/week even when you're not coding much

Architects who stop touching tools become out-of-touch architects within 2 years. Half-life of technical credibility is short, especially in ML where the tooling churns.

---

## Phase 15 — Senior → Architect Path

Concrete career advice.

### What Promotion Committees Look For

For staff/principal/architect at F50:

1. **Cross-team impact.** Your work affected multiple teams, not just yours.
2. **Long-horizon thinking.** Evidence you plan beyond the current quarter.
3. **Strategic decisions.** Architectural choices that paid off (or were made well even with mixed outcomes).
4. **Mentorship.** People grew because of you.
5. **Written artifacts.** Design docs, ADRs, strategy memos referenced after you wrote them.
6. **Executive presence.** Can be put in front of a VP without supervision.

Technical depth required but not sufficient. Many strong senior engineers plateau because they optimized for depth without building the rest of the surface area.

### Skills Gap Audit

Typical gaps from senior to architect:

- **Writing** (more, longer, polished, for more audiences)
- **Communication with non-technical stakeholders**
- **Strategic thinking** (3+ year horizons)
- **Comfort with ambiguity**
- **Comfort with politics** (= "people with different goals")
- **Negotiation**
- **Patience** (architecture moves slowly)

Hard to learn in a course. Best by doing — taking architecture-flavored work before the title.

### How to Get the Work Before the Title

1. **Volunteer to write the design doc** for a project, even if you're not leading it
2. **Propose ADRs** for decisions your team is making informally
3. **Run a cross-team initiative** — introducing a feature store, setting up ML observability standards
4. **Present at internal forums** — architecture reviews, all-hands, brown bags
5. **Mentor visibly** — formal programs, code reviews with detailed feedback
6. **Build a written body of work** — internal blog posts, RFCs, public posts

By the time the promotion conversation happens, you've been doing the work. The title catches up.

### Salary Reality

Approximate F50 ranges in 2026 for ML platform / staff ML / principal ML roles (USD, total comp, US-based):

- **Senior ML / MLOps engineer:** $300K–$500K
- **Staff:** $500K–$800K
- **Principal / Architect:** $700K–$1.2M
- **Distinguished / Director-IC:** $1M+

LLM/AI infrastructure roles at frontier labs (OpenAI, Anthropic, Google DeepMind, Meta AI) and the biggest F50s skew higher — staff there can hit $1M; principal $1.5M+ is real.

For remote / international, scale by local market. Gap between mid-senior and architect is often 1.5–2.5x.

The architect track is financially meaningful, intellectually meaningful, and harder. Choose with intention.

---

## Phase 16 — Common Failure Modes

Watch for these in yourself.

### Ivory Tower Architect

Stops touching code. Disconnected from reality. Decisions look elegant on paper, fail in practice. Engineers stop bringing real problems because the architect doesn't understand them.

**Fix:** 15–25% of your week in code or close-to-code. Incident reviews. Occasional pair programming. PR reviews in depth.

### Architecture Astronaut

Over-elaborate systems for problems that don't exist. Loves abstractions for their own sake. Architectures the team can't operate.

**Fix:** YAGNI ruthlessly. Build for the problem you have. Add complexity only when forced by reality.

### Tool Magpie

Adopts every new technology. Stack accumulates faster than the team can absorb. Each new tool brings hidden ops cost.

**Fix:** Boring technology mindset. Default to stack stability. Introduce new tech only when existing can't solve the problem, after proof.

### Decision Bottleneck

Every decision routes through the architect. Team can't move without approval. Architect = rate limiter.

**Fix:** Delegate aggressively. Push decisions to the lowest level with enough context. Reserve architect involvement for cross-team, long-shape, high-stakes irreversible.

### Yes-Architect

Agrees to everything. Roadmap grows infinitely. Nothing focused. Cost balloons.

**Fix:** Practice saying no. Simple framework for evaluating asks. Disappoint stakeholders in service of platform coherence.

### Detail-Distant Architect

Operates at such altitude they can't evaluate proposals. Approves bad designs because they couldn't tell. Loses technical credibility.

**Fix:** Depth in some specific area, even as you broaden. Maintain at least one specialty where you stay sharp.

### Politics-Avoidant Architect

Refuses organizational dynamics. Treats politics as beneath them. No influence over things they need to influence.

**Fix:** "Politics" = "people with different goals working out trade-offs." Engage. Build relationships. An architect who can't get budget from finance and headcount from HR is not effective regardless of technical skill.

### LLM-Era-Specific: The Hype Surfer

The 2026 trap. New model drops every Tuesday. Each one is "going to change everything." The architect who reorients the strategy every quarter is the architect with no platform.

**Fix:** Track the field, but commit on a 6–12 month cadence. Have an experimental track for the frontier; keep production stable.

---

## Phase 17 — Architect Interview Preparation

How architect interviews differ from engineer interviews.

### What's Tested

A typical F50 ML architect loop:

1. **Initial screen** — career story, scope of recent work
2. **System design** — design a system for a non-trivial problem
3. **Architecture deep dive** — present a system you architected; defend it
4. **Behavioral / leadership** — conflict, influence, decision-making
5. **Executive interview** — VP/director. Strategic thinking, communication
6. **Reference checks** — at this level references matter

What's missing or de-emphasized: coding (often no LeetCode), single-tool depth. What's added: storytelling, defending past decisions, executive presence.

### The Architecture Deep Dive

Most distinctive round. Present a system (45–60 min). Interviewers probe:

- Why decision X instead of Y?
- What would you do differently if starting over?
- What did you not anticipate?
- How did you handle this constraint?
- How did you communicate to stakeholders?
- How did the team execute?
- What was the outcome?

Preparation: 2–3 systems you can talk about. For each:

- Context (business problem, scale, team)
- Architecture (clean diagram)
- Key decisions (3–5 with trade-offs)
- What went well
- What didn't
- What you'd do differently
- Outcomes (measurable)

The portfolio projects covered in the projects track are great training for this. Even though they're not "real production," you can talk about your decisions at depth.

### System Design Round

Similar to senior MLOps system design but more emphasis on:

- Multi-year evolution (not just initial design)
- Team and org structure
- Cost projection
- Migration path from a hypothetical existing state
- Failure modes and recovery

Pattern:

1. Clarify aggressively (5–10 min): scale, constraints, team, timeline
2. High-level architecture (Context-level C4)
3. Drill into 2–3 components
4. Trade-offs explicitly
5. Failure modes, cost, evolution
6. Organizational implications

### Common Architect Failure Modes in Interviews

- **Too technical.** Treating it like a senior engineer interview. Going deep on PagedAttention when you should be talking about platform org dynamics.
- **Too vague.** Abstractions. "We use cloud-native patterns to enable agile delivery." Useless.
- **Defensive about past mistakes.** Strong candidates own mistakes specifically. Weak ones deflect.
- **No measurable outcomes.** Strong candidates know numbers. Cost reductions, latency improvements, team velocity. Weak ones describe work without quantifying.
- **Couldn't communicate with the VP.** Exec round often decides. Tone, brevity, business framing matter more than technical accuracy.

### LLM-Era-Specific Interview Topics

- "How would you build an LLM platform for a 200-team org?"
- "Walk me through your build-vs-buy logic for LLM hosting."
- "How do you think about LLM cost in 18 months?"
- "How would you migrate off OpenAI without breaking products?"
- "What's your view on agents in production?"
- "How do you handle the EU AI Act for your platform?"
- "Describe how you'd evaluate an LLM application."

Have a 5-minute answer for each.

---

## A Closing Thought

The architect track is *not* "senior MLOps plus more years." It's a different role with different skills. Some excellent senior engineers don't want this work — and that's a legitimate choice. The IC ladder at most F50s goes very high (distinguished engineer roles at $1M+) without requiring you to become an architect.

But if the work described in this track sounds interesting — the strategic thinking, the writing, the decisions, the cross-team scope — there's a real path. The hardest part isn't technical learning (you've done most of that). It's developing the surface area: writing, communicating, deciding under ambiguity, navigating orgs.

That takes years. Start now. Take architecture-shaped work before the title. Write design docs even when nobody asked. Volunteer for cross-team initiatives. Present at brown bags. Mentor visibly.

In 3–5 years, the architect track is yours if you want it.

---

## The Reading List for ML Architects Specifically

Beyond the advanced technical reading list covered earlier in the curriculum:

1. **The Pragmatic Programmer** (Hunt & Thomas, 20th anniversary) — foundational craft
2. **A Philosophy of Software Design** (Ousterhout) — short, dense, transformative
3. **Software Architecture: The Hard Parts** (Ford, Richards, Sadalage, Dehghani)
4. **Fundamentals of Software Architecture** (Richards, Ford)
5. **Team Topologies** (Skelton, Pais)
6. **Accelerate** (Forsgren, Humble, Kim)
7. **The Manager's Path** (Fournier) — even if you stay IC
8. **Working Backwards** (Bryar, Carr) — Amazon's process
9. **High Output Management** (Grove)
10. **Staff Engineer** (Larson) + [StaffEng.com](https://staffeng.com/)

Plus ML-specific architect-flavored:

11. **Machine Learning Engineering** (Andriy Burkov) — the practical engineer's view
12. **Designing Machine Learning Systems** (Chip Huyen) — read again, this time with architecture lens
13. **AI Engineering** (Chip Huyen) — when published, the closest thing to an LLM-platform-architect textbook

The single most valuable for the IC track at senior+: *Staff Engineer* by Will Larson, plus the StaffEng.com interviews.

---

## Where the Curriculum Ends

The full curriculum covers:

- MLOps foundations through core implementation
- Specialization in cloud platforms, LLMOps, and governance
- Portfolio projects for F50 roles
- Advanced topics — the deep technical body of knowledge
- The architect track — the long game (this chapter)

Work through this seriously over 2–4 years and you have a credible path to senior MLOps at any F50, and to staff/principal/architect (after the advanced topics and several years of practiced application).

The compound interest on this work is enormous. Most engineers never build this surface area. The ones who do become indispensable.

Now stop reading curricula. Go build the beginner project from the foundations track.

---

## You can now

- Adapt your communication across stakeholder groups (executives, engineers, peers, business) and push back productively when a stakeholder's ask would damage the platform.
- Produce the right document for the right audience (one-pager, ADR, design doc, RFC, runbook) and pick a diagramming tool that stays version-controlled.
- Read your org through Conway's Law, pick the right centralization-vs-federation balance, and match Team Topologies patterns to your org's maturity stage.
- Run a structured vendor evaluation (requirements doc, shortlist, POC, reference calls, contract review) and spot LLM-specific hazards — rapid model deprecation, data-training clauses, vendor lock-in.
- Recognize the common architect failure modes (ivory tower, architecture astronaut, tool magpie, decision bottleneck, yes-architect, hype surfer) in yourself, apply the fix, and build the track record that promotion committees and architecture-interview panels look for.

## Try it

Pick a real architectural decision your team has made informally in the last year — a serving stack, a feature store, an LLM provider, an experiment tracker — and write the ADR that should have existed. Use the standard structure from Phase 3: status, decision-makers, context (with the numbers you can find), the decision, at least two alternatives each with honest pros and cons, positive and negative consequences, and a phased implementation plan. Then classify the decision as Type 1 or Type 2 and note whether the process weight actually matched that classification. Keep it to 1–3 pages, commit it to your repo as `ADR-0001`, and share it with one teammate for the "would this orient a new hire six months from now?" test. This is the single highest-leverage habit on the senior-to-architect path, and doing it once before the title is exactly the kind of architecture-shaped work promotion committees look for.
