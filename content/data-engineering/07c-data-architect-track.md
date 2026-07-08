# 07 — The Data Architect Track — Part 3 of 4: Stakeholders, Documentation, Org Design, and Vendors

This is part 3 of 4 of the Data Architect Track. Parts 1–2 covered the architect's mindset and the technical/financial decisions (build vs buy, migration, cost, strategy). Here we cover the people-and-process side: managing stakeholders, writing documentation that holds up, shaping org design, and running vendor selection.

## Phase 10 — Stakeholder Management

The stakeholder skill set is the one most absent from technical training. Every architect serves the same groups — executive sponsors (outcomes, cost, risk), peer engineering leaders (coordination, dependencies), your own technical teams (being unblocked), business stakeholders (getting answers), compliance/security/legal (risk), vendors, and external auditors/regulators (governance posture) — and each wants something different from you. You communicate *up* by leading with the conclusion in the executive's vocabulary and quantifying trade-offs in dollars and time, *down* by being specific and explaining the why, and *across* by keeping commitments and surfacing dependencies early. The hardest skill is productive pushback: rather than a flat no, surface the underlying problem, offer an alternative, make the trade-off visible, or propose a pilot.

This is covered in depth in the MLOps course's ML Architect Track (Phase 10: Stakeholder Management) — the principle is identical across both tracks and the guidance is domain-agnostic; the only data-engineering framing is the vocabulary you speak to executives (revenue and compliance risk, not "Iceberg" or "DAG").

---

## Phase 11 — Documentation

Documentation is a distinct skill from writing code or strategy, and architects produce a lot of it. Match the document to the audience and cadence — one-pager (execs, monthly), ADR (per decision), design doc (per project), strategy doc (yearly), runbook (per system), architecture overview (quarterly refresh), RFC (per proposal). Good architectural documents lead with the conclusion, state the decision and its rationale explicitly, are honest about trade-offs, date everything, keep diagrams matching the words, prefer specific over abstract ("the customer master pipeline," not "the integration layer"), and are maintained rather than abandoned. Prefer version-controlled, text-based diagram tools (Mermaid, Structurizr, PlantUML/D2) over paid tools you lose access to when you change jobs, and fix under-investment in writing by writing more in low-stakes contexts.

This is covered in depth in the MLOps course's ML Architect Track (Phase 11: Documentation) — the principle is identical across both tracks and the guidance is domain-agnostic; the ML track adds a "model card" document type that has no data-engineering equivalent.

---

## Phase 12 — Organizational Design

Architecture and organizational structure are joined at the hip. Conway's Law says a system's structure mirrors the communication structure of the org that built it, so the architect's leverage is the "inverse Conway maneuver" — shape the org to produce the architecture you want rather than just reacting to the one you have. Team Topologies gives four team types (stream-aligned, platform, enabling, complicated-subsystem), and every org sits somewhere on a centralization-vs-federation spectrum: too central creates a bottleneck, too federated creates inconsistency and duplication, and the right balance is contextual and evolves with org maturity.

This is covered in depth in the MLOps course's ML Architect Track (Phase 12: Organizational Design) — the principle is identical; what follows is the data-engineering-specific delta.

**Data-engineering delta — the mapping.** The Team Topologies mapping for data: a central **data platform team** (warehouse, lake, orchestration, observability), **stream-aligned data teams** inside business domains (commerce, marketing), **enabling teams** for capability rollouts (a 6-month dbt or embedded-analytics enablement team), and **complicated-subsystem teams** for specialized work (real-time ML serving). The org-maturity path runs Solo DE → small central data team → platform + analytics-engineer split → full data org (platform, governance, ML-platform, domain-data, and central-analytics teams). Applying Stage-4 patterns to a Stage-2 company is over-engineering; the reverse is chaos.

---

## Phase 13 — Vendor Selection and Contracts

Vendor selection is a skill no one warns you about until you're doing it. The process: write a requirements doc *before* talking to vendors (or their salespeople write it for you), cut a long list of 8–10 to a short list of 3 on basic fit, run structured scored demos, run a real-workload proof-of-concept through the top 2 (2–4 weeks each — demos lie), take reference calls with customers you find yourself rather than the ones the vendor picks, then negotiate — 20–50% discounts are normal at F100 scale, and multi-year commits, volume tiers, reference-customer status, competing offers, end-of-quarter timing, and bundling are all leverage. Read contracts for term/renewal, price-escalation caps, termination and data-export rights, data ownership, SLAs, and liability caps; legal owns the review, but you raise the flag when something looks off.

This is covered in depth in the MLOps course's ML Architect Track (Phase 13: Vendor Selection and Contracts) — the principle is identical; what follows is the data-engineering-specific delta.

**Data-engineering delta — the lock-in signals to watch.** Vendor-specific SQL extensions you can't run elsewhere, proprietary file formats, custom UI tools that bake business logic outside version control, and engineering teams who only know one vendor. The defense is a periodic "what would migration look like?" exercise — even if you never migrate, knowing it's possible disciplines your buy decisions. (The ML track's equivalent hazards center on vendor-specific prompt formats, data-training clauses, and rapid model deprecation.)


---

## You can now

- Map your stakeholders (executive sponsors, peers, direct teams, business stakeholders, compliance, vendors) and adapt how you communicate up, down, and across to each.
- Push back productively when a stakeholder wants something that would damage the platform, using a concrete alternative rather than a flat no.
- Choose the right documentation type for an audience (one-pager, ADR, design doc, strategy doc, runbook, RFC) and apply the traits that make architectural documents actually get read.
- Apply Conway's Law and the Team Topologies framework to diagnose whether your org structure is fighting the architecture you want, and recognize which org-maturity stage your company is in.
- Run a structured vendor selection process (requirements doc, long-list to short-list, scored demos, proof of concept, reference calls, negotiation) and read a vendor contract for the clauses that create lock-in.

## Try it

Pick a real (or recent) stakeholder disagreement — a request you had to push back on, or a decision that needed executive buy-in. Write the one-page executive summary you would have sent, following the "communicating up" guidance in Phase 10: lead with the conclusion, use their vocabulary, quantify the trade-off in dollars or time, and state what you don't yet know. Then write the one paragraph of honest pushback you'd give the requester directly, using one of the four productive-disagreement patterns from Phase 10.
