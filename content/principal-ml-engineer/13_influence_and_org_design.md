# Module 13 — Influence Without Authority & Org Design

## Why this module matters

A principal engineer at a 300-engineer company touches the work of 10–30 teams and manages exactly zero of them. Every mechanism you used to get things done as a senior — doing the work yourself, convincing your manager, convincing your teammates over lunch — stops working, because the people whose behavior you need to change no longer share your standup, your manager, or your building. Alex Ewerlof's framing is the right one: the network a principal must influence is roughly 10× the network of a staff engineer, and at that multiplier, 1:1 influence arithmetic collapses — you cannot have enough coffees. The engineers who make this level work replace personal presence with *mechanisms*: documents that argue while they sleep, review systems that raise the bar without them in the room, and org structures that make the right thing the default thing. This module is those mechanisms, made explicit.

## 1. Writing is the scaling mechanism

At senior, your leverage is code. At principal, your leverage is decisions — and a decision only scales if it is written down, because a written decision can be read by 200 people simultaneously, argued with asynchronously, and cited two years later when everyone involved has changed teams. Will Larson's observation holds: at staff-plus, "writing is the closest thing to a superpower that exists in engineering organizations."

Three document types do most of the work. Learn to distinguish them, because writing the wrong one wastes weeks:

**The RFC (Request for Comments).** A proposal seeking input *before* a decision. Structure: context → problem → proposed approach → alternatives considered (with real steel-manning, not straw men) → open questions → rollout sketch. The open-questions section is not weakness; it is the invitation that makes reviewers collaborators instead of critics. An RFC with zero open questions reads as a decree wearing a costume, and reviewers respond accordingly.

**The decision doc.** A record written *after* a decision, for the people who weren't in the room and for your future self during the inevitable relitigating. One page: decision, date, deciders, options considered, why this one, what would make us revisit. Amazon-style orgs call the revisit clause the most important line — it converts "we decided forever" (which people fight) into "we decided until evidence X appears" (which people accept).

**The memo.** An argument meant to change minds before any specific proposal exists. "Our serving costs will exceed our infra budget by Q3 2027 at current growth" is a memo. It creates the shared diagnosis that later RFCs will build on — Module 02's point that strategy starts with diagnosis applies to influence too: you cannot pre-wire a solution into an org that does not yet agree there is a problem.

A calibration number: a working principal at a healthy org produces something like 1–2 substantial documents a month and reviews 5–10. If you are writing zero, you are operating as a very senior senior. If you are writing six, you are probably substituting document volume for the socializing that makes documents land (Section 3).

```text
RFC skeleton (steal this)

Title, author, status (draft/review/decided), reviewers named individually
1. Context        — 5 sentences max; link the memo/strategy, don't restate it
2. Problem        — what breaks or is lost if we do nothing, with a number
3. Proposal       — the design, at the altitude of decisions, not diffs
4. Alternatives   — 2–3, each with the strongest honest case FOR it
5. Costs & risks  — engineer-months, dollars, migration burden, exit cost
6. Open questions — the 3 things you genuinely want input on
7. Rollout        — sequencing, lighthouse, revisit trigger
```

## 2. The design-review system: raise the bar without becoming the bottleneck

Most orgs at 100+ engineers grow some design-review ritual. The principal usually inherits or builds it, and there is a narrow path between two failure modes: reviews so weak they are theater, and reviews so gated that you personally become the org's throughput ceiling — the *gatekeeper* anti-pattern from Module 01.

**Review the decision, not the diff.** Your job in a design review is the five questions that change outcomes: Is this the right problem? Does the approach survive 10× scale? What is the migration/exit cost? What breaks first? Who else is affected and do they know? It is explicitly *not* naming conventions, framework taste, or how you would have structured the modules. Every comment you spend on diff-level taste costs you credibility for the decision-level comment that matters, and trains the org to see review as hazing.

**Teach in comments.** The reader of your review comment is not just the author — it is everyone who reads the doc later. "This won't scale" helps nobody. "At 50k QPS this fan-out means ~400 feature-store reads per request; the recsys team hit this in 2025 and moved to a batched read — see their doc" teaches the author, the lurkers, and the next three teams with the same design. Write review comments as if they will be quoted, because the good ones are.

**Structural defenses against becoming the bottleneck:**

- **You review categories, not everything.** Publish explicit triggers: new externally-visible service, new datastore, cross-team interface change, new model family in production, anything touching money movement or PII. Everything else ships on team-level review.
- **Grow other reviewers deliberately.** Co-review with a staff engineer for three cycles, then hand them the category. A review system that only works when you are in the room has failed even if every individual review was excellent.
- **Timebox to 5 business days** for review cycles, with silence defaulting to approval on non-triggered docs. An unreviewed doc aging in a queue is the platform-org failure smell.
- **Keep an advice/consent distinction.** Most of your comments are advice the author may decline; mark the rare blocking concern explicitly. If more than ~10% of your comments are blocking, you are either reviewing too late or gatekeeping.

## 3. Pre-wiring: the meeting is the ratification, not the debate

The single most reliable tell of someone new to the level: they walk into a 12-person review meeting with a surprise proposal and get shredded — not because the proposal was wrong, but because six stakeholders each discovered a personal objection in real time, in public, with an audience to perform for. Decisions of consequence are not made in meetings. They are made in the 1:1s beforehand; the meeting ratifies.

The pre-wiring protocol:

1. **Map the stakeholders.** For a given decision, list everyone who can block it, everyone who must implement it, and everyone who will loudly comment. For each: what do they want, what do they fear, what did they propose last year that this displaces? (That last one is the most commonly missed — a surprising fraction of technical objections are authorship wounds.)
2. **Socialize individually, cheapest skeptic first... actually, allies first.** Walk the doc to likely allies first to sharpen it, then the most influential skeptic *before* the broad send. Skeptics who are consulted early become co-authors of the fix; skeptics ambushed in a meeting become permanent opponents.
3. **Integrate objections visibly.** When you incorporate someone's concern, name it in the doc ("Fraud-team constraint, raised by Dana: ..."). This converts a critic into a defender — people do not attack documents containing their own fingerprints.
4. **Walk in with the outcome already probable.** If you have 1:1 alignment from the people who matter, the meeting is 20 minutes of confirmation and edge cases. If you cannot get that alignment 1:1, the meeting was going to fail anyway — better to learn it privately, where positions can still change.

This is not manipulation; it is respect for how humans actually change their minds — privately, with time to think, without an audience. The failure mode to avoid is *fake* pre-wiring: collecting nods without genuinely updating the proposal. People notice within one cycle, and the currency you spend is trust, which does not refill quickly.

## 4. Disagree-and-commit, done honestly

You will lose arguments — at this level, on decisions with seven-figure consequences, against peers who are also usually right. Disagree-and-commit is the mechanism that keeps losing arguments from becoming either sabotage or silent resentment, but only if done with all three parts:

1. **State the disagreement in writing, once, at decision time.** "I think the vendor path underprices exit cost by roughly $2M; here is my estimate. If we proceed, I recommend contract terms X and Y as hedges." On the record, specific, constructive.
2. **Commit fully.** Not "comply while telling hallway audiences it will fail." Your teams read your actual energy with perfect fidelity; half-commitment from a principal licenses half-commitment from forty engineers, which *guarantees* the failure you predicted and teaches the org nothing.
3. **Set the revisit trigger.** "If integration cost exceeds 4 engineer-months by June, we re-open this." This is the part almost everyone skips, and it is what makes the mechanism honest — it converts your disagreement from a grievance into a falsifiable prediction. Either you were wrong (good: update publicly, per Module 14) or the trigger fires and the org corrects course with your credibility enhanced rather than spent.

## 5. Managing up and across: translation is the job

Executives do not resist technical arguments; they resist arguments in a foreign language. The principal is the translation layer between engineering language (skew, utilization, eval coverage) and executive language (risk, cost, time-to-market, optionality). Translation runs both ways — you also carry "the board wants a GenAI story by Q4" down to engineering as actual constraints rather than eye-rolls.

**The one-pager-for-the-CFO pattern.** When a technical decision needs executive money, the artifact is one page, structured as: what we recommend (one sentence) → what it costs (one number, loaded) → what it returns or de-risks (dollars or a named risk) → what happens if we don't (the expensive default nobody priced) → what we need from you (a specific approval). Every technical detail lives in a linked appendix the CFO will never open. Compare:

```text
Engineer version: "We need to consolidate our five serving stacks on a
unified inference platform to reduce operational entropy and enable
consistent eval gating."

Translation: "We run five copies of the same infrastructure. Each costs
~2 engineers to maintain — about $3.2M/yr loaded. Consolidating to one
costs 6 engineer-quarters once (~$900k) and removes the class of incident
that cost us $2M in March. Ask: priority for 3 engineers for 2 quarters."
```

The second version is not dumbed down. It is *finished* — the analysis carried all the way to the decision the reader actually owns. Module 09's TCO discipline provides the numbers; this section is about refusing to hand an executive an unfinished argument.

Two working rules: **never surprise your VP in a room** (bad news travels by 1:1 first — the message "here's the problem, here are two options, here's my recommendation" makes you the person execs pull into harder problems), and **know each executive's native metric** (CFO: unit economics and committed spend; CPO: time-to-market and user trust; General Counsel: liability surface and auditability — Module 12's material is largely GC translation).

## 6. Coalition mechanics: platform adoption without a mandate

The recurring principal assignment: get N teams onto one platform/standard/paved road, with no authority to compel any of them. The playbook that works has three laws:

**Make the paved road genuinely better.** Not better-for-the-org-on-average — better *for the adopting team, on their current quarter's goals*. If migration costs a team six weeks and saves the org money they never see, you are asking for charity, and teams have no charity budget. Either the platform gives them something they want (faster deploys, free eval infra, on-call relief) or you find funding to pay the migration cost centrally. Adoption friction is a price; someone pays it; decide who explicitly.

**Find the lighthouse team.** Choose your first adopter for signal value, not convenience: a team with a real workload, a respected tech lead, and a use case hard enough to be credible ("if it works for fraud's latency budget, it'll work for us"). Over-invest in their success shamelessly — embed an engineer, fix their bugs same-day. One skeptical team convinced by a lighthouse's numbers is worth ten mandates. The lighthouse's tech lead saying "the migration took three weeks and our p99 dropped 20%" at an eng all-hands does more than any document you will ever write.

**Never mandate before proving.** A mandate issued before the platform demonstrably works converts every rough edge into evidence of your bad judgment, and teams comply maliciously — minimal integration, loud complaints, workarounds that undermine the platform's economics. The correct sequence is prove → publicize → make default for new work → *then* (and only with leadership, for the genuine holdouts) mandate, by which point the mandate merely formalizes what 70% of teams already did. Module 03 covers the platform-as-product mechanics; the political sequencing here is what makes those mechanics land.

## 7. Mentoring as leverage: growing staff engineers is a deliverable

At this level, "the org has three more people who can do staff-level work" is a deliverable on par with a shipped platform — arguably above it, since people compound and platforms depreciate. Two distinctions matter:

**Sponsorship vs mentorship.** Mentorship is advice: reviewing their doc, debugging their career question — costs you an hour, transfers knowledge. Sponsorship is spending your credibility on their behalf: putting their name up for the visible migration, citing their design in the exec review, insisting they present their own work. Sponsorship is what actually moves careers, it is scarcer, and it is riskier — you are underwriting them. A principal doing five mentorships and zero sponsorships is doing the comfortable half of the job. Budget sponsorship deliberately: 2–3 active sponsorships is a full book.

**Delegate decisions, not tasks.** Delegating a task ("build the router") grows a senior into a faster senior. Delegating a decision ("own the routing-layer design; here are the constraints that are non-negotiable — p99 budget, the fraud team's veto on anything touching their features — everything else is yours; I want a pre-read before the review, and I will not override you in the room") grows a senior into a staff engineer. The guardrails are constraints and a review point, not a solution outline. Yes, they will make a choice you wouldn't have. If it is inside the guardrails, let it ship — the cost of a mildly suboptimal router is tuition, and it is cheaper than the alternative, which is an org where every consequential decision still routes through you (see: gatekeeper, again).

## 8. Org design: Conway's law is an architecture tool

Conway's law — systems mirror the communication structure of the orgs that build them — runs forward and backward. Forward: your five serving stacks exist because five teams didn't talk. Backward (the "inverse Conway maneuver"): if you want a unified serving architecture, the reliable move is often a team-boundary change, not a better diagram. Principals get pulled into org design precisely because at this level, *team topology is an architectural decision*, and you are the person who can see both layers.

**The three ML org shapes and their failure modes:**

- **Central platform team.** One team owns training infra, serving, evals; product teams consume. Wins: consistency, economies of scale, deep expertise. Failure mode: the ivory tower — the platform team optimizes for elegance over adoption, tickets queue for weeks, product teams quietly rebuild what they need (and you're back to five stacks, plus a resented platform team).
- **Embedded ML engineers.** Each product team has its own MLEs, full autonomy. Wins: speed, product proximity, no coordination tax. Failure mode: entropy — N teams make N copies of every decision, nothing is shared, no MLE has peers to grow against, and the org's aggregate ML maturity is the *minimum* of its teams'.
- **Hybrid: enable-and-embed.** A small platform core owns the paved road; platform engineers rotate into product teams for a quarter to onboard them, then rotate back carrying the product teams' pain into the roadmap. This is the shape most 200–1000-engineer orgs converge to. Its failure mode is under-resourcing the core: an enablement team of 3 serving 12 product teams becomes a consultancy that builds nothing durable.

**Interface ownership is the sharpest question.** For any two-team boundary, someone must own each surface, and ambiguity is where systems rot. The canonical ML boundaries: *features* (owner: usually platform for the store, producing team for each feature's correctness — with a data contract, Module 06), *serving* (platform owns the runtime and SLO machinery; product owns the model artifact and its quality), *evals* (platform owns the harness and gating infra; product owns the eval sets and thresholds — Module 07). When an incident review can't decide which team's action item it is, you have found an unowned interface; assigning it is a principal-sized deliverable.

**Proposing a reorg as an IC.** You do not own reorgs; directors do. Your move is a memo, not a plan: the diagnosis ("these four architectural problems are downstream of this team boundary"), 2–3 topology options with tradeoffs honestly priced (including people costs — reorgs burn trust and roughly a quarter of momentum), and a recommendation — then hand it to the director who owns the decision and let them lead it. An IC seen designing boxes-and-lines for other people's teams without invitation spends enormous credibility; an IC whose memo quietly becomes the reorg's rationale gains it. Timing filter: propose topology changes only when the architectural cost of the current shape is *demonstrated* (incidents, duplicated systems, stalled migrations you can name), never on aesthetics.

## 9. Owning the hiring bar

A principal who interviews 40 candidates a year shapes the org more durably than most architecture decisions — hiring compounds. Two responsibilities:

**Calibration.** Interview rubrics drift: each interviewer's "strong hire" quietly renormalizes to the candidates they've seen recently. The fix is mechanical, and someone at your level has to run it: written rubrics with behavioral anchors ("designs the feedback loop unprompted" not "good at system design"), periodic panel-calibration sessions replaying a real debrief, and auditing outcomes (do your "hire" calls succeed at 2 years? — most orgs never check).

**The bar-raiser role.** Borrowed from Amazon: a trained interviewer from *outside* the hiring team with veto power, whose incentive is the company's bar rather than the team's urgency. If your org has the role, hold it for ML hires; if not, propose it — a one-page RFC, piloted on your own org's loops first (Section 6's sequencing applies to process adoption too). The bar-raiser's highest-value veto is the "strong on skills, corrosive on collaboration" candidate whom a shorthanded team will always talk themselves into.

## You can now

- Write the right document for the moment — RFC to invite input before a decision, decision doc to record one, memo to build the shared diagnosis that makes RFCs land — and distinguish the three so you never spend three weeks writing the wrong one.
- Run a design review that asks the five decision-level questions (right problem? survives 10×? migration cost? first failure mode? who else is affected?) without becoming a bottleneck — enforcing a 5-day timebox, growing other reviewers through co-review cycles, and calibrating blocking versus advisory comments.
- Pre-wire a contentious proposal through ordered 1:1 stakeholder conversations so the review meeting ratifies in 20 minutes rather than debates for 90 — including identifying the authorship wounds that masquerade as technical objections before they surface in a room.
- Drive platform adoption across teams you do not manage by paying migration friction centrally, over-investing in a lighthouse team whose numbers do the persuading, and reserving executive mandate only to formalize behavior that 70% of teams already chose.
- Propose a team topology change as an IC by diagnosing the architectural cost of the current boundary in demonstrated terms (named incidents, duplicated systems, stalled migrations), pricing the momentum loss of a reorg honestly, and handing the memo to the director who owns the decision.

## Worked example — six product teams onto one serving platform in two quarters, no mandate

**Setup.** You are the principal at a ~350-engineer fintech. Six product teams (payments-fraud, credit-risk, support-copilot, personalization, KYC-docs, collections) run models on five serving stacks: two homegrown Flask fleets, one SageMaker estate, one team on a vendor, one on the platform team's year-old vLLM/KServe stack that only its builders use. Aggregate cost of the fragmentation, from your Module 02-style diagnosis: ~9 engineers' worth of duplicated maintenance ($1.6M/yr loaded), no shared eval gating (the March incident — a bad credit-risk model served for 11 days — traced directly to a stack with no gate), GPU utilization at 31% because capacity can't pool. Leadership agrees fragmentation is bad but has declined to mandate consolidation: "get them to want it."

**Step 1 — Stakeholder map (week 1–2).** One page, kept private:

```text
Team          Lead    Wants                  Fears                    Leverage point
fraud         Dana    p99 < 40ms guaranteed  migration breaks SLA     March incident hit them
credit-risk   Marcus  audit trail for regs   another platform promise Module 12 gating = his reg story
copilot       Priya   GPU quota for LLMs     losing vLLM tuning       pooled GPUs = 2x her capacity
personaliz.   Sam     ship velocity          any migration quarter    new-work-only path
KYC-docs      Alice   on-call relief         tiny team, no slack      we staff their migration
collections   Raj     built stack #2 himself being displaced         make him co-author
```

Raj is the political center of gravity: his objections will be technical in form and authorship in substance. Priya is the natural lighthouse: real GPU pain, respected, and her success story ("2× capacity, free") is the one other leads will envy.

**Step 2 — RFC v1 and pre-wiring (weeks 3–6).** Draft the RFC: one serving platform (extending the existing vLLM/KServe stack), migration funded centrally (2 platform engineers embedded per migrating team — the org pays the friction, not the teams), new models *must* launch on it, existing models migrate opportunistically. Walk it 1:1 in order: Priya (ally — sharpens the GPU-pooling math), Dana (skeptic — extracts a written p99 SLO with a rollback clause as her price), Raj last-but-before-the-broad-send, with the specific ask: "you've run serving here longer than anyone; I want you as co-author on the runtime design." Raj's condition — his stack's request-hedging feature gets ported — is genuinely good engineering; it goes in as his named contribution.

**Step 3 — RFC evolves through three revisions (weeks 6–10).** v1→v2: Dana's SLO annex and rollback rights; Marcus's audit-log requirement (which, translated per Section 5, becomes the compliance story that later gets the CFO's sign-off on GPU spend). v2→v3: Sam's objection — "I have zero migration capacity this half" — is accommodated structurally: existing personalization models are explicitly *out of scope* for two quarters; only new work lands on the platform. The alternatives section keeps honest cases for "do nothing" ($1.6M/yr, recurring) and "buy vendor serving" (Module 09 math: cheaper at their scale for 18 months, then worse, plus exit cost). Review meeting, week 10: 25 minutes, no ambushes — everyone influential already has fingerprints on the doc. Decision doc filed same week, revisit trigger included: "if two migrations exceed 6 weeks each, we halt and re-plan."

**Step 4 — Lighthouse quarter (Q1).** Copilot migrates first with two embedded platform engineers. It is deliberately over-supported: same-day bug fixes, a shared Slack channel the whole org can lurk in. Results at week 8: migration took 4 weeks, GPU utilization for Priya's workloads 31%→58% via pooling, p99 down 22% from vLLM tuning her team could never prioritize alone. Priya presents the numbers at the eng all-hands — *her* numbers, not yours. KYC-docs (smallest team, fully staffed by platform) goes second as the "it works for tiny teams too" proof.

**Step 5 — The holdout negotiation (Q2).** Fraud, credit-risk, collections migrate in Q2 on the strength of Q1's numbers and their extracted terms. Personalization (Sam) remains, per the RFC's own carve-out — honored publicly, because honoring your carve-outs is what makes the *next* RFC's carve-outs credible. The pressure that eventually moves Sam is structural, not personal: new personalization models launch on the platform (the new-work rule), the old stack's maintenance is now his team's cost alone, and at the Q3 planning cycle his own engineers ask to stop running it. Final state at two quarters: 5 of 6 teams migrated, the sixth on a dated glide path *they* proposed, zero mandates issued, and the mandate that leadership finally writes for new work merely codifies existing behavior.

**What made it work, compressed:** the friction was paid centrally; every skeptic's price was extracted 1:1 and paid in the document; the lighthouse's numbers did the persuasion; the holdout was given a legitimate path instead of a public defeat; and the whole thing rested on the platform actually being better — no coalition mechanics survive a bad product.

## Exercise

**The scenario.** You are the principal at a 400-engineer e-commerce company. Contentious decision on the table: consolidate three homegrown experimentation/A-B systems (owned by growth, search, and ads — each team's system built by its current tech lead) into one platform, because cross-team experiments currently take 6+ weeks of manual reconciliation and two recent launch decisions were made on statistically invalid cross-system comparisons. Ads' system is the most sophisticated but is coupled to their billing pipeline. Growth's tech lead was passed over for staff last cycle. Search's VP has publicly said "no infra projects this half." Finance wants the headline "faster experimentation" for the earnings narrative.

**Deliverable.** Two artifacts:

1. **The RFC** (2–3 pages) following Section 1's skeleton: context, problem with numbers (estimate the cost of 6-week reconciliation cycles and one bad launch decision; state your assumptions), proposal, two honestly-argued alternatives, costs, open questions, rollout with a lighthouse and a revisit trigger.
2. **The pre-wiring plan** (1 page): a stakeholder table in the worked example's format — every named party (three tech leads, search VP, finance, your own director), their want, their fear, the objection you predict verbatim, and your prepared response or concession for each; plus your socialization *sequence* with one sentence justifying the order.

**You're done when:**

- The RFC's alternatives section contains an argument for keeping three systems that its owner would recognize as fair.
- Every predicted objection in the pre-wiring plan has either a concession, a scope change, or an honest "this is the price, and here's why it's worth it" — no objection answered with adjectives.
- The rollout section names the lighthouse team and *why that team* in terms of signal value.
- There is a revisit trigger specific enough that a third party could adjudicate whether it fired.
- The passed-over tech lead appears in your plan with a role that is genuinely valuable, not a consolation title.

**Self-check questions:**

1. Whose fingerprints are on your RFC by the time it reaches the review meeting, and where exactly are they visible in the text?
2. If the search VP blocks anyway, what is your next move — and does it route through the VP's incentives or over their head? (One of these is recoverable.)
3. What did your plan pay the migrating teams, and who funds it? If the answer is "they absorb it for the org's good," which section of this module did you skip?
4. Which parts of your RFC would survive if the decision went the other way — i.e., did you write a document that informs the decision, or one that only argues for your preference?
5. Six months post-decision: what observable behavior (not sentiment) tells you the influence worked — and would your revisit trigger catch it if it didn't?
