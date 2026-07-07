# Module 01 — The Principal Delta: Scope, Judgment, and Operating Model

## Why this module matters

Most senior engineers pursue principal by doing senior work harder: more systems, more code, more heroics. It doesn't work, because principal is not senior-plus — it is a different job with different outputs, evaluated on a different axis. Companies are explicit about this in their ladders and opaque about it in practice, which is why so many strong seniors stall at the transition. This module defines the delta precisely: what the role actually consists of, hour by hour and artifact by artifact, and what "fifteen years of experience" actually encodes so you can acquire it deliberately instead of waiting for it. Everything else in this course builds on the operating model established here.

## 1. What "fifteen years of experience" actually encodes

Strip the mystique and long experience decomposes into three assets:

**A pattern library.** The 15-year engineer has *seen* the failure modes: the migration that stalled at 80% for two years, the platform team that built for a customer that never arrived, the model that aced offline evals and died in the A/B, the vendor whose pricing tripled after lock-in. When a new situation arrives, they pattern-match it against this library in seconds — not because they are smarter, but because they have the index. Chess research (de Groot, then Chase & Simon) found the same thing in grandmasters: not deeper calculation, but a library of ~50k–100k memorized positions. Recognition, not search.

**Judgment.** The pattern library plus the ability to notice *which* pattern applies — including when a situation superficially resembles pattern A but is structurally pattern B. Zillow's iBuyer team had strong forecasting patterns; what they lacked was the judgment to notice that a regime change (post-2020 housing market) invalidated the library itself (the full story is in Module 11). Judgment is knowing the boundaries of your own patterns.

**Calibrated confidence.** Fifteen years teaches you what you know at 95% confidence versus 60%, and — critically — makes you comfortable saying the number out loud. Miscalibration in either direction is expensive: overconfidence ships Watson Oncology; underconfidence produces the staff engineer who hedges every recommendation into uselessness and therefore decides nothing.

Here is the load-bearing claim of this course: **all three are compressible.** The 15-year engineer acquired their library *accidentally* — one company at a time, one failure every couple of years, with no spaced repetition and no forced extraction of the general pattern. Deliberate practice beats ambient exposure at building pattern libraries in every domain where it has been studied. Reading a hundred well-documented failures with the pattern extracted (Module 11), writing twenty strategy documents against realistic scenarios (Modules 02, 16), and auditing your own decision calibration monthly (this module's exercise) is not a substitute for experience — it *is* experience, at 10× the acquisition rate. What it cannot compress is the scar tissue of having been personally accountable when things failed; the exercises simulate accountability by forcing you to commit to positions in writing before seeing outcomes.

### The decision journal: calibration as a practice, not a trait

The single highest-yield tool for compressing the third asset is a decision journal, and it takes ten minutes a week. Every time you take a technical position with real stakes — a design-review objection, a build-vs-buy lean, a "this migration will slip" prediction — log it with a confidence number *before* the outcome is knowable:

```text
date: 2026-03-11
decision/prediction: The vector-DB vendor's quoted p99 (12ms at 50M
  vectors) will not hold on our filtered-query workload; expect >40ms.
confidence: 80%
basis: pattern — vendor benchmarks are unfiltered ANN; our queries
  carry metadata predicates that force post-filtering. Seen twice
  before (2023 search infra, 2024 doc-AI).
falsifier: load test with production query log, filters on.
resolve-by: 2026-04-01
--- resolution (filled in later) ---
outcome: p99 was 55ms at target QPS. Correct.
lesson: pattern held; promote to standing review question — "was
  this benchmarked with OUR filter cardinality?"
```

Review the journal monthly and score it like a forecaster: of the calls you made at 80% confidence, did roughly 80% resolve your way? Most engineers discover they are overconfident in their home domain and *under*confident outside it — which is exactly the miscalibration that makes principals either ship Watson or hedge into uselessness. Two further payoffs: the `basis` field forces pattern *extraction* (turning ambient experience into an indexed library entry, which is the whole compression trick), and after a year the journal is the raw material for the "tell me about a technical call you got wrong" interview question (Module 15) — answered with receipts instead of reconstruction. Every subsequent module's exercise assumes this journal exists; start it this week.

## 2. The level ladder: senior → staff → principal

Ladders differ in vocabulary but agree on structure. Google's L7/L8, Meta's E7/E8, Amazon's Principal/Senior Principal, and the composite ladder visible on levels.fyi all describe the same phase change. For scale: Meta's E7+ population is roughly **3% of engineers**; Amazon Principal Engineers are similarly rare, and Amazon considers the role important enough to maintain a public set of PE Tenets ("Exemplary Practitioner," "Technically Fearless," "Resolve the Unresolvable," "Ask Why") that read as a job description for judgment itself. Google's L7 is org-scope technical leadership; L8 (Principal) is company-scope, and there are usually fewer L8s in a 2 000-engineer org than there are directors.

The delta table:

| Dimension | Senior (L5/E5) | Staff (L6/E6) | Principal (L7+/E7+) |
|---|---|---|---|
| **Scope** | A system or feature area; one team | A group of systems; 2–4 teams | An org's or company's technical direction; 10+ teams affected by your decisions |
| **Horizon** | Weeks–quarters | Quarters–a year | 2–5 years; you own bets whose payoff outlives the org chart |
| **Primary output** | Working code and designs | Designs, cross-team projects, some strategy | Decisions, strategy docs, standards, reviews, unblocked teams |
| **Coding ratio** | 60–80% | 30–50% | **15–25%**, all high-leverage: prototypes, critical-path reviews, tooling |
| **Ambiguity handled** | "Build this well" | "Figure out what to build" | "Figure out whether this problem is worth anyone building anything" |
| **Influence** | Within team, through work | Across teams, through design and persuasion | Across org, **without authority** — through writing, trust, and sponsorship |
| **Business accountability** | Feature health | Project outcomes | ML investment ROI; you are expected to say "kill this $4M/yr effort" and be right |

Three notes on reading this table honestly.

**The scope row is about blast radius, not org chart.** A principal at a 200-engineer company and a staff engineer at Google can have identical jobs. Calibrate by "how many engineer-years does a wrong decision by this person waste?" — at principal, the answer is 20+.

**The coding-ratio row is the most misread.** It does not say "principals stop coding." It says the code changes character: from production features (which the team should own) to *prototypes that de-risk decisions* and *reviews of the paths where a defect is an incident*. Section 5 covers what stays technical.

**The ambiguity row is the actual promotion filter.** Senior engineers are given problems. Staff engineers are given problem *areas*. Principals are given a vague unease ("our inference costs feel high?", "should we be worried about agents?") and must return with the problem statement, the diagnosis, and a recommendation. If you need the problem defined for you, you are — by construction — not yet operating at this level.

## 3. Larson's archetypes, mapped to ML — and their limits

Will Larson's *Staff Engineer* (2021) identified four archetypes of the staff-plus role. Mapped to ML organizations:

- **Tech Lead** — steers one high-priority ML team's execution: the search-ranking lead who owns technical direction while an EM owns people. Most common first staff shape.
- **Architect** — owns a critical technical domain across teams: "the person accountable for our serving architecture" or "for eval infrastructure." Common in ML orgs at 100+ engineers, where cross-cutting domains (features, training infra, serving) outgrow any one team.
- **Solver** — parachuted into the burning problem: the engineer sent to figure out why the fraud model's precision collapsed, or to make the P0 latency target before the launch date. High variance, high visibility.
- **Right Hand** — extends an executive's bandwidth: attends the VP's staff meetings, operates with borrowed authority across their whole org. Rare below very large companies.

Learn the archetypes as a *vocabulary for negotiating your role*, not as a personality quiz. Alex Ewerlof (*Beyond Staff Engineer*) and Sean Goedecke have both made the sharper critique: real staff-plus work is situational, and the archetypes are **modes you switch between**, sometimes within a single week — Architect on Monday's serving-stack review, Solver on Wednesday's eval-regression fire, Right Hand in Friday's planning offsite. Engineers who identify *as* one archetype get stuck: the self-declared Architect refuses fire-fighting that would build the trust their architecture proposals need; the perpetual Solver never accumulates the durable ownership that promotion committees can see. The useful question is never "which archetype am I?" but "which mode does *this situation* need, and am I fluent in it?"

The archetype lens does earn its keep in one place: role negotiation. When you take a principal role, the biggest risk is that you and your VP have different archetypes in mind — they want a Solver, you expect to be an Architect, and six months later both parties are disappointed. Name the expected mode mix explicitly before accepting.

## 4. What a principal actually produces

If code is no longer the primary output, what is? Five artifact classes, in rough order of leverage:

**Decisions.** The core product. A principal exists to make (or force) the decisions that are too cross-cutting, too long-horizon, or too politically expensive for any single team: which serving stack survives, whether to build or buy the labeling pipeline, when the migration is killed. A decision has a written record — context, options, tradeoffs, choice, revisit trigger — or it will be relitigated quarterly forever. A one-page decision doc that prevents three teams from independently solving the same problem is worth more than any quarter of your code.

**Strategy documents.** The subject of Module 02. One or two a year, each moving tens of engineer-years. Rare and expensive; most of the work is the diagnosis behind them.

**Standards and golden paths.** "Every model ships with an eval and a rollback plan." "New services use the platform serving stack unless a written exception is approved." Standards are decisions amortized across the org — the mechanism by which your judgment scales past your calendar. The failure mode is standards nobody follows; a standard requires an enforcement joint (review gate, CI check, launch checklist) or it is a blog post.

**Reviews.** Design reviews, launch reviews, incident reviews. This is where the pattern library pays out: fifteen minutes of "this migration plan has no rollback story past step 3, and step 3 is where the last two migrations died" saves a team a quarter. A principal who reviews ten designs a month influences more architecture than they could ever build.

**Unblocking.** The least visible and often highest-value output: the two-day prototype that proves the vector-DB vendor's latency claims false before the contract is signed; the phone call that gets two directors to stop fighting over feature-store ownership; the reframing that turns a stuck 6-month debate into a 2-week experiment. Tanya Reilly (*The Staff Engineer's Path*) calls much of this "glue work" — at senior it is career-dangerous because it is unlegible; at principal it is the explicit job, and part of the skill is making it legible.

Notice what is absent: feature velocity. A principal measured on personal ticket throughput is being measured wrong, and — harder to accept — a principal who *optimizes* for personal ticket throughput is hiding from the actual job in comfortable work.

## 5. What stays technical

The counterweight to everything above: **you must remain the strongest technologist in the room, or the whole model collapses.** Influence without authority runs entirely on technical credibility. The day the org suspects your opinions are secondhand — that you're pattern-matching on blog posts instead of contact with the systems — your review comments become suggestions and your strategy docs become shelfware. The failed-principal population is bimodal: those who never stopped being seniors, and those who stopped being engineers.

Concretely, the 15–25% of time that stays hands-on goes to:

- **Prototypes that de-risk decisions.** Before recommending the org consolidate on vLLM, you have personally load-tested vLLM against the incumbent on your actual traffic shapes, and your strategy doc cites *your* numbers. A weekend prototype that kills a bad $2M platform bet is the highest-ROI code you will ever write. This is Amazon's "Technically Fearless" tenet in practice.
- **Code review on critical paths.** Not volume review — targeted review of the joints where defects become incidents: the feature-store point-in-time join, the eval harness's metric computation, the traffic-shifting logic in a migration. You review 2% of the org's code, chosen to be the 2% where a bug costs seven figures.
- **Reading code you didn't write.** Diagnosis (Module 02) requires ground truth. Org-level claims like "our five serving stacks share no code" are verified by reading, not by asking team leads, who will each tell you their stack is nearly done being great.
- **Staying current by building, not skimming.** One personal project per major technology wave, enough to have opinions with fingerprints on them. Depth in the new thing every 12–18 months beats breadth in everything.

The discipline is protecting this time *and keeping it off the critical path*. The moment your code blocks a team's launch, you have taken a senior engineer's job and abandoned your own — and you will do it badly, because you're in meetings all day.

## 6. The operating cadence

The delta becomes real at the level of a calendar week. A representative shape for a working principal MLE — not a prescription, but a calibration target:

```text
Deep work (design, writing, diagnosis)      ~30%   protected blocks, mornings
Prototyping / critical-path code review     ~20%   the "stays technical" budget
Design & launch reviews                     ~15%   2–4 reviews/week, prepared for
1:1s, mentoring, hallway unblocking         ~15%   staff+ engineers, leads, skip-levels
Planning, exec syncs, org meetings          ~10%   where decisions get pre-wired
Slack / interrupts / glue                   ~10%   deliberately capped
```

Three structural observations. First, **writing is not a line item because it permeates deep work** — a principal writes constantly (decision docs, review feedback, strategy drafts) because writing is the only influence mechanism that scales past your meeting capacity and survives your absence. Second, the calendar is **maker-manager hybrid** and degrades to pure manager without active defense; most principals who lose the 20% technical budget lose it 30 minutes at a time and notice a quarter later. Third, note what dominates: *reviewing and unblocking other people's work* now outweighs producing your own. If that trade feels like loss rather than leverage, that is worth knowing about yourself before pursuing the role — it is the honest reason some excellent engineers deliberately stay senior, and they are not wrong to.

## 7. Anti-patterns

Four failure shapes, each a virtue overextended:

**The Architecture Astronaut.** Produces frameworks, taxonomies, and five-year visions untethered from any shipping system; the strategy deck has 40 slides and zero numbers from production. Root cause: fled the technical work (section 5) and now operates on abstractions. Detection: when did they last read production code or a real incident review? Their proposals get polite nods and no adoption.

**The Gatekeeper.** Converts "reviews things" into "everything routes through me." Design reviews become approval tollbooths; the org's decision latency inflates; teams start routing around them, at which point they've lost both throughput *and* influence. Root cause: measuring their own value in vetoes. The correction is standards and golden paths — encode the judgment so it scales without your calendar. A principal's success metric is decisions that go *well without them*.

**The Ghost.** Operates only in exec meetings and 1:1s; produces no written artifacts; engineers two levels down cannot name a thing they've done. Often genuinely busy — but influence that lives only in meetings dies in those meetings, compounds nothing, and is invisible to promotion and calibration committees. Detection: grep for documents authored in the last two quarters.

**The Hero.** Still the best debugger in the org and proves it weekly — parachutes into every incident, personally rewrites the slow service, works 60-hour weeks shipping. Beloved, and a systemic failure: every fire they fight personally is a team not learning, a root cause not fixed, and a bus factor of one. The Solver archetype metastasized into an identity. The tell: their org's capability does not grow. The principal question is never "can I fix this?" — it is "why does this class of fire keep happening, and what makes it stop happening without me?"

All four share one root: continuing to optimize a metric from a previous level (elegance, correctness, presence, personal output) instead of the principal metric — **the org's decision quality and capability, compounding over years.**

## 8. How the promotion actually works

The mechanics matter because they explain the standard advice — "do the job before you have the title" — which sounds like exploitation until you see it from the committee's side.

**Committees promote on evidence, not potential.** At every large company, staff+ promotion runs through a packet: your artifacts, your scope, and testimony from people your work affected. The committee's question is not "could this person operate at L7?" but "has this person *been* operating at L7, such that the title is a correction?" This is rational from their side: the senior→principal transition is the one where extrapolation fails most often, precisely because the job changes rather than scales. The consequence for you: the exercises in this course exist to be packet evidence. A strategy doc that redirected a real decision, a migration plan a team executed, a standard the org adopted — these are the only currency the committee accepts.

**Scope is taken, not given — but it must be *available*.** The honest version of "act at the next level" has a precondition most advice omits: your org must contain unclaimed principal-shaped problems. Signs it does: recurring cross-team debates nobody resolves, decisions made by default rather than by anyone, a leadership team asking vague questions like the VP's email in the worked example below. Signs it doesn't: a principal already covering your area well, or a company too small to have org-level ML problems. In the second case the move is a team change or a company change, not harder work — a fact your gap analysis (this module's exercise) should state plainly if it's true, because two years of waiting for unavailable scope is the most common way strong seniors lose half a decade.

**Sponsorship is a mechanism, not favoritism.** Somebody in the promotion-committee room has to spend their own credibility asserting your scope claims are real. That person needs to have *seen* the work — which is why operating invisibly (the Ghost) is fatal even when the work is excellent, and why the written-artifact discipline of section 4 doubles as promotion infrastructure: documents travel into rooms you are not in. Module 13 covers cultivating sponsorship deliberately; for now, the test is one question — name the person two levels up who could describe, without preparation, three decisions you improved this year. No name, no packet.

**External loops shortcut differently.** Interviewing for principal externally (Module 15) replaces the packet with 5–6 hours of live performance, which trades a two-year evidence trail for the ability to *narrate* one convincingly. This is why the course's artifacts serve both routes: they are simultaneously packet evidence and the raw material for "tell me about a time you set technical direction."

## Going deeper

- Will Larson, *Staff Engineer* (2021) — the archetypes and the "getting the title where you are" mechanics; read it alongside the Ewerlof/Goedecke critique in section 3 rather than instead of it.
- Tanya Reilly, *The Staff Engineer's Path* (2022) — the best treatment of glue work, the maker-manager calendar problem, and staying technical while leading.
- Amazon's Principal Engineering Tenets (public) — worth reading verbatim; they are the most honest public description of the *judgment* expectations at this level, as opposed to the scope expectations ladders describe.
- levels.fyi's leveling comparisons — useful for calibrating what L7/E7/Principal means at a specific company before an interview loop or offer negotiation; titles are wildly non-portable and this is the exchange-rate table.
- The decision-journal practice in section 1 descends from forecasting research (Tetlock's *Superforecasting* is the accessible source); the transfer to engineering judgment is direct.

## You can now

- Distinguish principal-level work from senior work by artifact class — decisions, strategy docs, standards, reviews, unblocking — rather than by coding velocity or ticket throughput
- Use a decision journal to build calibrated confidence deliberately: log positions with explicit confidence percentages and falsifiers before outcomes are knowable, then score monthly as a forecaster would
- Identify which of the four failure anti-patterns — Architecture Astronaut, Gatekeeper, Ghost, Hero — is your nearest failure mode under pressure, and name the standing structural countermeasure (protected technical time, writing cadence, review rotation) that prevents it
- Separate scope-gap problems (the role doesn't offer principal-shaped problems) from operating-model problems (the scope exists and you're not taking it), and produce a one-page gap analysis citing specific artifacts and calendar facts — not vibes
- Explain the promotion mechanics end-to-end: packet evidence, scope as taken not granted, and sponsorship as credibility transfer — and name the specific artifact missing from your current trajectory

## Worked example

**Setting:** You are the principal MLE at a 400-engineer B2B SaaS company (~60 in ML across six teams: search, recommendations, fraud, support-automation, platform, data). Wednesday of last week, the VP Eng forwarded you a one-line email: *"Support-automation wants to fine-tune and self-host. Fraud says we should stay on the API. Both want budget. Thoughts by planning (next Friday)?"* This is the job: a vague, cross-team, seven-figure decision with a deadline. Here is the week.

**Monday.** Two protected morning hours: pull the actual numbers before talking to anyone, because both teams' slide decks will be advocacy. From the cost dashboards and a query against the LLM gateway logs: support-automation runs 1.4M requests/day at ~2 300 input / 400 output tokens against a frontier API — roughly **$14k/day, $420k/month**, growing 8%/month. Fraud runs 90k requests/day, spiky, hard-latency-bound at 150 ms p99 — ~$9k/month. Already the shape is visible: these are not one decision. One is a high-volume, stable-distribution workload (the self-host cost crossover from the ML System Design course clearly applies); the other is low-volume and latency-critical where GPU utilization would sit under 5%. Afternoon: three design reviews, one of which — the platform team's eval-service proposal — you spend real preparation on, because it intersects this decision: self-hosting a fine-tune without eval infrastructure is how companies ship regressions at $420k/month scale.

**Tuesday.** Solver-mode morning, unplanned: recommendations' nightly training job has failed three days running and the on-call is stuck; 45 minutes of reading the failing join with them locates a schema change upstream — but you don't fix it; you pair the on-call with the data-platform lead and file the pattern (no data contracts, third incident this quarter) into your notes as strategy-doc ammunition for next quarter. Afternoon: 1:1s with the support-automation and fraud leads, listening for what the email didn't say. Support-automation's lead reveals the real motivation is only 60% cost — they believe a fine-tuned model on their 2M-ticket corpus would *outperform* the general API on their distribution. Plausible, and testable. Fraud's lead mostly fears being conscripted onto someone else's GPU cluster. Real constraint surfaced: neither team has serving-infra experience; the platform team has two engineers with vLLM scars.

**Wednesday.** The technical 20%: you spend the day personally standing up an 8B open-weights model under vLLM on two spot H100s ($5.80/hr total), replaying 5 000 anonymized production requests from support-automation's logs. Result: ~6 100 decode tok/s sustained per GPU, quality on the team's own golden set at 71% versus the API's 78% — *before any fine-tuning*. You now own the two numbers the whole decision turns on: the cost floor is real (roughly $60–90k/month all-in at their volume, a ~4–6× saving), and the quality gap is real but plausibly closable with fine-tuning on their corpus. Nobody can wave either number away, because you measured both on production traffic.

**Thursday.** Writing day. A two-page decision doc: **Diagnosis** — one workload where self-hosting saves ~$4M/yr at current growth if fine-tuning closes a 7-point quality gap; one workload where it is strictly worse. **Recommendation** — (1) fraud stays on the API, revisit at 10× volume; (2) support-automation gets a *staged* bet: six weeks, two engineers plus platform support, to fine-tune and prove ≥ API-parity on the golden set and 500-ticket human-rated sample — kill criteria written down, in advance, in the doc; (3) serving runs on the platform team's stack, not a new team-owned one — explicitly forbidding the sixth serving stack; (4) the eval-service proposal from Monday gets funded first, as the gate for step 2. Late afternoon: pre-wiring (Module 02 covers the mechanics) — 20 minutes each with the platform lead, the fraud lead, and the VP's right-hand director. Two objections surface; one (spot-instance availability for the latency SLO) is real, and you amend the doc to reserved capacity, moving the estimate to $110k/month. Still a 4× saving. The doc is better because it was pre-wired; the meeting will be a formality because it was pre-wired.

**Friday.** Planning meeting: eight minutes on your doc, two clarifying questions, approved as written. The decision that arrived as a budget fight between two teams leaves as a staged experiment with kill criteria, a strengthened platform, and a precedent — the golden-set-plus-kill-criteria template will now be the org's default for every future self-hosting request. Afternoon: two hours of protected time drafting the data-contracts strategy doc that Tuesday's incident motivated, and a 30-minute mentoring session with the staff engineer on the fraud team, to whom you hand the follow-up work of writing the API-volume revisit trigger into the cost dashboard — visible, scoped, promotion-legible work for them.

**The audit.** Count the week in the table's currency: one seven-figure decision made with measured numbers and written kill criteria; one prototype that anchored it; three reviews; one incident converted into strategy ammunition and a pairing instead of a hero fix; one standard set by precedent; one staff engineer given legible scope. Lines of production code shipped: zero. Engineer-years redirected: about fifteen. That is the delta.

## Exercise

Audit your own last month against the delta table, then write two documents.

**Step 1 — Data collection (90 min).** Go through your last four work weeks — calendar, merged PRs, documents authored, reviews given, decisions you influenced. Classify every significant block of time into the section 6 categories (deep work / prototyping+review / reviews / mentoring / planning / interrupts) and compute your actual percentages. Then, for each row of the delta table in section 2, write one honest sentence placing yourself: your *actual* scope (blast radius in engineer-years, not job title), horizon of the furthest-out thing you influenced, artifacts produced, coding ratio and — more important — coding *character* (features vs prototypes/critical-path), the ambiguity level of the problems as they arrived at your desk, the furthest team your influence reached, and any business number you could be held accountable for.

**Step 2 — Gap analysis (one page, strict).** Three sections: (a) the two or three rows where your gap to principal is largest, with the evidence; (b) the *root cause* per gap — distinguish "my role doesn't offer this scope" (an environment problem) from "the scope is available and I don't take it" (an operating-model problem); most people find at least one of each; (c) which anti-pattern from section 7 you are nearest to under stress — everyone has one; naming it is the point.

**Step 3 — 90-day plan (half page).** Three commitments, each mapping to a delta-table row, each with a concrete artifact and a date. Good: "By March 15, write and socialize a decision doc for the embedding-model upgrade currently being argued about in Slack" or "Take over design-review facilitation for the two teams adjacent to mine; five reviews by end of quarter." Bad: "get more strategic," "improve communication." One of the three must produce a written artifact that someone two teams away will read.

**You're done when:** the gap analysis fits on one page and every claim in it cites a specific artifact or calendar fact from the last month (no vibes); the 90-day plan's three commitments each name a deliverable, a date, and an audience beyond your own team; and you have shown the gap analysis to one person who will tell you the truth — your manager or a staff+ peer — and recorded where they disagreed with your self-assessment.

**Self-check questions:**

1. Of the decisions made around you last month, how many did you influence with a *written* artifact versus a meeting opinion? What does that ratio say about whether your influence survives your absence?
2. What fraction of your coding time was prototypes-that-de-risk-decisions or critical-path review, versus feature work a strong senior on your team could have done?
3. When did you last change your public position on a technical question because of evidence? If you can't recall one, is that calibration — or is it that you never state positions crisply enough to be wrong?
4. Which problem arrived at your desk this quarter *least* defined, and did you return a problem statement and recommendation, or a request for clearer requirements?
5. If you were promoted to principal on Monday, which specific anti-pattern from section 7 would claim you by June — and what standing structure (protected technical time, a review rotation, a writing cadence) would you install now to prevent it?
