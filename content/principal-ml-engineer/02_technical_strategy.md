# Module 02 — Technical Strategy: Writing the Document That Moves the Org

## Why this module matters

Strategy documents are the highest-leverage artifact a principal produces: one good one redirects tens of engineer-years; the absence of one means fifty engineers each locally optimizing into five serving stacks, three feature stores, and no shared evals. Yet most documents titled "ML Strategy" are not strategies — they are roadmaps, wish lists, or vision decks, and the org can feel the difference even when it can't name it. This module teaches the Rumelt structure (diagnosis → guiding policy → coherent action), the ML-specific diagnosis work behind it, and the writing and socializing mechanics that determine whether the document moves anything. Module 01 established that decisions are your product; strategy is the document form of a decision that binds many future decisions.

## 1. Strategy ≠ roadmap

A roadmap answers "what will we build, in what order?" A strategy answers "given our situation, what approach do we bet on — and what do we therefore refuse to do?" The distinction is Richard Rumelt's (*Good Strategy Bad Strategy*, 2011), and his definition of bad strategy reads like an audit of ML org documents: fluff ("we will leverage state-of-the-art AI to delight customers"), failure to face the problem (no diagnosis anywhere in the deck), mistaking goals for strategy ("reduce inference cost 40%" — that's a wish; *how*, and at the cost of what?), and bad objectives (a list of 12 initiatives with no connecting logic).

The structural test: **a real strategy forbids things.** If nobody would object to your strategy — if it contains no sentence that costs some team something they want — it is not a strategy; it is a mood. "We will have great evals and low costs and fast serving" forbids nothing. "We consolidate on one serving stack, which means search re-platforms and loses their custom batching for ~2 quarters" is a strategy, and the search lead's objection is evidence it's real. Will Larson (*Crafting Engineering Strategy*, 2025) makes the same point operationally: strategy is "making a decision once instead of many times" — and a decision that excludes nothing decides nothing.

Roadmaps are downstream. Once the strategy says "API-first, then distill high-volume paths," the roadmap (which paths, which quarters) follows almost mechanically, and — the real payoff — it can be *re-derived* when circumstances change, because the logic that generated it is written down.

## 2. The Rumelt kernel, for engineers

Three parts, in order, no skipping:

**Diagnosis.** A falsifiable statement of what is actually going on — the constraint, the trend, the asymmetry that defines the situation. Not "AI is moving fast" (true of everyone, decides nothing) but "we run 23 models; four generate 91% of ML-attributed revenue; our 60 engineers spend an estimated 55% of their time on undifferentiated infrastructure that vendors now sell." A good diagnosis is the hard 80% of the work, and it is checkable: someone can audit your numbers and tell you you're wrong. That's a feature.

**Guiding policies.** The approach — a small number of rules (2–4; five is a smell) that channel every future decision without prescribing each one. Policies for the situation above might be "consolidate before building anything new" and "buy undifferentiated infra, build only where our data is the moat." Each policy must visibly *follow from* the diagnosis and visibly *exclude* alternatives someone reasonable would advocate.

**Coherent actions.** Concrete, resourced, sequenced commitments that implement the policies — named owners, dates, and the thing most bad strategies omit: what stops. Actions must cohere: if action 3 quietly assumes a platform team that action 1 defunds, the kernel is broken. Coherence is where wish lists die, because wish lists are, definitionally, actions with no shared logic.

## 3. Diagnosis for ML organizations

The generic kernel becomes an ML skill in the diagnosis. ML orgs hide their problems in places generic engineering strategy doesn't look, so the inventory has a specific shape. Budget 2–4 weeks of part-time work for an org of 50–100 engineers; the strategy is unwritable without it.

**Inventory the models.** Every model in production: owner, business metric it claims to move, last retrain date, last eval date, serving path, monthly cost. Expect three findings at almost any company: (1) a power law — a handful of models carry nearly all the value; (2) zombie models — still serving, still costing, owner departed, nobody dares turn them off; (3) claimed business impact that sums to more than the company's revenue. All three are diagnosis material.

**Inventory the pipelines and stacks.** Count the *distinct* ways the org does each fundamental: training orchestration, feature computation, serving, evals, monitoring. The count is usually a surprise to leadership ("wait, five serving stacks?") and each redundant stack has a carrying cost you can estimate: typically 0.5–1.5 engineers of maintenance plus the option-value loss of non-transferable engineers and non-shareable improvements.

**Inventory the spend.** GPU hours (training and inference separately), API tokens by workload, vendor contracts, and — the number nobody has — *loaded people-cost by activity*. If 60 engineers at $300k loaded cost each spend 55% of their time on infrastructure plumbing, that's a **$9.9M/year line item** that appears in no budget review because it's smeared across every team. Making smeared costs visible is half of what an ML diagnosis is for.

**Find the constraint.** Goldratt's question, applied to an ML org: what actually gates the rate at which the company turns ideas into shipped model improvements? Candidates, in rough order of how often they're the true answer: eval infrastructure (nobody can tell if a change is good, so everything ships slowly and fearfully), data access and labeling latency, serving-stack fragmentation (every improvement implemented five times), GPU quota politics, and — rarely but expensively — actual modeling talent. The constraint is where the strategy's force concentrates; a strategy that improves non-constraints is expensive noise. A useful probe: take the org's last three model launches and time-line them — where did each spend the most calendar days waiting? The answers cluster fast.

**Interview, then verify.** Ask each team lead: what would you build if the platform were perfect? What do you re-implement that someone else already built? What are you afraid to touch? Then verify the load-bearing claims by reading code and dashboards yourself (Module 01, section 5) — leads' answers are honest but each sees one facet, and at least one confident claim per diagnosis will turn out to be false on inspection.

## 4. Guiding policies for ML orgs — worked examples

Policies that recur across real ML strategies, with the tradeoff each one buys and the alternative it forbids:

- **"Consolidate on one serving stack; exceptions require a written waiver."** Buys: transferable engineers, shared optimizations, one on-call rotation. Costs: the best team's custom stack gets worse before the shared one gets better; 1–2 quarters of migration tax (Module 08). Forbids: the sixth stack, however locally justified.
- **"API-first; distill and self-host only paths exceeding a stated volume/cost threshold."** Buys: speed to product-market fit, no idle GPUs, deferred ops burden. Costs: unit economics at high volume, some data-flywheel latency. Forbids: speculative GPU purchases and "we'll self-host because it's cooler" projects. (The crossover math is in the ML System Design course; the *policy* is what makes the math binding.)
- **"No model ships without a versioned eval and a rollback plan."** Buys: the ability to move fast later — evals are the brakes that let you drive fast. Costs: real friction now, an eval-infra investment first. Forbids: vibes-based launches, including the CEO's favorite demo.
- **"Buy undifferentiated infrastructure; build only where proprietary data or workload shape gives us an edge."** Buys: focus. Costs: vendor risk, lock-in exit costs (Module 09 prices these). Forbids: the in-house feature store rewrite.
- **"One embedding/foundation model family per modality, upgraded on a cadence, not per-team."** Buys: shared caches, comparable evals, one fine-tuning pipeline. Costs: no team rides the newest release the week it drops. Forbids: per-team model zoos.

Note the pattern: every policy has a *forbids* clause. Drafting policies, write the forbids clause first — if you can't, the policy is fluff. And keep policies at the level of *approach*, not implementation: "consolidate serving" is policy; "use KServe 0.13" is an action (and next year's action may differ under the same policy).

## 5. Writing mechanics: the document that gets read

Execs and busy leads read the first page; everyone reads the first paragraph. Structure accordingly:

**BLUF — bottom line up front.** The first paragraph states the diagnosis in one sentence, the bet in one sentence, and the cost in one sentence. If the reader stops there, they still know what you're asking for. Burying the recommendation on page 4 "for context" is the most common structural mistake in engineer-written strategy.

**One page + appendix.** The strategy proper — kernel, costs, what-we-won't-do, asks — fits on one to two pages. Everything that makes it *credible* (the model inventory, the cost model, the stack comparison, the interview notes) goes in appendices that reviewers can spot-check. The page limit is not about attention spans; it forces you to know which sentences are load-bearing.

**Decision-ready framing.** The document ends with explicit asks the reader can approve or reject: headcount, budget, a mandate, a stop-work. A strategy with no ask is a memo. Name the decision-maker; name the date the decision is needed by; name what happens by default if no decision is made (there is always a default, and it is usually "the five stacks become seven").

**Numbers with stated assumptions.** Every claim that could be a number is a number, and every number shows its assumption ("assumes H100 at $2.50/hr reserved; sensitivity: at $4/hr spot-constrained, the crossover moves from 600k to 950k req/day"). Precise-looking numbers with hidden assumptions destroy trust exactly once.

**Steelman the alternatives.** One short section stating the strongest case for the paths not taken, in terms their advocates would accept. This is not politeness — it is how reviewers verify you understood the tradeoff, and it is what makes disagree-and-commit possible later, because dissenters see their argument recorded rather than strawmanned.

### The template

```text
# [Strategy title] — [org], [date], [author], status: DRAFT | IN REVIEW | ACTIVE | SUPERSEDED

## Bottom line
Three sentences: the diagnosis, the bet, the cost. Written so an exec
who reads nothing else can still make the decision.

## Diagnosis
What is actually going on. The constraint. 3-6 falsifiable claims,
each with a number and a pointer to appendix evidence.

## Guiding policies
2-4 rules that channel future decisions. Each policy states:
  - the rule
  - what it FORBIDS (mandatory — no forbids clause, no policy)
  - the tradeoff we are knowingly accepting

## What we will NOT do
Explicit list. The projects, purchases, and paths this strategy
declines, including ones with real advocates. If this section is
empty or uncontroversial, the strategy is fluff.

## Coherent actions
| # | Action | Owner | Resources | Start | Done-when | Depends on |
Sequenced, resourced, with kill/success criteria where applicable.
Includes STOP actions (what we cease doing), not just starts.

## Costs and risks
What this costs (money, time, morale, capability we give up), what
could make it wrong, and the leading indicator for each risk.

## Alternatives considered
Steelman of each rejected path: strongest honest case for it, and
the specific reason the diagnosis says otherwise.

## Revisit triggers
Conditions that reopen this strategy (metric thresholds, market
events, assumption failures) — and a default review date (6-12 mo).

## Asks
The specific approvals needed, from whom, by when, and the default
outcome if no decision is made.

## Appendices
A: model/pipeline/stack inventory   B: cost model with assumptions
C: constraint analysis / launch timelines   D: interview summaries
```

## 6. Socializing: strategy is a campaign, not a document

A strategy that debuts fully-formed in a large review meeting will be shredded — not because it's wrong, but because you asked twenty people to publicly update their positions in real time with their teams watching. The socializing sequence:

**Pre-wiring.** Before any formal review, walk the document through 1:1s with every person whose "no" could kill it and every person whose team pays a cost under it — roughly 5–10 conversations for an org-level strategy. In each: present the diagnosis first (people accept conclusions whose premises they helped verify), invite attack, and *visibly incorporate* what survives. Two outcomes, both wins: the document improves, and its eventual reviewers walk in having already engaged, half of them now minor co-authors. If a pre-wiring conversation surfaces an objection you can't answer, you found it in a room of two instead of twenty. This is standard operating procedure at Amazon and Google for exactly this reason; it is not politics, it is protocol design for how groups actually change their minds.

**Review circles, inside-out.** Sequence reviews from friendly-but-rigorous (a staff+ peer who will mark up every weak number) → the affected leads → the formal org review → the exec ask. Each circle hardens the document for the next. Skipping circles to "save time" moves the same objections into the highest-stakes room.

**Disagree-and-commit, explicitly invoked.** Some disagreements are real and survive good-faith argument — the search lead may *never* agree that losing custom batching is worth it, and may be reasonable. The resolution is not consensus (you'll wait forever) or steamrolling (you'll pay in sabotage-by-lethargy). It is the named decision-maker deciding, the dissent recorded in the doc ("Search disagrees with policy 1 on latency-risk grounds; see revisit trigger 3"), and an explicit commitment ask. Recording dissent is what makes commitment psychologically possible — the dissenter's position survives in writing, ready to say "told you so" if the trigger fires. Amazon's formulation of disagree-and-commit exists because the alternative equilibria — endless relitigation or silent non-compliance — are both worse than being wrong occasionally.

**After ratification: repetition.** A strategy is absorbed by an org after roughly the seventh repetition — in planning reviews ("this doesn't fit policy 2 — waiver or redesign?"), onboarding, design-review templates. The document is the constitution; you are, for two quarters, its broadcast mechanism. If you're not slightly bored of saying it, the org hasn't heard it yet.

## 7. Lifecycle: strategies expire

An ACTIVE strategy with no revisit mechanism becomes dogma, and in ML the half-life of load-bearing assumptions is short — API pricing has repeatedly dropped ~10× within 18 months, and a single model release can invalidate a build-vs-buy diagnosis. Write revisit triggers when you write the strategy, because you know the assumptions best at that moment: *"Reopen if: frontier API pricing drops below $X/1M (invalidates the distill-crossover math); support-ticket volume 2× (invalidates capacity assumptions); the platform team loses >2 of its 6 engineers; or 12 months elapse."* When a trigger fires, the update is cheap precisely because the kernel is explicit — you re-examine the diagnosis, and either it holds (re-ratify, new date) or it doesn't (revise the policies that depended on the failed claim, mark the old doc SUPERSEDED with a pointer forward). Orgs trust strategists who visibly update; the credibility cost of "the diagnosis changed, here's the revision" is negative. What kills trust is the strategy that everyone privately knows is dead but that still governs planning because nobody has the standing to say so. Killing your own strategy on-trigger is a principal-level flex; schedule the opportunity.

## 8. The signature failure: the wish list

You will read many documents shaped like this: a vision paragraph, then "our five strategic pillars" (quality, velocity, efficiency, innovation, talent), then 15 initiatives, every team's pet project among them, no costs, no sequencing, nothing declined. This is Rumelt's bad strategy in its purest corporate form, and it is *popular* because it is politically free — everyone's project is a pillar, nobody loses, the meeting ends warmly. It then changes nothing, because a document that ranks nothing cannot resolve any actual resource conflict, which is the only job a strategy has. The tell is always the same: **no tradeoffs, nothing forbidden, nobody mad.** When you review such a document — and as a principal, part of your job is reviewing them — the kindest useful question is: "Which of these fifteen things do we do *first*, and which two teams' requests get declined because of it?" If the room can't answer, there is no strategy yet; there is an inventory of hopes. The same test applies, uncomfortably, to your own drafts: find the sentence that will make a specific named person push back. If it's missing, you've written a wish list with better formatting.

## Worked example

**Setting.** Meridian, a B2B logistics-tech company: 340 engineers, ~60 in ML across five teams (ETA prediction, pricing, document-AI, support-automation, ML platform — the platform "team" is 3 engineers). You joined as the first principal MLE eight weeks ago. Leadership's vague unease, verbatim: "ML feels slow and expensive and we can't say why." Four weeks of diagnosis (inventories, launch timelines, 14 interviews, reading three of the serving codebases yourself) produced the document below — reproduced in full, at realistic length (~1.5 pages), because the artifact *is* the lesson.

```text
# Meridian ML Technical Strategy: Consolidate, Then Compound
Org: ML Engineering (60 eng) | Author: [Principal MLE] | 2026-03-09
Status: IN REVIEW | Decision needed by: 2026-03-27 (H2 planning lock)

## Bottom line
We run 21 production models on five serving stacks with no shared
eval infrastructure; four models generate 88% of measured ML value,
and our launch bottleneck is evaluation, not modeling. We bet H2 on
consolidation: one serving stack, one eval standard, and an API-first
policy for new GenAI work — deferring all new platform construction
and two approved model initiatives to fund it. Cost: ~14 engineer-
quarters and two quarters of reduced feature velocity, against an
estimated $4.1M/yr in recovered engineering capacity and a launch
cycle cut from 9 weeks to a target of 3.

## Diagnosis
1. Value is concentrated; effort is not. 4 of 21 models (ETA, spot-
   pricing, doc-extraction, support-triage) carry 88% of ML-attributed
   revenue impact (appendix A). 6 models are zombies: no owner, no
   retrain in >12 months, ~$41k/mo combined serving cost.
2. Five serving stacks (two homegrown Flask+Docker, one SageMaker,
   one KServe, one vendor GenAI gateway) cost ~5.5 eng in maintenance
   (appendix B) and make every cross-cutting improvement a 5x task.
   No engineer can work on another team's serving path without a
   multi-week ramp.
3. The constraint is evaluation. Time-lining our last 6 launches
   (appendix C): median 9 weeks idea-to-production, of which 4.5
   weeks is ad-hoc evaluation — every team hand-rolls eval sets,
   and two launches shipped regressions that evals would have caught,
   including the Q4 pricing incident (-$310k).
4. Undifferentiated toil dominates. Interview + calendar sampling
   puts 50-60% of ML engineering time on infra plumbing that is
   table-stakes elsewhere: at $300k loaded cost x 60 eng, a $9-11M/yr
   smeared line item (appendix B, assumptions stated).
5. Our durable advantage is data, not infrastructure: 9 years of
   shipment outcomes, 40M labeled freight documents, 2.1M support
   resolutions/yr. Nothing we build in serving or orchestration is
   a moat; what we do with this corpus is.

## Guiding policies
P1. One serving stack. KServe on EKS becomes the org standard.
    Forbids: new deployments on any other stack from 2026-04-01;
    the sixth stack, forever, absent a written waiver from the
    principal MLE + VP Eng.
    Accepted tradeoff: pricing team loses its custom dynamic-batching
    layer (~15ms p99 regression, appendix B4) until KServe parity
    work lands in Q4.
P2. No model ships without a versioned eval and a rollback plan.
    Enforced via launch checklist starting 2026-05-01, once the
    shared eval service (A2) reaches v1.
    Forbids: vibes-based launches; "we'll add evals after GA."
    Accepted tradeoff: real friction for small launches; est. +1 week
    on first launch per team, amortizing to ~zero by third use.
P3. API-first for new GenAI work; self-host only paths that exceed
    $50k/mo sustained API spend AND have a fine-tune quality case.
    Forbids: speculative GPU reservations; team-owned inference
    clusters. (Support-triage currently qualifies at $38k/mo and
    growing 9%/mo — expected to cross the threshold ~August; the
    eval standard (P2) must be in place first, which is sequencing,
    not coincidence.)
P4. Build only on the data moat. New platform construction beyond
    A1-A3 is deferred in H2.
    Forbids: the proposed feature-store rewrite; the internal
    experiment-tracking tool (buy: est. $60k/yr vs 2 eng to build).

## What we will NOT do in H2
- The feature-store rewrite (2 eng-quarters requested by platform).
- The pricing team's RL initiative (strong team advocacy; deferred
  until eval infrastructure can actually measure it — see P2).
- Self-host any LLM before support-triage crosses the P3 threshold.
- Keep the 6 zombie models: owners assigned or sunset by 2026-06-01.
- Grow ML headcount. The diagnosis says capacity is trapped, not
  absent; we ask for zero new heads and return velocity instead.

## Coherent actions
A1. Serving consolidation: migrate 15 active non-KServe models to
    KServe. Owner: platform lead + 1 eng borrowed from each team.
    8 eng-quarters, Apr-Sep. Done-when: old stacks decommissioned,
    on-call rotations merged 5->1. (Migration plan: Module-08-style
    strangler sequence, appendix E.)
A2. Shared eval service v1: versioned eval sets, judge-model harness,
    CI regression gates. Owner: [staff eng, doc-AI]. 4 eng-quarters,
    Apr-Jul. Done-when: all 4 top models have versioned evals wired
    into launch checklist.
A3. Zombie sunset + model registry of record. 1 eng-quarter.
    Done-when: 21 models -> ~14, each with a named owner.
STOP: feature-store rewrite (now), custom batching development
    (after parity work lands), per-team eval tooling (on A2 v1).

## Costs and risks
- ~14 eng-quarters of opportunity cost; roadmap features slip ~1
  quarter in Q2-Q3. Leading indicator: sprint velocity per team.
- Pricing p99 regression until Q4 parity work (risk: SLA breach on
  spot-pricing API; mitigation: parity work is A1's first milestone).
- Platform team (3 eng) is a single point of failure for A1+A2;
  mitigation: borrowed-engineer model doubles as cross-training.
- Morale risk on pricing (RL deferral) and platform (rewrite
  declined): named, owned by me and their leads in 1:1s, revisit
  trigger 4 gives the RL case its path back.

## Alternatives considered
- "Platform first: build the ideal stack, then migrate." Strongest
  case: migrating twice is waste. Rejected: our constraint is evals,
  not platform capability, and a bigger build extends the window in
  which five stacks keep diverging. KServe at parity is good enough
  to consolidate onto now.
- "Self-host GenAI now for support-triage." Strongest case: the
  cost crossover is ~5 months out and fine-tuning on 2.1M
  resolutions likely beats the API on-distribution. Rejected as
  sequencing, not direction: without P2's eval standard we cannot
  measure the fine-tune, which is how companies ship $400k/yr
  quality regressions with confidence.
- "Do nothing / grow into it." Default if no decision. Cost: stacks
  become 7 (two teams have prototypes), the $9-11M/yr toil line
  grows with headcount, and eval debt compounds at ~1 incident/2
  quarters (base rate from the last 18 months).

## Revisit triggers
1. Frontier API pricing drops >3x (reopens P3 threshold math).
2. Support-triage crosses $50k/mo sustained (activates P3 case).
3. A1 slips >6 weeks past the Jun milestone (re-scope or kill A1's
   long tail: migrate top-8 models only, sunset the rest harder).
4. Eval service v1 live + pricing team re-proposes RL with an eval
   plan (their path back, on the record).
5. Default review: 2026-09-15, H2 planning.

## Asks
- VP Eng: ratify P1-P4 by Mar 27; co-sign the waiver bar on P1.
- Directors (pricing, platform): disagree-and-commit recorded for
  the batching regression and rewrite deferral respectively;
  dissents logged in appendix F, tied to triggers 3-4.
- Finance: no new spend; approve $60k/yr experiment-tracking vendor
  from the recovered zombie-serving budget ($41k/mo).
```

**Why this document works — an audit against the module.** The bottom line is three sentences and decision-ready. Every diagnosis claim carries a number and an appendix pointer, and claim 3 — the constraint — came from time-lining real launches, not from asking teams what they wanted. Each policy has an explicit forbids clause and a named tradeoff with a specific loser (pricing's 15ms, platform's rewrite). The NOT-do list is genuinely contested — the RL initiative had director-level advocacy — which is how you know it's a strategy. Actions include stops. Dissent is recorded and wired to revisit triggers, which is what made disagree-and-commit acceptable to the pricing director in pre-wiring (conversation four of nine; the trigger-4 path-back was *his* amendment, which converted him from opponent to co-author). And the asks are approvable in one read by the person whose calendar matters. Total length: under two pages plus appendices. This document took four weeks of diagnosis and four days of writing; that ratio is correct and typical.

## Exercise

Write a technical strategy for your own ML org. If you don't currently have one (or can't use real numbers publicly), use this fictional brief: *a 45-engineer ML org at a marketplace company — 3 product ML teams and a 4-person platform team; 14 production models; three serving paths (SageMaker, a homegrown gRPC stack, and a vendor LLM gateway at $85k/mo growing 12%/mo); evals exist only on the search team; last two quarters shipped one revenue-negative launch that took 5 weeks to detect; leadership is asking whether to build a fine-tuning capability or "keep renting intelligence."*

**Deliverable.** A strategy document following the section 5 template, maximum two pages plus appendices. Real (or defensibly estimated) numbers in the diagnosis — for the fictional brief, invent internally consistent ones and state assumptions. At least two guiding policies, each with an explicit forbids clause and a named tradeoff. A **"What we will NOT do"** section containing at least three items, at least one of which a real constituency in the org actively wants. Revisit triggers tied to your diagnosis's load-bearing assumptions. An asks section naming the decision-maker and the decision date.

**Then pressure-test it (this is half the exercise):** give the document to one colleague and ask them to attack the diagnosis only — is the constraint claim falsifiable, and did you bring evidence or anecdote? Separately, identify the person (real or in the fictional org) angriest about your NOT-do list and write their strongest objection in their voice, then either amend the strategy or add their dissent + a revisit trigger.

**You're done when:** the bottom line survives being read aloud in three sentences to someone with no context and they can state back what you're betting on and what it costs; every diagnosis claim is falsifiable (a skeptic could check it against a dashboard, codebase, or timeline); every policy forbids something specific; the NOT-do section would make at least one named person push back; and the asks could be approved or rejected in a single meeting without follow-up questions.

**Self-check questions:**

1. Cover your policies and show a colleague only your diagnosis: do they derive approximately your policies from it? If they derive different ones, is your diagnosis underspecified — or are your policies not actually connected to it?
2. Which sentence in your document will make a specific, named person push back? If you can't name the person, which section is fluff?
3. If your central diagnosis claim turned out to be 2× off (the toil estimate, the constraint, the growth rate), which policies survive? A strategy where everything hinges on one uncheckable number is a bet dressed as an analysis — did you state it as one?
4. What is the default outcome if nobody approves your strategy — did you write it down, and is it genuinely worse than your proposal, or is "do nothing" secretly competitive?
5. Look at your coherent actions: which existing work do they STOP? If the answer is none, where is the capacity coming from — and have you just written a wish list with a template?
