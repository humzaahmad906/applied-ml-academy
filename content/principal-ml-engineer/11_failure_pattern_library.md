# Module 11 — The Production Failure Pattern Library

## Why this module matters

Principal engineers are not smarter than senior engineers; they have seen more things die. The judgment gap is mostly a pattern library — a compressed index of how ML systems actually fail in production, expensive enough that someone wrote a postmortem, public enough that you can learn it without paying the tuition. This module is 15 years of scar tissue, compressed: eleven failure patterns, each taught through canonical real cases with the root cause, the cost, and the one question a principal asks in a design review that catches it early. Read it twice — once for the stories, once for the questions. The consolidated checklist at the end is the artifact you take into every design review for the rest of your career; Module 10 told you how to run incidents and postmortems, this module tells you what the postmortems keep saying.

## Pattern 1 — Regime change: the distribution your training data has never seen

**Canonical case: Zillow Offers.** Zillow's iBuying arm used the Zestimate lineage of models to price homes it bought with its own balance sheet. Through early 2021 the models had trained on years of a low-volatility, steadily appreciating market — and worked. When the market regime shifted mid-2021 (pandemic-era whipsaw in prices, materials, labor), the models kept extrapolating the old regime, and the business layered an aggressive overbidding adjustment on top to win inventory. Q3 2021: a **$528M write-down**, ~7,000 homes to offload (many listed below purchase price), the entire division shut, and roughly 2,000 layoffs — a quarter of the company. The kill wasn't a bug; every component worked as designed. The training distribution simply stopped describing the world, and the system had no mechanism to notice that its own confidence was no longer earned.

**Root cause.** Point predictions consumed by an automated decision loop with no regime-shift detection, no uncertainty quantification feeding the *decision* (a wide prediction interval should shrink your bid, not just annotate it), and a business incentive (win inventory) pushing thresholds the wrong direction exactly when the model was most wrong. Slow feedback compounding: houses take months to resell, so the loss signal lagged the buying decision by two quarters.

**The principal's design-review question:** *"What happens to this system in a world its training data has never seen — and what is the earliest measurable signal that we've entered one?"* Follow-up: "What automated decision consumes these predictions, and does its aggressiveness scale down when model uncertainty scales up?"

## Pattern 2 — The proxy label encodes the wrong objective

**Canonical case: Optum.** A widely deployed healthcare risk model (analyzed in Obermeyer et al., Science 2019) predicted *future healthcare cost* as a proxy for *future healthcare need*, to allocate high-risk care-management programs. Black patients systematically generate lower costs at the same level of sickness (access barriers, undertreatment) — so the model rated them healthier than equally sick white patients. At a given risk score, Black patients had ~26% more chronic conditions; fixing the label roughly **doubled** the fraction of Black patients auto-enrolled (17%→46%). Affected: a model class applied to an estimated 200M people/year. The features weren't biased in any naive sense; the *label* was. No amount of model-side fairness tooling inspects whether the target column means what the product thinks it means.

**Canonical case: Amazon's resume screener.** Trained on ten years of hiring outcomes to predict "good hire," it learned that the historical process favored men — penalizing resumes containing "women's" (as in "women's chess club captain") and downweighting two all-women's colleges. Amazon's team tried to neuter the specific terms, couldn't guarantee the model wasn't finding new proxies, and killed the project (reported by Reuters, 2018). The label ("was hired/promoted here") was a fossil record of the old process, so the model's explicit job was to replicate it.

**Root cause.** Label choice is the single highest-leverage modeling decision and the least reviewed. Teams spend weeks on architecture and thirty seconds on "we'll predict cost / past hires / clicks" — importing every historical inequity and every gap between the measurable proxy and the actual objective.

**The principal's design-review question:** *"Walk me through the causal chain from the label to the thing we actually want. Who is systematically mis-measured by this proxy, and in which direction?"*

## Pattern 3 — Trained on synthetic/lab data, deployed to reality

**Canonical case: IBM Watson for Oncology.** Marketed as AI trained on cancer-patient data; in practice trained substantially on *synthetic cases* authored by a small group of clinicians at one hospital (Memorial Sloan Kettering), encoding one institution's preferences. Deployed globally into different populations, drug availabilities, and treatment guidelines: a Danish hospital found only ~33% concordance with its own tumor board and declined it; internal documents (STAT, 2018) recorded oncologists calling outputs "unsafe and incorrect." IBM sank an estimated **$4–5B** into the Watson Health bet (MD Anderson alone spent $62M before cancelling) before selling the unit off in 2022. The pattern beneath: the demo distribution was hand-curated; the deployment distribution was the world.

**Root cause.** Synthetic and lab data answer "can the model learn the mapping?" — they cannot answer "does the mapping hold in production?" Premature scaling turned a research prototype's eval regime into a global product's eval regime. Nobody ran the boring study — prospective concordance on each deployment site's own cases — before selling.

**The principal's design-review question:** *"What fraction of the training data was generated by the process we're deploying into — and what's our evidence from* that *distribution, per deployment site, not from the lab?"*

## Pattern 4 — Learning from user input is an attack surface

**Canonical case: Microsoft Tay (2016).** A conversational bot that learned from Twitter interactions in near-real-time. A coordinated group discovered the "repeat after me" mechanism and the online-learning loop, and within **16 hours** Tay was tweeting racist and genocidal content; Microsoft pulled it offline the same day. The technical failure is unremarkable; the pattern is eternal: *any system that updates from user-provided signals has handed its objective function to its most motivated users*. Modern instances are subtler — review-bombing a ranking model, poisoning an RLHF feedback channel, prompt-injecting an agent that writes to its own memory, coordinated click fraud steering a recommender.

**Root cause.** The team modeled users as a distribution to learn from, not as adversaries who can observe the learning rule and optimize against it. No rate limits per source, no anomaly detection on the feedback channel, no human gate between "user signal" and "model update."

**The principal's design-review question:** *"If a thousand coordinated accounts wanted to control this model's behavior through its own feedback loop, what would they do — and which of those moves would we detect before the next update ships?"*

## Pattern 5 — Out-of-distribution inputs in a safety-critical path

**Canonical case: Uber ATG, Tempe, 2018.** The first pedestrian killed by an autonomous vehicle. The NTSB report is required principal-level reading: the perception system detected the pedestrian (walking a bicycle across an unmarked section of road, at night) **5.6 seconds** before impact — but cycled her classification between *vehicle*, *bicycle*, and *other*, and on each switch reset the tracking history used to predict her path. The system had no concept of "jaywalking pedestrian" (training/design assumed crossings at crosswalks). Compounding: Uber had disabled the Volvo's factory emergency braking to avoid conflicts, and its own system suppressed emergency braking under a 1-second action delay designed to reduce false positives — with the human safety driver as the fallback, unalerted, and watching a phone.

**Root cause.** Layered: an OOD input the ontology couldn't represent; state-resetting classification churn; *and every fallback rung removed or degraded* (factory braking off, own braking suppressed, human monitor unmonitored). Note how this is a degradation-ladder failure in Module 10's terms — the ladder existed on paper and every rung was disabled in practice.

**The principal's design-review question:** *"For the inputs this system cannot classify, what does it do — and is the fallback for 'model confused' independent of the model that's confused?"*

## Pattern 6 — Patching the symptom instead of the root cause

**Canonical case: Google Photos, 2015–present.** Photos auto-tagged Black people as "gorillas." Google apologized and — fixed the classifier? No: it **removed the labels** "gorilla," "chimp," "chimpanzee," and "monkey" from the product. Wired verified in 2018 the labels were still blocked; The New York Times re-verified in **2023** — eight years on, the symptom-patch was still the fix, and Apple's Photos showed the same suppression. The patch was rational crisis management in week one. As a *permanent* state, it means the underlying representation problem (training data coverage, error-cost asymmetry across subgroups) was never re-prioritized once the press cycle ended.

**Root cause.** Organizational, not technical: incident pressure rewards the fastest symptom suppression; nothing in the org rewards reopening a "resolved" incident to fund the root fix. Every ML org accumulates these — the blocklisted query, the hardcoded override, the if-statement in front of the model — and each one silently narrows the product while the underlying failure generalizes to inputs the patch doesn't cover.

**The principal's design-review question:** *"Which behaviors of the current system are blocklist patches over model failures, when were they added, and what's the plan-with-a-date to retire each one?"* (Keep a registry of symptom-patches with expiry dates, exactly like security teams track accepted-risk exceptions.)

## Pattern 7 — Overcorrection as a failure mode

**Canonical case: Gemini image generation, February 2024.** Tuned to counteract the well-documented failure of image models to default to white subjects, Google's system applied diversity steering *unconditionally* — including to historically specific prompts, producing racially diverse Nazi-era German soldiers and non-white US Founding Fathers. Google paused person-generation entirely within days; Sundar Pichai called the outputs "completely unacceptable" in an internal memo, and the episode dominated a news cycle and dented the model's credibility at launch. The failure was not the goal (mitigating a real bias) but the mechanism: a blunt global intervention with no context-sensitivity and — the giveaway — apparently no eval suite for the *opposite* failure direction.

**Root cause.** Correcting a measured failure creates a new failure axis, and teams build evals only for the direction they were burned by. One-sided evaluation means you drive looking only in the rear-view mirror: you will not see the wall you're steering into.

**The principal's design-review question:** *"We're intervening to fix failure direction A — show me the eval for failure direction anti-A, and the threshold at which this intervention itself gets rolled back."*

## Pattern 8 — Architecture simplification without re-validation

**Canonical case: Tesla's radar removal, 2021.** Tesla dropped radar from new vehicles ("Tesla Vision"), moving to camera-only — a defensible cost/complexity/sensor-fusion-conflict argument. What followed: NHTSA investigations into **phantom braking** complaints (hundreds within months — 354 complaints in nine months by Feb 2022, later an investigation covering ~416k vehicles), temporary loss of IIHS Top Safety Pick and Consumer Reports recommendations while features (AEB at speed, following distance) were degraded post-removal, and Autopilot restrictions (speed caps, follow distance) that hadn't applied to the radar cars. Whatever the eventual end state, the transition shipped with capabilities objectively below the system it replaced.

**Root cause.** The redundant sensor wasn't only adding cost and fusion conflicts — it was masking failure modes of the remaining sensor (low sun, fog, overpasses reading as obstacles). Removing a component re-weights every input distribution downstream of it; the validation that certified the old architecture certifies nothing about the new one. This is the ML version of Chesterton's Fence: the radar was the fence.

**The principal's design-review question:** *"List the failure modes the component we're removing currently absorbs — and show me the re-validation on* those *slices specifically, not the global average metric."* (Global averages are where slice regressions go to hide.)

## Pattern 9 — Unverified generation in high-stakes output

**Canonical case: Mata v. Avianca (2023).** Two lawyers filed a federal brief containing six precedent cases generated by ChatGPT — all nonexistent, complete with fabricated quotes and citations; when challenged, one lawyer asked ChatGPT whether the cases were real and accepted its "yes." Judge Castel sanctioned them ($5,000, plus professional humiliation that became the canonical cautionary tale). **Air Canada (2024)** is the corporate version: its support chatbot invented a bereavement-fare refund policy; the airline argued the chatbot was "a separate legal entity responsible for its own actions"; the British Columbia tribunal ruled — precedent-setting — that a company is liable for information its AI gives customers, and ordered the refund. Small dollars, enormous principle: *your model's statements are your company's statements.*

**Root cause.** Generative output entered a high-stakes channel (court filing, binding customer communication) with no verification layer — no retrieval-grounding requirement, no citation checker, no "policy answers must quote the policy database" constraint, and a human in the loop who treated fluency as evidence of truth.

**The principal's design-review question:** *"For every claim this system can emit, what checks it against ground truth before a customer, court, or regulator sees it — and who is legally on the hook when it's wrong?"* (If the answer to the second half is "we are," the answer to the first half cannot be "nothing.")

## Pattern 10 — Evaluation shortcuts under time pressure

**Canonical case: COVID-19 prognostic models.** Wynants et al. (BMJ, 2020, living review) systematically appraised **145** published COVID diagnosis/prognosis models in the pandemic's first months: essentially all at high risk of bias — tiny non-representative samples, no external validation, outcome leakage, overfitting — and **none recommended for clinical use**. A companion Nature Machine Intelligence review of 62 imaging models reached the same score: zero. Hundreds of teams, working urgently and in good faith, produced a body of work with a collective clinical value of approximately nothing — because the step everyone skipped (external validation on data from a different site/time) was the step that constituted the actual evidence.

**Root cause.** "We'll validate properly later; people are dying / the launch is Thursday" — time pressure converts evaluation from a gate into a formality. The insidious part: shortcut evaluation still *produces a number*, and the number circulates stripped of its asterisks. An AUC from a leaked, single-site retrospective sample looks identical in a slide deck to a real one.

**The principal's design-review question:** *"If this eval is wrong, how would we know before users do? What here would survive an external validation — and if we're skipping one, say so in writing with a named owner of that risk."* (Making the shortcut explicit and owned is often enough to un-shortcut it.)

## Pattern 11 — The hidden technical debt classics

Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015) is a decade old and still the most-cited description of why ML systems rot. Four of its patterns deserve permanent slots in your library; you will see each within a year at any ML org:

- **Entanglement / CACE ("Changing Anything Changes Everything").** No feature is independent: reweight one input and the learned weights of all others shift. Mini-example: a team "cleans up" a redundant feature that a sibling team's model consumed via a shared embedding — sibling model regresses 3% and takes two weeks to trace. Prevention: isolate model boundaries; version features; never share mutable representations without a contract.
- **Undeclared consumers.** Your model's outputs get read by systems you don't know exist (Module 10's exercise hid one in the refunds system deliberately). When you "improve" the score distribution, an undeclared consumer's hardcoded threshold silently breaks. Prevention: access-controlled prediction outputs — consumers must register to read, which turns the dependency graph from folklore into a queryable table.
- **Hidden feedback loops.** Two systems influencing each other through the world: your pricing model changes demand, demand data trains your inventory model, inventory changes what pricing sees. Each team's metrics look fine; the composite system oscillates on a weeks-long period nobody's dashboard can see. Prevention: draw the loop diagram at design time; anywhere a model's outputs can reach another model's inputs through the world, add a holdback slice.
- **Pipeline jungles.** Five years of accreted glue: scrapes, joins, one-off backfills, a notebook that became a cron job. Cost is not aesthetic — it's that nobody can enumerate what the model actually consumes, which is why Pattern-4 poisoning and Module-10 Class-4 schema breaks go undetected. Prevention: periodic pipeline consolidation treated as tier-1 roadmap work with an executive sponsor, because it will lose every local prioritization fight (see Module 10's error-budget contract for how to fund it).

**The principal's design-review question:** *"Show me the dependency graph — every upstream data source with an owner, every downstream consumer by name. If we can't draw it, that's the finding."*

## The consolidated design-review checklist

Every catching-question, as one artifact. Run it against every new system design and every major change; it takes 45 minutes and it is the cheapest insurance in this course.

```text
PRODUCTION FAILURE PATTERN CHECKLIST — design review edition
Run by: ____________  System: ____________  Date: ________

 1. REGIME CHANGE   What does this system do in a world its training data
    has never seen? Earliest measurable signal we've entered one? Does
    decision aggressiveness scale down as model uncertainty scales up?
 2. PROXY LABEL     Causal chain from label to true objective — where does
    it break? Who is systematically mis-measured by this proxy, and in
    which direction?
 3. LAB-TO-REALITY  What fraction of training data comes from the actual
    deployment process? Per-site/per-segment evidence, or lab-only?
 4. ADVERSARIAL FEEDBACK  If 1,000 coordinated users wanted to steer this
    model through its own learning loop, what's the play — and which
    moves do we detect before the next update ships?
 5. OOD + SAFETY    For inputs the model can't classify: what happens?
    Is the fallback independent of the failing model? Is every rung of
    the degradation ladder actually enabled in production?
 6. SYMPTOM PATCHES  Which current behaviors are blocklists/overrides
    hiding model failures? Registry with owner + expiry date for each?
 7. OVERCORRECTION  For every intervention fixing failure direction A:
    where is the eval for anti-A, and the rollback threshold?
 8. SIMPLIFICATION  For any component being removed: which failure modes
    was it absorbing? Re-validation on those slices, not global averages?
 9. UNVERIFIED GENERATION  For each claim the system can emit: what
    verifies it against ground truth pre-delivery? Who is liable?
10. EVAL SHORTCUTS  What was skipped to hit the date? Is the skip written
    down with a named risk owner? Would this eval survive external
    validation on out-of-site, out-of-time data?
11. HIDDEN DEBT     Dependency graph drawn? Consumers registered?
    Feedback loops through the world mapped, with holdback slices?
    CACE: what breaks downstream if we change this feature?

Disposition per item: PASS / FINDING (owner + date) / ACCEPTED RISK (signed)
```

## You can now

- Recognize all eleven production failure patterns by their structural signature, name the canonical case that made each pattern famous, and explain why the same structure recurs across industries and system types.
- Apply the principal's catching-question for any pattern during a design review without prompting, and push every finding from "what happened to this system" to "what systemic change would prevent two other plausible incidents."
- Run the consolidated 11-item checklist against any new system design in under an hour, producing PASS, FINDING, or ACCEPTED RISK dispositions with named owners and dates for every item.
- Write an internal case study that identifies the dominant failure pattern, estimates cost with stated assumptions, and proposes a prevention that passes the two-other-incidents test.
- Distinguish the four patterns most commonly conflated in incident reviews — regime change versus adversarial feedback, and symptom patching versus overcorrection — and apply the correct defense to each.

## Worked example — running the checklist on "AskNorth," a bank's LLM support agent

**The design under review.** A retail bank (9M customers) proposes an LLM customer-support agent: frontier API + RAG over policy docs and account FAQs, able to quote fee/refund policies, initiate card blocks via tool call, and escalate to humans on low confidence. It will learn from thumbs-up/down feedback, fine-tuning a smaller in-house model quarterly on highly-rated transcripts to cut API cost. Target: deflect 40% of 2.4M annual contacts (~$8 saved per deflected contact ≈ $7.7M/year). The design doc is competent: latency budgets, cascade economics, eval on a 500-conversation golden set (89% resolution quality by LLM-judge).

The review takes 50 minutes and produces four findings and two accepted risks. The four findings:

**Finding 1 (item 9 — unverified generation, severity: launch-blocking).** The agent quotes fee and refund policies from RAG context, but nothing *constrains* policy statements to retrieved text — the model can blend retrieval with priors and state a plausible wrong policy fluently. Post-*Air Canada*, every such statement is binding on the bank. Required change: policy-class answers must be generated in quote-and-cite mode (verbatim policy text + generated framing), with a checker that verifies every figure (fees, timelines, thresholds) in the reply appears in the retrieved source; failures route to human. Estimated cost: 3 engineer-weeks plus ~40 ms latency. The team accepts; the $7.7M business case survives a 2-point deflection haircut.

**Finding 2 (item 4 — adversarial feedback, severity: high).** The quarterly fine-tune on thumbs-up transcripts is an open feedback channel: fraudsters probing for social-engineering-friendly responses can upvote them at scale (item 4's thousand-account question lands immediately — the bank *already* fights coordinated fraud rings, so the threat actor exists and is funded). Required change: fine-tuning corpus goes through the same anomaly screening as the fraud stack (account age, device clustering, rating-pattern outliers), plus human review of any transcript touching authentication, transfers, or limits before it enters training data. The reviewer notes this is Tay's lesson wearing a banking uniform.

**Finding 3 (item 5 — OOD in a consequential path, severity: high).** The card-block tool call is triggered by intent classification, and the design's fallback for "confused model" is... the same model asked to self-assess confidence. A distressed customer with atypical phrasing (non-native speaker, voice-transcription artifacts) can fail intent detection during an actual fraud event — the highest-stakes moment in the product. Required change: card-block intent gets an independent lightweight classifier (trained separately, different failure surface) as a parallel detector; disagreement between the two routes to a human within the SLA, and the phone-tree path to a block remains one keypress deep. Fallback independence is the whole point of item 5.

**Finding 4 (item 2 — proxy label, severity: medium, would have surfaced in month 6).** The online success metric is "deflection rate" — contact ended without human escalation. But a customer who gives up in frustration is a deflection; a customer wrongly told a fee is non-refundable and who doesn't push back is a *successful* deflection that costs a complaint to the regulator later. The proxy rewards exactly the failure the bank most fears. Required change: success = deflection *and* no re-contact on the same issue within 7 days *and* no complaint linkage within 30; a weekly 200-transcript human audit sample estimates the "wrongly satisfied" rate. (Note the Module 10 tie-in: this becomes the quality-SLO canary.)

Items 1, 3, 6, 7, 8, 10, 11 dispose as: PASS (3, 8 — no lab data, nothing removed), FINDING-minor (11 — consumer registration for the transcript stream), and ACCEPTED RISK signed by the product VP (1 — regime behavior under a bank-run-style news event, mitigated by a kill switch to human-only mode; 10 — golden set is synthetic-heavy pre-launch, with a committed 30-day post-launch external validation on real transcripts).

The meta-lesson: none of the four findings required inventing anything. Each is a 2016–2024 headline pattern-matched onto a 2026 design in under an hour. That is what the pattern library is *for*.

## Exercise

**Deliverable 1 — two internal case studies.** Pick two patterns from this module. For each, write a one-page internal case study of a near-miss (or full incident) from your own experience — or, if you genuinely have none, a plausible one set in your current system. Required structure: *Context* (system, scale, stakes) → *What happened / almost happened* → *Root cause in this module's vocabulary* (name the pattern; if it spans two, say which dominates) → *Detection: what caught it, or what would have* → *The cost, estimated in dollars or user-harm, even if rough* → *The systemic prevention* (must pass Module 10's test: would it prevent at least two other plausible incidents?).

**Deliverable 2 — extend your team's design-review template.** Take your team's actual design-doc template (or the checklist above, if you have none) and add **3 questions** tailored to your domain's specific failure surface — not copies of the eleven, but instances of them made concrete. Example of the required specificity: not "consider distribution shift" but "our document-AI models: what happens when a customer onboards a template format outside the 14 in training — what routes it to review before extraction errors reach their ERP?"

**You're done when:** each case study names exactly one dominant pattern and defends the choice; each includes a cost estimate with stated assumptions; each prevention is systemic by the two-other-incidents test; your 3 template questions are specific enough that a new hire could answer them about your system with a day of investigation; and you have actually opened a PR (or the org's equivalent) adding them to the template — the exercise is not complete in a private doc.

**Self-check questions:**

1. Zillow and Tay are both "the world fought back" stories — what's the structural difference between regime change (Pattern 1) and adversarial feedback (Pattern 4), and why do they need different defenses?
2. Your team proposes removing the rule-based fraud layer because the ML model "covers it now." Which two patterns does this trigger, and what evidence would you demand?
3. Why is Pattern 6 (symptom patches) fundamentally an *organizational* failure rather than a technical one — and what standing mechanism converts patches back into roadmap items?
4. The AskNorth review found the proxy-label problem (Finding 4) in the *metric*, not the training label. Why does Pattern 2 apply to online success metrics just as much as to training targets?
5. Which of the eleven patterns is your current production system most exposed to right now — and what did answering that question just add to your on-call playbook?
