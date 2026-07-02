# Module 16 — Career, Portfolio & Interview Strategy

## Why this module matters

Technical skill is necessary but not sufficient. Interviewers see dozens of candidates who can describe a two-tower retrieval system; they hire the ones who can also prove they built something, articulate impact, and communicate as a peer engineer. This module covers the human side of the ML career — portfolio construction, reading interviewers, progression from junior to staff, and the reading habits that keep technical knowledge from going stale. It is the consolidated home for career strategy in this course: other chapters point here rather than repeat it.

The framing throughout: a 45-minute design interview is a performance, not a test. The rubric from the interview playbook chapter is about how you navigate, not just what you know. Everything in this module is downstream of that distinction.

---

## 1. Portfolio building

The purpose of a portfolio is not to show that you know things — a cover letter does that. It is to show that you have *done* things, in a way a hiring manager can verify in under five minutes. Everything else is secondary.

### 1a. What to include beyond code

A GitHub repository full of notebooks is not a portfolio. The artefacts that differentiate:

**Design documents.** A one-to-three-page doc per project following the structure from the foundations chapter: problem statement, constraints, the design you chose, the alternatives you considered, the metrics you optimized for, and the failure modes you anticipated. This signals that you designed the system before you coded it — the most reliable predictor of a senior-capable engineer. Even a small project looks substantive with a design doc attached.

**Eval reports.** For any project involving a model: the offline metrics (what you measured, on what split, sliced by what dimensions), the numbers before and after key changes, and — for any GenAI project — judge calibration figures (agreement with a human-labeled sample). An eval report produced by the capstone project is richer interview ammunition than three toy model demos without one. The evaluation and observability chapter is the reference for what a credible eval report contains.

**Architecture diagrams.** A clean sequence or data-flow diagram for each project's online serving path. Use whatever tool renders to a readable image in the README — hand-drawn and photographed is fine; clarity matters, not the tool. Interviewers scanning your repo will spend more time on a diagram than on any amount of prose.

**Benchmarking results.** Latency-throughput curves, throughput-versus-batch-size sweeps, cost-per-1k-requests tables — any numbers you measured yourself. "I profiled this and found X" is a sentence that gets you deeper questions; "I implemented the system" is a sentence that gets you a nod.

**Incident postmortems.** A write-up of something that broke during a project, structured as root cause → detection lag → fix → prevention. Even a simulated incident — you introduced a feature-distribution shift and diagnosed it using the triage sequence from the evaluation and observability chapter — demonstrates operational thinking. It distinguishes engineers who have shipped from engineers who have prototyped.

The cumulative capstone from the preceding module is explicitly designed to produce all five of these artefacts. Treat that deliverable as the portfolio centerpiece; the earlier module projects as supporting evidence.

### 1b. Structuring the README as a story

The README is the interview before the interview. A reader should be able to answer three questions in under 90 seconds: What problem does this solve? What did you build, and what were the hard parts? What did you measure and what did you find?

The structure that works:

1. **One-sentence problem statement.** Specific, not generic. "A two-stage retrieval and ranking pipeline for a 200M-item product catalog, optimized for hybrid BM25+dense retrieval with RRF fusion" beats "an e-commerce recommendation system."
2. **Architecture summary with a diagram.** The diagram first, then a paragraph explaining the design choices — not the implementation, the *decisions*. What did you trade off and why?
3. **Key results.** Concrete numbers: latency p99 at X QPS, recall@100 before and after reranker, cost per 1k requests, eval metric on a fixed held-out set. Numbers you didn't measure are not results.
4. **What surprised you.** One honest paragraph about what didn't go as expected. This is the section interviewers read most carefully — it shows that you ran into real production behavior, not just happy-path training.
5. **How to reproduce.** A working command sequence. Not required for every project, but any project you reference in an interview should run.

Avoid: introductory sections that spend three paragraphs describing what transformers are; capability lists masquerading as results ("supports batch inference, multiple models, real-time serving"); and READMEs that describe the code structure rather than the system design.

### 1c. Linking projects to job descriptions

ML engineering job descriptions cluster around three archetypes, and each reads your portfolio differently.

**Product MLE** (own an end-to-end feature, work closely with PM and DS, ship to users): cares about business impact, feedback loop design, and A/B testing. Lead your portfolio walk-through with the eval report and the online-metric story; reference the design doc's metric-definition section.

**Infra/platform MLE** (build the serving platform, feature store, training infrastructure): cares about serving latency, throughput, reliability, and operational simplicity. Lead with the benchmarking results and the architecture diagram; reference incident postmortems.

**Research-adjacent MLE** (reproduce and adapt frontier research, run experiments, often at a lab-adjacent team): cares about empirical rigor — clean ablations, honest comparisons against baselines, understanding of the literature. Lead with the eval report and any quantization or training experiments; reference the baselines you beat and the ones you didn't.

Before any interview loop, read the job description carefully and map your three strongest projects onto what that role cares about. You will walk through one project in depth — know which one, and practice the five-minute version until it is automatic.

---

## 2. Interview strategy

### 2a. Reading the interviewer's focus

The same system design question asked by three different interviewers — a product-focused EM, a research scientist, a platform engineer — expects three materially different answers. Read the signals.

**Product/applied focus:** the interviewer asks about user experience, fallback behavior, how you'd know it's working in production, the A/B story, and what you'd monitor. They are less interested in training math; they are probing system thinking and product judgment. Lean into metrics, the feedback loop, and simplest-thing-that-works with named upgrade conditions.

**Research/modeling focus:** the interviewer asks about model architecture choices, why this loss over that one, how you'd handle label noise, what happens at scale in the training data distribution. They are probing whether you understand the ML, not just the infrastructure. Have the math ready — the scaling-laws reasoning from the training chapter, why BF16 not FP16, the RLVR-versus-DPO tradeoff — and be willing to go two levels deep.

**Infrastructure/platform focus:** the interviewer asks about latency breakdown, rolling deployment, rollback story, GPU count and justification, and what breaks at 10× scale. They are grading practicality and operational awareness. Lead with capacity math, failure modes, and deployment story.

Most interview loops contain all three lenses in different sessions. The skill is reading which lens is active and shifting emphasis mid-answer without losing coherence. The rubric from the interview playbook chapter — problem navigation, breadth, depth on demand, tradeoff reasoning, practicality — maps directly onto these three foci; knowing which axis the current interviewer weights most is how you allocate the remaining time.

### 2b. Handling "I don't know" better than bluffing

Bluffing is the fastest way to fail an interview. Senior interviewers hear dozens of candidates per quarter and detect hedged fabrication within two exchanges. Structured honesty with momentum is both more respected and more reliable:

"I haven't implemented that specific thing, but let me reason from first principles — [then actually reason]." A candidate who works toward a sensible answer from adjacent knowledge is more interesting to hire than one who recites a memorized answer that stops at the surface.

"I'm not certain of the exact number, but I can bound it — [upper-bound reasoning / lower-bound reasoning, with stated assumptions]." Back-of-envelope from stated assumptions beats confident fabrication and demonstrates the arithmetic instinct interviewers are grading.

"I know this is an active area; the last thing I read suggested X, but I'd verify before making a production decision." Intellectual honesty with awareness of your own staleness is a senior trait, not a weakness.

What you should not say: "I don't know" and stop. The silence kills momentum. Always follow "I don't know" with what you *do* know and a direction you'd explore. The interview is a conversation about tradeoffs, not a quiz with right answers.

### 2c. Asking good questions; effective follow-up

The questions you ask at the end of the interview are data about your priorities and your engineering judgment. Generic questions ("what does success look like in this role?") are forgettable. Questions that demonstrate you have thought about the team's actual technical problems are remembered:

- "The job description mentions [X system or challenge]. What's the hardest part of that problem at your current scale?"
- "How do you currently manage training-serving skew — is that solved infrastructure, or still a manual discipline?"
- "What's the eval story for [the team's main product]? Do you have a golden set with CI gates, or is it mostly A/B-driven?"

These questions serve two purposes: they get you useful information about whether the team operates with engineering discipline, and they signal to the interviewer that you think in systems rather than tasks. In an interview, volunteering the right question is as much a senior signal as volunteering the right failure mode.

**Follow-up after the loop.** A brief, specific note within 24 hours — not a generic thank-you, but a sentence referencing something concrete from the conversation. If you discussed a specific design problem, add one additional thought you had afterward. This is optional but takes five minutes, and the signal it sends about how you communicate is disproportionate to the effort.

---

## 3. Career progression

### 3a. What changes at each level

The skills are cumulative, but the *rate-limiter* shifts at each level.

**Junior MLE (0–2 years).** The rate-limiter is technical execution: given a well-specified task, can you ship it reliably? You are expected to go deep on a narrow problem, ramp quickly on an unfamiliar codebase, ask good clarifying questions, and produce code that doesn't break adjacent systems. The failure mode at this level is quality — bugs, missed edge cases, and implementations that work in isolation but break in production.

**Mid-level MLE (2–5 years).** The rate-limiter shifts to scope and judgment: given an under-specified problem, can you define the subproblems yourself, make the right tradeoffs, and deliver without being micromanaged? You are expected to own a feature end-to-end — data pipeline, training, serving, eval, monitoring — and to know when to escalate versus when to decide. The failure mode at this level is scope management: taking on too much, or not driving ambiguous projects to closure.

**Senior MLE (5–8 years).** The rate-limiter is influence: can you make the *team* better? A senior is expected to set technical direction for a system — not just a feature — unblock others, catch architectural mistakes before they are built, and produce design docs good enough to hand off to teammates who will implement them without your presence. Writing design docs that get implemented correctly without you in the room is the senior-level test. The failure mode is hoarding — doing the work yourself because it is faster, at the cost of team growth and your own leverage.

**Staff MLE (8+ years).** The rate-limiter is cross-functional leverage: can you change how the organization approaches a class of problems? Staff engineers define platforms, write standards that teams across the org adopt, and own the multi-year technical trajectory of a domain. The output is other teams' productivity, not your own code. The failure mode is staying local — solving the immediate team problem without asking whether a platform answer would solve it for ten teams at once.

In an interview, volunteering awareness of what your *next* level requires — "I'm mid-level and I'm actively working on driving ambiguous projects without a well-scoped spec" — is itself a senior signal. Most junior and mid-level candidates describe their current level's skills; candidates who get strong offers describe the gap they are working to close.

### 3b. The MLE promotion trap (and how to avoid it)

The most common reason technically strong MLEs stall at mid-level is invisible work. Training runs, debugging data pipelines, diagnosing model regressions — this work is high-value but low-visibility unless you make it visible. Concrete practices:

**Write design docs even for small projects.** A half-page doc before a two-week experiment is searchable, referable, and attributable six months later; a Slack message is none of those things. Design docs are the career artefact; experiments without docs are invisible.

**Present results, not work.** "I ran twelve experiments over four weeks" is invisible. "I ran twelve experiments over four weeks and concluded that preprocessing quality dominated model architecture — here is the table and here is what we should do next" is a result. Compress the journey; lead with the finding.

**Volunteer failure modes.** In design reviews, being the person who says "the classic failure here is X" signals that you have shipped before, not that you have been lucky. This pattern is practiced throughout this course's interview Q&A format — it transfers directly to promotion conversations.

**Pull scope toward you.** Stalling at mid-level almost always involves waiting to be handed a well-scoped problem. Engineers who get promoted are the ones who noticed something nobody had scoped yet and wrote the one-pager. The capstone project and the design journal recommended in the course README are explicit practice for this motion.

### 3c. Transition paths: SWE→MLE and DS→MLE

**SWE→MLE.** Software engineers transitioning to ML bring strong engineering fundamentals — systems thinking, code quality, debugging discipline — and typically weaker ML fundamentals: model mechanics, evaluation discipline, training infrastructure. The practical path: get ML fundamentals to the level covered in this course; contribute to an ML-adjacent system first (feature pipeline, serving infrastructure, eval tooling) before claiming model ownership; and demonstrate that you understand the eval story, not just the code. The gap that stalls SWE→MLE transitions is underestimating how different evaluation discipline is from software testing — model performance on a held-out set is not the same as unit tests passing, and the evaluation and observability chapter is where that difference is worked out.

**DS→MLE.** Data scientists transitioning to ML engineering bring strong statistical intuition and model selection judgment, and typically weaker systems thinking: serving, latency, production reliability, and scale. The practical path: build a production serving project end-to-end, even a small one; get comfortable with the serving and inference optimization chapters; and practice framing answers in terms of systems — what breaks at 10×, the rollback story, training-serving skew — rather than just experiments. The gap that stalls DS→MLE transitions is treating inference as "deploying the pickle." A serving infrastructure that is not thought through stops working the moment latency requirements become non-trivial, and debugging it without the vocabulary from the inference and serving chapters is painful.

---

## 4. Staying current — reading list and communities

ML engineering evolves faster than most engineering disciplines. Techniques from two years ago are production standard; techniques from one year ago are interview-expected; techniques from six months ago are frontier. The only sustainable approach is a reading habit, not a one-time course.

### 4a. Books — the durable foundations

Books age worse than papers for ML specifics but age well for foundations and systems reasoning. Read these once; they remain load-bearing:

- *Designing Machine Learning Systems* (Huyen) — the best single-volume treatment of production ML systems for practitioners. Read before any system design interview loop.
- *Designing Data-Intensive Applications* (Kleppmann) — not ML-specific, but the distributed-systems and database reasoning it contains underlies every feature store, data pipeline, and serving infrastructure decision in this course.
- *Deep Learning* (Goodfellow, Bengio, Courville) — the canonical reference for ML fundamentals. You do not need to read it cover-to-cover; the chapters on optimization, regularization, and sequence models are the most interview-relevant.
- *The Pragmatic Programmer* (Hunt, Thomas) — for engineering discipline. Its lessons on orthogonality and automation apply directly to ML system design and to the postmortem discipline this course emphasizes.

### 4b. Papers and engineering blogs — by course topic

Papers age fast but the seminal ones remain the vocabulary reference. Engineering blogs are the fastest way to see what production teams are actually doing. Read by topic, not exhaustively.

**Data engineering and feature platforms.** The original feature store papers from major recommendation teams are public and worth reading for the motivation, not just the solution. Engineering blogs from large-scale recommendation teams (search, social, commerce) document the real-world pressure that drove the design patterns in the data-engineering chapter.

**Training and post-training.** The Chinchilla-era scaling laws papers are required reading — the IsoFLOP analysis reasoning recurs in every training-efficiency conversation. LoRA and QLoRA papers are short and teach the PEFT mindset directly. For post-training, the original DPO paper and the GRPO paper are readable and explain the mechanics behind the techniques in the training infrastructure chapter.

**Inference optimization.** The FlashAttention papers and the vLLM paper (PagedAttention) are foundational for the serving chapter vocabulary. The speculative decoding papers — original and batched variants — are short and mechanically instructive.

**LLM serving.** System papers on disaggregated prefill-decode inference are the current frontier. Read the papers behind any open-source serving engine you use — the engineering motivation is exactly what you reference in a serving-infrastructure interview.

**Retrieval and RAG.** The DPR paper introduced bi-encoder fine-tuning on in-batch negatives; the ColBERT papers introduced late interaction; the HyDE paper is the cleanest treatment of query-side augmentation. For RAG evaluation, the RAGAS paper is the practical reference for the evaluation setup described in the retrieval and RAG chapter.

**Agentic systems.** The ReAct paper is the foundational reference for tool-use reasoning. Research on prompt injection in agentic contexts is active and publicly documented — reading current work before an interview on agent security is table stakes, given how aggressively that topic is probed in 2026.

**Classic ML systems.** The YouTube DNN paper (multi-task retrieval and ranking at scale), the DLRM paper (embedding tables at industrial scale), and the Wide & Deep paper (the two-stage architecture that recurs in every recsys design) are all short and readable. These three cover the structural vocabulary of the classic ML systems chapter.

**Evaluation and MLOps.** Evaluation harness and infrastructure papers from major labs document the eval-driven development discipline. For observability, the OpenTelemetry GenAI semantic conventions specification is the reference for the tracing data model described in the evaluation and observability chapter.

**Engineering blogs.** The most valuable reading for keeping current — faster signal than papers on what has crossed into production standard. Read the engineering blogs of companies you target before their interview loop; interviewers ask about problems their team is currently solving, and you want to have read their write-ups.

### 4c. Conferences and communities

**Conferences worth tracking.** NeurIPS, ICML, and ICLR publish the research that becomes production standard in 12–24 months. MLSys (the ML systems conference) publishes the infrastructure papers — the PagedAttention paper, FlashAttention, and most of the serving infrastructure work in this course originated or was formalized there. RecSys publishes industry-track papers on recommendation systems at scale. SysML/EuroMLSys are smaller but often have the highest practitioner signal-to-noise.

You do not need to read conference proceedings cover-to-cover. Track the best-paper lists and the proceedings of one or two tracks most relevant to your current role. Set a calendar reminder for each conference's proceedings release — the papers that define next year's interview vocabulary appear there first.

**Communities.** The most useful signal comes from practitioners who are shipping, not from commentary about shipping. Communities organized around specific tools — open-source serving engines, orchestration frameworks, observability platforms — tend to have higher practitioner density than general ML communities. GitHub issue trackers and Discord servers for actively-developed open-source projects are often the fastest place to encounter production failure modes and real workarounds, both of which are more interview-relevant than polished blog posts.

**A sustainable reading habit.** One practical system: one paper and one engineering blog post per week; a private document with a one-paragraph summary and a "so what for my current work" note per entry. After a month, the summaries are more valuable than the originals — you have built a retrievable corpus of applied knowledge rather than a list of links you will not revisit. After a year of this, the breadth and recency of your knowledge become a visible differentiator in interviews.

---

## References

- *Designing Machine Learning Systems* (Huyen) is the single-volume reference for the production systems topics across this course.
- MLSys conference proceedings are the primary archive for the serving infrastructure papers — PagedAttention, FlashAttention, disaggregated inference — referenced throughout the technical chapters.
- The post-training papers cited in the training chapter (LoRA, DPO, GRPO) are each readable in an afternoon and repay the investment in interview vocabulary many times over.
- The evaluation harness literature and the OpenTelemetry GenAI semantic conventions are the references for the eval-driven development and observability disciplines from the evaluation and observability chapter.
- Engineering blogs of companies operating at scale are the fastest signal on which research techniques have crossed into production standard; read the blogs of companies you target before each interview loop.

---

## Project 16 — Portfolio audit and mock interview debrief

This project has two parts, both done with a peer.

**(1) Portfolio audit (two hours).** Exchange GitHub portfolios with someone else in the same job search. For each project in your partner's portfolio, answer without asking them: (a) What problem did they solve? (b) What was the hardest design decision? (c) What did they measure — find a concrete number. Score each question 0–2. A project that scores below 4/6 needs a design doc, an eval report, or a results section before it goes in front of a hiring manager. Fix the weakest project first; the bottleneck is almost always the absence of an eval report, not the quality of the code.

**(2) Recorded mock, then critique (two hours).** Record yourself answering one of the mock questions from the interview playbook chapter — 45 minutes on a whiteboard or shared doc, timer running, no notes. Then watch the recording and count on the four axes from the rubric: assumptions stated, tradeoffs explicitly compared, numbers produced, failure modes volunteered. Fewer than three on any axis — drill it specifically before the next recording. Do the same for your peer and give written feedback. One recorded mock with honest critique against the rubric is worth more than ten self-assessed walkthroughs.

*Deliverable: the audit scorecard, the patched project README or new design doc, and the written critique of one mock.*

---

## Interview Q&A

**Q1. Walk me through a project from your portfolio.**
**A.** The structure that works: problem statement (one sentence, specific) → constraints that shaped the design → the key tradeoff you made and why → what you measured → what surprised you → what you'd do differently. The candidate who says "I built a recommendation system" loses to the candidate who says "I built a two-stage retrieval and ranking pipeline for a 200M-item catalog — made the architectural choice to keep retrieval as a bi-encoder rather than cross-encoder because the latency constraint ruled out full pairwise scoring, measured that position-bias correction improved offline nDCG by roughly 4%, and found the correction's calibration degraded on cold-start items, which only appeared when I sliced the eval by item age." The second answer demonstrates design thinking, tradeoff reasoning, and eval discipline in three sentences. Prepare a version at this level of specificity for each of your three strongest projects.

**Q2. What are you doing to stay current?**
**A.** The answer interviewers respect is a concrete, sustainable habit — not "I follow AI Twitter." Name one or two specific sources, name one specific thing you learned recently from them, and explain how it changed something you are doing or thinking about. "I read the disaggregated prefill-decode paper last quarter and realized our serving setup had a path to hit the latency SLO without scaling the GPU fleet, which changed my design proposal for the following quarter" — this is the answer. The ability to translate a research result into a concrete system decision is exactly the skill the interviewer is checking.

**Q3. What level are you targeting and why do you think you're ready for it?**
**A.** The answer is evidence, not assertion. Describe the skills the target level requires — using the framing from this module's career progression section — then map each to a concrete example from your work. If there are gaps, name them: "I'm confident on the technical execution and tradeoff reasoning axes. Where I'm still building is driving ambiguous projects without a well-scoped spec — I had one project like that last year and here is what I learned." Naming a known gap with a clear account of what you learned is more credible than claiming no gaps, and it is exactly how a mid-level-to-senior transition candidate should position themselves.

**Q4. Why should we hire you over a candidate with more years of experience?**
**A.** Evidence, not assertion: show what you built, what you measured, and how you think about systems — then demonstrate it rather than claim it. A junior candidate who has done the capstone project from this course, has an eval report with real numbers, and can navigate a system design question with explicit tradeoffs and volunteered failure modes is a stronger signal than a more experienced candidate who has been in CRUD-engineering roles and has never thought about training-serving skew or deployment risk. The portfolio is the evidence; the interview is the proof of concept. Frame it exactly that way.

---

*The technical skills in the preceding modules get you to the interview. The practices in this module get you through it — and into the role where you can actually use those skills.*
