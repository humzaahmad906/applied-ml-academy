# 15 — Risks and Safety: What Can Go Wrong and Who Owns It

Every model you ship is a liability surface. It will state falsehoods with total confidence, reproduce
the biases of its training data, occasionally regurgitate a phone number it memorized, and do whatever
a cleverly-worded input tells it to. None of this is exotic — it's the default behavior of a system
trained to imitate the internet and then optimized to be agreeable. This lesson is the practitioner's
taxonomy of those risks, the mitigations that actually move the needle, the 2026 governance reality you
now have to design around, and — the part that matters for your career — a concrete account of what an
*engineer* is responsible for versus what belongs to policy, legal, and leadership. Security-specific
attacks get a fuller treatment in the [AI threat landscape](../ai-security/01-the-ai-threat-landscape.md);
here we cover the whole risk surface and stay engineering-focused.

## The risk taxonomy

Four families cover most of what will bite you: hallucination, bias, privacy, and security. Misuse
(dual-use capability) cuts across all four.

### Hallucination: why models confabulate

A language model is trained to produce the *most probable continuation*, not the *true* one. When the
training distribution doesn't pin down an answer — an obscure fact, a made-up citation, a question
outside its knowledge — the objective still rewards a fluent, plausible completion. The model has no
built-in signal for "I don't know"; abstention is a *learned* behavior, and base models barely have it.
Kalai et al. (2025) framed it cleanly: standard training and evaluation *reward guessing over
abstaining*, because a benchmark that scores a blank as wrong and a lucky guess as right teaches the
model to always guess. Hallucination is thus partly a *training incentive* problem, not only a
capability gap.

Mitigations, in rough order of effectiveness:

- **Grounding / RAG.** Give the model the source text and instruct it to answer only from context (see
  [RAG and agents](09-rag-agents.md)). This is the single biggest lever — you convert an open-book
  recall task the model is bad at into a reading task it's good at.
- **Citations.** Require span-level attribution to retrieved sources so a human (or a checker) can
  verify. Citations that don't actually support the claim are themselves a failure mode to test for.
- **Abstention / calibration.** Train and prompt for "I don't know" and route low-confidence answers to
  a human. Confidence signals (verbalized, or logit-based) are noisy but usable as a gate.
- **Decoding and verification.** Lower temperature for factual tasks; use a verifier or self-consistency
  ([reasoning](11-reasoning.md)) for high-stakes answers.

There is no "turn off hallucination" switch. You reduce its *rate* on *your* distribution and you *catch*
the rest with grounding, verification, and human review. A useful framing for stakeholders: treat the
model like a fast, confident, sometimes-wrong intern — you don't fire the intern, you put review where
the cost of being wrong is high and let it run free where it isn't. The engineering deliverable is a
*measured* residual error rate on your task plus a routing policy that decides which outputs a human
sees, not a claim that the model is now accurate.

### Bias and fairness

Models inherit the statistical regularities of their data, including the harmful ones — occupational
gender stereotypes, dialect and name-based disparities, uneven quality across languages. This starts in
the embeddings (WEAT-style association tests, see [word vectors](02-word-vectors.md)) and survives
pretraining and, sometimes, post-training. Two things to keep straight. First, **measurement is the
hard part**: bias is task- and context-specific, so a generic "bias score" is nearly meaningless — you
measure disparity on *your* task (e.g. does resume screening rank equivalent candidates differently by
name?) with a counterfactual test set. Second, the **sources are the data and the objective**, so
mitigation lives across the stack: data curation, balanced fine-tuning, output filtering, and
post-hoc auditing. Don't promise "unbiased"; promise *measured, bounded, and monitored* disparity on
defined axes.

### Privacy: memorization, PII, and membership inference

LLMs memorize. Carlini et al. (2021, 2022) demonstrated **training-data extraction** — prompting a
model to regurgitate verbatim sequences from its training set, including names, addresses, and
secrets — and showed memorization *scales with model size, data duplication, and context length*.
Duplicated data is the biggest driver, which is why **deduplication** of the training corpus is both a
quality and a privacy control. Related attacks: **membership inference** (determine whether a specific
record was in training, a compliance problem under "right to be forgotten") and **PII extraction** (get
the model to emit personal data). Your controls: dedup and PII-scrub the training/fine-tuning data,
filter outputs for PII patterns, avoid fine-tuning on raw sensitive records, and treat anything the
model was trained on as *potentially extractable*. Deeper coverage in
[data privacy and leakage](../ai-security/04-data-privacy-and-leakage.md) if your role is security-adjacent.

### Security: injection, jailbreaks, misuse

**Prompt injection** — malicious instructions smuggled in through the model's *data* (a retrieved
document, a web page, a tool result) that hijack its behavior — is the defining vulnerability of
LLM applications, and it's unsolved in general because the model can't reliably distinguish trusted
instructions from untrusted content in the same context window (see
[prompting and PEFT](08-prompting-peft.md)). **Jailbreaks** are inputs that bypass safety training to
elicit prohibited content. **Misuse / dual-use** is the model being *capable* of harm (malware,
targeted disinformation, uplift for dangerous know-how) even when working as designed. These are
adversarial, ongoing arms races — you mitigate with defense-in-depth (input/output filtering, least-
privilege tool scopes, human approval for high-impact actions), never with a single fix. The full attack
and defense catalog is in the [AI security course](../ai-security/01-the-ai-threat-landscape.md); the
point for this lesson is that *security is a safety risk*, and shipping an agent with broad tool access
and no injection defense is negligence, not a v2 problem.

## Evaluating safety: red-teaming and benches

You cannot fix what you don't measure, and safety measurement is *adversarial*, which makes it different
from accuracy evals.

- **Red-teaming.** Actively try to break the model — manually and with automated attack generation
  (another LLM producing jailbreak attempts). The output is a growing corpus of failure cases that
  becomes a regression suite. Red-teaming is a *process*, not a one-time gate; capabilities and attacks
  both move.
- **Safety benchmarks.** Standardized suites (toxicity, refusal-on-harmful, over-refusal of benign
  requests) give comparable numbers but saturate and get gamed like any benchmark. Watch **over-refusal**
  specifically — a model tuned to refuse aggressively looks "safe" on harm benches while being useless,
  and that failure mode is easy to ship by accident.
- **Continuous monitoring.** Production traffic surfaces attacks your red team didn't imagine; log,
  sample, and review. Safety is an online metric, not just a pre-launch checkbox.

## Governance reality, 2026

The regulatory and licensing landscape stopped being optional. You don't need to be a lawyer, but you
need to know what constrains your design.

**EU AI Act.** In force and phasing in through 2025–2027. It's **risk-tiered**: outright *prohibited*
practices (e.g. social scoring, certain biometric categorization), *high-risk* systems (hiring, credit,
medical, critical infrastructure) with heavy obligations — risk management, data governance, logging,
human oversight, technical documentation, conformity assessment — and lighter *transparency* duties for
general-purpose and generative systems (disclose AI interaction, label synthetic media, publish training-
data summaries; the largest general-purpose models carry extra systemic-risk obligations). The practical
consequence: **if your NLP system touches hiring, credit, health, or education in the EU, you're likely
building a high-risk system** and the documentation and human-oversight requirements shape the
architecture from day one, not at the end.

**Model cards** (Mitchell et al., 2019) are the standard artifact for documenting a model: intended use,
out-of-scope use, training data provenance, eval results *broken down by relevant groups*, known
limitations, and ethical considerations. Datasheets do the same for datasets. In a governed setting these
aren't nice-to-haves — they're required documentation, and writing an honest one is an engineering task
that often falls to you. The value is in the *out-of-scope* and *limitations* sections: a card that only
lists strengths is marketing, and a reviewer will read the absence of stated limits as either
negligence or concealment.

**Licensing: open weight ≠ open source.** This distinction is routinely botched. **Open source** (by the
OSI definition) means source and freedom to use, modify, and redistribute for *any* purpose. Most
"open" LLMs are **open weight**: you get the weights under a *custom license* with restrictions —
Llama's community license has an acceptable-use policy and a >700M-monthly-active-user commercial gate;
Gemma has use restrictions; many "open" releases withhold the training data and code entirely, so they
aren't reproducible let alone open-source. Genuinely open-source-*ish* efforts (OLMo, with open data and
code) are the exception. Before you build on a model, **read its actual license** — commercial use,
distribution, and fine-tuning terms vary, and "it's on HuggingFace" tells you nothing about whether you
may ship it.

## What an engineer is actually responsible for

Cut through the noise. Not every risk is yours to *own*, but several are, and confusing the two is how
teams ship negligently or freeze uselessly. Yours to own:

- **Grounding, citations, and abstention** in the product surface — the technical hallucination
  controls.
- **Input/output filtering, least-privilege tool scopes, and injection defenses** for any agent — the
  technical security controls.
- **The safety eval and red-team regression suite** — building it, running it in CI, gating releases.
- **PII/dedup handling** in any data you fine-tune on, and not logging user data carelessly.
- **The honest model card** and the eval breakdowns that feed it.
- **Escalating** capability or misuse concerns you can see and product/policy can't.

Not solely yours (but you inform): the *risk tier* and legal classification, the acceptable-use policy,
which use cases are sanctioned, and the org's regulatory posture. The professional stance is: you build
the controls and the measurements, you document honestly, you escalate what you see, and you refuse to
ship an unmeasured high-stakes system. "The model said it, not me" is not a defense that survives contact
with a regulator or a postmortem.

## What interviews ask here

- Why do LLMs hallucinate, and what's the single best mitigation? Trained for plausible continuation with
  no "I don't know" signal and eval that rewards guessing; grounding/RAG is the biggest lever.
- How do you measure bias in a deployed NLP system? Task-specific counterfactual test set on defined
  axes — not a generic score; promise measured/bounded, not "unbiased."
- What is training-data extraction and what drives memorization? Prompting a model to emit verbatim
  training text; scales with size, duplication, context length — dedup is the main control.
- Prompt injection vs jailbreak? Injection hijacks via untrusted data in context (unsolved because the
  model can't separate instructions from data); jailbreak bypasses safety tuning to elicit banned content.
- Open weight vs open source? Weights under a restrictive custom license (Llama/Gemma) vs OSI freedom to
  use/modify/redistribute (rare, e.g. OLMo with open data+code). Read the license before shipping.
- What does the EU AI Act require of a hiring or credit NLP system? High-risk tier: risk management, data
  governance, logging, human oversight, documentation, conformity assessment.

## Where this shows up on the job

- Building the safety and red-team regression suite that gates every release, and defending its coverage.
- Designing hallucination controls (grounding, citations, abstention, human-in-the-loop) into a
  customer-facing feature, with a measured error rate rather than a promise of zero.
- Writing an honest model card and eval breakdown for a launch review, and clearing the licensing/legal
  gate before building on an "open" model.
- Being the person who escalates a misuse or injection risk — and who refuses to ship an unmeasured
  high-stakes system — which is a seniority signal, not an obstruction.
