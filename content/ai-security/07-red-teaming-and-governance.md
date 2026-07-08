# 07 — Red-Teaming and Governance

You have a threat model (Lesson 01), you understand the attacks (02–04), and you can layer filters and architecture against them (05–06). Two questions remain, and they are the ones that decide whether a system actually ships safely. First: *how do you know your defenses work?* Accuracy on a held-out set does not measure adversarial robustness — you have to attack your own system on purpose. Second: *what are you obligated to do?* Security is increasingly a legal and regulatory requirement, not just good practice. This final lesson covers systematic red-teaming, safety evaluation, and the governance frameworks (NIST AI RMF, EU AI Act) that turn all of it into a shipping decision.

## Red-teaming is the eval

The central mindset shift: for security, **the eval is an attack, not a benchmark**. A model can score 0.95 on your quality metric and leak its system prompt to the first Crescendo attempt. If you have not tried to break your system, you do not know whether it is broken — you only know it behaves on the inputs you happened to test. Red-teaming closes that gap by making the adversary's job your job, before the adversary gets there.

Two modes, and you need both:

- **Manual red-teaming.** Skilled humans probe the system creatively — novel jailbreaks, domain-specific abuse, chained attacks across surfaces. Humans find the *classes* of vulnerability that automated tools have not been told to look for. Expensive, slow, irreplaceable for discovery.
- **Automated red-teaming.** Tools generate and run thousands of attack variants, cover known families exhaustively, and — critically — run in CI so a regression is caught the day it is introduced. Cheap, fast, reproducible, but only finds what its probes know about.

The productive loop is manual discovery feeding automated regression: a human finds a new attack, you encode it as an automated probe, and it guards that hole forever after.

## The tooling (2025–2026)

Three open frameworks dominate, and a fourth is the emerging standard for evals:

- **garak** (NVIDIA) — an open-source LLM *vulnerability scanner*, the "nmap for LLMs." A plugin architecture with dozens of probe modules covering prompt injection, jailbreaks, encoding attacks, data leakage, toxicity, and more. Point it at a model endpoint and it runs a broad battery and reports what fired. Best for **initial, broad coverage scans**.
- **PyRIT** (Microsoft, the Python Risk Identification Toolkit) — an *orchestration* framework for building automated, **multi-turn and multi-modal** attack chains, where an adversarial LLM iteratively refines its attack against the target. It implements techniques like Crescendo and TAP. Best for **novel attack chains, multi-turn conversations, and generating adversarial datasets** — the deeper, adaptive testing.
- **Promptfoo** — red-teaming plus eval with strong **CI/CD integration**; covers a large catalog of vulnerability types and maps findings to the **OWASP LLM Top 10**. Best for **regression testing in your pipeline**.
- **Inspect AI** (UK AI Safety Institute) — a general evaluation framework increasingly used for safety and dangerous-capability evals; the reference for rigorous, reproducible eval harnesses.

For *agents* specifically, **AgentDojo** is the benchmark of record for prompt-injection-against-tool-use (it is where CaMeL's ~67% attack-neutralization figure comes from), and suites like **deepteam** package OWASP-aligned agent attacks. The practical recipe: **garak for the broad first pass, PyRIT for adaptive depth, Promptfoo in CI for regression, Inspect/AgentDojo for rigorous benchmarking.**

## Evaluating for safety

Red-teaming produces data; you need to turn it into a metric you can track. The safety-eval discipline borrows directly from eval-driven development:

- **Build a frozen adversarial eval set.** Curate attacks across every surface (injection, jailbreak, PII extraction, tool abuse) into a held-out set. Freeze it — adding samples is fine, but rebalancing or removing invalidates all prior numbers, exactly as with any eval set. A moving target measures nothing.
- **Measure Attack Success Rate (ASR)** — the fraction of attacks that achieve their goal — per attack family, not just in aggregate, so you can see *which* defense is weak. Track it over time and across model/prompt/config changes. A model upgrade that improves quality can *regress* safety; only a tracked ASR catches that.
- **Include the benign set too.** Measure over-refusal (false positives) alongside ASR (false negatives), because a system that refuses everything has ASR 0 and is useless. Safety is the two-sided error from Lesson 05.
- **Gate releases on it.** ASR on the frozen set becomes a release criterion, and the automated suite runs in CI so regressions block the merge. This is how "we red-teamed it once" becomes "we cannot ship a regression."

## Governance: NIST AI RMF

Governance frameworks give you a defensible structure for *all* of the above and are increasingly what auditors, customers, and regulators ask for.

The **NIST AI Risk Management Framework (AI RMF 1.0)** is the voluntary US standard, organized around four functions: **Govern** (culture, policies, accountability — the function that cuts across the others), **Map** (context and risk identification), **Measure** (assess and track risks — where your safety evals live), and **Manage** (prioritize and respond). Its companion, the **Generative AI Profile (NIST AI 600-1, published July 2024)**, is the piece to know for LLM work: it enumerates **12 risk categories** unique to or amplified by generative AI — including prompt-based manipulation, data memorization/leakage, harmful/unsafe content, hallucination, IP leakage, and CBRN/dangerous-capability uplift — and maps suggested actions back to the four core functions. It is voluntary and sector-agnostic, which makes it the natural backbone for an internal AI risk program even where no law requires one.

## Governance: the EU AI Act

The **EU AI Act** is the first comprehensive, binding AI law, and its reach is extraterritorial — it applies to providers placing AI on the EU market regardless of where they are based, so it affects US teams too. It is **risk-tiered**:

- **Prohibited practices** (unacceptable-risk systems — e.g. social scoring, certain biometric categorization) — banned, with these provisions applying from **February 2025**.
- **High-risk systems** (AI in hiring, credit, medical devices, critical infrastructure, and similar) carry the heaviest obligations: risk management, data governance, logging, human oversight, robustness and cybersecurity, and conformity assessment.
- **General-Purpose AI (GPAI) models** have their own obligations — technical documentation, training-data summaries, copyright policy, and for models posing *systemic risk*, additional duties including **adversarial testing / red-teaming, incident reporting, and cybersecurity protections.**
- **Limited-risk** systems (e.g. chatbots) carry transparency duties (disclose that users are interacting with AI).

The timeline to hold in mind: the Act entered into force in 2024; **prohibited-practice rules applied from February 2025**; **GPAI obligations began applying on 2 August 2025** (with a voluntary GPAI Code of Practice published to help providers comply); and the Act becomes **fully applicable on 2 August 2026**, with high-risk obligations and full enforcement powers phasing in around and after that date (some high-risk categories extend to 2027). *Verify current dates before relying on them* — the timeline has been subject to active political debate and possible adjustment, and regulatory details move faster than any lesson can.

Beyond these two, **ISO/IEC 42001** (AI management systems) is the emerging certifiable standard many organizations pursue to demonstrate governance maturity. The through-line across all of them: they demand exactly the practices this course teaches — documented risk assessment, adversarial testing, human oversight, logging, and incident response — which means doing the engineering well *is* most of the compliance.

## A shipping checklist

Consolidating the whole course into the walk-through before an AI system with real capability goes to production:

- **Threat model mapped.** Every one of the five surfaces (data, model, prompt, output, agent/tools) has been assessed against the OWASP LLM Top 10. (Lesson 01)
- **Injection contained, not just filtered.** No single session holds the full lethal trifecta; where it must, an isolation architecture (plan-then-execute / dual-LLM / CaMeL) sits between untrusted content and privileged action. (Lessons 02, 06)
- **Supply chain verified.** Models loaded from `safetensors` or scanned + sandboxed; datasets and weights pinned, hashed, and provenance-tracked; MCP tool definitions pinned + hashed and re-approved on change. (Lesson 04, 06)
- **Privacy addressed.** Training/fine-tuning data governed for PII; output-side PII redaction; awareness of memorization and membership-inference exposure. (Lesson 04)
- **Guardrails on both directions.** Input and output classifiers (Llama Guard / moderation), conversation-aware, with over-refusal measured — deployed as filters, not guarantees. (Lesson 05)
- **Least privilege + human gates.** Every tool minimally scoped, rate-limited, budget-capped; irreversible/outbound actions gated on out-of-band human confirmation with resolved-argument previews. (Lesson 06)
- **Output handling safe.** No model output into SQL/HTML/shell/`eval` unsanitized; no auto-rendering of model-supplied URLs. (Lesson 05)
- **Red-teamed and tracked.** A frozen adversarial eval set with per-family ASR, run in CI as a release gate, plus periodic manual red-teaming (garak → PyRIT → Promptfoo). (this lesson)
- **Observability + kill switch.** Every tool call logged with provenance; the ability to halt instantly; an incident-response path. (Lessons 06, this lesson)
- **Governance documented.** Risk assessment against NIST AI RMF / GenAI Profile; EU AI Act tier identified and obligations met if in scope. (this lesson)

The mindset that ties the whole course together, one last time: **assume the model will be compromised, and design so that a compromised model still cannot do serious harm.** Everything else is detail.

## Key takeaways

- For security, **the eval is an attack.** Use **manual red-teaming** for discovery and **automated** red-teaming for coverage and CI regression — manual finds new classes, automated guards them forever.
- Tooling: **garak** (broad vulnerability scan), **PyRIT** (adaptive multi-turn/multimodal chains), **Promptfoo** (CI regression, OWASP-mapped), **Inspect AI** (rigorous evals), **AgentDojo** (agent injection benchmark).
- Turn red-teaming into a tracked metric: a **frozen adversarial eval set**, **per-family Attack Success Rate**, **over-refusal measured alongside**, and ASR as a **CI release gate**.
- **NIST AI RMF** (Govern/Map/Measure/Manage) + the **GenAI Profile (AI 600-1, 12 risk categories)** is the voluntary backbone; the **EU AI Act** is binding and risk-tiered — prohibited practices from **Feb 2025**, **GPAI obligations from Aug 2025**, full applicability **Aug 2026** — with systemic-risk GPAI required to do adversarial testing. *Verify current dates.*
- Doing the engineering well *is* most of the compliance; the frameworks demand exactly this course's practices — documented risk assessment, adversarial testing, human oversight, logging, and incident response.

## Try it

Turn your favorite system from earlier lessons into a governed, red-teamed one. First, install **garak** and run it against your model endpoint for a broad first-pass scan; read the report and note which probe families fired. Then take the two or three most interesting hits and reproduce them as a small **frozen adversarial eval set** (attacks + expected-safe behavior), add a matching benign set, and write a script that computes per-family Attack Success Rate and over-refusal rate. Wire that script into your test suite so it runs on every change and fails the build if ASR rises above a threshold you pick. Finally, spend fifteen minutes as a manual red-teamer trying to beat your own defenses in ways garak did not — then encode whatever you find as a new case in the frozen set. You will end with the two artifacts that separate a demo from a shippable system: a red-team result you can *act on*, and a regression gate that keeps you from backsliding — which is exactly what NIST's "Measure and Manage" and the EU AI Act's adversarial-testing obligation are asking you to produce.
