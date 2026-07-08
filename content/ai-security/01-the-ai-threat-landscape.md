# 01 — The AI Threat Landscape

Every course you have taken so far asked one question of a model: *is it accurate?* This course asks a different one: *what happens when someone is actively trying to make it misbehave?* Those are not the same question, and a system that aces the first can fail catastrophically at the second. A fraud model with 0.99 AUC still leaks its training data under a membership-inference probe. A helpful RAG assistant still exfiltrates your inbox when it reads a poisoned web page. Security is an adversarial property, and adversarial properties are not measured on the validation set you optimized against.

This opening lesson builds the mental map for the rest of the course: *why* machine-learning systems — and large language models in particular — are uniquely exploitable, the industry-standard vocabulary for their risks (the OWASP LLM Top 10), and the attack-surface map that organizes the six lessons that follow. A sibling chapter in the GenAI course, *Agent Safety and Guardrails*, covers the agent-specific slice of this material; this course is broader (the whole ML lifecycle — data, model, supply chain, privacy, governance) and deeper. Read that chapter as the fast tour; read this as the full survey.

## Why ML systems are a new kind of attack surface

Classical software security rests on a boundary you can reason about. Code lives in one place, data in another; a SQL query has a template and parameters; a compiler enforces types; you escape untrusted input against a grammar. Attacks exploit *bugs* — a buffer you forgot to bound, an input you forgot to sanitize — and the fix is to patch the bug.

Machine-learning systems break three of these assumptions at once, and each break is structural rather than a bug you can patch.

**The behavior is learned, not written.** You did not author the decision boundary; gradient descent did, from data you probably did not fully inspect. That means the "specification" of what the model does is implicit and enormous, and an attacker can find corners of it you never tested. Adversarial examples — inputs perturbed imperceptibly to flip a classifier — are the canonical demonstration: the model is behaving exactly as trained, and the training simply never covered the adversary's input distribution.

**Instructions and data share one channel.** This is the defining vulnerability of LLM applications and it is worth stating precisely. A CPU separates code memory from data memory; SQL separates the query from its parameters. An LLM has neither. The system prompt, the user's message, a retrieved document, a tool's output, and a web page the model just fetched all arrive as *one flat token stream*, and the model was trained to follow instructions wherever it finds them. There is no reliable, model-level boundary between "this is my task" and "this is content I am supposed to be processing." You cannot escape or sanitize your way out, because there is no grammar to escape against — `ignore previous instructions` is ordinary English, and so is every paraphrase of it. Lesson 02 is entirely about the consequences.

**The model itself is data that can be poisoned or stolen.** The weights are a file. That file can carry a backdoor planted during training (Lesson 04), can be probed to reconstruct the private data it memorized (Lesson 04), and — when distributed as a Python pickle — can execute arbitrary code the moment you load it (Lesson 04 again; the supply chain is that important). The model is simultaneously the product, an attack surface, and a delivery vehicle.

Add to this that ML systems are increasingly *agentic* — placed in a loop with tools that read email, run code, query databases, and call APIs — and the blast radius of a manipulated model grows from "says something wrong" to "takes a harmful action in the world." That escalation is the throughline of Lesson 06.

## The OWASP LLM Top 10 (2025)

The shared vocabulary for these risks is the **OWASP Top 10 for Large Language Model Applications**. The 2025 edition (v2.0, published November 2024 by the OWASP GenAI Security Project) is the current reference list, and you should know it well enough to place any incident on it:

1. **LLM01 — Prompt Injection.** Getting the model to follow instructions its operator never intended. Holds the top spot two editions running. Lesson 02.
2. **LLM02 — Sensitive Information Disclosure.** The model reveals PII, secrets, or proprietary data — from training memorization, retrieved context, or the system prompt. Lesson 04.
3. **LLM03 — Supply Chain Vulnerabilities.** Compromised models, datasets, dependencies, or tool servers. Lessons 04 and 06.
4. **LLM04 — Data and Model Poisoning.** Corrupting training or fine-tuning data to plant backdoors or degrade behavior. Lesson 04.
5. **LLM05 — Improper Output Handling.** Passing model output into a downstream sink (SQL, HTML, shell) without treating it as untrusted. Lessons 05 and 06.
6. **LLM06 — Excessive Agency.** Too much functionality, too many permissions, too much autonomy — the thing that turns a manipulated model into real damage. Lesson 06.
7. **LLM07 — System Prompt Leakage.** Relying on the system prompt to stay secret when it will not, and putting secrets in it. Lesson 03.
8. **LLM08 — Vector and Embedding Weaknesses.** Attacks against the RAG layer — poisoned documents in the index, embedding inversion, cross-tenant leakage. Lessons 02 and 04.
9. **LLM09 — Misinformation.** Confident, wrong output — hallucination and over-reliance — treated as a security-relevant failure. Lesson 05.
10. **LLM10 — Unbounded Consumption.** Denial-of-wallet and denial-of-service through uncapped token, compute, or tool usage. Lesson 06.

Three entries are new to the 2025 list — System Prompt Leakage (LLM07), Vector and Embedding Weaknesses (LLM08), and Unbounded Consumption (LLM10, expanded from the older "Model Denial of Service"). That churn tells you something: the field is young, the threat model is still being written, and any list you memorize will move. Treat the Top 10 as a checklist and a shared language, not as a complete or permanent taxonomy.

## The attack-surface map

The cleanest way to organize LLM risk — and the structure of this course — is to walk the data path through the system and ask what an attacker controls at each stage. Five surfaces:

**1. The data.** Everything upstream of the weights: pretraining corpora, fine-tuning sets, RAG indexes, and the labels behind them. Attacks here are *poisoning* (inject crafted samples so the model learns a backdoor or a bias) and, on the flip side, *leakage* (the model memorizes and later regurgitates private data it was trained on). Because training data is scraped at web scale, the attacker often does not need insider access — they just need to publish content that gets scraped. Lesson 04.

**2. The model.** The weights and the file that ships them. Surfaces: *extraction* (steal the model or its behavior by querying it), *inversion / membership inference* (recover training data or confirm a record was in the training set), *backdoors* (trigger-activated misbehavior baked into the weights), and *supply chain* (a malicious model file that runs code on load, or a fine-tune of a trojaned base). Lesson 04.

**3. The prompt.** The input channel at inference time. This is where *prompt injection* (LLM01) lives — direct, from the user, and indirect, from content the model ingests — and where *jailbreaks* (LLM01 again, but a distinct problem) attack the model's safety training. This is the largest surface in practice and gets two lessons: 02 for injection, 03 for jailbreaks.

**4. The output.** What the model emits and where it flows. Surfaces: *improper output handling* (LLM05 — model output executed as SQL/HTML/shell), *sensitive disclosure* (LLM02), and *misinformation* (LLM09). The defensive counterpart — filtering, moderation, structured validation — is Lesson 05.

**5. The agent and its tools.** The loop, the tool permissions, the connectors (including MCP servers). Surfaces: *excessive agency* (LLM06), *tool poisoning and confused-deputy* attacks, and *unbounded consumption* (LLM10). This is where a text-only failure becomes an action in the world. Lesson 06.

Notice that a single real attack usually chains across surfaces: a poisoned document in the RAG index (data + prompt) delivers an injection (prompt) that hijacks a tool call (agent) to exfiltrate private data through an external channel (output). Security thinking means following that chain end to end, not defending one surface and declaring victory.

## The mindset: assume compromise, cap the blast radius

The single most important shift this course asks you to make is from *detection* to *containment*. Because the core vulnerabilities are structural — you cannot reliably detect a malicious instruction hidden in ordinary text, and you cannot perfectly filter a jailbreak — the reliable engineering move is to assume the model *will* be manipulated and design so that a manipulated model still cannot do serious harm. Least privilege, isolation architectures, human confirmation on irreversible actions, and provenance tracking are worth more than any single classifier, because they hold even when the classifier misses. Detection reduces volume; capability limits reduce damage. Keep that ordering in mind through everything that follows.

Two more framing habits. First, **governance is part of the system, not paperwork bolted on after.** The NIST AI Risk Management Framework and the EU AI Act (Lesson 07) increasingly make specific security and red-teaming practices legal obligations, not nice-to-haves — for general-purpose AI providers in the EU, obligations began applying in August 2025. Second, **red-teaming is the eval.** Accuracy on a held-out set does not measure adversarial robustness; you have to attack your own system on purpose, ideally with tooling built for it (PyRIT, garak — Lesson 07). If you have not tried to break it, you do not know whether it is broken.

## Key takeaways

- ML systems are a new attack surface because behavior is *learned* not written, instructions and data share *one channel*, and the model itself is *data* that can be poisoned, stolen, or weaponized on load. These are structural properties, not patchable bugs.
- The **OWASP LLM Top 10 (2025)** is the shared vocabulary: LLM01 Prompt Injection through LLM10 Unbounded Consumption. Know it, but treat it as a moving checklist, not a permanent truth — three entries are new this edition.
- The attack-surface map has five stages — **data, model, prompt, output, agent/tools** — and real attacks chain across them.
- The governing mindset is **assume compromise, cap the blast radius**: containment (least privilege, isolation, human gates) beats detection, because the core vulnerabilities cannot be reliably detected.
- **Governance and red-teaming are part of engineering**, not afterthoughts — increasingly they are legal obligations, and adversarial testing is the only eval that measures security.

## Try it

Pick one AI system you have actually built or used closely — a RAG chatbot, a classifier behind an API, a coding assistant, an internal agent. Draw its data path on paper as a pipeline: where does data enter, where do the weights come from, what reaches the prompt, where does the output go, what tools (if any) can it call? Now annotate each of the five surfaces with (a) what an attacker could influence there and (b) which OWASP LLM Top 10 entry it maps to. You will almost certainly find at least one surface you have never tested adversarially — most commonly the RAG index or the output-handling step. That gap is where the rest of this course is aimed, and finishing this exercise is the difference between "my model is accurate" and "I know where my system can be attacked."
