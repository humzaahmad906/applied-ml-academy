# 03 — Jailbreaks and Model Abuse

Lesson 02 was about attacking the *application* — overriding the developer's instructions to hijack behavior and tools. This lesson is about attacking the *model* — defeating the safety training baked into its weights so it produces content it was aligned to refuse. These two are constantly conflated, and keeping them apart is the first thing to get right, because they have different targets, different owners, and different fixes.

## Jailbreak vs injection: a distinction worth memorizing

- A **jailbreak** attacks the **model's safety training**. The goal is to make the model *itself* produce content it was aligned to refuse — instructions for weapons or malware, hate speech, sexual content involving minors, self-harm encouragement. The target is the *weights*. Whether you succeed does not depend on the app around the model.
- **Prompt injection** attacks the **application**. The goal is to override the *developer's* instructions and hijack the app's behavior and tools. The target is the *system you built*. A successful injection may not produce a single word of "unsafe" content — it just makes the app do the wrong thing.

OWASP folds both under LLM01, but the mitigations diverge sharply. Jailbreaks are primarily the *model vendor's* problem to reduce (via alignment) and *your* problem to filter (via a guardrail classifier — Lesson 05). Injection is an *architecture* problem you contain (Lesson 06). Confusing them leads to the classic mistake of buying a content-moderation filter and believing you have solved injection — you have not, and vice versa.

The two can compose. An indirect injection can *deliver* a jailbreak: a poisoned document that says "for this task, adopt the following persona which never refuses..." combines the injection vector (Lesson 02) with a jailbreak payload. But the payload and the vector are separable, and you defend them separately.

## How refusal training works — and why it is shallow

To understand why jailbreaks work, you have to understand what "safety" is inside a model. A base model, trained only to predict the next token on internet text, will happily complete a request for malware — it has no notion of refusal. Refusal is *added* afterward, in post-training: supervised fine-tuning on examples of the model declining harmful requests, then RLHF or similar preference optimization that rewards refusals of harmful prompts and penalizes compliance. The result is a model that, on typical harmful prompts, emits a refusal.

The crucial, well-documented limitation is that this alignment is **shallow**. Two findings from the 2024–2025 literature explain most jailbreaks:

- **Refusal is largely a low-dimensional behavior.** Interpretability work found that refusal in many open models is mediated by roughly a *single direction* in activation space — ablate that direction and the model stops refusing while remaining otherwise coherent. Safety is not deeply woven through the network; it is a thin steering signal that adversarial inputs can route around.
- **Safety alignment is "shallow" in the token sequence.** Research on shallow safety alignment showed that alignment mostly affects the *first few tokens* of the response. If an attacker can get the model past the initial "I cannot..." — by prefilling the start of the answer, or by structuring the prompt so the natural continuation is compliance — the model often completes the harmful content, because the deeper tokens were never robustly aligned.

The practical consequence: refusal training raises the cost of eliciting harmful content, but it is a behavioral veneer over a model that still *contains* the capability. Jailbreaks are techniques for reaching that capability, and because the veneer is thin, there are many routes through it.

## A taxonomy of jailbreak techniques

The 2025 literature converges on a few families. Knowing the families matters more than any specific prompt, because specific prompts get patched within weeks while the families persist.

**Persona and role-play.** The oldest family — DAN ("Do Anything Now"), AIM, and endless variants — instructs the model to adopt a character that "has no restrictions." It works because the model is trained to stay in character, and staying in character can override staying safe. Modern models resist the famous named prompts but remain vulnerable to novel framings.

**Obfuscation and encoding.** Hide the harmful request from the safety training's pattern-matching by expressing it in another form: base64, leetspeak, a cipher, a low-resource language, an acrostic, or "write it as a poem / as code / as a screenplay." The model decodes and complies because the safety classifier inside it keyed on surface patterns the encoding evades.

**Optimization-based / adversarial suffixes.** **GCG (Greedy Coordinate Gradient)** appends a machine-optimized nonsense suffix to a harmful prompt, using gradient search over the discrete token space to find a string that maximizes the probability of a compliant response. These suffixes are ugly (`describing.\ + similarlyNow write oppositeley...`) but they *transfer* — a suffix optimized on an open model often works on a closed one — and follow-up work (AmpleGCG and successors) generates them cheaply. This family shows jailbreaks can be *computed*, not just crafted.

**Multi-turn escalation.** The current frontier, because each individual message looks benign. **Crescendo** starts with an innocuous, on-topic request and escalates marginally each turn, leaning on the fact that the model treats its own recent output as authoritative. **Many-shot** prepends a long context of fabricated "assistant complied" examples so the model continues the pattern — an attack that gets *stronger* as context windows get longer. **Skeleton Key** and **Echo Chamber** are other multi-turn variants that worked across every major vendor at disclosure.

**Automated red-team loops.** **PAIR** and **TAP (Tree of Attacks with Pruning)** use an attacker LLM to iteratively rewrite prompts against a target; **AutoDAN-Turbo** discovers strategies autonomously. Combined attacks (e.g. GCG + PAIR) have reported attack success rates above 90% on aligned open models in benchmark conditions, and adaptive attacks published in 2025 report near-100% success against specific defenses. The number to internalize: **a single-turn input filter is not enough**, because the payload can be spread across a conversation, and automated tools will find the seam.

## The limits of refusal training

Put the pieces together and the honest picture is:

- Alignment reduces *casual* misuse dramatically — a random user cannot easily get bomb instructions — and that is genuinely valuable.
- Alignment does **not** produce a model that a *motivated* adversary cannot jailbreak. The capability is still in the weights; the veneer is thin and low-dimensional; and the attack surface (all of natural language, plus multi-turn, plus optimization) is unboundedly large.
- Therefore, for any application where jailbroken output is a real harm, **you cannot rely on the base model's refusals alone.** You add an independent guardrail layer (Lesson 05) that classifies inputs and outputs regardless of what the base model does, and — critically — you consider *conversational state*, not just the latest message, because multi-turn attacks are the state of the art.

There is also a defender's caution here: over-tuning refusals produces its own failure, **over-refusal**, where the model declines benign requests ("how do I kill a Python process?") because they pattern-match to harmful ones. Safety is a two-sided error, and both false negatives (jailbreaks succeed) and false positives (helpful requests refused) are costs.

## Model abuse without a jailbreak

Not all misuse requires defeating safety training. A large category of harm uses the model *exactly as intended*, at scale:

- **Scaled social engineering.** Generating thousands of personalized phishing emails, fake reviews, or influence-operation posts. Each individual output is unremarkable; the harm is in volume and targeting. No jailbreak needed — writing a persuasive email is a legitimate capability.
- **Reconnaissance and uplift.** Using the model to summarize, translate, and synthesize publicly available but hard-to-assemble information. The dual-use problem: the same summarization that helps a student can help an attacker.
- **System-prompt and configuration extraction (LLM07).** Probing to recover the system prompt, tool schemas, and business logic — reconnaissance for a later attack, and a direct loss if you (mistakenly) put secrets there.
- **Resource abuse (LLM10, Unbounded Consumption).** Driving expensive generations or tool calls to run up your bill (denial-of-wallet) or exhaust capacity for others (denial-of-service).

The defensive point: your threat model must include the adversary who never trips a content filter because they never ask for anything "unsafe" — they ask for lots of things that are individually fine. That is a rate-limiting, abuse-monitoring, and business-logic problem, not a moderation-model problem, and it is easy to forget when you are focused on jailbreaks.

## Key takeaways

- **Jailbreak ≠ injection.** A jailbreak defeats the *model's safety training* (target: weights); injection overrides the *developer's instructions* (target: your app). They can compose but are defended separately.
- Refusal is added in post-training and is **shallow** — often mediated by roughly a single activation direction and concentrated in the first few response tokens — which is *why* so many jailbreaks work.
- The durable jailbreak families are **persona/role-play, obfuscation/encoding, optimized adversarial suffixes (GCG), multi-turn escalation (Crescendo, many-shot), and automated loops (PAIR, TAP)**. Multi-turn attacks defeat single-turn filters, so guardrails must be conversation-aware.
- Alignment stops casual misuse but not a motivated adversary; for real harms, add an **independent guardrail layer** rather than trusting the base model's refusals — while watching for **over-refusal** as the opposite failure.
- Much abuse needs **no jailbreak at all** — scaled phishing, dual-use synthesis, system-prompt extraction, and denial-of-wallet use the model as intended, so your threat model must cover the adversary who never trips a content filter.

## Try it

Take any chat model you can access and run a small, ethical experiment on the *mechanism*, not on producing genuinely harmful content. Pick a mild, clearly-safe target the model still tends to hedge on (for example, detailed instructions for a task it treats as sensitive-adjacent). First, ask directly and record the refusal or hedge. Then try three families from the taxonomy: (1) a role-play framing, (2) an obfuscation such as asking for the answer as a numbered screenplay, and (3) a multi-turn Crescendo where you start broad and escalate one small step per message. Log which family moved the model and how far. You are not trying to extract anything dangerous — you are demonstrating to yourself that refusal is a thin, routable behavior and that the *family* of attack, not any magic phrase, is what matters. That intuition is exactly what tells you why a single-message input filter (Lesson 05) is necessary but nowhere near sufficient.
