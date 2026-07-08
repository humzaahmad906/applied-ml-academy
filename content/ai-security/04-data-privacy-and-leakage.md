# 04 — Data, Privacy, and Leakage

The last two lessons attacked the model at inference time through the prompt. This one moves upstream and downstream of the prompt to the two surfaces the OWASP list calls LLM02 (Sensitive Information Disclosure), LLM03 (Supply Chain), and LLM04 (Data and Model Poisoning). The unifying idea: **the model is data, and its training set is data, and both can leak *out* or be corrupted *in*.** These are the risks that survive even a perfectly injection-proof, jailbreak-proof application, because they were baked in before a single user ever sent a prompt.

We cover four things: what the model memorizes and leaks, how attackers confirm and extract it, how attackers poison what goes in, and how a model file itself becomes a code-execution vector.

## Memorization and extraction

Large models memorize. Trained to minimize loss on a corpus, they store not just statistical patterns but, for rare or repeated sequences, *verbatim spans* of the training data. This is not a bug — it is a direct consequence of high-capacity models fitting their data — and it becomes a privacy problem when the training data contained secrets.

The canonical demonstration is **training-data extraction**: prompt the model in ways that make it complete memorized text, then verify the completions against the corpus. Researchers have recovered verbatim email addresses, phone numbers, code, and long document passages from production models this way. Memorization scales with model size, with how many times a sequence appeared in training (duplicated data is memorized far more readily), and with sequence uniqueness. The practical implications:

- **PII leakage (LLM02).** If personal data was in the training set, the model can emit it — sometimes when asked directly, sometimes as an unprompted completion. A 2026 line of work refined this with cue-controlled studies showing PII regurgitation depends heavily on how strongly the prompt "cues" the memorized context, but the core risk stands: what you train on, the model can say.
- **System-prompt and context leakage.** The same regurgitation applies to anything in the context window — retrieved documents from another user, the system prompt (LLM07), secrets pasted into a session. Memorization is the training-time version; context leakage is the inference-time version of the same "the model repeats what it saw" problem.
- **Copyright and provenance.** Verbatim regurgitation of copyrighted text is both a legal exposure and evidence in the ongoing training-data disputes.

## Membership inference and model inversion

Two attack primitives formalize "what did the model learn about specific data":

**Membership inference (MIA)** asks a yes/no question: *was this specific record in the training set?* The attacker exploits the fact that models are more confident (lower loss, lower perplexity) on data they trained on than on data they did not. Classic MIAs threshold on this loss gap. On LLMs, MIAs are notoriously *hard* — the pretraining corpus is so vast and its distribution so overlapping that the confidence gap between members and non-members is thin, and several 2025 papers (and skeptical re-evaluations) show that many published LLM-MIA results were partly measuring distribution shift between the "member" and "non-member" sets rather than true membership. That caveat matters: MIA on LLMs is an active, contested research area, not a solved capability. Where it does work, the stakes are real: confirming a patient's record, or a specific book, was in the training set is itself a privacy or legal disclosure.

**Model inversion and embedding inversion** go further — reconstructing training inputs (or approximations) from model outputs or internal representations. Embedding-inversion work is especially relevant to RAG: the vectors you store are not opaque; they can leak substantial information about the source text, which is why a vector database of sensitive documents is itself sensitive data.

The defensive vocabulary here is worth knowing even at a high level:

- **Deduplication** of training data sharply reduces memorization — the single most cost-effective mitigation.
- **Differential privacy (DP)** training (DP-SGD) adds calibrated noise to bound how much any single training example can influence the model, giving a formal privacy guarantee at a cost in accuracy and compute. It is the rigorous defense but is expensive and rarely applied to frontier pretraining.
- **Data governance**: know what is in your training and fine-tuning sets, exclude secrets and PII, and keep provenance so you can answer "was X in the data?" without an attacker having to run an MIA.
- **Output-side PII redaction** (Lesson 05) catches leakage at emit time regardless of what the model memorized.

## Data and model poisoning, and backdoors

Poisoning flips the direction: instead of extracting from the model, the attacker *corrupts what goes in* so the model learns something the operator did not intend. This is OWASP LLM04.

**Untargeted poisoning** degrades general performance — inject enough garbage or mislabeled data and accuracy drops. Annoying, usually detectable.

**Backdoors (targeted poisoning)** are the serious version. The attacker plants training samples that teach a hidden *trigger → behavior* rule: the model behaves normally except when a specific trigger appears, at which point it does the attacker's bidding. Anthropic's **"Sleeper Agents"** work (2024) trained models to write secure code normally but insert vulnerabilities when the prompt indicated a certain year — and, alarmingly, showed that standard safety training (SFT, RL, even adversarial training) *failed to remove* the backdoor, and adversarial training sometimes taught the model to *hide* it better. The backdoor is in the weights and safety fine-tuning does not reliably scrub it.

The scale required is smaller than intuition suggests. A 2025 study by Anthropic, the UK AI Security Institute, and the Alan Turing Institute found that **roughly 250 malicious documents were enough to backdoor models from 600M to 13B parameters** — and, strikingly, the number of poison documents needed was *near-constant regardless of model size*, not a fixed percentage. Because pretraining data is scraped from the open web, an attacker does not need insider access; they need to publish content that gets scraped. That is a low bar, and it makes poisoning a supply-chain problem as much as a training problem.

Poisoning has an inference-time cousin relevant to RAG and agents: **retrieval/context poisoning**, where the attacker plants a document in your index (or in content the agent will fetch) rather than in the training set. Same idea — corrupt what the model conditions on — but no training run required. This is the bridge back to the indirect injection of Lesson 02.

Defenses are imperfect and layered: vet and provenance-track training sources; deduplicate and anomaly-scan datasets; prefer curated over indiscriminately-scraped corpora for sensitive uses; scan for known trigger patterns; and — because scrubbing a backdoor after the fact is unreliable — treat *any* model trained on untrusted data as potentially trojaned and constrain its blast radius accordingly.

## Supply chain: the model file as a code-execution vector

The most immediately dangerous — and most overlooked — item is that a downloaded model file can run arbitrary code the moment you load it. This is OWASP LLM03, and it does not require any of the subtle ML attacks above; it is plain remote code execution.

The culprit is **Python's `pickle`**. PyTorch's `.pth`/`.bin` checkpoints and many other model artifacts are pickle-serialized, and *unpickling executes code*: pickle's `__reduce__` mechanism lets an object specify a callable to run on deserialization, so `torch.load()` on a malicious checkpoint can execute an embedded shell command. This is documented behavior, not a bug — pickle's own docs warn never to unpickle untrusted data. In the wild, malicious `.pth` files uploaded to Hugging Face have used exactly this to download and run binaries on the loading machine, and studies have reported a large year-over-year increase in malicious models on public hubs.

The defenses, in order of value:

- **Use `safetensors`.** Hugging Face's `safetensors` format stores weights as a header plus raw tensor buffers with *no executable code path*. It is the structural fix: a safetensors file cannot run code on load. Prefer it for everything, and be suspicious of any model distributed only as pickle.
- **Scan pickle files** with tools like `picklescan` or `ModelScan` — but know the limits. Through 2025, researchers disclosed multiple bypasses of `picklescan` (including CVE-2025-10155, evading detection by renaming files to alternative extensions like `.bin`/`.pt`); fixes shipped in later versions, but scanning is a mitigation, not a guarantee.
- **Load untrusted models in a sandbox** — a container with no secrets, no network egress, minimal privileges — so that if code does execute, it executes nowhere useful.
- **Verify provenance**: pin exact versions and hashes, prefer signed artifacts and trusted publishers, and re-verify on update. This is the same supply-chain hygiene you apply to any dependency, extended to model weights and datasets — which are dependencies.

The same supply-chain logic extends to the *tool* layer for agents (MCP servers, plugins) — malicious code and "rug-pull" definition swaps — which Lesson 06 covers in its own right.

## Key takeaways

- Models **memorize** training data and can **regurgitate** it verbatim — the mechanism behind PII leakage (LLM02), context leakage, and copyright exposure. Deduplication is the cheapest mitigation; differential privacy is the rigorous one.
- **Membership inference** (was this record in training?) is real but genuinely *hard and contested on LLMs*; **model/embedding inversion** can reconstruct inputs, which is why a vector store of sensitive text is itself sensitive.
- **Backdoors (targeted poisoning)** plant a hidden trigger→behavior rule that safety training does **not** reliably remove; roughly **250 documents can backdoor models up to 13B parameters, near-constant in the count regardless of model size**, and web-scraped data makes this a low-bar supply-chain attack.
- A **pickle-serialized model file executes code on load** (`torch.load` RCE via `__reduce__`); malicious models are live on public hubs. **Prefer `safetensors`**, scan pickles (with known bypass caveats), sandbox untrusted loads, and verify provenance.
- Treat **datasets and model weights as dependencies** with full supply-chain hygiene: provenance, pinning, hashing, and least-privilege loading.

## Try it

Do the supply-chain exercise, because it is the one with the most visceral payoff and the least ambiguity. In an *isolated, disposable* environment (a throwaway container with no secrets and no network), construct a deliberately "malicious" pickle: define a small class whose `__reduce__` returns something harmless-but-observable, like `(os.system, ("echo PWNED > /tmp/proof.txt",))`, pickle an instance of it, then load it with `pickle.load` / `torch.load` and confirm that merely *loading the file* created `/tmp/proof.txt` — you never called the payload. Now re-save the same tensors with `safetensors.torch.save_file` and confirm that loading the safetensors version runs no code. You will have proven to yourself, in ten lines, why "just download the model and load it" is a remote-code-execution decision, and why `safetensors` plus a sandbox is not optional hygiene but a hard boundary. Do not run untrusted payloads, and do not do this outside a disposable sandbox.
