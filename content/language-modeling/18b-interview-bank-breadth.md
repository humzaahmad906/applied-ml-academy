# 18 — Interview Bank IV: Breadth & Rapid-Fire — Part 2 of 4: Scaling laws, data & evaluation

This is part 2 of 4 of the Interview Bank IV: Breadth & Rapid-Fire lesson. Here we cover scaling-law
fitting methodology, the Common Crawl data pipeline (dedup, quality filtering, mixing, contamination),
and evaluation protocol (perplexity, MMLU scoring, base-vs-chat evals, LLM-as-judge, Chatbot Arena,
benchmark saturation).

---

## Part D — Scaling laws (fitting, not lore)

**D1. Name the three ways to fit a scaling law and when you would use each.**

There are three. **(1) Minimum over training curves.** Train several model sizes, each on
many token counts, record loss along each curve; for every compute level `C`, take the *minimum* loss
across all runs hitting that `C`, and the `(N, D)` achieving it is compute-optimal at that `C`. Trace
the minima and fit `N_opt ∝ C^a`. Cheap because it reuses points along curves you trained anyway.
**(2) IsoFLOP profiles.** Pick several fixed compute budgets; within each, train several model sizes
`N`, giving each the matching `D = C/(6N)` so they all consume the *same* compute; plot final loss vs
`N` — it is a U-shaped bowl whose bottom is the optimal `N` for that budget (fit a parabola in
log-space, don't just take the lowest sampled point). Collect the bowl-bottoms and fit `N_opt ∝ C^a`.
This is the cleanest and the one the scaling-laws build centers on. **(3) Parametric fit.** Fit
$L(N,D) = E + A/N^\alpha + B/D^\beta$ to *all* your `(N, D, loss)` points at once (nonlinear regression,
Huber loss on log-loss for robustness), then compute the optimum analytically via `a = β/(α+β)`. Most
data-efficient, most sensitive to fitting choices. The discipline: all three must *agree* before you
trust an extrapolation.
*Probes: that you fit a scaling law empirically three ways, not derive it from theory.*

**D2. Kaplan said grow the model; Chinchilla said grow both. Why did they disagree?**

Both fit power laws; they got different *allocation* answers because of methodology. Kaplan (2020)
concluded `N ∝ C^0.73` — spend most of a new budget on a bigger model, barely grow the data — which
shaped GPT-3 (175B on only ~300B tokens, under 2 tokens/param). Chinchilla (2022) redid it and got
`N ∝ C^~0.5` — parameters and tokens scale *equally*, ~20 tokens/param. The discrepancy was that
Kaplan used a largely *fixed* learning-rate schedule and step count across model sizes, which
systematically *under-trained* the smaller models and biased the fit toward "bigger is better."
Chinchilla varied token counts properly and tuned the LR schedule to each run's length. The verdict
was stark: their 70B Chinchilla, trained compute-optimally on 1.4T tokens, *beat* the 280B Gopher
trained on 300B — a 4× smaller model won because it saw the right amount of data. GPT-3-era models
were over-parameterized and under-trained.
*Probes: that the "law" is only as good as the training discipline behind the fit.*

**D3. What do the exponents in `L(N,D) = E + A/N^α + B/D^β` mean, and why do both terms decay at comparable rates?**

`E` is the irreducible loss — the entropy of the data itself, the floor no model can beat (Chinchilla
fit ≈ 1.69 nats/token). `A/N^α` is the penalty for a model too small to represent the function; `B/D^β`
is the penalty for too little data. `α` and `β` are the power-law exponents governing how fast each
penalty shrinks as you grow parameters or tokens (Chinchilla fit α ≈ 0.34, β ≈ 0.28). They come out
*comparable* — neither dominates — which is precisely why the compute-optimal allocation grows both
together: substitute `D = C/(6N)` and minimize, and you get `N_opt ∝ C^(β/(α+β)) ≈ √C` and
`D_opt ∝ √C`, i.e. roughly 20 tokens/param. If one exponent were far larger, you would pour the budget
into whichever resource shrinks loss faster. (Note the Chinchilla α,β differ from the Kaplan
single-variable exponents α≈0.076, β≈0.095 — the two papers parameterize differently, part of why they
disagreed. Don't overread the exact constants; the *structure* is the durable insight.)
*Probes: reading the parametric form as "floor plus two decaying penalties" and why √C falls out.*

**D4. What are the ways extrapolating a scaling law loses you money?**

Four classic pitfalls. **Over-reaching the measured range:** a fit is trustworthy within and modestly
beyond where you measured; predicting 100× past your largest run is faith — hold out your largest
affordable point and check the fit predicts it before betting further. **A changing recipe:** the law
is fit for a *fixed* architecture, data distribution, optimizer, and schedule; change the data mix,
tokenizer, or LR schedule and you are on a different curve, so you cannot mix runs with different data
into one fit (better data shifts the whole line *down*, which is exactly why data work pays). **Under-
tuned small runs:** the Kaplan mistake in miniature — if your small models use a schedule tuned for
large ones, the small end is biased and the extrapolation tilts, so tune per run length. **Reading the
wrong bowl-bottom:** IsoFLOP bowls are flat near the minimum, so the naive lowest *sampled* point is
noisy — fit the parabola.
*Probes: the specific failure modes, especially "different data = different curve."*

**D5. You have a fixed FLOPs budget for a real run. Walk me through planning it.**

State the budget as `C` FLOPs. If I have scaling-law fits for this recipe, read `N_opt ∝ C^a` and
`D_opt = C/(6 N_opt)` straight off, sanity-checking that all three fitting methods agreed and that the
target `C` is not wildly beyond my measured range — if it is, I run a hold-out point first. Then I ask
the question the Chinchilla optimum ignores: *is this compute-optimal or deployment-optimal?* If the
model will be served heavily, especially on-device, I deliberately *overtrain* a smaller model past
its Chinchilla point — the loss bowl is flat near the minimum, so shifting toward a smaller,
cheaper-to-serve model costs almost nothing in quality while cutting inference cost on every forward
pass forever (LLaMA 3 8B on ~15T tokens, ~2000 tokens/param, is the example). Finally I budget for
reality: real runs lose time to restarts, data loading, and evaluation, and the recipe must stay fixed
or the fit no longer applies.
*Probes: turning a budget into `(N, D)` and immediately flagging deployment-vs-compute-optimal.*

---

## Part E — Data

**E1. Walk me through the Common Crawl pipeline from WARC to trainable tokens.**

The stages, each dropping a large fraction: **(1) Text extraction** — pull real content out of raw
WARC HTML (extract from WARC, *not* the crude WET files; the difference in the final model is large),
using tools like resiliparse/trafilatura, dropping nav, ads, boilerplate. **(2) Language ID** — score
each doc with a fastText language classifier and keep the languages you want above a threshold.
**(3) Quality filtering** — Gopher/C4 heuristics (symbol-to-word ratio, mean word length, fraction of
lines ending in punctuation, duplicate-line fraction) *plus*, increasingly, a fastText classifier
trained to distinguish curated high-quality text from raw web, which substantially outperforms
heuristics; plus a toxicity/NSFW filter. **(4) Deduplication** — exact (hash lines/docs) and fuzzy
(MinHash + LSH). **(5) PII removal** — detect and mask emails, phone numbers, IPs. **(6)
Decontamination** — remove docs overlapping your eval benchmarks by n-gram overlap so eval measures
capability, not memorization. Dedup and model-based quality filtering are the highest-return stages.
The order matters — you dedup and decontaminate after you have clean text to compare.
*Probes: knowing the full pipeline as an ordered sequence, not a bag of tricks.*

**E2. Explain MinHash + LSH dedup and the S-curve. What are you tuning?**

The goal is to find document pairs with high **Jaccard similarity** (|intersection| / |union| of their
n-gram sets) without the O(n²) all-pairs comparison over billions of docs. **MinHash:** represent each
doc as its set of word n-grams, apply many independent hash functions, and for each keep the *minimum*
value. The key theorem: the probability two docs share a given MinHash value equals their Jaccard
similarity — so a signature of `k` MinHashes estimates Jaccard as the fraction of matching entries, an
unbiased estimate, and each doc is now a fixed-length signature. **LSH (banding):** split the `k`-entry
signature into `b` bands of `r` rows each (`k = b·r`); two docs are *candidates* if they match exactly
in at least one band. The probability a pair with true Jaccard `s` becomes a candidate is
$1 - (1 - s^r)^b$, an S-curve in `s`. You tune `b` and `r` to place the steep threshold where you want
the cutoff (e.g. catch pairs above ~0.8, skip below): raising `r` sharpens and raises the threshold
(fewer false positives, more misses), raising `b` lowers it. Only candidate pairs get an exact
comparison, collapsing the quadratic blowup. That single S-curve is the whole tuning story.
*Probes: the Jaccard-equals-collision theorem and the b/r knobs on the S-curve.*

**E3. How does classifier-based quality filtering work, and what is its risk?**

You train a lightweight classifier — typically fastText, because it is fast enough to run over
petabytes — to distinguish "high quality" from "random web." Positives are drawn from curated
high-quality sources (Wikipedia-referenced pages, OpenWebText-style links, a known-good corpus);
negatives from raw Common Crawl. Then you keep documents the classifier scores above a threshold.
The empirical point is that this *substantially outperforms* heuristic rules — it is the lesson
behind DCLM and Nemotron-CC and the current frontier of open data. The risk is that the classifier
bakes in the biases of whatever you *called* "high quality": if your positive set is Wikipedia-like,
you quietly narrow the distribution toward encyclopedic prose and penalize legitimate text (forums,
dialects, code, non-Western sources) that simply does not look like the reference. Filtering is
destructive and you cannot undo it, so you inspect samples of what you keep *and* drop, because a
too-aggressive filter can silently delete all your code and math.
*Probes: knowing the method beat heuristics, and that "quality" is a value judgment with a cost.*

**E4. What is data mixing, and why is the mixture a tuned hyperparameter with a schedule?**

After you have clean data from many sources — web, books, code, arXiv, Wikipedia, synthetic — you
decide how much of each to include, and those weights meaningfully move the model: more code improves
reasoning and structured output *even for non-code tasks*, more math helps quantitative reasoning, but
too much of any narrow source narrows the model. So the mixture is tuned like any other
hyperparameter, on smaller runs, using scaling-law transfer. It is not a single decision but a
*schedule* (a curriculum): front-load broad filtered web in pre-training, raise the fraction of
high-quality curated / math / code toward mid-training, and often finish with an **annealing** phase
on a small, very-high-quality set. The three-stage view (pre-, mid-, post-training) has a *rising*
quality bar, and the same raw source can feed different stages at different filtering thresholds.
Domain sources also get domain handling — GitHub code needs license filtering (permissive only) and
its own dedup.
*Probes: mixture as a tuned, scheduled hyperparameter, not a fixed recipe.*

**E5. Quality versus quantity — how do you think about the tradeoff, and where does dedup sit?**

The field's verdict is that data *quality* moves the loss curve more than almost any architectural
change — two models with equal architecture and compute but different data are not close, which is why
labs guard data pipelines more than architecture. But you also need *volume*: trillions of tokens for
pretraining, so the pre-training stage runs the most permissive quality bar precisely because you
cannot be too picky at that scale. Deduplication is the elegant resolution of the tension: it improves
*quality* (heavy duplication drives memorization and degrades loss-per-unique-token) while *reducing*
wasted quantity (you stop re-learning the same article thousands of times), so it improves models at
fixed compute across multiple papers — a rare stage that costs you nothing to want. The general
instinct: aggressive dedup and model-based filtering first (highest return), then worry about squeezing
more raw volume.
*Probes: that quality dominates but volume is a floor, and dedup improves both.*

**E6. How does contamination sneak into a corpus despite a decontamination stage, and how do you catch it?**

Decontamination removes documents that overlap your *known* benchmarks by n-gram overlap — but it only
catches what you thought to check against. Contamination sneaks in through benchmarks you did not
decontaminate against, through *paraphrased* or reformatted versions of test items that dodge exact
n-gram matching, through synthetic data generated by a model that itself memorized the benchmark, and
through benchmarks released *after* your decontamination pass. Detection has two families: statistical
**exchangeability** tests — a model that memorized a benchmark assigns systematically higher likelihood
to the canonical ordering of examples than to a shuffled ordering, which a clean model would not — and
simply *encouraging providers to disclose* their measured train/test overlap. Your own defense is a
private held-out set created *after* the training cutoff that never touches training or tuning, plus
treating suspiciously high public-benchmark scores with skepticism. The moment an eval influences your
decisions, it starts leaking into the model through your choices.
*Probes: that decontamination is incomplete by construction and how exchangeability tests work.*

---

## Part F — Evaluation

**F1. What does perplexity measure, and what are its limits?**

Perplexity is the exponential of the average negative log-likelihood per token on held-out text —
lower is better, and intuitively a perplexity of `p` means the model is on average as uncertain as if
choosing uniformly among `p` options at each position. It is the direct summary of the training
objective, so it is cheap, continuous, and the right thing to watch during training and to compare
checkpoints or siblings of the *same* model. Its limits: it is only comparable across models with the
*same tokenizer* (perplexity is per-token, and different tokenizers cut text into different token
counts, so you must at least normalize per-byte or per-word to compare across families); it does not
directly measure downstream task quality; and it says nothing about instruction-following or reasoning.
Use it to track a run and rank siblings, never to rank different model families off the shelf.
*Probes: the per-token, same-tokenizer caveat, the most common perplexity misuse.*

**F2. How is a multiple-choice benchmark like MMLU actually scored, and why does the protocol change rankings?**

You do not usually let the model free-generate. The common protocol scores by *log-likelihood*: for
each option, compute the likelihood the model assigns to it, and pick the highest. But there are
distinct variants that give *different numbers on the same model*: score the likelihood of the answer
*letter* ("C"), or of the answer *text* ("Paris"), or *generate* an answer and parse it — and
text-likelihood favors different models than letter-likelihood, while generate-and-parse depends on
formatting robustness. Length-normalizing the option likelihood (dividing by token count) changes
rankings *again*, because otherwise longer options are penalized simply for having more tokens. This
is why two papers can differ mostly in *scoring protocol* rather than model quality, and why harnesses
like the EleutherAI LM-Eval-Harness and HELM exist — to fix a prompt, few-shot format, and scoring
rule per task so two models are at least measured the same way. When you see an MMLU number, ask how it
was scored before believing a comparison.
*Probes: that MC is log-likelihood-scored and the protocol is a hidden confound.*

**F3. Why do base and chat models need different evals?**

A base model is a next-token predictor; a chat model has been aligned to follow instructions, so they
fail different ways and you measure them with different instruments. Base models are evaluated on
knowledge/reasoning benchmarks scored by option likelihood (MMLU and its harder successors MMLU-Pro,
GPQA, HLE) and on checkable math/code (GSM8K, MATH, HumanEval) — the log-likelihood scoring works
because you are probing what the model *knows*. Chat models need to be judged on open-ended,
instruction-following behavior a base model cannot even format: Chatbot Arena (real users vote pairwise,
aggregated to Elo — the closest thing to ground-truth human preference at scale, but slow, uncontrolled,
and gameable by style), and cheaper reproducible proxies MT-Bench and AlpacaEval scored by an LLM
judge. Running a base-model MMLU protocol on a chat model, or an open-ended judge eval on a base model,
measures the wrong thing.
*Probes: the base-vs-chat split and the right instrument for each.*

**F4. What are the biases of LLM-as-judge, and how do you control each?**

LLM judges scale open-ended evaluation and correlate reasonably with human preference, but they have
systematic biases you must design around. **Position bias:** they favor the first (or second) option
in a pairwise comparison — control by randomizing order and averaging both orderings. **Length bias:**
they favor longer answers regardless of quality — control by matching or controlling for length.
**Self-preference:** a judge favors outputs from its own model family — control by using a judge from a
different family, or an ensemble. They can also be gamed by confident formatting and markdown. The
general discipline: use LLM judges for *relative* comparison and coarse signal, calibrate them against
human ratings on a sample, and never treat their scores as ground truth. A cheap sanity check is to
slip in a few items where you know the answer and confirm the judge gets them right.
*Probes: naming position/length/self-preference and the specific control for each.*

**F5. Explain Chatbot Arena and Elo. Why is it valued despite its flaws?**

In Chatbot Arena, real users submit their own prompt, receive two anonymous model responses, and vote
which is better; the pairwise votes are aggregated into an Elo-style rating (the chess system: each
model has a rating, a win against a higher-rated model moves you up more, and the ratings converge to
reflect pairwise win probabilities). It is valued because it is the closest thing to a *ground-truth
human-preference* signal at scale, on *real* prompts users actually cared about — unlike a fixed quiz,
it reflects genuine, diverse, information-seeking use. Its flaws: it is slow, uncontrolled (users pick
their own prompts, so the distribution drifts and cannot be held fixed), and gameable by *style* — a
chattier, more confidently formatted model can win votes without being more correct. So you read it as
a strong human-preference proxy, not as a capability measurement, and you pair it with checkable
benchmarks that Arena cannot provide.
*Probes: understanding Elo aggregation and why "real prompts, human votes" is both its strength and its weakness.*

**F6. Why do benchmarks saturate, and what does the field do about it?**

A benchmark discriminates between models only while they score meaningfully below the ceiling. Once
frontier models cluster near 100%, the remaining gap is noise, formatting, and mislabeled items rather
than capability, so the benchmark stops *ranking* — it saturates. Two things drive it: models genuinely
get better, and contamination inflates scores on public benchmarks that leaked into training. The field
responds with an arms race of harder successors: MMLU saturated, so MMLU-Pro widened each question to
10 choices to cut guessing, GPQA moved to PhD-level Google-proof expert-written questions, and HLE
("Humanity's Last Exam") pushes ~2,500 hard, often multimodal, frontier questions. The deeper fix is to
stop trusting leaderboard rank and evaluate on your *own* realistic, information-seeking task
distribution with a private held-out set — and to report capability alongside cost, since a marginally
better but 10× pricier model is not obviously better for a product.
*Probes: saturation as loss of discriminative power, the successor arms race, and the real fix.*

## You can now

- Name the three ways to fit a scaling law empirically and explain why all three must agree before you trust an extrapolation.
- Explain why Kaplan and Chinchilla disagreed, read the exponents of `L(N,D) = E + A/N^α + B/D^β`, and plan a real run's `(N, D)` from a fixed FLOPs budget, including deployment-vs-compute-optimal.
- Walk the Common Crawl pipeline end to end (extraction, langID, quality filtering, dedup, PII, decontamination) and explain MinHash+LSH's S-curve.
- Explain classifier-based quality filtering, the quality-vs-quantity tradeoff, and how contamination survives decontamination.
- Explain perplexity's limits, how MMLU is actually scored, why base and chat models need different evals, LLM-as-judge biases, Chatbot Arena's Elo, and why benchmarks saturate.
