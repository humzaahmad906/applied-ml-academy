# 08 — Scaling Laws and Compute-Optimal Training

You have a fixed compute budget. How big a model should you train, and on how much data? Guessing
wastes money at scale. Scaling laws turn this into arithmetic. This chapter is the payoff of the
`6ND` FLOP rule we derived earlier, and it is long because it spans a lot: the empirical laws
themselves and the theory behind compute-optimal training, the Kaplan-to-Chinchilla story, the
three concrete methods for *fitting* a scaling law (which is exactly what the build asks), and
how to extrapolate one to plan a real run without getting burned.

## The empirical fact

Across many orders of magnitude, the loss of a well-trained language model falls as a smooth power
law in the compute you spend, the parameters you use, and the tokens you train on. Plot loss
against any of these on log-log axes and you get close to a straight line over a wide range —
predictions hold across roughly **6 orders of magnitude in model size (10^6 to
10^12 params), 4 in data (10^9 to 10^13 tokens), and 3 in compute (10^18 to 10^21 FLOPs)**. This is
remarkable and not obvious a priori: it means you can fit the line on small, cheap runs and
extrapolate to predict the loss of a run far too expensive to do speculatively. That predictability
is what makes it safe to spend millions on a single training run — you are not gambling, you are
reading off a curve you already measured.

The single-variable forms, when the other resource is not the bottleneck, are pure power laws:

$$
L(N)\propto N^{-\alpha}\ (\alpha\approx0.076),
\qquad L(D)\propto D^{-\beta}\ (\beta\approx0.095),
\qquad L(C)\propto C^{-\gamma}\ (\gamma\approx0.057)
$$

(These are the Kaplan et al. 2020 exponents.) A power law in log-log space is a straight
line whose *slope* is the exponent, so fitting a scaling law is, at heart, fitting a line to a few
points — the whole difficulty is in doing it honestly.

## The parametric loss: L(N, D)

The single-variable laws are special cases of a joint form that separates the two ways a model can
be starved. Chinchilla (Hoffmann et al. 2022) writes:

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

Read it term by term:

- **E** is the irreducible loss — the entropy of the data itself, the floor you cannot model away
  no matter how large the model or dataset. In Chinchilla's fit, E ≈ 1.69 (nats/token on their
  data).
- **A / N^α** is the penalty for a model too small to represent the function. Chinchilla fit
  A ≈ 406, α ≈ 0.34.
- **B / D^β** is the penalty for too little data — the model overfits or simply hasn't seen enough.
  Chinchilla fit B ≈ 410, β ≈ 0.28.

(Do not overread the exact constants; they depend on the data and tokenizer. The *structure* — a
floor plus two decaying terms with comparable exponents — is the durable insight. Note also that
the Chinchilla α, β are much larger than the Kaplan single-variable exponents above; the two papers
parameterize differently, which is part of why they disagreed.) This joint form is the target of
the third fitting approach below, and it is what lets you answer the compute-optimal question by
calculus rather than by guesswork.

## The compute-optimal question

Fix total compute `C ≈ 6ND` (recall: ~6 FLOPs per parameter per token, forward + backward).
You can spend it on a big model trained on little data, a small model trained on lots of data, or
anything between. Which allocation minimizes loss? Substitute `D = C / (6N)` into `L(N, D)` and
minimize over `N`. Because the two penalty terms trade off — shrinking `N` inflates `A/N^α` while
the freed compute grows `D` and shrinks `B/D^β` — there is an interior minimum, and it scales as a
power of `C`:

$$
N_{\text{opt}}\propto C^{a},\ \ a = \frac{\beta}{\alpha + \beta}\approx 0.46\approx \tfrac12;
\qquad
D_{\text{opt}}\propto C^{b},\ \ b = \frac{\alpha}{\alpha + \beta}\approx 0.54\approx \tfrac12
$$

So **both parameters and tokens scale as roughly the square root of compute** — you grow them
*together*, in lock-step, as the budget grows. Equivalently `N_opt ∝ D_opt^(α/β) ≈ D_opt^0.8`.
Cranking through Chinchilla's constants gives the famous rule of thumb: **about 20 tokens per
parameter** at the optimum. A compute-optimal 1B model wants ~20B tokens; a 70B model wants ~1.4T
tokens.

## Kaplan vs Chinchilla: why the answer changed

The original scaling-laws work (Kaplan et al. 2020) concluded you should spend most of a new budget
on a *bigger model* and comparatively little extra on data — roughly `N ∝ C^0.73`, data barely
growing. That guidance shaped a whole generation of models, including GPT-3 (175B params trained on
only ~300B tokens — under 2 tokens per parameter).

Chinchilla (Hoffmann et al. 2022) redid the experiment more carefully and got a very different
answer: `N ∝ C^~0.5`, i.e. parameters and data scale *equally*. The discrepancy came down to
methodology. Kaplan largely used a *fixed* learning-rate schedule and a fixed number of steps
across model sizes, which systematically under-trained the smaller models and biased the fit toward
"bigger is better." Chinchilla varied the token count properly and tuned the schedule to the run
length. The practical verdict was stark: their 70B **Chinchilla**, trained compute-optimally on
1.4T tokens, *beat* the 280B **Gopher** trained on 300B tokens — a 4× smaller model won because it
was trained on the right amount of data. GPT-3-era models were badly *over*-parameterized and
*under*-trained. This is the correction that reset the field's defaults.

## The three approaches to fitting (this is the build)

You do not derive the optimum from theory; you *measure* it, and Chinchilla gives three
independent methods that must agree for you to trust the answer. The build has you run a small
version of this: you query a training API (a hosted service at a fixed FLOP budget that
returns the loss for a `(N, D)` configuration you request), spend your budget on a sweep, fit the
frontier, and extrapolate to a larger target budget you are *not* allowed to run directly. The
three approaches:

**Approach 1 — Fix model sizes, vary data (minimum over training curves).** Train several model
sizes, each on many token counts, and record the loss along each training curve. For every compute
level `C`, look across all the runs that hit that `C` and take the *minimum* loss; the `(N, D)` that
achieved it is the compute-optimal point at that `C`. Trace those minima across `C` and fit
`N_opt ∝ C^a`. Cheap because it reuses points along curves you were training anyway.

**Approach 2 — IsoFLOP profiles.** This is the cleanest and the one the build centers on:

1. Pick several fixed compute budgets (isoFLOP slices), each the curve `C = 6ND` for a chosen `C`.
2. Within each budget, train several model sizes `N`, giving each the matching `D = C / (6N)` so
   they all consume the *same* total compute.
3. For each budget, plot final loss against model size. It is a **U-shape (a bowl)**: too small a
   model underfits despite abundant data, too large a model saw too little data. The bottom of the
   bowl is the compute-optimal `N` for that budget (fit a parabola in log-space to locate it
   precisely, rather than just taking the lowest sampled point).
4. Collect the bowl-bottoms `(C, N_opt)` and fit `N_opt ∝ C^a`, then get `D_opt = C / (6 N_opt)`.
   Extrapolate the fitted line to your real budget.

**Approach 3 — Fit the parametric loss directly.** Fit `L(N, D) = E + A/N^α + B/D^β` to *all* your
`(N, D, loss)` points at once (a nonlinear regression, typically minimizing a Huber loss on
log-loss for robustness to outliers). Once you have `E, A, B, α, β`, you have the whole surface and
can compute the optimum analytically via the `a = β/(α+β)` formula — no need to hit each budget's
minimum empirically. Most data-efficient, most sensitive to fitting choices.

The point of the exercise is not the specific constants you recover on a toy corpus — it is
learning to turn a compute budget into a model size and token count with *evidence* instead of
vibes, and to sanity-check that the three methods agree before you trust an extrapolation.

## Extrapolation and its pitfalls

The entire value of a scaling law is extrapolation: fit on the cheap end, predict the expensive
end. The pitfalls are where people lose money:

- **Extrapolating too far.** A fit is trustworthy roughly within and modestly beyond the range you
  measured. Predicting 100× past your largest run is faith, not science — hold out your largest
  affordable point and *check* the fit predicts it before betting on points beyond it.
- **A changing recipe.** The law is fit *for a fixed architecture, data distribution, optimizer,
  and schedule*. Change the data mix, the tokenizer, or the LR schedule and you are on a different
  curve — the constants (and sometimes the exponents) move. Better data shifts the whole line
  *down*, which is why data work is so valuable and why you cannot mix runs with
  different data into one fit.
- **Under-tuned small runs.** The Kaplan mistake in miniature: if your small models are trained
  with a schedule tuned for large ones, the small end of your fit is biased and the extrapolation
  tilts. Tune per run length.
- **Reading the wrong bowl-bottom.** IsoFLOP bowls are flat near the minimum; the naive lowest
  sampled point is noisy. Fit the parabola.

## Compute-optimal is not deployment-optimal

Here is the subtlety that matters enormously in practice and that the Chinchilla rule alone
obscures. The Chinchilla optimum minimizes loss for a fixed *training* budget. It says nothing about
inference. If you will serve a model to millions of users, you pay the inference cost on every
forward pass, forever, and a smaller model is cheaper to serve at every one of them.

So it is often rational to deliberately **overtrain** a smaller model far past its Chinchilla point:
spend extra training compute once to get a smaller model that is cheaper to serve billions of times.
The training run costs more than compute-optimal, but the *lifetime* cost including inference is
much lower. This is exactly why LLaMA-series and most deployment-targeted models train on token
counts far above 20/param — LLaMA 3 8B saw ~15T tokens, nearly 2000 tokens per parameter. Formally,
the flatness of the loss bowl near its minimum is what makes this cheap: moving off the
compute-optimal `N` costs you very little loss, so a large shift toward a smaller, cheaper-to-serve
model is nearly free in quality.

For your work this is the whole game. The on-device models you ship are small on purpose, trained
hard on lots of data, because inference cost and memory on the device dominate everything. The
Chinchilla optimum is the wrong target when the device, not the training cluster, is the
constraint. Train the small model longer.

## What scaling laws do and do not tell you

They tell you: given a budget and a fixed recipe, what size and token count minimize loss, and
roughly what loss to expect. They also let you tune hyperparameters on small models and transfer
the settings up — the idea behind muP and related parameterizations that make learning rates
transfer across scale, saving the enormous cost of tuning at full size.

They do not tell you: whether lower loss means better downstream behavior on your actual task
(usually correlated, not guaranteed), how a change in data quality shifts the curves (it does, and
predictably — but you must re-fit), or anything about capabilities that appear discontinuously with
scale. Loss is smooth; benchmark accuracy sometimes is not. Treat scaling laws as a budgeting and
extrapolation tool, not a theory of everything.

## Key takeaways

Loss follows smooth power laws: `L(N)∝N^-α`, `L(D)∝D^-β`, `L(C)∝C^-γ` (Kaplan α≈0.076, β≈0.095,
γ≈0.057), which lets you fit cheap small runs and extrapolate to expensive ones. The joint law
`L(N,D) = E + A/N^α + B/D^β` separates the model-too-small and data-too-small penalties above an
irreducible floor E. Minimizing it under `C≈6ND` gives `N_opt ∝ C^(β/(α+β)) ≈ √C` and
`D_opt ∝ √C` — parameters and tokens scale together, ~20 tokens/param. Kaplan said grow the model
(`N∝C^0.73`); Chinchilla corrected the methodology and showed equal scaling, with 70B Chinchilla
beating 280B Gopher. You *fit* the law three ways — minimum over training curves, IsoFLOP bowls
(the core of the build: equal-compute models per budget, find the bottom of the U), and fitting the
parametric loss directly — and they must agree before you trust an extrapolation, whose pitfalls are
over-reaching the measured range, changing the recipe, and under-tuned small runs. Finally,
compute-optimal minimizes *training* cost only; when you will serve the model heavily — above all
on-device — overtrain a smaller model past its Chinchilla point to cut lifetime inference cost.

## You can now

- write the single-variable power laws and the joint Chinchilla form `L(N,D) = E + A/N^α + B/D^β`, and interpret each term.
- solve the compute-optimal allocation under `C ≈ 6ND` and recover both the `√C` scaling of `N` and `D` and the `~20 tokens/param` rule.
- explain why Kaplan and Chinchilla disagreed, and what methodological fix (per-run schedule tuning, proper token-count sweeps) changed the answer.
- fit a scaling law three ways — minimum over training curves, IsoFLOP bowls, and direct parametric regression — and cross-check them before trusting an extrapolation.
- decide when to deliberately overtrain a smaller model past its Chinchilla point to minimize lifetime inference cost, especially for on-device deployment.
