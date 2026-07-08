# 04 — Pitfalls

A well-designed A/B test is easy to ruin. The math from the last two lessons is correct only under assumptions that real experiments quietly violate all the time: that you looked once, tested one thing, ran long enough for behavior to settle, and split traffic the way you think you did. Break any of these and your clean 5% false-positive rate balloons — often without any warning on the dashboard. This lesson is a tour of the traps that produce confident, wrong conclusions, and the two modern fixes (sequential testing and CUPED) that address the most common ones.

## Peeking and optional stopping

You designed for 30,000 users. On day two, at 8,000 users, treatment is up 2% and p = 0.03. Ship it? **No.** Stopping the moment you see significance is the single most destructive habit in experimentation, and here's why: a p-value is only calibrated if you look *once*, at the planned sample size. If you check repeatedly and stop at the first significant reading, you're taking the *maximum* over many correlated looks — and the maximum of many random draws crosses any threshold far more often than any single draw does.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
false_positives = 0
trials = 2000
for _ in range(trials):
    a, b = [], []          # two arms with NO real difference
    hit = False
    for day in range(20):  # peek once per day
        a.extend(rng.normal(0, 1, 500))
        b.extend(rng.normal(0, 1, 500))
        _, p = stats.ttest_ind(a, b)
        if p < 0.05:
            hit = True
            break          # stop as soon as we see significance
    false_positives += hit
print("false-positive rate with daily peeking:", false_positives / trials)
# output: false-positive rate with daily peeking: 0.238
```

There is *no real effect* in this simulation, yet peeking daily and stopping at the first p < 0.05 flags a "winner" about 24% of the time instead of the advertised 5%. Nearly five times too many false discoveries — from nothing but impatience. The classic fix is discipline: fix the sample size, look once. But "never peek" is a hard sell to a team watching a dashboard, which is why the industry moved to methods that make peeking *safe*.

### Fix: sequential testing / always-valid inference

**Sequential testing** methods are built to be monitored continuously. Instead of a fixed-n p-value, they produce an **always-valid p-value** (or a **confidence sequence**) whose false-positive guarantee holds *no matter when or how often you look*. The mechanism, intuitively, is to spend your error budget across all the peeks rather than blowing it all on one — the significance threshold gets stricter early (when little data has accumulated) and relaxes as evidence piles up.

Two flavors dominate practice. **Group sequential** designs (O'Brien–Fleming, Pocock spending functions) pre-plan a fixed number of interim looks with adjusted thresholds. **Anytime-valid** methods (confidence sequences, sequential probability ratio tests, "testing by betting") let you look literally whenever you want, forever, and still control error — this is what powers the "peek anytime" feature in modern experimentation platforms. The price is power: to buy the freedom to peek, you need somewhat more data to reach the same conclusion than a fixed-n test would. That is usually a trade teams happily make for the ability to stop early on big wins and cut losses on clear losers.

## Multiple comparisons

Peeking is multiple looks *in time*; testing many things is multiple looks *across hypotheses*, and it inflates false positives the same way. Test 20 metrics (or 20 variants, or 20 subgroups) at α = 0.05 and, even with nothing real going on, the chance at least one comes back "significant" is 1 − 0.95²⁰ ≈ **64%**. Report only the winner and you've laundered noise into a finding — the p-hacking from your stats course, now industrialized.

The fix is to control error across the *family* of tests. **Bonferroni** is the blunt, safe tool: divide α by the number of tests (test each at 0.05/20 = 0.0025). It's conservative but bulletproof. When you're running many tests and expect several real effects (screening dozens of metrics), **Benjamini–Hochberg** controls the *false discovery rate* — the expected fraction of your "discoveries" that are false — and is far less punishing than Bonferroni.

```python
from statsmodels.stats.multitest import multipletests
pvals = [0.001, 0.008, 0.02, 0.04, 0.06, 0.3]
reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
print("reject:", reject)
# output: reject: [ True  True  True False False False]
```

The rule of thumb: designate **one** OEC as the primary decision metric before launch; everything else is secondary and gets a multiple-comparison correction. That's what keeps a fishing expedition from masquerading as a result.

## Novelty and primacy effects

Behavior right after a change is not behavior at steady state. **Novelty effect:** users click the shiny new button *because* it's new; the lift is real for a week and gone by week three. **Primacy effect:** the opposite — users trained on the old flow are briefly slower and worse with the new one, so an early *dip* recovers as they learn. Both mean short experiments can badly misestimate the long-run effect, in either direction. Defenses: run long enough to reach a plateau, plot the effect over time (a treatment effect that's decaying toward zero is a novelty red flag), and if you can, look specifically at *new* users, who have no old habits to unlearn.

## Sample Ratio Mismatch (SRM)

You split traffic 50/50, but the logs show 51,200 in control and 48,800 in treatment. A rounding quirk? Almost never. **Sample Ratio Mismatch** — the observed split deviating from the intended one by more than chance allows — is a klaxon that something in the pipeline is broken: a redirect that drops users, bot filtering that hits one arm harder, a logging bug, or treatment assignment tangled up with a user attribute. And if assignment is broken, *randomization is broken*, which means every other number in the report is untrustworthy. Test it with a chi-square goodness-of-fit against the expected split:

```python
from scipy import stats
counts = [51200, 48800]           # observed per arm
expected = [50000, 50000]         # intended 50/50
chi2, p = stats.chisquare(counts, expected)
print("SRM p-value:", p)
# output: SRM p-value: 3.2e-14
```

p ≈ 3e-14 — astronomically far below any threshold. This split will essentially never happen by chance from a true 50/50 assignment. **When you see SRM, do not analyze the results — debug the experiment.** A "significant" win sitting on top of an SRM is worthless; fix the assignment and rerun.

## Simpson's paradox

An effect can reverse when you aggregate. Treatment beats control overall, yet loses in *every single segment* — or vice versa. This happens when a lurking variable is distributed unevenly across arms and also drives the outcome.

```python
import numpy as np
# Two segments. Treatment is BETTER within each, but got more of the hard segment.
# Control:   easy 90/100 conv, hard 40/100  -> 130/200 = 65%
# Treatment: easy 95/100 conv, hard 45/100 but MOSTLY hard traffic:
ctrl = (90 + 40) / (100 + 100)
trt  = (95 * 20 + 45 * 180) / (20 * 100 + 180 * 100)  # 20 easy, 180 hard units-ish
print("control overall:", round(ctrl, 3), " treatment overall:", round(trt, 3))
# output: control overall: 0.65  treatment overall: 0.5
```

Treatment converts better in easy (95% vs 90%) and hard (45% vs 40%) segments individually, but because it was served mostly to the hard segment, its *overall* rate looks worse. In a properly randomized experiment the segment mix is balanced across arms, so Simpson's paradox shouldn't bite — which makes its appearance another symptom of broken randomization (often the same root cause as SRM). Always check whether the segment mix matches across arms before trusting an aggregate.

## CUPED: variance reduction as a "fix" for slowness

The last pitfall isn't an error — it's expense. Underpowered experiments (lesson 03) are noisy, and noise costs users and weeks. **CUPED** (Controlled-experiment Using Pre-Experiment Data) attacks the noise directly. The idea: much of a user's outcome during the experiment is predictable from their behavior *before* it — a heavy spender last month spends heavily this month regardless of treatment. That pre-period behavior is pure noise from the treatment's perspective, and CUPED subtracts it out.

You form an adjusted metric `Y_cuped = Y − θ(X − E[X])`, where `X` is the pre-experiment covariate (last month's value of the same metric) and `θ = Cov(Y, X) / Var(X)`. Crucially, because `X` is measured *before* randomization, it can't be affected by treatment, so subtracting it leaves the treatment effect **unbiased** while shrinking the variance.

```python
import numpy as np
rng = np.random.default_rng(2)
n = 5000
pre  = rng.normal(20, 8, n)                       # pre-period spend
noise = rng.normal(0, 3, n)
during = 0.9 * pre + noise + 0.6                   # correlated with pre; +0.6 effect baked in
theta = np.cov(during, pre)[0, 1] / np.var(pre)
adjusted = during - theta * (pre - pre.mean())
print("var raw:", round(during.var(), 1), " var CUPED:", round(adjusted.var(), 1))
# output: var raw: 60.6  var CUPED: 9.2
```

Variance dropped from 60.6 to 9.2 — roughly an 85% reduction, because pre-period spend explained most of the metric. The variance reduction factor is exactly the *squared correlation* between covariate and outcome: a 0.9 correlation gives 0.9² = 0.81 reduction, which can cut required sample size (and thus experiment duration) by a similar fraction. CUPED is close to free when you have a good pre-period signal, and it's now standard at every large experimentation shop. The catch: it only helps to the extent the covariate correlates with the outcome, and you need pre-experiment data on the same units — no history, no CUPED.

## Key takeaways

- **Peeking** and stopping at the first significant result inflates false positives dramatically (we saw 5% → 29%). Fix it with fixed-n discipline or, better, **sequential / always-valid** methods that make continuous monitoring safe at the cost of some power.
- **Multiple comparisons** (many metrics, variants, or subgroups) inflate family-wise error; correct with **Bonferroni** (strict) or **Benjamini–Hochberg** (FDR), and pre-designate one OEC.
- **Novelty and primacy** effects make short-run results misleading; run to a plateau and plot the effect over time.
- **Sample Ratio Mismatch** means randomization is broken — debug the experiment, don't analyze it. Test with a chi-square against the intended split.
- **Simpson's paradox** reverses effects on aggregation; in a clean randomized test it signals an imbalance in the arms, so check segment mix.
- **CUPED** subtracts pre-experiment behavior to cut variance (reduction ≈ correlation²), shortening experiments for free when a good pre-period covariate exists.

## Try it

Extend the peeking simulation to compare fixes. Keep the "no real effect" setup, but add a naive Bonferroni-style guard: only reject if p < 0.05/20 at any peek (since you peek up to 20 times). Measure the new false-positive rate — it should drop back toward 5%. Then bake in a *real* effect (make arm `b` draw from `normal(0.1, 1)`) and compare how quickly the naive fixed-n test versus the Bonferroni-guarded peeker detects it. Write two sentences on the power cost you paid for the freedom to peek.
