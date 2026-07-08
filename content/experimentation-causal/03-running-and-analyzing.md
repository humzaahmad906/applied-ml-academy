# 03 — Running and Analyzing

The experiment is designed, the sample size is set, and the traffic has flowed. Now two numbers sit on your dashboard: control converted at 10.0%, treatment at 10.8%. Did the change work? This lesson is the analyst's core loop — turn the raw counts into a test statistic, a p-value, and a confidence interval, then read those numbers honestly, which mostly means resisting the two opposite temptations: declaring victory on noise, and dismissing a real effect because it isn't "significant."

## The comparison, made precise

You have two groups and one metric. For a **rate** metric (conversion, click-through) the natural summary is a proportion in each arm; for a **continuous** metric (revenue per user, session minutes) it's a mean. In both cases the quantity you care about is the *difference*, and the question is whether that difference is larger than what random assignment alone would produce if the treatment did nothing. That last clause is the null hypothesis from your stats course, now with an experiment attached.

The engine is the same for every test: compute the observed difference, estimate its **standard error** (how much that difference would wobble from sheer sampling noise), form the ratio, and ask how extreme that ratio is under the null.

## The two-proportion z-test

Conversion rates are proportions, so we use a **z-test for two proportions**. Say control had 500 conversions out of 5,000 users (10.0%) and treatment had 540 out of 5,000 (10.8%). The observed lift is 0.8 pp. Is it real?

```python
import numpy as np
from scipy import stats

x = np.array([540, 500])       # conversions: treatment, control
n = np.array([5000, 5000])     # users per arm
p1, p2 = x / n
diff = p1 - p2

# Pooled proportion for the standard error under H0 (rates equal).
p_pool = x.sum() / n.sum()
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n[0] + 1 / n[1]))
z = diff / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))    # two-sided

print("lift:", round(diff, 4), " z:", round(z, 3), " p:", round(p_value, 3))
# output: lift: 0.008  z: 1.31  p: 0.19
```

The p-value is 0.19 — well above 0.05. Under the null that both arms convert identically, a gap of 0.8 pp or larger happens about 19% of the time by chance alone. That is not rare. **We do not have evidence the change worked.** Note the pooled proportion in the standard error: under H₀ the two rates are equal, so we pool all the data to estimate that single shared rate — the most honest denominator for the noise.

`statsmodels` wraps this exactly, and you should prefer it in real code:

```python
from statsmodels.stats.proportion import proportions_ztest
z, p = proportions_ztest(count=[540, 500], nobs=[5000, 5000])
print("z:", round(z, 3), " p:", round(p, 3))
# output: z: 1.31  p: 0.19
```

## The t-test for continuous metrics

When the metric is a mean rather than a proportion — revenue per user, minutes in app — you use a **two-sample t-test**. Same skeleton: difference in means over its standard error, but the reference distribution is Student's t (which accounts for estimating the variance from the data), and you should use **Welch's** version, which does *not* assume the two arms share a variance. That assumption is almost never safe in practice, and Welch costs you nothing when it holds.

```python
rng = np.random.default_rng(1)
control   = rng.normal(20.0, 8.0, 4000)     # revenue per user, $
treatment = rng.normal(20.6, 8.0, 4000)     # true +$0.60 lift

t, p = stats.ttest_ind(treatment, control, equal_var=False)   # Welch
print("mean lift:", round(treatment.mean() - control.mean(), 3),
      " t:", round(t, 3), " p:", round(p, 4))
# output: mean lift: 0.588  t: 3.294  p: 0.001
```

Here p ≈ 0.001, far below 0.05: the $0.59 observed lift is very unlikely under "no effect," so we reject the null. Revenue metrics are notoriously heavy-tailed (a few whales dominate), which inflates variance and can violate the t-test's normality assumption — for those, people often log-transform, cap outliers, or use a bootstrap. But for well-behaved continuous metrics, Welch's t-test is the workhorse.

## Confidence intervals: the number to actually report

A p-value answers a yes/no question. A **confidence interval** answers the question you usually care more about: *how big is the effect, and how sure are we?* Report the interval, not just the verdict. Take the revenue result:

```python
diff = treatment.mean() - control.mean()
se = np.sqrt(treatment.var(ddof=1) / len(treatment) +
             control.var(ddof=1)   / len(control))
lo, hi = diff - 1.96 * se, diff + 1.96 * se
print(f"lift ${diff:.2f}, 95% CI [${lo:.2f}, ${hi:.2f}]")
# output: lift $0.59, 95% CI [$0.24, $0.94]
```

Read this as: *our best estimate is +$0.59 per user, and the data are consistent with anything from +$0.24 to +$0.94.* Two things fall out for free. First, the interval excludes 0, which is the same information as p < 0.05 — a 95% CI that misses zero and a two-sided test at α = 0.05 always agree. Second, and more useful, the interval tells you the *plausible magnitudes*: even the pessimistic end is a real gain here. The correct reading of "95% confidence" is subtle — it's a statement about the *procedure* (95% of intervals built this way cover the truth), not a 95% probability that this particular interval contains it — but for decision-making, the practical reading "the effect is plausibly somewhere in this range" serves you well.

## Statistical vs practical significance

Here is the distinction that separates good analysts from p-value robots. **Statistical significance** means "probably not zero." **Practical significance** means "big enough to matter." They are independent, and all four combinations happen:

- **Significant and large:** ship it. The clean win.
- **Significant but tiny:** with millions of users, a +$0.002 lift can hit p < 0.001. It's real and it's worthless — the confidence interval sits at, say, [$0.001, $0.003], entirely inside the "who cares" zone. Reporting only "p < 0.001, significant!" here is misleading.
- **Not significant but large point estimate:** the estimate looks great (+$3) but the CI is [−$1, +$7] — you simply don't have enough data to tell a big win from a small loss. This is an *underpowered* result, not a negative one. "Not significant" ≠ "no effect."
- **Not significant and small:** a genuine null, if you were well-powered. Move on.

This is exactly why lesson 02 made you commit to an MDE, and why this lesson makes you report a confidence interval. The MDE told you what "big enough to matter" means *before* you saw the data; the CI tells you whether the experiment could distinguish that from zero. A result is only clean when the whole confidence interval lands on the same side of your MDE.

## A reading checklist

When results land, walk this in order:

1. **Sanity first.** Did the arms get the traffic split you expected (a 50/50 test showing 55/45 is a red flag we'll dissect in lesson 04)? Are the counts what you planned?
2. **Point estimate and CI.** What's the effect and its plausible range? Compare the range to your MDE.
3. **P-value / significance.** Is it distinguishable from zero at your α?
4. **Guardrails.** Did anything you promised not to harm move? A guardrail breach can veto an OEC win.
5. **Decision.** Ship, kill, or iterate — stated as a sentence, tied back to the hypothesis you wrote in design.

Resist two reflexes throughout: don't crown a winner because p squeaked under 0.05 on a trivial effect, and don't bury a promising-but-noisy result as a "failure" when it was really just underpowered. The honest report always carries both the magnitude and the uncertainty.

## Key takeaways

- Every test follows one skeleton: observed difference ÷ its standard error → a statistic → a p-value under the null.
- Use the **two-proportion z-test** for rates (pool the proportion for the standard error under H₀) and **Welch's t-test** for means (don't assume equal variances).
- Prefer library calls (`proportions_ztest`, `ttest_ind(..., equal_var=False)`) over hand math in production, but know the formula so you can debug it.
- Report a **confidence interval**, not just a p-value — it gives the effect's plausible magnitude, and excluding zero is equivalent to p < 0.05.
- **Statistical significance ("not zero") is not practical significance ("big enough").** All four combinations occur; judge the CI against your pre-registered MDE.
- "Not significant" often means *underpowered*, not "no effect" — a wide CI around a large estimate is a call for more data, not a rejection.

## Try it

Take the conversion example that came back non-significant (540 vs 500 out of 5,000). Build its 95% confidence interval for the difference in proportions by hand, using the *unpooled* standard error `sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)`. Confirm the interval straddles zero, consistent with p = 0.19. Then, keeping the rates fixed at 10.8% and 10.0%, re-run `proportions_ztest` with the sample size growing from 5,000 to 50,000 per arm and watch the p-value fall below 0.05. Write one sentence explaining why the *same* 0.8 pp effect flips from non-significant to significant purely by adding users — and what that says about reading significance without an MDE.
