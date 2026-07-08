# 05b — Regression Metrics and Class Imbalance

Lesson 05 covered classification metrics and the traps hiding inside them. This lesson picks up two threads it left dangling. First: what do you report when the target is a *number* — a house price, a delivery time — instead of a class? Regression has its own family of metrics, and each one lies in its own way. Second: we said accuracy collapses on imbalanced data, but *knowing* a metric is broken doesn't fix your model. Here we get concrete about what to actually do when the positive class is rare. We close with calibration — the often-ignored question of whether the probabilities your model prints can be trusted as probabilities at all.

## Regression metrics: four ways to measure "how far off"

When your model predicts a continuous value, every error is a distance: the gap between what you predicted and the truth. The metrics differ only in how they summarize a pile of those gaps into one number.

**MAE (Mean Absolute Error)** is the average of the absolute errors: `mean(|y_true - y_pred|)`. It's in the same units as your target and reads like plain English — "on average we're off by \$12,000." Because it treats an error of 10 as exactly twice as bad as an error of 5, it's **robust to outliers**: one wildly wrong prediction moves it only in proportion to its size.

**MSE (Mean Squared Error)** averages the *squared* errors. Squaring means a single large miss dominates the score — an error of 10 contributes 100, an error of 5 contributes 25, so that one bad prediction counts four times as much. MSE punishes big misses hard. That's a feature when large errors are genuinely catastrophic, and a bug when your data has outliers you don't actually care about, because a handful of them will hijack the whole metric. MSE is also in *squared* units (dollars-squared), which is meaningless to a human.

**RMSE (Root Mean Squared Error)** takes the square root of MSE to bring it back into the target's units. It keeps MSE's heavy penalty on large errors but is now readable as "\$14,000-ish." RMSE is always ≥ MAE; the gap between them tells you how much your errors vary in size. RMSE ≈ MAE means errors are uniform; RMSE ≫ MAE means a few big misses are inflating things.

**R² (coefficient of determination)** answers a different question: what fraction of the variance in the target does the model explain, compared to just predicting the mean every time? R² = 1.0 is perfect, R² = 0 means you're no better than the dumb mean-predictor, and R² can go *negative* when the model is worse than the mean. It's unitless, which makes it great for a quick "is this model doing anything at all?" sanity check — but that same unitlessness hides the actual error magnitude, and R² inflates as you add features whether or not they help.

**MAPE (Mean Absolute Percentage Error)** expresses each error as a percentage of the true value: `mean(|y_true - y_pred| / |y_true|)`. Percentages travel well across scales — "we're off by 8%" means the same thing for a \$100 item and a \$100,000 one. But MAPE has a nasty failure mode: it **explodes when true values are near zero** (dividing by a tiny number), and it's asymmetric, penalizing over-predictions and under-predictions unequally. Never use MAPE when your target can be zero or negative.

```python
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)

y_true = np.array([100, 200, 300, 400, 500])
y_pred = np.array([110, 190, 310, 380, 700])  # last one is a big miss

print("MAE :", mean_absolute_error(y_true, y_pred))
print("MSE :", mean_squared_error(y_true, y_pred))
print("RMSE:", root_mean_squared_error(y_true, y_pred))
print("R2  :", r2_score(y_true, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
# output:
# MAE : 46.0
# MSE : 8520.0
# RMSE: 92.3038...
# R2  : 0.574
# MAPE: 0.1793...   (≈ 18%)
```

Notice RMSE (92) is double MAE (46): that lone 200-unit miss on the last point is doing the damage, and RMSE screams about it while MAE stays calm. `root_mean_squared_error` is the modern function (added in scikit-learn 1.4); the old `mean_squared_error(..., squared=False)` trick is deprecated.

**Which to report?** Report MAE when you want an honest, outlier-resistant sense of typical error and your stakeholders need a number in real units. Report RMSE when large errors are disproportionately costly and you want the metric to reflect that. Report R² alongside, as a scale-free "is it better than nothing" check. Reach for MAPE only when relative error is what the business cares about *and* your targets are safely away from zero. As with classification: pick before you model, not after.

## Class imbalance: knowing the metric lies isn't enough

Recall the fraud example from lesson 05 — 99% of transactions are legitimate, so predicting "legitimate" always scores 99% accuracy while catching zero fraud. Lesson 05 told you to *distrust* accuracy here and look at precision, recall, F1, and PR-AUC. This section is about what you actually *change* so the model stops ignoring the rare class.

The root problem: most classifiers minimize total error, and when negatives outnumber positives 99-to-1, the cheapest way to be "right" is to bet on the majority every time. You have to tilt the game back toward the minority. There are four levers.

**Lever 1 — class weights.** Tell the model that mistakes on the rare class cost more. Most sklearn classifiers accept `class_weight='balanced'`, which automatically weights each class inversely to its frequency, so the rare positives pull as hard as the common negatives during training. This is the lowest-effort, first-thing-to-try fix — no new data, one keyword.

**Lever 2 — resampling.** Rebalance the training set itself. *Undersampling* throws away majority examples until the classes are even (fast, but discards data). *Oversampling* duplicates minority examples (keeps all data, risks overfitting to the copies). **SMOTE** (Synthetic Minority Over-sampling Technique) is smarter: instead of copying, it synthesizes new minority points by interpolating between real ones. SMOTE lives in the `imbalanced-learn` library, not sklearn itself. The one rule you must never break: **resample only the training split, never the test split** — your test set has to reflect the real, imbalanced world, or your metrics are fiction.

**Lever 3 — threshold moving.** Your model outputs a probability; the default 0.5 cutoff is just a convention, and it's usually wrong for imbalanced problems. Lowering the threshold catches more positives (higher recall) at the cost of more false alarms (lower precision) — exactly the tradeoff from lesson 05, now used deliberately. Sweep the threshold and pick the operating point your problem demands.

**Lever 4 — measure with the right AUC.** Lesson 05 warned that ROC-AUC flatters models on imbalanced data because the mountain of true negatives inflates it. When positives are rare, use **PR-AUC** (area under the precision-recall curve, computed by `average_precision_score`) instead. PR-AUC ignores true negatives entirely and focuses on how well you find the needles, so it drops honestly when the model is bad at the rare class.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=4,
    weights=[0.98, 0.02], random_state=42,   # 2% positive
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42,
)

plain = LogisticRegression(max_iter=1000).fit(X_train, y_train)
weighted = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, y_train)

for name, model in [("plain", plain), ("balanced", weighted)]:
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    rec = classification_report(y_test, preds, output_dict=True)["1"]["recall"]
    print(f"{name:9s} recall(+): {rec:.2f}  "
          f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}  "
          f"PR-AUC: {average_precision_score(y_test, probs):.3f}")
# output:
# plain     recall(+): 0.33  ROC-AUC: 0.951  PR-AUC: 0.646
# balanced  recall(+): 0.86  ROC-AUC: 0.951  PR-AUC: 0.640
```

Look at what happened. The plain model has a gorgeous 0.95 ROC-AUC but catches only a *third* of the positives — ROC-AUC hid that. `class_weight='balanced'` more than doubles recall on the rare class. And PR-AUC (~0.64) is far more sober than ROC-AUC (0.95) about how hard this problem actually is. Note `stratify=y` in the split — it keeps the 2% positive rate in both splits, which matters when the class is this rare.

## Calibration: are the probabilities real?

`predict_proba` returns a number between 0 and 1, and it's tempting to read 0.8 as "80% chance." Often that reading is wrong. A model can *rank* examples perfectly (great AUC) while its raw scores are systematically over- or under-confident. If you only ever threshold the score, that doesn't matter. But the moment you feed the probability into a downstream decision — expected-value math, a risk budget, a price — miscalibration silently corrupts everything built on top.

A **reliability curve** (calibration curve) checks this: bucket predictions by their claimed probability, then plot claimed-probability against the actual observed fraction of positives in each bucket. Perfect calibration is the diagonal line. A curve sagging below means the model is over-confident; bowing above means under-confident.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

frac_pos, mean_pred = calibration_curve(y_test, plain.predict_proba(X_test)[:, 1], n_bins=10)
print("claimed:", mean_pred.round(2))
print("actual :", frac_pos.round(2))   # gaps from 'claimed' = miscalibration

# Fix it: wrap the model in a calibrator (fits a correction via cross-validation)
calibrated = CalibratedClassifierCV(LogisticRegression(max_iter=1000), method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
```

`CalibratedClassifierCV` learns a correction mapping from raw scores to honest probabilities. Use `method='sigmoid'` (Platt scaling) when you have little data, or `method='isotonic'` for a more flexible fit when you have plenty. Calibration matters most for tree ensembles and SVMs, which are notoriously over-confident; well-tuned logistic regression is usually close to calibrated already. Rule of thumb: if a human or another system consumes the *probability itself*, calibrate. If you only consume the *decision*, don't bother.

## Key takeaways

- Regression metrics differ in how they punish errors: MAE is outlier-robust and readable; MSE/RMSE punish large misses hard (RMSE ≫ MAE signals a few big ones); R² is a scale-free "better than the mean?" check; MAPE is relative but explodes near zero.
- Diagnosing imbalance isn't fixing it. The levers are: `class_weight='balanced'`, resampling (under/over/SMOTE — train split only), threshold moving, and reporting PR-AUC instead of ROC-AUC when positives are rare.
- ROC-AUC can look excellent while recall on the rare class is terrible; always check recall and PR-AUC together on imbalanced data.
- A confident-looking `predict_proba` is not automatically a trustworthy probability. Check a reliability curve and calibrate with `CalibratedClassifierCV` whenever a downstream decision consumes the probability itself.

## Try it

Take the imbalanced synthetic dataset above and swap `LogisticRegression` for `RandomForestClassifier`. Train it plain, then again with `class_weight='balanced'`. Print recall on the positive class, ROC-AUC, and PR-AUC for both. Then plot a reliability curve for the plain random forest and one for a `CalibratedClassifierCV`-wrapped version. Write two sentences: one on which lever moved recall the most, and one on whether the random forest needed calibration more than logistic regression did.
