# 07 — Cross-Validation

You've been holding out a validation set to check your model. But there's a hidden problem with a single split: the number you get depends on *which* examples happened to land in that split. Get an easy validation set by luck and your model looks great; get a hard one and it looks weak. With a small dataset, that luck-of-the-draw noise can be larger than the real differences between models. Cross-validation is how you get a trustworthy estimate instead of a lucky or unlucky one.

## The problem with one split

Imagine you split off 20% for validation, train, and get 88% accuracy. Is your model really 88% good? Maybe. But if you'd made a different random split, you might have gotten 84% or 91%. You're basing an important decision — which model to ship — on a single roll of the dice.

The problem gets worse the smaller your data. With 200 examples, a 20% validation set is just 40 examples; a handful of tricky ones can swing the score wildly. You need a way to use your limited data more efficiently and to measure how *stable* your estimate is.

## K-fold cross-validation

The idea is elegant. Instead of one split, split the data into **k** equal chunks called **folds** (k=5 is common). Then train and evaluate k times. Each round, one fold is held out as the validation set and the other k-1 folds are used for training. Every fold gets its turn as the validation set exactly once.

With k=5, you train 5 models. Fold 1 validates on chunk 1 and trains on chunks 2–5; fold 2 validates on chunk 2 and trains on 1,3,4,5; and so on. You end up with 5 scores.

The payoffs are big. First, **every example is used for validation exactly once and for training k-1 times**, so you squeeze the most out of limited data. Second, you get 5 scores instead of 1, so you can report their **average** (a more stable estimate) *and* their **spread** (how much the estimate wobbles).

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=5)

print("Scores per fold:", scores)
print(f"Mean: {scores.mean():.3f}  Std: {scores.std():.3f}")
```

This prints something like `[0.86 0.91 0.88 0.84 0.90]`, mean `0.878`, std `0.026`. That standard deviation is gold: it tells you the honest uncertainty in your estimate. If model A scores 0.878 ± 0.026 and model B scores 0.882 ± 0.030, those error bars overlap heavily — the difference is probably noise, not a real improvement. Without cross-validation you'd never have known.

## Stratified folds for classification

For classification, especially with imbalanced classes, plain random folds can accidentally put too few (or zero) of a rare class in some folds. **Stratified** k-fold keeps each fold's class proportions the same as the full dataset. scikit-learn's `cross_val_score` does this automatically for classifiers, but you can be explicit:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv)
```

For time-series data, don't shuffle — use `TimeSeriesSplit` so you always train on the past and validate on the future, never the reverse.

## Where the test set still fits

Cross-validation replaces the *validation* set, not the test set. The workflow becomes:

1. Split off a **test set** and lock it away.
2. On the remaining data, use cross-validation to compare models and tune settings.
3. Pick your winner, retrain it on all the non-test data.
4. Evaluate **once** on the test set for your final honest number.

The test set is still your uncontaminated final exam. Cross-validation just makes step 2 far more reliable than a single validation split.

## Tuning hyperparameters with cross-validation

The biggest everyday use of cross-validation is choosing **hyperparameters** — the knobs you set before training, like a tree's `max_depth` or Ridge's `alpha`. You try a grid of values and use cross-validation to score each, picking the setting with the best average CV score. scikit-learn automates the whole search:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [3, 5, 10, None],
              "n_estimators": [100, 200]}

search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5
)
search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best CV score:", search.best_score_)
```

`GridSearchCV` runs cross-validation for every combination in the grid — here 4 depths × 2 tree counts × 5 folds = 40 model fits — and reports the best. One caution: the more combinations you search, the more chances you have to get a lucky-looking CV score, so the winner is slightly optimistic. That's precisely why you keep the untouched test set for the final word.

## A note on cost

Cross-validation trains k models instead of 1, and a grid search multiplies that by the number of combinations. It can get expensive on large data or slow models. When training is cheap, cross-validate freely. When it's costly, use a smaller k (say 3) or a single well-sized validation split, and accept a bit more uncertainty in exchange for speed.

## Key takeaways

- A single validation split gives a noisy estimate that depends on luck, especially on small data.
- K-fold cross-validation rotates the validation fold k times, using every example for both training and validation.
- It yields an average score (more stable) and a standard deviation (your honest uncertainty) — use the spread to judge whether a difference is real.
- Use stratified folds for classification and time-ordered folds for time series.
- Cross-validation replaces the validation set for tuning; the test set stays sealed for the final measurement.

## Try it

Run `cross_val_score` with `cv=5` on a model, then again with `cv=10`. Compare the mean and standard deviation. Then take a single `train_test_split` and score the same model on it three times using three different `random_state` values. How much does that single-split score bounce around compared to the cross-validated mean? Which would you trust to decide between two models?
