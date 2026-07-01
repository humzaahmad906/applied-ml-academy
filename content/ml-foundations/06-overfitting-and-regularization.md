# 06 — Overfitting and Regularization

Every model faces the same fundamental danger: it can learn the wrong lesson. It can memorize the quirks and noise of the training data instead of the real, general pattern. This is **overfitting**, and it's why a model can score 99% during development and then embarrass you in the real world. Understanding it — and the tools to fight it — separates people who can build reliable models from people who just get lucky.

## The student analogy

Picture two students preparing for an exam.

The first memorizes every practice question word-for-word without understanding the concepts. They ace the practice test. On the real exam, with new questions, they fail. This is **overfitting**: great on training data, poor on new data.

The second barely studies and only learns "there will be some math." They do mediocre on the practice test and mediocre on the real one. This is **underfitting**: the model is too simple to capture the pattern, so it's bad everywhere.

The student you want understands the underlying concepts. They do well on practice *and* on the real exam, and the two scores are close. That gap — between training performance and new-data performance — is your single most useful diagnostic.

## Diagnosing it: the train/test gap

You catch overfitting by comparing performance on training data versus held-out data:

```python
print("Train:", model.score(X_train, y_train))
print("Test: ", model.score(X_test, y_test))
```

- **Overfitting:** train score high, test score much lower. Big gap. The model memorized.
- **Underfitting:** both scores low. The model is too simple.
- **Good fit:** both scores high and close together.

A train accuracy of 0.99 and a test accuracy of 0.72 is a flashing red light. The model learned the training set's noise. Always print both — the training score alone tells you almost nothing.

## Bias and variance

There's a formal way to name these two failure modes.

**Bias** is error from wrong assumptions — the model is too simple to capture the truth. High bias causes underfitting. A straight line trying to fit a curve has high bias: it's systematically off no matter how much data you give it.

**Variance** is error from being too sensitive to the specific training data. High variance causes overfitting. A wildly flexible model contorts itself to fit every training point, so it changes dramatically if you swap in a different training sample.

The **bias-variance tradeoff** is the tension between them. Make a model more complex and bias falls but variance rises. Make it simpler and variance falls but bias rises. Your job is to find the sweet spot where total error is lowest — complex enough to capture the real pattern, simple enough not to chase noise.

## Regularization: a penalty for complexity

**Regularization** is the main tool for pulling an overfit model back toward simplicity. The idea: add a penalty to the model's training objective that grows as the model's weights get large. Now the model can't just crank up weights to fit every point — it pays a price for complexity, so it only uses complexity that genuinely helps.

For linear and logistic models, two flavors dominate:

- **L2 regularization (Ridge):** penalizes the *sum of squared weights*. It shrinks all weights toward zero, keeping them small and smooth but rarely exactly zero.
- **L1 regularization (Lasso):** penalizes the *sum of absolute weights*. It can drive some weights to *exactly* zero, effectively removing those features. This makes L1 a handy tool for automatic feature selection.

```python
from sklearn.linear_model import Ridge, Lasso

# alpha controls the strength of the penalty
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

print("Lasso zeroed-out features:", (lasso.coef_ == 0).sum())
```

The knob is `alpha` (in logistic regression it's `C`, which is the *inverse* — smaller `C` means stronger regularization). Higher `alpha` means a heavier penalty and a simpler model. Set it too high and you'll swing all the way into underfitting. You don't guess this value — you tune it, which is exactly what cross-validation (the next module) is for.

## Regularization beyond linear models

The same principle appears everywhere, just under different names:

- **Trees:** limit `max_depth`, require a minimum number of samples per leaf (`min_samples_leaf`), or prune. All of these stop the tree from carving out branches for individual noisy points.
- **Ensembles:** in gradient boosting, a lower `learning_rate` with more trees, plus subsampling, acts as regularization.
- **More data** is the ultimate regularizer. A model can't memorize what it can't fit; the more varied examples it sees, the harder it is to overfit and the more it's forced to learn the real pattern.

The unifying idea across all of these: **constrain the model's freedom so it captures signal, not noise.**

## Key takeaways

- Overfitting is memorizing training noise (high train score, low test score); underfitting is being too simple (both scores low).
- Diagnose by comparing train and test performance — a large gap means overfitting.
- Bias is error from oversimplification; variance is error from over-sensitivity; you trade one against the other.
- Regularization adds a penalty for complexity: L2/Ridge shrinks weights, L1/Lasso can zero them out and select features.
- The same principle appears as depth limits in trees, learning rate in boosting, and — most powerfully — more data.

## Try it

Fit a `DecisionTreeClassifier` at several `max_depth` values (say 1, 3, 5, 10, and unlimited). For each, print both the training and test accuracy. Plot or eyeball the two curves as depth increases. At what depth does the test score stop improving while the train score keeps climbing? That crossover point is overfitting starting.
