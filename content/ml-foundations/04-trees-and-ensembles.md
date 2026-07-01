# 04 — Decision Trees and Ensembles

Linear models draw straight lines. But a lot of real-world patterns aren't straight — they're full of "if this, then that" logic. Decision trees capture exactly that, and when you combine many of them into an **ensemble**, you get some of the most reliably strong models in all of classical machine learning. If you're working with tabular data (rows and columns), a tree ensemble is very often your best bet.

## A single decision tree

A decision tree is a flowchart of yes/no questions. To predict, you start at the top and follow the branches down until you reach a leaf, which holds the answer.

```
Is size > 1500 sqft?
├── No  → Is age > 30?
│         ├── Yes → predict $140k
│         └── No  → predict $210k
└── Yes → Is bedrooms > 3?
          ├── Yes → predict $460k
          └── No  → predict $350k
```

The tree *learns* these questions from data. At each step it searches every feature and every possible split point, and picks the one that best separates the examples — for classification, the split that makes the resulting groups as "pure" as possible (mostly one class); for regression, the split that most reduces the spread of the target.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))
```

Trees have real charms. They need no feature scaling. They handle non-linear patterns and feature interactions naturally. And a shallow tree is completely interpretable — you can read the rules.

But a single tree has a serious flaw: if you let it grow deep enough, it will carve out a tiny branch for every training example and memorize the data perfectly. That's textbook overfitting. Notice the `max_depth=3` above — that's a leash to stop the tree from growing too complex. A deep, unpruned tree usually performs badly on new data.

## The ensemble idea: many weak learners beat one

Here's the key insight that fixes trees: instead of trusting one tree, train *many* and combine their answers. A crowd of imperfect models, if their mistakes are somewhat independent, averages out to something far better than any single member. This is called an **ensemble**. Two dominant recipes exist.

## Random forests: average many independent trees

A **random forest** builds hundreds of trees, each trained on a random subset of the data and — at each split — allowed to consider only a random subset of the features. This forced diversity means the trees make *different* mistakes. Average their predictions (or take a majority vote for classification) and the errors cancel out.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
print(forest.score(X_test, y_test))
```

Random forests are the friendliest strong model in the toolbox. They work well out of the box, rarely overfit badly even with default settings, and barely need tuning. When in doubt, start here. The `n_estimators` (number of trees) is the main dial — more trees means more stability but slower training, with diminishing returns.

The trees are independent, so this is called a **parallel** ensemble — you could train them all at once.

## Gradient boosting: build trees that fix each other's mistakes

**Gradient boosting** takes the opposite approach. Instead of independent trees, it builds them one after another, and *each new tree is trained to correct the errors the previous trees made*. Tree 1 makes a rough prediction; tree 2 focuses on where tree 1 was wrong; tree 3 fixes what's left; and so on. This is a **sequential** ensemble.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
gb.fit(X_train, y_train)
print(gb.score(X_test, y_test))
```

When well-tuned, gradient boosting is often the single most accurate model on tabular data — it routinely wins competitions. The catch is that it's more sensitive to its settings. The `learning_rate` controls how much each tree corrects; smaller values are safer but need more trees. Push it too hard and it overfits, because it's actively chasing the training errors, including the noise.

## Random forest vs. gradient boosting

A practical way to choose:

- **Random forest** — reduces overfitting by averaging independent trees. Robust, forgiving, minimal tuning. Great default.
- **Gradient boosting** — reduces error by sequentially fixing mistakes. Often higher accuracy, but needs care with `learning_rate` and number of trees, and can overfit if pushed.

Both give you **feature importances** — a ranking of which features drove predictions — via `model.feature_importances_`. Treat these as a useful hint, not gospel: importances can be misleading when features are correlated, and they say nothing about direction or causation.

## Key takeaways

- A decision tree is a learned flowchart of yes/no questions; it handles non-linearity and interactions but overfits if grown too deep.
- Ensembles combine many trees so their individual mistakes cancel out.
- Random forests train many independent trees and average them — robust, low-maintenance, a great default.
- Gradient boosting trains trees sequentially, each fixing the last one's errors — often most accurate, but more tuning and overfitting risk.
- Trees need no feature scaling and offer feature importances, which are hints rather than proof.

## Try it

On any classification dataset, train three models: a single `DecisionTreeClassifier` with no depth limit, a `RandomForestClassifier`, and a `GradientBoostingClassifier`. For the single tree, also print its training accuracy alongside its test accuracy. What do you notice about the gap between train and test for the unpruned tree, and how do the two ensembles compare on the test set?
