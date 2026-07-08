# 07b — Unsupervised Learning and k-NN

Back in Lesson 01 we split machine learning into two camps: **supervised** learning, where every example comes with a correct answer, and **unsupervised** learning, where it doesn't. Since then we've lived entirely in the supervised world — regression, trees, and the rest all learn from labeled data. This lesson closes the loop. We'll look at unsupervised learning through its most common workhorse, **k-means clustering**, take a quick tour of **PCA** for dimensionality reduction, and then meet **k-nearest neighbors** — arguably the simplest classifier there is, and a nice bridge back to supervised territory.

## Supervised vs. unsupervised, revisited

In supervised learning you hand the model pairs: features `X` and a target `y`. It learns the mapping from one to the other. Unsupervised learning drops the `y` entirely. You give the model only `X` and ask it to *find structure on its own* — groups of similar points, a lower-dimensional shape the data lives on, unusual outliers.

There's no answer key, which changes everything. You can't compute accuracy, because there's nothing to be accurate *against*. Evaluation becomes fuzzier and more judgment-driven. Common goals include **clustering** (grouping similar examples) and **dimensionality reduction** (compressing many features into a few). We'll cover one of each.

## k-means clustering

k-means splits your data into `k` groups, where you pick `k` in advance. Each group is summarized by its **centroid** — the mean of the points assigned to it. The algorithm is a simple two-step loop:

1. **Assign** each point to the nearest centroid.
2. **Update** each centroid to the mean of the points now assigned to it.

Repeat until assignments stop changing. That's it. You start with `k` centroids placed somewhere, and this back-and-forth quickly settles into stable clusters. What it's really minimizing is **inertia** — the total squared distance from each point to its centroid.

### Choosing k

Since you have to name `k` up front, how do you pick a good value? Two standard tools:

- **The elbow method.** Run k-means for a range of `k` values and plot inertia against `k`. Inertia always drops as `k` grows (more centroids, tighter clusters), but at some point the gains flatten out. That bend — the "elbow" — is a reasonable choice.
- **Silhouette score.** For each point, this compares how close it is to its own cluster versus the nearest *other* cluster. It ranges from -1 to 1; higher is better. Averaged over all points, it gives a single number per `k`, and you pick the `k` that maximizes it.

Neither is gospel. They're guides, and often the right `k` comes from domain knowledge (you know there should be roughly three customer segments) as much as from a plot.

### Why scaling matters

k-means lives and dies by distance, and distance is dominated by whichever feature has the largest numeric range. If one feature is measured in dollars (0–100,000) and another in years (0–50), the dollar feature completely drowns out the year feature — the clusters will effectively ignore it. **Always standardize your features before k-means** so each one contributes on equal footing. `StandardScaler` (from Lesson 03) is the usual choice.

### Limitations

k-means is fast and intuitive, but it makes strong assumptions:

- It assumes clusters are **roughly round and similarly sized**. Long, stringy, or nested shapes confuse it.
- It's **sensitive to initialization** — where the centroids start affects where they end up, and a bad start can land in a poor solution. The fix is **k-means++**, a smarter initialization that spreads the starting centroids out. It's the default in scikit-learn, and running the whole thing a few times with different seeds (`n_init`) and keeping the best result guards against unlucky starts.
- You must choose `k` yourself.

### A worked example

Let's cluster a synthetic dataset with obvious blobs.

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Four true clusters, but pretend we don't know that
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

X_scaled = StandardScaler().fit_transform(X)

# Try a few values of k and compare silhouette scores
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(k, round(silhouette_score(X_scaled, labels), 3))
```

The silhouette score should peak at `k=4`, matching the four blobs we generated. Once you've settled on `k`, the fitted model gives you `km.labels_` (the cluster each point landed in) and `km.cluster_centers_` (the centroids). To cluster new points later, call `km.predict(X_new)`.

## PCA: fewer dimensions, same story

Real datasets often have dozens or hundreds of features, many of them correlated and redundant. **Principal Component Analysis (PCA)** compresses them into a handful of new features while keeping as much of the information as possible.

The intuition: PCA finds the **directions of most variance** in your data. The first principal component is the single direction along which the points are most spread out; the second is the next-most-spread direction perpendicular to the first; and so on. Projecting your data onto the first few components keeps the bulk of the structure while throwing away the low-variance noise. If you took the linear-algebra course, this is exactly the **eigenvector / SVD** idea in action — the principal components are the top eigenvectors of the data's covariance matrix, and the amount of variance each one captures is its eigenvalue.

Two everyday uses:

- **Visualization** — squash high-dimensional data down to 2 or 3 components so you can plot it and eyeball structure.
- **Compression / speed** — feed a smaller, denser feature set into a downstream model, which trains faster and can generalize better when the original features were noisy.

```python
from sklearn.decomposition import PCA

# X_scaled from before — PCA also expects scaled features
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(X_2d.shape)                        # (500, 2)
print(pca.explained_variance_ratio_)     # variance kept by each component
```

`explained_variance_ratio_` tells you the fraction of total variance each component preserves — sum it to see how much you kept. Like k-means, PCA is distance/variance based, so **scale your features first** or the highest-range feature will hijack the components.

## k-nearest neighbors (k-NN)

Now back to supervised learning, with the simplest classifier imaginable. k-NN doesn't really "train" at all — it just memorizes the training set. To predict the label of a new point, it finds the `k` closest training points and takes a **majority vote** of their labels. That's the whole algorithm.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```

### The role of k

`k` is a classic **bias/variance** dial:

- **Small k** (e.g. 1) makes the model very flexible — the prediction follows individual neighbors, so the decision boundary is jagged and it overfits, chasing noise.
- **Large k** smooths the boundary by averaging over many neighbors, but push it too far and it underfits, blurring the real distinctions between classes.

There's no formula for the best `k`; find it with cross-validation. Odd values help avoid ties in binary classification.

### Why k-NN needs scaling

Just like k-means, k-NN is built entirely on distance, so a feature with a large range dominates the "nearest" calculation. **Standardize first**, every time.

### The curse of dimensionality

k-NN's biggest weakness shows up when you have many features. In high dimensions, distances stop being meaningful — every point ends up roughly equidistant from every other point, so "nearest" loses its meaning and the voting degrades to noise. This is the **curse of dimensionality**, and it's one reason PCA and k-NN are natural partners: reduce the dimensions first, then let k-NN vote in the compressed space. k-NN also gets slow at prediction time on large datasets, since every prediction scans the training set for neighbors.

## When to reach for each

- **k-means** — you have unlabeled data and want to discover groups: customer segments, document topics, image color palettes. Start here for clustering, but remember the round-cluster assumption.
- **PCA** — you have too many features and want to visualize them, speed up a downstream model, or strip redundant noise. A preprocessing step, not a final model.
- **k-NN** — you have labeled data, a modest number of features, and want a dead-simple baseline. It's a great sanity check and shines on low-dimensional problems, but it fades as features and data volume grow. For serious tabular work, the tree ensembles from Lesson 04 usually win.

## Key takeaways

- Unsupervised learning finds structure in unlabeled data; there's no answer key, so evaluation is judgment-driven.
- k-means groups data into `k` clusters by alternating assign-to-nearest-centroid and recompute-centroid; choose `k` with the elbow method or silhouette score.
- k-means, PCA, and k-NN are all distance/variance based, so **scale your features first** — always.
- k-means assumes round, similar-sized clusters and is sensitive to initialization; k-means++ (the default) mitigates the second issue.
- PCA compresses features into the directions of most variance — the eigenvector/SVD idea — for visualization and speed.
- k-NN predicts by majority vote of the nearest neighbors; small `k` overfits, large `k` underfits, and it struggles in high dimensions (the curse of dimensionality).

## Try it

Generate a synthetic dataset with `make_blobs(n_samples=500, centers=4, random_state=42)`. First, run k-means for `k` from 2 to 7 and confirm the silhouette score peaks at 4. Then treat the blob assignments as labels and train a `KNeighborsClassifier` on a train/test split — once on the raw features and once after `StandardScaler` — and compare the test accuracy. How much did scaling matter, and what happens to the k-NN accuracy as you crank `n_neighbors` up toward the size of the training set?
