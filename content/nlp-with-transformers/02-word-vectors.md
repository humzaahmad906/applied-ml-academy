# 02 — Word Vectors: The Representation That Started It All

Before a model can do anything with language it has to turn discrete tokens into numbers, and *how* it
does that determines everything downstream. Represent "cat" as a one-hot vector — a 1 in one slot of a
50,000-dimensional vector, zeros everywhere else — and every word is equidistant from every other: "cat"
is as far from "dog" as it is from "parliament." No generalization is possible. **Word vectors** replaced
that with dense, low-dimensional representations where geometric closeness means semantic closeness, and
that single move made neural NLP work. The static embeddings of the 2013–2018 era have since been
absorbed into the input layers of transformers, but the *ideas* — the distributional hypothesis, the
skip-gram objective, negative sampling — are foundational, tested in interviews, and still deployed at
scale in retrieval and recommendation. This module derives them, math included.

## The distributional hypothesis

"You shall know a word by the company it keeps" (Firth, 1957). The claim: a word's meaning is captured
by the distribution of contexts it appears in. Words that occur in similar contexts — "coffee" and
"tea," "doctor" and "physician" — have similar meanings. This is not obviously true and it is not
*fully* true (it conflates synonyms with antonyms, since "hot" and "cold" share contexts), but it is
true enough to build on, and crucially it is **self-supervised**: you learn meaning from raw text with
no labels, just co-occurrence. Every representation-learning method in NLP, up to and including LLM
pretraining ([module 05](05-pretraining.md)), is a variation on this bet.

## Count vs predict

There are two ways to cash out the distributional hypothesis, and the field argued about them for years.

**Count-based** methods build a big word-context co-occurrence matrix by tallying how often each word
appears near each context word, then reduce its dimensionality — reweight with PMI (pointwise mutual
information), take an SVD, keep the top dimensions. This is the LSA / HAL lineage. It uses global corpus
statistics directly and trains fast, but the raw matrix is huge and the reweighting is hand-tuned.

**Predict-based** methods (word2vec) never build the matrix. They train a shallow network to *predict*
context from a word (or vice versa) and read the learned weights out as the embeddings. This was the
Mikolov et al. (2013) breakthrough, and it produced the famous result that vector arithmetic encodes
analogy: `king − man + woman ≈ queen`. The apparent gap between the two camps closed when Levy &
Goldberg (2014) proved that skip-gram with negative sampling is **implicitly factorizing a shifted PMI
matrix** — the predict method is doing a count method's job by another route. Understanding this
equivalence is a strong interview signal: they are two views of the same distributional statistics.

## Word2vec skip-gram

Skip-gram trains a model to predict the surrounding context words from a center word. Slide a window
over the corpus; for each center word $c$ and each context word $o$ inside the window, you want the model
to assign high probability to $o$ given $c$. Every word $w$ gets **two** vectors: $v_w$ when it acts as a
center word, and $u_w$ when it acts as a context word. (Two roles, two vectors — it simplifies the math
and the gradients; at the end you typically keep the $v$'s, or average.)

The probability of context word $o$ given center word $c$ is a softmax over the whole vocabulary $V$:

$$
P(o \mid c) = \frac{\exp\!\left(u_o^\top v_c\right)}{\sum_{w \in V} \exp\!\left(u_w^\top v_c\right)}
$$

The dot product $u_o^\top v_c$ scores how compatible the two words are; the softmax normalizes those
scores into a distribution. Training maximizes the log-probability of the true context words, i.e.
minimizes the negative log-likelihood. For a single (center, context) pair the loss is:

$$
J_{\text{softmax}} = -\log P(o \mid c)
     = -\,u_o^\top v_c + \log \sum_{w \in V} \exp\!\left(u_w^\top v_c\right)
$$

**The gradient (worth deriving).** Take the derivative with respect to the center vector $v_c$. The
first term is linear, giving $-u_o$. For the second term, differentiate the log-sum-exp:

$$
\frac{\partial}{\partial v_c}\log \sum_{w} \exp(u_w^\top v_c)
  = \sum_{w} \frac{\exp(u_w^\top v_c)}{\sum_{w'} \exp(u_{w'}^\top v_c)}\, u_w
  = \sum_{w} P(w \mid c)\, u_w
$$

So the full gradient is

$$
\frac{\partial J_{\text{softmax}}}{\partial v_c}
  = -\,u_o + \sum_{w \in V} P(w \mid c)\, u_w
  = -\Big(u_o - \mathbb{E}_{w \sim P(\cdot \mid c)}[u_w]\Big)
$$

Read it: the update pulls $v_c$ toward the *observed* context vector $u_o$ and pushes it away from the
model's *expected* context under the current distribution. It is the classic "observed minus expected"
gradient of any softmax model — and it exposes the fatal cost. That expectation sums over the **entire
vocabulary** on every single training pair. With $|V| = 100{,}000$ and billions of pairs, computing the
normalizer is hopeless. Negative sampling exists to kill this sum.

## Negative sampling

The fix (Mikolov et al., 2013): stop predicting a distribution over all words. Instead solve a much
cheaper binary problem — distinguish the real context word from a handful of random "noise" words. For
each true pair $(c, o)$, sample $K$ negative words $w_1, \dots, w_K$ (typically $K = 5$–$20$ for small
corpora, $2$–$5$ for large ones) and train a logistic classifier: label the true pair positive, the
$K$ sampled pairs negative. The per-pair objective becomes

$$
J_{\text{neg}} = -\log \sigma\!\left(u_o^\top v_c\right)
   \;-\; \sum_{k=1}^{K} \log \sigma\!\left(-\,u_{w_k}^\top v_c\right)
$$

where $\sigma(x) = 1/(1+e^{-x})$. The first term drives $\sigma(u_o^\top v_c) \to 1$ (real pairs score
high); each negative term drives $\sigma(u_{w_k}^\top v_c) \to 0$ (random pairs score low). Every term is
one dot product — no vocabulary-wide sum. Cost per pair drops from $O(|V|)$ to $O(K)$, which is the whole
game.

**The gradient (derive it, it comes up).** Use the two identities $\frac{d}{dx}\log\sigma(x) = 1-\sigma(x)$
and $\frac{d}{dx}\log\sigma(-x) = -\sigma(x)$. Differentiating $J_{\text{neg}}$ with respect to $v_c$:

$$
\frac{\partial J_{\text{neg}}}{\partial v_c}
  = -\big(1 - \sigma(u_o^\top v_c)\big)\,u_o
    \;+\; \sum_{k=1}^{K} \sigma\!\left(u_{w_k}^\top v_c\right)\,u_{w_k}
$$

The positive word is pulled in with weight $(1 - \sigma(u_o^\top v_c))$ — large when the model is wrong
about the true pair, near zero once it is confident — and each negative is pushed away with weight
$\sigma(u_{w_k}^\top v_c)$, its current (mis)predicted probability of being real. The gradients with
respect to the context vectors are symmetric:

$$
\frac{\partial J_{\text{neg}}}{\partial u_o} = -\big(1 - \sigma(u_o^\top v_c)\big)\,v_c,
\qquad
\frac{\partial J_{\text{neg}}}{\partial u_{w_k}} = \sigma\!\left(u_{w_k}^\top v_c\right)\,v_c
$$

Only $K+1$ context vectors update per pair instead of all $|V|$, which is why negative sampling trains
word2vec on billions of tokens in hours.

**The noise distribution matters.** Negatives are not sampled uniformly and not from the raw unigram
frequency $U(w)$, but from $U(w)^{3/4}$ renormalized:

$$
P_n(w) = \frac{U(w)^{3/4}}{\sum_{w'} U(w')^{3/4}}
$$

The $3/4$ power is a pure empirical hack — it damps very frequent words (so "the" isn't picked as a
negative constantly) while still oversampling common words relative to uniform. It has no derivation;
it just worked better than $1.0$ or $0.5$ in the original experiments, and it stuck. Be honest about this
in an interview: it is a tuned exponent, not a theorem. The companion trick is **subsampling frequent
words** — dropping "the," "of," "a" from the input with probability rising in their frequency — which
both speeds training and improves rare-word vectors.

## GloVe: the count-based counterpart

GloVe (Pennington et al., 2014) went back to global co-occurrence counts but framed them as a regression.
Let $X_{ij}$ be the number of times word $j$ appears in the context of word $i$. GloVe's insight is that
**ratios** of co-occurrence probabilities carry meaning: for a probe word $k$, $P(k \mid \text{ice}) /
P(k \mid \text{steam})$ is large for $k=$ "solid," small for $k=$ "gas," near 1 for "water" or "fashion."
To make dot products encode those log-ratios, GloVe fits

$$
J = \sum_{i,j} f(X_{ij})\,\big(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2
$$

a weighted least-squares regression of the dot product onto the log co-occurrence count, with bias terms.
The weighting $f(X_{ij}) = \min\!\big(1, (X_{ij}/x_{\max})^{0.75}\big)$ downweights rare, noisy pairs and
caps the influence of very frequent ones (there is that $0.75$ again). GloVe trains on the aggregated
matrix rather than streaming pairs, so it uses global statistics directly; in practice its embeddings and
skip-gram's are close in quality, and the choice was mostly taste. Both are static, and both were about
to be superseded.

## Evaluating embeddings — and why intrinsic eval misleads

Two intrinsic tests dominated the literature. **Word similarity**: correlate model cosine similarities
against human-rated pairs (WordSim-353, SimLex-999) via Spearman rank correlation. **Analogy**: solve
`a : b :: c : ?` by finding the word whose vector is nearest to $v_b - v_a + v_c$ (the `king − man +
woman ≈ queen` trick), scored by top-1 accuracy on the Google/BATS analogy sets.

Treat both with suspicion. The analogy result is partly an artifact of how the search is run — the
standard evaluation *excludes* the three input words from the candidate set, and without that exclusion
the answer is frequently just $c$ itself. Similarity benchmarks conflate genuine similarity with mere
relatedness ("car"/"road" score high but aren't similar), and high intrinsic scores routinely fail to
predict downstream task performance. The durable lesson, which generalizes to all of
[evaluation](10-evaluation.md): an intrinsic metric that is cheap to compute and easy to overfit is a
weak proxy for the thing you actually ship. Trust extrinsic evaluation — plug the embeddings into the
real task and measure the real metric.

## Bias in embeddings

Because embeddings absorb the statistics of their training text, they absorb its social biases too.
Bolukbasi et al. (2016) showed `man : computer_programmer :: woman : homemaker` falls straight out of
word2vec vectors trained on news text. The standard measurement is **WEAT** (Word Embedding Association
Test; Caliskan et al., 2017), which quantifies bias as a differential association: it compares the mean
cosine similarity of two *target* word sets (e.g. career vs family terms) to two *attribute* sets (e.g.
male vs female names), and reports an effect size and a permutation-test p-value — the same structure as
the implicit-association test from psychology. WEAT reliably recovers documented human biases from
embeddings, which is exactly the problem: **the geometry that makes embeddings useful is the geometry
that encodes the bias.** Debiasing by projecting out a "gender direction" mostly hides bias from the
specific metric rather than removing it. This matters on the job the moment embeddings feed a
consequential decision (résumé ranking, search), and it is a live thread through
[risks and safety](15-risks-and-safety.md).

## Subword embeddings: fastText

Word-level vectors have a hard failure: any word not seen in training has *no* vector, and morphology is
invisible ("run," "running," "runner" are unrelated slots). **fastText** (Bojanowski et al., 2017) fixed
both by representing a word as the sum of its character n-gram vectors — "where" becomes `<wh`, `whe`,
`her`, `ere`, `re>` plus the whole word. Now an unseen word still gets a vector from its pieces, and
morphological relatives share n-grams and land near each other. This is the same instinct that
[tokenization](03-tokenization.md) formalizes: operate below the word to handle the infinite tail. For
morphologically rich languages the gain is large.

## From static to contextual (the bridge)

The ceiling of everything above is a single vector per word type. "Bank" gets one embedding, averaging
the river and the money senses — polysemy is unrepresentable. **ELMo** (Peters et al., 2018) broke that
ceiling by making the embedding a function of the whole sentence: run a bidirectional LSTM language model
over the text and read out *contextual* vectors, so "bank" near "river" and "bank" near "loan" get
different representations. ELMo was the hinge between the static era and the transformer era — it proved
that pretrained contextual representations transfer, and within a year BERT and GPT replaced its LSTM
with a [transformer](04-transformer-architecture.md) and made contextual embeddings the default. Today
the "embedding" of a token *is* its transformer hidden state; static vectors are the historical layer
underneath.

## Why embeddings still matter in 2026

Static and sentence-level embeddings did not vanish; they moved to where dense vectors are the product.
**Retrieval** is the big one: [RAG](09-rag-agents.md) and semantic search embed queries and documents into
a shared space and rank by cosine similarity — the distributional hypothesis, industrialized. **Recommender
systems** learn item and user embeddings from co-occurrence (users who bought X bought Y) with objectives
that are word2vec in disguise; "prod2vec" is literally skip-gram over purchase sequences. **Cold-start**
leans on content embeddings to place a brand-new item with zero interaction history near similar known
items. And every classification or clustering pipeline that needs a cheap fixed-length text vector reaches
for a sentence embedding first. The frontier moved to contextual representations, but the geometry-of-meaning
idea is more deployed now than it ever was.

## What interviews ask here

- Derive the skip-gram softmax loss and its gradient with respect to the center vector. (Answer:
  $-u_o + \sum_w P(w\mid c)u_w$ — observed minus expected context vector.)
- Why negative sampling, and what does it optimize? (Replaces the $O(|V|)$ softmax with $K+1$ binary
  logistic decisions; implicitly factorizes shifted PMI.)
- What's the $3/4$ power in the noise distribution and where does it come from? (Empirical; damps frequent
  words. It is a tuned hack, not derived — say so.)
- Word2vec vs GloVe — predict vs count, and why they end up similar. (SGNS ≈ implicit PMI factorization,
  Levy & Goldberg 2014.)
- Why is intrinsic embedding evaluation (analogies, similarity) unreliable? (Overfittable proxies,
  relatedness vs similarity, artifacts in the analogy search; trust extrinsic metrics.)
- How do you measure bias in embeddings? (WEAT: differential cosine association with effect size + p-value.)

## Where this shows up on the job

- **Retrieval and RAG.** Every dense retriever is embeddings + cosine similarity; understanding what the
  geometry does (and where it fails — relatedness ≠ relevance) is daily work in search and RAG systems.
- **Recommendations and cold-start.** Product/user embeddings trained with skip-gram-style objectives
  power ranking and handle new items with no interaction history.
- **Cheap text features.** When you need a fast, private, calibrated fixed-length representation for
  clustering, dedup, or a lightweight classifier, a sentence embedding beats calling an LLM on cost and
  latency.
- **Bias and fairness reviews.** Any time embeddings feed a consequential decision, you will be asked to
  measure and mitigate the social bias they inherited from the corpus.
