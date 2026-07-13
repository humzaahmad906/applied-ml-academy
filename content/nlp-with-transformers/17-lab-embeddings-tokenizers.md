# 17 — Lab 1: Embeddings and Tokenizers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/nlp-with-transformers/notebooks/17-lab-embeddings-tokenizers.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup.

You will train word2vec skip-gram with negative sampling from scratch in PyTorch on a real corpus, check that the vectors capture meaning (nearest neighbors, analogies, a 2-D map), measure a gender bias baked into those vectors with a WEAT-lite test, then switch to subword tokenizers: train BPE and Unigram with the `tokenizers` library, compare how many tokens each language costs (fertility), and open up a production tokenizer (Qwen2.5) to see the same ideas shipped at scale. Everything here is CPU-only.

## Setup

Runs on free Colab (CPU is fine — no GPU needed). End-to-end runtime is about 4-6 minutes on Colab CPU, dominated by the ~3 epochs of skip-gram training. Peak memory stays under ~1.5 GB.

Seeds (`random`, `numpy`, `torch`) are all set to 42. The corpus is a 2.5M-token slice of **text8** (cleaned Wikipedia), which downloads in a few seconds.

```python
!pip install -q torch numpy tokenizers transformers
```

### Imports, seeds, and corpus

text8 is a single lowercase string of space-separated words with no punctuation — ideal for a from-scratch word2vec because you skip all the cleaning. We take the first 2.5M tokens to keep training under a few minutes.

```python
import time, zipfile, urllib.request, collections, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(42); np.random.seed(42); torch.manual_seed(42)

urllib.request.urlretrieve("http://mattmahoney.net/dc/text8.zip", "text8.zip")
with zipfile.ZipFile("text8.zip") as z:
    text = z.read("text8").decode("utf-8")

N_TOKENS = 2_500_000
words = text.split()[:N_TOKENS]
print(f"tokens={len(words):,}  sample: {' '.join(words[:12])}")
```

## Part A — Skip-gram with negative sampling, from scratch

**The model.** Skip-gram (Mikolov et al., 2013) learns two vectors per word: an *input* vector for when the word is the center, and an *output* vector for when it is a context word. Given a center word `c` and a true context word `o`, the training objective pushes their dot product up while pushing the center's dot product with `K` random *negative* words down. This is negative sampling — the loss for one (center, context) pair is

$$
\mathcal{L} = -\log\sigma(v_o^\top v_c) - \sum_{k=1}^{K}\log\sigma(-v_{n_k}^\top v_c)
$$

where $\sigma$ is the sigmoid, $v_c$ is the input (center) vector, and $v_o, v_{n_k}$ are output vectors. It is a set of $K+1$ independent binary logistic regressions: "is this word a real neighbor of the center, yes or no?" That side-steps the full softmax over the whole vocabulary, which is what makes it fast.

**Two preprocessing tricks that matter more than the architecture:**

- **Subsampling frequent words.** Words like "the" appear so often they drown the signal. Each token is kept with probability $p_{\text{keep}}(w) = (\sqrt{f(w)/t}+1)\cdot t/f(w)$ with $t=10^{-4}$, which aggressively drops the most frequent words.
- **Unigram$^{0.75}$ negatives.** Negatives are drawn from the unigram distribution raised to the 0.75 power, which lifts rare words relative to a plain frequency draw. This exponent is empirical, not derived — it just works better.

```python
MIN_COUNT = 5
counts = collections.Counter(words)
vocab = [w for w, c in counts.most_common() if c >= MIN_COUNT]
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# word2vec subsampling of frequent words (t = 1e-4)
total = sum(counts[w] for w in vocab)
freq = np.array([counts[w] / total for w in vocab])
keep = np.clip((np.sqrt(freq / 1e-4) + 1) * (1e-4 / freq), 0, 1)

rng = np.random.default_rng(42)
ids = np.array([w2i[w] for w in words if w in w2i], dtype=np.int64)
ids = ids[rng.random(len(ids)) < keep[ids]]
print(f"vocab={V:,}  training tokens after subsampling={len(ids):,}")
```

```python
# skip-gram pairs: (center, context) for offsets 1..W in both directions
W = 5
c_list, o_list = [], []
for d in range(1, W + 1):
    c_list += [ids[:-d], ids[d:]]
    o_list += [ids[d:],  ids[:-d]]
centers = np.concatenate(c_list)
contexts = np.concatenate(o_list)
order = rng.permutation(len(centers))
centers, contexts = centers[order], contexts[order]
print(f"training pairs={len(centers):,}")

# negative-sampling distribution: unigram ^ 0.75
neg_w = freq ** 0.75; neg_w /= neg_w.sum()
neg_t = torch.tensor(neg_w, dtype=torch.float)

DIM, K, BATCH, EPOCHS = 100, 5, 4096, 3
in_emb  = nn.Embedding(V, DIM)
out_emb = nn.Embedding(V, DIM)
nn.init.uniform_(in_emb.weight, -0.5 / DIM, 0.5 / DIM)
nn.init.zeros_(out_emb.weight)
opt = torch.optim.Adam(list(in_emb.parameters()) + list(out_emb.parameters()), lr=2e-3)
c_t, o_t = torch.from_numpy(centers), torch.from_numpy(contexts)

t0 = time.time()
for ep in range(EPOCHS):
    tot = nb = 0
    for s in range(0, len(c_t), BATCH):
        cb, ob = c_t[s:s+BATCH], o_t[s:s+BATCH]
        B = cb.size(0)
        negs = torch.multinomial(neg_t, B * K, replacement=True).view(B, K)
        vc, vo, vn = in_emb(cb), out_emb(ob), out_emb(negs)
        pos = (vc * vo).sum(-1)                             # <v_c, v_o>
        neg = torch.bmm(vn, vc.unsqueeze(-1)).squeeze(-1)   # <v_c, v_neg>
        loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    print(f"epoch {ep}  loss={tot/nb:.4f}  ({time.time()-t0:.0f}s)")

# unit-normalize the input vectors — these are the word embeddings
E = in_emb.weight.detach().numpy()
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
```

Loss falls from ~1.27 to ~1.09 over three epochs. It never approaches zero and it should not: negative sampling is a proxy objective, not a likelihood you can drive down. What matters is whether the *geometry* it produces is meaningful — which the next cells test directly.

## Part B — Similarity, analogies, and a 2-D map

The first sanity check is nearest neighbors by cosine similarity (already unit-normalized, so a dot product *is* cosine). If the distributional hypothesis held — words in similar contexts get similar vectors — the neighbors should be semantically related.

```python
def neighbors(word, k=8):
    q = E[w2i[word]]
    sims = E @ q
    return [(vocab[i], round(float(sims[i]), 3)) for i in np.argsort(-sims)[1:k+1]]

for w in ["king", "france", "water", "three", "music"]:
    print(f"{w:8s}", neighbors(w, 6))
```

You will see clusters like `three -> {two, four, five, six}`, `music -> {musical, pop, jazz, folk}`, `france -> {germany, italy, spain, paris}`. Nothing told the model these are numbers or countries; it fell out of co-occurrence.

**Analogies** test linear structure: is `king - man + woman` near `queen`? word2vec became famous for this. But be honest about scale — the original result used ~100B tokens. On our 2.5M-token slice you will see royal- and female-associated words near the top (queen, princess, isabella, daughter) but rarely a clean top-1 `queen`. The *direction* is right; the resolution is not.

```python
def analogy(a, b, c, k=5):
    """a is to b as c is to ? -> nearest word to (a - b + c)."""
    v = E[w2i[a]] - E[w2i[b]] + E[w2i[c]]
    v /= np.linalg.norm(v)
    sims = E @ v
    banned = {w2i[a], w2i[b], w2i[c]}
    return [(vocab[i], round(float(sims[i]), 3))
            for i in np.argsort(-sims) if i not in banned][:k]

print("king - man + woman     =>", analogy("king", "man", "woman"))
print("paris - france + italy =>", analogy("paris", "france", "italy"))
```

Finally, project a handful of words to 2-D with PCA (via SVD) to *see* the structure — countries, capitals, numbers, and royalty terms should each land in their own neighborhood.

```python
import matplotlib.pyplot as plt

sel = ["king", "queen", "man", "woman", "france", "germany", "italy", "spain",
       "paris", "london", "berlin", "rome", "water", "fire", "three", "four",
       "music", "jazz"]
sel = [w for w in sel if w in w2i]
M = E[[w2i[w] for w in sel]]
M = M - M.mean(0)
_, _, Vt = np.linalg.svd(M, full_matrices=False)   # PCA via SVD
P = M @ Vt[:2].T

plt.figure(figsize=(8, 6))
plt.scatter(P[:, 0], P[:, 1])
for (x, y), w in zip(P, sel):
    plt.annotate(w, (x + 0.01, y + 0.01), fontsize=11)
plt.title("Skip-gram embeddings, top-2 PCA")
plt.tight_layout(); plt.show()
```

**Why intrinsic eval misleads.** Analogy and similarity scores are cheap and satisfying, but they correlate weakly with how embeddings help a downstream task. A vector set can ace `king:queen` and still underperform on your classifier. Treat these checks as smoke tests, not as the metric you optimize.

## Part C — Measuring bias with a WEAT-lite test

Embeddings absorb whatever regularities are in the corpus — including social bias. The **Word Embedding Association Test** (Caliskan, Bryson, Narayanan, 2017) quantifies this. For a word `w`, define its differential association with two attribute sets `A` and `B` as the mean cosine similarity to `A` minus the mean to `B`. The effect size compares two *target* sets `X` and `Y`:

$$
d = \frac{\operatorname{mean}_{x\in X}s(x,A,B) - \operatorname{mean}_{y\in Y}s(y,A,B)}{\operatorname{std}_{w\in X\cup Y}\,s(w,A,B)}
$$

We test whether **science** terms sit closer to **male** words and **arts** terms closer to **female** words — one of the human-like biases Caliskan et al. found (they reported $d$ up to ~1.8 on web-scale embeddings).

```python
def assoc(w, A, B):
    return (np.mean([E[w2i[w]] @ E[w2i[a]] for a in A])
            - np.mean([E[w2i[w]] @ E[w2i[b]] for b in B]))

def weat_effect(X, Y, A, B):
    X = [w for w in X if w in w2i]; Y = [w for w in Y if w in w2i]
    A = [w for w in A if w in w2i]; B = [w for w in B if w in w2i]
    sx = [assoc(x, A, B) for x in X]
    sy = [assoc(y, A, B) for y in Y]
    d = (np.mean(sx) - np.mean(sy)) / np.std(sx + sy)
    return d, len(X), len(Y)

male    = ["he", "him", "his", "man", "men", "male", "father", "son", "boy"]
female  = ["she", "her", "hers", "woman", "women", "female", "mother", "daughter", "girl"]
science = ["science", "technology", "physics", "chemistry", "engineering", "math", "computation"]
arts    = ["poetry", "art", "dance", "literature", "novel", "drama", "sculpture"]

d, nx, ny = weat_effect(science, arts, male, female)
print(f"WEAT effect size d = {d:.3f}  (science/arts vs male/female, |X|={nx}, |Y|={ny})")
```

You get **d ≈ 1.2** — a large positive effect: this corpus ties science to male terms and arts to female terms, purely from co-occurrence. The number is noisier than a web-scale run (small corpus, small word sets), but the sign and magnitude reproduce the published finding. The lesson for the job: any system built on these vectors (resume screening, search ranking) inherits the bias unless you measure and mitigate it.

## Part D — Subword tokenizers: BPE vs Unigram, and fertility

Word vectors need a fixed vocabulary and choke on anything unseen. Subword tokenizers fix that by splitting text into reusable pieces. Two dominant training algorithms:

- **BPE** (Byte-Pair Encoding, Sennrich et al., 2016): start from bytes, greedily merge the most frequent adjacent pair, repeat until you hit the target vocab size. Merges are learned bottom-up.
- **Unigram** (Kudo, 2018, the SentencePiece default): start from a large candidate set and *prune* it, keeping the pieces that maximize the likelihood of the corpus under a unigram LM. Top-down.

Both run byte-level here (`ByteLevel` pre-tokenizer + seeded byte alphabet), so **no input is ever unknown** — worst case a character falls back to its raw bytes. We train each on a 1M-token slice.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

with open("tok_corpus.txt", "w") as f:
    f.write(" ".join(words[:1_000_000]))

BYTES = ByteLevel.alphabet()   # seed all 256 byte chars so nothing is OOV

def train_bpe(vocab_size=8000):
    tk = Tokenizer(BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tk.decoder = ByteLevelDecoder()
    tk.train(["tok_corpus.txt"], BpeTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]"], initial_alphabet=BYTES))
    return tk

def train_unigram(vocab_size=8000):
    tk = Tokenizer(Unigram())
    tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tk.decoder = ByteLevelDecoder()
    tk.train(["tok_corpus.txt"], UnigramTrainer(
        vocab_size=vocab_size, unk_token="[UNK]",
        special_tokens=["[UNK]"], initial_alphabet=BYTES))
    return tk

bpe = train_bpe()
uni = train_unigram()
print("trained BPE and Unigram, vocab_size=8000 each")
```

**Fertility** is tokens-per-word: how many tokens the tokenizer spends per whitespace word. Lower is better — it means more text fits in the context window and each word costs less to process. The critical question for multilingual systems: *does every language cost the same?* We compare English (in-corpus), Urdu (non-Latin script, out-of-corpus), and a code snippet.

```python
samples = {
    "english": "the government announced a new economic policy yesterday",
    "urdu":    "حکومت نے کل ایک نئی معاشی پالیسی کا اعلان کیا",
    "code":    "def add(a, b):\n    return a + b  # sum two numbers",
}

def fertility(tk, txt):
    return len(tk.encode(txt).ids) / max(1, len(txt.split()))

print(f"{'language':10s} {'words':>5s} {'BPE tok':>8s} {'BPE fert':>9s} {'Uni tok':>8s} {'Uni fert':>9s}")
for name, txt in samples.items():
    nb, nu, nw = len(bpe.encode(txt).ids), len(uni.encode(txt).ids), len(txt.split())
    print(f"{name:10s} {nw:5d} {nb:8d} {fertility(bpe,txt):9.2f} {nu:8d} {fertility(uni,txt):9.2f}")

# round-trip check: byte-level fallback is lossless even for unseen scripts
rt = bpe.decode(bpe.encode(samples["urdu"]).ids)
print("\nUrdu round-trip lossless:", rt == samples["urdu"])
```

The numbers tell the multilingual story bluntly:

| language | words | BPE fertility | Unigram fertility |
|---|---|---|---|
| english | 8 | 1.25 | 1.50 |
| urdu | 10 | 8.10 | 8.10 |
| code | 11 | 2.18 | 2.36 |

English costs ~1.25 tokens/word because the tokenizer *trained* on English — whole words became single tokens. Urdu was never in the training corpus, so it shatters into raw UTF-8 bytes: **~8 tokens per word, a 6x penalty.** Same sentence, same information, 6x the cost and 6x less of it fits in context. This is the **low-resource tax** — speakers of under-represented languages pay more for the identical service. Code lands in between: whitespace, operators, and identifiers fragment more than prose. This is exactly why frontier tokenizers deliberately train on balanced multilingual + code mixtures.

## Part E — Inspecting a production tokenizer

Now open a shipped tokenizer. Qwen2.5 uses byte-level BPE with a ~151K vocab trained on a huge multilingual + code corpus. Watch the fertility numbers collapse relative to our English-only toy.

```python
from transformers import AutoTokenizer

qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print(f"Qwen2.5 vocab size: {qwen.vocab_size:,}")

print(f"\n{'language':10s} {'fertility':>9s}")
for name, txt in samples.items():
    print(f"{name:10s} {len(qwen.encode(txt)) / max(1, len(txt.split())):9.2f}")

# see the actual pieces, including how a number and whitespace tokenize
demo = "Revenue grew 12.5% to $1,240,000 in Q3."
ids = qwen.encode(demo)
print("\ntext :", demo)
print("pieces:", [qwen.decode([i]) for i in ids])
```

With a 151K multilingual vocab, Qwen gets Urdu down to ~3.0 tokens/word (vs our 8.1) and English to ~1.0 — because it trained on all of it. Look at the piece list and you will see the classic gotchas from the tokenization chapter: numbers get chopped into odd digit groups (`12`, `.5`), the `%` and `$` are their own tokens, and leading spaces are glued onto the following word (`Ġgrew`). These are the seams where models get arithmetic wrong and where prompt formatting quietly changes token counts.

Cross-link: the mechanics here are covered in [word vectors](02-word-vectors.md) and [tokenization](03-tokenization.md); this lab is the hands-on companion to both.

## What you built

- **Skip-gram with negative sampling from scratch** in PyTorch — subsampling, unigram^0.75 negatives, the $K+1$ binary-logistic loss — trained on 2.5M real tokens.
- Verified the geometry with **nearest neighbors, analogies, and a PCA map**, and named why intrinsic eval misleads.
- Quantified a **gender bias** (science/male vs arts/female) with a WEAT-lite effect size of d ≈ 1.2.
- Trained **BPE and Unigram** tokenizers, measured **fertility** across English, Urdu, and code, and saw the ~6x low-resource tax.
- Inspected **Qwen2.5**'s production tokenizer and connected its behavior to real number/whitespace failure modes.

## Exercises

1. **Window and dimension sweep.** Retrain skip-gram with `W=2` and `W=10`, and with `DIM=50` vs `DIM=200`. Which analogies improve, which degrade, and how does training time scale? Report neighbors for `king` and `france` in each setting.
2. **CBOW instead of skip-gram.** Flip the objective: predict the center word from the average of its context vectors. Reuse the negative-sampling loss. Compare neighbor quality and training speed against skip-gram on the same corpus.
3. **A second WEAT test.** Build target/attribute sets for a different association (e.g. pleasant vs unpleasant words against two name groups, or flowers vs insects). Report the effect size and comment on whether the corpus supports the association.
4. **Vocab-size vs fertility curve.** Train BPE at vocab sizes {1k, 4k, 8k, 16k, 32k} and plot English and code fertility against vocab size. Where do the curves flatten, and what does that imply about picking a vocab size?
5. **Glitch-token hunt.** In the Qwen tokenizer, find tokens whose decoded string is empty, pure whitespace, or a rare fragment. Feed a few back through `encode`/`decode` and note any that behave oddly — these are candidate under-trained "glitch" tokens.

## What interviews ask here

- **Derive the skip-gram negative-sampling loss and its gradient.** Why does it replace the full softmax, and what is the role of the $K$ negatives?
- **Why the 0.75 exponent on the negative-sampling distribution, and why subsample frequent words?** Both are empirical — say so, and explain what breaks without them.
- **BPE vs WordPiece vs Unigram** — how does each decide what to merge or keep, and when would you pick one over the others?
- **What is fertility and why does it matter for cost and multilinguality?** Explain the low-resource tax with a number.
- **How do you measure bias in an embedding space?** Define WEAT's effect size and state its limits.
- **Why do LLMs fail at arithmetic and get confused by trailing whitespace?** Tie both to tokenization behavior you can demonstrate.

## Where this shows up on the job

- Picking or training a tokenizer for a non-English or code-heavy product, and defending the vocab-size and multilingual-coverage tradeoff with fertility numbers.
- Auditing an embedding-based system (search, RecSys, resume screening) for bias before it ships.
- Debugging "the model can't count" or "my prompt costs more tokens than I expected" incidents down to tokenizer behavior.
