# 24 — Interview Bank: Implementation Drills

The implementation round is where "I understand transformers" gets tested against "I can write
the twelve lines that make one work." These nine drills are the ones that recur: they are small
enough to finish in 20–30 minutes on a whiteboard or a shared editor, and each one hides a specific
trap that separates people who have typed the code from people who have only read about it. Every
solution below is complete, runnable, and seeded where anything is stochastic — copy it, run it,
then close the file and reproduce it from the hints. That reproduction is the actual practice.

Each drill is structured the way a real loop runs it: **the prompt** as an interviewer states it,
**hints** you can ask for (escalating, so take the least you need), the **solution**, and **what
they probe next** — because the code is rarely the end; the follow-ups are where senior signal
lives. Tensor shapes are in comments throughout. The mechanisms behind these appear in
[tokenization](03-tokenization.md), [the transformer](04-transformer-architecture.md),
[decoding](12-inference-decoding.md), [transfer tasks](06-transfer-learning-tasks.md),
[post-training](07-post-training.md), [PEFT](08-prompting-peft.md), and
[RAG](09-rag-agents.md) — reach for those if a drill exposes a gap.

---

## Drill 1 — BPE merge loop from scratch

**The prompt.** "Implement byte-pair encoding *training*: given a word-frequency dictionary, learn
`k` merges. Return the ordered merge list. Don't use a tokenizer library."

**Hints.**
1. Start each word as a tuple of characters plus an end-of-word marker so merges can't cross word
   boundaries.
2. One pass counts adjacent symbol pairs weighted by word frequency; pick the most frequent pair,
   merge it everywhere, repeat.
3. Break ties deterministically (by the pair itself), or your merge list won't reproduce across runs.

**Solution.**

```python
from collections import Counter

def pair_counts(word_freqs):
    # word_freqs: dict[tuple[str, ...], int]  -> Counter[(str, str), int]
    pairs = Counter()
    for symbols, freq in word_freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(pair, word_freqs):
    a, b = pair
    merged = a + b
    out = {}
    for symbols, freq in word_freqs.items():
        new_sym, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_sym.append(merged)   # collapse the adjacent pair
                i += 2
            else:
                new_sym.append(symbols[i])
                i += 1
        out[tuple(new_sym)] = freq
    return out

def learn_bpe(corpus, num_merges):
    # corpus: dict[str, int] word -> count
    word_freqs = {tuple(list(w) + ["</w>"]): c for w, c in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = pair_counts(word_freqs)
        if not pairs:
            break
        # max frequency; tie-break on the pair for a reproducible merge order
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        word_freqs = merge_pair(best, word_freqs)
    return merges

corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
print(learn_bpe(corpus, 5))
# [('t', '</w>'), ('s', 't</w>'), ('e', 'st</w>'), ('o', 'w'), ('l', 'ow')]
```

**What they probe next.** "Complexity?" Naive is `O(num_merges · N)` where `N` is total symbols —
each merge rescans the corpus; production trainers keep an incremental pair index so only affected
words are touched. "How do you *apply* the learned merges to new text?" Greedy: repeatedly apply the
highest-priority merge present. "Why `</w>`?" So `est` at word-end is a different token from `est`
mid-word, and so merges never span a space. "WordPiece difference?" WordPiece picks the merge that
most increases corpus likelihood, not raw frequency — that is the whole distinction.

---

## Drill 2 — Scaled dot-product attention with a causal mask

**The prompt.** "Write scaled dot-product attention for a batched, multi-head input, with a causal
mask, in plain PyTorch. No `F.scaled_dot_product_attention`. Write the softmax yourself."

**Hints.**
1. Scores are `q @ kᵀ` scaled by `1/√d_head` — the scale keeps the logits from growing with `d` and
   saturating the softmax.
2. The causal mask is lower-triangular *including* the diagonal: query `i` may see key `j` for
   `j ≤ i`.
3. Subtract the row max before `exp` or a large score overflows to `inf` → `NaN`.

**Solution.**

```python
import math, torch

def sdpa_causal(q, k, v):
    # q, k, v: (B, H, L, d_head)
    d_head = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / math.sqrt(d_head)          # (B, H, L, L)

    L = q.shape[-2]
    # True where attention is allowed: key index j <= query index i
    allowed = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~allowed, float("-inf"))

    scores = scores - scores.amax(dim=-1, keepdim=True)           # stable softmax
    weights = scores.exp()
    weights = weights / weights.sum(dim=-1, keepdim=True)         # -inf -> exp 0
    return weights @ v                                            # (B, H, L, d_head)

torch.manual_seed(42)
q = torch.randn(2, 4, 6, 16); k = torch.randn(2, 4, 6, 16); v = torch.randn(2, 4, 6, 16)
out = sdpa_causal(q, k, v)
print(out.shape)  # torch.Size([2, 4, 6, 16])
```

**What they probe next.** "Why `√d_head` and not `d_head`?" The dot product of two `d`-dim unit-ish
vectors has variance `∝ d`; dividing by `√d` renormalizes to unit scale so gradients don't vanish
into a saturated softmax. "Off-by-one on the mask?" `tril` keeps the diagonal so a token attends to
itself; `diagonal=-1` would forbid that and break the first position (all `-inf` → `NaN`). "Memory?"
The `(L, L)` score matrix is `O(L²)` — the reason FlashAttention tiles it and never materializes it.
"Extend to GQA?" `repeat_interleave` the KV heads up to the query-head count before the matmul.

---

## Drill 3 — Apply RoPE to queries and keys

**The prompt.** "Implement rotary position embeddings: build the cos/sin cache and apply it to `q`
and `k`. Explain why the relative offset falls out."

**Hints.**
1. Frequencies are `base^(-2i/d)` for `i = 0 … d/2−1`; the angle at position `t` is `t · freq`.
2. RoPE rotates pairs of dimensions. The common (GPT-NeoX/HF) layout splits the head dim in half and
   uses a `rotate_half` helper.
3. The `q·k` dot product ends up depending only on `(t_q − t_k)` — that relative offset is the point.

**Solution.**

```python
import torch

def build_rope_cache(seq_len, head_dim, base=10000.0):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (d/2,)
    t = torch.arange(seq_len).float()                       # (L,)
    freqs = torch.outer(t, inv_freq)                        # (L, d/2)
    emb = torch.cat([freqs, freqs], dim=-1)                 # (L, d)
    return emb.cos(), emb.sin()                             # each (L, d)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    # q, k: (B, H, L, d); cos, sin: (L, d)
    cos = cos[None, None, :, :]                             # (1, 1, L, d)
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

torch.manual_seed(42)
B, H, L, d = 1, 2, 8, 16
q = torch.randn(B, H, L, d); k = torch.randn(B, H, L, d)
cos, sin = build_rope_cache(L, d)
q_r, k_r = apply_rope(q, k, cos, sin)
print(q_r.shape)  # torch.Size([1, 2, 8, 16])
```

**What they probe next.** "Why does it encode *relative* position?" Rotating `q` by angle `θ·t_q` and
`k` by `θ·t_k`, the inner product of two rotated vectors depends only on `θ·(t_q − t_k)` — the same
identity as `cos(a)cos(b)+sin(a)sin(b)=cos(a−b)`. "Why not add sinusoids like the original
transformer?" RoPE injects position at *every* layer via the rotation, applies only to `q`/`k` (not
`v`), and extrapolates better. "Long context?" Position interpolation / YaRN scale the frequencies to
stretch the trained range — see [decoding](12-inference-decoding.md). "Where's the bug people ship?"
Applying RoPE after the head split is right; applying it to `v`, or forgetting to rebuild the cache
for longer sequences, are the classic errors.

---

## Drill 4 — Top-p (nucleus) sampler

**The prompt.** "Given a logits vector for one step, sample a token with nucleus sampling: keep the
smallest set of tokens whose cumulative probability exceeds `p`, renormalize, sample."

**Hints.**
1. Apply temperature to logits *before* the softmax.
2. Sort probabilities descending, take the cumulative sum, cut where it first crosses `p`.
3. Always keep the top-1 token even if it alone exceeds `p`, or a peaked distribution samples nothing.

**Solution.**

```python
import torch

def top_p_sample(logits, p=0.9, temperature=1.0, generator=None):
    # logits: (vocab,)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # cumsum-before-this-token > p  => this token is beyond the nucleus.
    # Using the shifted cumsum guarantees the top-1 token is always kept.
    remove = (cumsum - sorted_probs) > p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum()
    choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)  # (1,)
    return sorted_idx[choice]

g = torch.Generator().manual_seed(42)
logits = torch.tensor([2.0, 1.0, 0.5, 0.1, -1.0, -3.0])
print(top_p_sample(logits, p=0.9, temperature=0.8, generator=g).item())  # 0
```

**What they probe next.** "Why the shifted cumsum instead of `cumsum > p`?" The shift keeps the first
token whose *prior* cumulative mass was below `p`, i.e. it never empties the nucleus on a spiky
distribution. "top-p vs top-k?" top-k is a fixed count; top-p adapts the count to the distribution's
shape — narrow when the model is confident, wide when it's unsure. "min-p?" Keeps tokens with prob ≥
`min_p · max_prob`, which is even more shape-adaptive. "Batch it?" Same logic vectorized over a
`(B, vocab)` tensor with a `scatter` to undo the sort. "Determinism?" The `generator` argument is what
makes an eval reproducible — see [decoding](12-inference-decoding.md).

---

## Drill 5 — Beam search for a seq2seq decode

**The prompt.** "Implement beam search over a decoder. Keep `beam_width` hypotheses, expand each,
prune, stop at EOS. Apply length normalization."

**Hints.**
1. A beam is `(token_list, cumulative_logprob)`. Expand every live beam by its top-`beam_width` next
   tokens, then keep the global top-`beam_width` candidates.
2. Move finished (EOS) hypotheses aside so they stop being expanded but can still win.
3. Rank by `score / lengthᵃ` — raw log-prob sums always prefer shorter sequences.

**Solution.**

```python
import torch

def beam_search(step_fn, bos_id, eos_id, vocab, beam_width=4, max_len=20, alpha=0.6):
    # step_fn(tokens: LongTensor (t,)) -> log_probs (vocab,)
    norm = lambda toks, s: s / (len(toks) ** alpha)
    beams = [([bos_id], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for tokens, score in beams:
            log_probs = step_fn(torch.tensor(tokens))            # (vocab,)
            topv, topi = log_probs.topk(beam_width)
            for lp, idx in zip(topv.tolist(), topi.tolist()):
                seq = tokens + [idx]
                if idx == eos_id:
                    finished.append((seq, score + lp))
                else:
                    candidates.append((seq, score + lp))
        if not candidates:
            break
        candidates.sort(key=lambda c: norm(*c), reverse=True)
        beams = candidates[:beam_width]
    finished.extend(beams)
    finished.sort(key=lambda c: norm(*c), reverse=True)
    return finished[0][0]

# Toy stationary "model": a fixed log-prob table, seeded, for a runnable demo.
torch.manual_seed(42)
V = 7; EOS = 6
table = torch.log_softmax(torch.randn(V, V), dim=-1)   # next-token log-probs given last token
def step_fn(tokens):
    return table[tokens[-1].item()]

print(beam_search(step_fn, bos_id=0, eos_id=EOS, vocab=V, beam_width=3, max_len=10))
```

**What they probe next.** "Why does beam search hurt open-ended generation?" It optimizes total
log-prob, which collapses to bland, repetitive, high-probability text; sampling (top-p) is used for
open-ended, beam for constrained tasks like MT and summarization. "Length penalty derivation?" The
`length^alpha` term (GNMT, `alpha≈0.6`) counteracts the fact that every extra token adds a negative
log-prob. "Complexity?" `O(max_len · beam_width · vocab)` for the top-k each step. "Batching / KV
cache?" Real implementations expand along the batch dim and reorder the KV cache by the surviving
beam indices each step.

---

## Drill 6 — BIO decode to spans + entity-level F1

**The prompt.** "Given predicted and gold BIO tag sequences, decode each to typed spans and compute
entity-level precision, recall, and F1 — the seqeval metric, from scratch."

**Hints.**
1. A span opens on `B-X`, continues on `I-X`, closes on `O`, a new `B-`, or a type change.
2. Handle malformed sequences (an `I-X` with no matching open) — decide a policy and state it.
3. Entity-level means a true positive requires the *exact same* start, end, and type; count TP/FP/FN
   over spans, not tokens.

**Solution.**

```python
def bio_to_spans(tags):
    # tags: list[str], e.g. ["B-PER","I-PER","O","B-LOC"]
    spans, start, etype = [], None, None
    for i, tag in enumerate(tags):
        if tag == "O":
            if start is not None:
                spans.append((start, i, etype)); start = None
        elif tag.startswith("B-"):
            if start is not None:
                spans.append((start, i, etype))
            start, etype = i, tag[2:]
        elif tag.startswith("I-"):
            if start is None or tag[2:] != etype:   # illegal I-: treat as a fresh span
                if start is not None:
                    spans.append((start, i, etype))
                start, etype = i, tag[2:]
    if start is not None:
        spans.append((start, len(tags), etype))
    return spans                                    # list[(start, end_exclusive, type)]

def entity_f1(pred_seqs, gold_seqs):
    tp = fp = fn = 0
    for pred, gold in zip(pred_seqs, gold_seqs):
        p, g = set(bio_to_spans(pred)), set(bio_to_spans(gold))
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

gold = [["B-PER", "I-PER", "O", "B-LOC"]]
pred = [["B-PER", "I-PER", "O", "O"]]
print(entity_f1(pred, gold))  # (1.0, 0.5, 0.6666666666666666)
```

**What they probe next.** "Why entity-level not token-level?" Token accuracy is inflated by the `O`
majority class and gives partial credit for a half-right span; entity F1 is what the task actually
cares about. "Subword alignment?" A tokenizer splits `Washington` into pieces — you label the first
subword and set the rest to `-100` so loss ignores them, then realign before decoding; see
[transfer tasks](06-transfer-learning-tasks.md). "BIO vs BILOU?" BILOU adds explicit last/unit tags,
sometimes helping boundaries. "The malformed-`I` policy?" seqeval's default is stricter; naming your
choice and its effect on the score is the signal.

---

## Drill 7 — A LoRA linear layer (forward + merge)

**The prompt.** "Wrap a frozen `nn.Linear` with a LoRA adapter. Implement the forward pass and a
`merge()` that folds the adapter back into the base weight for inference."

**Hints.**
1. Freeze the base. The trainable update is `ΔW = (α/r)·B·A` with `A: (r, d_in)`, `B: (d_out, r)`.
2. Initialize `B` to zero so the adapter is a no-op at step 0 — training starts from the base model
   exactly.
3. Merge adds `(α/r)·B·A` into `base.weight` (shape `(d_out, d_in)`) so there is zero inference
   overhead.

**Solution.**

```python
import math, torch

class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False                       # freeze W
        d_out, d_in = base.weight.shape                   # nn.Linear weight is (out, in)
        self.r, self.scaling = r, alpha / r
        self.A = torch.nn.Parameter(torch.randn(r, d_in) / math.sqrt(d_in))  # (r, d_in)
        self.B = torch.nn.Parameter(torch.zeros(d_out, r))                   # (d_out, r), zero init
        self.dropout = torch.nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):                                 # x: (..., d_in)
        out = self.base(x)                                # (..., d_out)
        if not self.merged:
            delta = (self.dropout(x) @ self.A.t()) @ self.B.t()   # (..., r) -> (..., d_out)
            out = out + self.scaling * delta
        return out

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        self.base.weight.data += self.scaling * (self.B @ self.A)  # (d_out, d_in)
        self.merged = True

torch.manual_seed(42)
layer = LoRALinear(torch.nn.Linear(32, 8, bias=False), r=4, alpha=8)
x = torch.randn(2, 5, 32)
before = layer(x)
layer.merge()
after = layer(x)
print(torch.allclose(before, after, atol=1e-5))  # True — merge is exact
```

**What they probe next.** "Why zero-init `B` and not both?" If both were zero the gradient to `A` is
zero (product rule) — one side must be nonzero; convention is random `A`, zero `B`. "Why `α/r`
scaling?" It decouples the learning-rate-like update magnitude from the rank so you can change `r`
without silently rescaling the update; fix `α=2r` and tune `r`. "Param count?" `r·(d_in + d_out)`
per matrix vs `d_in·d_out` full. "When can't you merge?" Into a 4-bit (QLoRA) base — dequantize
first; and merging kills the ability to hot-swap adapters. See [PEFT](08-prompting-peft.md).

---

## Drill 8 — DPO loss from policy/reference log-probs

**The prompt.** "Implement the DPO loss. Assume you already have sequence log-probabilities for the
chosen and rejected responses under both the policy and the frozen reference model."

**Hints.**
1. DPO turns preference into a logistic loss on the *difference of log-ratios* between policy and
   reference.
2. `logits = β·[(logπ_c − logπ_r) − (logref_c − logref_r)]`; loss is `−logσ(logits)`.
3. Sequence log-prob = sum of per-token log-probs of the labels, with prompt tokens masked out.

**Solution.**

```python
import torch, torch.nn.functional as F

def sequence_logprob(logits, labels, mask):
    # logits: (B, T, V) shifted so position t predicts token t; labels: (B, T); mask: (B, T)
    logp = F.log_softmax(logits, dim=-1)
    tok = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # (B, T)
    return (tok * mask).sum(-1)                               # (B,) sum over response tokens

def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    # each arg: (B,) sequence log-probs (chosen/rejected under policy/reference)
    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))         # (B,)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (pi_c - ref_c).detach()            # implicit reward, for logging
    rejected_reward = beta * (pi_r - ref_r).detach()
    acc = (chosen_reward > rejected_reward).float().mean()    # preference accuracy
    return loss, acc

torch.manual_seed(42)
pi_c, pi_r = torch.randn(8), torch.randn(8)
ref_c, ref_r = torch.randn(8), torch.randn(8)
loss, acc = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
print(round(loss.item(), 4), round(acc.item(), 4))
```

**What they probe next.** "Where does this come from?" DPO reparameterizes the RLHF objective: the
optimal KL-constrained policy has a closed form whose reward is `β·log(π/π_ref)`, and substituting
that into the Bradley-Terry preference model gives this logistic loss — no reward model, no PPO. See
[post-training](07-post-training.md). "Role of `β`?" It's the implicit KL strength; small `β` lets
the policy drift further from the reference. "Why keep the reference?" It anchors the policy and
supplies the baseline that makes the log-ratio meaningful. "Failure mode?" Both chosen and rejected
log-probs can *drop* together while the margin grows — log the absolute rewards, not just the loss.
"IPO/KTO?" Variants that change the loss shape or drop the pairwise requirement.

---

## Drill 9 — Tiny dense retriever: embed, cosine top-k, MRR@10

**The prompt.** "Build a minimal dense retriever: embed queries and documents, retrieve the top-k by
cosine similarity, and evaluate with MRR@10."

**Hints.**
1. Cosine similarity = dot product of L2-normalized vectors; normalize once, then it's a matmul.
2. `topk` over the `(Q, N)` similarity matrix gives the ranked doc indices per query.
3. MRR = mean of `1/rank` of the first relevant doc (0 if it's not in the top-k).

**Solution.**

```python
import torch, torch.nn.functional as F

def cosine_topk(query_emb, doc_emb, k=10):
    # query_emb: (Q, d); doc_emb: (N, d)
    q = F.normalize(query_emb, dim=-1)
    d = F.normalize(doc_emb, dim=-1)
    sims = q @ d.t()                                   # (Q, N)
    scores, idx = sims.topk(min(k, d.shape[0]), dim=-1)
    return scores, idx                                 # idx: (Q, k) ranked doc ids

def mrr_at_k(ranked_idx, gold_idx, k=10):
    # ranked_idx: (Q, k); gold_idx: (Q,) the single relevant doc per query
    rr = 0.0
    for row, g in zip(ranked_idx.tolist(), gold_idx.tolist()):
        for rank, doc in enumerate(row[:k], start=1):
            if doc == g:
                rr += 1.0 / rank
                break
    return rr / len(gold_idx)

# In production the embeddings come from a real encoder, e.g.:
#   from sentence_transformers import SentenceTransformer
#   enc = SentenceTransformer("all-MiniLM-L6-v2")
#   doc_emb = torch.tensor(enc.encode(docs)); query_emb = torch.tensor(enc.encode(queries))
torch.manual_seed(42)
N, Q, d = 50, 5, 32
doc_emb = torch.randn(N, d)
gold = torch.randint(0, N, (Q,))
# make each query resemble its gold doc plus noise, so retrieval is non-trivial
query_emb = doc_emb[gold] + 0.5 * torch.randn(Q, d)
_, ranked = cosine_topk(query_emb, doc_emb, k=10)
print(round(mrr_at_k(ranked, gold, k=10), 4))
```

**What they probe next.** "Cosine vs dot product?" Cosine ignores magnitude — right when embeddings
aren't norm-calibrated; some models are trained for raw dot product. "Scaling to millions of docs?"
Exact `topk` is `O(N·d)` per query; swap in an ANN index (HNSW/IVF, FAISS) for sublinear search —
see [RAG](09-rag-agents.md). "MRR vs Recall@k vs nDCG?" MRR only credits the *first* hit (good for
single-answer QA); Recall@k counts any hit; nDCG weights graded relevance by rank. "Bi- vs
cross-encoder?" This is a bi-encoder (independent embeddings, fast); a cross-encoder rescores
query-doc pairs jointly for accuracy — retrieve with the bi-encoder, rerank the top-k with the
cross-encoder.

---

## How to practice

Do them cold. Read the prompt, cover everything below it, and write the solution in a blank editor —
then run it. The drills that matter are the ones where you reach for a hint: that hint marks the exact
edge of what you actually know versus what you recognize. Re-derive the three that hurt most a day
later. In the room, narrate the trap before you hit it ("I'll subtract the row max here so the softmax
doesn't overflow") — stating the failure mode you're avoiding is the clearest signal that you've
written this code before, not just read it.
