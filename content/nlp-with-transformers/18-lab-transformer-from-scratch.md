# 18 — Lab 2: Transformer From Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/nlp-with-transformers/notebooks/18-lab-transformer-from-scratch.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup.

You read the mechanism in [transformer architecture](04-transformer-architecture.md). Now you build it and prove it works. This lab implements scaled dot-product attention, multi-head attention with RoPE, and a pre-norm SwiGLU decoder block in plain PyTorch, then checks the attention against `torch.nn.functional.scaled_dot_product_attention` to a fixed numerical tolerance. You stack the block into a tiny decoder LM, train it on a slice of TinyStories, watch the loss curve, sample at several temperatures, and read the attention maps to find an induction-style head — the copy circuit behind in-context learning.

Tokenization is not the subject here; that was [lab 17](17-lab-embeddings-tokenizers.md), where you trained BPE. To keep the focus on the model, this lab uses raw UTF-8 bytes as tokens: a fixed vocabulary of 256, no training, no vocabulary file. A byte-level LM is a real (if small) language model, and it lets every character of TinyStories through with zero unknown tokens.

## Setup

```bash
pip install -q torch datasets matplotlib
```

Expected runtime on a free Colab **T4**: about **4-6 minutes** end to end, most of it the 2000-step training run. Peak GPU memory is under **1 GB** — the model is ~3.3M parameters. It also runs on CPU, where training takes roughly 15-20 minutes. Set `Runtime → Change runtime type → T4 GPU` before you start.

Seeds are set to 42 on `random`, `numpy`, and `torch` at the top of the first cell so every number below is reproducible.

```python
import math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
```

## Part A — Scaled Dot-Product Attention

Attention is a soft dictionary lookup: each query row scores every key row, the scores become a probability distribution, and the output is that distribution's weighted average of the value rows. The score is divided by $\sqrt{D_h}$ so that the dot product of two random $D_h$-dimensional vectors — whose variance grows linearly with $D_h$ — stays order-1 and the softmax does not saturate.

The causal mask forbids a position from attending to positions to its right. We add $-\infty$ to the upper triangle (excluding the diagonal) before the softmax, so those weights become exactly zero.

```python
def sdpa(q, k, v, causal=True, return_weights=False):
    """q, k, v: [B, H, T, D] -> out [B, H, T, D], weights [B, H, T, T]"""
    B, H, T, D = q.shape
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)      # [B, H, T, T]
    if causal:
        mask = torch.full((T, T), float("-inf"), device=q.device).triu(1)
        scores = scores + mask
    w = scores.softmax(-1)
    out = w @ v
    return (out, w) if return_weights else out
```

Now verify it. `F.scaled_dot_product_attention` is PyTorch's fused, FlashAttention-backed implementation with the same math and the same $1/\sqrt{D_h}$ default scale. If our version is correct, the outputs should agree to floating-point noise. The standard check for a hand-rolled kernel is the **max absolute difference** on a fixed random batch.

```python
B, H, T, D = 2, 4, 16, 32
torch.manual_seed(42)
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
mine = sdpa(q, k, v, causal=True)
ref  = F.scaled_dot_product_attention(q, k, v, is_causal=True)
max_abs_diff = (mine - ref).abs().max().item()
print(f"max abs diff vs F.sdpa: {max_abs_diff:.2e}")
assert max_abs_diff < 1e-5, "attention does not match the reference"
# and the mask actually holds: no weight above the diagonal
_, w = sdpa(q, k, v, causal=True, return_weights=True)
assert w.triu(1).abs().max() < 1e-6
print("causal mask verified: upper triangle is exactly zero")
```

Expect a max abs diff around `1e-7` — pure float32 rounding. That single number is the whole point: it is the difference between "I think my attention is right" and "my attention is right"; keep it as a unit test whenever you touch the kernel. Note that `F.scaled_dot_product_attention` materializes no `[B, H, T, T]` matrix — it is $O(T)$ memory, ours is $O(T^2)$ — which is why real models call it and we only hand-roll `sdpa` to inspect the weights, which we need for the induction analysis.

## Part B — RoPE and Multi-Head Attention

Multi-head attention runs `H` heads in parallel on `d_model / H`-dimensional slices, so different heads can specialize (one tracks the previous token, another the sentence subject), then concatenates and projects the results. Position information comes from **Rotary Position Embeddings**: instead of adding a position vector, RoPE rotates each 2D pair of query and key features by an angle proportional to absolute position. Because a dot product between two rotated vectors depends only on the *difference* of their angles, the attention score between positions $i$ and $j$ ends up depending on $i - j$ — relative position falls out for free and extrapolates past the training length better than learned absolute embeddings. RoPE is applied to Q and K only, never to V: it is about *where* things attend, not *what* gets copied.

```python
def rope_tables(head_dim, max_len, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t   = torch.arange(max_len, dtype=torch.float32)
    f   = torch.outer(t, inv)               # [max_len, head_dim // 2]
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):
    """x: [B, T, H, D]; cos/sin: [max_len, D // 2]"""
    B, T, H, D = x.shape
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    c = cos[:T].unsqueeze(0).unsqueeze(2)    # [1, T, 1, D // 2]
    s = sin[:T].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

class CausalMHA(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.H = n_heads; self.Dh = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        c, s = rope_tables(self.Dh, max_len)
        self.register_buffer("rc", c); self.register_buffer("rs", s)
        self.last_attn = None                # cache for inspection
    def forward(self, x, store_attn=False):
        B, T, _ = x.shape
        q = apply_rope(self.Wq(x).view(B, T, self.H, self.Dh), self.rc, self.rs).transpose(1, 2)
        k = apply_rope(self.Wk(x).view(B, T, self.H, self.Dh), self.rc, self.rs).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.H, self.Dh).transpose(1, 2)
        if store_attn:
            out, w = sdpa(q, k, v, causal=True, return_weights=True)
            self.last_attn = w.detach()
        else:
            out = sdpa(q, k, v, causal=True)
        return self.Wo(out.transpose(1, 2).contiguous().view(B, T, -1))
```

The `store_attn` flag is a teaching convenience — off during training, on during analysis to cache the `[B, H, T, T]` maps. Real interpretability tooling uses forward hooks for the same purpose.

## Part C — Pre-Norm SwiGLU Decoder Block and the Tiny LM

The block is the modern decoder recipe: **pre-norm** (normalize the input to each sublayer, add the residual after), **RMSNorm** instead of LayerNorm (no mean subtraction, one scale parameter, cheaper and empirically just as good), and a **SwiGLU** feed-forward network. SwiGLU replaces the ReLU MLP with a gated unit — one linear branch is passed through SiLU and multiplies a second linear branch — which is why its hidden width is set to $\tfrac{8}{3} d_{model}$ (rounded to a multiple of 64): the gating adds a third weight matrix, so you shrink the width to keep the parameter count comparable to a plain $4 d_{model}$ MLP.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return self.w * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = (int(d_model * 8 / 3) + 63) // 64 * 64
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x): return self.W2(F.silu(self.W1(x)) * self.W3(x))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.n1 = RMSNorm(d_model); self.attn = CausalMHA(d_model, n_heads, max_len)
        self.n2 = RMSNorm(d_model); self.ffn  = SwiGLUFFN(d_model)
    def forward(self, x, store_attn=False):
        x = x + self.attn(self.n1(x), store_attn=store_attn)   # pre-norm + residual
        return x + self.ffn(self.n2(x))

class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, max_len=256):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, max_len) for _ in range(n_layers)])
        self.norm   = RMSNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.embed.weight                   # weight tying
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, ids, store_attn=False):
        x = self.embed(ids)
        for blk in self.blocks: x = blk(x, store_attn=store_attn)
        return self.head(self.norm(x))                         # [B, T, vocab_size]
```

The `std=0.02` initialization matters more than it looks. With it, the untrained model's cross-entropy on 256 classes starts at $\ln 256 \approx 5.55$ — exactly the entropy of a uniform guess, which is what a sane initialization should predict. A first-step loss far above that means the logits are blowing up and you are fighting the optimizer from step one. Weight tying (`head.weight = embed.weight`) shares the input and output token matrices, saving parameters and usually helping small models.

## Part D — Data: A TinyStories Slice

TinyStories (Eldan and Li, 2023) is a corpus of very simple children's stories generated to use only words a young child knows. It is the standard toy corpus for exactly this exercise: small models learn to produce grammatical, on-topic English from it in minutes. We stream a few thousand stories so nothing large is downloaded, flatten them to a single byte tensor, and hold out the last 10% for validation.

```python
from datasets import load_dataset
N_STORIES = 4000
stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
texts = []
for i, ex in enumerate(stream):
    texts.append(ex["text"])
    if i + 1 >= N_STORIES: break
raw   = ("\n".join(texts)).encode("utf-8")            # bytes -> token ids 0..255
data  = torch.tensor(list(raw), dtype=torch.long)
n     = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
VOCAB = 256
print(f"{len(data):,} tokens  |  train {len(train_data):,}  val {len(val_data):,}")
BLOCK = 256
def get_batch(split, bs=32):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - BLOCK - 1, (bs,))
    x  = torch.stack([d[i:i + BLOCK]         for i in ix])
    y  = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])
    return x.to(device), y.to(device)
```

Each training example is a length-`BLOCK` window; the target is the same window shifted one position, because a language model predicts the next token at every position at once. This is the shift that makes the causal mask necessary: without it, position $t$ could peek at its own answer.

## Part E — Training and the Loss Curve

Standard next-token training: cross-entropy between the logits and the shifted targets, AdamW, 2000 steps. Every 100 steps we estimate the validation loss on a few batches so we can plot both curves and see whether the model is generalizing or memorizing.

```python
model = TinyLM(VOCAB, d_model=256, n_heads=4, n_layers=4, max_len=BLOCK).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

@torch.no_grad()
def eval_loss(split, iters=20):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(split)
        logits = model(x)
        tot += F.cross_entropy(logits.view(-1, VOCAB), y.view(-1)).item()
    model.train()
    return tot / iters

STEPS = 2000
hist = {"step": [], "train": [], "val": []}
for step in range(STEPS + 1):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        v = eval_loss("val")
        hist["step"].append(step); hist["train"].append(loss.item()); hist["val"].append(v)
        print(f"step {step:4d}  train {loss.item():.3f}  val {v:.3f}")
```

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(hist["step"], hist["train"], label="train")
plt.plot(hist["step"], hist["val"],   label="val")
plt.xlabel("step"); plt.ylabel("cross-entropy (nats/byte)")
plt.title("TinyLM training on TinyStories"); plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

The curve should fall from about 5.55 to roughly 1.3-1.5 nats/byte and the train and val lines should stay close — at this size and this many steps the model is nowhere near memorizing 3M+ bytes, so overfitting is not the failure mode. If the loss plateaus above ~2.5, the usual culprits are too-low a learning rate or too-short training, not the architecture. Cross-entropy in nats/byte is the byte-level analog of perplexity; $e^{\text{loss}}$ is the effective branching factor over the 256 possible next bytes.

## Part F — Sampling at Different Temperatures

Generation is autoregressive: feed the prompt, take the logits at the last position, sample one token, append it, repeat. **Temperature** divides the logits before the softmax. Low temperature sharpens the distribution toward the argmax (safe, repetitive); high temperature flattens it (varied, error-prone). Temperature 0 is pure greedy decoding.

```python
@torch.no_grad()
def generate(prompt, n_new=300, temp=1.0):
    model.eval()
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    for _ in range(n_new):
        logits = model(ids[:, -BLOCK:])[:, -1, :]
        if temp == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = (logits / temp).softmax(-1)
            nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")

for t in [0.2, 0.7, 1.0, 1.4]:
    print(f"\n----- temperature {t} -----")
    print(generate("Once upon a time", n_new=300, temp=t))
```

Read the four samples side by side. At `0.2` the model loops on the safest continuations and may repeat a phrase; at `0.7-1.0` you get the most story-like text; at `1.4` spelling and grammar start breaking as the tail of the distribution leaks in. This is the entire practical content of "sampling parameters" — you are trading coherence against diversity, and the sweet spot for a well-trained model is usually 0.7-1.0. `errors="replace"` guards against a sampled byte sequence that is not valid UTF-8.

## Part G — Attention Patterns and the Induction Head

Attention maps are not explanations, but they are evidence, and one pattern is worth hunting for specifically: the **induction head** (Olsson et al., 2022). An induction head implements copying — "the token that followed `X` last time is a good guess for what follows `X` now" — and it is the mechanistic basis of in-context learning. You detect it with a controlled input: a random token sequence repeated twice. In the second copy, position $t$ holds the same token as position $t - L$, so a head doing induction should attend from $t$ back to $t - L + 1$: the token that came *after* the previous occurrence.

```python
L = 48
torch.manual_seed(0)
base = torch.randint(0, VOCAB, (1, L), device=device)
rep  = torch.cat([base, base], dim=1)                  # [1, 2L], period L
with torch.no_grad():
    model(rep, store_attn=True)                        # populates every block's last_attn

def induction_score(w):                                # w: [1, H, 2L, 2L]
    H = w.shape[1]; out = []
    for h in range(H):
        vals = [w[0, h, t, t - L + 1].item() for t in range(L, 2 * L)]
        out.append(sum(vals) / len(vals))
    return out

print(f"chance level ~ {1.0 / (2 * L):.3f}\n")
best = (None, None, -1)
for li, blk in enumerate(model.blocks):
    scores = induction_score(blk.attn.last_attn)
    for h, s in enumerate(scores):
        if s > best[2]: best = (li, h, s)
    print(f"layer {li}: " + "  ".join(f"h{h}={s:.3f}" for h, s in enumerate(scores)))
print(f"\nstrongest induction head: layer {best[0]} head {best[1]} (score {best[2]:.3f})")
```

```python
li, h, _ = best
w = model.blocks[li].attn.last_attn[0, h].cpu()        # [2L, 2L]
plt.figure(figsize=(5, 5))
plt.imshow(w, cmap="viridis")
plt.title(f"layer {li} head {h} — repeated sequence (period {L})")
plt.xlabel("key position"); plt.ylabel("query position"); plt.colorbar(fraction=0.046)
plt.show()
```

Compare the strongest head's score against the chance level of $1/2L$. A genuine induction head lands several times above chance, and in the heatmap you see a bright off-diagonal stripe: for query positions in the second half ($\ge L$), attention concentrates on the position exactly $L - 1$ back. That stripe *is* the copy circuit. This model is small and trained briefly, so the effect is present but not razor-sharp — the honest lesson is that the *measurement* is what distinguishes "a head that copies" from "a head I hope copies," the same discipline as the max-abs-diff check in Part A. Induction only needs two layers to form (a previous-token head feeding a matching head), which is why it shows up even here.

## What you built

- `sdpa`: scaled dot-product attention with a causal mask, verified against `F.scaled_dot_product_attention` to `< 1e-5` max abs diff, with the mask checked to be exactly zero above the diagonal.
- `CausalMHA`: multi-head attention with RoPE applied to Q and K, plus an inspection hook for the weights.
- A pre-norm decoder block (RMSNorm → MHA → residual → RMSNorm → SwiGLU → residual) stacked into a 4-layer, ~3.3M-parameter byte-level `TinyLM` with weight-tied embeddings.
- A full training run on a TinyStories slice with train/val loss curves, temperature-controlled sampling, and an attention analysis that locates an induction-style head by a quantitative score, not by eyeballing.

## Exercises

1. **GQA.** Convert `CausalMHA` to grouped-query attention: use `n_kv_heads < n_heads` and repeat each K/V head across a group of Q heads. Report the parameter and KV-cache savings for `n_heads=8, n_kv_heads=2`, and confirm the loss curve is not materially worse.
2. **The √d ablation.** Remove the `/ math.sqrt(D)` scale and retrain. Plot the first 200 steps of loss against the scaled run and explain, in terms of softmax saturation, why the unscaled model learns slower or diverges.
3. **RoPE base sweep.** Retrain with `base` set to 500 and to 1,000,000 in `rope_tables`. Generate at a length longer than `BLOCK` and describe how the base changes length extrapolation quality.
4. **Two layers vs four.** Train a 2-layer and a 4-layer model to the same step count. Compare final val loss and the strongest induction score in each. Does the induction head strengthen with depth?
5. **min-p sampling.** Add min-p sampling (drop tokens whose probability is below `min_p * max_prob`) to `generate`, and compare its outputs at an aggressive setting against temperature 1.4. Which degrades more gracefully?

## What interviews ask here

- Why divide attention scores by $\sqrt{d_k}$, and what breaks in the softmax if you don't?
- How would you numerically verify a hand-written attention kernel against a reference — what do you measure and what tolerance is acceptable in float32?
- Why is RoPE applied to Q and K but not V, and how does rotating both give attention scores that depend only on relative position?
- Pre-norm vs post-norm: which trains more stably at depth without warmup, and why?
- What is an induction head, why does it need at least two layers, and what does its existence explain about in-context learning?
- What should the initial cross-entropy loss of a freshly initialized LM be, and what does a much larger value tell you about your initialization?
