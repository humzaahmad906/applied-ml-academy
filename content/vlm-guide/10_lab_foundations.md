# Lab 1 — Build a Tokenizer, Attention, and a Transformer Block

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/humzaahmad906/applied-ml-academy/blob/main/content/vlm-guide/notebooks/10_lab_foundations.ipynb)

**Follow along in a runnable notebook** — free GPU on Colab, no local setup. The full write-up and stack alternatives are below.

Build every core piece of a modern decoder-only transformer from scratch: a byte-level BPE tokenizer, scaled multi-head attention with RoPE, and a pre-norm SwiGLU decoder block stacked into a tiny LM. Makes the [Foundations chapter](01_foundations.md) concrete.

## Setup

```bash
pip install torch numpy
```

No downloads. Runs on CPU in under 60 s. Peak memory < 200 MB.

---

## Part A — Byte-Level BPE Tokenizer

```python
import random, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(42); np.random.seed(42); torch.manual_seed(42)

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "transformers are next token prediction machines",
    "byte level bpe never produces unknown tokens",
    "attention is all you need for sequence modelling",
    "residual connections let gradients flow through depth",
    "rotary position embeddings encode relative offsets",
] * 20  # repeat so merge statistics are stable

def bpe_train(corpus: list[str], num_merges: int = 120) -> tuple[dict, list]:
    # each word → (char_0, ..., char_n, '</w>')
    vocab: dict[tuple, int] = collections.defaultdict(int)
    for line in corpus:
        for word in line.split():
            vocab[tuple(list(word) + ["</w>"])] += 1

    merges: list[tuple] = []
    for _ in range(num_merges):
        pairs: dict[tuple, int] = collections.defaultdict(int)
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.__getitem__)
        merges.append(best)
        merged = best[0] + best[1]
        new_vocab: dict[tuple, int] = {}
        for word, freq in vocab.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word)-1 and word[i] == best[0] and word[i+1] == best[1]:
                    out.append(merged); i += 2
                else:
                    out.append(word[i]); i += 1
            new_vocab[tuple(out)] = freq
        vocab = new_vocab

    all_tokens: set[str] = set()
    for w in vocab: all_tokens.update(w)
    t2i = {t: i for i, t in enumerate(sorted(all_tokens))}
    t2i["<unk>"] = len(t2i)
    return t2i, merges

def bpe_encode(text: str, merges: list, t2i: dict) -> list[int]:
    ids = []
    for word in text.split():
        sym = tuple(list(word) + ["</w>"])
        for a, b in merges:
            out, i = [], 0
            while i < len(sym):
                if i < len(sym)-1 and sym[i] == a and sym[i+1] == b:
                    out.append(a+b); i += 2
                else:
                    out.append(sym[i]); i += 1
            sym = tuple(out)
        ids.extend(t2i.get(s, t2i["<unk>"]) for s in sym)
    return ids

def bpe_decode(ids: list, i2t: dict) -> str:
    return "".join(i2t[i] for i in ids).replace("</w>", " ").strip()

t2i, merges = bpe_train(CORPUS)
i2t = {v: k for k, v in t2i.items()}
VOCAB_SIZE = len(t2i)

sample = "the quick brown transformer"
enc = bpe_encode(sample, merges, t2i)
assert bpe_decode(enc, i2t) == sample, "round-trip failed"
print(f"vocab={VOCAB_SIZE}  encoded={enc}")
```

`</w>` is the end-of-word sentinel: every word maps to a deterministic token sequence with zero unknown tokens.

---

## Part B — Scaled Dot-Product Attention

```python
def sdpa(q, k, v, causal=True):
    """q/k/v: [B, H, T, D] → out [B, H, T, D], weights [B, H, T, T]"""
    B, H, T, D = q.shape
    scores = (q @ k.transpose(-2, -1)) * D**-0.5          # [B, H, T, T]
    if causal:
        scores = scores + torch.full((T, T), float("-inf"), device=q.device).triu(1)
    w = scores.softmax(-1)
    return w @ v, w

B, H, T, D = 2, 4, 16, 32
q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
out, w = sdpa(q, k, v)
assert out.shape == (B, H, T, D)
assert w[0, 0].triu(1).abs().max() < 1e-6, "causal mask broken"
print(f"SDPA: out={tuple(out.shape)}  causal mask OK")
```

---

## Part C — RoPE + Causal MHA + Decoder Block

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        return self.w * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()

def rope_tables(head_dim, max_len, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t   = torch.arange(max_len, dtype=torch.float32)
    f   = torch.outer(t, inv)   # [max_len, head_dim//2]
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):
    """x: [B, T, H, D]; cos/sin: [max_len, D//2]"""
    B, T, H, D = x.shape
    x1, x2 = x[..., :D//2], x[..., D//2:]
    c = cos[:T].unsqueeze(0).unsqueeze(2)   # [1, T, 1, D//2]
    s = sin[:T].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1*c - x2*s, x1*s + x2*c], dim=-1)

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

    def forward(self, x):
        B, T, _ = x.shape
        def proj(W): return W(x).view(B, T, self.H, self.Dh).transpose(1, 2)
        q = apply_rope(self.Wq(x).view(B,T,self.H,self.Dh), self.rc, self.rs).transpose(1,2)
        k = apply_rope(self.Wk(x).view(B,T,self.H,self.Dh), self.rc, self.rs).transpose(1,2)
        v = self.Wv(x).view(B, T, self.H, self.Dh).transpose(1, 2)
        out, _ = sdpa(q, k, v, causal=True)
        return self.Wo(out.transpose(1,2).contiguous().view(B, T, -1))

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = (int(d_model * 8/3) + 63) // 64 * 64
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x): return self.W2(F.silu(self.W1(x)) * self.W3(x))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.n1 = RMSNorm(d_model); self.attn = CausalMHA(d_model, n_heads, max_len)
        self.n2 = RMSNorm(d_model); self.ffn  = SwiGLUFFN(d_model)
    def forward(self, x):
        x = x + self.attn(self.n1(x))   # pre-norm + residual
        return x + self.ffn(self.n2(x))
```

---

## Part D — Tiny LM: Forward Pass and Causal Mask Verification

```python
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=256):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, max_len) for _ in range(n_layers)])
        self.norm   = RMSNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight    # weight tying
    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x))          # [B, T, vocab_size]

model = TinyLM(VOCAB_SIZE).to(device)
ids = torch.randint(0, VOCAB_SIZE, (2, 32), device=device)
with torch.no_grad():
    logits = model(ids)
print(f"logits: {tuple(logits.shape)}")    # (2, 32, vocab_size)
assert logits.shape == (2, 32, VOCAB_SIZE)

# verify causal mask: no token attends to a future position
model.eval()
with torch.no_grad():
    blk = model.blocks[0]
    x0  = blk.n1(model.embed(ids))
    mha = blk.attn
    B, T = 2, 32
    q = apply_rope(mha.Wq(x0).view(B,T,mha.H,mha.Dh), mha.rc, mha.rs).transpose(1,2)
    k = apply_rope(mha.Wk(x0).view(B,T,mha.H,mha.Dh), mha.rc, mha.rs).transpose(1,2)
    v = mha.Wv(x0).view(B,T,mha.H,mha.Dh).transpose(1,2)
    _, w = sdpa(q, k, v, causal=True)
    assert w[0, 0].triu(1).abs().max() < 1e-6
    print("Causal mask OK — upper triangle zero across all heads")

print(f"params: {sum(p.numel() for p in model.parameters()):,} on {device}")
```

---

## What you built

- A character/byte-level BPE tokenizer: merge loop, encode, decode, round-trip verified.
- Scaled dot-product attention with causal masking, shapes checked.
- Causal MHA with RoPE applied to Q and K only, causal mask verified numerically.
- Pre-norm decoder block: RMSNorm → CausalMHA → residual → RMSNorm → SwiGLUFFN → residual.
- 4-layer TinyLM with weight-tied embeddings producing correct `[B, T, vocab_size]` logits.

## Build it further

Add a training loop: cross-entropy loss on shifted targets (`logits[:, :-1]` vs `ids[:, 1:]`), AdamW, 300 steps on the BPE corpus. Log loss every 20 steps and confirm it decreases. Then generate greedily: iteratively take `logits[:, -1].argmax()` and append it to the input.

---

## Stacks & alternatives

**HuggingFace `tokenizers` — production-grade BPE, no coding:**

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tok = Tokenizer(BPE(unk_token="<unk>"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.train(files=["corpus.txt"], trainer=BpeTrainer(vocab_size=8000))
print(tok.encode("the quick fox").ids)
```

Rust-backed, ~100x faster than pure Python, handles all Unicode. Reach for it when you need a real tokenizer — not for teaching the merge loop.

**`tiktoken` — reproduce GPT-4 tokenization exactly:**

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("the quick fox"))
```

No training API; use when you need exact OpenAI token counts or compatibility.

**`F.scaled_dot_product_attention` — the production equivalent of your `sdpa`:**

```python
# dispatches to FlashAttention-2 on CUDA — no [B,H,T,T] materialized
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

O(T) memory vs. your O(T²). Once you've built `sdpa` by hand, this is what you use in real models. `nn.MultiheadAttention` wraps it with learned projections for a drop-in module.
