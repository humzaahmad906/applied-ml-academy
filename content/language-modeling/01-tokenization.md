# 01 — Tokenization

A language model does not see text. It sees a sequence of integers. Tokenization is the map from
one to the other, and it is the first thing you build because everything downstream depends on
it. Get it wrong and you pay for it in every training step and every inference call forever,
because the tokenizer is frozen once you start pretraining.

## Why not just use characters or words

Two obvious baselines, both bad on their own.

Characters (or raw bytes) give you a tiny vocabulary and never hit an unknown token, but your
sequences get very long. A 1000-word document is maybe 5000 bytes, so at the byte level your
context length has to be 5x larger to hold the same text, and attention cost grows with the
square of sequence length. You waste compute representing the fact that `t-h-e` is a common
triple. A pure-character vocabulary is also deceptively large: Unicode has ~150k assigned code
points, so `ord`/`chr` character tokenization gives you an enormous and mostly-unused vocabulary
whose compression ratio (bytes per token) is barely above 1.

Words give you short sequences but an unbounded vocabulary. Any tokenizer with a fixed word
list will meet words it has never seen (names, typos, code identifiers, other languages) and has
to fall back to an unknown token, which throws away information. Word-level also handles
morphology badly: `run`, `runs`, `running`, `runner` become four unrelated integers.

The working compromise is subword tokenization: common sequences get their own token, rare
sequences get broken into smaller pieces, and because the smallest pieces are bytes you can
always represent anything. The dominant algorithm for this is byte-pair encoding.

## Bytes and UTF-8, the actual substrate

Modern BPE operates on **bytes**, not characters. You encode the string as UTF-8 first:

```python
"hello".encode("utf-8")            # b'hello'  -> [104, 101, 108, 108, 111]
"héllo".encode("utf-8")           # b'h\xc3\xa9llo' -> the é is TWO bytes
"🌍".encode("utf-8")               # 4 bytes
```

UTF-8 is variable-length: ASCII is one byte, but accented Latin, CJK, emoji, and most non-Latin
scripts are two to four bytes each. This is why a byte-level BPE tokenizer trained on English
splits Urdu, Chinese, or emoji into many small tokens — those characters cost more bytes to begin
with, and the merges that would compress them were never learned.

Working on bytes has two consequences you must respect:

- **You can represent anything.** The base vocabulary is the 256 byte values, so there is no
  "unknown token." Any input round-trips.
- **Decoding can produce invalid UTF-8 at boundaries.** If you decode a partial token sequence,
  or a single mid-character byte, the bytes may not be a valid UTF-8 string. Decode with
  `errors="replace"` so a broken byte becomes the Unicode replacement character U+FFFD (`�`)
  rather than raising. This matters during streaming generation, where you emit tokens one at a
  time and a multi-byte character can straddle two tokens.

## Byte-pair encoding, precisely

BPE is a compression algorithm repurposed for tokenization. You train it once on a corpus to
learn a set of merges, then apply those merges at runtime to encode any text.

Training, in its naive form:

1. Start with the text as a sequence of bytes (0–255). The initial vocabulary is those 256
   byte values (plus any special tokens).
2. Count every adjacent pair of tokens in the corpus.
3. Find the most frequent pair. Merge it into a new single token, assigned the next free integer
   id. Record this merge in an ordered list.
4. Repeat from step 2 until the vocabulary reaches the target size (e.g. 10k, 32k, 50k, 128k).

The output of training is two things: the vocabulary (id → byte-string map) and the ordered
list of merges. The order matters because it defines priority at encode time. Here is the whole
naive training loop, which is worth reading once before you make it fast:

```python
def train_bpe(text: str, num_merges: int):
    indices = list(text.encode("utf-8"))               # start as raw bytes
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}                                         # (a, b) -> new_id, in order
    for i in range(num_merges):
        counts = collections.Counter(zip(indices, indices[1:]))
        pair = max(counts, key=counts.get)             # most frequent adjacent pair
        new_id = 256 + i
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        indices = merge(indices, pair, new_id)         # replace every occurrence
    return vocab, merges
```

There is one more detail the toy version skips: **deterministic tie-breaking**. When two pairs
have the same frequency, break the tie by preferring the lexicographically greater pair
(comparing the actual byte-strings). Without a fixed rule your merges are nondeterministic
and your tests will not reproduce.

## Pre-tokenization, the step everyone forgets

Real BPE tokenizers do not run merges across the entire raw byte stream. They first split text
into chunks — "pre-tokens" — with a regex, then run BPE **within each chunk** and never merge
across chunk boundaries. The reason is subtle but important: if you counted pairs across
boundaries, the tokenizer would learn a token that glues `dog` to a following `.` or space in
some contexts and not others, making the vocabulary depend on spacing accidents. By splitting
first, a pre-token that appears 10 times contributes exactly 10 to its internal pair counts and
zero to anything spanning its edges.

GPT-2's pattern is the de facto standard. We use the tidied form from tiktoken:

```python
import regex as re   # the `regex` package, not the stdlib `re` — you need \p{L}, \p{N}
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re.findall(PAT, "some text that i'll pre-tokenize")
# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

Read the alternatives left to right: common English contractions (`'s`, `'ll`, `'ve`, ...); an
optional leading space plus a run of letters; an optional leading space plus a run of digits; an
optional leading space plus a run of non-space-non-letter-non-digit characters (punctuation); and
finally two whitespace cases. The `\s+(?!\S)` clause captures trailing whitespace without eating
the space that belongs to the next word. Use `re.finditer`, not `re.findall`, so you stream over
matches instead of materializing the whole list while counting.

This is also where the **leading-space convention** comes from. Because ` ?\p{L}+` attaches the
space to the following word, ` cat` (with leading space) and `cat` (start of line) become
different tokens. This is deliberate, and it is why naive concatenation of decoded token strings
reproduces the original text with its spaces intact.

## Special tokens

You reserve a handful of ids for tokens that never appear in natural text: end-of-document,
padding, and later the chat-format markers (system/user/assistant turn boundaries). These are
added to the vocabulary explicitly, not learned by BPE. The canonical one is `<|endoftext|>`,
which delimits documents in the training files.

Special tokens interact with pre-tokenization in a way that trips people up, so handle them in a
specific order:

- **Before training and before encoding, split the text on the special tokens first**, then
  pre-tokenize each piece separately. If your corpus is `[Doc 1]<|endoftext|>[Doc 2]`, you split
  on `<|endoftext|>` and pre-tokenize `Doc 1` and `Doc 2` independently, so no merge can ever
  cross a document boundary. This is also how you parallelize pre-tokenization: chunk the corpus
  at `<|endoftext|>` occurrences and process chunks concurrently.
- **At encode time, a special token is always emitted as its single reserved id**, never run
  through the regex or the merges. `<|endoftext|>` in, one integer out.

## Making training fast: incremental pair counts

The naive loop above recounts every pair in the whole corpus on every merge. On a real corpus
that is hopelessly slow — this is the part of the build that separates a working solution from
a hanging one. The fix is to never recount from scratch.

Two ideas do the work:

1. **Count over unique pre-tokens with multiplicities, not over the raw stream.** Pre-tokenize
   once into a dictionary `{pre_token_bytes: frequency}`. The word `the` appearing 40,000 times
   is stored once with count 40,000. All pair counting happens over this deduplicated table
   weighted by frequency, which shrinks the work by orders of magnitude.

2. **After a merge, update only the counts it touched.** Maintain a global pair-count table and
   an index from each pair to the pre-tokens (and positions) containing it. When you merge
   `(A, B) -> X` inside a pre-token like `... A B C ...`:
   - the pairs `(A, B)` disappears,
   - the neighbor pair `(prev, A)` becomes `(prev, X)`,
   - the neighbor pair `(B, C)` becomes `(X, C)`,
   - and a brand-new pair `(X, next)`/`(prev, X)` may appear.

   You decrement the vanished pairs and increment the created ones, each change weighted by that
   pre-token's frequency. Every other pair in the corpus is untouched, so a single merge costs
   work proportional to the number of occurrences of the merged pair, not the size of the corpus.

Conceptually the merge itself is a linear scan that rewrites a sequence:

```python
def merge(indices, pair, new_id):
    out, i = [], 0
    while i < len(indices):
        if i + 1 < len(indices) and (indices[i], indices[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(indices[i]); i += 1
    return out
```

In the fast trainer you do not call this on the whole corpus; you apply the equivalent edit
in place on the affected pre-tokens and reconcile the pair-count deltas. That bookkeeping is the
real content of the build.

## Encoding and decoding

Encoding a new string mirrors training. Pre-tokenize, then within each pre-token apply the
learned merges **in the order they were created** — the earliest merge wins — until none applies:

```python
def encode(text, merges):                 # merges: dict (a,b)->id, insertion-ordered
    ids = []
    for pre_token in re.finditer(PAT, text):
        seq = list(pre_token.group().encode("utf-8"))
        for pair, new_id in merges.items():          # earliest merge first
            seq = merge(seq, pair, new_id)
        ids.extend(seq)
    return ids
```

For `the cat ate`, pre-tokenization gives `['the', ' cat', ' ate']`; `the` starts as
`[b't', b'h', b'e']`, the first applicable merge `(b't', b'h') -> b'th'` fires, then
`(b'th', b'e') -> b'the'`, and the pre-token collapses to a single id.

Decoding is trivial and is the source of the round-trip guarantee: look each id up in the
vocabulary, concatenate the byte-strings, and decode the bytes as UTF-8 (with `errors="replace"`
for safety):

```python
def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

A well-behaved tokenizer round-trips: `decode(encode(s)) == s` for any `s`.

## The corpus statistics are baked in

Because you merge on frequency, the tokenizer bakes in the statistics of your training corpus.
Train it on English and it will split Urdu or code into many small tokens, which makes those
inputs more expensive and, empirically, a bit worse-modeled. If your product runs on shipping
labels and BOLs, the token distribution of that domain matters — a tokenizer trained on generic
web text will over-fragment your part numbers and address formats.

## Vocabulary size is a real tradeoff

Bigger vocabulary means shorter sequences (cheaper attention, more text per context window) but
a larger embedding table and output projection, which is where a surprising fraction of a small
model's parameters live. For a model with hidden size `d` and vocab `V`, the embedding and the
tied or untied output head are each `V × d` parameters. At `V = 128k` and `d = 2048` that is 262M
parameters in the embedding alone, which for a sub-billion-parameter model is enormous. This is
exactly why small on-device models often use smaller vocabularies: the embedding table does not
shrink when you shrink the transformer body, so it dominates. Our own toy runs use
`vocab_size = 10000` precisely to keep the embedding cheap on a small model.

The other cost of a large vocabulary is that rare tokens get few gradient updates and stay poorly
trained. There is a sweet spot, and for most production models it lands somewhere between 32k and
128k.

## Measuring a tokenizer

Two numbers tell you most of what you need.

**Compression ratio**: bytes per token on held-out text of your target domain.

```python
def compression_ratio(text, ids):
    return len(text.encode("utf-8")) / len(ids)
```

Higher is better (fewer tokens for the same text) but only meaningful relative to a fixed
vocabulary size; you can always buy better compression by growing the vocab. A byte-level
baseline sits near 1.0; a decent BPE tokenizer on in-domain English lands around 4.

**Fertility**: average number of tokens a word is split into. Close to 1 for English on an
English-trained tokenizer; much higher for out-of-domain text. If your model will serve multiple
languages or code, check fertility on each and be honest about the ones that come out expensive.

## What you build, and how it is tested

The build is a full byte-level BPE tokenizer from scratch: train it on a corpus (TinyStories,
then OpenWebText) to a target vocab size, serialize the vocab and merges, and implement
`encode`/`decode` that round-trip losslessly. The two things that make it non-trivial are getting
pre-tokenization and special-token handling right, and making training fast enough on a real
corpus with the incremental pair-count scheme above.

Testing is done through a set of adapter functions wired into `pytest`. The suite checks the
learned merges and vocab against a reference on a fixed corpus (which is why deterministic
tie-breaking matters), checks that encoding produces the expected ids on known inputs, and checks
the round-trip property `decode(encode(s)) == s` including on strings that contain special tokens
and multi-byte UTF-8. You run it with `uv run pytest`.

## Key takeaways

Tokenization is a frozen, upfront decision that sets a floor on cost and a ceiling on which
inputs your model handles well. BPE gives you the always-representable safety of bytes with the
efficiency of subwords, operating over UTF-8 byte sequences so nothing is ever "unknown."
Pre-tokenization with the GPT-2 regex is not optional — it stops merges from crossing word
boundaries and is the part people implement wrong — and special tokens must be split out before
pre-tokenization and emitted as atomic ids at encode time. Naive training is too slow; the real
work is counting over deduplicated pre-tokens and updating only the pair counts each merge
touches. Vocabulary size trades sequence length against embedding parameters, and for small
models the embedding table can dominate the whole parameter budget.

## You can now

- explain why byte-level BPE never emits an out-of-vocabulary token, and why decoding must use `errors="replace"` during streaming generation.
- implement BPE training with incremental pair-count updates over deduplicated pre-tokens instead of recounting the whole corpus on every merge.
- apply the GPT-2 pre-tokenization regex and split special tokens out *before* merging, so no merge ever crosses a word or document boundary.
- measure a tokenizer with compression ratio and fertility, and reason about how the training corpus's statistics get baked into the merges.
- size a vocabulary against the embedding-table parameter cost for a given model scale, and know why small on-device models use smaller vocabularies.
