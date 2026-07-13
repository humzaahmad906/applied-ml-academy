# 03 — Tokenization: Turning Text into Model Inputs

A transformer never sees text. It sees a sequence of integers, and every one of those integers
indexes a row in an embedding table. Tokenization is the function that maps a string to that integer
sequence, and it is the most under-appreciated component in the whole stack: it fixes your vocabulary,
your sequence length, your per-language cost, and a surprising number of your failure modes — all
before the first matmul. A model that can't add two numbers, chokes on a rare language, or emits
garbage on a specific rare string is very often failing at the tokenizer, not the weights. This module
is how modern subword tokenizers actually work, why they were chosen, and where they break.

The unit you learn from [word vectors](02-word-vectors.md) was the whole word. That has a fatal
problem for open-vocabulary text: you can't enumerate every word (misspellings, names, code, new
slang, 100+ languages), and a fixed word vocabulary maps everything unseen to a single `<UNK>` token,
throwing away information. Character-level models fix coverage but blow up sequence length — English
averages ~4 characters per word, so a character model needs ~4× the sequence for the same text, and
attention is O(n²) (see [transformer architecture](04-transformer-architecture.md)). Subword
tokenization is the compromise that won: frequent words stay whole, rare words fragment into pieces,
nothing is ever truly out-of-vocabulary because the pieces bottom out at bytes or characters.

## BPE: merge the most frequent pair, repeat

**Byte-Pair Encoding** (Sennrich et al., 2016, adapting a 1994 compression algorithm) is the
dominant scheme. Training is a greedy loop:

1. Start with a base vocabulary of individual characters (or bytes).
2. Count all adjacent symbol pairs in the corpus.
3. Merge the single most frequent pair into a new symbol; record the merge rule.
4. Repeat until you hit your target vocabulary size.

The learned artifact is an *ordered list of merge rules*. Encoding a new word replays those merges in
the same order.

### Worked example

Take a tiny corpus with word counts:

```
low     : 5
lower   : 2
newest  : 6
widest  : 3
```

Split every word into characters plus an end-of-word marker `_` so the tokenizer knows where words
end (otherwise `est` inside `newest` and `est` at a word boundary are indistinguishable):

```
l o w _              (5)
l o w e r _          (2)
n e w e s t _        (6)
w i d e s t _        (3)
```

Base vocab: `{l, o, w, e, r, n, s, t, i, d, _}`.

Now count adjacent pairs across the corpus, weighted by word count:

- `e s`: appears in `newest` (6) and `widest` (3) → **9**
- `s t`: same two words → **9**
- `t _`: same → **9**
- `l o`: `low` (5) + `lower` (2) → 7
- `o w`: 7 …

`e s`, `s t`, `t _` tie at 9. BPE breaks ties by a fixed rule (first encountered / lexicographic —
implementations vary but it must be deterministic). Say we merge `s t` → `st`:

```
Merge 1:  (s, t) -> st
n e w e st _   (6)
w i d e st _   (3)
```

Recount. Now `e st` appears 9 times (both words), `st _` appears 9 times. Merge `e st` → `est`:

```
Merge 2:  (e, st) -> est
n e w est _    (6)
w i d est _    (3)
```

Now `est _` is 9. Merge `est _` → `est_`:

```
Merge 3:  (est, _) -> est_
n e w est_     (6)
w i d est_     (3)
```

`l o` is now the most frequent remaining pair at 7 (from `low` and `lower`). Merge `l o` → `lo`, then
`lo w` → `low` (7):

```
Merge 4:  (l, o) -> lo
Merge 5:  (lo, w) -> low
low _          (5)
low e r _      (2)
n e w est_     (6)
w i d est_     (3)
```

After five merges the vocabulary is the original characters plus `{st, est, est_, lo, low}`. Notice
what emerged with zero linguistic input: `est_` is essentially the superlative suffix, and `low` is a
whole common word. BPE rediscovers morphology as a side effect of frequency. To encode a new word
like `lowest`, you replay the merge list in order: `l o w e s t _` → (M1) `l o w e st _` → (M2)
`l o w est _` → (M3) `l o w est_` → (M4) `lo w est_` → (M5) `low est_`. Result: `[low, est_]`. The
unseen word costs two tokens and never hits `<UNK>`.

The BPE merge loop is a classic implementation interview question — see the drills in the
implementation bank for a from-scratch version.

## Byte-level BPE: never fall off the vocabulary

If your base units are Unicode *characters*, a character you never saw in training (a rare CJK glyph,
an emoji) is still out-of-vocabulary. GPT-2 fixed this by running BPE over **raw UTF-8 bytes**. There
are exactly 256 byte values, so the base vocabulary is 256 symbols and *every possible string*
decomposes into bytes with zero coverage gaps — true open-vocabulary. The cost is that non-ASCII text
fragments hard: a character outside ASCII takes 2–4 bytes, so before any merges a Chinese or Arabic
character is several tokens. Merges claw some of that back for frequent sequences, but the byte floor
is why byte-level tokenizers systematically over-charge non-English text. Llama, Qwen, and the GPT
family are all byte-level BPE.

## WordPiece and Unigram: two other ways to pick pieces

**WordPiece** (used by BERT) trains almost like BPE but changes the merge criterion. Instead of
merging the most *frequent* pair, it merges the pair that most increases the likelihood of the corpus
under a unigram language model — concretely, it picks the pair maximizing
`count(xy) / (count(x)·count(y))`, favoring pairs whose joint frequency exceeds what independence
predicts. WordPiece marks continuation pieces with `##` (`playing` → `play`, `##ing`).

**Unigram** (Kudo, 2018) works top-down instead of bottom-up. It starts from a large candidate
vocabulary, assigns each token a probability, and iteratively *removes* the tokens whose deletion
costs the corpus likelihood the least, until it reaches target size. At encode time a word can be
segmented many ways, and Unigram picks the highest-probability segmentation (Viterbi). This
probabilistic view enables **subword regularization**: sampling different segmentations during
training as data augmentation, which BPE can't do cleanly.

**SentencePiece** (Kudo & Richardson, 2018) is a *library*, not a fourth algorithm — it implements
both BPE and Unigram. Its real contribution is treating input as a raw stream with no pre-tokenization
and encoding spaces as a visible meta-symbol `▁` (U+2581). This makes it fully reversible and
language-agnostic: it works identically on English, on Thai (which has no spaces), and on code. When
people say "we used SentencePiece with Unigram," they mean the library with the Unigram objective.
T5 and many multilingual models use this combination.

## Vocabulary size is a real tradeoff

Vocabulary size is a hyperparameter with pull in both directions. **Larger vocab** → each token
carries more text, so sequences are shorter (less compute, more effective context) and common words
stay whole. But it also means a bigger embedding matrix and a bigger output softmax:
`vocab_size × hidden_dim` parameters at both ends, which for a small model can dominate the parameter
budget (see the [architecture](04-transformer-architecture.md) parameter accounting). Larger vocab
also means each embedding row is seen less often during training, so rare tokens are undertrained.
**Smaller vocab** → cheaper embeddings but longer sequences and more fragmentation. Modern models
landed on roughly 32K (Llama 2, older BERT), 128K (Llama 3), or 150K+ (Qwen3, GPT-4o) as vocabularies
grew to better cover multilingual and code data. There is no derived optimum; it's tuned against the
data mixture.

## Multilinguality: not all languages cost the same

Here is the fact that surprises people. Because tokenizers are trained on a corpus that is
overwhelmingly English, they learn long, efficient merges for English and short, fragmented ones for
everything else. The metric is **fertility**: the average number of tokens per word (or per
character). English fertility on a byte-level BPE tokenizer is close to 1 token per word; the same
tokenizer might spend 2–4× as many tokens on the same *meaning* in Hindi, Arabic, Burmese, or Urdu.

This is not cosmetic. It has three direct consequences:

- **Cost.** APIs bill per token. A sentence that costs 10 tokens in English can cost 30 in a
  low-resource language, so the same product is 3× more expensive to serve those users.
- **Context.** A fixed context window holds 3× less *content* in a high-fertility language. Your
  8K-token window is really 8K English words or ~2.5K words of Telugu.
- **Quality.** More tokens per unit meaning means longer sequences, more attention hops for the same
  dependency, and rarer, undertrained token embeddings — a real quality penalty on top of the cost
  one.

This is the **low-resource penalty**, and it compounds a bias that's already in the training data. If
you serve a multilingual product, measure fertility per language on *your* tokenizer against *your*
traffic before you trust a global token budget. The [embeddings/tokenizers lab](17-lab-embeddings-tokenizers.md)
walks through measuring it on English, Urdu, and code.

## Failure modes you will hit in production

- **Numbers.** How a tokenizer splits digits determines whether arithmetic is learnable. If `2024`
  is one token but `2025` splits as `20`,`25`, the model can't rely on positional digit structure.
  Llama 3 and others force each **digit to its own token** specifically to make arithmetic and dates
  more regular. If your model is bad at math, check digit tokenization first.
- **Whitespace and leading spaces.** In byte-level BPE, `"hello"` and `" hello"` are *different
  tokens* (the leading space is part of the token). This is why prompt formatting matters: a trailing
  space before a completion can change the token sequence and hurt output. It's also why naive
  string concatenation in prompt building introduces subtle bugs.
- **Code.** Indentation is semantic in Python, so tokenizers for code models add tokens for runs of
  spaces (e.g. a single token for 4 or 8 spaces). A general tokenizer that fragments indentation
  wastes tokens and blurs structure — a reason code models retrain the tokenizer.
- **Unicode.** Combining characters, normalization forms (NFC vs NFD), and zero-width joiners mean
  the "same" visible string can be different byte sequences. Emoji with skin-tone modifiers are
  multiple code points. Normalize consistently or you get train/serve skew.

### Glitch tokens

A famous failure class: tokens that exist in the vocabulary but were almost never seen during
training, so their embeddings are essentially random. The GPT-2/GPT-3 `SolidGoldMagikarp` family
(Rumbelow & Watkins, 2023) came from Reddit usernames and subreddit strings that made it into the
BPE training corpus but were scrubbed from the language-model training data. Feed the model one of
these and it behaves erratically — evasion, hallucination, refusal — because you've handed it an
embedding it never learned to use. Glitch tokens are a direct symptom of the tokenizer and the model
being trained on *different* data.

## Tokenizer–model lock-in

The tokenizer is chosen *before* pretraining and baked in permanently. Every embedding row and every
output logit is tied to a specific token id, so you cannot swap tokenizers on a trained model without
retraining those layers — and in practice, retraining the model. This has hard consequences: you
can't merge two models with different tokenizers, distillation across tokenizers needs alignment
tricks, and speculative decoding (see [inference](12-inference-decoding.md)) requires the draft and
target models to **share a tokenizer** or you need vocabulary mapping. When you pick a base model,
you're picking its tokenizer for the life of the project.

## The tokenizer-free direction

Tokenization is a hand-built preprocessing step bolted onto an otherwise end-to-end system, and it
causes most of the problems above, so there's active work to remove it. **ByT5** (Xue et al., 2021)
ran a transformer directly on bytes — robust but slow because sequences are long. **CANINE** and
**Charformer** learned to downsample characters. The most promising 2024–2025 line is the **Byte
Latent Transformer** (BLT, Meta 2024): instead of a fixed tokenizer it groups bytes into **patches
dynamically** based on a small byte-level model's next-byte entropy — spending more compute where the
text is unpredictable and less where it's routine — and reports matching BPE-based models at scale
while being more robust to noise and fairer across languages. Tokenizers are not going away this
year, but "the tokenizer is a bug we haven't removed yet" is a credible research position, and it's
worth being able to articulate why.

## What interviews ask here

- **Walk me through BPE training.** Base characters → count adjacent pairs → merge most frequent →
  record ordered merge rules → repeat to target vocab; encoding replays merges in order.
- **Why subword instead of word or character?** Words can't cover open vocabulary (`<UNK>` loss);
  characters blow up sequence length into O(n²) attention. Subword balances coverage and length.
- **Why is a token budget misleading across languages?** Fertility — byte-level BPE trained on
  English-heavy data spends 2–4× more tokens per unit meaning in low-resource languages, raising cost,
  shrinking effective context, and hurting quality.
- **BPE vs WordPiece vs Unigram?** BPE merges most-frequent pair; WordPiece merges the pair that most
  raises corpus likelihood; Unigram prunes a large vocab top-down and enables segmentation sampling.
- **What are glitch tokens and where do they come from?** Vocabulary tokens undertrained because the
  tokenizer corpus and the LM corpus differed; their embeddings are near-random.
- **Why can't you swap a model's tokenizer?** Embedding and output layers are indexed by token id;
  changing the tokenizer invalidates them, which is why speculative decoding needs a shared tokenizer.

## Where this shows up on the job

- **Cost and capacity planning for multilingual products.** You will measure fertility per language
  on your real traffic to forecast API spend and set realistic context budgets, and you'll flag when
  a "cheap" feature is 3× more expensive for non-English users.
- **Debugging model quality regressions.** Bad arithmetic, whitespace-sensitive prompt bugs, and
  erratic outputs on specific strings all trace back to tokenization before you ever touch weights.
- **Picking a base model.** Choosing the base fixes the tokenizer for the project's life, which
  constrains distillation, model merging, and speculative-decoding pairings downstream.
- **Prompt and data pipeline hygiene.** Consistent Unicode normalization and careful handling of
  leading spaces prevent train/serve skew that silently degrades production output.
