# 15 — Vision and Language

For fifteen modules you have treated an image as a self-contained object: pixels in, a label or a box or a mask out. But the label set was always fixed in advance — a classifier trained on ImageNet's 1,000 categories cannot name a thing outside those 1,000, and a detector knows only the classes its head was trained for. Language breaks that ceiling. If a model can relate an image to *any* piece of text, then its "label set" becomes the whole of language, and you can ask it about a category it never explicitly trained on. This module is about that bridge: how vision and text get mapped into a shared space, what that unlocks, and how it grows into the full vision-language models you will meet in the sibling courses.

## From captioning to a shared space

The first serious attempt to join vision and language was **image captioning**: given a photo, generate a sentence describing it. The early "Show and Tell" approach (2015) was a clean encoder-decoder — a CNN encoded the image into a single vector, and an RNN decoded that vector into words, exactly the sequence-to-sequence recipe from machine translation with an image standing in for the source sentence. "Show, Attend and Tell" then added attention so the decoder could look back at different image regions as it emitted each word, which both improved captions and produced interpretable heatmaps of where the model was looking.

Captioning is *generative* — it produces text — and it works, but it has two limits worth naming. First, it needs paired image-caption data, which is expensive to collect at the scale vision models want. Second, and more fundamentally, it does not give you the thing we most want for retrieval and zero-shot tasks: a fast way to score how well an *arbitrary* image and an *arbitrary* sentence match. To rank a million images against a query with a captioning model you would have to run generation a million times. What we want instead is a single embedding per image and per sentence, so matching is a cheap dot product. That is exactly what contrastive pretraining delivers, and it is why the field's center of gravity shifted from captioning to CLIP.

## CLIP: contrastive image-text pretraining

**CLIP** (Contrastive Language-Image Pre-training, 2021) trains two encoders at once. An image encoder — a ResNet or, more commonly now, a [vision transformer](07-vision-transformers.md) — maps an image to a vector. A text encoder — a transformer — maps a caption to a vector. Both vectors are projected to the same dimension and L2-normalized so they live on a shared unit sphere. The training signal is deceptively simple: images and their true captions should land close together on that sphere, and mismatched pairs should land far apart.

The trick that made it scale is that the negatives are *free*. Take a batch of $N$ image-text pairs scraped from the web. Compute all $N \times N$ pairwise similarities between the image vectors and the text vectors. The $N$ pairs on the diagonal are the true matches; the other $N^2 - N$ off-diagonal entries are negatives you got for nothing. Now treat it as two classification problems — for each image, pick its matching text out of the $N$ candidates, and for each text, pick its matching image — and optimize both with cross-entropy. This is the **batch-softmax** (InfoNCE) objective you saw for contrastive [self-supervised learning](11-self-supervised-learning.md), now spanning two modalities instead of two views of one image.

Written out, the loss for a single image $i$ is a softmax over its similarities to every text in the batch, with temperature $\tau$:

$$
\mathcal{L}_{\text{img}\to\text{txt}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\langle u_i, v_i\rangle / \tau)}{\sum_{j=1}^{N} \exp(\langle u_i, v_j\rangle / \tau)}
$$

where $u_i$ is the normalized image embedding and $v_j$ the normalized text embeddings. The full CLIP loss symmetrizes this — you add the identical term with the roles of image and text swapped, $\mathcal{L}_{\text{txt}\to\text{img}}$, and average the two. Because the embeddings are unit-normalized, the inner product $\langle u_i, v_j\rangle$ is just the cosine similarity, so the whole objective is asking each row and each column of the similarity grid to peak on its diagonal entry.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
N, d = 8, 512                          # batch of 8 pairs, 512-dim embeddings
img_emb = torch.randn(N, d)            # from the image encoder
txt_emb = torch.randn(N, d)            # from the text encoder

# project onto the unit sphere so dot product = cosine similarity
img_emb = F.normalize(img_emb, dim=1)
txt_emb = F.normalize(txt_emb, dim=1)

# learned temperature; CLIP stores log-temp and clamps it
logit_scale = torch.tensor(1 / 0.07).log().exp()
logits = logit_scale * img_emb @ txt_emb.t()   # (N, N) similarity grid

# the correct match for row i is column i (and vice-versa)
labels = torch.arange(N)
loss_i = F.cross_entropy(logits, labels)       # each image -> its text
loss_t = F.cross_entropy(logits.t(), labels)   # each text  -> its image
loss = (loss_i + loss_t) / 2
print(round(loss.item(), 3))
```

Three details matter in practice. The **temperature** (here `1/0.07`) sharpens the softmax; CLIP learns it as a parameter but clamps it so the logits do not explode. **Batch size is the quality knob** — bigger batches mean more negatives per step, so CLIP was trained with tens of thousands of pairs per batch across many GPUs, which is exactly the scale problem [module 16](16-scale-and-frontier.md) picks up. And the **data** — 400 million noisy image-caption pairs from the web — is what let a single model absorb enough of the visual-linguistic world to generalize.

The recipe has been iterated on since. **SigLIP** replaced the batch-wide softmax with a pairwise *sigmoid* loss, which removes the need to gather similarities across the whole batch and so trains well at smaller batch sizes; **EVA-CLIP** and the open **OpenCLIP** reproductions pushed accuracy up with better backbones and cleaner data. The mechanism you just implemented is the durable core; these are refinements of the loss and the data pipeline around it, and for applied work any of them drops into the same zero-shot and retrieval recipes below.

## Zero-shot classification

Once trained, CLIP classifies images from categories it never saw as labels, because a "label" is now just a piece of text. The full recipe is short and worth memorizing:

1. Write your class names and wrap each in one or more prompt templates.
2. Encode every prompt with the text encoder and L2-normalize the vectors (one vector per class, averaged over templates).
3. Encode the image with the image encoder and L2-normalize it.
4. Take the dot product of the image vector with every class vector — these are cosine similarities.
5. Apply softmax (scaled by the temperature) and pick the argmax.

```python
# pseudo-code with a real CLIP checkpoint (pip install open_clip_torch)
classes = ["a cat", "a dog", "a fire truck"]
prompts = [f"a photo of {c}." for c in classes]   # prompt template

text_feats = model.encode_text(tokenizer(prompts))   # (num_classes, d)
img_feats  = model.encode_image(preprocess(image)[None])  # (1, d)

text_feats = F.normalize(text_feats, dim=1)
img_feats  = F.normalize(img_feats, dim=1)
probs = (100.0 * img_feats @ text_feats.t()).softmax(dim=1)
pred = classes[probs.argmax()]
```

Notice the template `"a photo of {c}."` rather than the bare word. This is **prompt engineering for classifiers**, and it is not a gimmick: CLIP's captions were natural sentences, so a bare noun is out of distribution. Wrapping the class in a sentence, and often *averaging* the text embeddings over many templates measurably improves accuracy — you are ensembling over phrasings of the same concept.

```python
# prompt ensembling: average normalized text embeddings over templates
templates = ["a photo of a {}.", "a blurry photo of a {}.",
             "a close-up photo of a {}.", "a {} in the wild."]
def class_vector(name):
    prompts = [t.format(name) for t in templates]
    feats = F.normalize(model.encode_text(tokenizer(prompts)), dim=1)
    return F.normalize(feats.mean(dim=0), dim=0)   # ensemble, then renormalize
```

The whole appeal is that adding a new class costs one forward pass of the text encoder, with no retraining and no labeled images — so a classifier's vocabulary can change at inference time, driven by a config file rather than a training run.

## What the embeddings power

CLIP's real footprint is less about classification and more about the shared embedding space itself, which behaves like a universal "meaning" index for images and text.

- **Search and retrieval.** Embed a million images once, store the vectors, and you can retrieve by text query ("a red bicycle against a brick wall") with a single nearest-neighbor lookup. This is how modern image search and dataset-curation tools work under the hood.

  ```python
  # index once, query cheaply — the core of text-to-image search
  gallery = F.normalize(model.encode_image(all_images), dim=1)   # (M, d), stored
  q = F.normalize(model.encode_text(tokenizer(["a red bicycle"])), dim=1)  # (1, d)
  scores = q @ gallery.t()                 # (1, M) cosine similarities
  top5 = scores[0].topk(5).indices         # nearest images to the query
  ```

- **Conditioning generative models.** Text-to-image [diffusion models](13-diffusion-models.md) need a text representation to steer generation; CLIP (or a CLIP-like) text encoder was the original source of that signal, feeding the cross-attention that turns a prompt into a picture.
- **Data filtering.** CLIP similarity is used to score and filter web-scale image-text pairs when building the training sets for the next generation of models — including the diffusion models above.
- **Features for downstream heads.** A frozen CLIP image encoder is a strong general-purpose feature extractor; a **linear probe** on top of its embeddings often beats training a small model from scratch, and is a standard baseline when you have limited labels (the linear-probe-vs-fine-tune distinction from [module 11](11-self-supervised-learning.md) applies here unchanged).

The common thread is that once images and text live in the same metric space, a great many tasks reduce to *distance* — nearest neighbor, thresholding a similarity, or steering another model with a vector. That is why one pretrained model quietly sits underneath search, moderation, generation, and curation at once.

## Open-vocabulary detection and segmentation

The same "text is the label set" idea lifts [detection and segmentation](06-detection-and-segmentation.md) out of their fixed taxonomies. Recall that a standard detector's classification head has one output per class, fixed at training time — to add a class you retrain. **OWL-ViT** replaces that head with CLIP-style text embeddings, so instead of predicting scores over 80 fixed classes it scores each candidate box against arbitrary text queries — you ask for "a stethoscope" and it localizes stethoscopes, even though no detector was trained on that class. The class set is now a list of strings you pass at inference.

On the segmentation side, **SAM** (met in module 06) produces high-quality masks from a point or box prompt but does not itself know class *names* — it segments *what you point at*, not "the dog." The common pattern is therefore a pipeline: an open-vocabulary detector or CLIP finds *where* the named object is, then SAM cuts out the precise mask. Text supplies the semantics, SAM supplies the geometry. This composition, sometimes called "grounded" segmentation, is how promptable, label-free vision systems are actually assembled in 2026 — and it is a good example of the modern habit of *composing* foundation models rather than training one monolith end to end.

## From CLIP to full VLMs

CLIP aligns whole images with whole sentences, but it cannot *converse* — it has no language generation, so it can rank a caption but not write one, and it cannot answer "how many people are in this photo?" The step to a full vision-language model is surprisingly modest in concept: take a strong frozen image encoder (often a CLIP or [DINOv2](11-self-supervised-learning.md) backbone), take a pretrained language model, and train a small **projector** (an MLP or a handful of attention layers) that maps image features into the language model's token embedding space. The image becomes a few "visual tokens" the language model reads as if they were words, prepended to the text prompt, and now the system can caption, answer questions about an image, and follow visual instructions. The elegance is that both heavy components can stay frozen — only the small projector must be trained, which is cheap — and the whole design reuses a language model you did not have to build. That projector recipe, and everything about training and evaluating these models, is the subject of the [VLMs module](../vlm-guide/04_vlms.md) in the VLM guide, with the language-side view of fusing modalities covered in [multimodality](../nlp-with-transformers/14-multimodality.md) in the NLP course.

## Evaluation gotchas

CLIP-style models are strong but flawed in a specific, well-documented way: they behave like a **bag of words**. Because contrastive training only ever asked "does this text match this image better than the other texts in the batch?", the encoders learn to match *content words* — the nouns and adjectives that pin down which image is meant — without reliably encoding *how those words compose*. "A cat chasing a dog" and "a dog chasing a cat" produce nearly identical CLIP embeddings, and the model struggles to tell "a red cube on a blue sphere" from "a blue cube on a red sphere." Benchmarks like Winoground and ARO were built specifically to expose this **compositionality** failure.

Why does the objective cause this? To win the contrastive game you only need to be *distinguishable* from the other captions in the batch, and most random negatives differ in their nouns, so matching content words is enough to score well. Word order carries almost no gradient. The practical lesson: CLIP is excellent for retrieval and coarse zero-shot recognition, but do not trust it for relational or counting questions — those need the generative VLMs above, and even they are shaky. Treat a high retrieval score as evidence the right *objects* are present, not that the *relationships* are correct, and when you evaluate a CLIP-based system, include some hard negatives that differ only in word order so the metric cannot be gamed by object matching alone.

## Why this matters for ML

Vision-language models are where a large fraction of applied vision now lives. Product image search, content moderation that takes a policy written in English and flags matching images, "find me the frame where the driver is holding a phone" in a video pipeline, automatic alt-text, retrieval-augmented systems that ground a chatbot's answer in a document's figures — all of them rest on a shared image-text embedding or a projector-fused VLM.

The reason this shows up so widely is economic: zero-shot capability means a team can ship a new visual category by editing a list of prompt strings instead of collecting labels and retraining, which collapses the time-to-market for a new feature from weeks to minutes. You will rarely pretrain CLIP yourself; you will constantly fine-tune it, prompt it, use its embeddings as features, and — critically — know its failure modes well enough not to ship a relational query on top of a bag-of-words model.

Everything in this module — CLIP, open-vocabulary systems, full VLMs — was only possible because of scale: hundreds of millions of pairs, batches in the tens of thousands, many GPUs running for weeks. That is not a detail you can wave away, so the final module turns to it directly: how vision models are actually trained at scale, where the frontier is heading with world models and video, and the human responsibilities that come with deploying any of this. It also closes the course and lays out where to go next.

## Key takeaways

- **Image captioning** (Show and Tell → Show, Attend and Tell) joined vision and language generatively, but needs paired data and cannot cheaply *score* arbitrary image-text matches — the gap contrastive pretraining fills.
- **CLIP** trains an image encoder and a text encoder jointly so matching image-text pairs are close on a shared unit sphere; the **batch-softmax** objective turns every batch into two cross-entropy classification problems using off-diagonal pairs as free negatives.
- **Zero-shot classification** works by encoding class names as sentences and picking the nearest text vector — new classes cost one text forward pass, no retraining. Wrap labels in **prompt templates** and ensemble over several.
- CLIP's shared space powers **retrieval/search**, **conditioning for diffusion models**, and **data filtering** at web scale.
- Text-as-labels extends to **open-vocabulary detection** (OWL-ViT) and, paired with **SAM**, to promptable segmentation — the modern pattern of *composing* foundation models rather than training one monolith.
- Successor models (**SigLIP**'s sigmoid loss, **OpenCLIP**, **EVA-CLIP**) refine the loss and data but keep the same contrastive core and drop into the same recipes.
- A **projector** mapping frozen image features into a language model's token space turns CLIP-style alignment into a full **VLM** — see the VLM guide and the NLP multimodality module.
- CLIP behaves like a **bag of words**: strong on objects, weak on **compositionality** (relations, order, counting). Match the tool to the question, and put order-only hard negatives in your eval so object matching alone cannot game the score.

- The whole space rests on **scale**, which the final module examines head-on before closing the course.

## Try it

Install `open_clip_torch`, load a pretrained CLIP checkpoint (e.g. `ViT-B-32`), and reproduce zero-shot classification on a handful of your own photos: write 3–5 class names, wrap them in the `"a photo of a {c}."` template, and print the softmax probabilities. Then measure the effect of prompt engineering — compare accuracy with bare class names against the templated-and-ensembled version, and you should see the templates help. Next, test the bag-of-words failure directly — encode "a cat to the left of a dog" and "a dog to the left of a cat" and compare their cosine similarity to a single image; you should find the model can barely distinguish them. Finally, build a tiny text-to-image search: embed 20 images once, then retrieve the top match for a free-text query and confirm it returns something sensible.
