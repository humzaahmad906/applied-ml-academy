# 11 — Self-Supervised Learning

Every model you have built so far leaned on labels. Transfer learning in the [training lesson](05-training-and-transfer-learning.md) worked because someone hand-labeled a million ImageNet photos, and the [vision transformer](07-vision-transformers.md) was hungrier for that labeled data still. But labels are the expensive part. A radiologist annotating tumors, a team drawing boxes for a self-driving fleet — that work is slow, costly, and caps how much data you can actually train on. The internet, meanwhile, has billions of *unlabeled* images sitting idle. Self-supervised learning (SSL) is the set of tricks that turns those raw images into a training signal without anyone labeling them, and it is how nearly every strong vision backbone in 2026 is now pretrained.

## Labels are the bottleneck

The core idea is to invent a task where the *data supplies its own answer*. You hide part of the input, or transform it, and ask the network to recover something you already know — so the "label" is generated for free from the image itself. Do this over enough images and the network is forced to learn general-purpose features (edges, textures, parts, objects) to succeed, and those features transfer to real tasks you *do* have labels for, just like supervised ImageNet pretraining did, only without the annotation bill.

The first wave of these invented tasks are called **pretext tasks**, and it is worth knowing them mostly to understand why they faded.

## Pretext tasks, and why they faded

Early SSL leaned on clever puzzles:

- **Rotation** — rotate an image by 0°, 90°, 180°, or 270° and train a 4-way classifier to predict the angle. To know a photo is upside down, the network has to recognize what is in it.
- **Jigsaw** — cut the image into tiles, shuffle them, and predict the permutation.
- **Colorization** — feed a grayscale image and predict the color channels.

These work, and they were an important proof that useful features can come from no labels at all. But they share a weakness: the network can often cheat by latching onto low-level cues (a patch of blue is probably sky, a chromatic-aberration edge betrays the tile boundary) rather than learning semantics. The features they produce are decent but not competitive. By around 2020 the field had converged on two families that dominate today — **contrastive learning** and **masked image modeling**.

## Contrastive learning: InfoNCE

The contrastive idea is simple to state. Take an image, make two different augmented views of it (a random crop, a color jitter, a flip). Those two views are a **positive pair** — they should map to nearby points in feature space. Every *other* image in the batch is a **negative** — it should map far away. The network learns by pulling positives together and pushing negatives apart.

The loss that makes this precise is **InfoNCE** (noise-contrastive estimation). For a query representation $q$ with its matching positive key $k^+$ and a set of negative keys $k^-_i$, using cosine similarity and a temperature $\tau$:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log
\frac{\exp\!\left(\mathrm{sim}(q, k^+)/\tau\right)}
{\exp\!\left(\mathrm{sim}(q, k^+)/\tau\right) + \sum_{i}\exp\!\left(\mathrm{sim}(q, k^-_i)/\tau\right)}
$$

Read it as a classification problem: among all the keys, identify which one is the positive. It is a softmax cross-entropy where the "correct class" is the positive pair. The temperature $\tau$ sharpens or softens that distribution — small $\tau$ makes the model focus hard on the toughest negatives. The quality of the features you get out depends heavily on having *many* good negatives, which is the practical tension the next two methods resolve differently.

## SimCLR: augmentation is the supervision

**SimCLR** is the clean, direct implementation of the idea. For each image in a batch, generate two augmented views, encode all of them with the same network plus a small projection head, and apply InfoNCE using the other views in the batch as negatives. There is no memory bank and no second network — just a big batch.

```python
import torch
import torch.nn.functional as F

def nt_xent(z1, z2, tau=0.5):
    # z1, z2: (N, D) projected embeddings of the two views
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2N, D)
    sim = z @ z.t() / tau                                # (2N, 2N) cosine / tau
    n = z1.size(0)
    # positive of row i is its partner view; build the target index
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    sim.fill_diagonal_(float("-inf"))                    # a view is not its own negative
    return F.cross_entropy(sim, targets)

torch.manual_seed(0)
z1, z2 = torch.randn(8, 128), torch.randn(8, 128)
print(round(nt_xent(z1, z2).item(), 3))
```

Two lessons hide in that code. First, **the augmentation policy *is* the task** — what you teach the model to be invariant to is exactly what you choose to augment away. SimCLR found strong color jitter and aggressive cropping essential; without them the model solves the puzzle with color histograms and learns little. Second, it needs a **large batch** (thousands) so that each positive is contrasted against enough negatives, which makes it compute-hungry.

## The projection head: train it, then throw it away

One detail in SimCLR trips people up and is worth stating plainly, because every method here does it. The InfoNCE loss is not applied to the backbone's features directly. A small MLP **projection head** sits on top of the backbone, and the contrastive loss acts on *its* output. But when you are done pretraining, you **discard the head and keep the backbone**. The features you actually transfer are the ones *below* the projection, not the ones the loss was computed on.

Why train a head only to throw it away? The projection lets the backbone hold on to information that is useful for downstream tasks but that the contrastive objective would otherwise destroy — color, orientation, and other properties the augmentations tell the model to ignore. The head absorbs that invariance so the backbone does not have to, leaving richer features underneath. It is a small architectural choice with a large effect on transfer quality, and forgetting to strip the head is a common source of disappointing linear-probe numbers.

## MoCo: a queue and a momentum encoder

**MoCo** (Momentum Contrast) removes the giant-batch requirement with two ideas. Keep a **queue** of encoded keys from recent batches so you have thousands of negatives without recomputing them, and encode those keys with a **momentum encoder** — a slowly-moving copy of the main network whose weights are an exponential moving average of it:

$$
\theta_k \leftarrow m\,\theta_k + (1 - m)\,\theta_q, \qquad m \approx 0.999
$$

Why the slow copy? The queue holds keys computed at slightly different past moments; if the key encoder changed fast, those old keys would be stale and inconsistent. A momentum encoder drifts slowly enough that keys stay comparable across batches. The payoff is a large, consistent negative set on a modest GPU budget — the same momentum-encoder trick reappears in the methods below.

## BYOL and DINO: dropping the negatives

Here is the surprise that reshaped the field: you may not need negatives at all. **BYOL** keeps two views and two networks — an online network trained by gradient descent and a momentum "target" network — and simply trains the online network to *predict* the target's representation of the other view. There is no push-apart term. Naively this should collapse (map everything to a constant and the loss is zero), but the momentum target plus a predictor head and stop-gradient prevent it. This is **self-distillation**: the network teaches itself using a slowly-updated version of itself as the teacher.

**DINO** brings self-distillation to the vision transformer and produces its most striking result. Student and teacher (again an EMA of the student) each see different crops; the student is trained to match the teacher's output distribution, with centering and sharpening on the teacher side to stop collapse. When you then look at the ViT's [self-attention](07-vision-transformers.md) maps, the `[CLS]` token attends to clean object boundaries — **the model learned to segment foreground objects with no segmentation labels, no masks, nothing but images**. That emergent structure is the clearest evidence that these objectives learn genuine semantics rather than shortcuts.

## One enemy in common: collapse

Once you see the sequence — negatives, then a queue, then a momentum teacher, then centering and sharpening — the design of every contrastive method makes sense as a defense against a single failure mode called **collapse**. Any objective that only rewards making two views agree has a trivial cheat: map *every* image to the same constant vector, and the views always agree perfectly. That solution has zero loss and zero usefulness.

The push-apart term in InfoNCE prevents it directly: you cannot collapse everything to one point if you are also being penalized for putting different images close together. The negative-free methods must prevent it structurally instead — BYOL's predictor and stop-gradient break the symmetry that would allow the shortcut, and DINO's teacher centering keeps the output distribution from concentrating on a single value. When you read a new SSL paper, the fastest way to understand it is to ask: *what stops this from collapsing?* The answer is usually the heart of the method.

## Masked image modeling: MAE

The other dominant family borrows directly from how language models pretrain — mask part of the input and reconstruct it. The **Masked Autoencoder (MAE)** does this for images and makes two choices that matter.

First, it masks **75%** of the image patches, far more than the ~15% used for text. Images are spatially redundant; a light mask can be solved by copying neighboring pixels, so you must hide most of the image to force real understanding. Second, it uses an **asymmetric encoder-decoder**: a heavy ViT encoder processes only the *visible* 25% of patches, and a lightweight decoder reconstructs the missing pixels from those plus mask tokens. Because the expensive encoder never sees the masked patches, pretraining is several times cheaper than it would otherwise be.

```python
def random_masking(x, mask_ratio=0.75):
    # x: (N, L, D) patch embeddings; returns kept patches + the mask
    N, L, D = x.shape
    keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    idx = noise.argsort(dim=1)                 # shuffle patch order
    keep_idx = idx[:, :keep]
    x_kept = torch.gather(x, 1, keep_idx.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(N, L, device=x.device)
    mask.scatter_(1, keep_idx, 0)              # 1 = masked, 0 = visible
    return x_kept, mask

torch.manual_seed(0)
x = torch.randn(2, 196, 768)                   # 14x14 patches, ViT-Base
x_kept, mask = random_masking(x)
print(x_kept.shape, int(mask[0].sum()))        # (2, 49, 768), 147 masked
```

Contrastive and masked methods have different flavors: contrastive features tend to be strong for linear probing off the shelf, while masked-modeling features often shine after fine-tuning. In practice the frontier has combined them.

## DINOv2: the 2026 default backbone

That combination is **DINOv2**, which pairs DINO-style self-distillation with a masked-modeling objective and trains on a large, carefully curated (but still unlabeled) image set. The result is a general-purpose ViT backbone whose frozen features are strong enough that you can attach a simple linear head for classification, a small decoder for depth or segmentation, or use them directly for retrieval — often with no fine-tuning at all. For most applied vision work in 2026, "load a pretrained DINOv2 backbone and put a small head on top" has become the sensible starting point, the way "load an ImageNet ResNet" was a few years ago. The same self-supervised recipe underlies the promptable [foundation models](06-detection-and-segmentation.md) like SAM you met earlier.

## Linear probe vs. fine-tune

Because SSL produces a backbone rather than a finished classifier, you evaluate its features two ways, and the distinction matters:

- **Linear probe** — freeze the entire backbone and train only a single linear layer on top for your labeled task. This measures how good and how *linearly separable* the learned features already are. It is cheap, fast, and the honest test of representation quality.
- **Fine-tune** — unfreeze the backbone and train it end-to-end on your labels. This almost always scores higher, but it also lets the network adapt (or overfit) to your specific data, so a strong fine-tune number can hide weak underlying features.

A third option is even simpler and popular for SSL: **kNN evaluation**. Freeze the backbone, embed the training set once, and classify a query image by the labels of its nearest neighbors in feature space — no training of any head at all. Because it fits nothing, it is the least forgiving and most direct probe of whether the raw feature geometry already groups classes together.

Report what you can. A method that wins on linear probe or kNN learned better *general* features; a method that only wins after full fine-tuning may just be a good initialization.

## Where this shows up

Self-supervised backbones are now the quiet default in production vision. When a company has a mountain of unlabeled domain images — satellite tiles, retail shelves, manufacturing-line frames, medical scans — pretraining a DINOv2-style model on that unlabeled pile and then fine-tuning on the small labeled set they can afford routinely beats training from ImageNet weights. It is also the standard first move in medical imaging, where labels are scarce and expensive, and it is the feature extractor behind large-scale image search and deduplication systems.

In robotics and autonomous driving the same logic holds even harder: sensors generate enormous streams of unlabeled frames, and self-supervised pretraining lets a perception stack learn from all of it while human annotation is reserved for the handful of safety-critical cases where a label truly pays for itself.

## Key takeaways

- Labels are the bottleneck; SSL invents tasks where the image supplies its own target, unlocking billions of unlabeled images.
- Pretext tasks (rotation, jigsaw, colorization) proved the concept but let networks cheat with low-level cues, so they faded.
- Contrastive learning pulls augmented views of the same image together and pushes others apart via the **InfoNCE** loss; SimCLR needs big batches, MoCo replaces them with a queue plus a momentum encoder.
- BYOL and DINO drop negatives entirely and use **self-distillation** with a momentum teacher; DINO's attention maps segment objects with no labels.
- **MAE** masks 75% of patches and reconstructs them with an asymmetric encoder-decoder; **DINOv2** combines the families into the go-to 2026 backbone.
- Evaluate features with a **linear probe** (frozen backbone, honest quality signal) and **fine-tuning** (end-to-end, higher but less diagnostic).

## Try it

Load a pretrained DINOv2 backbone (`torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`), freeze it, and extract features for a small labeled dataset of your choice. Train a single `nn.Linear` on top (a linear probe) and record the accuracy. Then unfreeze and fine-tune the whole model and compare. Separately, implement the SimCLR `nt_xent` loss above and confirm that as you make `z2` a near-copy of `z1`, the loss drops — the positives are being pulled together.

The features you just froze and probed learn *what* is in an image without ever being told. The next question is harder: can a network learn to *generate* new images at all? The [next module](12-generative-models-vae-gan.md) turns from understanding images to creating them, starting with VAEs and GANs — and you will see the encoder-decoder and the adversarial ideas that later power the diffusion models this course builds toward.
