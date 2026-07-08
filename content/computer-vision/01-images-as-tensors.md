# 01 — Images as Tensors

Before a neural network can learn anything about a photo, the photo has to become numbers. A model never "sees" a cat — it sees a grid of intensities that, after training, it has learned to associate with the label "cat." So the very first thing to get right in computer vision is the representation: how an image turns into a tensor, what each dimension means, and why the *shape* of that tensor matters enormously for the kind of model you should reach for.

## A pixel is just a number

A grayscale image is a rectangular grid of pixels, and each pixel is a single number describing brightness. In the most common convention that number is an integer from 0 (black) to 255 (white), because it fits in one byte. A small 28×28 grayscale image — the size of a handwritten digit in the classic MNIST dataset — is therefore a grid of 784 numbers.

```python
import numpy as np

# a tiny 3x3 grayscale image: 0 = black, 255 = white
img = np.array([
    [  0,   0, 255],
    [  0, 128, 255],
    [255, 255, 255],
], dtype=np.uint8)

print(img.shape)   # (3, 3)
print(img.dtype)   # uint8
```

That `(3, 3)` shape is `(height, width)`. The first index picks a row (how far down), the second picks a column (how far across). Already a subtle point: images are indexed row-first, which is `(H, W)`, not `(x, y)`. Mixing these up is one of the most common early bugs.

## Color adds a channel dimension

A color image can't be described by a single brightness per pixel. The standard trick is to store three numbers per pixel — how much **R**ed, **G**reen, and **B**lue light it emits. Any color you can see on a screen is a mix of those three. So a color image is three stacked grids, one per **channel**.

That gives a three-dimensional tensor. There are two conventions for the order:

- **H×W×C** — height, width, channels. This is how libraries like NumPy, PIL, and OpenCV store images, and how you'd think of "a grid of pixels, each holding 3 values."
- **C×H×W** — channels first. This is what PyTorch expects for its models.

```python
# a 32x32 RGB image, channels-last (the "natural" layout)
rgb = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
print(rgb.shape)          # (32, 32, 3)  -> H, W, C

# PyTorch wants channels-first: C, H, W
import torch
t = torch.from_numpy(rgb).permute(2, 0, 1)
print(t.shape)            # torch.Size([3, 32, 32])
```

The `permute(2, 0, 1)` reorders the axes: old axis 2 (channels) moves to the front, then height, then width. Nothing about the data changes — only the bookkeeping of which axis is which. Remembering that **torchvision and `nn.Conv2d` speak C×H×W** will save you a lot of shape errors.

## Batches: the fourth dimension

Networks almost never process one image at a time during training — they process a **batch** of them together, because GPUs are far more efficient at chewing through many examples in parallel. So we prepend one more axis for the batch:

```
(N, C, H, W)
 │  │  │  └── width
 │  │  └───── height
 │  └──────── channels (3 for RGB, 1 for grayscale)
 └─────────── batch size: how many images at once
```

A batch of 64 color images at 224×224 (a very common size for pretrained models) is a tensor of shape `(64, 3, 224, 224)`. That single 4-D tensor is what flows into the first layer of nearly every vision model you'll build. When a model throws a shape error, your first move should be to `print(x.shape)` and check it against `(N, C, H, W)`.

## Normalization: from 0–255 to something a network likes

Raw pixel values run 0–255, but neural networks train best when their inputs are small and centered near zero. Large, all-positive inputs make gradients unbalanced and slow learning — the same reasoning behind zero-centered activations. So we always **normalize** before feeding pixels to a model.

The simplest step is to scale into `[0, 1]` by dividing by 255. A stronger, standard step is to then subtract a mean and divide by a standard deviation per channel, so each channel is centered at 0 with roughly unit spread:

```python
x = t.float() / 255.0              # now in [0, 1]

# per-channel normalize (these are the famous ImageNet stats)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
x = (x - mean) / std               # centered near 0, per channel
print(x.mean().item(), x.std().item())   # roughly 0-ish, ~1-ish
```

Those specific numbers `[0.485, 0.456, 0.406]` are the mean pixel values of the ImageNet dataset. If you use a model pretrained on ImageNet (which you will, constantly), you **must** normalize your inputs with the same stats the model was trained on, or its learned features won't line up with what it sees. This is such a common requirement that modern torchvision bundles the correct transforms with each set of pretrained weights — more on that in the transfer-learning lesson.

## Why MLPs waste an image's structure

Here's the motivating problem for this whole course. Suppose you flatten that 28×28 digit into a 784-length vector and feed it to a multilayer perceptron (MLP), exactly as you learned in the deep-learning course. It works — MLPs can classify MNIST — but it's doing so wastefully, for two deep reasons.

**First, it destroys spatial structure.** Flattening throws away the fact that pixel (5, 5) sits right next to pixel (5, 6). To the MLP, the 784 inputs are just an unordered list; the notion that some pixels are neighbors and form edges, corners, and textures is gone. The network has to *relearn* geometry from scratch, from data, which takes far more examples.

**Second, it doesn't reuse what it learns.** In an MLP's first layer, every input pixel connects to every hidden unit with its own weight. A feature detector that learns "there's a vertical edge here" in the top-left corner has a completely separate set of weights from one that would detect the same vertical edge in the bottom-right. The network can't share the concept of "vertical edge" across locations — it must learn it independently everywhere. A 224×224 color image flattened is 150,528 inputs; a single hidden layer of 1,000 units then needs over 150 million weights just for layer one. That's enormous, slow, and prone to overfitting.

The fix, which the next lesson builds, is the **convolution**: a small, reusable filter that slides across the image, sharing one set of weights across every location and preserving the grid. That single idea — local, shared, position-aware filters — is what makes convolutional neural networks so much better suited to images than plain MLPs.

## Why this matters for ML

Almost every vision bug you'll hit early on is a representation bug, not a modeling bug: an image in H×W×C when the model wanted C×H×W, a batch dimension you forgot to add, pixels left in 0–255 when the model expected normalized floats, or the wrong normalization stats for a pretrained backbone. Getting the tensor right is not a preliminary you rush past — it's half of making a vision model work at all. And understanding *why* the naive flatten-into-an-MLP approach is wasteful is exactly the motivation for everything that follows.

## Key takeaways

- An image is a tensor of pixel intensities; grayscale is `(H, W)`, color adds a channel axis for R, G, B.
- Watch the layout: NumPy/PIL use **H×W×C**, but PyTorch models expect **C×H×W**. Use `permute` to convert.
- Training feeds a **batch**, so the full tensor is `(N, C, H, W)` — check this shape first when debugging.
- Always **normalize** pixels (at least ÷255; usually also per-channel mean/std). Match a pretrained model's original stats exactly.
- Flattening an image into an MLP throws away spatial structure and can't share features across locations — the motivation for convolutions.

## Try it

Load any image with PIL (`from PIL import Image`), convert it to a NumPy array, and print its shape and dtype. Then turn it into a normalized PyTorch tensor of shape `(1, 3, H, W)` — channels-first, scaled to `[0, 1]`, with a batch dimension of 1. Finally, compute how many weights a single dense layer of 512 units would need if you flattened this image and fed it to an MLP. Compare that number to the ~5×5×3×64 ≈ 4,800 weights a small convolutional filter bank uses, and write one sentence on why that gap matters.
