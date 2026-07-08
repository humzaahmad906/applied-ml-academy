# 02 — The Convolution Operation

The last lesson ended with a promise: instead of connecting every pixel to every neuron, we'll use a small, reusable filter that slides across the image. That filter, and the sliding, is the **convolution** — the single operation that defines convolutional neural networks. Once you see how one small kernel can detect an edge anywhere in an image using a handful of shared weights, the whole architecture makes sense.

## A kernel is a tiny window of weights

A **kernel** (or **filter**) is a small grid of numbers — typically 3×3 or 5×5. You slide it across the image, and at every position you multiply the kernel's numbers by the pixels underneath it, add up the results, and write that single sum into an output grid. That's it. The kernel is a little pattern-matcher: it produces a large value where the image locally looks like the kernel, and a small value where it doesn't.

Here's the whole operation on one patch, by hand:

```python
import numpy as np

patch = np.array([          # a 3x3 patch of the image
    [10, 10, 10],
    [10, 10, 10],
    [80, 80, 80],
])

kernel = np.array([         # a horizontal-edge detector
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
])

response = (patch * kernel).sum()
print(response)             # 210  -> a strong edge response
```

The kernel above has negatives on top and positives on bottom. Where the image is flat (top rows equal bottom rows) the response is near zero. Where there's a sharp jump from dark to light going downward — a horizontal edge — the response is large. We didn't tell the kernel what an edge is; we chose weights that *happen* to respond to edges. In a CNN, the network **learns** these weights from data.

## Sliding the window: the feature map

To convolve the whole image, we place the kernel at the top-left, compute one response, slide it one pixel right, compute again, and continue across and down. The grid of responses we build up is called a **feature map** (or activation map): a new "image" where each pixel says how strongly the kernel's pattern appeared at that location.

```python
def convolve2d(img, kernel):
    kh, kw = kernel.shape
    H, W = img.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = img[i:i+kh, j:j+kw]
            out[i, j] = (patch * kernel).sum()
    return out

img = np.random.randint(0, 255, (8, 8)).astype(float)
fmap = convolve2d(img, kernel)
print(fmap.shape)          # (6, 6)  -> smaller than the 8x8 input
```

Notice the output shrank from 8×8 to 6×6. A 3×3 kernel can't be centered on the very edge pixels — it would hang off the image — so we lose a one-pixel border. In general, a `k×k` kernel on an `H×W` image gives an `(H-k+1)×(W-k+1)` output. We'll fix that shrinkage with padding in a moment.

## Real CNNs use many filters, across channels

One kernel detects one kind of pattern. A convolutional layer uses **many** kernels in parallel — say 64 of them — each learning to detect something different: one for vertical edges, one for horizontal, one for a particular color blob, and so on. Each kernel produces its own feature map, so a layer with 64 filters turns its input into a stack of 64 feature maps. Those 64 maps become the *channels* of the layer's output, which the next layer's filters read in turn.

And on a color input, a kernel isn't really 3×3 — it's 3×3×(number of input channels). A filter in the first layer of an RGB network is 3×3×3: it looks at all three color channels at once and sums across them to produce a single number. So a filter's full shape is `(out_channels, in_channels, kh, kw)`. This is exactly what PyTorch stores:

```python
import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
print(conv.weight.shape)   # torch.Size([64, 3, 3, 3])
```

That's 64 filters, each 3×3 over 3 input channels — 1,728 weights plus 64 biases. Compare that to the 150-million-weight dense layer from the last lesson. The savings come from two properties we look at next.

## Stride and padding: controlling the output size

Two knobs shape the geometry of the output.

**Stride** is how many pixels the kernel jumps each step. Stride 1 (the default) moves one pixel at a time and looks at every position. Stride 2 skips every other position, which roughly halves the output's height and width — a cheap way to downsample and shrink the spatial size as you go deeper.

**Padding** adds a border of zeros around the input so the kernel can sit on the edge pixels. With the right padding (for a 3×3 kernel, one pixel of zeros all around), the output keeps the same height and width as the input — often called "same" padding. Without it, every conv layer shrinks the image, and a deep stack would erode away to nothing.

```python
# same-size output: 3x3 kernel, padding=1, stride=1
same = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)

# downsampling: stride 2 roughly halves H and W
down = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
```

The general output-size formula is `out = (in + 2*padding - kernel) / stride + 1`, rounded down. It's worth keeping handy, because getting spatial sizes to line up is a routine part of designing a CNN.

## The two ideas that make convolution powerful

Everything above rests on two properties that give CNNs their edge over MLPs on images.

**Parameter sharing.** The same kernel — the same handful of weights — is applied at every position in the image. A network doesn't learn a separate "vertical edge" detector for each location; it learns *one* and reuses it everywhere. That's why a conv layer needs thousands of weights where an MLP needed millions, and why CNNs generalize from far less data.

**Translation equivariance.** Because the same kernel slides everywhere, if the pattern it detects moves — a cat shifts to the right — the response in the feature map moves right by the same amount. The detector doesn't care *where* the pattern appears; it fires wherever it is. This matches how images actually work: a cat is a cat in the corner or the center. An MLP, with separate weights per position, has no such built-in guarantee and must learn location-invariance the hard way.

These two properties aren't tuning tricks; they're the **inductive bias** of convolutions — assumptions baked into the architecture that happen to be true for natural images. That bias is exactly why CNNs learn vision efficiently, and (as a later lesson explores) it's also why vision transformers, which lack it, need much more data to compete.

## Why this matters for ML

Convolution is the load-bearing idea in computer vision. Understanding it as "a small learned pattern-matcher slid across the image, with weights shared everywhere" demystifies almost everything downstream: why feature maps have the shapes they do, why we pad and stride, why CNNs need so much less data than a naive dense network, and why they're naturally robust to objects moving around the frame. When you later stack these layers, tune kernel sizes, or debug a shape mismatch, you're reasoning directly about the operation in this lesson.

## Key takeaways

- A **kernel** is a small grid of weights; convolution slides it over the image, computing a dot product at each position to produce a **feature map**.
- A conv layer uses **many** filters, each producing one output channel; a filter spans all input channels, so its shape is `(out_ch, in_ch, kh, kw)`.
- **Stride** controls the step size (stride 2 downsamples); **padding** adds a zero border so outputs don't shrink. Output size = `(in + 2p - k)/s + 1`.
- **Parameter sharing** (one kernel reused everywhere) makes CNNs far smaller than MLPs; **translation equivariance** means detectors fire wherever a pattern appears.
- These built-in assumptions — the convolutional inductive bias — are why CNNs learn vision efficiently from limited data.

## Try it

Take a grayscale image and apply three hand-built 3×3 kernels with the `convolve2d` function above: a horizontal-edge detector, its transpose (a vertical-edge detector), and a 3×3 box blur (all values 1/9). Print the shape of each feature map and describe what each one emphasizes. Then create an `nn.Conv2d(1, 8, kernel_size=3, padding=1)`, pass a `(1, 1, 28, 28)` tensor through it, and confirm the output shape is `(1, 8, 28, 28)` — explaining where the 8 and the preserved 28×28 come from.
