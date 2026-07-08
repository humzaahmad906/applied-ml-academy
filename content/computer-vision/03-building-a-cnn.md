# 03 — Building a CNN

We now have the one operation that matters — convolution — and we know why it beats a dense layer on images. But a single conv layer only detects simple, local patterns like edges. Real recognition needs to build from edges to textures to parts to whole objects. That's what a full **convolutional neural network** does: it stacks conv layers into a pipeline where each layer sees a little more of the image and describes it a little more abstractly. This lesson assembles that stack.

## The repeating block: conv → activation → pooling

Almost every classic CNN is built from one repeating motif:

1. **Convolution** — apply a bank of learned filters to produce feature maps.
2. **Activation** — pass the feature maps through a nonlinearity, almost always ReLU, so the network can represent non-linear patterns (the same reasoning as in any neural net).
3. **Pooling** — shrink the spatial size, condensing each small neighborhood into a summary.

We already know convolution and ReLU. **Pooling** is the new piece. Max pooling with a 2×2 window slides over the feature map in non-overlapping 2×2 blocks and keeps only the largest value in each block, throwing away three-quarters of the numbers and halving height and width:

```python
import torch
import torch.nn as nn

x = torch.tensor([[[[1., 3., 2., 4.],
                    [5., 6., 1., 2.],
                    [0., 1., 8., 9.],
                    [2., 3., 4., 7.]]]])   # shape (1, 1, 4, 4)

pool = nn.MaxPool2d(kernel_size=2)
print(pool(x).shape)      # torch.Size([1, 1, 2, 2])
print(pool(x))
# tensor([[[[6., 4.],
#           [3., 9.]]]])   each value is the max of its 2x2 block
```

Pooling does two useful things. It reduces computation and memory by shrinking the maps. And it adds a little **translation invariance**: if the strongest response wiggles by a pixel, the max is usually unchanged, so the network cares *that* a feature is present in a region more than exactly where. (Many modern architectures downsample with strided convolutions instead of pooling, but the concept — periodically shrink the spatial size — is the same.)

## The core trade-off: channels grow, space shrinks

Watch what happens as data flows through the stack. Each pooling step (or strided conv) halves the height and width. Meanwhile, we deliberately *increase* the number of filters in deeper conv layers — 32, then 64, then 128 channels. This is the defining rhythm of a CNN:

> **Spatial dimensions shrink; channel depth grows.**

The intuition is a trade of *where* for *what*. Early layers keep high spatial resolution but few channels: they know precisely where things are but describe them crudely (just edges). Deep layers have tiny spatial maps but many channels: they've lost fine position but describe rich, abstract concepts ("this region contains something fur-like and eye-like"). A 224×224×3 image might become 7×7×512 near the end — almost no spatial resolution, but 512 semantic feature channels.

## The receptive field: how a neuron sees more of the image

Here's the idea that makes stacking worthwhile. A single 3×3 conv sees only a 3×3 patch of its input. But stack a second 3×3 conv on top, and each of *its* neurons reads a 3×3 patch of the first layer's output — and each of those already summarized a 3×3 patch of the original image. So a neuron two layers deep effectively depends on a 5×5 region of the input. This growing window is the **receptive field**.

Add pooling and the receptive field grows even faster, because each pooled neuron already covers a shrunk-down, wider area. After a handful of conv-and-pool blocks, a single deep neuron's receptive field can span the entire image — which is exactly what you need to recognize a whole object. So depth isn't just "more capacity"; it's the mechanism by which local edge-detectors compose into global object-detectors. The hierarchy is real: edges → textures → object parts → whole objects, one level per few layers.

## From feature maps to a prediction

The conv-pool tower produces a compact stack of feature maps — say `(128, 4, 4)`. To turn that into a class prediction we need a fixed-length vector, so we **flatten** the maps into one long vector and pass it through a small MLP "head" ending in one output per class:

```python
feat = torch.randn(1, 128, 4, 4)
flat = feat.flatten(start_dim=1)   # keep batch dim, flatten the rest
print(flat.shape)                  # torch.Size([1, 2048])
```

Flattening here is fine, and different from the mistake in lesson 1. Back then we flattened *raw pixels*, destroying structure before any spatial processing. Here we flatten only *after* the conv stack has already extracted spatial features — we're summarizing learned concepts, not throwing away geometry. (A common modern alternative is global average pooling, which averages each channel's map down to a single number, giving a 128-vector directly and coping with variable input sizes.)

## A small CNN in PyTorch

Here's a complete, runnable classifier for 28×28 grayscale digits, assembling everything above:

```python
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1 -> 32 channels
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 -> 64 channels
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7
        )
        self.head = nn.Sequential(
            nn.Flatten(),                # (N, 64, 7, 7) -> (N, 3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes), # one logit per class
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

model = SmallCNN()
dummy = torch.randn(8, 1, 28, 28)        # a batch of 8 images
out = model(dummy)
print(out.shape)                         # torch.Size([8, 10])
```

Trace the shapes. Input `(8, 1, 28, 28)`. After the first conv+pool block, `(8, 32, 14, 14)` — channels up to 32, spatial halved. After the second, `(8, 64, 7, 7)` — channels 64, spatial halved again. Flatten to `(8, 3136)`, dense down to 128, then to 10 logits. Those 10 numbers per image are the raw class scores; a softmax and cross-entropy loss (covered in the training lesson) turn them into a trained classifier. The whole model has a few hundred thousand parameters — orders of magnitude smaller than the MLP we sketched in lesson 1, and far more accurate on images.

## Where the parameters actually live

It's worth pausing on where those parameters sit, because it explains a design tension in every CNN. The conv layers are cheap: the first has `1×32×3×3 + 32 = 320` parameters, the second `32×64×3×3 + 64 = 18,496` — because of parameter sharing, kernel count and size are all that matter, not image size. The dense head is where the weight explodes: `3136×128 = 401,408` parameters in the first linear layer alone. So a CNN spends most of its convolutional budget learning transferable feature detectors, but most of its *parameter* budget on the final classifier — which is exactly why the classification head is the part we replace and retrain in transfer learning, and why architectures like ResNet shrink or eliminate the dense head with global average pooling. Keeping an eye on where parameters concentrate tells you where a model is most likely to overfit and what's cheap to change.

## A note on kernel and pool choices

You'll see the same handful of settings over and over, and they're conventions worth internalizing rather than tuning blindly. 3×3 convolutions with padding 1 (keeping spatial size fixed) are the near-universal default — small enough to be cheap, and, stacked, they build up any receptive field you need. 2×2 max pooling with stride 2 (exactly halving the map) is the standard downsampler. Channel counts almost always grow in powers of two — 32, 64, 128, 256 — roughly doubling each time the spatial size halves, so the total amount of information per layer stays in the same ballpark as depth increases. When you read an unfamiliar architecture, these patterns let you predict its shapes before running a single tensor through it.

## Why this matters for ML

Every image classifier you'll build or fine-tune — from a toy MNIST net to a giant ResNet — is this same skeleton: a tower of conv-activation-downsample blocks that grows channels while shrinking space, topped by a small head that maps features to classes. Reading a model's architecture, debugging a shape mismatch, or deciding where to cut a network for transfer learning all come down to tracking the `(N, C, H, W)` tensor through exactly the stages in this lesson. The receptive-field idea, in particular, explains *why* these networks need to be deep at all.

## Key takeaways

- A CNN stacks a repeating block: **conv → ReLU → downsample** (max pooling or strided conv), then a small dense head for the final prediction.
- **Pooling** shrinks the spatial size and grants a bit of translation invariance by summarizing local neighborhoods.
- The defining rhythm: **spatial dimensions shrink while channel depth grows** — trading fine position for rich, abstract features.
- The **receptive field** of a neuron widens with depth, so deep neurons can "see" the whole image — this is why depth lets edges compose into objects.
- Flatten (or global-average-pool) only *after* the conv tower; flattening raw pixels, as an MLP does, is the mistake CNNs are designed to avoid.

## Try it

Build the `SmallCNN` above and, after each layer in `self.features`, print the tensor shape (add temporary prints inside `forward`, or run the layers one at a time). Confirm the channels-up / spatial-down pattern with your own eyes. Then change the first conv to `out_channels=16` and the second to `out_channels=32`, fix the `nn.Linear` input size accordingly, and verify the model still produces a `(N, 10)` output. Finally, reason on paper: after two 3×3 convs (no pooling), how many input pixels does one output neuron depend on?
