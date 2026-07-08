# 04 — Classic Architectures

You now know how to stack conv, activation, and pooling into a working CNN. The natural next question is: how big, how deep, arranged how? For a decade the field answered that question with a series of landmark architectures, each fixing a specific problem the last one hit. Walking through them — LeNet, AlexNet, VGG, ResNet — isn't history for its own sake. Each design encodes a lesson about *why* CNNs are built the way they are, and ResNet's key idea (skip connections) is still in essentially every modern network, including the transformers you'll meet later.

## LeNet-5 (1998): the template

LeNet, built by Yann LeCun to read handwritten digits on bank checks, is the CNN you already know how to build. Two convolutional layers with pooling in between, then a couple of dense layers to a 10-class output. By modern standards it's tiny — about 60,000 parameters — and it used tanh activations rather than ReLU. But it established the enduring template: **convolve to extract features, pool to downsample, then classify with dense layers.** Every network below is a scaling-up of this same skeleton.

For years LeNet-scale networks stalled. The data (small labeled sets), the compute (no GPUs), and a few training tricks weren't there yet. Then all three arrived at once.

## AlexNet (2012): the breakthrough

AlexNet is the network that started the deep learning boom. Entered in the ImageNet competition — classifying 1.2 million images into 1,000 categories — it crushed the field, cutting the error rate by roughly a third overnight and convincing the world that deep CNNs work. It was structurally a bigger LeNet (five conv layers, three dense layers, ~60 million parameters), but it combined several things that mattered:

- **ReLU activations** instead of tanh, which trained far faster and sidestepped vanishing gradients (the reason ReLU became the default everywhere).
- **GPU training**, splitting the model across two GPUs to make the compute feasible at all.
- **Dropout** in the dense layers to fight overfitting on that many parameters.
- **Data augmentation** — random crops and flips — to stretch the training set.

The takeaway: the ideas weren't all new, but **scale plus a few practical fixes** turned CNNs from a niche method into the dominant one. Most of AlexNet's tricks (ReLU, dropout, augmentation, GPUs) are still standard practice.

## VGG (2014): deeper, with a simple rule

VGG asked a clean question: what if we just go deeper, using only small 3×3 convolutions throughout? Its rule was almost monotonous — stack 3×3 convs, occasionally max-pool to halve the spatial size and double the channels, repeat until you have 16 or 19 layers. The insight it popularized: **two stacked 3×3 convolutions have the same receptive field as one 5×5, but with fewer parameters and an extra nonlinearity in between.** Small filters, stacked deep, beat large filters.

VGG's weakness was its size. Its dense layers made it huge — over 130 million parameters, most of them in the classifier head — and it's slow and memory-hungry. But its clean, uniform design made it a favorite for feature extraction, and "stacks of 3×3 convs" became the default building block.

## The problem VGG hit: deeper stopped helping

Here's the wall the field ran into around 2015. If depth is so good, just keep adding layers — 50, 100, 150. But experiments showed the opposite: past a point, deeper plain networks got *worse*, and not from overfitting — their **training** error went up too. A 56-layer plain net underperformed a 34-layer one on the training set itself.

The culprit is optimization. In a very deep stack, gradients have to flow backward through dozens of layers, and they tend to shrink (or blow up) along the way — the **vanishing gradient** problem again, now at architectural scale. The early layers get almost no useful signal, so the network can't even learn to match a shallower one. Depth, the thing that made CNNs powerful, had become the thing blocking further progress.

## ResNet (2015): skip connections

ResNet's fix is one of the most important ideas in deep learning, and it's beautifully simple. Instead of asking a block of layers to compute a full transformation `H(x)`, ask it to compute only the *change* — the **residual** `F(x)` — and add the input back:

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU()

    def forward(self, x):
        identity = x                       # the "skip" / shortcut
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity               # add the input back
        return self.relu(out)
```

That `out + identity` line — the **skip connection** (or shortcut) — changes everything. Two things follow:

**Gradients get a highway.** During backprop, the addition passes the gradient straight through to earlier layers unchanged, alongside the path through the convs. Even in a 152-layer network, early layers now receive a strong learning signal, so vanishing gradients stop blocking depth.

**Layers only need to learn a small correction.** If a block isn't helping, it can drive `F(x)` toward zero and just pass the input through unchanged — so adding layers can't easily make things *worse*. This made networks of 50, 101, even 152 layers not only trainable but genuinely better, and ResNet won ImageNet 2015 decisively.

The code above also shows **batch normalization** (`nn.BatchNorm2d`), a companion technique from the same era that normalizes each layer's activations during training. It stabilizes and speeds up training of deep nets and appears in nearly every architecture since. ResNet-50 remains, a decade later, one of the most common backbones in practice — the sensible default when someone says "just use a CNN."

## A one-line nod to what came after

Progress didn't stop at ResNet, but the later refinements are variations on these themes:

- **EfficientNet (2019)** studied how to scale a network's depth, width, and input resolution *together* in a principled ratio, hitting a given accuracy with far fewer parameters — the go-to when compute or model size is tight.
- **ConvNeXt (2022)** took a plain ResNet and modernized it piece by piece with tricks borrowed from vision transformers (larger kernels, fewer activations, different normalization), showing that a well-tuned pure CNN can still match transformers on large-scale benchmarks.

You rarely implement these from scratch. In practice you load one pretrained (the next lesson), and knowing this lineage tells you *which* to reach for: ResNet-50 as a robust default, EfficientNet when size matters, ConvNeXt for a modern high-accuracy CNN.

## Why this matters for ML

You will almost never design a classification backbone from zero — you'll pick a proven one and fine-tune it. This lesson is how you make that choice intelligently, and how you read a model card that says "ResNet-50 backbone" and know what that implies about depth, skip connections, and normalization. More broadly, the residual connection is not a CNN-only trick: it's in transformers, diffusion models, and LLMs. Understanding *why* it exists — to let gradients survive extreme depth — is understanding a load-bearing idea of all modern deep learning.

## Key takeaways

- **LeNet** set the template (conv → pool → dense); **AlexNet** scaled it up with ReLU, dropout, augmentation, and GPUs, and launched the deep learning era.
- **VGG** showed that deep stacks of small 3×3 convs beat shallow stacks of large filters — but its dense head made it huge.
- Plain networks stop improving past a certain depth because gradients vanish over many layers — deeper made *training* error worse.
- **ResNet's skip connections** (`out + input`) give gradients a highway and let blocks learn small corrections, making 50–152 layer networks trainable and better. Batch norm helps too.
- Later nets (**EfficientNet**, **ConvNeXt**) refine scaling and borrow from transformers; in practice you pick a pretrained backbone rather than build one.

## Try it

Load a pretrained ResNet-18 with `torchvision.models.resnet18(weights="DEFAULT")` and print it. Find the residual blocks and count the layers. Then contrast: instantiate `torchvision.models.vgg16` and compare the two models' parameter counts with `sum(p.numel() for p in m.parameters())`. Which is deeper, and which has more parameters? Write one sentence explaining why the deeper network can have *fewer* parameters than the shallower one.
