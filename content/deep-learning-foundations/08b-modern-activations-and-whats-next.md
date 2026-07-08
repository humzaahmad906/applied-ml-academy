# 08b — Modern Activations and What's Next

You just built a full neural network from scratch — a neuron, a nonlinearity, stacked layers, backprop, a loss, an optimizer, a training loop, all wired together and confirmed by the curves. That skeleton is the real thing, and it scales. But two questions are hanging in the air. First: you used ReLU and sigmoid because they were simple to reason about — is that what the big models actually use? Second: you've built an MLP (a "multi-layer perceptron," fully connected layers stacked on top of each other) — is that the whole field, or the ground floor? This lesson answers both. It's a bridge: no new problem to solve, just a map of where the road goes from here.

## Activations after ReLU

ReLU (`max(0, x)`) is still a great default and still everywhere, especially in vision networks. It's cheap, it doesn't saturate on the positive side, and it made deep networks trainable in the first place. But it has a rough edge — literally. At zero it has a kink, and for any negative input the gradient is exactly zero. A neuron pushed into that negative region stops receiving gradient and can get stuck there permanently, a failure mode with the memorable name *dead ReLU*.

Modern architectures, especially transformers, mostly reach for **smooth** activations instead. The two you'll see everywhere:

- **GELU (Gaussian Error Linear Unit).** Instead of the hard on/off switch of ReLU, GELU multiplies the input by a smooth gate — roughly, "how far above zero is this input, in probabilistic terms." Small negative inputs still pass a little signal through instead of being clamped to zero. GELU was the default activation in BERT, GPT-2, and GPT-3, and it's still the standard in a huge fraction of transformer models.
- **SiLU / Swish.** `SiLU(x) = x · sigmoid(x)`. Same smooth, slightly-negative-allowing shape as GELU, but cheaper to compute (it's just a sigmoid and a multiply — using the exact same sigmoid you already wrote). At the scale of modern LLMs, SiLU tends to match or slightly beat GELU, and it's simpler, so a lot of newer models default to it. "SiLU" and "Swish" are two names for the same function.

Why does *smooth* help? Two reasons worth holding onto. A smooth curve has a smooth derivative, and gradient descent is happier with smooth gradients — no sudden cliffs where the gradient jumps from zero to one. And letting a trickle of signal through for small negative inputs means fewer neurons go permanently dead. It's a small change to the shape of one function, but across dozens of layers and billions of parameters it adds up to more stable training.

One more term you'll run into, worth knowing by name even though we won't build it: **GLU variants**, especially **SwiGLU**. These aren't just a different activation curve — they change the shape of the whole feed-forward block, splitting it into two paths where one path *gates* (multiplies, element-wise) the other. SwiGLU is the feed-forward design inside LLaMA, PaLM, DeepSeek, and most current large language models. If you remember one thing: the field moved from ReLU to smooth activations *and* from plain two-layer MLPs to these gated blocks. You now know enough vocabulary to read that sentence in a paper and not bounce off it.

## Vanishing and exploding gradients

Back in the activations lesson you saw that sigmoid flattens out at its tails, so its gradient goes to nearly zero there. Now that you've watched backprop run through a network, you can see why that matters at depth — and it deserves its proper name.

Backprop computes a gradient by *multiplying* together the local gradients of every layer on the way back from the loss. Chain rule, one factor per layer. If those factors are consistently a bit less than 1, multiplying twenty or fifty of them together drives the result toward zero: the early layers get a gradient so tiny they barely update. That's the **vanishing gradient** problem, and it's exactly why very deep networks were hard to train before people figured out better activations and initializations. Flip it around — if the factors are consistently bigger than 1, the product explodes, gradients become huge, and a single update flings the weights to nonsense (you'll often see the loss print as `NaN`). That's the **exploding gradient** problem.

Smooth activations and good initialization (like the He-style init in the last lesson) fight the vanishing side. For the exploding side there's a blunt, reliable tool: **gradient clipping**. Before applying the update, you measure the overall size (the *norm*) of the gradient vector, and if it's above a threshold you scale the whole thing down to that threshold — same direction, capped magnitude. It's a seatbelt: most steps it does nothing, but on the occasional step that would have exploded, it saves the run. Clipping by norm (rather than clipping each value independently) is the standard choice because it preserves the direction of the update. It's near-universal in training RNNs and transformers.

## The map: MLPs aren't the end

Here's the honest framing. What you've built — fully connected layers, every input wired to every neuron — is the *foundation*, not the destination. It works, but it treats all inputs as an unordered bag of numbers. Real data usually has structure, and the major architecture families are each a way of building that structure into the network instead of forcing it to learn from scratch.

- **Images → CNNs (convolutional neural networks).** A photo has spatial structure: nearby pixels relate, and an edge is an edge wherever it appears. CNNs bake that in with small filters that slide across the image and share weights, so the network doesn't have to relearn "what a corner looks like" in every position. This is the entire next course — see the **computer-vision** track to pick up from here.
- **Sequences → RNNs, then transformers.** Text, audio, and time series have order: what came before shapes what comes next. RNNs process a sequence step by step, carrying a hidden state. They mostly worked, but they struggled with long-range dependencies (partly *because* of the vanishing gradients above, stretched across time steps). Then **attention** — a mechanism that lets every position look directly at every other position — and the **transformer** architecture built on it changed everything. Transformers now dominate not just language but increasingly vision and multimodal models too. For where that road goes, see the **language-modeling** track and the **vlm-guide**.

The reassuring part: every one of these is still forward pass → loss → backprop → update, looped, on a proper data split. CNNs and transformers change *what a layer does*, not *how the network learns*. The training loop you wrote by hand in numpy is running, essentially unchanged, inside GPT and inside every image model. You didn't learn a toy — you learned the engine. What's next is a catalog of better bodies to put around it.

## A small taste in PyTorch

You don't need to build these from scratch — the frameworks provide them. Here's GELU as a drop-in activation and gradient clipping wired into a training step, so the names above are concrete:

```python
import torch
import torch.nn as nn

# A tiny MLP, but with GELU instead of ReLU — one line different from what you know
net = nn.Sequential(
    nn.Linear(2, 8),
    nn.GELU(),          # smooth activation; swap for nn.SiLU() to try Swish
    nn.Linear(8, 1),
)

opt = torch.optim.SGD(net.parameters(), lr=0.5)
loss_fn = nn.BCEWithLogitsLoss()   # sigmoid + BCE fused, numerically safer

# One training step, with gradient clipping as the seatbelt
opt.zero_grad()
pred = net(X_batch)                 # X_batch: a tensor of shape (n, 2)
loss = loss_fn(pred, y_batch)
loss.backward()                     # same backprop you derived by hand
nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # cap gradient size
opt.step()
```

Every piece here has a counterpart in the numpy network you already wrote. `nn.GELU()` replaces the hand-written `relu`. `loss.backward()` is your `backward()` function, computed automatically. `clip_grad_norm_` is the one genuinely new idea — the seatbelt for exploding gradients — and it's a single line.

## Key takeaways

- **ReLU is still a fine default**, but modern transformers favor **smooth activations** — GELU (BERT, GPT-2/3) and SiLU/Swish (`x · sigmoid(x)`, cheaper, common in newer models). Smooth curves give smooth gradients and fewer permanently-dead neurons.
- **GLU variants like SwiGLU** change the whole feed-forward block into a gated design and are standard in current LLMs (LLaMA, PaLM). Know the name; you'll meet it.
- **Vanishing / exploding gradients** come from *multiplying* per-layer gradients across a deep stack. Smooth activations and good init help the vanishing side; **clip the gradient by norm** to tame the exploding side — a near-universal seatbelt in RNN and transformer training.
- **The MLP is the foundation, not the field.** Images → CNNs (the computer-vision course), sequences → RNNs then transformers, with **attention** now dominant (language-modeling, vlm-guide).
- Under every one of those, it's still forward → loss → backprop → update. You learned the engine; what's next is better bodies around it.

## Try it

Take the PyTorch snippet above and (1) swap `nn.GELU()` for `nn.ReLU()` and then `nn.SiLU()` — plot each activation over the range `-4` to `4` with `torch.linspace` and eyeball how ReLU's hard corner differs from the two smooth curves. (2) Print the gradient norm *before* clipping (`clip_grad_norm_` returns it) across a few steps — on this tiny problem it'll sit well under 1.0, so clipping never fires; that's the point of a seatbelt. (3) Write yourself two sentences: which architecture family you'd reach for if your next dataset were photographs, and which if it were sentences — and why the plain MLP would be the wrong tool for both.

---

*Sources on the current landscape: [SwiGLU: The Activation Function Powering Modern LLMs](https://saeedmehrang.github.io/blogs/language-modeling/llm-2025-overview/swiglu/), [SiLU / Swish — Sebastian Raschka](https://sebastianraschka.com/llms-from-scratch/ch04/11_silu/), [Activation Functions — ReLU, GELU, SiLU, and SwiGLU](https://tutorialq.com/ai/dl-foundations/activation-functions).*
