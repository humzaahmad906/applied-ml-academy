# 05 — Loss Functions and Optimizers

Backpropagation gives us gradients — but gradients *of what*, and what do we *do* with them? This module fills in both ends. The **loss function** defines the single number the network is trying to make small; it's the target of the whole enterprise. The **optimizer** is the rule that turns gradients into actual weight updates. Get these two right and training works; get them wrong and the network flails.

## The loss function: what "wrong" means

A loss function takes the network's prediction and the true answer and returns one number: how wrong was that prediction. Bigger loss = more wrong. Training is nothing more than adjusting weights to drive the average loss down across your data. Different tasks need different notions of "wrong."

### Mean squared error (regression)

When you're predicting a continuous number (a price, a temperature), squared error is the natural choice:

```python
import numpy as np

def mse(pred, target):
    return np.mean((pred - target)**2)
```

It penalizes large mistakes much more than small ones (the square), and it's smooth, which optimizers like. Its gradient is simply proportional to `(pred - target)` — the further off you are, the bigger the push.

### Cross-entropy (classification)

When you're predicting a probability, squared error is a poor fit. Instead we use cross-entropy, which measures how surprised the network is by the true label. For binary classification with a sigmoid output:

```python
def binary_cross_entropy(pred, target, eps=1e-9):
    pred = np.clip(pred, eps, 1 - eps)   # avoid log(0)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
```

If the true label is 1 and the network confidently predicts 0.99, the loss is tiny. If it confidently predicts 0.01, the `log` term explodes — a big penalty for being confidently wrong. That steep penalty is exactly what pushes classifiers to calibrate. For multi-class problems, the softmax output pairs with categorical cross-entropy, the same idea generalized across several classes.

The rule of thumb: **squared error for regression, cross-entropy for classification.** Match the loss to the task and the output activation, and gradients behave.

## The optimizer: turning gradients into steps

Once backprop gives us `grad` for every weight, we update. The simplest rule is **gradient descent**: step each weight a little in the direction that lowers the loss (i.e. opposite the gradient).

```python
w = w - learning_rate * grad
```

That `learning_rate` (often `lr`) controls step size, and it's the single most important knob in all of training. Too small and learning crawls. Too large and you overshoot the minimum, bouncing around or diverging entirely. Typical starting values sit around `0.01` to `0.001`, but it's problem-dependent.

### SGD

Computing the gradient over your *entire* dataset before each step is expensive. **Stochastic Gradient Descent (SGD)** instead estimates the gradient from a small random mini-batch of examples, and steps immediately. The estimate is noisy, but you take vastly more steps per pass over the data, and the noise can even help escape shallow bad spots. Plain mini-batch SGD is the baseline every other optimizer improves on:

```python
def sgd_step(params, grads, lr):
    return [p - lr * g for p, g in zip(params, grads)]
```

### Momentum

SGD can be slow when the loss surface is a long narrow valley — it zig-zags across the walls instead of rolling down the floor. **Momentum** fixes this by accumulating a running average of past gradients, like a ball gathering speed downhill. It damps the zig-zag and accelerates along consistent directions.

```python
# velocity carries over between steps; beta ~ 0.9
velocity = beta * velocity + (1 - beta) * grad
w = w - lr * velocity
```

The `velocity` term remembers where we've been heading. Consistent gradients build it up; noisy back-and-forth gradients cancel out. The result is faster, smoother descent.

### Adam

**Adam** is the workhorse default for most deep learning today. It combines two ideas: momentum (a running average of the gradient) *and* per-parameter learning rates (a running average of the gradient's *squared* magnitude, used to scale each weight's step). Parameters with large, erratic gradients get smaller steps; parameters with small, steady gradients get larger ones. In effect, every weight gets its own adaptive learning rate.

```python
def adam_step(w, grad, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * grad          # 1st moment: mean of gradients
    v = b2 * v + (1 - b2) * grad**2       # 2nd moment: mean of squared grads
    m_hat = m / (1 - b1**t)               # bias correction (early steps)
    v_hat = v / (1 - b2**t)
    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w, m, v
```

Adam usually "just works" with its default settings, which is why it's the go-to when you don't want to fuss over tuning. The tradeoff: it has more moving parts and more memory (two running averages per weight), and well-tuned SGD with momentum sometimes generalizes slightly better on certain tasks. For a first network, reach for Adam.

## How loss and optimizer fit together

Here's the full loop in miniature, so you can see where each piece lands:

```python
for batch in data:
    preds = forward(batch.x, params)          # forward pass
    loss = binary_cross_entropy(preds, batch.y)   # measure wrongness
    grads = backprop(loss, params)            # gradients of loss wrt weights
    params = adam_step(params, grads, ...)     # optimizer updates weights
```

Loss defines the target. Backprop measures how each weight affects it. The optimizer takes the step. Repeat, and the loss falls.

## Key takeaways

- The **loss function** turns a prediction and a truth into one number measuring wrongness; training minimizes it.
- Use **MSE for regression**, **cross-entropy for classification** — match the loss to the task and output activation.
- The **optimizer** converts gradients into weight updates; the **learning rate** is the most important knob.
- **SGD** steps on noisy mini-batch gradients; **momentum** smooths and accelerates by averaging past gradients; **Adam** adds per-parameter adaptive rates and is the common default.
- Loss, backprop, and optimizer form the core cycle: measure → attribute → step.

## Try it

Take the single neuron from the backpropagation module and wire in a full training step. Use MSE loss, compute the gradient by hand (you already did), and implement plain gradient descent: `w = w - lr * grad_w`, `b = b - lr * grad_b`. Run 100 steps with `lr = 0.5` on a fixed input/target pair and print the loss every 10 steps — watch it fall. Then rerun with `lr = 5.0` and `lr = 0.001` and describe what goes wrong in each case.
