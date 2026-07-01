# 05 — Gradient Descent

Everything so far has been building to this. **Gradient descent** is the algorithm that lets a model learn, and it's astonishingly simple: figure out which way is downhill, take a small step that way, and repeat. If you can walk down a hill in fog, you understand gradient descent.

## The hill-in-fog picture

Imagine you're on a hillside in thick fog and you want to reach the lowest point. You can't see the valley, but you can feel the ground under your feet. So you do the obvious thing: feel which direction slopes down most steeply, take a step that way, and check again. Step by step, you drift toward the bottom.

That's the whole algorithm. The "slope under your feet" is the derivative (or gradient). The valley bottom is where the error is smallest. Learning is just walking downhill on the error landscape.

## The one-variable version

Let's make it concrete with an error function shaped like a valley:

```
f(x) = (x - 4)^2
```

The bottom sits at `x = 4`, but pretend we don't know that and have to find it. Its derivative is:

```
f'(x) = 2(x - 4)
```

The update rule for gradient descent is:

```
new_x = x - learning_rate * f'(x)
```

Read it carefully. We *subtract* the slope, because the derivative points uphill and we want to go the other way. The `learning_rate` is a small positive number that controls how big each step is.

Let's start at `x = 0` with a learning rate of `0.1` and walk:

| step | x     | f'(x) = 2(x-4) | new_x |
|------|-------|----------------|-------|
| 0    | 0.00  | -8.00          | 0.80  |
| 1    | 0.80  | -6.40          | 1.44  |
| 2    | 1.44  | -5.12          | 1.95  |
| 3    | 1.95  | -4.10          | 2.36  |

Each step, `x` creeps toward `4`. Keep going and it homes right in on the bottom. Notice the slope shrinks as we approach — the steps naturally get smaller near the target.

```python
def grad(x):
    return 2 * (x - 4)

x = 0.0
lr = 0.1
for step in range(30):
    x = x - lr * grad(x)
print(round(x, 4))   # ≈ 4.0
```

Thirty steps and we've essentially found `x = 4` without ever being told the answer. The model *learned* it from the slope alone.

## The learning rate

The `learning_rate` (often called `lr`) is the most important dial to understand.

- **Too small:** tiny steps. You'll get there eventually, but it wastes time — thousands of steps for a journey a hundred would do.
- **Too large:** huge steps. You can overshoot the bottom and bounce to the *other* side of the valley, sometimes climbing higher each time and flying off to infinity. This is called *diverging*.
- **Just right:** steady, confident steps that settle into the bottom.

Try changing `lr` to `1.1` in the loop above and watch `x` explode instead of converge. Then try `0.001` and watch it barely move. Tuning the learning rate is one of the everyday crafts of training models.

## Minima: where we're headed

The bottom of the valley is called a **minimum** — the input where the function is smallest. At a minimum, the ground is flat, which means the slope is zero. That's the signal that gradient descent has arrived: the derivative is (near) zero, so the update `x - lr * 0` stops moving `x`.

One honest caveat: some landscapes have more than one valley. A spot that's the lowest *in its neighborhood* but not the lowest overall is a **local minimum**, versus the true bottom, the **global minimum**. Gradient descent, being a fog-walker, can settle into a local minimum because from where it's standing, every direction is uphill. In practice this is often fine, and there are tricks to escape, but it's worth knowing the algorithm follows the slope it can feel, not a map it can't see.

## Scaling up to many knobs

Real models have many inputs, so instead of a single derivative we use the **gradient** — the vector of all the partial derivatives. The update looks identical, just applied to every knob at once:

```
new_weights = weights - learning_rate * gradient
```

The gradient points uphill in the full high-dimensional landscape; we subtract it to head downhill. Every knob gets nudged in proportion to how much it was contributing to the error. Run this over and over on real data, and the model's knobs settle into values that make good predictions. That repeated loop is training.

## Key takeaways

- **Gradient descent** = feel the slope, step downhill, repeat.
- The update rule is `new = old - learning_rate * slope`. We subtract because the slope points *uphill*.
- The **learning rate** sets step size: too small is slow, too large overshoots and diverges, just right converges smoothly.
- A **minimum** is where the function is smallest and the slope is zero — that's where descent settles.
- Beware **local minima**: descent follows the slope it feels, which isn't always the global bottom.
- With many knobs, swap the single slope for the **gradient** and update every knob at once. That loop is how models train.

## Try it

Use the error function `f(x) = (x - 4)^2` from above.

1. Run the gradient-descent loop with `lr = 0.1` and print `x` every 5 steps. Watch it approach `4`.
2. Change `lr` to `1.1`. What happens to `x`? Explain why in terms of overshooting.
3. Change the starting point to `x = 10`. Does it still reach `4`? From which direction does it approach this time, and what sign does the slope have along the way?
4. Bonus: modify the loop to also print `f(x)` each step and confirm the error shrinks toward `0`.
