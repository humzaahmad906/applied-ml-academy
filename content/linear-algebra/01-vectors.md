# 01 — Vectors and Geometry

Machine learning runs on numbers, but rarely one number at a time. Almost everything — an image, a sentence, a user's preferences, a sound clip — gets turned into a *list* of numbers. That list is a **vector**, and learning to think about vectors geometrically is the single most useful habit you can build early on.

## What a vector actually is

A vector is just an ordered list of numbers. Here is one with two entries:

```
v = [3, 2]
```

That's it. The order matters — `[3, 2]` is not the same as `[2, 3]` — and each number is called a **component** or **entry**. A vector with two components is a 2-vector, one with three is a 3-vector, and machine learning routinely uses vectors with hundreds or thousands of components.

There are two ways to picture a vector, and both are worth holding in your head at once.

**As a point.** The vector `[3, 2]` is the location you reach by going 3 steps right and 2 steps up from the origin. It's a spot on the plane.

**As an arrow.** The same `[3, 2]` is an arrow pointing from the origin to that spot. The arrow has a *direction* (which way it points) and a *length* (how far it reaches). This picture is the more powerful one, because it tells you what happens when you combine vectors.

For a 2-vector we can draw it on paper. For a 3-vector, imagine an arrow in the room you're sitting in. Beyond three components we lose the ability to draw, but — and this is the important part — the *rules* stay exactly the same. A 768-component vector behaves like a really long arrow in a space we can't visualize, and every intuition you build in 2D still applies.

## Adding vectors

To add two vectors, add them component by component:

```
[3, 2] + [1, 4] = [3+1, 2+4] = [4, 6]
```

The shapes must match — you can only add a 2-vector to a 2-vector.

Geometrically, addition is "walk one arrow, then walk the other." Start at the origin, follow the first arrow to `[3, 2]`, then from there follow the second arrow (1 right, 4 up), and you land at `[4, 6]`. The sum is the arrow from the origin straight to where you ended up. This is sometimes called the "tip-to-tail" rule, and it works no matter how many components you have.

Subtraction works the same way: `[4, 6] - [1, 4] = [3, 2]`. Geometrically, `a - b` is the arrow pointing *from* `b` *to* `a`. That fact quietly powers a lot of ML — the difference between two vectors tells you the direction and distance from one to the other.

## Scaling vectors

Multiplying a vector by a single number (a **scalar**) stretches or shrinks it:

```
2 * [3, 2] = [6, 4]
0.5 * [3, 2] = [1.5, 1]
```

You multiply every component by the scalar. Geometrically, the arrow keeps its direction but changes length. Multiply by 2 and it's twice as long; multiply by 0.5 and it's half as long.

A negative scalar flips the arrow to point the opposite way:

```
-1 * [3, 2] = [-3, -2]
```

Scaling and adding together let you build *any* vector out of a few building blocks. For example, every 2-vector is some amount of `[1, 0]` plus some amount of `[0, 1]`: the vector `[3, 2]` is just `3 * [1, 0] + 2 * [0, 1]`. This idea — combining scaled vectors — is called a **linear combination**, and it's the beating heart of everything that follows.

## Why this matters for ML

When a model turns a word into a vector (an "embedding"), similar words end up as arrows pointing in similar directions. When a recommendation system represents you and a movie as vectors, it's asking whether your arrows point the same way. Adding and scaling vectors is how models blend information — averaging a batch of examples, nudging a prediction, mixing features. Every one of those operations is the component-wise addition and scaling you just learned.

## A quick worked example

Suppose a tiny model represents "warmth" and "brightness" of a color as a 2-vector. Red is `[8, 3]` and yellow is `[6, 9]`. The *average* color is:

```
0.5 * ([8, 3] + [6, 9]) = 0.5 * [14, 12] = [7, 6]
```

That's you blending two vectors with addition and scaling — the same math a real model does with thousands of components.

## Key takeaways

- A vector is an ordered list of numbers; picture it as a point *or* an arrow from the origin.
- Add vectors component by component; geometrically this is "tip-to-tail."
- Scaling by a number stretches, shrinks, or flips the arrow without changing (or reversing) its direction.
- A **linear combination** — scaled vectors added together — can build any vector, and it underlies how models combine information.
- The rules are identical in 2 dimensions or 2000; the geometry you learn in the plane carries all the way up.

## Try it

By hand, with `a = [2, 1]` and `b = [-1, 3]`:

1. Compute `a + b` and `a - b`.
2. Compute `3 * a` and `-2 * b`.
3. Compute the linear combination `2 * a + 1 * b`.
4. Sketch `a`, `b`, and `a + b` as arrows on graph paper and confirm the tip-to-tail rule holds visually.
5. Bonus: what single scalar times `a` gives an arrow twice as long pointing the *opposite* direction?
