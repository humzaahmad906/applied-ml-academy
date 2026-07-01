# 02 — Images vs Containers

Two words come up constantly in the container world: *image* and *container*. Beginners mix them up all the time, and it causes real confusion later. This lesson pins down the difference, then walks through the lifecycle of a container from birth to cleanup.

## The two core concepts

Here is the one-sentence version, and it is worth memorizing:

> An **image** is a blueprint. A **container** is a running instance of that blueprint.

An image is a read-only package sitting on disk. It contains a filesystem snapshot: the operating system files, your installed libraries, your code, and instructions for what to run. It does nothing on its own. It just sits there, inert, like a `.zip` file or a class definition in code.

A container is what you get when you *start* an image. The container is alive: it has a running process, memory, its own writable layer on top of the image, and a lifecycle. It can be started, stopped, and deleted.

The class-and-object analogy from programming fits perfectly:

- The **image** is like a class. You define it once.
- The **container** is like an object. You can create many independent objects from one class.

From a single image you can launch one container or a hundred, and each is isolated from the others. Delete a container and the image is untouched, ready to spawn more.

## Why the distinction matters

Because images are read-only, they are stable and shareable. You build an image, and that exact byte-for-byte package can be copied to another machine and produce identical containers. This is what gives containers their reproducibility.

Containers, on the other hand, are disposable. The recommended mindset is to treat them as *cattle, not pets*: you do not lovingly maintain a single long-lived container, you throw it away and start a fresh one from the image whenever you need to. Any data you care about lives outside the container (you will learn how in a later lesson), so destroying a container loses nothing important.

This split, stable images and disposable containers, is the foundation everything else builds on.

## Images are built in layers

An image is not one solid block. It is a stack of read-only layers, each representing a change: one layer installs the base operating system, the next adds Python, the next installs your libraries, the next copies in your code. When you build an image, each instruction adds a layer on top.

Layers are cached and shared. If two images both start from the same base operating system layer, that layer is stored once on disk and reused. When you rebuild after changing only your code, the earlier layers (the OS, Python, the libraries) are reused from cache and only the changed layers are rebuilt. This is why the second build of an image is usually much faster than the first.

When a container runs, it adds one more layer on top of the image's stack: a thin, writable layer. Anything the container writes goes there. Delete the container and that writable layer disappears, leaving the image's read-only layers untouched.

## The container lifecycle

A container moves through a predictable set of states. Here are the commands that drive it.

**List available images** on your machine:

```bash
# Show all images stored locally
docker images
```

**Create and start a container** from an image. The following starts a container from the small `hello-world` image, which prints a message and exits:

```bash
# Run a container from the hello-world image
docker run hello-world
```

`docker run` does two things at once: it creates a container from the image and starts it.

**See what is running** and what has stopped:

```bash
# List currently running containers
docker ps

# List all containers, including stopped ones
docker ps -a
```

`docker ps` shows only live containers. Adding `-a` reveals stopped ones too, which is useful because a container that finished its job sticks around in a stopped state until you remove it.

**Stop a running container** by its name or ID (you can see IDs in `docker ps`):

```bash
# Gracefully stop a running container
docker stop <container-id-or-name>
```

**Remove a stopped container** to clean up:

```bash
# Delete a stopped container
docker rm <container-id-or-name>
```

**Remove an image** you no longer need:

```bash
# Delete an image from local storage
docker rmi <image-name>
```

Put together, the typical loop is: `docker run` to create and start, `docker ps` to observe, `docker stop` to halt, and `docker rm` to clean up. Images arrive on your machine either by building them (next lesson) or by pulling them from a registry (a later lesson), and `docker rmi` removes them when you are done.

## A note on naming and IDs

Every container gets a unique ID (a long hex string) and a human-friendly name (auto-generated, like `nervous_tesla`, unless you set one). You can refer to a container by either. You will learn to assign meaningful names with the `--name` flag soon, which makes commands far easier to type and scripts far easier to read.

## Key takeaways

- An image is a read-only blueprint stored on disk; a container is a running instance of that image. Think class versus object.
- One image can produce many independent containers.
- Images are built from cached, reusable layers, which makes rebuilds fast and storage efficient.
- Containers are disposable. Treat them as cattle, not pets, and keep important data outside them.
- The core lifecycle commands are `docker run`, `docker ps`, `docker stop`, `docker rm`, and `docker rmi`.

## Try it

With Docker installed and running:

1. Run `docker run hello-world`. Read the message it prints; it explains what just happened under the hood.
2. Run `docker ps`. Notice the container is not listed, because it already exited.
3. Now run `docker ps -a`. There it is, in a stopped state. Note its container ID and its auto-generated name.
4. Run `docker images` and find the `hello-world` image that was downloaded.
5. Clean up: remove the stopped container with `docker rm <name>`, then remove the image with `docker rmi hello-world`. Run `docker ps -a` and `docker images` again to confirm both are gone.
