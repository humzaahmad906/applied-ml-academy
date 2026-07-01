# 07 — Registries and Sharing

You have built an image and run it on your machine. But the whole promise of containers is that the artifact runs *anywhere*. To get your image onto another machine, a teammate's laptop, a server, a cloud platform, you need a place to store and share it. That place is a *registry*. This final lesson covers tagging, pushing, pulling, and what deployment actually looks like.

## What a registry is

A registry is a storage service for container images, much like a package repository stores libraries or a Git host stores code. You push an image up to the registry, and anyone with access can pull it down and run it. The image arrives byte-for-byte identical, so it produces the exact environment you built and tested.

There are public registries (the default one Docker uses when you pull common images like `python` or `postgres`) and private registries run by cloud providers or hosted internally for a team. The mechanics are the same across all of them; only the address changes.

## Image names and tags

Before you can push an image, you need to understand how images are named. A full image reference has three parts:

```
registry-host/repository:tag
```

For example, `registry.example.com/myteam/model-server:v1.0`. Breaking that down:

- **registry-host** is where the image lives (`registry.example.com`). If you omit it, the default public registry is assumed.
- **repository** is the name of the image, often namespaced by team or user (`myteam/model-server`).
- **tag** is a label for a specific version (`v1.0`). If you omit the tag, it defaults to `latest`.

Tags are how you version images. `model-server:v1.0`, `model-server:v1.1`, and `model-server:latest` can all coexist in one repository. This matters enormously for ML: you want to tag the image that produced a given set of results so you can reproduce or roll back to it later.

A word on `latest`: it is not magic. It is just the default tag name, and it does *not* automatically mean the newest image. Relying on `latest` in production is a common trap because you can never be sure which build it points to. Use explicit version tags for anything you deploy.

## Tagging an image

When you built images earlier with `-t`, you were already tagging them, just with a short local name. To push to a registry, you give the image a full reference. The `docker tag` command adds a new name to an existing image:

```bash
# Give your local image a full registry reference
docker tag model-server registry.example.com/myteam/model-server:v1.0
```

This does not copy or rebuild anything; it just adds another name pointing at the same image. Now the image can be pushed to that registry.

## Logging in

Registries require authentication before you can push. You log in once with `docker login`, giving the registry host:

```bash
# Authenticate to a registry
docker login registry.example.com
```

It prompts for a username and password or token. For the default public registry, you can omit the host. Once logged in, your credentials are cached and you do not need to repeat this for every push.

## Pushing

With the image tagged and yourself logged in, push it up:

```bash
# Upload the image to the registry
docker push registry.example.com/myteam/model-server:v1.0
```

Docker uploads the image layer by layer. Layers already present in the registry are skipped, so pushing a small change to a large image only transfers the changed layers, not the whole thing. This is the layer system from Lesson 2 paying off again.

## Pulling and running elsewhere

On any other machine, a server, a colleague's laptop, a cloud instance, pull the image and run it. This is the moment the container promise is fulfilled:

```bash
# Download the image from the registry
docker pull registry.example.com/myteam/model-server:v1.0

# Run it exactly as you did locally
docker run -d -p 8000:8000 registry.example.com/myteam/model-server:v1.0
```

The machine doing this needs Docker installed but nothing else, no Python, no libraries, no matching setup. The environment came inside the image. Your inference service now runs on that machine identically to how it ran on yours.

You often do not even need an explicit `docker pull`; `docker run` will pull the image automatically if it is not already present locally.

## What deployment looks like

"Deploying" a containerized model is, at its core, the pull-and-run you just saw, done on a server that stays up and is reachable by other systems. In practice, teams automate and harden this:

- A build pipeline builds the image, tags it with a version, and pushes it to a registry every time code is merged.
- A deployment step pulls that specific tagged image onto the target servers and starts it, often replacing the previous version with zero downtime.
- Orchestration platforms manage many copies of the container across many machines, restarting any that crash and scaling the count up or down with traffic.

The details of those platforms are beyond this course, but the foundation is exactly what you now know. Every one of them ultimately pulls a tagged image from a registry and runs a container from it. The image is the unit of deployment, and you can build, tag, push, and run it.

## A clean workflow

Putting the whole lifecycle together, here is the end-to-end path for shipping a model:

```bash
# 1. Build and tag with a version
docker build -t registry.example.com/myteam/model-server:v1.0 .

# 2. Log in to the registry (once)
docker login registry.example.com

# 3. Push the versioned image
docker push registry.example.com/myteam/model-server:v1.0

# 4. On the target machine: pull and run
docker pull registry.example.com/myteam/model-server:v1.0
docker run -d -p 8000:8000 registry.example.com/myteam/model-server:v1.0
```

That sequence takes a model from your laptop to running in production, reproducibly, with an explicit version you can always return to.

## Key takeaways

- A registry stores and shares images so they can run on any machine that has Docker.
- An image reference is `registry-host/repository:tag`; tags are how you version images, and `latest` is just a default name, not "the newest."
- `docker tag` adds a full registry name to an image; `docker login` authenticates; `docker push` uploads; `docker pull` downloads.
- Pushing and pulling transfer only changed layers, so updates are efficient.
- Deployment is fundamentally pull-and-run on a server. Everything on top, pipelines and orchestration, automates that same core step.

## Try it

1. Create a free account on a container registry of your choice and run `docker login` against it.
2. Take your `model-server` image from the previous lesson and tag it with your registry's full reference and a version, for example `docker tag model-server <your-registry>/<you>/model-server:v1.0`.
3. Push it with `docker push`. Watch the layers upload.
4. Remove the local image with `docker rmi` to prove it is really gone from your machine.
5. Pull it back with `docker pull`, then `docker run` it and hit the `/health` endpoint. You just round-tripped your model through a registry, exactly as a deployment would.
