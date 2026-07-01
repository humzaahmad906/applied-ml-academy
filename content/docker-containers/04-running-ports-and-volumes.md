# 04 — Running: Ports and Volumes

Building an image is half the job. The other half is running it *usefully*: reaching a web server inside a container, keeping data around after the container dies, and passing in configuration. This lesson covers the flags on `docker run` that make containers actually do work.

## docker run, revisited

You have used `docker run` in its simplest form. In practice it takes flags that connect the container to the outside world. The general shape is:

```bash
docker run [flags] <image-name>
```

The flags are where all the interesting behavior lives. We will cover the three you will reach for constantly: ports, volumes, and environment variables. Two more small but important flags come first.

### Foreground, background, and cleanup

By default a container runs in the foreground and ties up your terminal. To run it in the background (detached), use `-d`:

```bash
# Run the container in the background
docker run -d myapp
```

To give the container a memorable name instead of a random one, use `--name`:

```bash
# Run a named container in the background
docker run -d --name web myapp
```

To automatically delete the container when it stops (great for one-off runs so you do not accumulate stopped containers), use `--rm`:

```bash
# Run and auto-remove when finished
docker run --rm myapp
```

## Ports: reaching a server inside the container

Here is a subtlety that trips up every beginner. Suppose your app runs a web server on port 8000 *inside* the container. If you just run it, then open `localhost:8000` in your browser, nothing happens. Why? Because the container has its own isolated network. Port 8000 inside the container is not the same as port 8000 on your machine.

To bridge them, you *publish* a port with `-p`, mapping a host port to a container port:

```bash
# Map host port 8000 to container port 8000
docker run -p 8000:8000 myapp
```

The format is `-p HOST:CONTAINER`. The number on the left is the port on your machine; the number on the right is the port the app listens on inside the container. They do not have to match:

```bash
# Reach the container's port 8000 via port 3000 on your machine
docker run -p 3000:8000 myapp
```

With that running, `localhost:3000` on your machine reaches the server inside the container. Getting the direction right, host on the left, container on the right, saves a lot of confusion.

## Volumes: keeping data alive

Recall that containers are disposable and their writable layer vanishes when they are removed. That is fine for the app itself, but disastrous for data you care about: a database, model checkpoints, logs, training outputs. You do not want to lose those when a container is replaced.

A *volume* solves this by connecting a folder on your machine to a folder inside the container. Data written there lives on your machine and survives the container's death. You mount one with `-v`:

```bash
# Mount ./data on the host to /app/data inside the container
docker run -v $(pwd)/data:/app/data myapp
```

The format is `-v HOST_PATH:CONTAINER_PATH`. `$(pwd)` expands to your current directory, so this mounts a local `data` folder into `/app/data` inside the container. Anything the app writes to `/app/data` actually lands in your local `data` folder, and it is still there after the container is gone.

Volumes also work the other direction: they are how you feed data *in*. Mount a folder of input files and the container can read them without baking them into the image. This is especially handy in ML, where datasets are far too large to copy into an image.

## Environment variables: passing configuration

Hard-coding settings into an image is a bad idea; you would need to rebuild for every change. Instead, pass configuration at run time with environment variables using `-e`:

```bash
# Set an environment variable inside the container
docker run -e MODEL_NAME=resnet50 -e LOG_LEVEL=info myapp
```

Inside the container, the app reads these like any environment variable (for example, `os.environ["MODEL_NAME"]` in Python). This lets one image behave differently across environments: point it at a different model, flip debug logging, or supply a port, all without rebuilding.

When you have many variables, keep them in a file and load it with `--env-file`:

```bash
# Load all variables from a file named .env
docker run --env-file .env myapp
```

A word of caution: never put secrets like API keys directly in your `Dockerfile` or image, because anyone with the image can read them. Pass secrets at run time through environment variables or a file instead.

## Putting it all together

A realistic run for a web service that reads and writes data usually combines all of these:

```bash
# Named, backgrounded, port-mapped, with a volume and config
docker run -d --name inference \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_NAME=resnet50 \
  myapp
```

This starts `myapp` in the background as `inference`, exposes its port 8000 on your machine's port 8000, mounts your local `models` folder so the container can load model files, and sets a configuration variable. When you no longer need it, `docker stop inference` and `docker rm inference` clean it up.

## Inspecting a running container

Two commands help when something is not working. View the container's output (its logs):

```bash
# Stream the logs of a running container
docker logs -f inference
```

Open an interactive shell *inside* a running container to poke around its filesystem:

```bash
# Get a shell inside the running container
docker exec -it inference bash
```

`-it` makes the session interactive so you can type commands, look at files, and confirm your volume mounted where you expected.

## Key takeaways

- `docker run` flags connect a container to the outside world; `-d` backgrounds it, `--name` names it, `--rm` auto-cleans it.
- Ports are isolated. Publish them with `-p HOST:CONTAINER` to reach a server inside the container from your machine.
- Volumes (`-v HOST_PATH:CONTAINER_PATH`) persist data outside the container and feed large data in without baking it into the image.
- Environment variables (`-e` or `--env-file`) pass configuration at run time so one image works everywhere. Keep secrets out of images.
- Use `docker logs` and `docker exec -it ... bash` to inspect and debug a running container.

## Try it

1. Take the `hello-app` image from the previous lesson, or any image that runs a simple web server on a known port.
2. Run it with a port mapping, `docker run -d --name test -p 8080:8000 <image>`, then open `localhost:8080`. Confirm you can reach it.
3. Stop and remove it, then run it again with a different host port like `-p 9000:8000`. Confirm the app is now reachable at the new port, proving the mapping is what controls access.
4. Create a local `data` folder. Run a container with `-v $(pwd)/data:/app/data`, use `docker exec -it <name> bash` to write a file into `/app/data`, then exit and remove the container. Check your local `data` folder; the file is still there.
