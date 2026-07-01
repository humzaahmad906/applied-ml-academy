# 05 — Docker Compose

Real applications are rarely a single container. An ML service might have an inference API, a database, and a cache, all running together and talking to each other. Managing that by hand with long `docker run` commands gets painful fast. Docker Compose fixes this by letting you describe the whole stack in one file and control it with one command.

## The problem Compose solves

Imagine your app needs three containers: your API, a PostgreSQL database, and a Redis cache. Without Compose, you would run three separate `docker run` commands, each with its own ports, volumes, environment variables, and network settings. You would need to start them in the right order, remember every flag, and repeat the whole ritual every time. Get one flag wrong and something silently fails to connect.

Compose replaces all of that with a single configuration file, `docker-compose.yml`, and a handful of short commands. You declare *what* you want, and Compose figures out how to make it happen.

## The compose file

A compose file is written in YAML, a text format built around indented key-value pairs. It lists the *services* (containers) that make up your application, and for each one, the same settings you would otherwise pass to `docker run`.

Here is a compose file for a web API plus a database:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      - db

  db:
    image: postgres:16
    volumes:
      - db-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb

volumes:
  db-data:
```

Let's read it top to bottom.

- `services:` begins the list of containers. Here there are two, named `api` and `db`.
- Under `api`, `build: .` tells Compose to build an image from the `Dockerfile` in the current directory. This is the equivalent of `docker build`.
- `ports:` publishes port mappings, exactly like `-p 8000:8000`.
- `environment:` sets environment variables, exactly like `-e`.
- `depends_on:` tells Compose to start `db` before `api`, so the database exists when the API comes up.
- Under `db`, `image: postgres:16` pulls a ready-made image instead of building one. A service either builds an image or uses an existing one.
- `volumes:` under `db` persists the database's data. It uses a *named volume* (`db-data`) rather than a host path.
- The top-level `volumes:` block at the bottom declares that named volume so Compose manages it for you.

## Services talk by name

Here is the feature that makes Compose feel magical. Notice the `DATABASE_URL` in the `api` service points at `db:5432`, using the word `db`, the service name, as if it were a hostname.

Compose automatically puts all services on a shared private network and lets them find each other by their service name. The `api` container can reach the `db` container simply by connecting to the host `db`. You never look up IP addresses or wire up networking by hand. Name a service `db`, and `db` becomes its address to every other service in the file.

This is why multi-container apps become easy. The plumbing between containers is handled for you.

## The core commands

You run everything from the folder containing `docker-compose.yml`.

**Start the whole stack.** This builds images as needed, creates the network, and starts every service:

```bash
# Build (if needed) and start all services in the background
docker compose up -d
```

The `-d` runs it detached, in the background, just like `docker run -d`. Leave it off and you will see the combined logs of every service streaming in your terminal.

**Check what is running:**

```bash
# Show the status of services in this compose project
docker compose ps
```

**Watch the logs**, optionally for a single service:

```bash
# Follow logs for all services
docker compose logs -f

# Follow logs for just the api service
docker compose logs -f api
```

**Rebuild after changing your code or Dockerfile:**

```bash
# Rebuild images, then start
docker compose up -d --build
```

**Stop and remove everything** the stack created, containers and network:

```bash
# Stop and tear down the stack
docker compose down
```

By default `docker compose down` leaves named volumes intact, so your database data survives. If you truly want to wipe the data too, add `-v` to remove the named volumes as well.

## Why this matters for ML

An ML application often is a stack. A typical inference setup might combine:

- an **API service** that receives requests and returns predictions,
- a **database** that logs inputs and outputs for monitoring,
- a **cache** that stores recent results so repeated requests are fast.

With Compose, that entire system is one file that anyone on your team can start with `docker compose up`. The setup that used to be a page of instructions becomes a single command, and everyone runs the identical stack. It is the reproducibility promise of containers, extended from one container to a whole application.

## Key takeaways

- Compose describes a multi-container application in one YAML file, `docker-compose.yml`, and controls it with short commands.
- Each entry under `services:` is a container, configured with the same options you would pass to `docker run` (ports, environment, volumes, and either `build` or `image`).
- Services find each other by service name on an automatic shared network. No manual networking required.
- Core commands: `docker compose up -d` to start, `docker compose ps` to inspect, `docker compose logs` to watch, and `docker compose down` to tear down.
- Named volumes persist data across `up`/`down` cycles unless you explicitly remove them.

## Try it

1. Take an app image you built earlier (or the `build: .` pattern with a `Dockerfile`) and write a `docker-compose.yml` with a single `api` service that maps a port.
2. Run `docker compose up -d`, then `docker compose ps` to confirm it is running. Visit the mapped port to check.
3. Add a second service, a database using `image: postgres:16` with the environment variables shown above and a named volume. Add `depends_on: [db]` to your `api`.
4. Run `docker compose up -d` again and confirm both services start with `docker compose ps`.
5. Tear it down with `docker compose down`. Then bring it back up and confirm, via `docker compose logs db`, that the database data persisted thanks to the named volume.
