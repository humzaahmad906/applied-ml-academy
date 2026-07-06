# 09 — Configuration and Secrets Management

Every real program needs settings: which database to talk to, how many worker threads to run, which API key to authenticate with. The tempting shortcut is to write these values directly into your code. That shortcut causes some of the most painful failures in software — leaked passwords, code that only runs on one machine, and secrets baked forever into git history. This lesson teaches the disciplined alternative: keep configuration in the environment, keep secrets out of your repository, and validate both.

## The twelve-factor principle: config lives in the environment

There is a widely followed set of guidelines for building good services called the Twelve-Factor App, and one of its rules is simple and powerful: **configuration belongs in the environment, not in the code.** Anything that changes between your laptop, a teammate's laptop, a staging server, and production — that is configuration, and it should come from outside the program.

Why? Because the *same* code should run everywhere, with only the environment differing. If your database URL is hard-coded, you must edit source to deploy. If it comes from an environment variable, the exact same artifact runs in every setting and reads its address from the machine it lands on. It also means secrets never sit in the code that gets shared, reviewed, and stored in version control.

## Reading environment variables

Python exposes environment variables through `os.environ`. Read them with `.get()`, and always decide what happens when the variable is missing:

```python
import os

# with a sensible default
workers = int(os.environ.get("WEB_CONCURRENCY", "4"))

# required — fail loudly if absent
db_url = os.environ.get("DATABASE_URL")
if db_url is None:
    raise RuntimeError("DATABASE_URL is not set")
```

The important habit: *either* supply a sensible default *or* fail loudly. Never let a missing configuration value silently become `None` and drift downstream — that produces confusing failures far from the real cause (exactly the fail-loud principle from Lesson 08).

## .env files for local development

Setting environment variables by hand every time you open a terminal is tedious. For local development, the convention is a `.env` file in your project holding `KEY=value` lines:

```bash
# .env
DATABASE_URL=postgresql://localhost:5432/dev
WEB_CONCURRENCY=2
OPENAI_API_KEY=sk-local-dev-key
```

The `python-dotenv` library loads this file into the environment when your program starts:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env into os.environ
api_key = os.environ["OPENAI_API_KEY"]
```

Two rules make this safe and useful:

- **`.env` goes in `.gitignore`.** It holds real values, often secrets. It must never be committed.
- **Commit a `.env.example` instead.** This file lists every variable your program needs, with blank or dummy values, so a new teammate knows exactly what to set — without ever exposing a real secret.

```bash
# .env.example  (committed)
DATABASE_URL=
WEB_CONCURRENCY=4
OPENAI_API_KEY=
```

The `.example` file is documentation: it answers "what do I need to configure to run this?" without leaking anything.

## pydantic-settings: typed, validated configuration

Reading each variable with `os.environ.get`, converting strings to ints, and checking for missing values by hand gets repetitive and error-prone. `pydantic-settings` (version 2.x, installed separately with `uv add pydantic-settings`) turns your configuration into a validated, typed object.

You declare a class describing your settings, and pydantic reads them from the environment, coerces the types, and raises a clear error if anything required is missing or malformed:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env")

    database_url: str
    web_concurrency: int = 4
    debug: bool = False


settings = Settings()
print(settings.web_concurrency)  # a real int, not a string
```

Several things happen here for free. `env_prefix="APP_"` means the field `database_url` is read from the environment variable `APP_DATABASE_URL`, keeping your app's variables namespaced and unambiguous. `env_file=".env"` loads the local file automatically, so you do not even need `python-dotenv`. Types are enforced: `web_concurrency` arrives as a proper `int`, and `debug` accepts `true`/`false`/`1`/`0` and gives you a real `bool`. A missing required field (one with no default, like `database_url`) raises a validation error at startup — loud and early, exactly where you want it.

You can also nest configuration for larger apps, grouping related settings into sub-models, but a single flat `Settings` class covers most needs and is the right place to start.

## Secrets: the values that must never leak

Some configuration is merely environmental — a port number, a worker count. Other configuration is *secret*: API keys, database passwords, authentication tokens, signing keys. These demand stricter handling. The absolute rule is: **secrets never go into source code and never go into git history.**

There are three acceptable homes for a secret:

- **Environment variables**, set on the machine or container that runs the program.
- **A secret manager** — HashiCorp Vault, or a cloud secret store like AWS Secrets Manager, GCP Secret Manager, or Azure Key Vault. The program fetches the secret at runtime and it is never written to disk in the repo.
- **A `.env` file that lives outside version control** (in `.gitignore`), for local development only.

## What happens when a secret leaks

Understanding the cost makes the discipline stick. If you commit an API key and push it to GitHub, assume it is compromised the moment it lands — automated scanners crawl public repositories for exactly these patterns, and keys have been abused within minutes. Deleting the line in a later commit does **not** fix it: git keeps history, so the secret is still sitting in an earlier commit for anyone who clones the repo.

The correct response is twofold. First, **rotate the secret immediately** — revoke the leaked key and issue a new one, because the old one must be considered burned. Second, **purge it from history**. The modern tool for rewriting git history to remove a secret from every past commit is `git filter-repo`:

```bash
git filter-repo --path .env --invert-paths
```

This removes the file from the entire history. It rewrites commit hashes, so it is disruptive on a shared repository and everyone must re-clone — which is exactly why prevention (`.gitignore` from day one) is so much cheaper than the cure. Rotating the key is non-negotiable regardless, because you can never be sure who already saw it.

## Key takeaways

- Configuration belongs in the environment, not in code, so the same artifact runs unchanged everywhere (the twelve-factor principle).
- Read variables with `os.environ.get(...)` and always either give a default or fail loudly — never let a missing value silently become `None`.
- Use a `.env` file for local dev, put it in `.gitignore`, and commit a `.env.example` that documents required variables without values.
- `pydantic-settings` (2.x) validates and type-coerces your config into a typed object via a `BaseSettings` subclass and `SettingsConfigDict(env_prefix=..., env_file=...)`.
- Secrets — keys, passwords, tokens — never go in source or git history; use environment variables, a secret manager, or a `.env` kept out of the repo.
- A leaked secret must be rotated immediately and purged from history with `git filter-repo`; deleting the line in a new commit is not enough.

## Try it

Create a small project with `uv init`. Add a `.env` file with `APP_DATABASE_URL=postgresql://localhost/dev` and `APP_DEBUG=true`, add `.env` to `.gitignore`, and create a matching `.env.example` with the keys but blank values. Then `uv add pydantic-settings` and write a `Settings(BaseSettings)` class with `SettingsConfigDict(env_prefix="APP_", env_file=".env")`, a required `database_url: str`, and a `debug: bool = False`. Instantiate it and print the values, confirming `debug` is a real boolean. Finally, comment out `database_url` in your `.env` and run again — watch pydantic raise a clear validation error at startup instead of failing mysteriously later.
