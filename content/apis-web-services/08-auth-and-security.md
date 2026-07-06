# 08 — Authentication, Rate Limiting, and Security Basics

The moment your model API is reachable by anyone, you have to answer two questions on every request: *who is this?* (authentication) and *are they allowed to do this?* (authorization). On top of that come the practical concerns of any public service — limiting abuse, letting browsers call you safely, and keeping secrets out of your code. This lesson gives you the working knowledge to secure an ML API without pretending you can build a full auth system from scratch — because you should not, and this lesson will show you what to reach for instead.

## API keys via a header

The simplest form of authentication is an **API key**: a secret string the caller sends with each request, which you check against a set of known keys. By convention it travels in a custom header, `X-API-Key`. FastAPI's `APIKeyHeader` reads that header, and you wrap the check in a dependency (Lesson 06) so any route can require it:

```python
import os
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key")
VALID_KEYS = set(os.environ["API_KEYS"].split(","))

def require_api_key(key: str = Security(api_key_header)) -> str:
    if key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

@app.post("/predict")
def predict(api_key: str = Depends(require_api_key)) -> dict:
    return {"prediction": 1}
```

A request without a valid `X-API-Key` header is rejected with `401` before the route runs. Because the check is a dependency, adding `Depends(require_api_key)` to any route protects it — one function, reused everywhere. API keys are ideal for service-to-service calls where you control both ends.

## Bearer tokens and JWT

For anything user-facing, keys give way to **tokens** carried in the standard `Authorization` header: `Authorization: Bearer <token>`. The word "Bearer" means "whoever bears this token is granted access," so the token must be protected in transit (always HTTPS) and short-lived. The common token format is a **JWT** (JSON Web Token): a signed, self-contained blob encoding who the user is and when it expires. Your API *validates* the signature and expiry rather than looking the token up in a database.

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException

bearer = HTTPBearer()

def verify_token(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    token = creds.credentials
    try:
        payload = decode_and_verify(token)   # validates signature + expiry
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload
```

The critical rule: **do not roll your own auth.** Signing, verifying, and expiring tokens correctly is full of subtle security traps. Use a vetted library — `python-jose` or `PyJWT` — to encode and decode JWTs, and lean on `HTTPBearer` to extract the token. Writing your own crypto is how APIs get breached.

## OAuth2 — recognize the flow

You will constantly see **OAuth2**, the protocol behind "Log in with Google" and most third-party API access. The idea: instead of handing your password to every app, an *authorization server* issues a scoped, expiring token that the app presents to the API. FastAPI has helpers (`OAuth2PasswordBearer` and friends) for implementing it, but a full OAuth2 provider is a substantial system. For this course the goal is to **recognize the flow** — client gets a token from an auth server, sends it as a Bearer token, your API validates it — not to implement a provider yourself. In practice you delegate this to an identity provider (Auth0, Cognito, Keycloak) and your API just validates the tokens they issue.

## Rate limiting

A public endpoint needs protection from being hammered — accidentally by a buggy client or deliberately by an abuser. **Rate limiting** caps how many requests a caller may make in a window. When a caller exceeds it, the server responds with HTTP `429 Too Many Requests`, usually with a `Retry-After` header saying how many seconds to wait.

The two sides of rate limiting matter for ML engineers in different ways. When you *serve* an API, you enforce limits (often at the API gateway or with a library, keyed by API key or IP). When you *consume* someone else's API (Lesson 03) — a hosted LLM, say — you must **respect their limits**: on a `429`, read `Retry-After` and back off rather than retrying immediately, or you will get throttled harder.

```python
# When consuming an API, honor 429 + Retry-After instead of hammering.
if response.status_code == 429:
    wait = int(response.headers.get("Retry-After", "5"))
    time.sleep(wait)   # then retry
```

## CORS — why browsers enforce it

If a website's JavaScript tries to call your API from a *different* domain, the browser blocks the response unless your API explicitly permits that origin. This is **CORS** (Cross-Origin Resource Sharing), and it is a browser-enforced safety rule — it exists so a malicious page cannot silently make authenticated calls to other sites you are logged into. It only affects browser clients; `curl`, a Python script, or another server are unaffected. To allow your legitimate front-end to call your API, add the CORS middleware (introduced in Lesson 06) and list the origins you trust:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.example.com"],  # never "*" with credentials
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

Resist the temptation to set `allow_origins=["*"]` on a real service — that opens your API to every website. List the specific front-ends you control.

## Never put secrets in code or URLs

The last rule underpins all the others: **secrets — API keys, tokens, database passwords — must never live in your source code or in URLs.** A key committed to git is a key leaked forever, even after you delete it, because it lives in the history. A key in a URL ends up in server logs, browser history, and proxy caches. Instead, read secrets from **environment variables** at runtime (note the `os.environ["API_KEYS"]` in the first example). The clean way to manage them is `pydantic-settings`, which loads config from env vars and a local `.env` file that you never commit:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_keys: str
    jwt_secret: str
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env")

settings = Settings()
```

This bridges directly into the **Software Engineering Practices course**, which covers configuration and secret management in depth. The habit to build now: secrets come from the environment, never from a string literal.

## Key takeaways

- API keys sent in an `X-API-Key` header, checked via an `APIKeyHeader` dependency, are the simplest auth — ideal for service-to-service calls.
- `Authorization: Bearer <token>` carries a token (often a JWT); your API validates the signature and expiry with a library like `python-jose` or `PyJWT`.
- Never roll your own auth or crypto — use vetted libraries, and delegate full login flows to an identity provider.
- OAuth2 is the token-issuing flow behind "Log in with X"; recognize it rather than implementing a provider yourself.
- Rate limiting returns `429`; when consuming an API, honor `Retry-After` and back off instead of retrying immediately.
- CORS is a browser-enforced rule; add `CORSMiddleware` with an explicit origin list, never `"*"` on a real service.
- Keep secrets out of source code and URLs — read them from environment variables, managed cleanly with `pydantic-settings`.

## Try it

Take the predict API from Lesson 07 and secure it. Add an `APIKeyHeader` dependency that reads valid keys from an `API_KEYS` environment variable and rejects missing or wrong keys with `401`; attach it to the `/predict` route. Test it three ways: no header (expect `401`), a wrong key (`401`), and a correct key (`200`). Then add `CORSMiddleware` allowing a single made-up front-end origin. Finally, move the key list into a `pydantic-settings` `Settings` class backed by a `.env` file, add that file to `.gitignore`, and confirm the app still starts by reading the keys from the environment rather than any literal in your code.
