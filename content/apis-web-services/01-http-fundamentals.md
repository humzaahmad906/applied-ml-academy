# 01 — How the Web Works: HTTP Fundamentals

Almost every ML system you will build talks to the outside world over HTTP: you call an API to fetch training data, you download a model from a hub, and eventually you serve your own model behind an endpoint that other people call. Before you can serve or consume APIs confidently, you need a clear picture of what HTTP actually is and what happens on the wire when two machines talk. This lesson builds that picture from the ground up, using nothing but `curl` so you can see the raw conversation.

## The client-server model

The web runs on a simple idea: one program asks, another answers. The asker is the **client** (your browser, a Python script, a mobile app) and the answerer is the **server** (a machine listening for requests). The client sends a **request**; the server sends back a **response**. That is the whole relationship. There is no ongoing connection you have to maintain and no shared memory between the two — each request stands alone, which is why HTTP is called *stateless*. If the server needs to remember who you are between requests, it does so with tokens or cookies you send each time, not with a live link.

This request-response pattern is the backbone of REST APIs, model-serving endpoints, and nearly everything you will do in the rest of this course.

## What happens when you curl a URL

When you run a command like the one below, a surprising amount happens in the second before you see output:

```bash
curl https://api.github.com/users/torvalds
```

First, the client takes the hostname `api.github.com` and asks DNS to translate it into an IP address, then opens a connection to that address on a port (443 for HTTPS by default). The mechanics of DNS, ports, and IP addresses are covered in the Cloud & Linux course, so we will not re-teach them here — just know that by the time HTTP starts, the client already knows *where* to send bytes. HTTP is the language spoken *over* that connection: a text-based protocol where the client writes a request and reads a response.

## HTTP methods

Every request names a **method** (also called a verb) that tells the server what kind of action you intend. The five you will use constantly:

- **GET** — read a resource. Should never change anything on the server.
- **POST** — create a new resource, or trigger an action (like "run a prediction").
- **PUT** — replace a resource entirely with the data you send.
- **PATCH** — update part of a resource, leaving the rest alone.
- **DELETE** — remove a resource.

An important property is **idempotency**: calling the same request many times has the same effect as calling it once. GET, PUT, and DELETE are idempotent — deleting an already-deleted item still leaves it deleted. POST is *not*: two POSTs to "create a prediction" create two predictions. GET is additionally **safe**, meaning it has no side effects at all. These distinctions matter later when you add retry logic, because retrying a POST can accidentally duplicate work.

## Status codes

Every response starts with a three-digit status code that tells you, at a glance, how things went. They come in families:

- **2xx — success.** `200 OK` (the request worked), `201 Created` (a POST created something new).
- **3xx — redirection.** The resource lives elsewhere; the client is told where to look.
- **4xx — client error.** *You* got something wrong. `400 Bad Request` (malformed input), `401 Unauthorized` (you are not authenticated), `403 Forbidden` (authenticated but not allowed), `404 Not Found`, `422 Unprocessable Entity` (the input parsed but failed validation — you will see this constantly with FastAPI), `429 Too Many Requests` (rate-limited).
- **5xx — server error.** The *server* broke. `500 Internal Server Error` (an unhandled crash), `503 Service Unavailable` (temporarily down or overloaded).

The mental rule: 4xx means fix your request, 5xx means the problem is on their end (and is often worth retrying). When your model-serving endpoint returns a 500, that is your code throwing an exception; when it returns 422, the caller sent bad data.

## Headers that matter

Alongside the method and body, each request and response carries **headers** — key-value metadata. A handful come up again and again in ML work:

- **Content-Type** — what format the body is in, e.g. `application/json`. The server needs this to know how to parse what you sent.
- **Authorization** — your credentials, usually `Bearer <token>` for API keys.
- **Accept** — what format you *want* back, e.g. `application/json`.
- **User-Agent** — identifies the client software; some APIs reject requests without one.

## Request and response bodies

The **body** carries the actual data. It can be plain text, HTML, form-encoded key-value pairs (`key=value&other=thing`, the old HTML-form format), or — dominantly in modern APIs — **JSON**. JSON won because it maps cleanly onto the data structures every language already has (objects, arrays, strings, numbers) and is readable by humans while still being compact. When you POST features to a model endpoint or read predictions back, the body is almost always JSON. Lesson 02 covers JSON in depth.

## HTTPS and TLS, briefly

The `s` in `https` means the connection is encrypted with **TLS**. Conceptually: before any HTTP is exchanged, the client and server perform a handshake that verifies the server's identity (via a certificate) and agrees on encryption keys, so no one in between can read or tamper with the traffic. You do not implement any of this — libraries and the OS handle it — but you should default to HTTPS for anything carrying credentials or real data, which is essentially everything.

## Seeing it for real

The `-v` (verbose) flag makes `curl` print the entire conversation, so you can watch the request go out and the response come back:

```bash
curl -v https://api.github.com/users/torvalds
```

Lines beginning with `>` are what the client *sent*; lines with `<` are what the server *returned*:

```
> GET /users/torvalds HTTP/2
> Host: api.github.com
> User-Agent: curl/8.4.0
> Accept: */*
>
< HTTP/2 200
< content-type: application/json; charset=utf-8
<
{ "login": "torvalds", "id": 1024025, ... }
```

You can read the whole exchange here: a GET request with three headers, a `200` response with a `content-type` header, and a JSON body. To send a header yourself, use `-H`:

```bash
curl -H "Accept: application/json" https://api.github.com/users/torvalds
```

Everything you build in this course is a variation on this exchange — a method, a URL, some headers, maybe a body, and a status code coming back.

## Key takeaways

- HTTP is a stateless request-response protocol: the client asks, the server answers, and each request stands on its own.
- The core methods are GET (read), POST (create/act), PUT (replace), PATCH (partial update), and DELETE (remove).
- GET, PUT, and DELETE are idempotent; POST is not — this matters when you add retries.
- Status codes group into families: 2xx success, 3xx redirect, 4xx your fault, 5xx server's fault.
- Headers like Content-Type, Authorization, Accept, and User-Agent carry the metadata that makes a request work.
- JSON dominates request and response bodies because it maps cleanly onto ordinary data structures.
- `curl -v` shows you the raw request and response, which is the best way to build intuition.

## Try it

Pick a public API that needs no authentication — GitHub's user endpoint (`https://api.github.com/users/<name>`) works well. Run `curl -v` against it and read the output line by line: identify the method, the request headers (`>`), the status code, and the response headers (`<`). Then try `curl -H "Accept: application/xml" ...` and see whether the status code or Content-Type changes. Finally, hit a URL that does not exist (add `/nope` to the path) and confirm you get a 404. Write down, in your own words, one thing each header in the request seemed to do.
