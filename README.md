# Applied ML Academy

The academy website: a landing page, a public certificate **verify** page, and a
password-protected **admin** where you issue certificates. Certificates only mint
at or above the pass mark (80% by default), each gets a unique verification code,
and anyone can check a code with no login.

Built with Flask. Runs on Render's free tier.

## Pages

- `/` landing page (courses, positioning)
- `/verify` public verification; `/verify/<code>` and `/c/<code>` are shareable links
- `/certificate/<code>` the printable certificate (Print to PDF, or Download PNG)
- `/admin` issue and manage certificates (login required)

## Run locally

```bash
pip install -r requirements.txt
export ADMIN_PASSWORD=pick-a-password        # Windows: set ADMIN_PASSWORD=...
python app.py                                # http://localhost:5000
```

Admin is at `/admin`. The default password is `change-me` until you set
`ADMIN_PASSWORD`.

## Deploy on Render

1. Push this folder to a **GitHub** repo.
2. In Render: **New +** -> **Web Service** -> connect the repo.
3. Configure:
   - **Runtime**: Python (auto-detected)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Instance Type**: Free
4. **Environment** (Environment tab -> Add):
   - `ADMIN_PASSWORD` -> your admin password (required)
   - `SECRET_KEY` -> any long random string (Render can generate one)
   - `PASS_MARK` -> `80` (optional; change to move the bar)
5. Create the service. First build takes a couple of minutes.

The repo also includes `render.yaml`, so you can instead use **New + -> Blueprint**
and Render reads the config automatically. You'll still set `ADMIN_PASSWORD`.

Free tier sleeps after ~15 min idle and takes ~30s to wake on the next visit.

## Keeping issued certificates (important)

By default the app uses SQLite, stored on the instance disk. **On Render's free
tier that disk is wiped on every redeploy and restart**, so issued certificates
would disappear. For durable records, attach a database and point the app at it:

1. In Render: **New + -> PostgreSQL** (free), or use a free
   [Supabase](https://supabase.com) Postgres.
2. Copy its connection URL.
3. On the web service, add env var `DATABASE_URL` = that URL.

The app auto-detects `DATABASE_URL`, fixes the `postgres://` scheme, and uses it.
No code changes needed.

## Customize

Brand details (academy name, signatures, established year) are constants at the
top of `app.py`. Courses shown on the landing page are the `COURSES` list in the
same file. The certificate design lives in `templates/certificate.html` and the
seal/border art in `art.py`.
