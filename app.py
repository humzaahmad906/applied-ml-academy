import os
import re
import secrets
import hmac
from functools import wraps
from datetime import datetime, date
from pathlib import Path

import markdown as md
from flask import (Flask, render_template, request, redirect, url_for,
                   session, abort, flash)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

import art

# ---------------------------------------------------------------- config
BRAND = "Applied ML Academy"
MONOGRAM = "A"
EST_YEAR = "2026"
PROGRAM_LINE = "Professional Certification Program"
INSTRUCTOR = "Humza Ahmad"
INSTRUCTOR_TITLE = "Program Director & Lead Instructor"
COSIGNER = "A. Rahman"
COSIGNER_TITLE = "Dean of Curriculum"

PASS_MARK = int(os.environ.get("PASS_MARK", "80"))
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "change-me")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Render's free Postgres hands out postgres:// ; SQLAlchemy wants postgresql://
db_url = os.environ.get("DATABASE_URL", "sqlite:///academy.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ---------------------------------------------------------------- models
class Certificate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(48), unique=True, nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    course = db.Column(db.String(200), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    hours = db.Column(db.String(80), default="")
    issued_on = db.Column(db.String(10), nullable=False)   # YYYY-MM-DD
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    revoked = db.Column(db.Boolean, default=False)

    @property
    def band(self):
        return "Distinction" if self.score >= 90 else "Pass"

    @property
    def issued_display(self):
        try:
            return datetime.strptime(self.issued_on, "%Y-%m-%d").strftime("%d %B %Y")
        except ValueError:
            return self.issued_on


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(200), unique=True, nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    enrollments = db.relationship("Enrollment", backref="user",
                                  cascade="all, delete-orphan")


class Enrollment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    course_slug = db.Column(db.String(80), nullable=False)
    last_module = db.Column(db.String(120), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("user_id", "course_slug",
                                          name="uq_user_course"),)


class Note(db.Model):
    """A private, per-user note attached to one module of one course."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    course_slug = db.Column(db.String(80), nullable=False)
    module_id = db.Column(db.String(120), nullable=False)
    body = db.Column(db.Text, default="")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow,
                           onupdate=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("user_id", "course_slug", "module_id",
                                          name="uq_user_course_module"),)


with app.app_context():
    db.create_all()


# ---------------------------------------------------------------- courses
# Static metadata per course; module lists are loaded from content/<slug>/*.md.
COURSES = [
    {"code": "LANGMDL", "title": "Language Modeling from Scratch",
     "blurb": "Build a modern LLM end to end — tokenizer, transformer, kernels, "
              "parallelism, scaling, inference, alignment. Four interview banks included.",
     "hours": "48", "level": "Advanced", "tag": "LLMs & Systems", "accent": "#c6a04e",
     "slug": "language-modeling",
     "outcomes": ["Implement a BPE tokenizer, transformer, and training loop from scratch",
                  "Reason about FLOPs, memory, and parallelism for real training runs",
                  "Optimize inference: KV cache, quantization, speculative decoding",
                  "Align a base model with SFT, DPO, and GRPO",
                  "Answer frontier-lab interview questions across four banks"]},
    {"code": "LLMVLM", "title": "LLM · VLM · RAG · Agents",
     "blurb": "Foundations through frontier: attention, KV cache, RAG, and agents, "
              "with the tradeoffs behind each. Senior-level throughout.",
     "hours": "36", "level": "Foundations", "tag": "Generative AI", "accent": "#8b5cf6",
     "slug": "vlm-guide",
     "outcomes": ["Explain attention, KV cache, and modern decoder design",
                  "Design retrieval-augmented generation pipelines",
                  "Build and reason about agentic systems",
                  "Read papers and speak the vocabulary fluently"]},
    {"code": "MLSYS", "title": "ML System Design",
     "blurb": "Design ML systems the way a staff engineer does: data platforms, "
              "training and serving infra, RAG, agents, recsys, and the interview playbook.",
     "hours": "40", "level": "Advanced", "tag": "System Design", "accent": "#0ea5e9",
     "slug": "ml-system-design",
     "outcomes": ["Frame any ML system-design interview with a repeatable structure",
                  "Design feature platforms and training/serving infrastructure",
                  "Architect retrieval, agents, recsys, search, and fraud systems",
                  "Handle evaluation, observability, and MLOps at scale"]},
    {"code": "MLOPS", "title": "MLOps: Production Machine Learning Systems",
     "blurb": "Serving, monitoring, CI/CD for models, and the failure modes nobody "
              "warns you about. Beginner to architect track.",
     "hours": "40", "level": "Intermediate → Advanced", "tag": "MLOps", "accent": "#ec4899",
     "slug": "mlops",
     "outcomes": ["Serve, monitor, and version models in production",
                  "Build CI/CD pipelines for ML",
                  "Diagnose the failure modes that break deployed models",
                  "Grow from practitioner to ML architect"]},
    {"code": "DATAENG", "title": "Data Engineering",
     "blurb": "Pipelines, warehouses, and the architecture behind them — from first "
              "principles to Fortune-100 scale and the data-architect track.",
     "hours": "36", "level": "Beginner → Advanced", "tag": "Data", "accent": "#f59e0b",
     "slug": "data-engineering",
     "outcomes": ["Build reliable batch and streaming data pipelines",
                  "Model data warehouses and lakehouses",
                  "Design data architecture at Fortune-100 scale",
                  "Follow the path to data architect"]},
]
COURSE_BY_SLUG = {c["slug"]: c for c in COURSES}

CONTENT_DIR = Path(__file__).parent / "content"


def _title_of(md_path):
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return md_path.stem.replace("_", " ").replace("-", " ").title()


def load_modules():
    mods = {}
    for c in COURSES:
        folder = CONTENT_DIR / c["slug"]
        items = []
        if folder.is_dir():
            for p in sorted(folder.glob("*.md")):
                items.append({"id": p.stem, "file": p.name, "title": _title_of(p)})
        mods[c["slug"]] = items
    return mods


MODULES = load_modules()

MD_EXT = ["fenced_code", "tables", "sane_lists", "toc", "attr_list"]


def render_markdown(slug, module_id):
    path = CONTENT_DIR / slug / f"{module_id}.md"
    if not path.is_file():
        return None
    return md.markdown(path.read_text(encoding="utf-8"), extensions=MD_EXT)


def course_view(slug):
    """Meta dict augmented with its module list and count."""
    c = COURSE_BY_SLUG.get(slug)
    if not c:
        return None
    v = dict(c)
    v["modules"] = MODULES.get(slug, [])
    v["module_count"] = len(v["modules"])
    return v


# ---------------------------------------------------------------- auth
def current_user():
    uid = session.get("uid")
    return db.session.get(User, uid) if uid else None


def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not current_user():
            return redirect(url_for("login", next=request.path))
        return f(*a, **kw)
    return wrapper


def enrollment_of(user, slug):
    if not user:
        return None
    return Enrollment.query.filter_by(user_id=user.id, course_slug=slug).first()


def logged_in():
    return session.get("admin") is True


@app.context_processor
def inject_globals():
    return {"BRAND": BRAND, "PASS_MARK": PASS_MARK, "YEAR": datetime.utcnow().year,
            "user": current_user()}


# ---------------------------------------------------------------- helpers
BRAND_CODE = "AMLA"


def course_slug_code(title):
    s = "".join(c for c in title.upper() if c.isalnum())
    return s[:5] or "CERT"


def mint_code(course, year):
    for _ in range(30):
        rnd = secrets.token_hex(2).upper()
        code = f"{BRAND_CODE}-{course_slug_code(course)}-{year}-{rnd}"
        if not Certificate.query.filter_by(code=code).first():
            return code
    return f"{BRAND_CODE}-{course_slug_code(course)}-{year}-{secrets.token_hex(3).upper()}"


# ---------------------------------------------------------------- public
@app.route("/")
def index():
    cards = [course_view(c["slug"]) for c in COURSES]
    total_modules = sum(c["module_count"] for c in cards)
    return render_template("index.html", courses=cards,
                           total_modules=total_modules)


@app.route("/course/<slug>")
def course_detail(slug):
    c = course_view(slug)
    if not c:
        abort(404)
    enr = enrollment_of(current_user(), slug)
    return render_template("course_detail.html", course=c, enrollment=enr)


@app.route("/course/<slug>/enroll", methods=["POST"])
@login_required
def enroll(slug):
    c = course_view(slug)
    if not c:
        abort(404)
    user = current_user()
    if not enrollment_of(user, slug):
        db.session.add(Enrollment(user_id=user.id, course_slug=slug))
        db.session.commit()
        flash(f"You're enrolled in {c['title']}.")
    first = c["modules"][0]["id"] if c["modules"] else None
    if first:
        return redirect(url_for("module", slug=slug, module_id=first))
    return redirect(url_for("course_detail", slug=slug))


@app.route("/course/<slug>/<module_id>")
@login_required
def module(slug, module_id):
    c = course_view(slug)
    if not c:
        abort(404)
    user = current_user()
    enr = enrollment_of(user, slug)
    if not enr:
        flash("Enroll in this course to read its modules.")
        return redirect(url_for("course_detail", slug=slug))
    ids = [m["id"] for m in c["modules"]]
    if module_id not in ids:
        abort(404)
    html = render_markdown(slug, module_id)
    idx = ids.index(module_id)
    prev_m = c["modules"][idx - 1] if idx > 0 else None
    next_m = c["modules"][idx + 1] if idx < len(ids) - 1 else None
    # remember last position for "continue"
    enr.last_module = module_id
    db.session.commit()
    note = Note.query.filter_by(user_id=user.id, course_slug=slug,
                                module_id=module_id).first()
    return render_template("module.html", course=c, current=c["modules"][idx],
                           body=html, prev_m=prev_m, next_m=next_m, index=idx,
                           note_body=(note.body if note else ""))


@app.route("/course/<slug>/<module_id>/note", methods=["POST"])
@login_required
def save_note(slug, module_id):
    user = current_user()
    if not enrollment_of(user, slug):
        abort(403)
    if slug not in COURSE_BY_SLUG or module_id not in [m["id"] for m in MODULES.get(slug, [])]:
        abort(404)
    data = request.get_json(silent=True) or {}
    body = (data.get("body") or "")[:20000]
    note = Note.query.filter_by(user_id=user.id, course_slug=slug,
                                module_id=module_id).first()
    if note:
        note.body = body
    else:
        db.session.add(Note(user_id=user.id, course_slug=slug,
                            module_id=module_id, body=body))
    db.session.commit()
    return {"ok": True}


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    enrolled = []
    for e in Enrollment.query.filter_by(user_id=user.id).order_by(
            Enrollment.created_at.desc()).all():
        c = course_view(e.course_slug)
        if not c:
            continue
        resume = e.last_module or (c["modules"][0]["id"] if c["modules"] else None)
        enrolled.append({"course": c, "enrollment": e, "resume": resume})
    return render_template("dashboard.html", enrolled=enrolled)


# ---------------------------------------------------------------- student auth
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        if not name or not EMAIL_RE.match(email) or len(pw) < 8:
            flash("Enter a name, a valid email, and a password of at least 8 characters.")
            return render_template("register.html", name=name, email=email)
        if User.query.filter_by(email=email).first():
            flash("That email is already registered. Try logging in.")
            return render_template("register.html", name=name, email=email)
        u = User(email=email, name=name, password_hash=generate_password_hash(pw))
        db.session.add(u)
        db.session.commit()
        session["uid"] = u.id
        nxt = request.args.get("next")
        return redirect(nxt or url_for("dashboard"))
    return render_template("register.html", name="", email="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        u = User.query.filter_by(email=email).first()
        if u and check_password_hash(u.password_hash, pw):
            session["uid"] = u.id
            nxt = request.args.get("next")
            return redirect(nxt or url_for("dashboard"))
        flash("Wrong email or password.")
        return render_template("login.html", email=email)
    return render_template("login.html", email="")


@app.route("/logout")
def logout():
    session.pop("uid", None)
    return redirect(url_for("index"))


# ---------------------------------------------------------------- verify
@app.route("/verify", methods=["GET"])
def verify():
    return render_template("verify.html", code="", result=None)


@app.route("/verify/<code>")
@app.route("/c/<code>")
def verify_code(code):
    code = code.strip().upper()
    cert = Certificate.query.filter_by(code=code).first()
    if not cert:
        result = {"status": "invalid"}
    elif cert.revoked:
        result = {"status": "revoked", "cert": cert}
    else:
        result = {"status": "valid", "cert": cert}
    return render_template("verify.html", code=code, result=result)


@app.route("/certificate/<code>")
def certificate(code):
    cert = Certificate.query.filter_by(code=code.strip().upper()).first()
    if not cert:
        abort(404)
    verify_url = request.host_url.rstrip("/") + url_for("verify_code", code=cert.code)
    return render_template(
        "certificate.html", cert=cert, verify_url=verify_url,
        brand=BRAND, monogram=MONOGRAM, est=EST_YEAR, program=PROGRAM_LINE,
        instructor=INSTRUCTOR, instructor_title=INSTRUCTOR_TITLE,
        cosigner=COSIGNER, cosigner_title=COSIGNER_TITLE,
        art_bg=art.bg_svg(), art_seal=art.seal_svg(BRAND),
        art_corner=art.corner_svg(), art_crest=art.crest_svg(MONOGRAM),
    )


# ---------------------------------------------------------------- admin
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        pw = request.form.get("password", "")
        if hmac.compare_digest(pw, ADMIN_PASSWORD):
            session["admin"] = True
            return redirect(url_for("admin"))
        flash("Wrong password. Try again.")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("index"))


@app.route("/admin")
def admin():
    if not logged_in():
        return redirect(url_for("admin_login"))
    certs = Certificate.query.order_by(Certificate.created_at.desc()).all()
    valid = sum(1 for c in certs if not c.revoked)
    default_pw = ADMIN_PASSWORD == "change-me"
    return render_template("admin.html", certs=certs, valid=valid,
                           courses=COURSES, today=date.today().isoformat(),
                           default_pw=default_pw,
                           students=User.query.count(),
                           enrollments=Enrollment.query.count())


@app.route("/admin/issue", methods=["POST"])
def admin_issue():
    if not logged_in():
        abort(403)
    name = request.form.get("name", "").strip()
    course = request.form.get("course", "").strip()
    hours = request.form.get("hours", "").strip()
    issued_on = request.form.get("issued_on", "").strip() or date.today().isoformat()
    try:
        score = int(request.form.get("score", ""))
    except ValueError:
        score = -1

    if not name or not course or score < 0:
        flash("Fill in a name, a course, and a score.")
        return redirect(url_for("admin"))
    if score < PASS_MARK:
        flash(f"Score {score}% is below the {PASS_MARK}% pass mark. "
              f"No certificate issued.")
        return redirect(url_for("admin"))

    year = issued_on[:4]
    code = mint_code(course, year)
    cert = Certificate(code=code, name=name, course=course, score=score,
                       hours=hours, issued_on=issued_on)
    db.session.add(cert)
    db.session.commit()
    flash(f"Issued {code} to {name}.")
    return redirect(url_for("admin"))


@app.route("/admin/toggle/<code>", methods=["POST"])
def admin_toggle(code):
    if not logged_in():
        abort(403)
    cert = Certificate.query.filter_by(code=code).first_or_404()
    cert.revoked = not cert.revoked
    db.session.commit()
    flash(("Revoked " if cert.revoked else "Restored ") + code + ".")
    return redirect(url_for("admin"))


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
