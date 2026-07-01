import os
import secrets
import hmac
from datetime import datetime, date

from flask import (Flask, render_template, request, redirect, url_for,
                   session, abort, flash)
from flask_sqlalchemy import SQLAlchemy

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


# ---------------------------------------------------------------- model
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


with app.app_context():
    db.create_all()


# ---------------------------------------------------------------- courses (static content)
# `slug` and `start` map each course to the embedded reader (static/viewer.html):
# the landing card deep-links to `/learn#<slug>/<start>`. Keep these in sync with
# the viewer's course ids if you rebuild it.
COURSES = [
    {"code": "LANGMDL", "title": "Language Modeling from Scratch",
     "blurb": "Build a modern LLM end to end — tokenizer, transformer, kernels, "
              "parallelism, scaling, inference, alignment. Four interview banks included.",
     "hours": "48 hours", "level": "Advanced", "tag": "LLMs & Systems", "modules": 19,
     "accent": "#c6a04e", "slug": "language-modeling", "start": "00-README"},
    {"code": "LLMVLM", "title": "LLM · VLM · RAG · Agents",
     "blurb": "Foundations through frontier: attention, KV cache, RAG, and agents, "
              "with the tradeoffs behind each. Senior-level throughout.",
     "hours": "36 hours", "level": "Foundations", "tag": "Generative AI", "modules": 11,
     "accent": "#6ea8fe", "slug": "vlm-guide", "start": "00_README_and_roadmap"},
    {"code": "MLSYS", "title": "ML System Design",
     "blurb": "Design ML systems the way a staff engineer does: data platforms, "
              "training and serving infra, RAG, agents, recsys, and the interview playbook.",
     "hours": "40 hours", "level": "Advanced", "tag": "System Design", "modules": 11,
     "accent": "#7ee0b8", "slug": "ml-system-design", "start": "00_README_syllabus"},
    {"code": "MLOPS", "title": "MLOps: Production Machine Learning Systems",
     "blurb": "Serving, monitoring, CI/CD for models, and the failure modes nobody "
              "warns you about. Beginner to architect track.",
     "hours": "40 hours", "level": "Intermediate → Advanced", "tag": "MLOps", "modules": 13,
     "accent": "#d08bd0", "slug": "mlops", "start": "00-overview-and-prereqs"},
    {"code": "DATAENG", "title": "Data Engineering",
     "blurb": "Pipelines, warehouses, and the architecture behind them — from first "
              "principles to Fortune-100 scale and the data-architect track.",
     "hours": "36 hours", "level": "Beginner → Advanced", "tag": "Data", "modules": 9,
     "accent": "#e6a366", "slug": "data-engineering", "start": "00-overview-and-prereqs"},
]


# ---------------------------------------------------------------- helpers
BRAND_CODE = "AMLA"   # prefix on every verification code


def course_slug(title):
    s = "".join(c for c in title.upper() if c.isalnum())
    return s[:5] or "CERT"


def mint_code(course, year):
    for _ in range(30):
        rnd = secrets.token_hex(2).upper()
        code = f"{BRAND_CODE}-{course_slug(course)}-{year}-{rnd}"
        if not Certificate.query.filter_by(code=code).first():
            return code
    return f"{BRAND_CODE}-{course_slug(course)}-{year}-{secrets.token_hex(3).upper()}"


def logged_in():
    return session.get("admin") is True


@app.context_processor
def inject_globals():
    return {"BRAND": BRAND, "PASS_MARK": PASS_MARK, "YEAR": datetime.utcnow().year}


# ---------------------------------------------------------------- public routes
@app.route("/")
def index():
    return render_template("index.html", courses=COURSES,
                           issued=Certificate.query.count())


@app.route("/learn")
def learn():
    # The reader is a single self-contained file (all course content + the
    # gamified viewer inline). Client-side hash routing (#course/doc) handles
    # deep links, so one route serving the file is enough.
    return app.send_static_file("viewer.html")


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
                           default_pw=default_pw)


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
