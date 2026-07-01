import os
import re
import json
import secrets
import hmac
from functools import wraps
from datetime import datetime, date
from pathlib import Path

import markdown as md
from flask import (Flask, render_template, request, redirect, url_for,
                   session, abort, flash)
from werkzeug.security import generate_password_hash, check_password_hash

import art
import data

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

# Data lives in Firestore (see data.py) so it survives redeploys. Credentials come
# from FIREBASE_CREDENTIALS / GOOGLE_APPLICATION_CREDENTIALS / FIRESTORE_EMULATOR_HOST.


# ---------------------------------------------------------------- courses
# Static metadata per course; module lists are loaded from content/<slug>/*.md.
COURSES = [
    {"code": "LANGMDL", "title": "Language Modeling from Scratch",
     "blurb": "Build a modern LLM end to end — tokenizer, transformer, kernels, "
              "parallelism, scaling, inference, alignment. Four interview banks included.",
     "hours": "48", "level": "Advanced", "tag": "LLMs & Systems", "accent": "#c6a04e",
     "order": 4, "path_note": "Go deep — build a modern LLM end to end. The hardest track.",
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
     "order": 1, "path_note": "Start here — the vocabulary and mental models the rest build on.",
     "slug": "vlm-guide",
     "outcomes": ["Explain attention, KV cache, and modern decoder design",
                  "Design retrieval-augmented generation pipelines",
                  "Build and reason about agentic systems",
                  "Read papers and speak the vocabulary fluently"]},
    {"code": "MLSYS", "title": "ML System Design",
     "blurb": "Design ML systems the way a staff engineer does: data platforms, "
              "training and serving infra, RAG, agents, recsys, and the interview playbook.",
     "hours": "40", "level": "Advanced", "tag": "System Design", "accent": "#0ea5e9",
     "order": 5, "path_note": "Capstone — design end-to-end systems and prep for interviews.",
     "slug": "ml-system-design",
     "outcomes": ["Frame any ML system-design interview with a repeatable structure",
                  "Design feature platforms and training/serving infrastructure",
                  "Architect retrieval, agents, recsys, search, and fraud systems",
                  "Handle evaluation, observability, and MLOps at scale"]},
    {"code": "MLOPS", "title": "MLOps: Production Machine Learning Systems",
     "blurb": "Serving, monitoring, CI/CD for models, and the failure modes nobody "
              "warns you about. Beginner to architect track.",
     "hours": "40", "level": "Intermediate → Advanced", "tag": "MLOps", "accent": "#ec4899",
     "order": 3, "path_note": "Take models to production: serving, monitoring, CI/CD.",
     "slug": "mlops",
     "outcomes": ["Serve, monitor, and version models in production",
                  "Build CI/CD pipelines for ML",
                  "Diagnose the failure modes that break deployed models",
                  "Grow from practitioner to ML architect"]},
    {"code": "DATAENG", "title": "Data Engineering",
     "blurb": "Pipelines, warehouses, and the architecture behind them — from first "
              "principles to Fortune-100 scale and the data-architect track.",
     "hours": "36", "level": "Beginner → Advanced", "tag": "Data", "accent": "#f59e0b",
     "order": 2, "path_note": "The data layer every model and system sits on top of.",
     "slug": "data-engineering",
     "outcomes": ["Build reliable batch and streaming data pipelines",
                  "Model data warehouses and lakehouses",
                  "Design data architecture at Fortune-100 scale",
                  "Follow the path to data architect"]},
]
# Open foundation track: free, login required (so notes/progress work), no certificate.
# Content is authored course-by-course; until a course has content files its syllabus
# is shown as "lessons coming soon". `order` is the recommended beginner sequence.
def _syl(*pairs):
    return [{"title": t, "desc": d} for t, d in pairs]


FOUNDATIONS = [
    {"code": "PY", "title": "Python for Data & ML", "slug": "python-foundations",
     "tag": "Programming", "accent": "#3776ab", "level": "Beginner", "hours": "12", "order": 1,
     "foundation": True, "certificate": False,
     "blurb": "Start from zero: Python syntax through NumPy, the language every other course assumes.",
     "syllabus": _syl(
        ("Setup and the REPL", "Install Python, run code, use a notebook and an editor."),
        ("Variables, types, and control flow", "Numbers, strings, booleans, if/for/while."),
        ("Data structures", "Lists, dicts, sets, tuples and when to use each."),
        ("Functions and modules", "Write reusable functions; organize code into modules."),
        ("Files and error handling", "Read/write files; handle exceptions cleanly."),
        ("Object-oriented basics", "Classes, methods, and when objects help."),
        ("Comprehensions and iterators", "Pythonic loops, generators, and lazy iteration."),
        ("NumPy and vectorized math", "Arrays, broadcasting, and why loops are slow."))},
    {"code": "LINALG", "title": "Linear Algebra for ML", "slug": "linear-algebra",
     "tag": "Math", "accent": "#16a34a", "level": "Beginner", "hours": "10", "order": 2,
     "foundation": True, "certificate": False,
     "blurb": "Vectors, matrices, and matrix multiplication — the language tensors are written in.",
     "syllabus": _syl(
        ("Vectors and geometry", "What a vector is; addition, scaling, and intuition."),
        ("Matrices and operations", "Shapes, transpose, and basic operations."),
        ("Matrix multiplication", "The one operation that dominates ML compute."),
        ("Norms and distances", "Measuring size and similarity."),
        ("Dot products and projections", "Similarity, angles, and why attention uses them."),
        ("Eigenvalues and SVD", "Decompositions, intuitively, and where they show up."),
        ("Linear algebra in NumPy", "Do all of the above in code."))},
    {"code": "CALC", "title": "Calculus & Gradients", "slug": "calculus-gradients",
     "tag": "Math", "accent": "#0d9488", "level": "Beginner", "hours": "9", "order": 3,
     "foundation": True, "certificate": False,
     "blurb": "Derivatives, the chain rule, and gradient descent — how models actually learn.",
     "syllabus": _syl(
        ("Functions and limits", "The setup you need, no more."),
        ("Derivatives", "Slope, rate of change, and notation."),
        ("The chain rule", "Composing functions — the heart of backprop."),
        ("Partial derivatives and gradients", "Slopes in many dimensions."),
        ("Gradient descent", "Follow the slope downhill to minimize loss."),
        ("Backprop intuition", "How the chain rule trains a network."))},
    {"code": "PROB", "title": "Probability & Statistics for ML", "slug": "probability-stats",
     "tag": "Math", "accent": "#7c3aed", "level": "Beginner", "hours": "11", "order": 4,
     "foundation": True, "certificate": False,
     "blurb": "Distributions, Bayes, and estimation — the reasoning under every metric and loss.",
     "syllabus": _syl(
        ("Probability basics", "Events, conditional probability, independence."),
        ("Random variables and distributions", "Discrete and continuous; the common ones."),
        ("Expectation and variance", "Averages, spread, and what they tell you."),
        ("Bayes' theorem", "Updating beliefs from evidence."),
        ("Sampling and the CLT", "Why sample means behave nicely."),
        ("Estimation and maximum likelihood", "Fitting parameters to data."),
        ("Hypothesis testing basics", "p-values and what they do and don't mean."))},
    {"code": "CLI", "title": "Command Line & Git", "slug": "cli-git",
     "tag": "Tooling", "accent": "#475569", "level": "Beginner", "hours": "8", "order": 5,
     "foundation": True, "certificate": False,
     "blurb": "The terminal, shell, and version control — the daily workflow of every engineer.",
     "syllabus": _syl(
        ("The terminal and shell", "Where work happens; navigating the filesystem."),
        ("Files and text tools", "cat, grep, pipes, and redirection."),
        ("Bash scripting basics", "Variables, loops, and small automations."),
        ("Git fundamentals", "Commits, history, and the staging area."),
        ("Branching and merging", "Work in parallel without chaos."),
        ("GitHub and pull requests", "Collaborate and review changes."))},
    {"code": "SQL", "title": "SQL & Databases", "slug": "sql-databases",
     "tag": "Data", "accent": "#0891b2", "level": "Beginner", "hours": "10", "order": 6,
     "foundation": True, "certificate": False,
     "blurb": "Query and model relational data — the backbone of data engineering and analytics.",
     "syllabus": _syl(
        ("The relational model", "Tables, rows, keys, and relationships."),
        ("SELECT and filtering", "Get exactly the rows you want."),
        ("Joins", "Combine tables the right way."),
        ("Aggregation and GROUP BY", "Summarize data."),
        ("Subqueries and CTEs", "Compose complex queries readably."),
        ("Indexing and performance", "Why some queries are slow."),
        ("Schema design basics", "Model a domain without pain later."))},
    {"code": "PANDAS", "title": "Data Analysis with Pandas", "slug": "pandas-analysis",
     "tag": "Data", "accent": "#db2777", "level": "Beginner", "hours": "10", "order": 7,
     "foundation": True, "certificate": False,
     "blurb": "Load, clean, and explore real data with DataFrames — the analyst's core tool.",
     "syllabus": _syl(
        ("Series and DataFrames", "The two objects everything is built on."),
        ("Loading and inspecting data", "CSV, JSON, and a first look."),
        ("Selection and filtering", "Slice data by label and condition."),
        ("Cleaning and missing data", "Handle nulls, types, and duplicates."),
        ("GroupBy and aggregation", "Split-apply-combine."),
        ("Merging and reshaping", "Join, pivot, and melt."),
        ("Plotting and EDA", "See the data before you model it."))},
    {"code": "MLF", "title": "Machine Learning Foundations", "slug": "ml-foundations",
     "tag": "Machine Learning", "accent": "#ea580c", "level": "Beginner → Intermediate", "hours": "14", "order": 8,
     "foundation": True, "certificate": False,
     "blurb": "Classical ML end to end: regression, trees, evaluation, and the traps that fool beginners.",
     "syllabus": _syl(
        ("What machine learning is", "Supervised, unsupervised, and the workflow."),
        ("Data splits and leakage", "Train/validation/test done right."),
        ("Linear and logistic regression", "The workhorse models."),
        ("Decision trees and ensembles", "Trees, random forests, gradient boosting."),
        ("Evaluation metrics", "Accuracy, precision/recall, ROC, and when each lies."),
        ("Overfitting and regularization", "Bias, variance, and controlling them."),
        ("Cross-validation", "Trustworthy estimates of performance."),
        ("A first end-to-end project", "From raw data to an evaluated model."))},
    {"code": "DLF", "title": "Deep Learning & Neural Nets", "slug": "deep-learning-foundations",
     "tag": "Deep Learning", "accent": "#6366f1", "level": "Beginner → Intermediate", "hours": "14", "order": 9,
     "foundation": True, "certificate": False,
     "blurb": "From a single neuron to a trained network: forward pass, backprop, and optimization.",
     "syllabus": _syl(
        ("From linear models to neurons", "Why stack and nonlinearize."),
        ("Activation functions", "ReLU and friends, and what they do."),
        ("The forward pass", "How a network computes an output."),
        ("Backpropagation", "Gradients through the whole network."),
        ("Loss functions and optimizers", "What you minimize and how."),
        ("Training loops and batching", "Epochs, mini-batches, learning rate."),
        ("Regularization", "Dropout, weight decay, early stopping."),
        ("Your first neural network", "Train one end to end."))},
    {"code": "PYT", "title": "PyTorch Essentials", "slug": "pytorch-essentials",
     "tag": "Deep Learning", "accent": "#ee4c2c", "level": "Beginner → Intermediate", "hours": "10", "order": 10,
     "foundation": True, "certificate": False,
     "blurb": "The framework the deep courses assume: tensors, autograd, modules, and a real training loop.",
     "syllabus": _syl(
        ("Tensors", "Creation, shapes, and operations."),
        ("Autograd", "Automatic differentiation, demystified."),
        ("nn.Module and layers", "Build models from components."),
        ("Datasets and DataLoaders", "Feed data efficiently."),
        ("The training loop", "Forward, loss, backward, step."),
        ("Saving and loading", "Checkpoints and inference."),
        ("Devices and GPUs", "Move work to the accelerator."))},
    {"code": "DOCK", "title": "Docker & Containers for ML", "slug": "docker-containers",
     "tag": "Ops", "accent": "#2496ed", "level": "Beginner", "hours": "8", "order": 11,
     "foundation": True, "certificate": False,
     "blurb": "Package anything to run anywhere — the unit of deployment for modern ML.",
     "syllabus": _syl(
        ("Why containers", "The 'works on my machine' problem, solved."),
        ("Images vs containers", "The two core concepts."),
        ("Dockerfile basics", "Build a reproducible image."),
        ("Running: ports and volumes", "Talk to a container and persist data."),
        ("Docker Compose", "Run multi-service stacks."),
        ("Packaging an ML model", "Containerize an inference service."),
        ("Registries and sharing", "Push, pull, and deploy."))},
    {"code": "CLOUD", "title": "Cloud & Linux Fundamentals", "slug": "cloud-linux",
     "tag": "Ops", "accent": "#d97706", "level": "Beginner", "hours": "10", "order": 12,
     "foundation": True, "certificate": False,
     "blurb": "Linux and cloud building blocks — compute, storage, networking, and IAM — without the jargon.",
     "syllabus": _syl(
        ("Linux essentials", "The filesystem, shell, and packages."),
        ("Users, permissions, processes", "Who can do what, and what's running."),
        ("Compute: virtual machines", "Rent a computer in the cloud."),
        ("Storage and object stores", "Disks, buckets, and when to use each."),
        ("Networking basics", "IPs, ports, DNS, and firewalls."),
        ("IAM and security", "Identities, roles, and least privilege."),
        ("Cost and trade-offs", "Pick the right service without overspending."))},
]

# Cloud provider deep-dives: detailed, certificate-eligible, in the Cloud category
# (not part of the 5-step recommended path). Ordered after the core certificate track.
CLOUD = [
    {"code": "AWS", "title": "AWS for ML Engineers", "slug": "aws-for-ml",
     "tag": "Cloud", "accent": "#ff9900", "level": "Beginner → Advanced", "hours": "34", "order": 6,
     "certificate": True,
     "blurb": "Amazon Web Services from first principles to building and shipping ML systems — "
              "IAM, compute, storage, containers, SageMaker AI, and Bedrock.",
     "outcomes": ["Navigate AWS core services (IAM, EC2, S3, VPC) confidently",
                  "Train and deploy models with Amazon SageMaker AI",
                  "Serve models with real-time, serverless, and batch inference",
                  "Architect a production ML system on AWS end to end"]},
    {"code": "AZURE", "title": "Azure for ML Engineers", "slug": "azure-for-ml",
     "tag": "Cloud", "accent": "#0078d4", "level": "Beginner → Advanced", "hours": "34", "order": 7,
     "certificate": True,
     "blurb": "Microsoft Azure end to end for ML — Entra ID, compute, storage, AKS, "
              "Azure Machine Learning, and Azure AI Foundry / OpenAI.",
     "outcomes": ["Work with Azure core services and Entra ID securely",
                  "Train and deploy with Azure Machine Learning",
                  "Use Azure AI Foundry and Azure OpenAI for GenAI systems",
                  "Architect a production ML system on Azure end to end"]},
    {"code": "GCP", "title": "GCP for ML Engineers", "slug": "gcp-for-ml",
     "tag": "Cloud", "accent": "#34a853", "level": "Beginner → Advanced", "hours": "34", "order": 8,
     "certificate": True,
     "blurb": "Google Cloud from fundamentals to production ML — IAM, Compute Engine, "
              "Cloud Storage, BigQuery, GKE, and Vertex AI with Gemini.",
     "outcomes": ["Navigate GCP core services (IAM, Compute Engine, Cloud Storage)",
                  "Use BigQuery and Vertex AI for data and training",
                  "Deploy models and Gemini-based GenAI on Vertex AI",
                  "Architect a production ML system on GCP end to end"]},
]

ALL_COURSES = COURSES + CLOUD + FOUNDATIONS
COURSE_BY_SLUG = {c["slug"]: c for c in ALL_COURSES}

# ---------------------------------------------------------------- categories
# Browse-by-category taxonomy across every course. Order = display order.
CATEGORIES = [
    ("Programming & Tools", ["python-foundations", "cli-git"]),
    ("Mathematics", ["linear-algebra", "calculus-gradients", "probability-stats"]),
    ("Data & Analytics", ["sql-databases", "pandas-analysis", "data-engineering"]),
    ("Machine Learning", ["ml-foundations", "deep-learning-foundations", "pytorch-essentials"]),
    ("LLMs & Systems", ["vlm-guide", "language-modeling", "ml-system-design"]),
    ("Cloud & Infrastructure",
     ["cloud-linux", "docker-containers", "aws-for-ml", "azure-for-ml", "gcp-for-ml", "mlops"]),
]
CATEGORY_OF = {slug: name for name, slugs in CATEGORIES for slug in slugs}

CONTENT_DIR = Path(__file__).parent / "content"


def _title_of(md_path):
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return md_path.stem.replace("_", " ").replace("-", " ").title()


def load_modules():
    mods = {}
    for c in ALL_COURSES:
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
    v["certificate"] = c.get("certificate", True)
    v["foundation"] = c.get("foundation", False)
    v["syllabus"] = c.get("syllabus", [])
    v["category"] = CATEGORY_OF.get(slug, "Other")
    v["has_content"] = bool(v["modules"])
    v["module_count"] = len(v["modules"]) if v["modules"] else len(v["syllabus"])
    return v


# ---------------------------------------------------------------- auth
def current_user():
    uid = session.get("uid")
    return data.get_user(uid) if uid else None


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
    return data.get_enrollment(user.id, slug)


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
        if not data.certificate_exists(code):
            return code
    return f"{BRAND_CODE}-{course_slug_code(course)}-{year}-{secrets.token_hex(3).upper()}"


# ---------------------------------------------------------------- public
@app.route("/")
def index():
    path_cards = sorted((course_view(c["slug"]) for c in COURSES),
                        key=lambda c: c["order"])
    foundations = sorted((course_view(c["slug"]) for c in FOUNDATIONS),
                         key=lambda c: c["order"])
    grouped = [(name, [course_view(s) for s in slugs]) for name, slugs in CATEGORIES]
    total_modules = sum(len(MODULES.get(c["slug"], [])) for c in ALL_COURSES)
    return render_template("index.html", courses=path_cards, foundations=foundations,
                           grouped=grouped, n_courses=len(ALL_COURSES),
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
        data.create_enrollment(user.id, slug)
        flash(f"You're enrolled in {c['title']}.")
    first = c["modules"][0]["id"] if c["modules"] else None
    if first:
        return redirect(url_for("module", slug=slug, module_id=first))
    flash("Lessons for this course are coming soon — you'll find them here when ready.")
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
    data.set_last_module(user.id, slug, module_id)
    note = data.get_note(user.id, slug, module_id)
    return render_template("module.html", course=c, current=c["modules"][idx],
                           body=html, prev_m=prev_m, next_m=next_m, index=idx,
                           note_body=(note.body if note else ""),
                           note_highlights=json.dumps(note.highlights if note else []))


def _module_guard(user, slug, module_id):
    if not enrollment_of(user, slug):
        abort(403)
    if slug not in COURSE_BY_SLUG or module_id not in [m["id"] for m in MODULES.get(slug, [])]:
        abort(404)


@app.route("/course/<slug>/<module_id>/note", methods=["POST"])
@login_required
def save_note(slug, module_id):
    user = current_user()
    _module_guard(user, slug, module_id)
    payload = request.get_json(silent=True) or {}
    data.save_note_body(user.id, slug, module_id, (payload.get("body") or "")[:20000])
    return {"ok": True}


@app.route("/course/<slug>/<module_id>/highlights", methods=["POST"])
@login_required
def save_highlights(slug, module_id):
    user = current_user()
    _module_guard(user, slug, module_id)
    payload = request.get_json(silent=True) or {}
    items = payload.get("highlights")
    if not isinstance(items, list):
        return {"ok": False, "error": "highlights must be a list"}, 400
    # keep only the fields we expect, cap size defensively
    clean = []
    for h in items[:500]:
        if not isinstance(h, dict):
            continue
        clean.append({
            "id": str(h.get("id", ""))[:40],
            "start": int(h.get("start", 0)),
            "end": int(h.get("end", 0)),
            "text": str(h.get("text", ""))[:2000],
            "color": (h.get("color") if h.get("color") in ("y", "g", "b", "p") else "y"),
            "note": str(h.get("note", ""))[:4000],
        })
    data.save_highlights(user.id, slug, module_id, clean)
    return {"ok": True, "count": len(clean)}


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    enrolled = []
    for e in data.list_enrollments(user.id):
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
        if data.get_user_by_email(email):
            flash("That email is already registered. Try logging in.")
            return render_template("register.html", name=name, email=email)
        uid = data.create_user(email, name, generate_password_hash(pw))
        session["uid"] = uid
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
        u = data.get_user_by_email(email)
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
    cert = data.get_certificate(code)
    if not cert:
        result = {"status": "invalid"}
    elif cert.revoked:
        result = {"status": "revoked", "cert": cert}
    else:
        result = {"status": "valid", "cert": cert}
    return render_template("verify.html", code=code, result=result)


@app.route("/certificate/<code>")
def certificate(code):
    cert = data.get_certificate(code.strip().upper())
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
    certs = data.list_certificates()
    valid = sum(1 for c in certs if not c.revoked)
    default_pw = ADMIN_PASSWORD == "change-me"
    return render_template("admin.html", certs=certs, valid=valid,
                           courses=COURSES, today=date.today().isoformat(),
                           default_pw=default_pw,
                           students=data.count_users(),
                           enrollments=data.count_enrollments())


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
    data.create_certificate(code, name, course, score, hours, issued_on)
    flash(f"Issued {code} to {name}.")
    return redirect(url_for("admin"))


@app.route("/admin/toggle/<code>", methods=["POST"])
def admin_toggle(code):
    if not logged_in():
        abort(403)
    cert = data.toggle_certificate(code)
    if not cert:
        abort(404)
    flash(("Revoked " if cert.revoked else "Restored ") + code + ".")
    return redirect(url_for("admin"))


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
