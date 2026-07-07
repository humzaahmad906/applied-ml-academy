"""Firestore-backed data layer for Applied ML Academy.

Server-side access via the firebase-admin SDK. Auth (password hashing, sessions)
stays in the Flask app; this module only persists and reads data.

Credentials (pick one, checked in this order):
  - FIREBASE_CREDENTIALS      : the service-account JSON, as a single string
  - GOOGLE_APPLICATION_CREDENTIALS : path to a service-account JSON file (ADC)
  - FIRESTORE_EMULATOR_HOST   : local emulator (no real credentials needed)

Collections:
  users         doc id = auto            {email, name, password_hash, created_at}
  enrollments   doc id = uid__slug       {user_id, course_slug, last_module, created_at}
  notes         doc id = uid__slug__mod  {user_id, course_slug, module_id, body, highlights[], updated_at}
  certificates  doc id = code            {code, name, course, score, hours, issued_on, created_at, revoked}
"""
import os
import json
from types import SimpleNamespace
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, firestore

_db = None


def get_db():
    global _db
    if _db is not None:
        return _db
    if not firebase_admin._apps:
        cred_json = os.environ.get("FIREBASE_CREDENTIALS")
        project = (os.environ.get("FIREBASE_PROJECT_ID")
                   or os.environ.get("GOOGLE_CLOUD_PROJECT"))
        if cred_json:
            firebase_admin.initialize_app(credentials.Certificate(json.loads(cred_json)))
        elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            firebase_admin.initialize_app()
        elif os.environ.get("FIRESTORE_EMULATOR_HOST"):
            firebase_admin.initialize_app(options={"projectId": project or "demo-mlcourses"})
        else:
            raise RuntimeError(
                "No Firebase credentials. Set FIREBASE_CREDENTIALS (service-account "
                "JSON string) or GOOGLE_APPLICATION_CREDENTIALS (path), or "
                "FIRESTORE_EMULATOR_HOST for local dev.")
    _db = firestore.client()
    return _db


def available():
    """True if Firestore credentials are configured (used for a friendly error page)."""
    return bool(os.environ.get("FIREBASE_CREDENTIALS")
                or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                or os.environ.get("FIRESTORE_EMULATOR_HOST"))


def _ns(doc_id, d):
    return SimpleNamespace(id=doc_id, **d)


# ---------------------------------------------------------------- users
def get_user(uid):
    if not uid:
        return None
    snap = get_db().collection("users").document(uid).get()
    return _ns(snap.id, snap.to_dict()) if snap.exists else None


def get_user_by_email(email):
    q = (get_db().collection("users").where("email", "==", email).limit(1).stream())
    for snap in q:
        return _ns(snap.id, snap.to_dict())
    return None


def create_user(email, name, password_hash):
    ref = get_db().collection("users").document()
    ref.set({"email": email, "name": name, "password_hash": password_hash,
             "created_at": firestore.SERVER_TIMESTAMP})
    return ref.id


def count_users():
    return get_db().collection("users").count().get()[0][0].value


# ---------------------------------------------------------------- enrollments
def _enroll_id(uid, slug):
    return f"{uid}__{slug}"


def get_enrollment(uid, slug):
    if not uid:
        return None
    snap = get_db().collection("enrollments").document(_enroll_id(uid, slug)).get()
    return _ns(snap.id, snap.to_dict()) if snap.exists else None


def create_enrollment(uid, slug):
    ref = get_db().collection("enrollments").document(_enroll_id(uid, slug))
    if not ref.get().exists:
        ref.set({"user_id": uid, "course_slug": slug, "last_module": "",
                 "created_at": firestore.SERVER_TIMESTAMP})


def set_last_module(uid, slug, module_id):
    get_db().collection("enrollments").document(_enroll_id(uid, slug)).update(
        {"last_module": module_id})


def list_enrollments(uid):
    q = (get_db().collection("enrollments").where("user_id", "==", uid).stream())
    items = [_ns(s.id, s.to_dict()) for s in q]
    items.sort(key=lambda e: getattr(e, "created_at", None) or datetime.min, reverse=True)
    return items


def count_enrollments():
    return get_db().collection("enrollments").count().get()[0][0].value


# ---------------------------------------------------------------- progress + gamification
# Completion lives on the enrollment doc (completed: [module_id]); XP/streak/badges
# live on the user doc so a single read (current_user) already carries xp/streak.
def _today():
    return datetime.utcnow().date().isoformat()


def _yesterday():
    return (datetime.utcnow().date() - timedelta(days=1)).isoformat()


def mark_module_complete(uid, slug, module_id):
    """Add module to enrollment.completed if not already there. Returns True if newly added."""
    ref = get_db().collection("enrollments").document(_enroll_id(uid, slug))
    snap = ref.get()
    if not snap.exists:
        return False
    if module_id in (snap.to_dict().get("completed") or []):
        return False
    ref.update({"completed": firestore.ArrayUnion([module_id])})
    return True


def completed_modules(uid, slug):
    if not uid:
        return set()
    snap = get_db().collection("enrollments").document(_enroll_id(uid, slug)).get()
    return set(snap.to_dict().get("completed") or []) if snap.exists else set()


def completed_counts(uid):
    """slug -> number of completed modules, across all enrollments (for the dashboard)."""
    return {e.course_slug: len(getattr(e, "completed", []) or []) for e in list_enrollments(uid)}


def award_activity(uid, xp_gain):
    """Add XP and roll the daily streak forward. Returns the updated stats."""
    ref = get_db().collection("users").document(uid)
    d = ref.get().to_dict() or {}
    last = d.get("last_active")
    streak = d.get("streak", 0) or 0
    if last == _today():
        pass                      # already counted today
    elif last == _yesterday():
        streak += 1
    else:
        streak = 1                # first activity, or a day was missed
    longest = max(d.get("longest_streak", 0) or 0, streak)
    xp = (d.get("xp", 0) or 0) + xp_gain
    ref.update({"xp": xp, "streak": streak, "longest_streak": longest, "last_active": _today()})
    return {"xp": xp, "streak": streak, "longest_streak": longest}


def get_user_stats(uid):
    d = get_db().collection("users").document(uid).get().to_dict() or {}
    return {"xp": d.get("xp", 0) or 0, "streak": d.get("streak", 0) or 0,
            "longest_streak": d.get("longest_streak", 0) or 0,
            "badges": d.get("badges", []) or [], "last_active": d.get("last_active")}


def add_badges(uid, badge_ids):
    if badge_ids:
        get_db().collection("users").document(uid).update(
            {"badges": firestore.ArrayUnion(list(badge_ids))})


def leaderboard(limit=20):
    q = (get_db().collection("users")
         .order_by("xp", direction=firestore.Query.DESCENDING).limit(limit).stream())
    return [_ns(s.id, s.to_dict()) for s in q]


# ---------------------------------------------------------------- notes + highlights
def _note_id(uid, slug, module_id):
    return f"{uid}__{slug}__{module_id}"


def get_note(uid, slug, module_id):
    snap = get_db().collection("notes").document(_note_id(uid, slug, module_id)).get()
    if not snap.exists:
        return None
    d = snap.to_dict()
    d.setdefault("body", "")
    d.setdefault("highlights", [])
    return _ns(snap.id, d)


def save_note_body(uid, slug, module_id, body):
    ref = get_db().collection("notes").document(_note_id(uid, slug, module_id))
    ref.set({"user_id": uid, "course_slug": slug, "module_id": module_id,
             "body": body, "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)


def save_highlights(uid, slug, module_id, highlights):
    ref = get_db().collection("notes").document(_note_id(uid, slug, module_id))
    ref.set({"user_id": uid, "course_slug": slug, "module_id": module_id,
             "highlights": highlights, "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)


# ---------------------------------------------------------------- certificates
def _cert_ns(doc_id, d):
    score = d.get("score", 0)
    band = "Distinction" if score >= 90 else "Pass"
    try:
        issued_display = datetime.strptime(d.get("issued_on", ""), "%Y-%m-%d").strftime("%d %B %Y")
    except ValueError:
        issued_display = d.get("issued_on", "")
    return SimpleNamespace(id=doc_id, band=band, issued_display=issued_display, **d)


def get_certificate(code):
    snap = get_db().collection("certificates").document(code).get()
    return _cert_ns(snap.id, snap.to_dict()) if snap.exists else None


def certificate_exists(code):
    return get_db().collection("certificates").document(code).get().exists


def create_certificate(code, name, course, score, hours, issued_on, kind="course"):
    get_db().collection("certificates").document(code).set(
        {"code": code, "name": name, "course": course, "score": score,
         "hours": hours, "issued_on": issued_on, "revoked": False, "kind": kind,
         "created_at": firestore.SERVER_TIMESTAMP})


def list_certificates():
    q = get_db().collection("certificates").stream()
    items = [_cert_ns(s.id, s.to_dict()) for s in q]
    items.sort(key=lambda c: getattr(c, "created_at", None) or datetime.min, reverse=True)
    return items


def toggle_certificate(code):
    ref = get_db().collection("certificates").document(code)
    snap = ref.get()
    if not snap.exists:
        return None
    revoked = not snap.to_dict().get("revoked", False)
    ref.update({"revoked": revoked})
    return _cert_ns(snap.id, {**snap.to_dict(), "revoked": revoked})


# ---------------------------------------------------------------- capstones
# One capstone submission per (user, certificate-course). doc id = uid__slug.
# Phase 1: learners submit; an admin reviews and decides. Peer/AI review is layered
# on later (see CAPSTONE_REVIEW_SPEC.md) — no grading logic lives here yet.
def _capstone_id(uid, slug):
    return f"{uid}__{slug}"


def get_capstone(uid, slug):
    if not uid:
        return None
    snap = get_db().collection("capstones").document(_capstone_id(uid, slug)).get()
    return _ns(snap.id, snap.to_dict()) if snap.exists else None


def submit_capstone(uid, slug, repo_url, summary, artifact_urls):
    """Create or re-submit a capstone. Resubmitting bumps the version and resets
    the review state back to `submitted` (clearing any prior decision)."""
    ref = get_db().collection("capstones").document(_capstone_id(uid, slug))
    snap = ref.get()
    version = (snap.to_dict().get("version", 0) + 1) if snap.exists else 1
    doc = {"user_id": uid, "course_slug": slug, "repo_url": repo_url,
           "summary": summary, "artifact_urls": artifact_urls, "version": version,
           "status": "submitted", "score": None, "verdict": None, "decided_at": None,
           "submitted_at": firestore.SERVER_TIMESTAMP}
    if not snap.exists:
        doc["created_at"] = firestore.SERVER_TIMESTAMP
    ref.set(doc, merge=True)
    return version


def list_capstones():
    """All submissions, newest submission first (admin review queue)."""
    q = get_db().collection("capstones").stream()
    items = [_ns(s.id, s.to_dict()) for s in q]
    items.sort(key=lambda c: getattr(c, "submitted_at", None) or datetime.min, reverse=True)
    return items


def decide_capstone(uid, slug, status, score=None, verdict=None):
    ref = get_db().collection("capstones").document(_capstone_id(uid, slug))
    if not ref.get().exists:
        return None
    upd = {"status": status, "verdict": verdict, "decided_at": firestore.SERVER_TIMESTAMP}
    if score is not None:
        upd["score"] = score
    ref.update(upd)
    return get_capstone(uid, slug)
