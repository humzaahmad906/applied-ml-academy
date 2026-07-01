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
from datetime import datetime

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


def create_certificate(code, name, course, score, hours, issued_on):
    get_db().collection("certificates").document(code).set(
        {"code": code, "name": name, "course": course, "score": score,
         "hours": hours, "issued_on": issued_on, "revoked": False,
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
