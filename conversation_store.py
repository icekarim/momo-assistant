"""Persistent conversation memory using Firestore."""

import threading
from datetime import datetime, timezone
from google.cloud import firestore
from cachetools import TTLCache
import config

_db = None

_conversation_cache = TTLCache(maxsize=64, ttl=60)
_cache_lock = threading.Lock()


def _safe_doc_id(user_id):
    """Sanitize user_id for use as a Firestore document ID.
    Add-on payloads use 'users/123456' which contains '/' — replace with '_'."""
    return user_id.replace("/", "_")


def get_db():
    global _db
    if _db is None:
        _db = firestore.Client(project=config.GCP_PROJECT_ID, database=config.FIRESTORE_DATABASE)
    return _db


def get_conversation(user_id):
    """Get conversation history for a user. Cached for 60s to avoid
    redundant Firestore reads during rapid back-and-forth exchanges."""
    cache_key = _safe_doc_id(user_id)
    with _cache_lock:
        cached = _conversation_cache.get(cache_key)
    if cached is not None:
        return cached

    db = get_db()
    doc = db.collection(config.FIRESTORE_COLLECTION).document(cache_key).get()

    turns = doc.to_dict().get("turns", []) if doc.exists else []
    with _cache_lock:
        _conversation_cache[cache_key] = turns
    return turns


def add_turn(user_id, role, content):
    """Add a conversation turn and trim to max length."""
    db = get_db()
    cache_key = _safe_doc_id(user_id)
    doc_ref = db.collection(config.FIRESTORE_COLLECTION).document(cache_key)

    doc = doc_ref.get()
    turns = doc.to_dict().get("turns", []) if doc.exists else []

    turns.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    if len(turns) > config.MAX_CONVERSATION_TURNS:
        turns = turns[-config.MAX_CONVERSATION_TURNS:]

    doc_ref.set({
        "turns": turns,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })

    with _cache_lock:
        _conversation_cache[cache_key] = turns


def clear_conversation(user_id):
    """Clear conversation history for a user."""
    db = get_db()
    cache_key = _safe_doc_id(user_id)
    db.collection(config.FIRESTORE_COLLECTION).document(cache_key).delete()

    with _cache_lock:
        _conversation_cache.pop(cache_key, None)


def has_email_alert_been_sent(message_id):
    """Return True if this Gmail message already triggered a proactive alert."""
    db = get_db()
    doc = db.collection(config.FIRESTORE_EMAIL_ALERTS_COLLECTION).document(message_id).get()
    return doc.exists


def mark_email_alert_sent(email):
    """Persist a sent-email alert marker to avoid duplicate notifications."""
    db = get_db()
    db.collection(config.FIRESTORE_EMAIL_ALERTS_COLLECTION).document(email["id"]).set(
        {
            "message_id": email["id"],
            "thread_id": email.get("thread_id", ""),
            "subject": email.get("subject", ""),
            "from": email.get("from", ""),
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def has_debrief_been_sent(calendar_event_id):
    """Return True if a post-meeting debrief was already sent for this event."""
    db = get_db()
    doc = db.collection(config.FIRESTORE_MEETING_DEBRIEFS_COLLECTION).document(calendar_event_id).get()
    return doc.exists


def mark_debrief_sent(calendar_event_id, meeting_title):
    """Record that a debrief was sent so we don't send it again."""
    db = get_db()
    db.collection(config.FIRESTORE_MEETING_DEBRIEFS_COLLECTION).document(calendar_event_id).set(
        {
            "event_id": calendar_event_id,
            "title": meeting_title,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def has_prep_been_sent(calendar_event_id):
    """Return True if a pre-meeting prep was already sent for this event."""
    db = get_db()
    doc = db.collection(config.FIRESTORE_MEETING_PREP_COLLECTION).document(calendar_event_id).get()
    return doc.exists


def mark_prep_sent(calendar_event_id, meeting_title):
    """Record that a meeting prep was sent so we don't send it again."""
    db = get_db()
    db.collection(config.FIRESTORE_MEETING_PREP_COLLECTION).document(calendar_event_id).set(
        {
            "event_id": calendar_event_id,
            "title": meeting_title,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def has_nudge_been_sent(nudge_key):
    """Return True if this nudge was already sent within the cooldown window."""
    db = get_db()
    doc = db.collection(config.FIRESTORE_NUDGES_COLLECTION).document(nudge_key).get()
    if not doc.exists:
        return False
    data = doc.to_dict()
    sent_at = data.get("sent_at", "")
    if not sent_at:
        return False
    try:
        sent_dt = datetime.fromisoformat(sent_at)
        from datetime import timedelta
        cooldown = timedelta(days=config.NUDGE_COOLDOWN_DAYS)
        return (datetime.now(timezone.utc) - sent_dt) < cooldown
    except (ValueError, TypeError):
        return False


def mark_nudge_sent(nudge_key, nudge_type, title):
    """Record that a nudge was sent to enforce cooldown."""
    db = get_db()
    db.collection(config.FIRESTORE_NUDGES_COLLECTION).document(nudge_key).set(
        {
            "nudge_key": nudge_key,
            "nudge_type": nudge_type,
            "title": title,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
    )
