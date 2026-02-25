"""Persistent conversation memory using Firestore."""

from datetime import datetime, timezone
from google.cloud import firestore
import config

_db = None


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
    """Get conversation history for a user."""
    db = get_db()
    doc = db.collection(config.FIRESTORE_COLLECTION).document(_safe_doc_id(user_id)).get()

    if doc.exists:
        data = doc.to_dict()
        return data.get("turns", [])

    return []


def add_turn(user_id, role, content):
    """Add a conversation turn and trim to max length."""
    db = get_db()
    doc_ref = db.collection(config.FIRESTORE_COLLECTION).document(_safe_doc_id(user_id))

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


def clear_conversation(user_id):
    """Clear conversation history for a user."""
    db = get_db()
    db.collection(config.FIRESTORE_COLLECTION).document(_safe_doc_id(user_id)).delete()


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
