"""Persistent conversation memory using Firestore."""

import threading
from datetime import datetime, timezone
from google.cloud import firestore
from cachetools import TTLCache
import config

_db = None

_conversation_cache = TTLCache(maxsize=64, ttl=60)
_cache_lock = threading.Lock()


def _safe_doc_id(scope_id):
    """Sanitize a conversation or task scope ID for Firestore doc IDs."""
    return scope_id.replace("/", "_")


def conversation_scope(user_id: str = "", space: str = "") -> str:
    """Return the conversation scope for the current chat context.

    Google Chat replies happen inside a space, so scope chat history to that
    space when available. This lets follow-up questions reference proactive
    messages that were posted into the same room or DM.
    """
    if space:
        return f"space:{space}"
    if user_id:
        return f"user:{user_id}"
    return "user:unknown"


def _pending_task_doc_id(scope_id: str = "latest") -> str:
    """Map a pending-task scope to a Firestore-safe document ID."""
    if not scope_id or scope_id == "latest":
        return "latest"
    return _safe_doc_id(scope_id)


def get_db():
    global _db
    if _db is None:
        _db = firestore.Client(project=config.GCP_PROJECT_ID, database=config.FIRESTORE_DATABASE)
    return _db


def get_conversation(scope_id):
    """Get conversation history for a scope. Cached for 60s to avoid
    redundant Firestore reads during rapid back-and-forth exchanges."""
    cache_key = _safe_doc_id(scope_id)
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


def add_turn(scope_id, role, content):
    """Add a conversation turn and trim to max length."""
    db = get_db()
    cache_key = _safe_doc_id(scope_id)
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


def clear_conversation(scope_id):
    """Clear conversation history for a scope."""
    db = get_db()
    cache_key = _safe_doc_id(scope_id)
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


def store_pending_task_actions(actions, scope_id="latest", meeting_title="", approval_message=""):
    """Store pending task mutations awaiting explicit approval.

    Each action is a dict with an ``action`` key (create/update/complete/delete)
    plus the fields needed to execute that mutation. Only one pending action set
    is kept per scope; newer requests replace older pending ones.
    """
    db = get_db()
    db.collection(config.FIRESTORE_PENDING_TASKS_COLLECTION).document(
        _pending_task_doc_id(scope_id)
    ).set(
        {
            "actions": actions,
            "meeting_title": meeting_title,
            "approval_message": approval_message,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def store_pending_task_actions_if_empty(actions, scope_id="latest", meeting_title="", approval_message=""):
    """Create a pending task request only if the scope has no live request."""
    db = get_db()
    doc_ref = db.collection(config.FIRESTORE_PENDING_TASKS_COLLECTION).document(
        _pending_task_doc_id(scope_id)
    )
    payload = {
        "actions": actions,
        "meeting_title": meeting_title,
        "approval_message": approval_message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        doc_ref.create(payload)
        return True
    except Exception:
        existing = get_pending_task_actions(scope_id=scope_id)
        if existing:
            return False
        doc_ref.set(payload)
        return True


def get_pending_task_actions(scope_id="latest"):
    """Retrieve pending task actions for a scope.

    Returns a dict with ``actions``, ``meeting_title``, and ``approval_message``,
    or ``None`` if nothing is pending or the request expired.
    """
    db = get_db()
    doc = db.collection(config.FIRESTORE_PENDING_TASKS_COLLECTION).document(
        _pending_task_doc_id(scope_id)
    ).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    created = data.get("created_at", "")
    if created:
        try:
            created_dt = datetime.fromisoformat(created)
            age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
            if age_hours > 24:
                db.collection(config.FIRESTORE_PENDING_TASKS_COLLECTION).document(
                    _pending_task_doc_id(scope_id)
                ).delete()
                return None
        except (ValueError, TypeError):
            pass

    actions = data.get("actions")
    if actions is None:
        # Backward compatibility for older pending create-only proposals.
        actions = data.get("tasks", [])

    normalized_actions = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if "action" not in action:
            normalized_actions.append({"action": "create", **action})
        else:
            normalized_actions.append(action)

    if not normalized_actions:
        return None

    return {
        "actions": normalized_actions,
        "meeting_title": data.get("meeting_title", ""),
        "approval_message": data.get("approval_message", ""),
    }


def clear_pending_task_actions(scope_id="latest"):
    """Remove pending task actions after they've been approved or canceled."""
    db = get_db()
    db.collection(config.FIRESTORE_PENDING_TASKS_COLLECTION).document(
        _pending_task_doc_id(scope_id)
    ).delete()


def store_pending_tasks(tasks, meeting_title="", scope_id="latest"):
    """Store proposed create-task suggestions for user confirmation later."""
    actions = []
    for task in tasks:
        action = {"action": "create", "title": task["title"]}
        if task.get("due"):
            action["due"] = task["due"]
        if task.get("notes"):
            action["notes"] = task["notes"]
        actions.append(action)
    store_pending_task_actions(actions, scope_id=scope_id, meeting_title=meeting_title)


def get_pending_tasks(scope_id="latest"):
    """Backward-compatible wrapper for create-task proposals."""
    pending = get_pending_task_actions(scope_id=scope_id)
    if not pending:
        return [], ""

    tasks = []
    for action in pending["actions"]:
        if action.get("action", "create") != "create":
            continue
        task = {"title": action["title"]}
        if action.get("due"):
            task["due"] = action["due"]
        if action.get("notes"):
            task["notes"] = action["notes"]
        tasks.append(task)
    return tasks, pending.get("meeting_title", "")


def clear_pending_tasks(scope_id="latest"):
    """Backward-compatible wrapper for pending create-task proposals."""
    clear_pending_task_actions(scope_id=scope_id)
