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


# ── Inbound-message idempotency guard ─────────────────────────────────────────
# Google Chat retries the webhook when the synchronous agent loop exceeds its 30s
# deadline; the retry would otherwise re-run the whole loop and duplicate its
# side effects (store_task_batch / create_task). A claim doc keyed by the Chat
# message_name makes processing exactly-once. Built on the same create-if-absent
# pattern as store_pending_task_actions_if_empty.


def claim_message_once(message_name: str, ttl_hours: int = 1) -> bool:
    """Atomically claim an inbound message so it is processed exactly once.

    Returns True if the claim is NEW (caller should process), or False if a live
    claim already exists (a duplicate/retry — caller should no-op). A claim older
    than ``ttl_hours`` is lazily reclaimed so a message that crashed before
    release isn't wedged forever.
    """
    db = get_db()
    doc_ref = db.collection(config.FIRESTORE_PROCESSED_MESSAGES_COLLECTION).document(
        _safe_doc_id(message_name)
    )
    payload = {
        "message_name": message_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ttl_hours": ttl_hours,
    }
    try:
        doc_ref.create(payload)
        return True
    except Exception:
        existing = doc_ref.get()
        if existing.exists:
            created = existing.to_dict().get("created_at", "")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
                    if age_hours > ttl_hours:
                        doc_ref.set(payload)
                        return True
                except (ValueError, TypeError):
                    pass
        return False


def release_message_claim(message_name: str) -> None:
    """Delete a message claim so a genuine processing crash can be retried."""
    db = get_db()
    db.collection(config.FIRESTORE_PROCESSED_MESSAGES_COLLECTION).document(
        _safe_doc_id(message_name)
    ).delete()


# ── Task-batch state store ────────────────────────────────────────────────────
# Keyed by batch_id so multiple concurrent batches never overwrite each other.
# Fixes the "vanishing task" bug caused by the single-slot pending model.
#
# Row schema: {"taskId": str, "title": str, "due": str|None, "owner": str|None,
#              "priority": str|None, "state": str}
# state values: "pending" | "added" | "already_exists" | "dismissed"
#
# Doc schema (Firestore): source, space, rows, created_at (ISO UTC), expires_hours.


def store_task_batch(batch_id: str, source: str, space: str, rows: list,
                     expires_hours: int = 24) -> None:
    """Write a new task-batch document keyed by batch_id.

    Each batch_id gets its own Firestore document so concurrent batches
    are fully independent — the document key is the batch_id itself.
    """
    db = get_db()
    db.collection(config.FIRESTORE_TASK_BATCHES_COLLECTION).document(batch_id).set(
        {
            "source": source,
            "space": space,
            "rows": rows,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_hours": expires_hours,
        }
    )


def get_task_batch(batch_id: str) -> dict | None:
    """Retrieve a task-batch by batch_id.

    Returns the batch dict (keys: source, space, rows, created_at) or None
    if the document does not exist.  Lazily deletes and returns None when the
    batch is older than its configured expiry window (mirrors the 24-hour lazy-
    expiry pattern used by get_pending_task_actions).
    """
    db = get_db()
    doc = db.collection(config.FIRESTORE_TASK_BATCHES_COLLECTION).document(batch_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    created = data.get("created_at", "")
    if created:
        try:
            created_dt = datetime.fromisoformat(created)
            expires_hours = data.get("expires_hours", 24)
            age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
            if age_hours > expires_hours:
                db.collection(config.FIRESTORE_TASK_BATCHES_COLLECTION).document(batch_id).delete()
                return None
        except (ValueError, TypeError):
            pass
    return data


def update_task_batch(batch_id: str, rows: list) -> None:
    """Replace the rows array on an existing task-batch document."""
    db = get_db()
    db.collection(config.FIRESTORE_TASK_BATCHES_COLLECTION).document(batch_id).update(
        {"rows": rows}
    )
