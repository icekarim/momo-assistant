"""Persistent user memory for corrections, preferences, and facts.

Stores per-user memories in Firestore so Momo can learn from corrections
and remember preferences across conversations.
"""

import json
import threading
from datetime import datetime, timezone

from cachetools import TTLCache

from google.cloud.firestore_v1.base_query import FieldFilter

import config
from conversation_store import get_db
from langsmith_config import traceable


_memory_cache = TTLCache(maxsize=64, ttl=120)   # 2-min TTL, keyed by user_id
_cache_lock = threading.Lock()

_VALID_TYPES = {"correction", "preference", "fact"}


def _safe_key(user_id: str) -> str:
    return user_id.replace("/", "_")


def _collection():
    return get_db().collection(config.FIRESTORE_USER_MEMORY_COLLECTION)


# ── Read ────────────────────────────────────────────────────


@traceable(run_type="tool", name="get-user-memories")
def get_user_memories(user_id: str) -> list[dict]:
    """Load all active memories for a user. Cached for 2 minutes."""
    key = _safe_key(user_id)
    with _cache_lock:
        if key in _memory_cache:
            return _memory_cache[key]

    docs = (
        _collection()
        .where(filter=FieldFilter("user_id", "==", user_id))
        .where(filter=FieldFilter("active", "==", True))
        .order_by("created_at")
        .stream()
    )
    memories = []
    for doc in docs:
        d = doc.to_dict()
        memories.append({
            "id": doc.id,
            "memory_type": d.get("memory_type", "preference"),
            "content": d.get("content", ""),
            "created_at": d.get("created_at", ""),
        })

    with _cache_lock:
        _memory_cache[key] = memories
    return memories


# ── Write ───────────────────────────────────────────────────


@traceable(run_type="tool", name="add-user-memory")
def add_memory(
    user_id: str,
    content: str,
    memory_type: str = "preference",
    source_message: str = "",
) -> dict:
    """Store a new memory. Deduplicates via substring check.

    Enforces USER_MEMORY_MAX_PER_USER — soft-deletes oldest if at cap.
    Returns the created (or existing duplicate) memory dict.
    """
    if memory_type not in _VALID_TYPES:
        memory_type = "preference"

    content = content.strip()
    if not content:
        return {"status": "error", "message": "Empty content"}

    # ── Dedup: skip if a very similar memory already exists ──
    existing = get_user_memories(user_id)
    content_lower = content.lower()
    for mem in existing:
        existing_lower = mem["content"].lower()
        if content_lower in existing_lower or existing_lower in content_lower:
            return {"status": "duplicate", "content": mem["content"]}

    # ── Cap enforcement: soft-delete oldest if at limit ──
    if len(existing) >= config.USER_MEMORY_MAX_PER_USER:
        oldest = existing[0]  # ordered by created_at asc
        _collection().document(oldest["id"]).update({"active": False})
        print(f"[user_memory] cap reached for {user_id}, deactivated oldest: {oldest['id']}")

    now = datetime.now(timezone.utc).isoformat()
    doc_data = {
        "user_id": user_id,
        "memory_type": memory_type,
        "content": content,
        "source_message": source_message,
        "created_at": now,
        "active": True,
    }
    ref = _collection().add(doc_data)
    doc_id = ref[1].id
    print(f"[user_memory] stored memory for {user_id}: {content[:60]}")

    # Invalidate cache
    key = _safe_key(user_id)
    with _cache_lock:
        _memory_cache.pop(key, None)

    return {"status": "stored", "id": doc_id, "content": content}


# ── Delete (soft) ───────────────────────────────────────────


@traceable(run_type="tool", name="remove-user-memory")
def remove_memory(user_id: str, content_hint: str) -> dict | None:
    """Soft-delete the memory best matching content_hint.

    Uses Gemini Flash for fuzzy matching when there are multiple memories.
    Returns the deactivated memory dict, or None if no match.
    """
    memories = get_user_memories(user_id)
    if not memories:
        return None

    match = _find_best_match(memories, content_hint)
    if match is None:
        return None

    _collection().document(match["id"]).update({"active": False})
    print(f"[user_memory] forgot memory for {user_id}: {match['content'][:60]}")

    # Invalidate cache
    key = _safe_key(user_id)
    with _cache_lock:
        _memory_cache.pop(key, None)

    return {"status": "forgotten", "content": match["content"]}


def _find_best_match(memories: list[dict], hint: str) -> dict | None:
    """Find the memory that best matches the hint.

    For 1 memory, returns it directly. For multiple, uses Gemini Flash.
    """
    if len(memories) == 1:
        return memories[0]

    # Try simple substring match first
    hint_lower = hint.lower()
    for mem in memories:
        if hint_lower in mem["content"].lower() or mem["content"].lower() in hint_lower:
            return mem

    # Fall back to Gemini Flash for fuzzy matching
    try:
        import google.genai as genai
        model = genai.GenerativeModel(model_name=config.GEMINI_MODEL_FLASH)
        numbered = "\n".join(f"{i+1}. {m['content']}" for i, m in enumerate(memories))
        prompt = (
            f"The user wants to forget a memory. Their hint: \"{hint}\"\n\n"
            f"Which of these memories is the best match? Reply with ONLY the number.\n\n"
            f"{numbered}"
        )
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        # Extract first number from response
        digits = "".join(c for c in text if c.isdigit())
        if digits:
            idx = int(digits) - 1
            if 0 <= idx < len(memories):
                return memories[idx]
    except Exception as exc:
        print(f"[user_memory] Gemini fuzzy match failed: {exc}")

    return None


# ── Formatting ──────────────────────────────────────────────


def format_memories_for_context(memories: list[dict]) -> str:
    """Format memories into a text block for injection into conversation history.

    Returns empty string if no memories.
    """
    if not memories:
        return ""

    lines = ["[USER MEMORIES]"]
    for mem in memories:
        prefix = f"[{mem['memory_type']}]" if mem.get("memory_type") else ""
        lines.append(f"- {prefix} {mem['content']}")
    lines.append("[END USER MEMORIES]")
    return "\n".join(lines)
