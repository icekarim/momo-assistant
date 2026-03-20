"""Cross-Meeting Intelligence Graph — entity extraction, storage, and querying.

Extracts people, projects, decisions, commitments, blockers, and topics from
meetings, emails, calendar events, tasks, and Granola notes, stores them in
Firestore, and provides query functions for surfacing institutional memory in
Momo's chat responses.
"""

import json
import re
import threading
import traceback
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import google.generativeai as genai
from cachetools import TTLCache
from google.cloud.firestore_v1.base_query import FieldFilter

import config
from conversation_store import get_db

_kg_cache = TTLCache(maxsize=128, ttl=300)
_kg_cache_lock = threading.Lock()

_embedding_cache = TTLCache(maxsize=1, ttl=1800)  # 30 min — embeddings rarely change
_embedding_cache_lock = threading.Lock()

genai.configure(api_key=config.GEMINI_API_KEY)


# ── Embedding helpers ────────────────────────────────────────


def _build_embedding_text(entry: dict, source_type: str = "") -> str:
    """Build a combined text string from an entity for embedding generation."""
    parts = [
        entry.get("entity_type", ""),
        entry.get("name", ""),
        entry.get("content", ""),
    ]
    if entry.get("owner"):
        parts.append(f"owner: {entry['owner']}")
    if entry.get("related_people"):
        parts.append(f"people: {', '.join(entry['related_people'])}")
    if entry.get("related_projects"):
        parts.append(f"projects: {', '.join(entry['related_projects'])}")
    if source_type:
        parts.append(f"source: {source_type}")
    return " | ".join(p for p in parts if p)


def _get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generate an embedding vector for a text string using Gemini."""
    result = genai.embed_content(
        model=config.GEMINI_EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (fallback, prefer _batch_cosine)."""
    import numpy as np
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


_EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")


def _parse_source_date(value: str | datetime | None) -> datetime | None:
    """Parse a variety of source date formats into a comparable UTC-naive date."""
    if not value:
        return None

    parsed = value if isinstance(value, datetime) else None
    text = "" if isinstance(value, datetime) else str(value).strip()

    if parsed is None and text:
        for candidate in (text.replace("Z", "+00:00"), text[:10]):
            try:
                parsed = datetime.fromisoformat(candidate)
                break
            except ValueError:
                continue

    if parsed is None and text:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            parsed = None

    if parsed is None and text:
        for fmt in ("%Y/%m/%d", "%b %d, %Y", "%b %d, %I:%M %p"):
            try:
                parsed = datetime.strptime(text, fmt)
                if fmt == "%b %d, %I:%M %p":
                    parsed = parsed.replace(year=datetime.now().year)
                break
            except ValueError:
                continue

    if parsed is None:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)


def _normalize_source_date(value: str | datetime | None) -> str:
    """Store source dates in YYYY-MM-DD whenever possible."""
    parsed = _parse_source_date(value)
    if parsed:
        return parsed.strftime("%Y-%m-%d")
    return str(value or "").strip()


def _entry_date_key(entry: dict) -> datetime:
    parsed = _parse_source_date(entry.get("source_date"))
    return parsed or datetime.min


def _normalize_text(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _person_tokens(value: str | None) -> list[str]:
    text = (value or "").strip().lower()
    if not text:
        return []

    tokens = []
    for email in _EMAIL_RE.findall(text):
        tokens.extend(re.findall(r"[a-z0-9]+", email.split("@", 1)[0]))

    text = _EMAIL_RE.sub(" ", text)
    text = re.sub(r"[<>]", " ", text)
    tokens.extend(re.findall(r"[a-z0-9]+", text))
    return [token for token in tokens if token]


def _person_matches(query: str, candidate: str) -> bool:
    q_tokens = _person_tokens(query)
    c_tokens = _person_tokens(candidate)
    if not q_tokens or not c_tokens:
        return False
    if q_tokens == c_tokens:
        return True

    q_set = set(q_tokens)
    c_set = set(c_tokens)

    if len(q_tokens) >= 2 and len(c_tokens) >= 2:
        if q_tokens[0] == c_tokens[0] and q_tokens[-1] == c_tokens[-1]:
            return True
        if q_set.issubset(c_set) or c_set.issubset(q_set):
            return True

    if len(q_tokens) == 1:
        return q_tokens[0] in c_set
    if len(c_tokens) == 1:
        return c_tokens[0] in q_set
    return False


def _project_matches(query: str, candidate: str) -> bool:
    q = _normalize_text(query)
    c = _normalize_text(candidate)
    if not q or not c:
        return False
    if q == c:
        return True
    if min(len(q), len(c)) < 5:
        return False
    return q in c or c in q


def _load_all_entries() -> list[dict]:
    cache_key = ("all_entries",)
    with _kg_cache_lock:
        cached = _kg_cache.get(cache_key)
    if cached is not None:
        return cached

    db = get_db()
    entries = [_doc_to_dict(doc) for doc in db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).stream()]
    with _kg_cache_lock:
        _kg_cache[cache_key] = entries
    return entries

# ── Extraction prompt ────────────────────────────────────────
# Literal braces in the JSON example are doubled ({{ }}) so
# str.format() doesn't treat them as placeholders.

_EXTRACTION_PROMPT = """You are an entity extraction engine. Extract structured knowledge from the content below.

Return a JSON array of objects. Each object represents one piece of knowledge:

{{
  "entity_type": "decision" | "commitment" | "action_item" | "blocker" | "topic" | "update",
  "name": "short label (2-8 words)",
  "content": "1-2 sentence description of what was said/decided/committed",
  "status": "open" | "completed" | "resolved" | null,
  "owner": "person responsible (if known, else null)",
  "related_people": ["list of people mentioned or involved"],
  "related_projects": ["list of projects, clients, or initiatives mentioned"],
  "tags": ["lowercase keywords for search, 3-6 tags"]
}}

Entity type guidelines:
- "decision": something that was decided or agreed upon
- "commitment": something someone promised to do (action item with an owner)
- "action_item": a task that needs to happen (may or may not have an owner)
- "blocker": something blocking progress
- "topic": a subject that was discussed without a clear decision/action
- "update": a status update or progress report on something

Rules:
- Extract ONLY what is explicitly stated. Do not infer or fabricate.
- Be specific in "content" — include names, dates, numbers when present.
- "related_people" should include first names or full names as they appear.
- "related_projects" should include client names, product names, initiative names.
- "tags" should be lowercase, no spaces, useful for search (e.g. "pricing", "q2", "launch").
- If no meaningful knowledge can be extracted, return an empty array: []
- Return ONLY the JSON array, no markdown fences, no explanation.

SOURCE TYPE: {source_type}
SOURCE TITLE: {source_title}
ATTENDEES: {attendees}

CONTENT:
{content}
"""


# ── Extraction & storage ─────────────────────────────────────


def _already_extracted(source_id: str) -> bool:
    """Check if we've already extracted from this source (idempotency)."""
    try:
        db = get_db()
        docs = list(
            db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
            .where(filter=FieldFilter("source_id", "==", source_id))
            .limit(1)
            .stream()
        )
        return len(docs) > 0
    except Exception as exc:
        print(f"Knowledge graph: dedup check failed ({exc}), proceeding with extraction")
        return False


def _run_extraction(source_type: str, source_title: str, content: str,
                    attendees: list[str]) -> list[dict]:
    """Call Gemini Flash to extract structured entities from content."""
    if not content or not content.strip():
        return []

    attendees_str = ", ".join(attendees) if attendees else "unknown"
    prompt = _EXTRACTION_PROMPT.format(
        source_type=source_type,
        source_title=source_title,
        attendees=attendees_str,
        content=content[:8000],
    )

    model = genai.GenerativeModel(model_name=config.GEMINI_MODEL_FLASH)

    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        parsed = json.loads(text.strip())

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            print(f"Knowledge graph: unexpected response type {type(parsed)}, skipping")
            return []

        return [e for e in parsed if isinstance(e, dict)]
    except Exception as exc:
        print(f"Knowledge graph extraction failed: {exc}")
        return []


def _store_entries(entries: list[dict], source_type: str, source_id: str,
                   source_title: str, source_date: str):
    """Persist extracted entries to Firestore with embeddings and invalidate caches."""
    db = get_db()
    collection = db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
    now_iso = datetime.now(timezone.utc).isoformat()

    for entry in entries:
        doc = {
            "entity_type": entry.get("entity_type", "topic"),
            "name": entry.get("name", ""),
            "content": entry.get("content", ""),
            "status": entry.get("status"),
            "owner": entry.get("owner"),
            "related_people": entry.get("related_people", []),
            "related_projects": entry.get("related_projects", []),
            "tags": entry.get("tags", []),
            "source_type": source_type,
            "source_id": source_id,
            "source_title": source_title,
            "source_date": _normalize_source_date(source_date),
            "extracted_at": now_iso,
        }
        try:
            text = _build_embedding_text(entry, source_type=source_type)
            doc["embedding"] = _get_embedding(text)
        except Exception as exc:
            print(f"  Knowledge graph: embedding generation failed ({exc}), storing without")
        collection.add(doc)

    with _kg_cache_lock:
        _kg_cache.clear()
    with _embedding_cache_lock:
        _embedding_cache.clear()


def extract_and_store(source_type: str, source_id: str, source_title: str,
                      source_date: str, content: str,
                      attendees: list[str] | None = None):
    """Extract entities from content and store in the knowledge graph.

    Idempotent — skips if source_id was already processed.
    """
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return

    if _already_extracted(source_id):
        print(f"  Knowledge graph: already extracted for {source_id}, skipping")
        return

    entries = _run_extraction(source_type, source_title, content, attendees or [])

    if entries:
        _store_entries(entries, source_type, source_id, source_title, source_date)
        print(f"  Knowledge graph: stored {len(entries)} entries from '{source_title}'")
    else:
        print(f"  Knowledge graph: no entities extracted from '{source_title}'")


def extract_and_store_background(source_type: str, source_id: str,
                                 source_title: str, source_date: str,
                                 content: str,
                                 attendees: list[str] | None = None):
    """Fire-and-forget wrapper — runs extraction in a daemon thread."""
    thread = threading.Thread(
        target=_safe_extract,
        args=(source_type, source_id, source_title, source_date, content, attendees),
        daemon=True,
    )
    thread.start()


def _safe_extract(source_type, source_id, source_title, source_date, content, attendees):
    try:
        extract_and_store(source_type, source_id, source_title, source_date,
                          content, attendees)
    except Exception:
        print("  Knowledge graph background extraction failed:")
        traceback.print_exc()


# ── Source-specific extraction helpers ───────────────────────


def extract_from_calendar_events(events: list[dict]):
    """Extract knowledge from calendar events (background, per-event).

    Skips all-day events (holidays, OOO). Uses the Google Calendar event ID
    as source_id so dedup prevents re-extraction of the same event.
    """
    if not config.KNOWLEDGE_GRAPH_ENABLED or not events:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    for event in events:
        if event.get("is_all_day"):
            continue

        attendees = [a["name"] for a in event.get("attendees", [])]
        parts = [f"Meeting: {event.get('title', '')}"]
        parts.append(f"Time: {event.get('start_time', '')} – {event.get('end_time', '')}")
        if event.get("location"):
            parts.append(f"Location: {event['location']}")
        if attendees:
            parts.append(f"Attendees: {', '.join(attendees)}")
        if event.get("organizer"):
            parts.append(f"Organizer: {event['organizer']}")
        if event.get("description"):
            parts.append(f"Description: {event['description']}")

        extract_and_store_background(
            source_type="calendar",
            source_id=event.get("id", ""),
            source_title=event.get("title", ""),
            source_date=today,
            content="\n".join(parts),
            attendees=attendees,
        )


def extract_from_tasks(tasks: list[dict]):
    """Extract knowledge from the current task list (batched, once per day).

    Tasks are small individually, so we batch them into one extraction call.
    Uses a date-based source_id so it runs once per day.
    """
    if not config.KNOWLEDGE_GRAPH_ENABLED or not tasks:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    source_id = f"tasks-daily-{today}"

    lines = []
    for t in tasks:
        line = f"- {t['title']}"
        if t.get("due"):
            overdue = " (OVERDUE)" if t.get("is_overdue") else ""
            line += f" [due: {t['due']}{overdue}]"
        if t.get("notes"):
            line += f" — {t['notes']}"
        if t.get("list_name"):
            line += f" (list: {t['list_name']})"
        lines.append(line)

    content = f"Open tasks as of {today}:\n" + "\n".join(lines)

    extract_and_store_background(
        source_type="tasks",
        source_id=source_id,
        source_title=f"Task snapshot {today}",
        source_date=today,
        content=content,
        attendees=[],
    )


def extract_from_granola_notes(granola_context: str, source_date: str | None = None):
    """Extract knowledge from Granola meeting notes context.

    Used by the morning briefing to capture yesterday's notes that may not
    have been caught by the post-meeting debrief pipeline.
    """
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return
    if not granola_context or not granola_context.strip():
        return

    date = source_date or datetime.now().strftime("%Y-%m-%d")
    source_id = f"granola-briefing-{date}"

    extract_and_store_background(
        source_type="meeting_notes",
        source_id=source_id,
        source_title=f"Granola meeting notes ({date})",
        source_date=date,
        content=granola_context,
        attendees=[],
    )


# ── Query functions ──────────────────────────────────────────


def query_by_person(name: str, since: str | None = None,
                    limit: int = 50) -> list[dict]:
    """All knowledge nodes mentioning a person, with fuzzy name/email matching."""
    cache_key = ("person", _normalize_text(name), since, limit)
    with _kg_cache_lock:
        cached = _kg_cache.get(cache_key)
    if cached is not None:
        return cached

    since_dt = _parse_source_date(since) if since else None
    results = []
    for entry in sorted(_load_all_entries(), key=_entry_date_key, reverse=True):
        if since_dt:
            entry_dt = _parse_source_date(entry.get("source_date"))
            if not entry_dt or entry_dt < since_dt:
                continue

        people = entry.get("related_people", [])
        owner = entry.get("owner") or ""
        if any(_person_matches(name, person) for person in people) or _person_matches(name, owner):
            results.append(entry)
            if len(results) >= limit:
                break

    with _kg_cache_lock:
        _kg_cache[cache_key] = results
    return results


def query_by_project(name: str, since: str | None = None,
                     limit: int = 50) -> list[dict]:
    """All knowledge nodes related to a project/client, case-insensitive."""
    cache_key = ("project", _normalize_text(name), since, limit)
    with _kg_cache_lock:
        cached = _kg_cache.get(cache_key)
    if cached is not None:
        return cached

    since_dt = _parse_source_date(since) if since else None
    results = []
    for entry in sorted(_load_all_entries(), key=_entry_date_key, reverse=True):
        if since_dt:
            entry_dt = _parse_source_date(entry.get("source_date"))
            if not entry_dt or entry_dt < since_dt:
                continue

        if any(_project_matches(name, project) for project in entry.get("related_projects", [])):
            results.append(entry)
            if len(results) >= limit:
                break

    with _kg_cache_lock:
        _kg_cache[cache_key] = results
    return results


def query_by_type(entity_type: str, since: str | None = None,
                  limit: int = 50) -> list[dict]:
    """All nodes of a given type (decision, commitment, blocker, etc.)."""
    cache_key = ("type", entity_type, since, limit)
    with _kg_cache_lock:
        cached = _kg_cache.get(cache_key)
    if cached is not None:
        return cached

    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where(filter=FieldFilter("entity_type", "==", entity_type))
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    if since:
        query = query.where(filter=FieldFilter("source_date", ">=", since))
    results = [_doc_to_dict(doc) for doc in query.stream()]
    with _kg_cache_lock:
        _kg_cache[cache_key] = results
    return results


def query_open_commitments(since: str | None = None,
                           limit: int = 50) -> list[dict]:
    """Commitments and action items that haven't been completed."""
    cache_key = ("open_commitments", since, limit)
    with _kg_cache_lock:
        cached = _kg_cache.get(cache_key)
    if cached is not None:
        return cached

    since_dt = _parse_source_date(since) if since else None
    results = []
    for entry in sorted(_load_all_entries(), key=_entry_date_key, reverse=True):
        if entry.get("entity_type") not in ("commitment", "action_item"):
            continue
        if entry.get("status") != "open":
            continue

        if since_dt:
            entry_dt = _parse_source_date(entry.get("source_date"))
            if not entry_dt or entry_dt < since_dt:
                continue

        results.append(entry)
        if len(results) >= limit:
            break

    with _kg_cache_lock:
        _kg_cache[cache_key] = results
    return results


def query_all_entries(limit: int | None = None) -> list[dict]:
    """Return all KG entries sorted by source date descending."""
    entries = sorted(_load_all_entries(), key=_entry_date_key, reverse=True)
    if limit is None:
        return entries
    return entries[:limit]


def search_knowledge(keywords: list[str], limit: int = 50) -> list[dict]:
    """Search by tags. Firestore array-contains only supports a single value,
    so we query the first keyword and filter the rest in-memory."""
    if not keywords:
        return []

    cache_key = ("search", tuple(keywords), limit)
    with _kg_cache_lock:
        cached = _kg_cache.get(cache_key)
    if cached is not None:
        return cached

    db = get_db()
    primary = keywords[0].lower().strip()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where(filter=FieldFilter("tags", "array_contains", primary))
        .order_by("source_date", direction="DESCENDING")
        .limit(limit * 2)
    )
    results = [_doc_to_dict(doc) for doc in query.stream()]

    if len(keywords) > 1:
        extra = set(k.lower().strip() for k in keywords[1:])
        results = [
            r for r in results
            if extra.intersection(set(r.get("tags", [])))
        ]

    results = results[:limit]
    with _kg_cache_lock:
        _kg_cache[cache_key] = results
    return results


def query_recent(days: int = 7, limit: int = 50) -> list[dict]:
    """All entries from the last N days."""
    from datetime import timedelta
    since_dt = (datetime.now() - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    results = []
    for entry in sorted(_load_all_entries(), key=_entry_date_key, reverse=True):
        entry_dt = _parse_source_date(entry.get("source_date"))
        if not entry_dt or entry_dt < since_dt:
            continue
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def query_open_by_age(min_days: int = 3, limit: int = 50) -> list[dict]:
    """Open commitments/action items older than min_days."""
    from datetime import timedelta
    cutoff_dt = (datetime.now() - timedelta(days=min_days)).replace(hour=0, minute=0, second=0, microsecond=0)
    results = []
    for entry in sorted(_load_all_entries(), key=_entry_date_key):
        if entry.get("entity_type") not in ("commitment", "action_item"):
            continue
        if entry.get("status") != "open":
            continue

        entry_dt = _parse_source_date(entry.get("source_date"))
        if not entry_dt or entry_dt > cutoff_dt:
            continue

        results.append(entry)
        if len(results) >= limit:
            break
    return results


def update_entity_status(doc_id: str, new_status: str):
    """Update the status of a knowledge graph entity (e.g. open -> resolved)."""
    db = get_db()
    doc_ref = db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).document(doc_id)
    doc_ref.update({
        "status": new_status,
        "status_updated_at": datetime.now(timezone.utc).isoformat(),
    })
    with _kg_cache_lock:
        _kg_cache.clear()
    with _embedding_cache_lock:
        _embedding_cache.clear()


# ── Semantic search ──────────────────────────────────────────


def _get_all_embeddings() -> tuple[list[dict], "np.ndarray | None"]:
    """Cached bulk read of all KG entities + pre-computed embedding matrix.

    Returns (entities, matrix) where matrix is a numpy array of shape
    (N, embedding_dim) for entities that have embeddings. Cached for 30 min.
    """
    import numpy as np

    with _embedding_cache_lock:
        cached = _embedding_cache.get("all")
    if cached is not None:
        return cached

    t0 = __import__("time").time()
    db = get_db()
    docs = db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).stream()
    entities = [_doc_to_dict(doc) for doc in docs]

    # Pre-compute numpy matrix for vectorized cosine similarity
    indexed_entities = []
    vectors = []
    for e in entities:
        emb = e.get("embedding")
        if emb:
            indexed_entities.append(e)
            vectors.append(emb)

    matrix = np.array(vectors, dtype=np.float32) if vectors else None

    # Pre-compute norms for fast cosine similarity
    if matrix is not None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        matrix = matrix / norms  # normalize once, reuse on every query

    result = (indexed_entities, matrix)
    with _embedding_cache_lock:
        _embedding_cache["all"] = result

    elapsed = __import__("time").time() - t0
    print(f"Knowledge graph: loaded {len(indexed_entities)} embeddings in {elapsed:.2f}s")
    return result


def warm_embedding_cache():
    """Pre-load the embedding cache. Call on startup to avoid cold-start latency."""
    try:
        _get_all_embeddings()
    except Exception as exc:
        print(f"Knowledge graph: embedding cache warmup failed: {exc}")


def semantic_search(query: str, limit: int | None = None,
                    threshold: float | None = None) -> list[dict]:
    """Search the knowledge graph using vectorized cosine similarity.

    Embeds the query, compares against the pre-computed normalized embedding
    matrix in a single numpy operation, and returns the top matches.
    """
    import numpy as np

    limit = limit or config.SEMANTIC_SEARCH_LIMIT
    threshold = threshold if threshold is not None else config.SEMANTIC_SEARCH_THRESHOLD

    try:
        query_embedding = _get_embedding(query, task_type="RETRIEVAL_QUERY")
    except Exception as exc:
        print(f"Knowledge graph: query embedding failed ({exc}), falling back to empty")
        return []

    indexed_entities, matrix = _get_all_embeddings()

    if matrix is None or len(indexed_entities) == 0:
        return []

    # Vectorized cosine similarity: one matrix-vector dot product
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    q = q / q_norm

    scores = matrix @ q  # (N,) — all similarities in one operation

    # Filter and sort
    mask = scores >= threshold
    if not mask.any():
        return []

    indices = np.where(mask)[0]
    top_indices = indices[np.argsort(scores[indices])[::-1][:limit]]

    return [indexed_entities[i] for i in top_indices]


def embed_backfill() -> dict:
    """Add embeddings to existing KG entities that don't have one.

    Returns counts of updated and skipped entities.
    """
    import time as _time

    db = get_db()
    docs = list(db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).stream())

    updated, skipped, failed = 0, 0, 0
    for doc in docs:
        data = doc.to_dict()
        if "embedding" in data:
            skipped += 1
            continue

        try:
            text = _build_embedding_text(data, source_type=data.get("source_type", ""))
            embedding = _get_embedding(text)
            doc.reference.update({"embedding": embedding})
            updated += 1
        except Exception as exc:
            print(f"  Embed backfill failed for {doc.id}: {exc}")
            failed += 1
        _time.sleep(0.1)

    with _embedding_cache_lock:
        _embedding_cache.clear()

    print(f"Embed backfill: {updated} updated, {skipped} already had embeddings, {failed} failed")
    return {"updated": updated, "skipped": skipped, "failed": failed}


def _doc_to_dict(doc) -> dict:
    d = doc.to_dict()
    d["id"] = doc.id
    d["source_date"] = _normalize_source_date(d.get("source_date"))
    return d


# ── Context formatting ───────────────────────────────────────


def format_knowledge_for_context(entries: list[dict]) -> str:
    """Format knowledge graph entries into a text block for Gemini context."""
    if not entries:
        return ""

    entries = entries[:15]
    lines = []
    for e in entries:
        owner = f" (owner: {e['owner']})" if e.get("owner") else ""
        status = f" [{e['status']}]" if e.get("status") else ""
        people = ", ".join(e.get("related_people", []))
        projects = ", ".join(e.get("related_projects", []))
        refs = []
        if people:
            refs.append(f"people: {people}")
        if projects:
            refs.append(f"projects: {projects}")
        ref_str = f" | {'; '.join(refs)}" if refs else ""

        lines.append(
            f"- [{e.get('source_date', '?')}] [{e.get('entity_type', '?')}]{status}{owner} "
            f"{e.get('name', '')}: {e.get('content', '')}"
            f" (source: {e.get('source_title', '?')}{ref_str})"
        )

    return "\n".join(lines)


# ── Smart query dispatcher ───────────────────────────────────


def query_knowledge_graph(user_message: str) -> str:
    """Given a user message, find relevant knowledge using semantic search.

    Embeds the user message, compares against all stored entity embeddings,
    and returns the top matches formatted for Gemini context.
    """
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return ""

    results = semantic_search(user_message)

    if not results:
        return ""

    return format_knowledge_for_context(results)
