"""Cross-Meeting Intelligence Graph — entity extraction, storage, and querying.

Extracts people, projects, decisions, commitments, blockers, and topics from
meetings, emails, calendar events, tasks, and Granola notes, stores them in
Firestore, and provides query functions for surfacing institutional memory in
Momo's chat responses.
"""

import json
import threading
import traceback
from datetime import datetime, timezone

import google.generativeai as genai
from cachetools import TTLCache
from google.cloud.firestore_v1.base_query import FieldFilter

import config
from conversation_store import get_db

_kg_cache = TTLCache(maxsize=128, ttl=300)
_kg_cache_lock = threading.Lock()

_embedding_cache = TTLCache(maxsize=1, ttl=300)
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
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

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
            "source_date": source_date,
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
    """All knowledge nodes mentioning a person (array-contains on related_people)."""
    cache_key = ("person", name, since, limit)
    with _kg_cache_lock:
        cached = _kg_cache.get(cache_key)
    if cached is not None:
        return cached

    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where(filter=FieldFilter("related_people", "array_contains", name))
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    if since:
        query = query.where(filter=FieldFilter("source_date", ">=", since))
    results = [_doc_to_dict(doc) for doc in query.stream()]
    with _kg_cache_lock:
        _kg_cache[cache_key] = results
    return results


def query_by_project(name: str, since: str | None = None,
                     limit: int = 50) -> list[dict]:
    """All knowledge nodes related to a project/client."""
    cache_key = ("project", name, since, limit)
    with _kg_cache_lock:
        cached = _kg_cache.get(cache_key)
    if cached is not None:
        return cached

    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where(filter=FieldFilter("related_projects", "array_contains", name))
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    if since:
        query = query.where(filter=FieldFilter("source_date", ">=", since))
    results = [_doc_to_dict(doc) for doc in query.stream()]
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

    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where(filter=FieldFilter("entity_type", "in", ["commitment", "action_item"]))
        .where(filter=FieldFilter("status", "==", "open"))
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    if since:
        query = query.where(filter=FieldFilter("source_date", ">=", since))
    results = [_doc_to_dict(doc) for doc in query.stream()]
    with _kg_cache_lock:
        _kg_cache[cache_key] = results
    return results


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
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where(filter=FieldFilter("source_date", ">=", since))
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    return [_doc_to_dict(doc) for doc in query.stream()]


def query_open_by_age(min_days: int = 3, limit: int = 50) -> list[dict]:
    """Open commitments/action items older than min_days."""
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=min_days)).strftime("%Y-%m-%d")
    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where(filter=FieldFilter("entity_type", "in", ["commitment", "action_item"]))
        .where(filter=FieldFilter("status", "==", "open"))
        .order_by("source_date", direction="ASCENDING")
        .limit(limit)
    )
    results = []
    for doc in query.stream():
        d = _doc_to_dict(doc)
        if d.get("source_date", "9999") <= cutoff:
            results.append(d)
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


def _get_all_embeddings() -> list[dict]:
    """Cached bulk read of all KG entities with embeddings from Firestore."""
    with _embedding_cache_lock:
        cached = _embedding_cache.get("all")
    if cached is not None:
        return cached

    db = get_db()
    docs = db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).stream()
    entities = [_doc_to_dict(doc) for doc in docs]

    with _embedding_cache_lock:
        _embedding_cache["all"] = entities
    return entities


def semantic_search(query: str, limit: int | None = None,
                    threshold: float | None = None) -> list[dict]:
    """Search the knowledge graph using vector similarity.

    Embeds the query, compares against all stored entity embeddings,
    and returns the top matches above the similarity threshold.
    """
    limit = limit or config.SEMANTIC_SEARCH_LIMIT
    threshold = threshold if threshold is not None else config.SEMANTIC_SEARCH_THRESHOLD

    try:
        query_embedding = _get_embedding(query, task_type="RETRIEVAL_QUERY")
    except Exception as exc:
        print(f"Knowledge graph: query embedding failed ({exc}), falling back to empty")
        return []

    entities = _get_all_embeddings()

    scored = []
    for entity in entities:
        if "embedding" not in entity:
            continue
        score = _cosine_similarity(query_embedding, entity["embedding"])
        if score >= threshold:
            scored.append((score, entity))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entity for _score, entity in scored[:limit]]


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
