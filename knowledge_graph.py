"""Cross-Meeting Intelligence Graph — entity extraction, storage, and querying.

Extracts people, projects, decisions, commitments, blockers, and topics from
meeting debriefs and emails, stores them in Firestore, and provides query
functions for surfacing institutional memory in Momo's chat responses.
"""

import json
import threading
from datetime import datetime, timezone

import google.generativeai as genai

import config
from conversation_store import get_db

genai.configure(api_key=config.GEMINI_API_KEY)

# ── Extraction prompt ────────────────────────────────────────

_EXTRACTION_PROMPT = """You are an entity extraction engine. Extract structured knowledge from the content below.

Return a JSON array of objects. Each object represents one piece of knowledge:

{
  "type": "decision" | "commitment" | "action_item" | "blocker" | "topic" | "update",
  "name": "short label (2-8 words)",
  "content": "1-2 sentence description of what was said/decided/committed",
  "status": "open" | "completed" | "resolved" | null,
  "owner": "person responsible (if known, else null)",
  "related_people": ["list of people mentioned or involved"],
  "related_projects": ["list of projects, clients, or initiatives mentioned"],
  "tags": ["lowercase keywords for search, 3-6 tags"]
}

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
    db = get_db()
    existing = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where("source_id", "==", source_id)
        .limit(1)
        .get()
    )
    return len(existing) > 0


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
        return json.loads(text.strip())
    except Exception as exc:
        print(f"Knowledge graph extraction failed: {exc}")
        return []


def _store_entries(entries: list[dict], source_type: str, source_id: str,
                   source_title: str, source_date: str):
    """Persist extracted entries to Firestore."""
    db = get_db()
    collection = db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
    now_iso = datetime.now(timezone.utc).isoformat()

    for entry in entries:
        doc = {
            "type": entry.get("type", "topic"),
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
        collection.add(doc)


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
    except Exception as exc:
        print(f"  Knowledge graph background extraction failed: {exc}")


# ── Query functions ──────────────────────────────────────────


def query_by_person(name: str, since: str | None = None,
                    limit: int = 50) -> list[dict]:
    """All knowledge nodes mentioning a person (array-contains on related_people)."""
    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where("related_people", "array_contains", name)
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    if since:
        query = query.where("source_date", ">=", since)
    return [_doc_to_dict(doc) for doc in query.stream()]


def query_by_project(name: str, since: str | None = None,
                     limit: int = 50) -> list[dict]:
    """All knowledge nodes related to a project/client."""
    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where("related_projects", "array_contains", name)
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    if since:
        query = query.where("source_date", ">=", since)
    return [_doc_to_dict(doc) for doc in query.stream()]


def query_by_type(entity_type: str, since: str | None = None,
                  limit: int = 50) -> list[dict]:
    """All nodes of a given type (decision, commitment, blocker, etc.)."""
    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where("type", "==", entity_type)
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    if since:
        query = query.where("source_date", ">=", since)
    return [_doc_to_dict(doc) for doc in query.stream()]


def query_open_commitments(since: str | None = None,
                           limit: int = 50) -> list[dict]:
    """Commitments and action items that haven't been completed."""
    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where("type", "in", ["commitment", "action_item"])
        .where("status", "==", "open")
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    if since:
        query = query.where("source_date", ">=", since)
    return [_doc_to_dict(doc) for doc in query.stream()]


def search_knowledge(keywords: list[str], limit: int = 50) -> list[dict]:
    """Search by tags. Firestore array-contains only supports a single value,
    so we query the first keyword and filter the rest in-memory."""
    if not keywords:
        return []

    db = get_db()
    primary = keywords[0].lower().strip()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where("tags", "array_contains", primary)
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

    return results[:limit]


def query_recent(days: int = 7, limit: int = 50) -> list[dict]:
    """All entries from the last N days."""
    from datetime import timedelta
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    db = get_db()
    query = (
        db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION)
        .where("source_date", ">=", since)
        .order_by("source_date", direction="DESCENDING")
        .limit(limit)
    )
    return [_doc_to_dict(doc) for doc in query.stream()]


def _doc_to_dict(doc) -> dict:
    d = doc.to_dict()
    d["id"] = doc.id
    return d


# ── Context formatting ───────────────────────────────────────


def format_knowledge_for_context(entries: list[dict]) -> str:
    """Format knowledge graph entries into a text block for Gemini context."""
    if not entries:
        return ""

    entries = entries[:50]
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
            f"- [{e.get('source_date', '?')}] [{e.get('type', '?')}]{status}{owner} "
            f"{e.get('name', '')}: {e.get('content', '')}"
            f" (source: {e.get('source_title', '?')}{ref_str})"
        )

    return "\n".join(lines)


# ── Smart query dispatcher ───────────────────────────────────


def query_knowledge_graph(user_message: str) -> str:
    """Given a user message, figure out what to query and return formatted context.

    Uses keyword matching and simple NLP to determine the right query strategy,
    then formats results for injection into Gemini context.
    """
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return ""

    lower = user_message.lower()
    results = []

    person_name = _extract_person_name(user_message)
    project_name = _extract_project_name(user_message)

    commitment_keywords = ["commitment", "committed", "promised", "owe", "action item",
                           "haven't done", "outstanding", "follow up", "follow-up"]
    wants_commitments = any(kw in lower for kw in commitment_keywords)

    blocker_keywords = ["blocker", "blocked", "blocking", "stuck", "impediment"]
    wants_blockers = any(kw in lower for kw in blocker_keywords)

    decision_keywords = ["decided", "decision", "agreed", "alignment", "conclusion"]
    wants_decisions = any(kw in lower for kw in decision_keywords)

    if wants_commitments:
        results.extend(query_open_commitments())
    if wants_blockers:
        results.extend(query_by_type("blocker"))
    if wants_decisions:
        results.extend(query_by_type("decision"))

    if person_name:
        results.extend(query_by_person(person_name))
    if project_name:
        results.extend(query_by_project(project_name))

    if not results:
        search_terms = _extract_knowledge_search_terms(user_message)
        if search_terms:
            results = search_knowledge(search_terms)

    if not results:
        results = query_recent(days=14, limit=30)

    seen_ids = set()
    unique = []
    for r in results:
        rid = r.get("id")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            unique.append(r)

    return format_knowledge_for_context(unique)


def _extract_person_name(message: str) -> str | None:
    """Try to extract a person's name from the user message."""
    import re
    lower = message.lower()

    patterns = [
        r"(?:with|from|about|by|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\b",
        r"(?:met|meeting|call|sync|chat)\s+(?:with\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]

    skip_words = {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December", "Today", "Tomorrow",
        "Momo", "Granola", "Gmail", "Google", "Gemini",
    }

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            name = match.group(1).strip()
            if name not in skip_words and len(name) > 1:
                return name
    return None


def _extract_project_name(message: str) -> str | None:
    """Try to extract a project/client name from the user message.

    Looks for known patterns and falls back to capitalized multi-word phrases
    that might be client/project names (e.g. "BJ's", "DSW", "Lowes").
    """
    import re

    patterns = [
        r"(?:history of|about|on|with|for|the)\s+(?:the\s+)?([A-Z][A-Za-z']+(?:\s+[A-Za-z']+){0,3}?)(?:\s+(?:project|deal|account|client|discussion|meeting|topic))",
        r"([A-Z]{2,}(?:'s)?)\b",
    ]

    skip_words = {"AI", "API", "CEO", "CTO", "CFO", "COO", "VP", "PM", "QA",
                  "HR", "IT", "OK", "AM", "PM", "US", "UK", "EU"}

    for pattern in patterns:
        for match in re.finditer(pattern, message):
            name = match.group(1).strip()
            if name not in skip_words and len(name) > 1:
                return name
    return None


def _extract_knowledge_search_terms(message: str) -> list[str]:
    """Extract search keywords from a user message for tag-based search."""
    import re
    stopwords = {
        "what", "whats", "how", "is", "are", "the", "a", "an", "my", "me", "i",
        "from", "with", "them", "as", "well", "going", "on", "about", "any",
        "history", "full", "story", "tell", "show", "give", "last", "time",
        "since", "changed", "happened", "discussed", "decided", "committed",
        "do", "does", "did", "can", "could", "would", "should", "will",
        "in", "to", "for", "of", "and", "or", "but", "not", "also",
        "please", "hey", "hi", "hello", "momo", "check", "look",
        "has", "have", "had", "been", "there", "they", "this", "that",
        "up", "out", "all", "some", "just", "like", "know", "see", "want",
    }
    words = re.findall(r'\b[a-zA-Z0-9]+\b', message.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords[:5]
