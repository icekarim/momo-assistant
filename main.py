"""
Momo — FastAPI Application

Endpoints:
  POST /chat                    — Google Chat webhook (receives user messages)
  POST /briefing                — Trigger morning briefing (called by Cloud Scheduler)
  POST /email-alerts            — Trigger proactive important email checks
  POST /meeting-debrief         — Post-meeting debrief with Granola notes (Cloud Scheduler)
  POST /meeting-prep            — Pre-meeting prep briefs with KG context (Cloud Scheduler)
  POST /knowledge-backfill      — Backfill knowledge graph from recent meetings/emails
  POST /knowledge-embed-backfill — Add vector embeddings to existing KG entities
  GET  /health                  — Health check
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import traceback
import threading

import re
import config
from briefing import run_morning_briefing, run_proactive_email_alerts, run_post_meeting_debrief
from gmail_service import (
    fetch_unread_client_emails,
    search_emails,
    format_emails_for_context,
)
from calendar_service import (
    fetch_todays_meetings,
    format_meetings_for_context,
)
from tasks_service import (
    fetch_open_tasks,
    format_tasks_for_context,
    create_task,
    update_task,
    complete_task,
    delete_task,
)
from gemini_service import chat_response, transcribe_audio
from chat_service import format_for_google_chat, send_chat_message, download_attachment, _SUPPORTED_AUDIO_TYPES
from conversation_store import (
    get_conversation,
    add_turn,
    clear_conversation,
    conversation_scope,
    get_pending_task_actions,
    clear_pending_task_actions,
    store_pending_task_actions,
    store_pending_task_actions_if_empty,
)

app = FastAPI(title="Momo")

# ── API Secret Middleware ────────────────────────────────────
# Protects all endpoints except /health, /, and /chat (Google Chat webhook)

_OPEN_PATHS = {"/health", "/", "/chat"}


@app.middleware("http")
async def api_secret_middleware(request: Request, call_next):
    if not config.MOMO_API_SECRET:
        return await call_next(request)
    if request.url.path in _OPEN_PATHS:
        return await call_next(request)
    if request.headers.get("X-Momo-Secret") != config.MOMO_API_SECRET:
        return JSONResponse(
            status_code=403,
            content={"detail": "Forbidden: invalid or missing X-Momo-Secret header"},
        )
    return await call_next(request)


@app.on_event("startup")
async def startup_warmup():
    """Pre-initialize Google credentials, discovery docs, and KG embeddings on startup."""
    from google_auth import warmup
    warmup()
    try:
        from google_auth import get_credentials
        from googleapiclient.discovery import build
        creds = get_credentials()
        build("gmail", "v1", credentials=creds)
        build("calendar", "v3", credentials=creds)
        build("tasks", "v1", credentials=creds)
        print("Discovery docs cached on startup")
    except Exception as e:
        print(f"Discovery cache warmup failed (will retry on first request): {e}")

    # Pre-load KG embeddings so first semantic search is fast
    if config.KNOWLEDGE_GRAPH_ENABLED:
        def _warm_kg():
            from knowledge_graph import warm_embedding_cache
            warm_embedding_cache()
        threading.Thread(target=_warm_kg, daemon=True).start()


# ── Helpers for Workspace Add-on format ──────────────────────

def _parse_event(body: dict) -> dict:
    """Parse both standard Chat events and Workspace Add-on events into a
    normalized dict with keys: event_type, text, user_id, user_name, space,
    is_addon, attachments."""
    is_addon = "commonEventObject" in body or "chat" in body

    event_type = body.get("type")
    text = ""
    user_id = "unknown"
    user_name = "there"
    space = ""
    attachments = []

    if is_addon:
        chat_payload = body.get("chat", {})
        if not event_type:
            if "messagePayload" in chat_payload:
                event_type = "MESSAGE"
            elif "addedToSpacePayload" in chat_payload:
                event_type = "ADDED_TO_SPACE"
            elif "removedFromSpacePayload" in chat_payload:
                event_type = "REMOVED_FROM_SPACE"
            elif "buttonClickedPayload" in chat_payload:
                event_type = "CARD_CLICKED"

        msg_payload = chat_payload.get("messagePayload", {})
        msg = msg_payload.get("message", {})
        text = msg.get("argumentText", msg.get("text", "")).strip()
        attachments = msg.get("attachment", [])

        chat_user = chat_payload.get("user", {})
        user_id = chat_user.get("name", "unknown")
        user_name = chat_user.get("displayName", "there")

        space_info = msg_payload.get("space", {})
        space = space_info.get("name", "")
    else:
        msg = body.get("message", {})
        text = msg.get("argumentText", msg.get("text", "")).strip()
        attachments = msg.get("attachment", [])
        user = body.get("user", {})
        user_id = user.get("name", "unknown")
        user_name = user.get("displayName", "there")
        space = body.get("space", {}).get("name", "")

    return {
        "event_type": event_type,
        "text": text,
        "user_id": user_id,
        "user_name": user_name,
        "space": space,
        "is_addon": is_addon,
        "attachments": attachments,
    }


def _make_response(text: str, is_addon: bool) -> dict:
    """Build the response in the correct format (add-on vs standard Chat)."""
    if not text:
        return {}
    if is_addon:
        return {
            "hostAppDataAction": {
                "chatDataAction": {
                    "createMessageAction": {
                        "message": {
                            "text": text
                        }
                    }
                }
            }
        }
    return {"text": text}


# ── Health Check ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return {"status": "ok", "service": "momo"}


# ── Morning Briefing Trigger ─────────────────────────────────

@app.post("/briefing")
async def trigger_briefing():
    """Called by Cloud Scheduler at 8 AM daily."""
    try:
        result = run_morning_briefing()
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/email-alerts")
async def trigger_email_alerts():
    """Called by Cloud Scheduler to proactively alert on important/client emails."""
    try:
        result = run_proactive_email_alerts()
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Post-Meeting Debrief Trigger ─────────────────────────────

@app.post("/meeting-debrief")
async def trigger_meeting_debrief():
    """Called by Cloud Scheduler every ~10 min during work hours.
    Sends short debriefs for recently ended meetings using Granola notes."""
    try:
        result = run_post_meeting_debrief()
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Pre-Meeting Prep Trigger ─────────────────────────────────

@app.post("/meeting-prep")
async def trigger_meeting_prep():
    """Called by Cloud Scheduler every ~10 min during work hours.
    Sends pre-meeting prep briefs for upcoming meetings using KG context."""
    try:
        from proactive_intelligence import run_meeting_prep
        result = run_meeting_prep()
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── KG + Meeting Prep E2E Test ────────────────────────────────

@app.get("/test-kg")
async def test_kg(query: str = "test", person: str = ""):
    """E2E test for knowledge graph queries and meeting prep generation.
    Usage: /test-kg?query=ProjectName&person=Person+Name"""
    import time
    results = {}

    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return {"error": "knowledge graph disabled"}

    from knowledge_graph import query_by_project, query_by_person, semantic_search

    # Test query_by_project
    t0 = time.time()
    try:
        project_results = query_by_project(query)
        results["query_by_project"] = {
            "query": query,
            "count": len(project_results),
            "time_ms": round((time.time() - t0) * 1000),
            "sample": [{"name": e.get("name"), "type": e.get("entity_type"),
                        "date": e.get("source_date")} for e in project_results[:5]],
        }
    except Exception as e:
        results["query_by_project"] = {"error": str(e), "time_ms": round((time.time() - t0) * 1000)}

    # Test query_by_person (if provided)
    if person:
        t0 = time.time()
        try:
            person_results = query_by_person(person)
            results["query_by_person"] = {
                "query": person,
                "count": len(person_results),
                "time_ms": round((time.time() - t0) * 1000),
                "sample": [{"name": e.get("name"), "type": e.get("entity_type"),
                            "date": e.get("source_date")} for e in person_results[:5]],
            }
        except Exception as e:
            results["query_by_person"] = {"error": str(e), "time_ms": round((time.time() - t0) * 1000)}

    # Test semantic_search
    t0 = time.time()
    try:
        semantic_results = semantic_search(query)
        results["semantic_search"] = {
            "query": query,
            "count": len(semantic_results),
            "time_ms": round((time.time() - t0) * 1000),
            "sample": [{"name": e.get("name"), "type": e.get("entity_type"),
                        "date": e.get("source_date")} for e in semantic_results[:5]],
        }
    except Exception as e:
        results["semantic_search"] = {"error": str(e), "time_ms": round((time.time() - t0) * 1000)}

    # Test simulated meeting prep
    t0 = time.time()
    try:
        from proactive_intelligence import _build_meeting_prep
        fake_meeting = {
            "title": f"Internal Sync: {query}",
            "start_time": "3:00 PM",
            "attendees": [{"name": person}] if person else [{"name": "Team"}],
        }
        prep = _build_meeting_prep(fake_meeting)
        results["meeting_prep"] = {
            "time_ms": round((time.time() - t0) * 1000),
            "generated": bool(prep),
            "length": len(prep) if prep else 0,
            "preview": (prep[:500] + "...") if prep and len(prep) > 500 else prep,
        }
    except Exception as e:
        results["meeting_prep"] = {"error": str(e), "time_ms": round((time.time() - t0) * 1000)}

    return results


# ── LangSmith Evals Trigger ──────────────────────────────────

@app.post("/run-evals")
async def trigger_evals():
    """Called by Cloud Scheduler once daily to run LLM-as-judge evals
    against production traces collected in the momo-prod-traces dataset."""
    from datetime import datetime
    try:
        from scripts.run_langsmith_evals import run_evals
        prefix = f"momo-eval-{datetime.now().strftime('%Y%m%d')}"
        run_evals(prefix=prefix, limit=50)
        return {"status": "ok", "experiment": prefix}
    except SystemExit:
        # run_evals calls sys.exit if dataset is empty — catch it
        return {"status": "skipped", "reason": "dataset empty or not found"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/promote-eval-failures")
async def trigger_promote_eval_failures():
    """Auto-promote staged production failures to the eval regression dataset.

    Reads pending failures from Firestore eval_failures collection and adds
    them to the momo-eval-golden dataset for regression testing."""
    try:
        from scripts.promote_failures_to_evals import auto_promote
        result = auto_promote()
        return {"status": "ok", **result}
    except SystemExit:
        return {"status": "skipped", "reason": "dataset not found"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Granola Token Refresh ────────────────────────────────────

@app.post("/granola-token-refresh")
async def granola_token_refresh():
    """Proactively refresh the Granola OAuth token.

    Designed to be called by Cloud Scheduler every 4 hours to keep the
    refresh_token alive. Granola access tokens last 6h; refreshing before
    expiry prevents the refresh_token from going stale.
    """
    if not config.GRANOLA_ENABLED:
        return {"status": "skipped", "reason": "granola disabled"}

    try:
        from granola_service import _load_token, _cached_token
        token = _load_token()
        if token:
            return {"status": "ok", "message": "Granola token is valid"}
        else:
            return {"status": "error", "message": "Granola token refresh failed — re-run granola_auth_setup.py"}
    except Exception as e:
        return {"status": "error", "message": f"Granola token refresh failed: {str(e)}"}


# ── Knowledge Graph Backfill ─────────────────────────────────

@app.post("/knowledge-backfill")
async def trigger_knowledge_backfill():
    """Reprocess recent meetings and emails into the knowledge graph.
    Returns immediately and processes in a background thread."""
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return {"status": "skipped", "reason": "knowledge graph disabled"}

    thread = threading.Thread(target=_run_backfill, daemon=True)
    thread.start()
    return {"status": "started", "message": "backfill running in background, check logs for progress"}


@app.get("/embedding-health")
async def get_embedding_health():
    """Return coverage stats for the embedding system."""
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return {"status": "skipped", "reason": "knowledge graph disabled"}
    from knowledge_graph import embedding_health
    return embedding_health()


@app.post("/knowledge-embed-backfill")
async def trigger_embed_backfill(include_stale: bool = False):
    """Add vector embeddings to existing KG entities that don't have one.
    Pass ?include_stale=true to also re-embed entities on an older model.
    Returns immediately and processes in a background thread."""
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return {"status": "skipped", "reason": "knowledge graph disabled"}

    def _run():
        from knowledge_graph import embed_backfill
        embed_backfill(include_stale=include_stale)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "message": "embed backfill running in background, check logs for progress"}


@app.post("/knowledge-search-index-backfill")
async def trigger_search_index_backfill():
    """Add _search_people/_search_projects tokens to existing KG docs.
    Required once after deploying Firestore-native query changes."""
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return {"status": "skipped", "reason": "knowledge graph disabled"}

    def _run():
        from knowledge_graph import search_index_backfill
        search_index_backfill()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "started", "message": "search index backfill running in background, check logs for progress"}


def _run_backfill():
    import time as _time

    from knowledge_graph import extract_and_store

    print("Backfill: starting knowledge graph backfill...")
    meetings_ok, meetings_fail = 0, 0
    emails_ok, emails_fail = 0, 0
    calendar_ok, tasks_ok = 0, 0

    if config.GRANOLA_ENABLED:
        try:
            from granola_service import list_granola_meetings, fetch_meeting_notes_batch
            import re as _re

            xml = list_granola_meetings("last_30_days")
            if xml:
                meetings = []
                for match in _re.finditer(r'<meeting\s+([^>]+)>', xml):
                    attrs = dict(_re.findall(r'(\w+)="([^"]*)"', match.group(1)))
                    meeting_id = attrs.get("id")
                    title = attrs.get("title")
                    if not meeting_id or not title:
                        continue
                    meetings.append({
                        "id": meeting_id,
                        "title": title,
                        "source_date": (attrs.get("date", "") or attrs.get("start_date", ""))[:10],
                    })

                print(f"Backfill: found {len(meetings)} meetings to process")

                batch_size = 10
                for i in range(0, len(meetings), batch_size):
                    batch = meetings[i:i + batch_size]
                    ids = [m["id"] for m in batch]
                    try:
                        notes_by_id = fetch_meeting_notes_batch(ids)
                    except Exception as e:
                        print(f"  Backfill: batch fetch failed: {e}")
                        meetings_fail += len(batch)
                        continue

                    for m in batch:
                        notes = notes_by_id.get(m["id"], "")
                        if not notes:
                            continue
                        try:
                            extract_and_store(
                                source_type="meeting",
                                source_id=m["id"],
                                source_title=m["title"],
                                source_date=m.get("source_date", ""),
                                content=notes,
                                attendees=[],
                            )
                            meetings_ok += 1
                        except Exception as e:
                            print(f"  Backfill: extraction failed for '{m['title']}': {e}")
                            traceback.print_exc()
                            meetings_fail += 1
                        _time.sleep(0.5)
        except Exception as e:
            print(f"  Backfill: Granola processing failed: {e}")
            traceback.print_exc()

    try:
        from gmail_service import fetch_unread_client_emails
        emails = fetch_unread_client_emails(max_results=50)
        print(f"Backfill: found {len(emails)} emails to process")
        for email in emails:
            try:
                extract_and_store(
                    source_type="email",
                    source_id=email["id"],
                    source_title=email.get("subject", ""),
                    source_date=email.get("date_ymd") or email.get("date", "")[:10] or email.get("date_human", ""),
                    content=email.get("body", ""),
                    attendees=[email.get("from", "")],
                )
                emails_ok += 1
            except Exception as e:
                print(f"  Backfill: email extraction failed: {e}")
                traceback.print_exc()
                emails_fail += 1
            _time.sleep(0.5)
    except Exception as e:
        print(f"  Backfill: email processing failed: {e}")
        traceback.print_exc()

    # Calendar events — extract from today's meetings
    try:
        from calendar_service import fetch_todays_meetings
        from knowledge_graph import extract_from_calendar_events
        cal_events = fetch_todays_meetings()
        print(f"Backfill: found {len(cal_events)} calendar events to process")
        extract_from_calendar_events(cal_events)
        calendar_ok = len([e for e in cal_events if not e.get("is_all_day")])
    except Exception as e:
        print(f"  Backfill: calendar processing failed: {e}")
        traceback.print_exc()

    # Tasks — extract current open tasks
    try:
        from tasks_service import fetch_open_tasks
        from knowledge_graph import extract_from_tasks
        open_tasks = fetch_open_tasks()
        print(f"Backfill: found {len(open_tasks)} open tasks to process")
        extract_from_tasks(open_tasks)
        tasks_ok = len(open_tasks)
    except Exception as e:
        print(f"  Backfill: tasks processing failed: {e}")
        traceback.print_exc()

    print(f"Backfill complete: meetings={meetings_ok} ok/{meetings_fail} fail, "
          f"emails={emails_ok} ok/{emails_fail} fail, "
          f"calendar={calendar_ok}, tasks={tasks_ok}")


# ── Google Chat Webhook ──────────────────────────────────────

@app.api_route("/chat", methods=["GET", "POST"])
async def chat_webhook(request: Request):
    """Receives messages from Google Chat (supports both standard and Add-on format)."""
    if request.method == "GET":
        return {"status": "ok", "message": "Momo Chat endpoint"}

    body = await request.json()
    ev = _parse_event(body)
    event_type = ev["event_type"]
    is_addon = ev["is_addon"]

    if event_type == "ADDED_TO_SPACE":
        space = ev["space"]
        user_name = ev["user_name"]
        print(f"Momo added to space: {space}")
        reply = (
            f"Heyyy {user_name}! I'm Momo — your personal briefing sidekick.\n\n"
            "Every morning at 8 AM I'll hit you with the rundown: client emails, "
            "meetings, tasks — the whole vibe.\n\n"
            "You can also just ask me stuff like:\n"
            "- *What's on my schedule today?*\n"
            "- *Any urgent emails?*\n"
            "- *What did [client] send me?*\n"
            "- *What are my overdue tasks?*\n"
            "- *Help me draft a reply to [person]*\n\n"
            "Type *clear* to wipe our chat history and start fresh.\n\n"
            "Let's get it"
        )
        return _make_response(reply, is_addon)

    if event_type == "REMOVED_FROM_SPACE":
        print("Momo removed from space.")
        return {}

    if event_type == "MESSAGE":
        return await handle_message(ev)

    return _make_response("Momo is here. Send me a message to get started.", is_addon)


_APPROVE_WORDS = {
    "yes",
    "approve",
    "approved",
    "confirm",
    "confirmed",
    "yes please",
    "yes approve",
    "approve them",
    "approve please",
    "confirm them",
    "create them",
    "create these tasks",
}
_DECLINE_WORDS = {
    "no",
    "nope",
    "no thanks",
    "decline",
    "cancel",
    "cancel them",
    "skip",
    "skip them",
    "don't do it",
    "dont do it",
}
_APPROVE_PREFIXES = tuple(sorted(_APPROVE_WORDS, key=len, reverse=True))
_DECLINE_PREFIXES = tuple(sorted(_DECLINE_WORDS, key=len, reverse=True))

# Verbs that mean "decline this specific task" without a leading no/cancel.
_DECLINE_ACTION_WORDS = {
    "drop",
    "remove",
    "ditch",
    "scratch",
    "lose",
    "dont need",
    "don't need",
    "i dont need",
    "i don't need",
    "get rid of",
    "dont want",
    "don't want",
    "i dont want",
    "i don't want",
    "no need for",
}
_DECLINE_ACTION_PREFIXES = tuple(sorted(_DECLINE_ACTION_WORDS, key=len, reverse=True))

# Qualifiers after a decline word that invert the meaning to "keep only X".
_KEEP_ONLY_QUALIFIERS = ("but keep", "but only", "but just", "keep", "just", "only")

# Standalone keep-only patterns (no leading decline word required).
_KEEP_ACTION_WORDS = {
    "i just want",
    "i only want",
    "i just need",
    "i only need",
    "keep only",
    "only keep",
    "keep just",
    "just keep",
    "keep",
    "just",
    "only",
}
_KEEP_ACTION_PREFIXES = tuple(sorted(_KEEP_ACTION_WORDS, key=len, reverse=True))

_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_REFERENCE_FILLER_WORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "task", "tasks", "item", "items", "one", "ones",
    "please", "pls", "do", "for", "me",
}


def _check_pending_task_intent(lower: str) -> str | None:
    """Detect explicit approval or rejection of a pending task request."""
    normalized = re.sub(r"\s+", " ", lower).strip().rstrip(".,!?")
    if normalized in _APPROVE_WORDS:
        return "confirm"
    if normalized in _DECLINE_WORDS:
        return "decline"
    return None


def _extract_pending_task_command(lower: str) -> tuple[str | None, str]:
    """Return (intent, remainder) for explicit approve/cancel commands."""
    normalized = re.sub(r"\s+", " ", lower).strip().rstrip(".,!?")

    for phrase in _APPROVE_PREFIXES:
        if normalized == phrase:
            return "confirm", ""
        if normalized.startswith(f"{phrase} ") or normalized.startswith(f"{phrase},"):
            return "confirm", normalized[len(phrase):].lstrip(" ,;:-")

    for phrase in _DECLINE_PREFIXES:
        if normalized == phrase:
            return "decline", ""
        if normalized.startswith(f"{phrase} ") or normalized.startswith(f"{phrase},"):
            return "decline", normalized[len(phrase):].lstrip(" ,;:-")

    return None, ""


def _action_reference_texts(action: dict) -> list[str]:
    """Return the task phrases a user might use to reference an action."""
    refs = []
    if action.get("find"):
        refs.append(action["find"])
    if action.get("title"):
        refs.append(action["title"])
    return refs


def _normalize_reference_tokens(text: str) -> tuple[list[str], str]:
    """Normalize a phrase for fuzzy task-reference matching."""
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    filtered = [token for token in tokens if token not in _REFERENCE_FILLER_WORDS]
    return filtered, " ".join(filtered)


def _match_action_by_reference(reference: str, actions: list[dict]) -> set[int] | None:
    """Try to match a user-supplied task reference to pending actions."""
    ref_tokens, ref_phrase = _normalize_reference_tokens(reference)
    if not ref_tokens:
        return None

    ref_compact = ref_phrase.replace(" ", "")
    substring_matches = set()
    overlap_matches = set()

    for idx, action in enumerate(actions):
        for candidate in _action_reference_texts(action):
            candidate_tokens, candidate_phrase = _normalize_reference_tokens(candidate)
            if not candidate_tokens:
                continue

            candidate_compact = candidate_phrase.replace(" ", "")

            # Substring match: handles normal case and compound words
            # e.g. "footlocker" matches "foot locker", "foot locker" matches "footlocker"
            if ref_phrase and (
                ref_phrase in candidate_phrase
                or ref_compact in candidate_compact
                or candidate_compact in ref_compact
            ):
                substring_matches.add(idx)
                continue

            overlap = len(set(ref_tokens) & set(candidate_tokens))
            if len(ref_tokens) == 1:
                if overlap == 1 and ref_tokens[0] in candidate_tokens:
                    overlap_matches.add(idx)
            elif overlap >= 2 and overlap / len(ref_tokens) >= 0.6:
                overlap_matches.add(idx)

    if len(substring_matches) == 1:
        return substring_matches
    if len(substring_matches) > 1:
        return None
    if len(overlap_matches) == 1:
        return overlap_matches
    return None


def _select_pending_action_indices(reference: str, actions: list[dict]) -> set[int] | None:
    """Resolve a partial approval/cancel reference to pending action indices."""
    normalized = re.sub(r"\s+", " ", reference).strip().rstrip(".,!?")
    if not normalized:
        return set(range(len(actions)))

    all_refs = {
        "all",
        "all of them",
        "them",
        "these",
        "those",
        "these tasks",
        "those tasks",
        "everything",
    }
    singular_refs = {"it", "this one", "that one", "one of them", "one task"}

    if normalized in all_refs:
        return set(range(len(actions)))
    if normalized in singular_refs:
        return {0} if len(actions) == 1 else None

    indices = set()

    for match in re.findall(r"#?(\d+)", normalized):
        idx = int(match)
        if 1 <= idx <= len(actions):
            indices.add(idx - 1)

    for word, idx in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", normalized):
            if idx <= len(actions):
                indices.add(idx - 1)
    if re.search(r"\blast\b", normalized) and actions:
        indices.add(len(actions) - 1)

    if indices:
        return indices

    return _match_action_by_reference(normalized, actions)


def _parse_pending_task_reply(lower: str, actions: list[dict]) -> dict:
    """Parse an approval/cancel reply and resolve which actions it targets.

    Returns a dict with keys:
      intent: "confirm" | "decline" | None
      selected_indices: set of action indices the intent applies to
      ambiguous: True if we detected intent but couldn't resolve the target
      dismiss_rest: True if non-selected actions should be cleared (not kept pending)
    """
    exact_intent = _check_pending_task_intent(lower)
    if exact_intent:
        return {
            "intent": exact_intent,
            "selected_indices": set(range(len(actions))),
            "ambiguous": False,
            "dismiss_rest": False,
        }

    intent, remainder = _extract_pending_task_command(lower)

    # "no just X" / "no only X" / "no, keep X" → approve only X, dismiss rest.
    # The user is declining *everything else*, not the referenced task.
    if intent == "decline" and remainder:
        for qualifier in _KEEP_ONLY_QUALIFIERS:
            if remainder == qualifier or remainder.startswith(f"{qualifier} "):
                keep_ref = remainder[len(qualifier):].lstrip(" ,:-") if remainder.startswith(f"{qualifier} ") else ""
                selected = _select_pending_action_indices(keep_ref, actions)
                if selected is None:
                    return {"intent": "confirm", "selected_indices": set(), "ambiguous": True, "dismiss_rest": True}
                return {
                    "intent": "confirm",
                    "selected_indices": selected,
                    "ambiguous": False,
                    "dismiss_rest": True,
                }

    if intent:
        selected_indices = _select_pending_action_indices(remainder, actions)
        if selected_indices is None:
            return {"intent": intent, "selected_indices": set(), "ambiguous": True, "dismiss_rest": False}
        return {
            "intent": intent,
            "selected_indices": selected_indices,
            "ambiguous": False,
            "dismiss_rest": False,
        }

    # No standard approve/decline prefix — check action-verb patterns.
    normalized = re.sub(r"\s+", " ", lower).strip().rstrip(".,!?")

    # Decline action verbs: "drop X", "remove X", "dont need X", etc.
    for phrase in _DECLINE_ACTION_PREFIXES:
        if normalized == phrase:
            return {
                "intent": "decline",
                "selected_indices": set(range(len(actions))),
                "ambiguous": False,
                "dismiss_rest": False,
            }
        if normalized.startswith(f"{phrase} "):
            action_ref = normalized[len(phrase):].lstrip(" ,:-")
            selected = _select_pending_action_indices(action_ref, actions)
            if selected is None:
                return {"intent": "decline", "selected_indices": set(), "ambiguous": True, "dismiss_rest": False}
            return {
                "intent": "decline",
                "selected_indices": selected,
                "ambiguous": False,
                "dismiss_rest": False,
            }

    # Keep-only patterns: "just X", "keep X", "only X", "i just want X"
    for phrase in _KEEP_ACTION_PREFIXES:
        if normalized == phrase:
            return {
                "intent": "confirm",
                "selected_indices": set(range(len(actions))),
                "ambiguous": False,
                "dismiss_rest": True,
            }
        if normalized.startswith(f"{phrase} "):
            keep_ref = normalized[len(phrase):].lstrip(" ,:-")
            selected = _select_pending_action_indices(keep_ref, actions)
            if selected is None:
                return {"intent": "confirm", "selected_indices": set(), "ambiguous": True, "dismiss_rest": True}
            return {
                "intent": "confirm",
                "selected_indices": selected,
                "ambiguous": False,
                "dismiss_rest": True,
            }

    return {"intent": None, "selected_indices": set(), "ambiguous": False, "dismiss_rest": False}


def _user_task_scope(user_id: str, space: str) -> str:
    """Scope direct task approvals to the current user in the current space."""
    return f"user:{space or 'direct'}:{user_id}"


def _space_task_scope(space: str) -> str | None:
    """Scope scheduled debrief approvals to the chat space."""
    if not space:
        return None
    return f"space:{space}"


def _get_pending_task_request(user_id: str, space: str) -> tuple[dict | None, str | None]:
    """Fetch the highest-priority pending task request for this message."""
    user_scope = _user_task_scope(user_id, space)
    pending = get_pending_task_actions(scope_id=user_scope)
    if pending:
        return pending, user_scope

    space_scope = _space_task_scope(space)
    if space_scope:
        pending = get_pending_task_actions(scope_id=space_scope)
        if pending:
            return pending, space_scope

    return None, None


def _format_pending_task_action(action: dict) -> str:
    """Render a short human-readable summary of a pending task mutation."""
    op = action.get("action", "create")
    if op == "create":
        due = f" (due {action['due']})" if action.get("due") else ""
        return f"create *{action['title']}*{due}"

    if op == "update":
        changes = []
        if action.get("title"):
            changes.append(f"rename to *{action['title']}*")
        if action.get("due"):
            changes.append(f"set due {action['due']}")
        if "notes" in action:
            changes.append("clear notes" if not action.get("notes") else "update notes")
        change_str = "; ".join(changes) if changes else "update details"
        return f"update *{action['find']}* ({change_str})"

    if op == "complete":
        return f"mark *{action['find']}* complete"

    if op == "delete":
        return f"delete *{action['find']}*"

    return f"{op} task request"


def _build_task_approval_block(actions: list[dict]) -> str:
    """Build the standard approval block appended to pending task replies."""
    lines = ["📝 *Approve these Google Tasks changes*"]
    lines.extend(f"  {idx}. {_format_pending_task_action(action)}" for idx, action in enumerate(actions, start=1))
    lines.append("")
    if len(actions) == 1:
        lines.append('_Reply *yes* to apply this change in Google Tasks, or *no* to cancel_')
    else:
        lines.append(
            f"_Reply *yes* to apply all {len(actions)} changes in Google Tasks, "
            "*approve 2* to apply only item #2, *remove 1* to drop item #1, or *no* to cancel_"
        )
    return "\n".join(lines)


def _append_task_approval_block(response: str, actions: list[dict]) -> str:
    """Append the pending-approval summary to the agent's response."""
    block = _build_task_approval_block(actions)
    if not response:
        return block
    return f"{response.rstrip()}\n\n{block}"


def _build_pending_selection_help(intent: str, actions: list[dict]) -> str:
    """Prompt the user to clarify which pending task(s) they mean."""
    verb = "approve" if intent == "confirm" else "cancel"
    return (
        f"not sure which item you want to {verb}. reply with `{verb} 1` to choose item #1 "
        f"or paste part of the task name.\n\n{_build_task_approval_block(actions)}"
    )


def _build_pending_conflict_reply(pending_request: dict) -> str:
    """Explain that a new task request was not queued because one is already pending."""
    prefix = "you already have a pending Google Tasks approval"
    if pending_request.get("meeting_title"):
        prefix += f" from *{pending_request['meeting_title']}*"
    prefix += ", so i didn't queue this new task change."
    return (
        f"{prefix}\n\n"
        "reply `yes` to apply it or `no` to cancel it, then resend your new task request.\n\n"
        f"{_build_task_approval_block(pending_request['actions'])}"
    )


def _persist_pending_request(scope_id: str, pending_request: dict, actions: list[dict]) -> None:
    """Update or clear the pending request for a scope."""
    if actions:
        store_pending_task_actions(
            actions,
            scope_id=scope_id,
            meeting_title=pending_request.get("meeting_title", ""),
            approval_message=pending_request.get("approval_message", ""),
        )
        return
    clear_pending_task_actions(scope_id=scope_id)


async def handle_message(ev: dict) -> dict:
    """Process an incoming chat message (text and/or voice).
    Returns an immediate ack and processes the real response in a background
    thread, sending it via the Chat API when ready."""
    text = ev["text"]
    user_id = ev["user_id"]
    space = ev["space"]
    is_addon = ev["is_addon"]
    attachments = ev.get("attachments", [])

    audio_attachments = [
        a for a in attachments
        if a.get("contentType", "").startswith("audio/")
    ]
    has_non_audio_attachments = len(attachments) > len(audio_attachments) and not audio_attachments

    if space and not config.CHAT_SPACE_ID:
        print(f"Detected space ID: {space}")
        print(f"   Set CHAT_SPACE_ID={space} in your environment variables")

    if not text and not audio_attachments:
        if has_non_audio_attachments:
            return _make_response(
                "i can only handle voice messages for now — try sending an audio clip or just type it out",
                is_addon,
            )
        return _make_response(
            "Hmm, I got nothing there. Try asking about your emails, meetings, or tasks",
            is_addon,
        )

    if text and not audio_attachments:
        lower = text.lower().strip()
        if lower in ("clear", "reset", "start over"):
            clear_conversation(conversation_scope(user_id=user_id, space=space))
            return _make_response("Slate wiped. What can Momo do for you?", is_addon)

        if lower in ("briefing", "morning briefing", "daily briefing"):
            try:
                run_morning_briefing(space or config.CHAT_SPACE_ID)
                return _make_response("Morning briefing sent!", is_addon)
            except Exception as e:
                return _make_response(f"Error generating briefing: {str(e)}", is_addon)

        pending_request, pending_scope_id = _get_pending_task_request(
            user_id,
            space or config.CHAT_SPACE_ID,
        )
        if pending_request:
            parsed_reply = _parse_pending_task_reply(lower, pending_request["actions"])
            if parsed_reply["ambiguous"]:
                return _make_response(
                    _build_pending_selection_help(parsed_reply["intent"], pending_request["actions"]),
                    is_addon,
                )
            if parsed_reply["intent"] == "confirm":
                selected_indices = parsed_reply["selected_indices"]
                selected_actions = [
                    action for idx, action in enumerate(pending_request["actions"])
                    if idx in selected_indices
                ]
                remaining_actions = [] if parsed_reply.get("dismiss_rest") else [
                    action for idx, action in enumerate(pending_request["actions"])
                    if idx not in selected_indices
                ]
                target_space = space or config.CHAT_SPACE_ID
                _persist_pending_request(pending_scope_id, pending_request, remaining_actions)
                thread = threading.Thread(
                    target=_apply_pending_task_actions_background,
                    args=(
                        selected_actions,
                        remaining_actions,
                        pending_request.get("meeting_title", ""),
                        target_space,
                        pending_scope_id,
                    ),
                    daemon=True,
                )
                thread.start()
                return _make_response("", is_addon)
            if parsed_reply["intent"] == "decline":
                selected_indices = parsed_reply["selected_indices"]
                remaining_actions = [
                    action for idx, action in enumerate(pending_request["actions"])
                    if idx not in selected_indices
                ]
                if remaining_actions:
                    _persist_pending_request(pending_scope_id, pending_request, remaining_actions)
                    reply = (
                        "got it — removed that task from pending approval.\n\n"
                        + _build_task_approval_block(remaining_actions)
                    )
                else:
                    _persist_pending_request(pending_scope_id, pending_request, remaining_actions)
                    reply = "Got it — canceled that pending task request."
                return _make_response(reply, is_addon)

    target_space = space or config.CHAT_SPACE_ID
    thread = threading.Thread(
        target=_process_message_background,
        args=(text, user_id, target_space, audio_attachments),
        daemon=True,
    )
    thread.start()

    return _make_response("", is_addon)


def _apply_pending_task_actions_background(
    pending_actions,
    remaining_actions,
    meeting_title,
    space,
    scope_id,
):
    """Apply approved pending task mutations and report back."""
    try:
        results = []
        skipped = []
        errors = []
        for action in pending_actions:
            op = action.get("action", "create")
            label = action.get("title") or action.get("find") or op
            try:
                if op == "create":
                    result = create_task(
                        title=action["title"],
                        notes=action.get("notes", ""),
                        due_date=action.get("due"),
                    )
                    status = result.get("status", "error")
                    if "error" in result:
                        errors.append(f"{label}: {result['error']}")
                    elif status in ("already_exists", "already_completed"):
                        skipped.append(f"*{result['title']}* — {status.replace('_', ' ')}")
                    else:
                        results.append(f"created *{result['title']}*")
                elif op == "update":
                    result = update_task(
                        task_title=action["find"],
                        new_title=action.get("title"),
                        new_notes=action.get("notes"),
                        new_due=action.get("due"),
                    )
                    if "error" in result:
                        errors.append(f"{label}: {result['error']}")
                    else:
                        results.append(f"updated *{result['title']}*")
                elif op == "complete":
                    result = complete_task(task_title=action["find"])
                    if "error" in result:
                        errors.append(f"{label}: {result['error']}")
                    else:
                        results.append(f"completed *{result['title']}*")
                elif op == "delete":
                    result = delete_task(task_title=action["find"])
                    if "error" in result:
                        errors.append(f"{label}: {result['error']}")
                    else:
                        results.append(f"deleted *{result['title']}*")
                else:
                    errors.append(f"{label}: unknown task action '{op}'")
            except Exception as e:
                errors.append(f"{label}: {str(e)}")

        lines = []
        if meeting_title:
            lines.append(f"📝 *approved from {meeting_title}*")
        if results:
            lines.append(f"✅ *{len(results)} task change(s) applied:*")
            lines.extend(f"  • {result}" for result in results)
        if skipped:
            lines.append(f"🟡 *{len(skipped)} skipped:*")
            lines.extend(f"  • {s}" for s in skipped)
        if errors:
            lines.append(f"🔴 *{len(errors)} failed:*")
            lines.extend(f"  • {e}" for e in errors)

        if remaining_actions:
            lines.append("")
            lines.append(_build_task_approval_block(remaining_actions))

        reply = "\n".join(lines) if lines else "No task changes were applied."
        formatted = format_for_google_chat(reply)
        send_chat_message(space, formatted)
    except Exception as e:
        traceback.print_exc()
        try:
            send_chat_message(space, f"sorry, something went wrong applying that task request: {str(e)}")
        except Exception:
            print(f"Failed to send error message: {e}")


def _transcribe_voice_message(audio_attachments, existing_text, space):
    """Download and transcribe audio attachments. Returns the final text to
    process, or None if transcription fails entirely (error already sent)."""
    for attachment in audio_attachments:
        content_type = attachment.get("contentType", "")
        if content_type not in _SUPPORTED_AUDIO_TYPES:
            print(f"Unsupported audio type: {content_type}")
            continue

        resource_name = attachment.get("attachmentDataRef", {}).get("resourceName")
        if not resource_name:
            resource_name = attachment.get("name", "")
        if not resource_name:
            print("Attachment missing resource name, skipping")
            continue

        result = download_attachment(resource_name)
        if result is None:
            continue

        audio_bytes, detected_type = result
        mime = detected_type if detected_type.startswith("audio/") else content_type

        transcription = transcribe_audio(audio_bytes, mime)
        if transcription:
            if existing_text:
                return f"{existing_text}\n\n[voice message]: {transcription}"
            return transcription

    if not existing_text:
        send_chat_message(space, "couldn't process that voice message — try typing it out?")
        return None
    return existing_text


def _process_message_background(text, user_id, space, audio_attachments=None):
    """Heavy processing in background thread — no 30s webhook pressure."""
    import time
    _t0 = time.time()
    print(f"[perf] processing message ({len(text or '')} chars)")
    try:
        if audio_attachments:
            text = _transcribe_voice_message(audio_attachments, text, space)
            if text is None:
                return

        conversation_id = conversation_scope(user_id=user_id, space=space)
        history = get_conversation(conversation_id)
        _t1 = time.time()
        print(f"[perf] get_conversation: {_t1 - _t0:.2f}s (history={len(history)} turns)")

        pending_task_actions = []
        if config.AGENTIC_MODE_ENABLED:
            from agent import run_agent_loop
            response, pending_task_actions = run_agent_loop(text, history, thread_id=conversation_id, user_id=user_id)
            if pending_task_actions:
                pending_scope_id = _user_task_scope(user_id, space)
                approval_response = _append_task_approval_block(response, pending_task_actions)
                if store_pending_task_actions_if_empty(
                    pending_task_actions,
                    scope_id=pending_scope_id,
                    approval_message=approval_response,
                ):
                    response = approval_response
                else:
                    existing_pending = get_pending_task_actions(scope_id=pending_scope_id)
                    if existing_pending:
                        response = _build_pending_conflict_reply(existing_pending)
                    else:
                        response = "sorry, something went wrong queueing that task change — try again?"
        else:
            context_data = _build_context(text)
            response = chat_response(text, history, context_data, thread_id=conversation_id)
            response = _remove_task_tags(response)

        _t2 = time.time()
        print(f"[perf] response: {_t2 - _t1:.2f}s ({len(response or '')} chars)")

        add_turn(conversation_id, "user", text)
        add_turn(conversation_id, "assistant", response)

        if config.KNOWLEDGE_GRAPH_ENABLED:
            from datetime import datetime as _dt
            from knowledge_graph import extract_and_store_background
            _now = _dt.now()
            extract_and_store_background(
                source_type="chat",
                source_id=f"chat-{user_id}-{_now.strftime('%Y%m%d%H%M%S')}",
                source_title="Chat message",
                source_date=_now.strftime("%Y-%m-%d"),
                content=text,
                attendees=[],
            )

        formatted = format_for_google_chat(response)
        _t3 = time.time()
        send_chat_message(space, formatted)
        _t4 = time.time()
        print(f"[perf] send_chat_message: {_t4 - _t3:.2f}s")
        print(f"[perf] TOTAL: {_t4 - _t0:.2f}s")

    except Exception as e:
        traceback.print_exc()
        try:
            send_chat_message(space, f"sorry, something went wrong: {str(e)}")
        except Exception:
            print(f"Failed to send error message: {e}")


def _remove_task_tags(response):
    """Remove all task action tags from the response shown to the user.
    Used by the legacy (non-agentic) fallback path."""
    cleaned = re.sub(r'\[(CREATE|UPDATE|COMPLETE|DELETE)_TASK\][^\n]*\n?', '', response)
    return cleaned.strip()


def _extract_search_terms(user_message):
    """Extract entity names/keywords for targeted email search."""
    stopwords = {
        "what", "whats", "how", "is", "are", "the", "a", "an", "my", "me", "i",
        "from", "with", "them", "as", "well", "going", "on", "about", "any",
        "emails", "email", "meetings", "meeting", "tasks", "task", "update",
        "status", "today", "tell", "show", "do", "does", "did", "can", "could",
        "would", "should", "will", "in", "to", "for", "of", "and", "or", "but",
        "not", "also", "please", "hey", "hi", "hello", "momo", "check", "look",
        "whats", "hows", "anything", "give", "get", "got", "has", "have", "had",
        "been", "their", "there", "they", "this", "that", "those", "these",
        "up", "out", "all", "some", "just", "like", "know", "see", "want",
        "right", "now", "currently", "latest", "recent", "recently", "so",
    }
    words = re.findall(r'\b[a-zA-Z0-9]+\b', user_message.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(keywords[:3]) if keywords else None


def _build_context(user_message):
    """Fetch relevant context based on what the user is asking about.
    All API calls run in parallel. Runs in background thread so no webhook
    timeout pressure."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    lower = user_message.lower()
    context = {}

    email_keywords = [
        "email", "mail", "sent", "inbox", "message", "client",
        "unread", "from", "urgent", "reply", "respond",
    ]
    wants_emails = any(kw in lower for kw in email_keywords)

    general_keywords = ["what", "how", "any", "update", "status", "summary", "briefing", "today"]
    is_general = any(kw in lower for kw in general_keywords)

    meeting_keywords = [
        "meeting notes", "meeting note", "discussed", "action items",
        "transcript", "granola", "notes from", "what happened in",
        "debrief", "takeaways", "decisions", "follow up", "follow-up",
    ]
    wants_meeting_notes = any(kw in lower for kw in meeting_keywords)

    search_terms = _extract_search_terms(user_message)
    has_specific_entity = bool(search_terms)

    wants_granola = config.GRANOLA_ENABLED and (wants_meeting_notes or is_general or has_specific_entity)

    jira_keywords = [
        "jira", "ticket", "issue", "sprint", "backlog",
        "story", "bug", "epic", "kanban", "board",
    ]
    wants_jira = config.JIRA_ENABLED and (any(kw in lower for kw in jira_keywords) or is_general)

    def _timed_fetch(name, fn):
        t = time.time()
        result = fn()
        print(f"[perf]   source '{name}': {time.time() - t:.2f}s")
        return result

    def _fetch_meetings():
        meetings = fetch_todays_meetings()
        return "meetings", format_meetings_for_context(meetings)

    def _fetch_tasks():
        tasks = fetch_open_tasks()
        return "tasks", format_tasks_for_context(tasks)

    def _fetch_unread_emails():
        emails = fetch_unread_client_emails(max_results=config.MAX_CHAT_EMAILS)
        return "emails", format_emails_for_context(emails)

    def _fetch_targeted_emails():
        targeted = search_emails(search_terms, days_back=90, max_results=10)
        return "targeted_emails", format_emails_for_context(targeted)

    def _fetch_granola():
        from granola_service import query_granola
        return "granola", query_granola(user_message)

    def _fetch_jira():
        import re as _re
        from jira_service import fetch_active_jira_tickets, get_jira_issue

        # Always fetch active tickets as the base context
        result = fetch_active_jira_tickets()

        # If a specific issue key is mentioned (e.g. OSD-106548), fetch its details too
        issue_keys = _re.findall(r'\b[A-Z]{2,10}-\d+\b', user_message)
        for key in issue_keys[:3]:
            detail = get_jira_issue(key)
            if detail and key not in result:
                result = result + "\n\n" + detail if result else detail

        return "jira", result

    wants_knowledge = config.KNOWLEDGE_GRAPH_ENABLED

    def _fetch_knowledge():
        from knowledge_graph import query_knowledge_graph
        return "knowledge_graph", query_knowledge_graph(user_message)

    per_source_timeout = {
        "meetings": 10,
        "tasks": 10,
        "emails": 15,
        "targeted_emails": 15,
        "granola": 12,
        "jira": 12,
        "knowledge_graph": 10,
    }

    pool = ThreadPoolExecutor(max_workers=8)
    futures = {}
    futures["meetings"] = pool.submit(_timed_fetch, "meetings", _fetch_meetings)
    futures["tasks"] = pool.submit(_timed_fetch, "tasks", _fetch_tasks)
    if wants_emails or is_general:
        futures["emails"] = pool.submit(_timed_fetch, "emails", _fetch_unread_emails)
    if search_terms:
        futures["targeted_emails"] = pool.submit(_timed_fetch, "targeted_emails", _fetch_targeted_emails)
    if wants_granola:
        futures["granola"] = pool.submit(_timed_fetch, "granola", _fetch_granola)
    if wants_jira:
        futures["jira"] = pool.submit(_timed_fetch, "jira", _fetch_jira)
    if wants_knowledge:
        futures["knowledge_graph"] = pool.submit(_timed_fetch, "knowledge_graph", _fetch_knowledge)

    timed_out_sources = []
    for key, future in futures.items():
        timeout = per_source_timeout.get(key, 10)
        try:
            ctx_key, value = future.result(timeout=timeout)
            if value:
                if ctx_key == "targeted_emails" and "emails" in context:
                    context["emails"] = context["emails"] + "\n\n--- Targeted search results ---\n\n" + value
                elif ctx_key == "targeted_emails":
                    context["emails"] = value
                else:
                    context[ctx_key] = value
        except TimeoutError:
            timed_out_sources.append(key)
            print(f"Source '{key}' timed out after {timeout}s — proceeding without it")
            future.cancel()
        except Exception as e:
            print(f"Error fetching {key}: {e}")

    if timed_out_sources:
        human_names = {"meetings": "calendar", "tasks": "tasks", "emails": "email",
                       "targeted_emails": "email search", "granola": "meeting notes",
                       "jira": "jira tickets", "knowledge_graph": "knowledge graph"}
        names = [human_names.get(s, s) for s in timed_out_sources]
        context["_unavailable_sources"] = (
            f"Note: I couldn't reach your {', '.join(names)} right now (timed out). "
            "The rest of the info below is still up to date."
        )

    pool.shutdown(wait=False)
    return context


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
