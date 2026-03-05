"""
Momo — FastAPI Application

Endpoints:
  POST /chat               — Google Chat webhook (receives user messages)
  POST /briefing           — Trigger morning briefing (called by Cloud Scheduler)
  POST /email-alerts       — Trigger proactive important email checks
  POST /meeting-debrief    — Post-meeting debrief with Granola notes (Cloud Scheduler)
  POST /meeting-prep       — Pre-meeting prep briefs with KG context (Cloud Scheduler)
  POST /knowledge-backfill — Backfill knowledge graph from recent meetings/emails
  GET  /health             — Health check
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
from tasks_service import fetch_open_tasks, format_tasks_for_context, create_task, update_task, complete_task, delete_task
from gemini_service import chat_response, transcribe_audio
from chat_service import format_for_google_chat, send_chat_message, download_attachment, _SUPPORTED_AUDIO_TYPES
from conversation_store import get_conversation, add_turn, clear_conversation, get_pending_tasks, clear_pending_tasks

app = FastAPI(title="Momo")


@app.on_event("startup")
async def startup_warmup():
    """Pre-initialize Google credentials and cache discovery docs on startup."""
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


def _run_backfill():
    import time as _time

    from knowledge_graph import extract_and_store

    print("Backfill: starting knowledge graph backfill...")
    meetings_ok, meetings_fail = 0, 0
    emails_ok, emails_fail = 0, 0

    if config.GRANOLA_ENABLED:
        try:
            from granola_service import list_granola_meetings, fetch_meeting_notes_batch
            import re as _re

            xml = list_granola_meetings("last_30_days")
            if xml:
                meetings = []
                for match in _re.finditer(r'<meeting\s+id="([^"]+)"\s+title="([^"]+)"', xml):
                    meetings.append({"id": match.group(1), "title": match.group(2)})

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
                                source_date="",
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
                    source_date=email.get("date_human", ""),
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

    print(f"Backfill complete: meetings={meetings_ok} ok/{meetings_fail} fail, "
          f"emails={emails_ok} ok/{emails_fail} fail")


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
            clear_conversation(user_id)
            return _make_response("Slate wiped. What can Momo do for you?", is_addon)

        if lower in ("briefing", "morning briefing", "daily briefing"):
            try:
                run_morning_briefing()
                return _make_response("Morning briefing sent!", is_addon)
            except Exception as e:
                return _make_response(f"Error generating briefing: {str(e)}", is_addon)

        _CONFIRM_PHRASES = {
            "yes", "yep", "yeah", "yea", "y", "sure", "do it", "go ahead",
            "create them", "create those", "create those tasks", "create tasks",
            "add them", "add those", "add those tasks", "go for it", "please",
            "yes please", "yep do it", "sounds good", "let's do it",
        }
        _DECLINE_PHRASES = {
            "no", "nah", "nope", "n", "skip", "skip those", "no thanks",
            "don't create", "never mind", "nevermind", "cancel",
        }
        if lower in _CONFIRM_PHRASES:
            pending, meeting_title = get_pending_tasks()
            if pending:
                target_space = space or config.CHAT_SPACE_ID
                thread = threading.Thread(
                    target=_create_pending_tasks_background,
                    args=(pending, meeting_title, target_space),
                    daemon=True,
                )
                thread.start()
                return _make_response("", is_addon)
        if lower in _DECLINE_PHRASES:
            pending, _ = get_pending_tasks()
            if pending:
                clear_pending_tasks()
                return _make_response("Got it — skipped those tasks.", is_addon)

    target_space = space or config.CHAT_SPACE_ID
    thread = threading.Thread(
        target=_process_message_background,
        args=(text, user_id, target_space, audio_attachments),
        daemon=True,
    )
    thread.start()

    return _make_response("", is_addon)


def _create_pending_tasks_background(pending_tasks, meeting_title, space):
    """Create confirmed pending tasks from a debrief and report back."""
    try:
        results = []
        skipped = []
        errors = []
        for task in pending_tasks:
            try:
                result = create_task(
                    title=task["title"],
                    notes=task.get("notes", ""),
                    due_date=task.get("due"),
                )
                status = result.get("status", "error")
                if "error" in result:
                    errors.append(f"{task['title']}: {result['error']}")
                elif status in ("already_exists", "already_completed"):
                    skipped.append(f"*{result['title']}* — {status.replace('_', ' ')}")
                else:
                    results.append(f"*{result['title']}* — {status}")
            except Exception as e:
                errors.append(f"{task['title']}: {str(e)}")

        lines = []
        if results:
            lines.append(f"✅ *{len(results)} task(s) created:*")
            lines.extend(f"  • {r}" for r in results)
        if skipped:
            lines.append(f"🟡 *{len(skipped)} skipped (already existed):*")
            lines.extend(f"  • {s}" for s in skipped)
        if errors:
            lines.append(f"🔴 *{len(errors)} failed:*")
            lines.extend(f"  • {e}" for e in errors)

        reply = "\n".join(lines) if lines else "No tasks to create."
        formatted = format_for_google_chat(reply)
        send_chat_message(space, formatted)
        clear_pending_tasks()
    except Exception as e:
        traceback.print_exc()
        try:
            send_chat_message(space, f"sorry, something went wrong creating those tasks: {str(e)}")
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

        history = get_conversation(user_id)
        _t1 = time.time()
        print(f"[perf] get_conversation: {_t1 - _t0:.2f}s (history={len(history)} turns)")

        context_data = _build_context(text)
        _t2 = time.time()
        print(f"[perf] build_context: {_t2 - _t1:.2f}s (sources={list(context_data.keys())})")

        response = chat_response(text, history, context_data)
        _t3 = time.time()
        print(f"[perf] chat_response: {_t3 - _t2:.2f}s (response={len(response or '')} chars)")

        task_actions = _extract_all_task_proposals(response, text)

        if not task_actions:
            bulk = _detect_bulk_task_action(text)
            if bulk:
                task_actions = bulk

        task_summary = ""
        if task_actions:
            results = []
            skipped = []
            errors = []
            for action_item in task_actions:
                try:
                    result = _execute_task_action(action_item)
                    if "error" in result:
                        errors.append(f"{action_item.get('title', action_item.get('find', '?'))}: {result['error']}")
                    elif result.get("status") in ("already_exists", "already_completed"):
                        skipped.append(f"*{result['title']}* — {result['status'].replace('_', ' ')}")
                    else:
                        results.append(f"*{result['title']}* — {result['status']}")
                except Exception as e:
                    errors.append(f"{action_item.get('title', action_item.get('find', '?'))}: {str(e)}")

            lines = []
            if results:
                lines.append(f"\n✅ *done — {len(results)} task(s):*")
                lines.extend(f"  - {r}" for r in results)
            if skipped:
                lines.append(f"\n🟡 *skipped — {len(skipped)} already existed:*")
                lines.extend(f"  - {s}" for s in skipped)
            if errors:
                lines.append(f"\n🔴 *couldn't do {len(errors)}:*")
                lines.extend(f"  - {e}" for e in errors)
            task_summary = "\n".join(lines)

        clean_response = _remove_task_tags(response)
        full_response = clean_response + task_summary if task_summary else clean_response

        add_turn(user_id, "user", text)
        add_turn(user_id, "assistant", clean_response)

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

        formatted = format_for_google_chat(full_response)
        _t4 = time.time()
        send_chat_message(space, formatted)
        _t5 = time.time()
        print(f"[perf] send_chat_message: {_t5 - _t4:.2f}s")
        print(f"[perf] TOTAL: {_t5 - _t0:.2f}s")

    except Exception as e:
        traceback.print_exc()
        try:
            send_chat_message(space, f"sorry, something went wrong: {str(e)}")
        except Exception:
            print(f"Failed to send error message: {e}")


def _extract_all_task_proposals(response, user_message=""):
    """Extract ALL task action proposals from Gemini's response.
    Returns a list of action dicts. Supports bulk operations."""
    proposals = []

    # CREATE
    for match in re.finditer(
        r'\[CREATE_TASK\]\s*title="([^"]+)"(?:\s*due="([^"]*)")?(?:\s*notes="([^"]*)")?',
        response,
    ):
        task = {"action": "create", "title": match.group(1)}
        if match.group(2):
            task["due"] = match.group(2)
        if match.group(3):
            task["notes"] = match.group(3)
        proposals.append(task)

    # UPDATE
    for match in re.finditer(
        r'\[UPDATE_TASK\]\s*find="([^"]+)"(?:\s*title="([^"]*)")?(?:\s*due="([^"]*)")?(?:\s*notes="([^"]*)")?',
        response,
    ):
        task = {"action": "update", "find": match.group(1)}
        if match.group(2):
            task["new_title"] = match.group(2)
        if match.group(3):
            task["new_due"] = match.group(3)
        if match.group(4):
            task["new_notes"] = match.group(4)
        proposals.append(task)

    # COMPLETE
    for match in re.finditer(r'\[COMPLETE_TASK\]\s*find="([^"]+)"', response):
        proposals.append({"action": "complete", "find": match.group(1)})

    # DELETE
    for match in re.finditer(r'\[DELETE_TASK\]\s*find="([^"]+)"', response):
        proposals.append({"action": "delete", "find": match.group(1)})

    if proposals:
        return proposals

    if _user_has_task_intent(user_message):
        return _fallback_parse_prose(response)

    return None


def _user_has_task_intent(user_message):
    """Only use fallback prose parsing when the user's message clearly
    indicates they want to create, update, complete, or delete a task."""
    lower = user_message.lower()
    task_intent_patterns = [
        r'\b(?:add|create|make|set up|new)\b.*\btask',
        r'\btask\b.*\b(?:add|create|make)',
        r'\b(?:move|push|reschedule|change|update)\b.*\b(?:task|due|deadline)',
        r'\b(?:complete|finish|done|mark)\b.*\btask',
        r'\b(?:delete|remove)\b.*\btask',
        r'\bremind\s+me\b',
        r'\b(?:add|create)\b.*\bto.do\b',
        r'\b(?:move|push)\b.*\bto\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
    ]
    return any(re.search(p, lower) for p in task_intent_patterns)


def _fallback_parse_prose(response):
    """Fallback: detect task actions from Gemini's prose when it forgets tags.
    Looks for patterns like 'moved "X" to today' or 'added "X"'."""
    from datetime import datetime, timedelta

    lower = response.lower()
    proposals = []

    now = datetime.now()
    today_iso = now.strftime("%Y-%m-%d")
    tomorrow_iso = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    weekday_map = {}
    for i in range(1, 8):
        d = now + timedelta(days=i)
        weekday_map[d.strftime("%A").lower()] = d.strftime("%Y-%m-%d")

    def _resolve_date(text):
        t = text.lower().strip().rstrip(".")
        if t == "today":
            return today_iso
        if t == "tomorrow":
            return tomorrow_iso
        for day_name, iso in weekday_map.items():
            if day_name in t:
                return iso
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', t)
        if date_match:
            return date_match.group(1)
        return None

    for match in re.finditer(
        r'(?:moved|updated|pushed|changed|rescheduled)\s+"([^"]+)"\s+(?:to|due)\s+(\w+[\w\s,]*)',
        lower,
    ):
        title = match.group(1)
        date = _resolve_date(match.group(2))
        if date:
            orig_match = re.search(re.escape(match.group(1)), response, re.IGNORECASE)
            orig_title = orig_match.group(0) if orig_match else title
            proposals.append({"action": "update", "find": orig_title, "new_due": date})

    for match in re.finditer(
        r'(?:added|created)\s+"([^"]+)"(?:\s+(?:to your tasks?|for you))?(?:[,.]?\s*due\s+(\w+[\w\s,]*))?',
        lower,
    ):
        title = match.group(1)
        orig_match = re.search(re.escape(match.group(1)), response, re.IGNORECASE)
        orig_title = orig_match.group(0) if orig_match else title
        task = {"action": "create", "title": orig_title}
        if match.group(2):
            date = _resolve_date(match.group(2))
            if date:
                task["due"] = date
        proposals.append(task)

    for match in re.finditer(
        r'(?:marked|completed|finished|crossed off)\s+"([^"]+)"',
        lower,
    ):
        orig_match = re.search(re.escape(match.group(1)), response, re.IGNORECASE)
        orig_title = orig_match.group(0) if orig_match else match.group(1)
        proposals.append({"action": "complete", "find": orig_title})

    for match in re.finditer(
        r'(?:deleted|removed)\s+"([^"]+)"',
        lower,
    ):
        orig_match = re.search(re.escape(match.group(1)), response, re.IGNORECASE)
        orig_title = orig_match.group(0) if orig_match else match.group(1)
        proposals.append({"action": "delete", "find": orig_title})

    return proposals if proposals else None


def _remove_task_tags(response):
    """Remove all task action tags from the response shown to the user."""
    cleaned = re.sub(r'\[(CREATE|UPDATE|COMPLETE|DELETE)_TASK\][^\n]*\n?', '', response)
    return cleaned.strip()


def _execute_task_action(action_item):
    """Execute a single confirmed task action."""
    action = action_item.get("action", "create")
    if action == "create":
        return create_task(
            title=action_item["title"],
            notes=action_item.get("notes", ""),
            due_date=action_item.get("due"),
        )
    elif action == "update":
        return update_task(
            task_title=action_item["find"],
            new_title=action_item.get("new_title"),
            new_notes=action_item.get("new_notes"),
            new_due=action_item.get("new_due"),
        )
    elif action == "complete":
        return complete_task(task_title=action_item["find"])
    elif action == "delete":
        return delete_task(task_title=action_item["find"])
    return {"error": f"Unknown action: {action}"}


def _detect_bulk_task_action(user_message):
    """Detect bulk task operations like 'change all tasks to due today' and
    programmatically generate update actions for every open task.
    Skips tasks that already have the target due date."""
    from datetime import datetime, timedelta

    lower = user_message.lower().strip()

    all_patterns = [
        r'(?:change|move|push|set|update|reschedule)\s+(?:all|every|each)\s+(?:my\s+)?(?:pending\s+|open\s+)?tasks?\b',
        r'(?:make|set)\s+(?:all|every|each)\s+(?:my\s+)?(?:pending\s+|open\s+)?tasks?\s+(?:due|to)',
        r'(?:push|move)\s+everything\s+(?:to|due)',
    ]
    if not any(re.search(p, lower) for p in all_patterns):
        return None

    now = datetime.now()
    today_iso = now.strftime("%Y-%m-%d")
    tomorrow_iso = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    weekday_map = {}
    for i in range(1, 8):
        d = now + timedelta(days=i)
        weekday_map[d.strftime("%A").lower()] = d.strftime("%Y-%m-%d")

    target_date = None
    if "today" in lower:
        target_date = today_iso
    elif "tomorrow" in lower:
        target_date = tomorrow_iso
    else:
        for day_name, iso in weekday_map.items():
            if day_name in lower:
                target_date = iso
                break
        if not target_date:
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', lower)
            if date_match:
                target_date = date_match.group(1)

    if not target_date:
        return None

    try:
        tasks = fetch_open_tasks()
    except Exception:
        return None

    if not tasks:
        return None

    target_human = datetime.strptime(target_date, "%Y-%m-%d").strftime("%b %d, %Y")

    actions = []
    for t in tasks:
        if t.get("due") == target_human:
            continue
        if not t.get("is_overdue"):
            continue
        actions.append({
            "action": "update",
            "find": t["title"],
            "new_due": target_date,
        })

    return actions if actions else None


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
        "knowledge_graph": 10,
    }

    pool = ThreadPoolExecutor(max_workers=7)
    futures = {}
    futures["meetings"] = pool.submit(_timed_fetch, "meetings", _fetch_meetings)
    futures["tasks"] = pool.submit(_timed_fetch, "tasks", _fetch_tasks)
    if wants_emails or is_general:
        futures["emails"] = pool.submit(_timed_fetch, "emails", _fetch_unread_emails)
    if search_terms:
        futures["targeted_emails"] = pool.submit(_timed_fetch, "targeted_emails", _fetch_targeted_emails)
    if wants_granola:
        futures["granola"] = pool.submit(_timed_fetch, "granola", _fetch_granola)
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
                       "knowledge_graph": "knowledge graph"}
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
