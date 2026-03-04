"""Momo briefing + proactive email alert orchestrator."""

import json
import google.generativeai as genai
from gmail_service import (
    fetch_unread_client_emails,
    format_emails_for_context,
    fetch_email_alert_candidates,
)
from calendar_service import (
    fetch_todays_meetings,
    fetch_recently_ended_meetings,
    format_meetings_for_context,
)
from tasks_service import fetch_open_tasks, format_tasks_for_context
from gemini_service import generate_morning_briefing, generate_post_meeting_debrief
from chat_service import send_chat_message, format_for_google_chat
from conversation_store import (
    has_email_alert_been_sent,
    mark_email_alert_sent,
    has_debrief_been_sent,
    mark_debrief_sent,
)
import config


def run_morning_briefing():
    """Full morning briefing pipeline.
    Fetches emails, meetings, tasks, Granola notes, and nudges in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Momo is preparing the morning briefing...")

    def _fetch_emails():
        emails = fetch_unread_client_emails()
        print(f"     Found {len(emails)} unread client email(s)")
        return "emails", emails

    def _fetch_meetings():
        meetings = fetch_todays_meetings()
        print(f"     Found {len(meetings)} meeting(s)")
        return "meetings", meetings

    def _fetch_tasks():
        tasks = fetch_open_tasks()
        print(f"     Found {len(tasks)} open task(s)")
        return "tasks", tasks

    def _fetch_granola():
        try:
            from granola_service import fetch_yesterday_meeting_notes, format_granola_notes_for_context
            raw_notes = fetch_yesterday_meeting_notes()
            ctx = format_granola_notes_for_context(raw_notes)
            print(f"     Granola notes loaded ({len(ctx)} chars)")
            return "granola", ctx
        except Exception as e:
            print(f"     Granola fetch failed: {e}")
            return "granola", ""

    def _fetch_nudges():
        try:
            from proactive_intelligence import generate_daily_nudges
            ctx = generate_daily_nudges()
            if ctx:
                print(f"     Nudges generated ({len(ctx)} chars)")
            else:
                print("     No nudges to report")
            return "nudges", ctx or ""
        except Exception as e:
            print(f"     Proactive intelligence failed: {e}")
            return "nudges", ""

    futures = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures["emails"] = pool.submit(_fetch_emails)
        futures["meetings"] = pool.submit(_fetch_meetings)
        futures["tasks"] = pool.submit(_fetch_tasks)
        if config.GRANOLA_ENABLED:
            futures["granola"] = pool.submit(_fetch_granola)
        if config.PROACTIVE_INTELLIGENCE_ENABLED and config.KNOWLEDGE_GRAPH_ENABLED:
            futures["nudges"] = pool.submit(_fetch_nudges)

    data = {}
    for key, future in futures.items():
        try:
            label, value = future.result(timeout=120)
            data[label] = value
        except Exception as e:
            print(f"  Error fetching {key}: {e}")
            data[key] = [] if key in ("emails", "meetings", "tasks") else ""

    emails = data.get("emails", [])
    meetings = data.get("meetings", [])
    tasks = data.get("tasks", [])
    granola_ctx = data.get("granola", "")
    nudges_ctx = data.get("nudges", "")

    if not emails and not meetings and not tasks and not granola_ctx and not nudges_ctx:
        print("  Nothing to report. Skipping.")
        return {"status": "skipped", "reason": "nothing to report"}

    emails_ctx = format_emails_for_context(emails)
    meetings_ctx = format_meetings_for_context(meetings)
    tasks_ctx = format_tasks_for_context(tasks)

    print("  Generating briefing with Gemini...")
    summary = generate_morning_briefing(
        emails_ctx, meetings_ctx, tasks_ctx,
        granola_context=granola_ctx, nudges_context=nudges_ctx,
    )

    if config.CHAT_SPACE_ID:
        print("  Sending to Google Chat...")
        formatted = format_for_google_chat(summary)
        send_chat_message(config.CHAT_SPACE_ID, formatted)
    else:
        print("  No CHAT_SPACE_ID configured. Printing to console:")
        print(summary)

    print("Momo's morning briefing delivered.")
    return {
        "status": "sent",
        "emails": len(emails),
        "meetings": len(meetings),
        "tasks": len(tasks),
    }


def run_proactive_email_alerts():
    """Notify user when a new client/important email arrives.
    Uses Gemini to triage emails the same way Momo would in conversation."""
    if not config.EMAIL_ALERTS_ENABLED:
        return {"status": "skipped", "reason": "email alerts disabled"}
    if not config.CHAT_SPACE_ID:
        return {"status": "skipped", "reason": "CHAT_SPACE_ID not configured"}

    emails = fetch_email_alert_candidates()
    unseen = [e for e in emails if not has_email_alert_been_sent(e["id"])]

    if not unseen:
        return {"status": "no_alerts", "alerts_sent": 0, "checked": len(emails)}

    # Batch up to 10 unseen emails for a single Gemini triage call
    batch = unseen[: config.EMAIL_ALERTS_MAX_PER_RUN * 2]
    triage_results = _gemini_triage_emails(batch)
    sent_count = 0

    for result in triage_results:
        if sent_count >= config.EMAIL_ALERTS_MAX_PER_RUN:
            break

        email = result["email"]
        if has_email_alert_been_sent(email["id"]):
            continue

        message = _format_email_alert_message(
            email, result["reason"], result["summary"], result["priority"]
        )
        formatted = format_for_google_chat(message)
        send_chat_message(config.CHAT_SPACE_ID, formatted)
        mark_email_alert_sent(email)
        sent_count += 1

        from knowledge_graph import extract_and_store_background
        extract_and_store_background(
            source_type="email",
            source_id=email["id"],
            source_title=email.get("subject", ""),
            source_date=email.get("date_human", ""),
            content=email.get("body", ""),
            attendees=[email.get("from", "")],
        )

    return {
        "status": "sent" if sent_count else "no_alerts",
        "alerts_sent": sent_count,
        "checked": len(emails),
    }


_TRIAGE_PROMPT = """You are an email triage assistant. Your job is to decide which emails are important enough to proactively notify someone about.

An email is worth alerting on if it's:
- From a client, partner, or external stakeholder (not internal newsletters, automated notifications, marketing, or system alerts)
- Requires action or a timely response
- Contains urgent or time-sensitive information (deadlines, escalations, blockers)
- Is from a real person about something that matters (not spam, promotions, or automated digests)

An email is NOT worth alerting on if it's:
- Automated notifications (JIRA, GitHub, Slack digests, CI/CD, monitoring)
- Marketing, newsletters, or promotional emails
- Internal FYI or low-priority updates
- Calendar invites or routine scheduling
- Anything the user would not want to be interrupted for

For each email below, respond with a JSON array. Each element should be:
{"id": "<email_id>", "alert": true/false, "priority": "high"/"medium", "reason": "<1 short phrase>", "summary": "<1-2 sentence summary>"}

Only set "alert": true for emails that genuinely deserve an interruption. Be selective — when in doubt, don't alert.

EMAILS:
"""


def _gemini_triage_emails(emails):
    """Use Gemini to decide which emails deserve a proactive alert."""
    genai.configure(api_key=config.GEMINI_API_KEY)

    email_block = ""
    for i, e in enumerate(emails):
        body_preview = (e.get("body", "") or "")[:800]
        email_block += (
            f"\n--- Email {i+1} (id={e['id']}) ---\n"
            f"From: {e.get('from', 'Unknown')}\n"
            f"Subject: {e.get('subject', '(no subject)')}\n"
            f"Date: {e.get('date_human', '')}\n"
            f"Labels: {', '.join(e.get('labels', []))}\n"
            f"Body preview:\n{body_preview}\n"
        )

    prompt = _TRIAGE_PROMPT + email_block

    model = genai.GenerativeModel(model_name=config.GEMINI_MODEL)

    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        results = json.loads(text.strip())
    except Exception as exc:
        print(f"Gemini triage failed: {exc}")
        return []

    email_map = {e["id"]: e for e in emails}
    flagged = []
    for item in results:
        if not item.get("alert"):
            continue
        email = email_map.get(item.get("id"))
        if not email:
            continue
        flagged.append({
            "email": email,
            "reason": item.get("reason", "flagged by momo"),
            "summary": item.get("summary", ""),
            "priority": item.get("priority", "medium"),
        })

    return flagged


def _process_debrief_tasks(debrief_text, meeting_title=""):
    """Parse [CREATE_TASK] tags from a debrief, store them as pending
    proposals for user confirmation, and return cleaned text with a
    nicely formatted suggestion section replacing the raw tags."""
    import re
    from conversation_store import store_pending_tasks

    pattern = r'\[CREATE_TASK\]\s*title="([^"]+)"(?:\s*due="([^"]*)")?(?:\s*notes="([^"]*)")?'
    matches = list(re.finditer(pattern, debrief_text))
    if not matches:
        return debrief_text

    cleaned = re.sub(r'\[CREATE_TASK\][^\n]*\n?', '', debrief_text).rstrip()

    pending = []
    suggestion_lines = []
    for match in matches:
        title = match.group(1)
        due = match.group(2) or None
        notes = match.group(3) or ""
        task = {"title": title}
        if due:
            task["due"] = due
        if notes:
            task["notes"] = notes
        pending.append(task)
        due_str = f" (due {due})" if due else ""
        suggestion_lines.append(f"  • {title}{due_str}")

    store_pending_tasks(pending, meeting_title=meeting_title)

    cleaned += "\n\n📋 *Suggested tasks:*\n" + "\n".join(suggestion_lines)
    cleaned += "\n\n_Reply *yes* to create these tasks_"

    return cleaned


def run_post_meeting_debrief():
    """Check for recently ended meetings and send short debriefs with Granola notes.

    Notes-gated: a debrief is only sent once Granola notes are available.
    Meetings that run over their scheduled time are handled naturally — the
    lookback window is wide enough to keep retrying on subsequent scheduler
    runs until the notes appear.

    If Granola itself is erroring (auth, network, etc.) the debrief is sent
    without notes after MEETING_DEBRIEF_GRACE_MINUTES so the user isn't left
    with nothing.
    """
    if not config.GRANOLA_ENABLED:
        return {"status": "skipped", "reason": "granola disabled"}
    if not config.CHAT_SPACE_ID:
        return {"status": "skipped", "reason": "CHAT_SPACE_ID not configured"}

    lookback = config.MEETING_DEBRIEF_LOOKBACK_MINUTES
    grace = config.MEETING_DEBRIEF_GRACE_MINUTES
    ended = fetch_recently_ended_meetings(lookback_minutes=lookback)

    if not ended:
        return {"status": "no_meetings", "debriefs_sent": 0}

    from granola_service import build_meeting_id_map, match_meeting_id, fetch_meeting_notes_batch
    from datetime import datetime

    now = datetime.now().astimezone()
    sent_count = 0
    deferred_count = 0

    min_wait = config.MEETING_DEBRIEF_MIN_WAIT_MINUTES
    pending = []
    for meeting in ended:
        event_id = meeting.get("id", "")
        if not event_id or has_debrief_been_sent(event_id):
            continue
        try:
            end_dt = datetime.fromisoformat(meeting["end_iso"])
            minutes_since_end = (now - end_dt).total_seconds() / 60
        except (ValueError, TypeError):
            minutes_since_end = 0
        if minutes_since_end < min_wait:
            print(f"  Skipping {meeting['title']}: only {minutes_since_end:.0f}m since end, waiting at least {min_wait}m")
            continue
        pending.append((meeting, minutes_since_end))

    if not pending:
        return {"status": "no_debriefs", "debriefs_sent": 0}

    # Single list_meetings call → title-to-ID map
    granola_error = None
    id_map: dict[str, str] = {}
    try:
        id_map = build_meeting_id_map()
    except Exception as e:
        print(f"  Granola list_meetings failed: {e}")
        granola_error = str(e)

    # Match titles → IDs, then batch-fetch notes in one get_meetings call
    id_to_meeting: dict[str, tuple] = {}
    no_match: list[tuple] = []
    for meeting, mins in pending:
        title = meeting["title"]
        mid = match_meeting_id(title, id_map) if id_map else None
        if mid:
            id_to_meeting[mid] = (meeting, mins)
        else:
            no_match.append((meeting, mins))

    notes_by_id: dict[str, str] = {}
    if id_to_meeting:
        try:
            notes_by_id = fetch_meeting_notes_batch(list(id_to_meeting.keys()))
        except Exception as e:
            print(f"  Granola get_meetings failed: {e}")
            granola_error = str(e)

    # Process meetings with matched IDs
    for mid, (meeting, minutes_since_end) in id_to_meeting.items():
        title = meeting["title"]
        attendees = [a["name"] for a in meeting.get("attendees", [])]
        granola_notes = notes_by_id.get(mid, "")

        if granola_notes:
            import re as _re
            stripped = _re.sub(r'<[^>]+>', '', granola_notes).strip()
            if len(stripped) < 20:
                granola_notes = ""

        print(f"  Checking debrief for: {title} (scheduled end was {minutes_since_end:.0f}m ago)")

        if not granola_notes:
            if granola_error and minutes_since_end >= grace:
                print(f"    Granola unavailable past grace window ({grace}m), sending without notes")
            else:
                reason = "Granola error, retrying" if granola_error else "notes not available yet"
                print(f"    {reason}, deferring to next run")
                deferred_count += 1
                continue

        try:
            end_time = meeting.get("end_time", "")
            event_id = meeting.get("id", "")
            debrief = generate_post_meeting_debrief(title, attendees, granola_notes, end_time)
            debrief = _process_debrief_tasks(debrief, meeting_title=title)
            formatted = format_for_google_chat(debrief)
            send_chat_message(config.CHAT_SPACE_ID, formatted)
            mark_debrief_sent(event_id, title)
            sent_count += 1
            print(f"    Debrief sent for: {title}")
            if granola_notes:
                from knowledge_graph import extract_and_store_background
                extract_and_store_background(
                    source_type="meeting",
                    source_id=event_id,
                    source_title=title,
                    source_date=now.strftime("%Y-%m-%d"),
                    content=granola_notes,
                    attendees=attendees,
                )
        except Exception as e:
            print(f"    Debrief generation/send failed for '{title}': {e}")

    # Meetings with no Granola match — defer or send without notes
    for meeting, minutes_since_end in no_match:
        title = meeting["title"]
        attendees = [a["name"] for a in meeting.get("attendees", [])]
        print(f"  Checking debrief for: {title} (scheduled end was {minutes_since_end:.0f}m ago)")

        if granola_error and minutes_since_end >= grace:
            print(f"    Granola unavailable past grace window ({grace}m), sending without notes")
            try:
                end_time = meeting.get("end_time", "")
                debrief = generate_post_meeting_debrief(title, attendees, "", end_time)
                debrief = _process_debrief_tasks(debrief, meeting_title=title)
                formatted = format_for_google_chat(debrief)
                send_chat_message(config.CHAT_SPACE_ID, formatted)
                mark_debrief_sent(meeting.get("id", ""), title)
                sent_count += 1
                print(f"    Debrief sent for: {title}")
            except Exception as e:
                print(f"    Debrief generation/send failed for '{title}': {e}")
        else:
            print(f"    No Granola match for '{title}', deferring to next run")
            deferred_count += 1

    return {
        "status": "sent" if sent_count else ("deferred" if deferred_count else "no_debriefs"),
        "debriefs_sent": sent_count,
        "deferred": deferred_count,
    }


def _format_email_alert_message(email, reason, summary, priority):
    priority_icon = "🔴" if priority == "high" else "🟡"
    lines = [
        f"{priority_icon} *new email needs your attention*",
        f"- *From:* {email.get('from', 'Unknown')}",
        f"- *Subject:* {email.get('subject', '(no subject)')}",
        f"- *Why:* {reason}",
    ]
    if summary:
        lines.append(f"- *TLDR:* {summary}")
    lines.append("ask me to `summarize this email` or `draft a reply` if you need more.")
    return "\n".join(lines)
