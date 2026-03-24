"""Proactive Intelligence Engine — surfaces insights before the user asks.

Four engines:
  1. Pre-Meeting Prep  — briefs you before upcoming meetings with KG context
  2. Commitment Follow-Up — flags unfulfilled commitments with cross-referencing
  3. Pattern Detection — spots recurring topics, frequent collaborators
  4. Drift Detection — flags stale projects and aging open items

Coordinator functions:
  run_meeting_prep()     — called by /meeting-prep endpoint (every ~10 min)
  generate_daily_nudges() — called during morning briefing (daily)
"""

import hashlib
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import google.generativeai as genai

import config
from langsmith_config import traceable, traced_generate_content
from calendar_service import fetch_upcoming_meetings
from chat_service import format_for_google_chat, send_chat_message
from conversation_store import (
    add_turn,
    conversation_scope,
    has_nudge_been_sent,
    has_prep_been_sent,
    mark_nudge_sent,
    mark_prep_sent,
)
from knowledge_graph import (
    format_knowledge_for_context,
    query_all_entries,
    query_by_person,
    query_by_project,
    query_open_by_age,
    query_recent,
    update_entity_status,
)

genai.configure(api_key=config.GEMINI_API_KEY)


def _store_proactive_message(message: str, space_id: str) -> None:
    """Persist a proactive assistant message into the matching chat history."""
    if not message or not space_id:
        return
    try:
        add_turn(conversation_scope(space=space_id), "assistant", message)
    except Exception as exc:
        print(f"  Failed to store proactive message in conversation history: {exc}")


def _nudge_key(nudge_type: str, identifier: str) -> str:
    """Deterministic key for dedup. Hashed so Firestore doc IDs stay clean."""
    raw = f"{nudge_type}:{identifier}".lower().strip()
    return hashlib.md5(raw.encode()).hexdigest()[:16]


_DRIFT_ACTIVITY_SOURCES = {"meeting", "meeting_notes", "email"}


def _is_drift_activity_entry(entry: dict) -> bool:
    return entry.get("source_type") in _DRIFT_ACTIVITY_SOURCES


def _has_recent_activity(entry: dict, recent_entries: list[dict]) -> bool:
    """Return True if the item showed up in recent meeting/email activity."""
    target_name = (entry.get("name") or "").strip().lower()
    target_projects = {
        project.strip().lower()
        for project in entry.get("related_projects", [])
        if project
    }
    if not target_name and not target_projects:
        return False

    for other in recent_entries:
        if other.get("id") == entry.get("id"):
            continue

        if target_name and (other.get("name") or "").strip().lower() == target_name:
            return True

        other_projects = {
            project.strip().lower()
            for project in other.get("related_projects", [])
            if project
        }
        if target_projects and target_projects & other_projects:
            return True

    return False


# ── Engine 1: Pre-Meeting Prep ───────────────────────────────


_PREP_PROMPT = """You are Momo, preparing a quick pre-meeting intel brief. Be casual, concise, and useful.

Meeting: {title}
Attendees: {attendees}
Starts: {start_time}

Here is everything Momo knows about the people and topics involved (from past meetings, emails, and conversations):

{knowledge_context}

Write a short pre-meeting prep (3-6 bullet points max). Include:
- Key context about the attendees from past interactions
- Any open commitments or action items involving these people
- Relevant decisions or blockers from previous meetings
- Anything the user should be prepared to discuss

If there's very little context, just say so briefly — don't pad it out.
Format for Google Chat: use *bold* for names and topics, bullet points for items.
Start with: 📋 *meeting prep — {title}*"""


def _build_meeting_prep(meeting: dict) -> str | None:
    """Gather KG context for a meeting and generate a prep brief via Gemini."""
    attendee_names = [a["name"] for a in meeting.get("attendees", [])]
    if not attendee_names:
        return None

    all_entries = []
    seen_ids = set()
    title = meeting.get("title", "")

    # Query KG in parallel: by each attendee AND by meeting title (semantic search)
    from knowledge_graph import semantic_search as _semantic_search
    workers = max(len(attendee_names) + 1, 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        person_futures = {
            pool.submit(query_by_person, attendee, None, 8): f"person:{attendee}"
            for attendee in attendee_names
        }
        # Also search by meeting title to catch topic-based KG entries
        title_future = pool.submit(_semantic_search, title, 10)
        person_futures[title_future] = f"title:{title}"

        for future in as_completed(person_futures):
            label = person_futures[future]
            try:
                for entry in future.result():
                    if entry["id"] not in seen_ids:
                        seen_ids.add(entry["id"])
                        all_entries.append(entry)
            except Exception as exc:
                print(f"    KG query failed ({label}): {exc}")

    # Also query by projects found in existing results
    projects = set()
    for e in all_entries:
        projects.update(e.get("related_projects", []))
    project_list = list(projects)[:3]

    if project_list:
        with ThreadPoolExecutor(max_workers=len(project_list)) as pool:
            proj_futures = {
                pool.submit(query_by_project, proj, None, 5): proj
                for proj in project_list
            }
            for future in as_completed(proj_futures):
                try:
                    for entry in future.result():
                        if entry["id"] not in seen_ids:
                            seen_ids.add(entry["id"])
                            all_entries.append(entry)
                except Exception as exc:
                    print(f"    KG project query failed: {exc}")

    attendees_str = ", ".join(attendee_names)

    if not all_entries:
        # No KG context — generate a minimal prep with just attendee + time info
        knowledge_context = "(No prior context found for these attendees or topics.)"
    else:
        knowledge_context = format_knowledge_for_context(all_entries[:20])

    prompt = _PREP_PROMPT.format(
        title=meeting["title"],
        attendees=attendees_str,
        start_time=meeting.get("start_time", "soon"),
        knowledge_context=knowledge_context,
    )

    model = genai.GenerativeModel(model_name=config.GEMINI_MODEL_FLASH)
    try:
        resp = traced_generate_content(model, prompt, model_name=config.GEMINI_MODEL_FLASH)
        return resp.text.strip()
    except Exception as exc:
        print(f"  Meeting prep generation failed: {exc}")
        return None


def run_meeting_prep() -> dict:
    """Check for upcoming meetings and send prep briefs for unsent ones.

    Only creates a LangSmith trace when there are actual meetings to prep,
    so idle polling runs don't flood the trace dashboard.
    """
    if not config.PROACTIVE_INTELLIGENCE_ENABLED or not config.MEETING_PREP_ENABLED:
        return {"status": "skipped", "reason": "meeting prep disabled"}
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return {"status": "skipped", "reason": "knowledge graph disabled"}
    if not config.CHAT_SPACE_ID:
        return {"status": "skipped", "reason": "CHAT_SPACE_ID not configured"}

    upcoming = fetch_upcoming_meetings(hours=config.MEETING_PREP_LOOKAHEAD_HOURS)
    if not upcoming:
        return {"status": "no_meetings", "preps_sent": 0}

    # Filter to meetings that actually need prep
    meetings_to_prep = []
    for meeting in upcoming:
        event_id = meeting.get("id", "")
        if not event_id or meeting.get("is_all_day"):
            continue
        if has_prep_been_sent(event_id):
            continue
        meetings_to_prep.append(meeting)

    if not meetings_to_prep:
        return {"status": "no_preps", "preps_sent": 0}

    # Only trace when we're doing real work
    return _run_meeting_prep_traced(meetings_to_prep)


@traceable(name="meeting-prep", tags=["proactive", "scheduled"])
def _run_meeting_prep_traced(meetings: list) -> dict:
    """Traced inner function — only called when there are meetings to prep."""
    sent_count = 0
    for meeting in meetings:
        event_id = meeting.get("id", "")
        print(f"  Generating meeting prep for: {meeting['title']}")
        try:
            brief = _build_meeting_prep(meeting)
            if brief:
                formatted = format_for_google_chat(brief)
                send_chat_message(config.CHAT_SPACE_ID, formatted)
                _store_proactive_message(brief, config.CHAT_SPACE_ID)
                mark_prep_sent(event_id, meeting["title"])
                sent_count += 1
                print(f"    Prep sent for: {meeting['title']}")
            else:
                print(f"    No KG context for: {meeting['title']}, will retry on next run")
        except Exception as exc:
            print(f"    Prep failed for '{meeting['title']}': {exc}")
            traceback.print_exc()

    return {"status": "sent" if sent_count else "no_preps", "preps_sent": sent_count}


# ── Engine 2: Commitment Follow-Up ──────────────────────────


_EVIDENCE_PROMPT = """Does this email provide evidence that the following commitment was fulfilled?

COMMITMENT: {commitment}

EMAIL:
From: {sender}
Subject: {subject}
Body: {body}

Answer with ONLY "yes" or "no". "yes" means the email clearly shows the commitment was completed (e.g. the thing was sent, delivered, finished). "no" means the email is unrelated or doesn't prove completion."""


def _check_commitment_evidence(commitment: dict) -> str | None:
    """Cross-reference a commitment against Gmail and Tasks for evidence of completion.
    Returns a reason string if evidence found, else None."""
    name = commitment.get("name", "")
    content = commitment.get("content", "")
    search_terms = name if len(name) > 3 else content[:50]

    try:
        from gmail_service import search_emails
        emails = search_emails(search_terms, days_back=30, max_results=3)
        if emails:
            model = genai.GenerativeModel(model_name=config.GEMINI_MODEL_FLASH)
            commitment_desc = f"{name}: {content}"
            for email in emails:
                prompt = _EVIDENCE_PROMPT.format(
                    commitment=commitment_desc,
                    sender=email.get("from", "?"),
                    subject=email.get("subject", "?"),
                    body=(email.get("body", "") or "")[:500],
                )
                try:
                    resp = traced_generate_content(model, prompt, model_name=config.GEMINI_MODEL_FLASH)
                    if resp.text.strip().lower().startswith("yes"):
                        return f"Found matching email: {email.get('subject', '?')}"
                except Exception:
                    pass
    except Exception:
        pass

    try:
        from tasks_service import find_completed_task
        result = find_completed_task(name, days_back=30)
        if result:
            return f"Matching task '{result['title']}' is completed"
    except Exception:
        pass

    return None


def _run_commitment_engine() -> list[dict]:
    """Find overdue open commitments, cross-reference for evidence, return nudges."""
    overdue = query_open_by_age(min_days=config.COMMITMENT_FOLLOWUP_DAYS, limit=20)
    if not overdue:
        return []

    nudges = []
    for entry in overdue:
        nudge_id = _nudge_key("commitment", entry.get("id", entry.get("name", "")))
        if has_nudge_been_sent(nudge_id):
            continue

        evidence = _check_commitment_evidence(entry)
        if evidence:
            try:
                update_entity_status(entry["id"], "resolved")
                print(f"    Auto-resolved commitment '{entry.get('name')}': {evidence}")
            except Exception:
                pass
            continue

        source_date = entry.get("source_date", "?")
        try:
            days_ago = (datetime.now() - datetime.strptime(source_date, "%Y-%m-%d")).days
        except (ValueError, TypeError):
            days_ago = config.COMMITMENT_FOLLOWUP_DAYS

        owner = entry.get("owner") or "you"
        source = entry.get("source_title", "a meeting")
        priority = "high" if days_ago > config.COMMITMENT_FOLLOWUP_DAYS * 2 else "medium"

        nudges.append({
            "type": "commitment",
            "priority": priority,
            "title": entry.get("name", "Unnamed commitment"),
            "body": (
                f"{days_ago} days ago, {owner} committed to: {entry.get('content', entry.get('name', '?'))} "
                f"(from: {source}). no matching sent email or completed task found."
            ),
            "related_entity_ids": [entry.get("id", "")],
            "delivery": "both" if priority == "high" else "briefing",
            "_nudge_key": nudge_id,
        })

    return nudges


# ── Engine 3: Pattern Detection ──────────────────────────────

_PATTERN_PROMPT = """You are Momo, analyzing patterns in recent workplace activity. Be casual, insightful, and concise.

Here are patterns detected from the last 30 days of meetings, emails, and conversations:

{patterns}

Generate 1-3 short, actionable insights based on these patterns. Each insight should be 1-2 sentences.
Focus on things like:
- Recurring topics that might need a dedicated discussion
- People who keep coming up together (potential collaboration opportunities)
- Topics evolving from discussion to decision to blocker (trajectory)

If the patterns aren't interesting enough to mention, return exactly: NO_INSIGHTS
Otherwise, return just the insights as bullet points (- ), no headers, no preamble."""


def _run_pattern_engine() -> list[dict]:
    """Analyze recent KG entries for recurring patterns."""
    entries = query_recent(days=30, limit=300)
    if len(entries) < 5:
        return []

    people_counter: Counter = Counter()
    project_counter: Counter = Counter()
    tag_counter: Counter = Counter()
    project_types: dict[str, list[str]] = defaultdict(list)

    for e in entries:
        for person in e.get("related_people", []):
            people_counter[person] += 1
        for project in e.get("related_projects", []):
            project_counter[project] += 1
            project_types[project].append(e.get("entity_type", "topic"))
        for tag in e.get("tags", []):
            tag_counter[tag] += 1

    pattern_lines = []

    frequent_people = [(p, c) for p, c in people_counter.most_common(5) if c >= 3]
    if frequent_people:
        pattern_lines.append("Frequent people across meetings/emails:")
        for person, count in frequent_people:
            pattern_lines.append(f"  - {person}: mentioned in {count} entries")

    hot_projects = [(p, c) for p, c in project_counter.most_common(5) if c >= 3]
    if hot_projects:
        pattern_lines.append("Hot projects/topics:")
        for proj, count in hot_projects:
            types = project_types.get(proj, [])
            type_summary = ", ".join(f"{t}({types.count(t)})" for t in set(types))
            pattern_lines.append(f"  - {proj}: {count} mentions ({type_summary})")

    hot_tags = [(t, c) for t, c in tag_counter.most_common(8) if c >= 3]
    if hot_tags:
        pattern_lines.append("Recurring keywords:")
        for tag, count in hot_tags:
            pattern_lines.append(f"  - {tag}: {count} mentions")

    if not pattern_lines:
        return []

    nudge_id = _nudge_key("pattern", "\n".join(pattern_lines))
    if has_nudge_been_sent(nudge_id):
        return []

    model = genai.GenerativeModel(model_name=config.GEMINI_MODEL_FLASH)
    try:
        prompt = _PATTERN_PROMPT.format(patterns="\n".join(pattern_lines))
        resp = traced_generate_content(model, prompt, model_name=config.GEMINI_MODEL_FLASH)
        text = resp.text.strip()
    except Exception as exc:
        print(f"  Pattern insight generation failed: {exc}")
        return []

    if text == "NO_INSIGHTS" or not text:
        return []

    return [{
        "type": "pattern",
        "priority": "low",
        "title": "Patterns from the last 30 days",
        "body": text,
        "related_entity_ids": [],
        "delivery": "briefing",
        "_nudge_key": nudge_id,
    }]


# ── Engine 4: Drift Detection ───────────────────────────────


def _run_drift_engine() -> list[dict]:
    """Flag open items and projects with no recent activity."""
    threshold = config.DRIFT_THRESHOLD_DAYS
    cutoff = (datetime.now() - timedelta(days=threshold)).strftime("%Y-%m-%d")

    stale_commitments = query_open_by_age(min_days=threshold, limit=30)
    recent_activity = [
        entry
        for entry in query_recent(days=threshold, limit=500)
        if _is_drift_activity_entry(entry)
    ]

    project_last_seen: dict[str, tuple[str, str]] = {}
    for e in query_all_entries(limit=5000):
        if not _is_drift_activity_entry(e):
            continue

        source_date = e.get("source_date", "")
        if not source_date:
            continue

        for proj in e.get("related_projects", []):
            normalized = proj.strip().lower()
            if not normalized:
                continue
            _, existing_date = project_last_seen.get(normalized, (proj, ""))
            if source_date > existing_date:
                project_last_seen[normalized] = (proj, source_date)

    stale_projects = [
        (proj, last_date)
        for proj, last_date in project_last_seen.values()
        if last_date <= cutoff
    ]

    nudges = []

    for entry in stale_commitments[:5]:
        if _has_recent_activity(entry, recent_activity):
            continue

        nudge_id = _nudge_key("drift_commitment", entry.get("id", ""))
        if has_nudge_been_sent(nudge_id):
            continue

        source_date = entry.get("source_date", "?")
        try:
            days_ago = (datetime.now() - datetime.strptime(source_date, "%Y-%m-%d")).days
        except (ValueError, TypeError):
            days_ago = threshold

        nudges.append({
            "type": "drift",
            "priority": "medium",
            "title": entry.get("name", "Unnamed item"),
            "body": (
                f"this {entry.get('entity_type', 'item')} has been open for {days_ago} days "
                f"and hasn't shown up in recent meeting/email activity "
                f"(from: {entry.get('source_title', '?')}). still active?"
            ),
            "related_entity_ids": [entry.get("id", "")],
            "delivery": "briefing",
            "_nudge_key": nudge_id,
        })

    for proj, last_date in stale_projects[:3]:
        nudge_id = _nudge_key("drift_project", proj)
        if has_nudge_been_sent(nudge_id):
            continue

        try:
            days_ago = (datetime.now() - datetime.strptime(last_date, "%Y-%m-%d")).days
        except (ValueError, TypeError):
            days_ago = threshold

        nudges.append({
            "type": "drift",
            "priority": "low",
            "title": f"{proj} — gone quiet",
            "body": (
                f"the '{proj}' project hasn't shown up in meetings or emails "
                f"for {days_ago} days. is this still active?"
            ),
            "related_entity_ids": [],
            "delivery": "briefing",
            "_nudge_key": nudge_id,
        })

    return nudges


# ── Coordinators ─────────────────────────────────────────────


@traceable(name="daily-nudges", tags=["proactive", "scheduled"])
def generate_daily_nudges() -> str:
    """Run commitment, pattern, and drift engines. Returns formatted text
    for inclusion in the morning briefing, or empty string if nothing to report."""
    if not config.PROACTIVE_INTELLIGENCE_ENABLED:
        return ""
    if not config.KNOWLEDGE_GRAPH_ENABLED:
        return ""

    all_nudges = []

    engines = {
        "commitment": _run_commitment_engine,
        "pattern": _run_pattern_engine,
        "drift": _run_drift_engine,
    }

    print("  Proactive intelligence: running all engines in parallel...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_to_name = {pool.submit(fn): name for name, fn in engines.items()}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                all_nudges.extend(future.result())
            except Exception as exc:
                print(f"    {name} engine failed: {exc}")
                traceback.print_exc()

    if not all_nudges:
        print("  Proactive intelligence: no nudges to report")
        return ""

    for nudge in all_nudges:
        key = nudge.get("_nudge_key", "")
        if key:
            mark_nudge_sent(key, nudge["type"], nudge["title"])

    standalone = [n for n in all_nudges if n["delivery"] in ("standalone", "both")]
    if standalone and config.CHAT_SPACE_ID:
        _send_standalone_nudges(standalone)

    briefing_nudges = [n for n in all_nudges if n["delivery"] in ("briefing", "both")]
    if not briefing_nudges:
        return ""

    return _format_nudges_for_briefing(briefing_nudges)


def _send_standalone_nudges(nudges: list[dict]):
    """Send high-priority nudges as standalone Chat messages."""
    lines = ["🔔 *momo's nudges*", ""]
    for n in nudges:
        priority_icon = "🔴" if n["priority"] == "high" else "🟡"
        lines.append(f"{priority_icon} *{n['title']}*")
        lines.append(f"  {n['body']}")
        lines.append("")

    text = "\n".join(lines).strip()
    try:
        formatted = format_for_google_chat(text)
        send_chat_message(config.CHAT_SPACE_ID, formatted)
        _store_proactive_message(text, config.CHAT_SPACE_ID)
    except Exception as exc:
        print(f"  Failed to send standalone nudges: {exc}")


def _format_nudges_for_briefing(nudges: list[dict]) -> str:
    """Format nudges into a text block for inclusion in the morning briefing prompt."""
    sections: dict[str, list[dict]] = defaultdict(list)
    for n in nudges:
        sections[n["type"]].append(n)

    lines = []

    if sections.get("commitment"):
        lines.append("OPEN COMMITMENTS NEEDING FOLLOW-UP:")
        for n in sections["commitment"]:
            priority_icon = "🔴" if n["priority"] == "high" else "🟡"
            lines.append(f"  {priority_icon} {n['title']}: {n['body']}")

    if sections.get("pattern"):
        lines.append("")
        lines.append("PATTERNS & INSIGHTS:")
        for n in sections["pattern"]:
            lines.append(f"  {n['body']}")

    if sections.get("drift"):
        lines.append("")
        lines.append("STALE ITEMS / GONE QUIET:")
        for n in sections["drift"]:
            lines.append(f"  🟡 {n['title']}: {n['body']}")

    return "\n".join(lines)
