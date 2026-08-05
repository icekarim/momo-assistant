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
import re
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import config
from observability import observe
from claude_client import generate, extract_text, TaskComplexity
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
    get_canonical_aliases,
    query_all_entries,
    query_by_person,
    query_by_project,
    query_open_by_age,
    query_recent,
    resolve_canonical,
    stable_key,
    update_entity_status,
)



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

Relevance rules:
- Only include context clearly connected to the non-owner attendees or this meeting's stated topic.
- Ignore context connected only to the user themselves — their own unrelated projects and history do not belong in this prep.
- Prefer recent information, and always show the date for anything older than ~3 months.
- For intro meetings, or attendees with no history, say plainly that there's no prior context — never pad the brief with unrelated projects.

If there's very little context, just say so briefly — don't pad it out.
Format for Google Chat: use *bold* for names and topics, bullet points for items.
Do NOT write a header or title line — output only the bullet points (and an optional short closing line)."""


def _name_tokens(name: str) -> set[str]:
    """Lowercase alphanumeric tokens of a name. For emails only the local part
    is tokenized (domains like 'gmail.com' would cause false owner matches)."""
    local = name.split("@", 1)[0] if "@" in name else name
    return {t for t in re.findall(r"[a-z0-9]+", local.lower()) if len(t) >= 2}


def _owner_identity_tokens() -> set[str]:
    """Token set identifying the owner: OWNER_NAME plus any canonical aliases
    identity resolution knows about. Empty set when OWNER_NAME is unset."""
    owner = (config.OWNER_NAME or "").strip()
    if not owner:
        return set()
    names = [owner]
    try:
        names.extend(get_canonical_aliases(owner))
    except Exception as exc:
        print(f"    Owner alias lookup failed: {exc}")
    tokens: set[str] = set()
    for candidate in names:
        tokens.update(_name_tokens(candidate))
    return tokens


def _matches_owner(name: str, owner_tokens: set[str]) -> bool:
    """Case-insensitive token overlap with the owner's identity
    (e.g. 'Karim' matches owner 'Karim X')."""
    return bool(owner_tokens) and bool(_name_tokens(name) & owner_tokens)


def _entry_within_cutoff(entry: dict, cutoff: datetime) -> bool:
    """True if the entry's source_date is on/after the cutoff. Entries with a
    missing or unparseable source_date are KEPT (don't over-filter)."""
    from knowledge_graph import _parse_source_date
    entry_dt = _parse_source_date(entry.get("source_date"))
    if entry_dt is None:
        return True
    return entry_dt >= cutoff


def _rerank_prep_entries(entries: list[dict], title: str,
                         attendee_names: list[str]) -> list[dict]:
    """Rerank the merged KG context (person + semantic + project fan-out)
    against THIS meeting — title plus non-owner attendees. The reranker drops
    entries it judges irrelevant; on any failure the unreranked list is
    returned (graceful fallback, mirrors semantic_search)."""
    if not config.RERANK_ENABLED or len(entries) <= 1:
        return entries

    query = (
        f"Upcoming meeting: {title}. Attendees: {', '.join(attendee_names)}. "
        "Keep only entries plausibly relevant to THIS meeting and its "
        "non-owner attendees; drop entries about unrelated people or projects."
    )
    texts = []
    for e in entries:
        people = ", ".join(e.get("related_people", []))
        projects = ", ".join(e.get("related_projects", []))
        texts.append(
            f"[{e.get('source_date', '?')}] {e.get('name', '')}: {e.get('content', '')}"
            f" (people: {people}; projects: {projects})"
        )
    try:
        from claude_client import rerank as _rerank
        order = _rerank(query, texts)
        return [entries[i] for i in order]
    except Exception as exc:
        print(f"    KG context rerank failed, using unreranked list: {exc}")
        return entries


def _build_meeting_prep(meeting: dict) -> str | None:
    """Gather KG context for a meeting and generate a prep brief via Gemini."""
    attendee_names = [a["name"] for a in meeting.get("attendees", [])]
    if not attendee_names:
        return None

    all_entries = []
    seen_ids = set()
    title = meeting.get("title", "")

    # Recency window: KG entries older than this are stale for meeting prep.
    cutoff_dt = (
        datetime.now() - timedelta(days=config.MEETING_PREP_CONTEXT_MAX_AGE_DAYS)
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d")

    # The prep is FOR the owner — their own KG trail is noise here. Drop any
    # attendee (or alias below) whose name overlaps the owner's identity.
    owner_tokens = _owner_identity_tokens()
    non_owner_attendees: list[str] = []
    for attendee in attendee_names:
        if _matches_owner(attendee, owner_tokens):
            print(f"    Meeting prep: dropping owner attendee '{attendee}' from KG retrieval")
        else:
            non_owner_attendees.append(attendee)

    # Expand each attendee with their canonical aliases. get_canonical_aliases
    # returns [] when KG_RESOLUTION_ENABLED is off, so query_names equals the
    # attendee set then — a no-op. Deduped by normalized form.
    query_names: list[str] = []
    query_attendee: dict[str, str] = {}  # query name -> base attendee
    seen_query: set[str] = set()
    for attendee in non_owner_attendees:
        for candidate in [attendee, *get_canonical_aliases(attendee)]:
            normalized = candidate.strip().lower()
            if not normalized or normalized in seen_query:
                continue
            if _matches_owner(candidate, owner_tokens):
                print(f"    Meeting prep: dropping owner alias '{candidate}' from KG retrieval")
                continue
            seen_query.add(normalized)
            query_names.append(candidate)
            query_attendee[candidate] = attendee

    # Query KG in parallel: by each attendee AND by meeting title (semantic search)
    attendees_with_context: set[str] = set()
    from knowledge_graph import semantic_search as _semantic_search
    workers = max(len(query_names) + 1, 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict = {
            pool.submit(query_by_person, name, cutoff_str, 8): ("person", name)
            for name in query_names
        }
        # Also search by meeting title to catch topic-based KG entries.
        # rerank=False: the merged set (person + semantic + project fan-out)
        # is reranked ONCE below against title + attendees — reranking here
        # too would double the latency for no precision gain.
        title_future = pool.submit(lambda: _semantic_search(title, limit=10, rerank=False))
        futures[title_future] = ("title", title)

        for future in as_completed(futures):
            kind, name = futures[future]
            try:
                for entry in future.result():
                    # semantic_search has no date awareness (embeddings omit
                    # source_date) — post-filter stale entries; undated kept.
                    if kind == "title" and not _entry_within_cutoff(entry, cutoff_dt):
                        continue
                    if kind == "person":
                        base_attendee = query_attendee.get(name)
                        attendees_with_context.add(base_attendee if base_attendee else name)
                    if entry["id"] not in seen_ids:
                        seen_ids.add(entry["id"])
                        all_entries.append(entry)
            except Exception as exc:
                print(f"    KG query failed ({kind}:{name}): {exc}")

    # Also query by projects found in existing results
    projects = set()
    for e in all_entries:
        projects.update(e.get("related_projects", []))
    project_list = list(projects)[:3]

    if project_list:
        with ThreadPoolExecutor(max_workers=len(project_list)) as pool:
            proj_futures = {
                pool.submit(query_by_project, proj, cutoff_str, 5): proj
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

    # Rerank the merged, deduped set against THIS meeting so context that is
    # merely owner-adjacent (old unrelated projects) gets dropped.
    all_entries = _rerank_prep_entries(
        all_entries, title, non_owner_attendees or attendee_names
    )

    attendees_str = ", ".join(attendee_names)

    # Per-attendee gaps must be explicit — otherwise sparse attendees get
    # silently padded with other attendees' (or the owner's) context.
    missing_context_notes = [
        f"(No prior context found for {attendee}.)"
        for attendee in non_owner_attendees
        if attendee not in attendees_with_context
    ]

    if not all_entries:
        # No KG context — generate a minimal prep with just attendee + time info
        knowledge_context = "(No prior context found for these attendees or topics.)"
    else:
        knowledge_context = format_knowledge_for_context(all_entries[:20])
        if missing_context_notes:
            knowledge_context += "\n" + "\n".join(missing_context_notes)

    prompt = _PREP_PROMPT.format(
        title=meeting["title"],
        attendees=attendees_str,
        start_time=meeting.get("start_time", "soon"),
        knowledge_context=knowledge_context,
    )

    try:
        msg = generate(prompt=prompt, tier=TaskComplexity.LIGHT)
        body = extract_text(msg).strip()
        # Defensive: drop any header line the model emits despite instructions.
        # The header is prepended deterministically below so the meeting title
        # is never LLM-transcribed (a model slip once glued words onto it).
        lines = body.split("\n")
        if lines and ("meeting prep" in lines[0].lower() or lines[0].lstrip().startswith("📋")):
            body = "\n".join(lines[1:]).lstrip("\n")
        if not body:
            return None
        return f"📋 *meeting prep — {title}*\n\n{body}"
    except Exception as exc:
        print(f"  Meeting prep generation failed: {exc}")
        return None


def run_meeting_prep() -> dict:
    """Check for upcoming meetings and send prep briefs for unsent ones.

    Only creates a Langfuse trace when there are actual meetings to prep,
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


@observe(name="meeting-prep", capture_input=False)
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
            commitment_desc = f"{name}: {content}"
            for email in emails:
                prompt = _EVIDENCE_PROMPT.format(
                    commitment=commitment_desc,
                    sender=email.get("from", "?"),
                    subject=email.get("subject", "?"),
                    body=(email.get("body", "") or "")[:500],
                )
                try:
                    msg = generate(prompt=prompt, tier=TaskComplexity.LIGHT)
                    if extract_text(msg).lower().startswith("yes"):
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
        # Dual-read transition: legacy dedup docs use the old Firestore-id key;
        # read both so the cooldown keeps suppressing. Only new_key is written.
        new_key = _nudge_key("commitment", stable_key(entry))
        old_key = _nudge_key("commitment", entry.get("id", entry.get("name", "")))
        if has_nudge_been_sent(new_key) or has_nudge_been_sent(old_key):
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
            "_nudge_key": new_key,
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

    # Canonicalize counter keys so split variants ("Sarah" / "Sarah Chen") merge.
    # resolve_canonical returns an identity map when KG_RESOLUTION_ENABLED is off,
    # so counting is unchanged in that case.
    person_canonical = resolve_canonical(
        list({p for e in entries for p in e.get("related_people", [])})
    )
    project_canonical = resolve_canonical(
        list({p for e in entries for p in e.get("related_projects", [])})
    )

    for e in entries:
        for person in e.get("related_people", []):
            people_counter[person_canonical.get(person, person)] += 1
        for project in e.get("related_projects", []):
            canonical_project = project_canonical.get(project, project)
            project_counter[canonical_project] += 1
            project_types[canonical_project].append(e.get("entity_type", "topic"))
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

    try:
        prompt = _PATTERN_PROMPT.format(patterns="\n".join(pattern_lines))
        msg = generate(prompt=prompt, tier=TaskComplexity.LIGHT)
        text = extract_text(msg).strip()
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


@observe(name="daily-nudges", capture_input=False)
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
