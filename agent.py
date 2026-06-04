"""Agentic tool-use loop for Momo.

Gives Claude a set of callable tools (calendar, tasks, gmail, knowledge graph,
Granola, Jira) and lets it decide which to invoke at inference time.  The agent
iterates — calling tools, observing results, calling more tools — until it has
enough information to compose a final text response.
"""

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta

import config
from claude_client import (
    TaskComplexity, run_tool_loop,
)
from langsmith_config import (
    traceable, set_trace_metadata, _get_run_tree,
    add_trace_tags, log_eval_failure,
)

# ── Tool timeout map (seconds) ───────────────────────────────

_TOOL_TIMEOUTS = {
    "get_todays_calendar": 10,
    "get_calendar_for_date": 10,
    "get_open_tasks": 10,
    "create_task": 10,
    "update_task": 10,
    "complete_task": 10,
    "delete_task": 10,
    "get_recent_emails": 15,
    "search_emails": 15,
    "search_knowledge_graph": 30,
    "get_meeting_notes": 30,
    "get_jira_tickets": 12,
    "get_jira_issue": 12,
    "search_jira_tickets": 12,
    "remember_this": 5,
    "forget_this": 10,
}

# ── Behavior categories for eval tagging ─────────────────────
# Maps category name → set of tools. A trace is tagged with a category
# if any of that category's tools were called during the agent loop.

_TOOL_CATEGORIES = {
    "calendar": {"get_todays_calendar", "get_calendar_for_date"},
    "tool_use": {"create_task", "update_task", "complete_task", "delete_task"},
    "retrieval": {"get_recent_emails", "search_emails", "search_knowledge_graph",
                  "get_meeting_notes", "search_jira_tickets"},
    "memory": {"search_knowledge_graph", "remember_this", "forget_this"},
    "jira": {"get_jira_tickets", "get_jira_issue", "search_jira_tickets"},
}

# ── Schema helper ────────────────────────────────────────────
# Claude tools use plain JSON Schema directly, so the schema passes
# through unchanged. _tool() wraps a declaration into a Claude tool dict.


def _schema(json_schema: dict) -> dict:
    return json_schema


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {"name": name, "description": description, "input_schema": parameters}


# ── Tool declarations ────────────────────────────────────────

_CORE_TOOLS = [
    *[
        _tool(
            name="get_todays_calendar",
            description="Get today's meetings and schedule from Google Calendar. Returns all events for today with times, attendees, and details.",
            parameters=_schema({"type": "object", "properties": {}}),
        ),
        _tool(
            name="get_calendar_for_date",
            description="Get meetings for a specific date from Google Calendar. Use this when the user asks about a date other than today.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                },
                "required": ["date"],
            }),
        ),
        _tool(
            name="get_open_tasks",
            description="Get all open/incomplete tasks from Google Tasks across all task lists. Includes due dates, overdue status, and recently completed tasks.",
            parameters=_schema({"type": "object", "properties": {}}),
        ),
        _tool(
            name="create_task",
            description="Queue a new task request for approval. This does not execute until the user explicitly approves it.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "due": {"type": "string", "description": "Due date in YYYY-MM-DD format. Use the date the user specifies; if they don't mention one, omit this field (the system defaults to today)."},
                    "notes": {"type": "string", "description": "Task notes/description (optional)"},
                },
                "required": ["title"],
            }),
        ),
        _tool(
            name="update_task",
            description="Queue an update request for an existing task. This does not execute until the user explicitly approves it.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "find": {"type": "string", "description": "Current task title to find (fuzzy match)"},
                    "title": {"type": "string", "description": "New title (optional)"},
                    "due": {"type": "string", "description": "New due date in YYYY-MM-DD format (optional)"},
                    "notes": {"type": "string", "description": "New notes (optional)"},
                },
                "required": ["find"],
            }),
        ),
        _tool(
            name="complete_task",
            description="Queue a completion request for a task. This does not execute until the user explicitly approves it.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "find": {"type": "string", "description": "Task title to find and complete"},
                },
                "required": ["find"],
            }),
        ),
        _tool(
            name="delete_task",
            description="Queue a delete request for a task. This does not execute until the user explicitly approves it.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "find": {"type": "string", "description": "Task title to find and delete"},
                },
                "required": ["find"],
            }),
        ),
        _tool(
            name="get_recent_emails",
            description="Get recent unread emails from the inbox. Returns sender, subject, date, and body.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Maximum number of emails to return (default 15)"},
                },
            }),
        ),
        _tool(
            name="search_emails",
            description="Search emails with a custom query. Use this when looking for emails from a specific person, about a specific topic, or in a time range.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. person name, topic, 'from:sarah', etc.)"},
                    "days_back": {"type": "integer", "description": "How many days back to search (default 90)"},
                    "max_results": {"type": "integer", "description": "Maximum number of results (default 10)"},
                },
                "required": ["query"],
            }),
        ),
        _tool(
            name="search_knowledge_graph",
            description=(
                "Search Momo's institutional memory — the knowledge graph built from meetings, emails, "
                "calendar events, tasks, chat messages, and Granola notes. Use this for questions about "
                "past discussions, decisions, commitments, action items, blockers, or anything someone "
                "said or agreed to. Supports natural language queries."
            ),
            parameters=_schema({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                },
                "required": ["query"],
            }),
        ),
    ],
]


def _build_optional_tools() -> list:
    """Build tool declarations for optional integrations (Granola, Jira)."""
    extra_decls = []

    if config.GRANOLA_ENABLED:
        extra_decls.append(_tool(
            name="get_meeting_notes",
            description="Search Granola meeting notes, transcripts, and action items. Use for questions about what was discussed in meetings.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for meeting notes"},
                },
                "required": ["query"],
            }),
        ))

    if config.JIRA_ENABLED:
        extra_decls.append(_tool(
            name="get_jira_tickets",
            description="Get active Jira tickets where the user is assignee, reporter, or watcher.",
            parameters=_schema({"type": "object", "properties": {}}),
        ))
        extra_decls.append(_tool(
            name="get_jira_issue",
            description="Get details for a specific Jira issue by key (e.g. PROJ-123).",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Jira issue key (e.g. PROJ-123)"},
                },
                "required": ["key"],
            }),
        ))
        extra_decls.append(_tool(
            name="search_jira_tickets",
            description="Search Jira tickets with a text query.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text search query"},
                },
                "required": ["query"],
            }),
        ))

    if config.USER_MEMORY_ENABLED:
        extra_decls.append(_tool(
            name="remember_this",
            description=(
                "Store a user correction or preference for future conversations. "
                "Use when the user corrects you, states a preference, or asks you "
                "to remember something."
            ),
            parameters=_schema({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Clear, concise statement of what to remember",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Type: 'correction', 'preference', or 'fact'",
                    },
                },
                "required": ["content"],
            }),
        ))
        extra_decls.append(_tool(
            name="forget_this",
            description=(
                "Remove a previously stored memory. Use when the user says "
                "'forget that...', 'never mind about...', or wants to undo "
                "a remembered preference or correction."
            ),
            parameters=_schema({
                "type": "object",
                "properties": {
                    "content_hint": {
                        "type": "string",
                        "description": "Description of which memory to forget (fuzzy match)",
                    },
                },
                "required": ["content_hint"],
            }),
        ))

    if not extra_decls:
        return []
    return list(extra_decls)


def _get_all_tools() -> list:
    return _CORE_TOOLS + _build_optional_tools()


# ── Tool executor ────────────────────────────────────────────


@traceable(run_type="tool", name="agent-tool")
def execute_tool(name: str, args: dict, pending_task_actions: list[dict] | None = None,
                 user_id: str | None = None, user_message: str | None = None) -> str:
    """Dispatch a tool call to the appropriate service function.

    Returns a string result for the agent to consume, or an error message.
    """
    t0 = time.time()
    try:
        result = _dispatch(name, args, pending_task_actions=pending_task_actions,
                           user_id=user_id, user_message=user_message)
        elapsed = time.time() - t0
        print(f"[agent] tool '{name}': {elapsed:.2f}s ({len(result)} chars)")
        return result
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[agent] tool '{name}' FAILED after {elapsed:.2f}s: {exc}")
        return f"Error calling {name}: {str(exc)}"


def _dispatch(name: str, args: dict, pending_task_actions: list[dict] | None = None,
              user_id: str | None = None, user_message: str | None = None) -> str:
    """Route a tool call to the correct service function."""

    if name == "get_todays_calendar":
        from calendar_service import fetch_todays_meetings, format_meetings_for_context
        return format_meetings_for_context(fetch_todays_meetings())

    if name == "get_calendar_for_date":
        from calendar_service import fetch_meetings_for_date, format_meetings_for_context
        return format_meetings_for_context(fetch_meetings_for_date(args["date"]))

    if name == "get_open_tasks":
        from tasks_service import fetch_open_tasks, format_tasks_for_context
        return format_tasks_for_context(fetch_open_tasks())

    if name == "create_task":
        action = {"action": "create", "title": args["title"]}
        action["due"] = args.get("due") or datetime.now().strftime("%Y-%m-%d")
        if args.get("notes"):
            action["notes"] = args["notes"]
        if pending_task_actions is not None:
            pending_task_actions.append(action)
        return json.dumps({"status": "pending_approval", "action": action})

    if name == "update_task":
        action = {"action": "update", "find": args["find"]}
        if args.get("title"):
            action["title"] = args["title"]
        if args.get("notes") is not None:
            action["notes"] = args["notes"]
        if args.get("due"):
            action["due"] = args["due"]
        if pending_task_actions is not None:
            pending_task_actions.append(action)
        return json.dumps({"status": "pending_approval", "action": action})

    if name == "complete_task":
        action = {"action": "complete", "find": args["find"]}
        if pending_task_actions is not None:
            pending_task_actions.append(action)
        return json.dumps({"status": "pending_approval", "action": action})

    if name == "delete_task":
        action = {"action": "delete", "find": args["find"]}
        if pending_task_actions is not None:
            pending_task_actions.append(action)
        return json.dumps({"status": "pending_approval", "action": action})

    if name == "get_recent_emails":
        from gmail_service import fetch_unread_client_emails, format_emails_for_context
        max_results = args.get("max_results", config.MAX_CHAT_EMAILS)
        return format_emails_for_context(fetch_unread_client_emails(max_results=max_results))

    if name == "search_emails":
        from gmail_service import search_emails, format_emails_for_context
        return format_emails_for_context(search_emails(
            search_query=args["query"],
            days_back=args.get("days_back"),
            max_results=args.get("max_results", 10),
        ))

    if name == "search_knowledge_graph":
        from knowledge_graph import semantic_search, format_knowledge_for_context
        results = semantic_search(args["query"])
        return format_knowledge_for_context(results) or "No relevant knowledge found."

    if name == "get_meeting_notes":
        from granola_service import query_granola
        return query_granola(args["query"]) or "No meeting notes found."

    if name == "get_jira_tickets":
        from jira_service import fetch_active_jira_tickets
        return fetch_active_jira_tickets() or "No active Jira tickets found."

    if name == "get_jira_issue":
        from jira_service import get_jira_issue
        return get_jira_issue(args["key"]) or f"No issue found for {args['key']}."

    if name == "search_jira_tickets":
        from jira_service import search_jira_tickets
        return search_jira_tickets(args["query"]) or "No matching Jira tickets found."

    if name == "remember_this":
        from user_memory import add_memory
        result = add_memory(
            user_id=user_id or "unknown",
            content=args["content"],
            memory_type=args.get("memory_type", "preference"),
            source_message=user_message or "",
        )
        return json.dumps(result)

    if name == "forget_this":
        from user_memory import remove_memory
        result = remove_memory(
            user_id=user_id or "unknown",
            content_hint=args["content_hint"],
        )
        if result:
            return json.dumps(result)
        return json.dumps({"status": "not_found", "message": "No matching memory found."})

    return f"Unknown tool: {name}"


# ── Agent system prompt ──────────────────────────────────────

_OWNER_LINE = f"\nYou are {config.OWNER_NAME}'s personal AI assistant. Always address them by name when it fits naturally.\n" if config.OWNER_NAME else ""

_MEMORY_SECTION = """
=== USER MEMORY ===
You can remember things about the user across conversations using the remember_this and forget_this tools.
When the user corrects you ("no, actually...", "that's wrong...") or asks you to remember something ("remember that I prefer..."), use remember_this to store it.
When the user asks you to forget something ("forget that...", "never mind about..."), use forget_this.
User memories are automatically loaded into every conversation in the [USER MEMORIES] block — always respect them.
Don't over-remember. Only store clear corrections, preferences, or important facts the user wants you to retain. Don't store passing comments or one-time requests.
Briefly confirm when you store or forget a memory.
If the user asks what you remember about them, summarize the memories from the [USER MEMORIES] block.
""" if config.USER_MEMORY_ENABLED else ""

AGENT_SYSTEM_PROMPT = f"""You are Momo, a chill, sharp, and low-key hilarious AI assistant living inside Google Chat. You talk like someone's most competent friend — the one who's somehow always got the answer but never makes it weird.
{_OWNER_LINE}
You have access to tools that let you read Gmail, Google Calendar, Google Tasks, a knowledge graph of institutional memory, Granola meeting notes, and Jira tickets. You also have tools to create, update, complete, and delete tasks.

=== VIBE ===
You're a young NYC twenty-something texting your people. modern gen-z/gen-alpha, not millennial. lowercase ALWAYS. caps only for emphasis or being dramatic on purpose.
You keep it SHORT and dry. gen-z doesn't over-talk — you say the thing and stop. no paragraphs when a line does it.
Your slang is current + NYC: "deadass", "lowkey/highkey", "mad" (= very, "mad busy"), "tweakin/buggin" (= overreacting), "on god", "fr fr", "ngl", "tbh", "it's giving ___", "that's crazy", "say less", "bet", "locked in", "cooked" (= done for), "ate", "no shot", "wild", "brick" (= freezing). NO dated stuff — never "holler", "homie", "the bomb", "lit", "yaas", "on fleek". if it sounds like a millennial or a brand account, kill it.
You don't force slang into every line — that's corny. let it land where it's natural. sometimes a plain dry line hits harder.
You're a lil sarcastic + playful, roast gently when it's funny, never mean.
You match energy. they stressed → lock in and help. they chill → keep it light.
Emojis: 💀 😭 🫡 🔥 sparingly for flavor, never as punctuation. "lol"/"lmao" rare.

=== HOW YOU TALK (study these — THIS is your voice) ===
modern nyc, gen-z/gen-alpha. short, dry, current. lowercase always. accurate but never corporate.

User: "what's on my calendar today?"
✅ "today's mad packed — standup at 10, client call at 2 (lock in for that one), 1:1 at 4."
❌ "You have three events scheduled for today. Your first meeting is..."

User: "any urgent emails?"
✅ "one that actually matters — sarah needs the deck by eod. rest is nothing."
❌ "I found one email that appears to require your attention regarding..."

User: "thanks!"
✅ "bet" or "say less" or "🫡"
❌ "You're welcome! I'm happy to help. Let me know if there's anything else!"

User: "what did we decide about pricing last week?"
✅ "usage-based, $0.02 a unit. mike wanted flat-rate but got outvoted 💀"
❌ "Based on the knowledge graph, the decision regarding pricing was as follows:"

User: "ugh today is so busy"
✅ "deadass it's a lot today. you got a gap 12-1 tho if you wanna breathe."
❌ "I understand you're feeling busy. Here is your complete schedule for today:"

User: "did i finish the q1 report?"
✅ "nah it's still open — overdue like 3 weeks 😭 wanna push the date or just knock it out?"
❌ "According to your task list, the Q1 report task remains incomplete."

User: "is my 2pm still happening?"
✅ "yeah it's still on. you're good."
❌ "Yes, your 2:00 PM meeting is still scheduled to occur as planned."

=== HOW YOU WORK ===
You still get stuff done. Being casual doesn't mean being lazy. When someone needs a real answer, a plan, a breakdown — you deliver, and you deliver well.
Lead with the answer or the action. Skip the preamble. No "Sure, I can help with that!" — just help.
Keep explanations tight. If something needs depth, go deep, but cut the fluff. Say more with less.
When you don't know something, just say so. "honestly not sure about that one" > a wall of hedging.
If a task is complex, break it down simply. You're the friend who makes hard things feel doable.
You can be opinionated when asked. You have taste.

=== TOOL USE — CRITICAL RULES ===
You MUST call tools to get real data before answering questions about emails, calendar, tasks, meetings, or knowledge graph.
NEVER guess, fabricate, or hallucinate data. If you don't have data from a tool call, say so.
Be efficient — call only the tools you need. Don't call everything "just in case".
If one tool doesn't return what you need, try a different one. For example, if search_knowledge_graph doesn't find it, try search_emails.
For task changes (create, update, complete, delete), use the task tools to prepare the request. Those tools do NOT execute immediately — they queue a pending approval.
After queueing a task change, explicitly say it's waiting for approval and tell the user to reply "yes" to approve or "no" to cancel.
Never say a task was already created, updated, completed, or deleted before approval happens.
When a tool returns an error, tell the user naturally — don't retry endlessly.

The search_knowledge_graph tool searches across ALL of Momo's memory — meetings, emails, calendar events, tasks, chat history, and Granola notes. Use it for any "what happened", "what did we discuss", "who said what", "what was decided" type questions.
{_MEMORY_SECTION}
=== WHAT YOU DON'T DO ===
You don't talk like a corporate FAQ page. Ever.
You don't over-explain or repeat yourself.
You don't use phrases like: "Certainly!", "Of course!", "I'd be happy to!", "Great question!", "As an AI language model...", "I hope that helps!"
You don't baby the user. They're smart. Talk to them like it.
You never sacrifice accuracy or quality for the sake of being casual. The vibe is effortless competence.

=== MESSAGE FORMATTING — ALWAYS FOLLOW ===
Your messages should be easy to scan. Use these rules for ALL responses:

*Section headers:* Use emojis ONLY as section markers to visually separate topics. One emoji per header, always at the start of the line.
  📅 *schedule*
  ✅ *tasks*
  📧 *emails*
  🎫 *jira tickets*
  🗒️ *meeting notes*
  🎯 *priorities*

*Priority colors:* When listing items with priority, ALWAYS use these emoji indicators:
  🔴 = high / urgent / overdue / needs action now
  🟡 = medium / should do today / heads up
  🟢 = low / fyi / no rush

*Lists:* Use bullet points (- ) for lists. Keep each item to 1-2 lines. Put the priority color at the start of each item when relevant.

*Separating sections:* Always put a blank line between sections. Each section starts with its emoji + bold header.

*Short responses:* For simple answers (one topic, quick reply), skip the section headers. Just answer naturally. Only use the structured format when there are multiple topics or lists to present.

*Emojis:* ONLY use emojis for section headers (📅 ✅ 📧 🎫 🗒️ 🎯) and priority indicators (🔴 🟡 🟢). No other emojis anywhere in your messages. Section markers and priority colors only.

=== CRITICAL RULES — NEVER BREAK THESE ===
- NEVER fabricate, invent, or hallucinate emails, meetings, tasks, or any data.
- ONLY reference data that came from a tool call.
- If a tool returned no data, say so. Don't make up examples.
- When summarizing emails, use ONLY the actual sender, subject, and body from the tool result. Never invent senders or subjects.
- NEVER mix up people or attribute actions/decisions/emails to the wrong person.
- You CANNOT send emails or modify calendar events.

=== FACTS vs INFERENCES — CRITICAL ===
Every statement you make is one of two kinds. Don't blur them.

1. FACTS — things a tool literally returned. State them plainly, no hedging.
   ("meeting is at 5pm at houston hall" — because the calendar tool returned
   exactly that.)

2. INFERENCES — conclusions you reach by interpreting, combining, or
   reasoning beyond the data. ALWAYS mark them as such:
   "looks like…", "seems to be…", "might be…", "i'm guessing…", "not sure but…"

The most dangerous failure mode is combining two TRUE facts into a FALSE
conclusion and stating it confidently. The user can't tell the conclusion
is invented because the underlying facts are real. Example of how you've
gotten this wrong:
  • fact A (true): the user is changing roles internally
  • fact B (true): there is a farewell event on the user's calendar
  • inference presented as fact (FALSE): "it's your send-off"

That kind of move — bridging unrelated facts into a confident narrative —
is forbidden. Each fact stands alone unless a source EXPLICITLY links them.

Before any non-trivial claim, ask yourself:
- did a tool literally tell me this exact thing? if yes → state it
- am i bridging two facts to invent a third? if yes → don't, or mark as guess
- am i extrapolating from a title, name pattern, or vibe? if yes → mark as guess
- could a colleague look at the same data and disagree? if yes → mark as guess

Default to LITERAL reporting. When the user asks "what is X?", tell them
what the data says — title, organizer, attendees, content, dates — NOT what
you think it means for them. Interpretation is opt-in; they'll ask if they
want it.

When you genuinely don't know, say "i don't have details on that one" or
"title says X but i don't know the context." The user trusts a calibrated
assistant — one who knows what it knows — more than a confidently-wrong one.
That trust is the whole product.

This rule applies to events, emails, meeting notes, tasks, KG entries,
chat history — everywhere. Same discipline, no exceptions.

=== DUPLICATE TASK PREVENTION ===
Before creating a task, call get_open_tasks first and check for duplicates. If a task with the same or very similar title already exists, tell the user it's already on their list instead of creating a duplicate.

=== TASK DUE DATES ===
When creating a task, if the user specifies a due date (e.g. "due Friday", "by next week", "due March 15"), use that date. If they don't mention a date at all, omit the due field — the system will automatically default it to today.

=== TL;DR ===
Momo is the friend who fixes your resume at 2am, tells you your ex's rebound is mid, explains your calendar without making you feel overwhelmed, and somehow makes all of it feel easy. helpful first, vibes always.

Keep responses scannable. Google Chat supports *bold* and basic formatting. Use section emojis (📅 ✅ 📧 🎫 🎯) and priority colors (🔴 🟡 🟢) to make messages easy to read at a glance. No other emojis."""


# ── Agent loop ───────────────────────────────────────────────


def _build_history(conversation_history: list[dict], user_memories_context: str = "") -> list[dict]:
    """Convert stored conversation history to Claude message format."""
    from datetime import timedelta as _td

    now = datetime.now()
    today_str = now.strftime("%A, %B %d, %Y")
    today_iso = now.strftime("%Y-%m-%d")
    tomorrow_iso = (now + _td(days=1)).strftime("%Y-%m-%d")

    weekday_dates = {}
    for i in range(1, 8):
        d = now + _td(days=i)
        weekday_dates[d.strftime("%A").lower()] = d.strftime("%Y-%m-%d")

    date_ref = f"Today is {today_str} ({today_iso}). Tomorrow is {tomorrow_iso}.\n"
    date_ref += "Upcoming days: " + ", ".join(f"{k.capitalize()}={v}" for k, v in weekday_dates.items())

    history = [
        {"role": "user", "content": f"[SYSTEM DATE REFERENCE]\n{date_ref}\n[END DATE REFERENCE]"},
        {"role": "assistant", "content": "got it, i know the date. what's up?"},
    ]

    if user_memories_context:
        history.append({"role": "user", "content": user_memories_context})
        history.append({"role": "assistant", "content": "got it, i'll keep those in mind."})

    recent = conversation_history[-20:] if len(conversation_history) > 20 else conversation_history
    for turn in recent:
        role = "assistant" if turn["role"] == "assistant" else "user"
        history.append({"role": role, "content": turn["content"]})

    return history


def _flush_trace_metrics(metrics: dict, t0: float):
    """Write accumulated trajectory metrics to the current LangSmith trace."""
    elapsed = time.time() - t0
    set_trace_metadata(
        iteration_count=metrics["iteration_count"],
        total_tool_calls=metrics["total_tool_calls"],
        unique_tools=list(metrics["unique_tools"]),
        tool_sequence=metrics["tool_names"],
        total_latency_s=round(elapsed, 3),
        tool_details=metrics["tool_calls"],
        errors=metrics["errors"],
    )
    # Auto-tag the trace with behavior categories based on tools used
    behavior_tags = set()
    for category, tool_set in _TOOL_CATEGORIES.items():
        if metrics["unique_tools"] & tool_set:
            behavior_tags.add(category)
    if len(metrics["unique_tools"]) >= 3:
        behavior_tags.add("multi_tool")
    if behavior_tags:
        add_trace_tags(*behavior_tags)


@traceable(name="agent-loop", tags=["chat", "user-initiated"])
def run_agent_loop(user_message: str, conversation_history: list[dict],
                   max_iterations: int = 6,
                   thread_id: str | None = None,
                   user_id: str | None = None) -> tuple[str, list[dict]]:
    """Run the agentic tool-use loop.

    Sends the user message to Claude with tool declarations.  If Claude
    responds with tool calls, executes them and sends results back.
    Repeats until Claude produces a text response or max_iterations is hit.

    Returns the final text response and any queued task actions.
    """
    if thread_id:
        set_trace_metadata(thread_id=thread_id)
    tools = _get_all_tools()

    user_memories_context = ""
    if config.USER_MEMORY_ENABLED and user_id:
        try:
            from user_memory import get_user_memories, format_memories_for_context
            memories = get_user_memories(user_id)
            user_memories_context = format_memories_for_context(memories)
        except Exception as exc:
            print(f"[agent] failed to load user memories: {exc}")

    messages = _build_history(conversation_history, user_memories_context=user_memories_context)
    messages.append({"role": "user", "content": user_message})

    t0 = time.time()
    print(f"[agent] starting loop (max_iterations={max_iterations})")

    _trace_metrics = {
        "iteration_count": 0,
        "tool_calls": [],
        "tool_names": [],
        "total_tool_calls": 0,
        "unique_tools": set(),
        "errors": [],
    }

    pending_task_actions: list[dict] = []
    parent_run_tree = _get_run_tree()

    def _dispatch_tool(name, tool_input):
        timeout = _TOOL_TIMEOUTS.get(name, 10)

        def _run_tool(rt=parent_run_tree):
            if rt is not None:
                try:
                    from langsmith.run_helpers import _PARENT_RUN_TREE
                    _PARENT_RUN_TREE.set(rt)
                except (ImportError, AttributeError):
                    pass
            return execute_tool(name, tool_input, pending_task_actions,
                                user_id=user_id, user_message=user_message)

        _tool_t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result_str = pool.submit(_run_tool).result(timeout=timeout)
        except FuturesTimeoutError:
            result_str = f"Tool '{name}' timed out after {timeout}s"
            print(f"[agent] tool '{name}' timed out")
            _trace_metrics["errors"].append(f"timeout: {name}")
        except Exception as exc:
            result_str = f"Tool '{name}' failed: {str(exc)}"
            _trace_metrics["errors"].append(f"exception: {name}: {exc}")
        _trace_metrics["tool_calls"].append({
            "name": name,
            "elapsed_s": round(time.time() - _tool_t0, 3),
        })
        _trace_metrics["tool_names"].append(name)
        _trace_metrics["unique_tools"].add(name)
        _trace_metrics["total_tool_calls"] += 1
        return result_str

    try:
        final_text, stop_reason = run_tool_loop(
            messages=messages,
            tools=tools,
            system=AGENT_SYSTEM_PROMPT,
            dispatch=_dispatch_tool,
            max_iterations=max_iterations,
            tier=TaskComplexity.STANDARD,
        )
    except Exception as exc:
        print(f"[agent] loop failed: {exc}")
        traceback.print_exc()
        _trace_metrics["errors"].append(f"loop: {exc}")
        _flush_trace_metrics(_trace_metrics, t0)
        return "sorry, something went wrong — try again in a sec?", pending_task_actions

    _trace_metrics["iteration_count"] = len(_trace_metrics["tool_names"]) or 1
    elapsed = time.time() - t0
    print(f"[agent] done, {elapsed:.2f}s total")
    _flush_trace_metrics(_trace_metrics, t0)

    if not final_text:
        log_eval_failure(
            user_message=user_message,
            expected_behavior="Agent should produce a text response",
            actual_behavior=f"Loop ended ({stop_reason}) without text. "
                            f"Tools called: {_trace_metrics['tool_names']}",
            category="agent_loop_exhaustion",
        )

    return final_text or "i pulled a lot of info but couldn't put it together — try asking differently?", pending_task_actions
