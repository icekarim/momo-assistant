"""Agentic tool-use loop for Momo.

Gives Gemini a set of callable tools (calendar, tasks, gmail, knowledge graph,
Granola, Jira) and lets it decide which to invoke at inference time.  The agent
iterates — calling tools, observing results, calling more tools — until it has
enough information to compose a final text response.
"""

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta

import google.generativeai as genai

import config
from langsmith_config import (
    traceable, traced_chat_send, set_trace_metadata, _get_run_tree,
    add_trace_tags, log_eval_failure,
)

genai.configure(api_key=config.GEMINI_API_KEY)

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

_TYPE_MAP = {
    "string": genai.protos.Type.STRING,
    "integer": genai.protos.Type.INTEGER,
    "number": genai.protos.Type.NUMBER,
    "boolean": genai.protos.Type.BOOLEAN,
    "object": genai.protos.Type.OBJECT,
    "array": genai.protos.Type.ARRAY,
}


def _schema(json_schema: dict) -> genai.protos.Schema:
    """Convert a JSON-Schema-style dict to a genai.protos.Schema."""
    schema_type = _TYPE_MAP.get(json_schema.get("type", "object"), genai.protos.Type.OBJECT)

    properties = {}
    for key, prop in json_schema.get("properties", {}).items():
        properties[key] = genai.protos.Schema(
            type=_TYPE_MAP.get(prop.get("type", "string"), genai.protos.Type.STRING),
            description=prop.get("description", ""),
        )

    required = json_schema.get("required") or None

    return genai.protos.Schema(
        type=schema_type,
        properties=properties if properties else None,
        required=required,
    )


# ── Tool declarations ────────────────────────────────────────

_CORE_TOOLS = [
    genai.protos.Tool(function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_todays_calendar",
            description="Get today's meetings and schedule from Google Calendar. Returns all events for today with times, attendees, and details.",
            parameters=_schema({"type": "object", "properties": {}}),
        ),
        genai.protos.FunctionDeclaration(
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
        genai.protos.FunctionDeclaration(
            name="get_open_tasks",
            description="Get all open/incomplete tasks from Google Tasks across all task lists. Includes due dates, overdue status, and recently completed tasks.",
            parameters=_schema({"type": "object", "properties": {}}),
        ),
        genai.protos.FunctionDeclaration(
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
        genai.protos.FunctionDeclaration(
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
        genai.protos.FunctionDeclaration(
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
        genai.protos.FunctionDeclaration(
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
        genai.protos.FunctionDeclaration(
            name="get_recent_emails",
            description="Get recent unread emails from the inbox. Returns sender, subject, date, and body.",
            parameters=_schema({
                "type": "object",
                "properties": {
                    "max_results": {"type": "integer", "description": "Maximum number of emails to return (default 15)"},
                },
            }),
        ),
        genai.protos.FunctionDeclaration(
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
        genai.protos.FunctionDeclaration(
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
    ]),
]


def _build_optional_tools() -> list:
    """Build tool declarations for optional integrations (Granola, Jira)."""
    extra_decls = []

    if config.GRANOLA_ENABLED:
        extra_decls.append(genai.protos.FunctionDeclaration(
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
        extra_decls.append(genai.protos.FunctionDeclaration(
            name="get_jira_tickets",
            description="Get active Jira tickets where the user is assignee, reporter, or watcher.",
            parameters=_schema({"type": "object", "properties": {}}),
        ))
        extra_decls.append(genai.protos.FunctionDeclaration(
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
        extra_decls.append(genai.protos.FunctionDeclaration(
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
        extra_decls.append(genai.protos.FunctionDeclaration(
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
        extra_decls.append(genai.protos.FunctionDeclaration(
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
    return [genai.protos.Tool(function_declarations=extra_decls)]


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
You're casual. Like texting-your-friend casual. Lowercase is your default. capitalization is for emphasis or when you're being dramatic on purpose.
You're warm but not try-hard. No "certainly!" no "absolutely!" no "great question!" — that energy is dead to you.
You use gen-z slang naturally, not like a brand account trying to go viral. If it doesn't fit, you don't force it.
You're a little sarcastic, a little playful, but never mean. You roast gently and only when it's funny.
You match the user's energy. If they're stressed, you dial it back and actually help. If they're vibing, you vibe back.
You say "ngl", "lowkey", "fr", "tbh", "bet", "no cap" etc. when it flows — but you're not spamming them in every sentence like a parody.
You use "lol", "lmao" sparingly for flavor — not as punctuation.

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

=== DUPLICATE TASK PREVENTION ===
Before creating a task, call get_open_tasks first and check for duplicates. If a task with the same or very similar title already exists, tell the user it's already on their list instead of creating a duplicate.

=== TASK DUE DATES ===
When creating a task, if the user specifies a due date (e.g. "due Friday", "by next week", "due March 15"), use that date. If they don't mention a date at all, omit the due field — the system will automatically default it to today.

=== TL;DR ===
Momo is the friend who fixes your resume at 2am, tells you your ex's rebound is mid, explains your calendar without making you feel overwhelmed, and somehow makes all of it feel easy. helpful first, vibes always.

Keep responses scannable. Google Chat supports *bold* and basic formatting. Use section emojis (📅 ✅ 📧 🎫 🎯) and priority colors (🔴 🟡 🟢) to make messages easy to read at a glance. No other emojis."""


# ── Agent loop ───────────────────────────────────────────────


def _build_history(conversation_history: list[dict], user_memories_context: str = "") -> list[dict]:
    """Convert stored conversation history to Gemini chat format."""
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
        {"role": "user", "parts": [f"[SYSTEM DATE REFERENCE]\n{date_ref}\n[END DATE REFERENCE]"]},
        {"role": "model", "parts": ["got it, i know the date. what's up?"]},
    ]

    if user_memories_context:
        history.append({"role": "user", "parts": [user_memories_context]})
        history.append({"role": "model", "parts": ["got it, i'll keep those in mind."]})

    recent = conversation_history[-20:] if len(conversation_history) > 20 else conversation_history
    for turn in recent:
        role = "model" if turn["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [turn["content"]]})

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

    Sends the user message to Gemini with tool declarations.  If Gemini
    responds with function calls, executes them and sends results back.
    Repeats until Gemini produces a text response or max_iterations is hit.

    Returns the final text response and any queued task actions.
    """
    if thread_id:
        set_trace_metadata(thread_id=thread_id)
    tools = _get_all_tools()
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL_FLASH,
        system_instruction=AGENT_SYSTEM_PROMPT,
        tools=tools,
    )

    # Load user memories for context injection
    user_memories_context = ""
    if config.USER_MEMORY_ENABLED and user_id:
        try:
            from user_memory import get_user_memories, format_memories_for_context
            memories = get_user_memories(user_id)
            user_memories_context = format_memories_for_context(memories)
        except Exception as exc:
            print(f"[agent] failed to load user memories: {exc}")

    history = _build_history(conversation_history, user_memories_context=user_memories_context)
    chat = model.start_chat(history=history)

    t0 = time.time()
    print(f"[agent] starting loop (max_iterations={max_iterations})")

    _trace_metrics = {
        "iteration_count": 0,
        "tool_calls": [],       # list of {"name", "elapsed_s", "success"}
        "tool_names": [],       # ordered tool names called
        "total_tool_calls": 0,
        "unique_tools": set(),
        "errors": [],
    }

    pending_task_actions: list[dict] = []

    try:
        response = traced_chat_send(chat, user_message, model_name=config.GEMINI_MODEL_FLASH)
    except Exception as exc:
        print(f"[agent] initial send failed: {exc}")
        traceback.print_exc()
        _trace_metrics["errors"].append(f"initial_send: {exc}")
        _flush_trace_metrics(_trace_metrics, t0)
        return "sorry, something went wrong talking to gemini — try again in a sec?", pending_task_actions

    for iteration in range(max_iterations):
        _trace_metrics["iteration_count"] = iteration + 1

        candidate = response.candidates[0] if response.candidates else None
        if not candidate:
            break

        parts = candidate.content.parts
        function_calls = [p for p in parts if p.function_call and p.function_call.name]
        text_parts = [p.text for p in parts if hasattr(p, "text") and p.text]

        if not function_calls:
            if text_parts:
                final = "\n".join(text_parts)
                print(f"[agent] done in {iteration + 1} iteration(s), {time.time() - t0:.2f}s total")
                _flush_trace_metrics(_trace_metrics, t0)
                return final, pending_task_actions
            break

        print(f"[agent] iteration {iteration + 1}: {len(function_calls)} tool call(s): "
              f"{[fc.function_call.name for fc in function_calls]}")

        tool_responses = []
        # Capture the current run tree so child threads inherit the trace context
        parent_run_tree = _get_run_tree()
        for part in function_calls:
            fc = part.function_call
            timeout = _TOOL_TIMEOUTS.get(fc.name, 10)

            def _run_tool(name, args, pending, rt=parent_run_tree):
                """Execute tool with LangSmith context propagated."""
                if rt is not None:
                    import contextvars
                    try:
                        from langsmith.run_helpers import _PARENT_RUN_TREE
                        _PARENT_RUN_TREE.set(rt)
                    except (ImportError, AttributeError):
                        pass
                return execute_tool(name, args, pending,
                                    user_id=user_id, user_message=user_message)

            _tool_t0 = time.time()
            tool_success = True
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_run_tool, fc.name, dict(fc.args), pending_task_actions)
                    result_str = future.result(timeout=timeout)
                    if result_str.startswith(("Tool '", "Error calling")):
                        tool_success = False
            except FuturesTimeoutError:
                result_str = f"Tool '{fc.name}' timed out after {timeout}s"
                print(f"[agent] tool '{fc.name}' timed out")
                tool_success = False
                _trace_metrics["errors"].append(f"timeout: {fc.name}")
            except Exception as exc:
                result_str = f"Tool '{fc.name}' failed: {str(exc)}"
                tool_success = False
                _trace_metrics["errors"].append(f"exception: {fc.name}: {exc}")

            # Record tool metrics for trajectory evaluation
            _trace_metrics["tool_calls"].append({
                "name": fc.name,
                "elapsed_s": round(time.time() - _tool_t0, 3),
                "success": tool_success,
            })
            _trace_metrics["tool_names"].append(fc.name)
            _trace_metrics["unique_tools"].add(fc.name)
            _trace_metrics["total_tool_calls"] += 1

            tool_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": result_str},
                    )
                )
            )

        if text_parts:
            tool_responses.append(genai.protos.Part(text="\n".join(text_parts)))

        try:
            response = traced_chat_send(chat, tool_responses, model_name=config.GEMINI_MODEL_FLASH, iteration=iteration + 1)
        except Exception as exc:
            print(f"[agent] send_message failed on iteration {iteration + 1}: {exc}")
            _trace_metrics["errors"].append(f"send_message: {exc}")
            _flush_trace_metrics(_trace_metrics, t0)
            if text_parts:
                return "\n".join(text_parts), pending_task_actions
            return "sorry, hit a snag pulling your data — try again?", pending_task_actions

    final_text = ""
    try:
        if response.candidates:
            for p in response.candidates[0].content.parts:
                if hasattr(p, "text") and p.text:
                    final_text += p.text
    except Exception:
        pass

    elapsed = time.time() - t0
    print(f"[agent] loop ended after {max_iterations} iterations, {elapsed:.2f}s total")

    _flush_trace_metrics(_trace_metrics, t0)

    # Log exhausted loops as eval failure candidates
    if not final_text:
        log_eval_failure(
            user_message=user_message,
            expected_behavior="Agent should produce a text response",
            actual_behavior=f"Exhausted {max_iterations} iterations without text response. "
                            f"Tools called: {_trace_metrics['tool_names']}",
            category="agent_loop_exhaustion",
        )

    return final_text or "i pulled a lot of info but couldn't put it together — try asking differently?", pending_task_actions
