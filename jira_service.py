"""Jira REST API client — fetches tickets via Jira Cloud REST API v3.

Auth: Basic auth with email + API token.
Generate a token at: https://id.atlassian.com/manage-profile/security/api-tokens
"""

import base64

import httpx

import config

_TIMEOUT = 15


def _get_auth_header() -> dict[str, str]:
    """Build Basic auth header from email + API token."""
    raw = f"{config.JIRA_USER_EMAIL}:{config.JIRA_API_TOKEN}"
    encoded = base64.b64encode(raw.encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    site = config.JIRA_SITE_URL.rstrip("/")
    if not site.startswith("http"):
        site = f"https://{site}"
    return f"{site}/rest/api/3"


def _search(jql: str, max_results: int = 50, fields: list[str] | None = None) -> list[dict]:
    """Run a JQL search via POST /search/jql and return the list of issues."""
    url = f"{_base_url()}/search/jql"
    body: dict = {"jql": jql, "maxResults": max_results}
    if fields:
        body["fields"] = fields

    try:
        resp = httpx.post(url, headers=_get_auth_header(), json=body, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("issues", [])
    except Exception as exc:
        print(f"Jira: search failed — {exc}")
        return []


# ── Public helpers ───────────────────────────────────────────


_ISSUE_FIELDS = ["summary", "status", "priority", "assignee", "reporter", "updated", "created", "issuetype"]

_KG_FIELDS = _ISSUE_FIELDS + ["description", "project", "labels"]


def fetch_active_jira_tickets() -> str:
    """Fetch active Jira tickets using the configured JQL filter."""
    issues = _search(config.JIRA_JQL_FILTER, fields=_ISSUE_FIELDS)
    return _format_issues(issues)


def fetch_active_jira_tickets_data() -> list[dict]:
    """Fetch active Jira tickets as normalized dicts for knowledge-graph extraction.

    Unlike fetch_active_jira_tickets (which returns a formatted text block for
    Gemini context), this returns structured records including the ticket
    description so the knowledge graph can extract decisions, blockers, and
    owners from each ticket.
    """
    issues = _search(config.JIRA_JQL_FILTER, fields=_KG_FIELDS)
    return [_normalize_issue(i) for i in issues]


def _normalize_issue(issue: dict) -> dict:
    """Flatten a raw Jira issue dict into a normalized record."""
    fields = issue.get("fields", {})
    return {
        "key": issue.get("key", ""),
        "summary": fields.get("summary", ""),
        "status": (fields.get("status") or {}).get("name", ""),
        "priority": (fields.get("priority") or {}).get("name", ""),
        "issue_type": (fields.get("issuetype") or {}).get("name", ""),
        "assignee": (fields.get("assignee") or {}).get("displayName", ""),
        "reporter": (fields.get("reporter") or {}).get("displayName", ""),
        "project": (fields.get("project") or {}).get("name", ""),
        "labels": fields.get("labels", []) or [],
        "updated": (fields.get("updated") or "")[:10],
        "description": _adf_to_text(fields.get("description")).strip(),
    }


def _adf_to_text(node) -> str:
    """Flatten an Atlassian Document Format (ADF) node into plain text.

    Jira Cloud REST v3 returns rich-text fields (e.g. description) as nested
    ADF JSON rather than plain strings. This walks the tree collecting text
    nodes, inserting newlines at block boundaries for readability.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_adf_to_text(n) for n in node)
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type == "text":
            return node.get("text", "")
        if node_type == "hardBreak":
            return "\n"
        text = _adf_to_text(node.get("content"))
        if node_type in ("paragraph", "heading", "blockquote", "listItem", "codeBlock"):
            return text + "\n"
        return text
    return ""


def search_jira_tickets(query: str) -> str:
    """Search Jira tickets with a text query (wrapped in JQL text search)."""
    sanitized = query.replace("\\", "\\\\").replace('"', '\\"')
    jql = f'text ~ "{sanitized}" ORDER BY updated DESC'
    issues = _search(jql, max_results=20, fields=_ISSUE_FIELDS)
    return _format_issues(issues)


def get_jira_issue(issue_key: str) -> str:
    """Fetch details for a specific Jira issue by key (e.g. PROJ-123)."""
    url = f"{_base_url()}/issue/{issue_key}"
    try:
        resp = httpx.get(url, headers=_get_auth_header(), timeout=_TIMEOUT)
        resp.raise_for_status()
        return _format_issues([resp.json()])
    except Exception as exc:
        print(f"Jira: get_issue({issue_key}) failed — {exc}")
        return ""


def _format_issues(issues: list[dict]) -> str:
    """Format a list of Jira issue dicts into a text block for Gemini context."""
    if not issues:
        return ""

    lines = []
    for issue in issues:
        key = issue.get("key", "?")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "(no summary)")

        status = (fields.get("status") or {}).get("name", "Unknown")
        priority = (fields.get("priority") or {}).get("name", "")
        issue_type = (fields.get("issuetype") or {}).get("name", "")
        assignee = (fields.get("assignee") or {}).get("displayName", "Unassigned")
        reporter = (fields.get("reporter") or {}).get("displayName", "")
        updated = fields.get("updated", "")[:10]

        parts = [f"- {key}: {summary}"]
        parts.append(f"  Type: {issue_type} | Status: {status} | Priority: {priority}")
        parts.append(f"  Assignee: {assignee}")
        if reporter:
            parts.append(f"  Reporter: {reporter}")
        if updated:
            parts.append(f"  Updated: {updated}")

        lines.append("\n".join(parts))

    return "\n\n".join(lines)


def format_jira_tickets_for_context(tickets_text: str) -> str:
    """Format raw Jira ticket data into a text block for Gemini context."""
    if not tickets_text:
        return "No active Jira tickets found."
    return tickets_text


# ── Write operations ─────────────────────────────────────────
# These mutate shared, team-visible Jira data. They are the raw REST layer and
# MUST only be invoked from the approved-execution path (never directly by the
# agent loop). The agent layer queues these behind explicit user approval.


def _text_to_adf(text: str) -> dict:
    """Build a minimal Atlassian Document Format doc from plain text.

    Jira Cloud REST v3 requires rich-text fields (description, comment body) as
    ADF JSON. Each non-empty line becomes a paragraph.
    """
    lines = (text or "").split("\n")
    content = []
    for line in lines:
        para: dict = {"type": "paragraph", "content": []}
        if line:
            para["content"].append({"type": "text", "text": line})
        content.append(para)
    if not content:
        content = [{"type": "paragraph", "content": []}]
    return {"type": "doc", "version": 1, "content": content}


def _issue_browse_url(key: str) -> str:
    site = config.JIRA_SITE_URL.rstrip("/")
    if not site.startswith("http"):
        site = f"https://{site}"
    return f"{site}/browse/{key}"


def create_jira_ticket(project_key: str, summary: str, description: str = "",
                       issue_type: str = "Task", priority: str | None = None) -> dict:
    """Create a Jira issue. Returns {success, key, url} or {success: False, error}."""
    fields: dict = {
        "project": {"key": project_key},
        "summary": summary,
        "issuetype": {"name": issue_type},
    }
    if description:
        fields["description"] = _text_to_adf(description)
    if priority:
        fields["priority"] = {"name": priority}

    try:
        resp = httpx.post(f"{_base_url()}/issue", headers=_get_auth_header(),
                          json={"fields": fields}, timeout=_TIMEOUT)
        resp.raise_for_status()
        key = resp.json().get("key", "")
        return {"success": True, "key": key, "url": _issue_browse_url(key)}
    except Exception as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")
        print(f"Jira: create_ticket failed — {exc} {detail[:300]}")
        return {"success": False, "error": str(exc), "detail": detail[:300]}


def add_jira_comment(issue_key: str, comment: str) -> dict:
    """Add a comment to a Jira issue. Returns {success, key, url} or error."""
    try:
        resp = httpx.post(
            f"{_base_url()}/issue/{issue_key}/comment",
            headers=_get_auth_header(),
            json={"body": _text_to_adf(comment)},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return {"success": True, "key": issue_key, "url": _issue_browse_url(issue_key)}
    except Exception as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")
        print(f"Jira: add_comment({issue_key}) failed — {exc} {detail[:300]}")
        return {"success": False, "error": str(exc), "detail": detail[:300]}


def list_jira_transitions(issue_key: str) -> list[dict]:
    """List available status transitions for an issue as [{id, name, to_status}]."""
    try:
        resp = httpx.get(f"{_base_url()}/issue/{issue_key}/transitions",
                         headers=_get_auth_header(), timeout=_TIMEOUT)
        resp.raise_for_status()
        out = []
        for tr in resp.json().get("transitions", []):
            out.append({
                "id": tr.get("id", ""),
                "name": tr.get("name", ""),
                "to_status": (tr.get("to") or {}).get("name", ""),
            })
        return out
    except Exception as exc:
        print(f"Jira: list_transitions({issue_key}) failed — {exc}")
        return []


def transition_jira_ticket(issue_key: str, transition_name: str) -> dict:
    """Move an issue to a new status by transition name (case-insensitive).

    Resolves the transition name to its id against the issue's currently
    available transitions, then applies it. Returns {success, ...} or an error
    listing the valid transitions when the name does not match.
    """
    transitions = list_jira_transitions(issue_key)
    if not transitions:
        return {"success": False, "error": f"No transitions available for {issue_key}"}

    target = (transition_name or "").strip().lower()
    match = next(
        (t for t in transitions
         if t["name"].lower() == target or t["to_status"].lower() == target),
        None,
    )
    if not match:
        valid = ", ".join(f"{t['name']} -> {t['to_status']}" for t in transitions)
        return {"success": False,
                "error": f"No transition '{transition_name}' for {issue_key}",
                "valid_transitions": valid}

    try:
        resp = httpx.post(
            f"{_base_url()}/issue/{issue_key}/transitions",
            headers=_get_auth_header(),
            json={"transition": {"id": match["id"]}},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return {"success": True, "key": issue_key, "to_status": match["to_status"],
                "url": _issue_browse_url(issue_key)}
    except Exception as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")
        print(f"Jira: transition({issue_key}) failed — {exc} {detail[:300]}")
        return {"success": False, "error": str(exc), "detail": detail[:300]}
