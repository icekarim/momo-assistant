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


def fetch_active_jira_tickets() -> str:
    """Fetch active Jira tickets using the configured JQL filter."""
    issues = _search(config.JIRA_JQL_FILTER, fields=_ISSUE_FIELDS)
    return _format_issues(issues)


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
