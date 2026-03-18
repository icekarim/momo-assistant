"""Jira MCP client — fetches tickets via the Atlassian Remote MCP Server.

Auth: static API token from JIRA_API_TOKEN env var, sent as Bearer auth.
Generate one at:
  https://id.atlassian.com/manage-profile/security/api-tokens?autofillToken&expiryDays=max&appId=mcp&selectedScopes=all
"""

import asyncio

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import config


# ── MCP transport ────────────────────────────────────────────


class _BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


async def _call_tool(tool_name: str, arguments: dict | None = None):
    """Open a short-lived MCP session and call a single tool."""
    token = config.JIRA_API_TOKEN
    if not token:
        print("Jira: no API token configured. Set JIRA_API_TOKEN in .env")
        return None

    auth = _BearerAuth(token)

    try:
        async with streamablehttp_client(config.JIRA_MCP_URL, auth=auth) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments or {})
                return result
    except Exception as exc:
        print(f"Jira MCP call '{tool_name}' failed: {exc}")
        raise


def _run(coro, timeout=20):
    """Run an async coroutine from sync code with a timeout."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=timeout)
    return asyncio.run(coro)


def _extract_text(result) -> str:
    """Pull plain text out of an MCP tool result."""
    if result is None:
        return ""
    if hasattr(result, "content"):
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)
    return str(result)


# ── Public helpers ───────────────────────────────────────────


def fetch_active_jira_tickets() -> str:
    """Fetch active Jira tickets using the configured JQL filter.

    Default JQL targets tickets where the user is a request participant
    with status category != Done.
    """
    try:
        result = _run(_call_tool("search_issues", {
            "jql": config.JIRA_JQL_FILTER,
        }))
        return _extract_text(result)
    except Exception as exc:
        print(f"Jira: fetch_active_jira_tickets failed: {exc}")
        return ""


def search_jira_tickets(query: str) -> str:
    """Search Jira tickets with a natural-language or JQL query."""
    try:
        result = _run(_call_tool("search_issues", {
            "query": query,
        }))
        return _extract_text(result)
    except Exception as exc:
        print(f"Jira: search_jira_tickets failed: {exc}")
        return ""


def get_jira_issue(issue_key: str) -> str:
    """Fetch details for a specific Jira issue by key (e.g. PROJ-123)."""
    try:
        result = _run(_call_tool("get_issue", {
            "issue_key": issue_key,
        }))
        return _extract_text(result)
    except Exception as exc:
        print(f"Jira: get_jira_issue({issue_key}) failed: {exc}")
        return ""


def format_jira_tickets_for_context(tickets_text: str) -> str:
    """Format raw Jira ticket data into a text block for Gemini context."""
    if not tickets_text:
        return "No active Jira tickets found."
    return tickets_text
