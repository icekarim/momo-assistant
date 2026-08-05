"""Shared typed errors for external connector failures.

Every transport boundary (Jira, Google Workspace, Google Chat, Granola, MCP,
Anthropic, Gemini) classifies auth/permission failures into these types so
callers — the agent loop, briefings, and /connection-health — can distinguish
"credentials expired" from "genuinely no data". Deliberately dependency-free
so any module can import it without circular-import risk.
"""


class ExternalConnectionError(Exception):
    """Base class for classified external-connector failures."""

    def __init__(self, connector: str, message: str = ""):
        self.connector = connector
        self.message = message or f"{connector} connection failed"
        super().__init__(self.message)


class ExternalAuthError(ExternalConnectionError):
    """Credentials expired/invalid (HTTP 401 or equivalent). Reconnect needed."""

    def __init__(self, connector: str, message: str = "", *,
                 status: int | None = None, reconnect_hint: str = ""):
        super().__init__(connector, message or f"{connector} credentials expired or invalid — reconnection required")
        self.status = status
        self.reconnect_hint = reconnect_hint


class ExternalPermissionError(ExternalAuthError):
    """Authenticated but not authorized (HTTP 403 / missing scope)."""


class ExternalUnavailableError(ExternalConnectionError):
    """Connector unreachable or erroring for non-auth reasons."""


def format_for_agent(err: ExternalConnectionError) -> str:
    """Stable one-line tool-result string the agent is prompted to recognize.

    Format (do not change — the system prompt and tests key off it):
        CONNECTION_AUTH_ERROR connector=<name> action=reconnect message="<user_message>"
    """
    message = (getattr(err, "message", "") or str(err)).replace('"', "'")
    hint = getattr(err, "reconnect_hint", "") or ""
    if hint:
        message = f"{message} ({hint})"
    return (
        f'CONNECTION_AUTH_ERROR connector={err.connector} '
        f'action=reconnect message="{message}"'
    )
