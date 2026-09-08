"""Reconnect links and delivery/history helpers, independent of live credentials."""

import re
from urllib.parse import urlsplit

import config


class ReauthLinkError(RuntimeError):
    """A safe, user-facing link failure (never include credentials or tickets)."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def get_service_url(service_url: str = "") -> str:
    """Return the trusted configured HTTPS origin, never a request-derived host.

    Auth routes live at the root. Reject paths, userinfo, query/fragment data,
    malformed authorities and Chat-link markup rather than produce a bad link.
    Legacy caller overrides may only repeat the configured origin.
    """
    configured = config.MOMO_SERVICE_URL
    if not configured:
        raise ReauthLinkError(
            "missing_config", "MOMO_SERVICE_URL is not configured; a reconnect link is unavailable."
        )
    try:
        if not isinstance(configured, str) or any(
            char.isspace() or ord(char) < 32 or char in '\\<>"\'|`'
            for char in configured
        ):
            raise ValueError
        parsed = urlsplit(configured)
        hostname = parsed.hostname or ""
        labels = hostname.split(".")
        if (
            parsed.scheme != "https"
            or not hostname
            or len(hostname) > 253
            or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                   for label in labels)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or "?" in configured or "#" in configured
            or parsed.netloc.endswith(":")
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        ):
            raise ValueError
        base_url = configured.rstrip("/")
        if service_url and service_url.rstrip("/") != base_url:
            raise ValueError
        return base_url
    except (AttributeError, TypeError, ValueError):
        raise ReauthLinkError(
            "invalid_config", "MOMO_SERVICE_URL must be a trusted absolute HTTPS origin."
        ) from None


def get_reauth_links(service: str = "all") -> dict:
    """Create fresh links on explicit request, with no notification/cooldown IO.

    Each provider is isolated so a failed Google ticket write cannot hide a
    usable Granola link. Imports are lazy; neither provider checks credentials.
    """
    providers = ("google_workspace", "granola")
    if not isinstance(service, str) or service not in (*providers, "all"):
        return {
            "status": "error", "error": "invalid_service",
            "message": "Choose google_workspace, granola, or all.", "providers": {},
        }
    outcomes = {}
    for provider in providers if service == "all" else (service,):
        try:
            if provider == "google_workspace":
                from google_auth import create_reauth_link
            else:
                from granola_service import create_reauth_link
            url = create_reauth_link()
            outcomes[provider] = {"status": "ok", "url": url}
            if provider == "google_workspace":
                outcomes[provider].update(single_use=True, expires_in_seconds=600)
        except ReauthLinkError as exc:
            outcomes[provider] = {"status": "error", "error": exc.code, "message": str(exc)}
        except Exception:
            # Provider/SDK exception text can contain secret material.
            outcomes[provider] = {
                "status": "error", "error": "link_unavailable",
                "message": "Could not create this reconnect link. Please request a fresh link later.",
            }
    successes = sum(outcome["status"] == "ok" for outcome in outcomes.values())
    return {
        "status": "ok" if successes == len(outcomes) else "partial" if successes else "error",
        "providers": outcomes,
    }


def send_auth_notification(text: str) -> bool:
    """Send to the configured Chat space; save the exact delivered text afterward.

    History is best effort and must not turn a successful delivery into a retry.
    Never log message bodies or exceptions that could contain ticket URLs.
    """
    space = config.CHAT_SPACE_ID
    if not space:
        return False
    try:
        from chat_service import send_chat_message
        if send_chat_message(space, text) is not True:
            return False
    except Exception:
        print("Auth notification: Chat delivery failed")
        return False
    try:
        from conversation_store import add_turn, conversation_scope
        add_turn(conversation_scope(space=space), "assistant", text)
    except Exception:
        print("Auth notification: delivered, but conversation history could not be saved")
    return True
