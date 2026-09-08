"""Connection health checks for all of Momo's external connectors.

Called by POST /connection-health (Cloud Scheduler, e.g. hourly). Each probe
does the cheapest possible authenticated round-trip and classifies the result:

    ok           - authenticated round-trip succeeded
    auth_failed  - credentials expired/invalid (401/403/dead-token quirks)
    unavailable  - reachable problem that is NOT auth (network, 5xx, timeout)
    skipped      - connector disabled by config

State is persisted per connector in the `connection_health` Firestore
collection so Chat alerts fire ONLY on transitions (→ auth_failed and the
recovery back to ok), plus a reminder at most every 12h while still failing.
Google/Granola failure delivery uses the providers' shared cooldowns, including
alerts sent inside probes. If Firestore is down, state persistence is skipped;
failed provider delivery falls back to a notice asking for a fresh link.

All heavyweight imports happen inside the probes so importing this module is
cheap and probes stay independently mockable.
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import config
from connection_errors import ExternalAuthError

_COLLECTION = "connection_health"
_PROBE_TIMEOUT_SECONDS = 10
_REMINDER_COOLDOWN_SECONDS = 12 * 60 * 60


def get_db():
    """Thin seam over conversation_store.get_db (monkeypatchable in tests)."""
    from conversation_store import get_db as _get_db
    return _get_db()


def _send_alert(text: str) -> bool:
    """Best-effort Chat alert; never raises into the health check."""
    from reauth_service import send_auth_notification
    return send_auth_notification(text)


def _result(connector: str, enabled: bool, status: str, *,
            error_kind: str | None = None, error_message: str = "",
            user_message: str = "") -> dict:
    return {
        "connector": connector,
        "enabled": enabled,
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "error_kind": error_kind,
        "error_message": error_message,
        "user_message": user_message,
    }


def _classify_exception(connector: str, enabled: bool, exc: Exception) -> dict:
    if isinstance(exc, ExternalAuthError):
        return _result(
            connector, enabled, "auth_failed",
            error_kind="auth", error_message=str(exc),
            user_message=getattr(exc, "reconnect_hint", "") or f"{connector} needs re-auth",
        )
    return _result(
        connector, enabled, "unavailable",
        error_kind="unavailable", error_message=str(exc),
        user_message=f"{connector} is unreachable (non-auth failure)",
    )


# ── Probes ───────────────────────────────────────────────────
# Each probe returns a full result dict and must not raise (raising is treated
# as unavailable by the runner).


def probe_jira() -> dict:
    if not config.JIRA_ENABLED:
        return _result("jira", False, "skipped")
    from jira_service import check_jira_auth
    ok, detail = check_jira_auth()
    if ok is True:
        return _result("jira", True, "ok")
    if ok is False:
        return _result(
            "jira", True, "auth_failed",
            error_kind="auth", error_message=detail,
            user_message="Jira credentials expired — regenerate the API token and update JIRA_API_TOKEN",
        )
    return _result("jira", True, "unavailable",
                   error_kind="unavailable", error_message=detail,
                   user_message="Jira is unreachable (non-auth failure)")


def probe_google_workspace() -> dict:
    try:
        from google_auth import get_credentials, ReauthRequiredError
        from googleapiclient.discovery import build
        try:
            creds = get_credentials()
            svc = build("gmail", "v1", credentials=creds)
            svc.users().getProfile(userId="me").execute()
            return _result("google_workspace", True, "ok")
        except ReauthRequiredError as exc:
            return _result(
                "google_workspace", True, "auth_failed",
                error_kind="auth", error_message=str(exc),
                user_message="Google Workspace needs re-auth — ask momo for a fresh reconnect link",
            )
        except Exception as exc:
            from google_auth import classify_google_auth_error
            auth_err = classify_google_auth_error(exc, source="connection_health")
            if auth_err is not None:
                return _classify_exception("google_workspace", True, auth_err)
            return _classify_exception("google_workspace", True, exc)
    except Exception as exc:
        return _classify_exception("google_workspace", True, exc)


def probe_google_chat() -> dict:
    if not config.CHAT_SPACE_ID:
        return _result("google_chat", False, "skipped")
    try:
        from chat_service import _get_chat_session
        session = _get_chat_session()
        # Non-mutating space GET (best effort — chat.bot scope can read spaces
        # the bot belongs to). 401/403 = auth; anything else unsupported/other
        # is unavailable, NOT auth.
        resp = session.get(f"https://chat.googleapis.com/v1/{config.CHAT_SPACE_ID}",
                           timeout=_PROBE_TIMEOUT_SECONDS)
        if resp.status_code in (401, 403):
            return _result(
                "google_chat", True, "auth_failed",
                error_kind="auth", error_message=f"space GET returned {resp.status_code}",
                user_message="Google Chat bot credentials rejected — check the service account",
            )
        if 200 <= resp.status_code < 300:
            return _result("google_chat", True, "ok")
        return _result("google_chat", True, "unavailable",
                       error_kind="unavailable",
                       error_message=f"space GET returned {resp.status_code}",
                       user_message="Google Chat probe unsupported/unavailable (not an auth failure)")
    except Exception as exc:
        text = str(exc)
        if "invalid_grant" in text.lower() or "RefreshError" in type(exc).__name__:
            return _result("google_chat", True, "auth_failed",
                           error_kind="auth", error_message=text,
                           user_message="Google Chat ADC credentials failed to refresh")
        return _classify_exception("google_chat", True, exc)


def probe_granola() -> dict:
    if not config.GRANOLA_ENABLED:
        return _result("granola", False, "skipped")
    try:
        import granola_service
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        token = granola_service._load_token()
        if not token:
            return _result(
                "granola", True, "auth_failed",
                error_kind="auth", error_message="no Granola token available",
                user_message="Granola needs re-auth — ask momo for a fresh reconnect link",
            )

        async def _initialize_and_list():
            auth = granola_service._BearerAuth(token)
            async with streamablehttp_client(config.GRANOLA_MCP_URL, auth=auth) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await session.list_tools()

        granola_service._run(_initialize_and_list(), timeout=_PROBE_TIMEOUT_SECONDS)
        return _result("granola", True, "ok")
    except Exception as exc:
        try:
            from granola_service import _is_auth_error
            if _is_auth_error(exc):
                return _result(
                    "granola", True, "auth_failed",
                    error_kind="auth", error_message=str(exc),
                    user_message="Granola needs re-auth — ask momo for a fresh reconnect link",
                )
        except Exception:
            pass
        return _classify_exception("granola", True, exc)


def _probe_mcp_server(srv: dict) -> dict:
    name = srv.get("name", "")
    connector = f"mcp:{name}"
    try:
        # Bypass the discovery cache: initialize + list_tools directly.
        from mcp_client import _async_list_tools, _is_auth_error, _load_token, _run

        if srv.get("auth") == "oauth":
            token = _load_token(name)
            if not token:
                return _result(
                    connector, True, "auth_failed",
                    error_kind="auth", error_message=f"no auth token for MCP server '{name}'",
                    user_message=f"run `python mcp_auth_setup.py {name}` to reconnect",
                )
        else:
            token = srv.get("bearer_token", "")

        _run(_async_list_tools(srv["url"], token), timeout=_PROBE_TIMEOUT_SECONDS)
        return _result(connector, True, "ok")
    except Exception as exc:
        try:
            from mcp_client import _is_auth_error
            if _is_auth_error(exc):
                return _result(
                    connector, True, "auth_failed",
                    error_kind="auth", error_message=str(exc),
                    user_message=f"run `python mcp_auth_setup.py {name}` to reconnect",
                )
        except Exception:
            pass
        return _classify_exception(connector, True, exc)


def probe_anthropic() -> dict:
    try:
        import anthropic
        from claude_client import get_client
        get_client().messages.create(
            model=config.CLAUDE_MODEL_HAIKU,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        return _result("anthropic", True, "ok")
    except Exception as exc:
        try:
            import anthropic
            if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
                return _result(
                    "anthropic", True, "auth_failed",
                    error_kind="auth", error_message=str(exc),
                    user_message="Anthropic API key rejected — check/rotate ANTHROPIC_API_KEY",
                )
        except Exception:
            pass
        return _classify_exception("anthropic", True, exc)


def probe_gemini() -> dict:
    if not config.KNOWLEDGE_GRAPH_ENABLED or not config.GEMINI_API_KEY:
        return _result("gemini", False, "skipped")
    try:
        from knowledge_graph import _get_embedding
        _get_embedding("health")
        return _result("gemini", True, "ok")
    except Exception as exc:
        return _classify_exception("gemini", True, exc)


def probe_firestore() -> dict:
    try:
        db = get_db()
        db.collection(_COLLECTION).document("_probe").get()
        return _result("firestore", True, "ok")
    except Exception as exc:
        text = str(exc)
        if "403" in text or "401" in text or "PermissionDenied" in type(exc).__name__:
            return _result("firestore", True, "auth_failed",
                           error_kind="auth", error_message=text,
                           user_message="Firestore credentials rejected — check the service account")
        return _classify_exception("firestore", True, exc)


def probe_langfuse() -> dict:
    if (not config.LANGFUSE_TRACING_ENABLED
            or not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY):
        return _result("langfuse", False, "skipped")
    try:
        import httpx
        resp = httpx.get(
            f"{config.LANGFUSE_BASE_URL.rstrip('/')}/api/public/projects",
            auth=(config.LANGFUSE_PUBLIC_KEY, config.LANGFUSE_SECRET_KEY),
            timeout=5,
        )
        if resp.status_code in (401, 403):
            return _result("langfuse", True, "auth_failed",
                           error_kind="auth",
                           error_message=f"/api/public/projects returned {resp.status_code}",
                           user_message="Langfuse keys rejected — check LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY")
        resp.raise_for_status()
        return _result("langfuse", True, "ok")
    except Exception as exc:
        # Langfuse is observability-only: classify but it never blocks health.
        return _classify_exception("langfuse", True, exc)


def _build_registry() -> list:
    """(connector_name, probe_callable) for every known connector."""
    probes = [
        ("jira", probe_jira),
        ("google_workspace", probe_google_workspace),
        ("google_chat", probe_google_chat),
        ("granola", probe_granola),
    ]
    if config.MCP_ENABLED:
        for srv in config.MCP_SERVERS:
            if not srv.get("enabled", True) or not srv.get("name"):
                continue
            probes.append((f"mcp:{srv['name']}",
                           lambda s=srv: _probe_mcp_server(s)))
    else:
        probes.append(("mcp", lambda: _result("mcp", False, "skipped")))
    probes.extend([
        ("anthropic", probe_anthropic),
        ("gemini", probe_gemini),
        ("firestore", probe_firestore),
        ("langfuse", probe_langfuse),
    ])
    return probes


# ── State persistence + transition alerts ────────────────────


def _alert_text(result: dict, kind: str) -> str:
    connector = result["connector"]
    if kind == "failure":
        detail = result.get("user_message") or result.get("error_message") or ""
        return (
            f"🔴 *{connector} connection needs re-auth*\n\n"
            f"momo's {connector} credentials stopped working — data from this "
            "source is unavailable until it's reconnected.\n\n"
            + (f"👉 {detail}" if detail else "")
        ).strip()
    if kind == "reminder":
        return (
            f"🔴 *{connector} still needs re-auth* (reminder)\n\n"
            f"{result.get('user_message') or ''}"
        ).strip()
    return f"🟢 *{connector} connection restored* — the latest connection check passed."


def _send_failure_alert(result: dict, kind: str) -> bool:
    """One automatic failure/reminder delivery path per reauth provider."""
    if result["connector"] == "google_workspace":
        from google_auth import _send_throttled_reauth_alert
        return _send_throttled_reauth_alert()
    if result["connector"] == "granola":
        from granola_service import send_reauth_alert
        return send_reauth_alert()
    return _send_alert(_alert_text(result, kind))


def _clear_provider_alert_cooldown(connector: str) -> None:
    """A confirmed recovery ends the provider's previous failure episode."""
    try:
        if connector == "google_workspace":
            from google_auth import _clear_reauth_alert_cooldown
        elif connector == "granola":
            from granola_service import _clear_reauth_alert_cooldown
        else:
            return
        _clear_reauth_alert_cooldown()
    except Exception:
        print(f"connection_health: could not clear reauth cooldown for '{connector}'")


def _persist_and_alert(result: dict) -> None:
    """Write per-connector state and alert ONLY on auth transitions.

    Alerts: ok/unknown→auth_failed (immediately), auth_failed→auth_failed
    (at most every 12h), auth_failed→ok (recovery note). If Firestore is down,
    skip persistence and best-effort alert on failures."""
    connector = result["connector"]
    status = result["status"]
    now = time.time()

    try:
        doc_ref = get_db().collection(_COLLECTION).document(connector)
        snapshot = doc_ref.get()
        prev = snapshot.to_dict() if getattr(snapshot, "exists", False) else {}
    except Exception as exc:
        print(f"connection_health: Firestore unavailable for '{connector}' ({exc}) — "
              "skipping persistence, best-effort alerting")
        if status == "auth_failed":
            try:
                delivered = _send_failure_alert(result, "failure")
            except Exception:
                delivered = False
            if not delivered and connector in ("google_workspace", "granola"):
                # Without state storage the provider may be unable to check its
                # cooldown or mint a usable ticket. Send a link-free notice.
                _send_alert(_alert_text(result, "failure"))
        return

    prev = prev or {}
    prev_status = prev.get("status")
    last_alerted_at = float(prev.get("last_alerted_at") or 0)
    last_changed_at = float(prev.get("last_changed_at") or 0) or now

    alerted_at = last_alerted_at
    if status == "auth_failed":
        # Provider senders own the cooldown, not this independent health doc.
        # In particular, a probe may already have delivered the alert.
        if (connector in ("google_workspace", "granola")
                or prev_status != "auth_failed"
                or not last_alerted_at
                or now - last_alerted_at >= _REMINDER_COOLDOWN_SECONDS):
            kind = "reminder" if prev_status == "auth_failed" else "failure"
            if _send_failure_alert(result, kind):
                alerted_at = now
    elif status == "ok" and prev_status == "auth_failed":
        _clear_provider_alert_cooldown(connector)
        if _send_alert(_alert_text(result, "recovery")):
            alerted_at = now

    doc = {
        "status": status,
        "enabled": result.get("enabled", True),
        "last_checked_at": now,
        "last_changed_at": now if status != prev_status else last_changed_at,
        "last_alerted_at": alerted_at,
        "error_kind": result.get("error_kind"),
        "error_message": (result.get("error_message") or "")[:1000],
        "user_message": result.get("user_message") or "",
    }
    try:
        doc_ref.set(doc)
    except Exception as exc:
        print(f"connection_health: failed to persist state for '{connector}': {exc}")


# ── Runner ───────────────────────────────────────────────────


def run_connection_health_check(probes=None) -> dict:
    """Run all probes in parallel, persist state, alert on transitions.

    Returns {"status": "ok"|"degraded", "connectors": [...]}. `probes` is
    injectable for tests: a list of (name, callable) pairs."""
    registry = probes if probes is not None else _build_registry()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(len(registry), 8) or 1) as pool:
        futures = {pool.submit(fn): name for name, fn in registry}
        for future, name in futures.items():
            try:
                results.append(future.result(timeout=_PROBE_TIMEOUT_SECONDS + 5))
            except Exception as exc:
                traceback.print_exc()
                results.append(_result(name, True, "unavailable",
                                       error_kind="unavailable",
                                       error_message=f"probe error/timeout: {exc}",
                                       user_message=f"{name} probe did not complete"))

    for result in results:
        try:
            _persist_and_alert(result)
        except Exception as exc:
            print(f"connection_health: persist/alert failed for "
                  f"'{result.get('connector')}': {exc}")

    degraded = any(r["status"] in ("auth_failed", "unavailable") for r in results)
    return {"status": "degraded" if degraded else "ok", "connectors": results}
