"""Granola MCP client — fetches meeting notes via the official Granola MCP server.

Token lifecycle:
  1. User runs `python granola_auth_setup.py` once
     → saves to granola_token.json AND Firestore
  2. This module loads the token: in-memory cache → local file → Firestore
  3. When the access_token nears expiry, auto-refreshes via refresh_token
  4. Refreshed token is written back to both file (if writable) and Firestore
  5. On Cloud Run there's no file — Firestore handles everything automatically
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import config

_TOKEN_FILE = os.getenv("GRANOLA_TOKEN_FILE", "granola_token.json")
_GRANOLA_TOKEN_JSON_ENV = os.getenv("GRANOLA_TOKEN_JSON", "")
_FIRESTORE_GRANOLA_COLLECTION = "granola_auth"
_FIRESTORE_GRANOLA_DOC = "token"

_cached_token: dict | None = None
_token_loaded_at: float = 0


# ── Token persistence (Firestore) ───────────────────────────


def _get_db():
    """Lazy import to avoid circular deps and let Firestore init on first use."""
    from conversation_store import get_db
    return get_db()


def _read_token_from_firestore() -> dict | None:
    try:
        db = _get_db()
        doc = db.collection(_FIRESTORE_GRANOLA_COLLECTION).document(_FIRESTORE_GRANOLA_DOC).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as exc:
        print(f"Granola: Firestore token read failed: {exc}")
    return None


def _write_token_to_firestore(token_data: dict):
    try:
        db = _get_db()
        db.collection(_FIRESTORE_GRANOLA_COLLECTION).document(_FIRESTORE_GRANOLA_DOC).set(token_data)
    except Exception as exc:
        print(f"Granola: Firestore token write failed: {exc}")


def _write_token_to_file(token_data: dict):
    try:
        with open(_TOKEN_FILE, "w") as f:
            json.dump(token_data, f, indent=2)
    except OSError:
        pass  # read-only filesystem on Cloud Run — that's fine


def _persist_token(token_data: dict):
    """Write refreshed token to all available stores."""
    _write_token_to_file(token_data)
    _write_token_to_firestore(token_data)


# ── Token loading & refresh ──────────────────────────────────


def _load_token() -> str | None:
    """Return a valid access token. Resolution order:
    1. In-memory cache (if not expired)
    2. Local file (granola_token.json)
    3. Firestore (granola_auth/token)
    4. GRANOLA_TOKEN env var (static fallback, no refresh)
    """
    global _cached_token, _token_loaded_at

    if _cached_token and not _is_expired():
        return _cached_token.get("access_token")

    # Try local file first (fast, works offline)
    if os.path.exists(_TOKEN_FILE):
        with open(_TOKEN_FILE) as f:
            _cached_token = json.load(f)
        _token_loaded_at = time.time()

        if _is_expired():
            _refresh()
        return _cached_token.get("access_token") if _cached_token else None

    # Firestore (Cloud Run path)
    fs_token = _read_token_from_firestore()
    if fs_token:
        _cached_token = fs_token
        _token_loaded_at = time.time()

        if _is_expired():
            _refresh()
        return _cached_token.get("access_token") if _cached_token else None

    # GRANOLA_TOKEN_JSON env var — seed Firestore on first boot, then Firestore
    # handles all future refreshes automatically
    if _GRANOLA_TOKEN_JSON_ENV:
        try:
            _cached_token = json.loads(_GRANOLA_TOKEN_JSON_ENV)
            _token_loaded_at = time.time()
            _write_token_to_firestore(_cached_token)
            print("Granola: seeded Firestore from GRANOLA_TOKEN_JSON env var")

            if _is_expired():
                _refresh()
            return _cached_token.get("access_token") if _cached_token else None
        except json.JSONDecodeError:
            print("Granola: GRANOLA_TOKEN_JSON is not valid JSON")

    # Static env var (no refresh possible, last resort)
    if config.GRANOLA_TOKEN:
        return config.GRANOLA_TOKEN

    print("Granola: no token found. Run `python granola_auth_setup.py` first.")
    return None


def _is_expired() -> bool:
    if not _cached_token:
        return True
    expires_in = _cached_token.get("expires_in", 21600)
    elapsed = time.time() - _token_loaded_at
    return elapsed >= (expires_in - 300)  # refresh 5 min early


def _refresh():
    """Use the refresh_token to get a new access_token."""
    global _cached_token, _token_loaded_at

    refresh_token = _cached_token.get("refresh_token") if _cached_token else None
    if not refresh_token:
        print("Granola: no refresh_token available, re-run granola_auth_setup.py")
        _cached_token = None
        return

    token_endpoint = (_cached_token or {}).get("_token_endpoint")
    client_id = (_cached_token or {}).get("_client_id")

    if not token_endpoint or not client_id:
        token_endpoint, client_id = _discover_refresh_params()
        if not token_endpoint:
            print("Granola: could not discover OAuth endpoints for refresh")
            _cached_token = None
            return

    try:
        resp = httpx.post(
            token_endpoint,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
        new_tokens = resp.json()

        new_tokens.setdefault("refresh_token", refresh_token)
        new_tokens["_token_endpoint"] = token_endpoint
        new_tokens["_client_id"] = client_id

        _cached_token = new_tokens
        _token_loaded_at = time.time()

        _persist_token(new_tokens)

        print("Granola: token refreshed successfully")
    except Exception as exc:
        print(f"Granola: token refresh failed: {exc}")
        _cached_token = None


def _discover_refresh_params() -> tuple[str | None, str | None]:
    """Discover token_endpoint and register a client for refresh."""
    base = config.GRANOLA_MCP_URL.rstrip("/")

    for url in [
        f"{base}/.well-known/oauth-authorization-server",
        f"{httpx.URL(base).scheme}://{httpx.URL(base).host}/.well-known/oauth-authorization-server",
    ]:
        try:
            resp = httpx.get(url, follow_redirects=True)
            if resp.status_code == 200:
                meta = resp.json()
                token_endpoint = meta.get("token_endpoint")
                reg_endpoint = meta.get("registration_endpoint")

                client_id = "momo-assistant"
                if reg_endpoint:
                    try:
                        reg_resp = httpx.post(
                            reg_endpoint,
                            json={
                                "client_name": "Momo Assistant",
                                "redirect_uris": ["http://localhost:9876/callback"],
                                "grant_types": ["authorization_code", "refresh_token"],
                                "response_types": ["code"],
                                "token_endpoint_auth_method": "none",
                            },
                            follow_redirects=True,
                        )
                        reg_resp.raise_for_status()
                        client_id = reg_resp.json().get("client_id", client_id)
                    except Exception:
                        pass

                return token_endpoint, client_id
        except Exception:
            continue

    return None, None


# ── MCP transport ────────────────────────────────────────────


class _BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


async def _call_tool(tool_name: str, arguments: dict | None = None):
    """Open a short-lived MCP session and call a single tool."""
    token = _load_token()
    if not token:
        return None

    auth = _BearerAuth(token)

    async with streamablehttp_client(config.GRANOLA_MCP_URL, auth=auth) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments or {})
            return result


def _run(coro, timeout=50):
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


def list_granola_meetings(start_date: str, end_date: str) -> str:
    """List meetings in a date range (YYYY-MM-DD)."""
    result = _run(_call_tool("list_meetings", {
        "start_date": start_date,
        "end_date": end_date,
    }))
    return _extract_text(result)


def get_granola_meeting_notes(query: str) -> str:
    """Search meeting content (notes, action items, attendees)."""
    result = _run(_call_tool("get_meetings", {"query": query}))
    return _extract_text(result)


def get_granola_transcript(meeting_id: str) -> str:
    """Retrieve the raw transcript for a specific meeting."""
    result = _run(_call_tool("get_meeting_transcript", {
        "meeting_id": meeting_id,
    }))
    return _extract_text(result)


def query_granola(query: str) -> str:
    """Natural-language query across all meetings."""
    result = _run(_call_tool("query_granola_meetings", {"query": query}))
    return _extract_text(result)


def fetch_yesterday_meeting_notes() -> str:
    """Fetch notes for all of yesterday's meetings (used in morning briefings)."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        notes = get_granola_meeting_notes(
            f"meetings from {yesterday} with action items and key decisions"
        )
        return notes if notes else ""
    except Exception as exc:
        print(f"Granola: error fetching yesterday's notes: {exc}")
        return ""


def fetch_meeting_notes_for_context(meeting_title: str, meeting_date: str | None = None) -> str:
    """Find Granola notes matching a specific calendar event."""
    query = meeting_title
    if meeting_date:
        query = f"{meeting_title} on {meeting_date}"
    try:
        notes = get_granola_meeting_notes(query)
        return notes if notes else ""
    except Exception as exc:
        print(f"Granola: error fetching notes for '{meeting_title}': {exc}")
        return ""


def format_granola_notes_for_context(notes: str) -> str:
    """Format raw Granola notes into a text block suitable for Gemini context."""
    if not notes:
        return "No meeting notes available from Granola."
    return notes
