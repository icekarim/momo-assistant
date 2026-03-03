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

        if "_expires_at" not in _cached_token:
            file_mtime = os.path.getmtime(_TOKEN_FILE)
            _cached_token["_expires_at"] = file_mtime + _cached_token.get("expires_in", 21600)

        if _is_expired():
            _refresh()
        return _cached_token.get("access_token") if _cached_token else None

    # Firestore (Cloud Run path)
    fs_token = _read_token_from_firestore()

    # GRANOLA_TOKEN_JSON env var — used to seed Firestore or as fallback
    env_token = None
    if _GRANOLA_TOKEN_JSON_ENV:
        try:
            env_token = json.loads(_GRANOLA_TOKEN_JSON_ENV)
        except json.JSONDecodeError:
            print("Granola: GRANOLA_TOKEN_JSON is not valid JSON")

    # Prefer whichever source has the required _client_id for refresh
    chosen = None
    if fs_token and fs_token.get("_client_id"):
        chosen = fs_token
    elif env_token and env_token.get("_client_id"):
        chosen = env_token
        _write_token_to_firestore(chosen)
        print("Granola: seeded Firestore from GRANOLA_TOKEN_JSON env var")
    elif fs_token:
        chosen = fs_token
    elif env_token:
        chosen = env_token

    if chosen:
        _cached_token = chosen
        _token_loaded_at = time.time()

        if "_expires_at" not in _cached_token and _cached_token.get("expires_in"):
            _cached_token["_expires_at"] = time.time() + _cached_token["expires_in"]

        if _is_expired():
            _refresh()
        return _cached_token.get("access_token") if _cached_token else None

    # Static env var (no refresh possible, last resort)
    if config.GRANOLA_TOKEN:
        return config.GRANOLA_TOKEN

    print("Granola: no token found. Run `python granola_auth_setup.py` first.")
    return None


def _is_expired() -> bool:
    if not _cached_token:
        return True
    expires_at = _cached_token.get("_expires_at")
    if expires_at:
        return time.time() >= (expires_at - 300)
    return True


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

    if not client_id:
        print("Granola: _client_id missing from token — re-run `python granola_auth_setup.py`")
        _cached_token = None
        return

    if not token_endpoint:
        token_endpoint = _discover_token_endpoint()
        if not token_endpoint:
            print("Granola: could not discover token endpoint for refresh")
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
        new_tokens["_expires_at"] = time.time() + new_tokens.get("expires_in", 21600)

        _cached_token = new_tokens
        _token_loaded_at = time.time()

        _persist_token(new_tokens)

        print("Granola: token refreshed successfully")
    except Exception as exc:
        print(f"Granola: token refresh failed: {exc}")
        _cached_token = None


def _discover_token_endpoint() -> str | None:
    """Discover just the token_endpoint via OAuth well-known metadata.

    Does NOT register a new client — the refresh_token is bound to the
    client_id from the original auth flow and a new registration would
    produce a mismatched client_id, causing 400 on refresh.
    """
    base = config.GRANOLA_MCP_URL.rstrip("/")

    for url in [
        f"{base}/.well-known/oauth-authorization-server",
        f"{httpx.URL(base).scheme}://{httpx.URL(base).host}/.well-known/oauth-authorization-server",
    ]:
        try:
            resp = httpx.get(url, follow_redirects=True)
            if resp.status_code == 200:
                return resp.json().get("token_endpoint")
        except Exception:
            continue

    return None


# ── MCP transport ────────────────────────────────────────────


class _BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


async def _call_tool(tool_name: str, arguments: dict | None = None):
    """Open a short-lived MCP session and call a single tool.

    Automatically retries once on 401 after forcing a token refresh.
    """
    for attempt in range(2):
        token = _load_token()
        if not token:
            return None

        auth = _BearerAuth(token)

        try:
            async with streamablehttp_client(config.GRANOLA_MCP_URL, auth=auth) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments or {})
                    return result
        except Exception as exc:
            if attempt == 0 and _is_auth_error(exc):
                print("Granola: 401 received, forcing token refresh and retrying...")
                global _cached_token
                _cached_token = None
                continue
            raise


def _is_auth_error(exc: Exception) -> bool:
    """Check if an exception (possibly wrapped in ExceptionGroup) is a 401."""
    if hasattr(exc, 'status_code') and exc.status_code == 401:
        return True
    if hasattr(exc, 'response') and hasattr(exc.response, 'status_code') and exc.response.status_code == 401:
        return True
    if hasattr(exc, 'exceptions'):
        return any(_is_auth_error(sub) for sub in exc.exceptions)
    if '401' in str(exc):
        return True
    return False


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


def list_granola_meetings(time_range: str = "last_30_days") -> str:
    """List meetings in a time range (this_week, last_week, last_30_days)."""
    result = _run(_call_tool("list_meetings", {
        "time_range": time_range,
    }))
    return _extract_text(result)


def get_granola_meeting_notes(query: str) -> str:
    """Search meeting content (notes, action items, attendees)."""
    result = _run(_call_tool("query_granola_meetings", {"query": query}))
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
    """Fetch notes for yesterday's meetings via list_meetings + batch fetch.

    Uses the same reliable list→batch path as the debrief flow instead of
    the flaky query_granola_meetings semantic search.
    """
    import re
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    weekday = datetime.now().weekday()

    try:
        time_range = "last_week" if weekday == 0 else "this_week"
        xml = list_granola_meetings(time_range)
        if not xml:
            return ""

        all_ids = []
        yesterday_ids = []
        for tag in re.finditer(r'<meeting\s+([^>]+)>', xml):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag.group(1)))
            mid = attrs.get("id")
            if not mid:
                continue
            all_ids.append(mid)
            date_val = attrs.get("date", "") or attrs.get("start_date", "")
            if yesterday in date_val:
                yesterday_ids.append(mid)

        target_ids = yesterday_ids if yesterday_ids else all_ids
        if not target_ids:
            return ""

        notes_by_id = fetch_meeting_notes_batch(target_ids[:10])
        return "\n\n".join(notes_by_id.values()).strip()

    except Exception as exc:
        print(f"Granola: error fetching yesterday's notes via list+batch: {exc}")
        return ""


def build_meeting_id_map() -> dict[str, str]:
    """Fetch this week's Granola meetings and return a {lowercase_title: id} map.

    Called once before processing meetings so we don't re-list per meeting.
    """
    import re
    xml = list_granola_meetings("this_week")
    if not xml:
        return {}

    id_map: dict[str, str] = {}
    for match in re.finditer(r'<meeting\s+id="([^"]+)"\s+title="([^"]+)"', xml):
        id_map[match.group(2).strip().lower()] = match.group(1)
    return id_map


def match_meeting_id(title: str, id_map: dict[str, str]) -> str | None:
    """Fuzzy-match a calendar title against the pre-built Granola ID map."""
    title_lower = title.strip().lower()
    for granola_title, mid in id_map.items():
        if title_lower in granola_title or granola_title in title_lower:
            return mid
    return None


def fetch_meeting_notes_batch(meeting_ids: list[str]) -> dict[str, str]:
    """Fetch notes for multiple meetings in a single get_meetings call (max 10).

    Returns {meeting_id: notes_text}. Raises on transport/auth errors.
    """
    if not meeting_ids:
        return {}

    result = _run(_call_tool("get_meetings", {"meeting_ids": meeting_ids[:10]}))
    text = _extract_text(result)

    import re
    notes_by_id: dict[str, str] = {}
    blocks = re.split(r'(?=<meeting\s+id=")', text)
    for block in blocks:
        m = re.match(r'<meeting\s+id="([^"]+)"', block)
        if m:
            notes_by_id[m.group(1)] = block
    return notes_by_id


def format_granola_notes_for_context(notes: str) -> str:
    """Format raw Granola notes into a text block suitable for Gemini context."""
    if not notes:
        return "No meeting notes available from Granola."
    return notes
