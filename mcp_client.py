"""Generic MCP client for Momo.

Token lifecycle (per server):
  1. Run `python mcp_auth_setup.py <server_name>` once locally.
  2. Tokens resolve: in-memory cache → local file → Firestore → env seed.
  3. Auto-refresh when access_token nears expiry (300s buffer).
  4. Rotated refresh tokens (RoktGPT issues a new one per refresh) are
     persisted immediately after every successful refresh.

Server names MUST NOT contain underscores — qualified tool names use the
format `mcp_{server}_{tool}`, so underscore-free names make splitting
unambiguous at call time.
"""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import config

_FIRESTORE_MCP_COLLECTION = "mcp_auth"
_TOOL_CACHE_TTL = 3600
_EXPIRY_BUFFER = 300

_token_cache: dict[str, dict | None] = {}
_tool_discovery_cache: dict[str, dict] = {}


def _sanitize_server_name(name: str) -> str:
    if "_" in name:
        raise ValueError(
            f"MCP server name '{name}' contains underscores, which break the "
            "`mcp_{{server}}_{{tool}}` naming scheme. Use a hyphen-free, "
            "underscore-free name (e.g. 'roktgpt')."
        )
    return name


def _get_server_config(server_name: str) -> dict | None:
    for srv in config.MCP_SERVERS:
        if srv.get("name") == server_name and srv.get("enabled", True):
            return srv
    return None


def _get_db():
    from conversation_store import get_db
    return get_db()


def _read_token_from_firestore(server_name: str) -> dict | None:
    try:
        db = _get_db()
        doc = db.collection(_FIRESTORE_MCP_COLLECTION).document(server_name).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as exc:
        print(f"MCP[{server_name}]: Firestore token read failed: {exc}")
    return None


def _write_token_to_firestore(server_name: str, token_data: dict) -> None:
    try:
        db = _get_db()
        db.collection(_FIRESTORE_MCP_COLLECTION).document(server_name).set(token_data)
    except Exception as exc:
        print(f"MCP[{server_name}]: Firestore token write failed: {exc}")


def _token_file_candidates(server_name: str) -> list[str]:
    primary = os.getenv(f"MCP_TOKEN_FILE_{server_name.upper()}", f"mcp_token_{server_name}.json")
    return [primary, f"/tmp/mcp_token_{server_name}.json"]


def _read_token_from_file(server_name: str) -> tuple[dict | None, str | None]:
    for path in _token_file_candidates(server_name):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f), path
            except Exception as exc:
                print(f"MCP[{server_name}]: token file read error ({path}): {exc}")
    return None, None


def _write_token_to_file(server_name: str, token_data: dict) -> None:
    for path in _token_file_candidates(server_name):
        try:
            with open(path, "w") as f:
                json.dump(token_data, f, indent=2)
            return
        except OSError:
            continue


def _persist_token(server_name: str, token_data: dict) -> None:
    _write_token_to_file(server_name, token_data)
    _write_token_to_firestore(server_name, token_data)


def _is_expired(token: dict | None) -> bool:
    if not token:
        return True
    expires_at = token.get("_expires_at")
    if expires_at:
        return time.time() >= (expires_at - _EXPIRY_BUFFER)
    return True


def _discover_token_endpoint(server_name: str) -> str | None:
    srv = _get_server_config(server_name)
    if not srv:
        return None
    base = srv["url"].rstrip("/")
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


def _refresh(server_name: str) -> None:
    token = _token_cache.get(server_name)
    refresh_token = (token or {}).get("refresh_token")
    if not refresh_token:
        print(f"MCP[{server_name}]: no refresh_token — run `python mcp_auth_setup.py {server_name}`")
        _token_cache[server_name] = None
        return

    token_endpoint = (token or {}).get("_token_endpoint") or _discover_token_endpoint(server_name)
    client_id = (token or {}).get("_client_id")

    if not token_endpoint:
        print(f"MCP[{server_name}]: could not discover token endpoint for refresh")
        _token_cache[server_name] = None
        return

    if not client_id:
        print(f"MCP[{server_name}]: _client_id missing — re-run `python mcp_auth_setup.py {server_name}`")
        _token_cache[server_name] = None
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
        new_tokens["_expires_at"] = time.time() + new_tokens.get("expires_in", 28800)

        _token_cache[server_name] = new_tokens
        _persist_token(server_name, new_tokens)
        print(f"MCP[{server_name}]: token refreshed successfully")
    except Exception as exc:
        print(f"MCP[{server_name}]: token refresh failed: {exc}")
        _token_cache[server_name] = None


def _load_token(server_name: str) -> str | None:
    cached = _token_cache.get(server_name)
    if cached and not _is_expired(cached):
        return cached.get("access_token")

    file_token, file_path = _read_token_from_file(server_name)
    if file_token:
        if "_expires_at" not in file_token:
            try:
                mtime = os.path.getmtime(file_path)
                file_token["_expires_at"] = mtime + file_token.get("expires_in", 28800)
            except OSError:
                file_token["_expires_at"] = time.time() + file_token.get("expires_in", 28800)
        _token_cache[server_name] = file_token
        if _is_expired(file_token):
            _refresh(server_name)
        return (_token_cache.get(server_name) or {}).get("access_token")

    fs_token = _read_token_from_firestore(server_name)

    env_key = f"MCP_TOKEN_JSON_{server_name.upper()}"
    env_token = None
    env_raw = os.getenv(env_key, "")
    if env_raw:
        try:
            env_token = json.loads(env_raw)
        except json.JSONDecodeError:
            print(f"MCP[{server_name}]: {env_key} is not valid JSON")

    chosen = None
    if fs_token and fs_token.get("_client_id"):
        chosen = fs_token
    elif env_token and env_token.get("_client_id"):
        chosen = env_token
        _write_token_to_firestore(server_name, chosen)
        print(f"MCP[{server_name}]: seeded Firestore from {env_key}")
    elif fs_token:
        chosen = fs_token
    elif env_token:
        chosen = env_token

    if chosen:
        if "_expires_at" not in chosen and chosen.get("expires_in"):
            chosen["_expires_at"] = time.time() + chosen["expires_in"]
        _token_cache[server_name] = chosen
        if _is_expired(chosen):
            _refresh(server_name)
        return (_token_cache.get(server_name) or {}).get("access_token")

    print(f"MCP[{server_name}]: no token found — run `python mcp_auth_setup.py {server_name}`")
    return None


class _BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


def _is_auth_error(exc: Exception) -> bool:
    if hasattr(exc, "status_code") and exc.status_code == 401:
        return True
    if hasattr(exc, "response") and hasattr(exc.response, "status_code") and exc.response.status_code == 401:
        return True
    if hasattr(exc, "exceptions"):
        return any(_is_auth_error(sub) for sub in exc.exceptions)
    if "401" in str(exc):
        return True
    return False


def _run(coro, timeout: int = 60):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=timeout)
    return asyncio.run(coro)


def _extract_text(result) -> str:
    if result is None:
        return ""
    if hasattr(result, "content"):
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)
    return str(result)


async def _async_list_tools(server_url: str, token: str) -> list:
    auth = _BearerAuth(token)
    async with streamablehttp_client(server_url, auth=auth) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools if hasattr(result, "tools") else []


async def _async_call_tool(server_url: str, tool_name: str, arguments: dict, token: str):
    auth = _BearerAuth(token)
    async with streamablehttp_client(server_url, auth=auth) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments=arguments)


def _build_input_schema(tool) -> dict:
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        return {"type": "object", "properties": {}}
    if isinstance(schema, dict):
        result = schema
    elif hasattr(schema, "model_dump"):
        result = schema.model_dump(exclude_none=True)
    else:
        try:
            result = dict(schema)
        except Exception:
            result = {}
    if not result.get("type"):
        result.setdefault("type", "object")
        result.setdefault("properties", {})
    return result


def list_server_tools(server_name: str) -> list[dict]:
    _sanitize_server_name(server_name)
    cached = _tool_discovery_cache.get(server_name)
    if cached and cached["expires_at"] > time.time():
        return cached["tools"]

    srv = _get_server_config(server_name)
    if not srv:
        print(f"MCP: server '{server_name}' not found in MCP_SERVERS config")
        return []

    if srv.get("auth") == "oauth":
        token = _load_token(server_name)
        if not token:
            print(f"MCP[{server_name}]: no token — skipping tool discovery")
            _tool_discovery_cache[server_name] = {"tools": [], "expires_at": time.time() + 60}
            return []
    else:
        token = srv.get("bearer_token", "")

    allowlist = srv.get("tools")

    try:
        raw_tools = _run(_async_list_tools(srv["url"], token), timeout=15)
    except Exception as exc:
        print(f"MCP[{server_name}]: tool discovery failed: {exc}")
        _tool_discovery_cache[server_name] = {"tools": [], "expires_at": time.time() + 60}
        return []

    decls = []
    for tool in raw_tools:
        tool_name = tool.name if hasattr(tool, "name") else str(tool)
        if allowlist and tool_name not in allowlist:
            continue
        qualified = f"mcp_{server_name}_{tool_name}"
        raw_desc = getattr(tool, "description", "") or ""
        description = f"[via {server_name} MCP] {raw_desc}".strip()
        decls.append({
            "name": qualified,
            "description": description,
            "input_schema": _build_input_schema(tool),
        })

    _tool_discovery_cache[server_name] = {"tools": decls, "expires_at": time.time() + _TOOL_CACHE_TTL}
    print(f"MCP[{server_name}]: discovered {len(decls)} tool(s)")
    return decls


def list_all_mcp_tools() -> list[dict]:
    if not config.MCP_ENABLED:
        return []
    tools = []
    for srv in config.MCP_SERVERS:
        if not srv.get("enabled", True):
            continue
        name = srv.get("name", "")
        if not name:
            continue
        try:
            tools.extend(list_server_tools(name))
        except Exception as exc:
            print(f"MCP: failed to list tools for '{name}': {exc}")
    return tools


def call_mcp_tool(qualified_name: str, args: dict) -> str:
    parts = qualified_name.split("_", 2)
    if len(parts) != 3 or parts[0] != "mcp":
        return f"MCP: invalid qualified tool name '{qualified_name}' (expected mcp_{{server}}_{{tool}})"

    server_name, tool_name = parts[1], parts[2]

    srv = _get_server_config(server_name)
    if not srv:
        return f"MCP: server '{server_name}' not found or disabled"

    for attempt in range(2):
        if srv.get("auth") == "oauth":
            if attempt == 1:
                _token_cache[server_name] = None
            token = _load_token(server_name)
            if not token:
                return (
                    f"MCP[{server_name}]: no auth token — "
                    f"run `python mcp_auth_setup.py {server_name}`"
                )
        else:
            token = srv.get("bearer_token", "")

        try:
            result = _run(
                _async_call_tool(srv["url"], tool_name, args, token),
                timeout=config.MCP_DEFAULT_TIMEOUT,
            )
            return _extract_text(result) or ""
        except Exception as exc:
            if attempt == 0 and _is_auth_error(exc):
                print(f"MCP[{server_name}]: 401 received, forcing token refresh and retrying...")
                continue
            print(f"MCP[{server_name}]: tool call '{tool_name}' failed: {exc}")
            return f"MCP tool error ({server_name}/{tool_name}): {exc}"

    return f"MCP[{server_name}]: tool call failed after retry"


def refresh_all_tokens() -> dict[str, bool]:
    results: dict[str, bool] = {}
    for srv in config.MCP_SERVERS:
        if not srv.get("enabled", True) or srv.get("auth") != "oauth":
            continue
        name = srv.get("name", "")
        if not name:
            continue
        try:
            _token_cache.pop(name, None)
            token = _load_token(name)
            results[name] = bool(token)
        except Exception as exc:
            print(f"MCP[{name}]: proactive refresh failed: {exc}")
            results[name] = False
    return results
