"""One-time OAuth setup for an MCP server.

Usage:
    python mcp_auth_setup.py <server_name>
    python mcp_auth_setup.py roktgpt

The server must be registered in config.MCP_SERVERS with auth=oauth.

Token is saved to:
  1. mcp_token_{server_name}.json  (local dev)
  2. Firestore mcp_auth/{server_name}  (Cloud Run — auto-refreshes there too)

The callback path is read from the server config (callback_path field,
default /oauth/callback).  RoktGPT allowlists exactly /oauth/callback on
loopback URIs, so the default is correct for that server.
"""

import asyncio
import base64
import hashlib
import json
import os
import socket
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import httpx

import config

_auth_code: str | None = None
_auth_error: str | None = None
_callback_path: str = "/oauth/callback"


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _auth_code, _auth_error
        parsed = urlparse(self.path)
        if parsed.path != _callback_path:
            self.send_response(404)
            self.end_headers()
            return
        query = parse_qs(parsed.query)
        error = query.get("error", [None])[0]
        if error:
            _auth_error = error
            import html as _html
            body = f"<html><body><h2>Auth failed: {_html.escape(error)}</h2></body></html>".encode()
        else:
            _auth_code = query.get("code", [None])[0]
            body = b"<html><body><h2>Auth complete \xe2\x80\x94 you can close this tab.</h2></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


async def _discover_oauth_metadata(server_url: str) -> dict | None:
    base = server_url.rstrip("/")
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for url in [
            f"{base}/.well-known/oauth-authorization-server",
            f"{httpx.URL(base).scheme}://{httpx.URL(base).host}/.well-known/oauth-authorization-server",
        ]:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                continue
    return None


async def _register_client(registration_endpoint: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.post(
            registration_endpoint,
            json={
                "client_name": "Momo Assistant",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        resp.raise_for_status()
        return resp.json()


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


async def _exchange_code(
    token_endpoint: str,
    code: str,
    client_id: str,
    code_verifier: str,
    redirect_uri: str,
) -> dict:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": code_verifier,
            },
        )
        resp.raise_for_status()
        return resp.json()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


async def run_auth_flow(server_name: str) -> None:
    global _auth_code, _callback_path

    srv = next((s for s in config.MCP_SERVERS if s.get("name") == server_name), None)
    if not srv:
        available = [s["name"] for s in config.MCP_SERVERS if s.get("name")]
        print(f"Error: server '{server_name}' not found in MCP_SERVERS.")
        print(f"Available: {available}")
        sys.exit(1)

    server_url = srv["url"]
    _callback_path = srv.get("callback_path", "/oauth/callback")
    port = _free_port()
    redirect_uri = f"http://localhost:{port}{_callback_path}"

    print(f"Discovering OAuth endpoints for {server_name}...")
    meta = await _discover_oauth_metadata(server_url)
    if not meta:
        print(f"Could not discover OAuth metadata from {server_url}")
        sys.exit(1)

    auth_endpoint = meta["authorization_endpoint"]
    token_endpoint = meta["token_endpoint"]
    reg_endpoint = meta.get("registration_endpoint")

    if reg_endpoint:
        print("Registering dynamic client...")
        reg = await _register_client(reg_endpoint, redirect_uri)
        client_id = reg["client_id"]
        print(f"  Client ID: {client_id}")
    else:
        client_id = f"momo-{server_name}"

    code_verifier, code_challenge = _pkce_pair()

    auth_url = (
        f"{auth_endpoint}"
        f"?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&code_challenge={code_challenge}"
        f"&code_challenge_method=S256"
    )

    print(f"\nOpening browser for {server_name} sign-in...\n")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", port), _CallbackHandler)
    print(f"Waiting for callback on {redirect_uri} ...")
    while _auth_code is None and _auth_error is None:
        server.handle_request()
    server.server_close()

    if _auth_error:
        print(f"\nAuth failed: {_auth_error}")
        sys.exit(1)

    print("Exchanging code for token...")
    import time as _time
    tokens = await _exchange_code(token_endpoint, _auth_code, client_id, code_verifier, redirect_uri)

    tokens["_token_endpoint"] = token_endpoint
    tokens["_client_id"] = client_id
    tokens["_expires_at"] = _time.time() + tokens.get("expires_in", 28800)

    token_file = f"mcp_token_{server_name}.json"
    with open(token_file, "w") as f:
        json.dump(tokens, f, indent=2)
    print(f"\nToken saved to {token_file}")

    try:
        from mcp_client import _write_token_to_firestore
        _write_token_to_firestore(server_name, tokens)
        print("Token synced to Firestore (Cloud Run will use this)")
    except Exception as exc:
        print(f"Firestore sync skipped: {exc}")

    print(f"\nDone. Momo will auto-refresh the {server_name} token automatically.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python mcp_auth_setup.py <server_name>")
        print("Example: python mcp_auth_setup.py roktgpt")
        sys.exit(1)
    asyncio.run(run_auth_flow(sys.argv[1]))


if __name__ == "__main__":
    main()
