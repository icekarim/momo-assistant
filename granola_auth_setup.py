"""
Run this ONCE locally to authenticate with Granola and obtain an OAuth token.

Granola MCP uses browser-based OAuth 2.0 with Dynamic Client Registration (DCR).
No client ID or client secret is required — credentials are handled automatically.

Usage:
    python granola_auth_setup.py

The token is saved to:
  1. granola_token.json (local dev)
  2. Firestore granola_auth/token (Cloud Run — auto-refreshes there too)

No manual token management needed after this.
"""

import asyncio
import base64
import hashlib
import json
import os
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import httpx

import config

GRANOLA_MCP_URL = config.GRANOLA_MCP_URL
TOKEN_FILE = "granola_token.json"
REDIRECT_PORT = 9876
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"

_auth_code = None
_auth_error = None


class _CallbackHandler(BaseHTTPRequestHandler):
    """Tiny HTTP handler that captures the OAuth callback."""

    def do_GET(self):
        global _auth_code, _auth_error
        query = parse_qs(urlparse(self.path).query)
        error = query.get("error", [None])[0]
        if error:
            _auth_error = error
            body = f"<html><body><h2>Auth failed: {error}</h2></body></html>".encode()
        else:
            _auth_code = query.get("code", [None])[0]
            body = b"<html><body><h2>Granola auth complete - you can close this tab.</h2></body></html>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


async def _discover_oauth_metadata():
    """Fetch the MCP server's OAuth metadata via RFC 8414 discovery."""
    base = GRANOLA_MCP_URL.rstrip("/")
    well_known = f"{base}/.well-known/oauth-authorization-server"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(well_known)
        if resp.status_code == 200:
            return resp.json()

        # Fallback: try the base domain
        from urllib.parse import urlparse as _up

        parsed = _up(base)
        root_wk = f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-authorization-server"
        resp = await client.get(root_wk)
        if resp.status_code == 200:
            return resp.json()

    return None


async def _register_client(registration_endpoint: str):
    """Dynamically register a client (DCR)."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.post(
            registration_endpoint,
            json={
                "client_name": "Momo Assistant",
                "redirect_uris": [REDIRECT_URI],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        resp.raise_for_status()
        return resp.json()


def _pkce_pair():
    """Generate a PKCE code_verifier and code_challenge (S256)."""
    verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


async def _exchange_code(token_endpoint: str, code: str, client_id: str, code_verifier: str):
    """Exchange the authorization code for tokens."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": client_id,
                "code_verifier": code_verifier,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def run_auth_flow():
    global _auth_code

    print("Discovering Granola OAuth endpoints...")
    meta = await _discover_oauth_metadata()
    if not meta:
        print("Could not discover OAuth metadata from the Granola MCP server.")
        print(f"Tried: {GRANOLA_MCP_URL}/.well-known/oauth-authorization-server")
        print("\nFallback: if you already have a Granola token, set it directly:")
        print("  export GRANOLA_TOKEN=<your-token>")
        return

    auth_endpoint = meta["authorization_endpoint"]
    token_endpoint = meta["token_endpoint"]
    reg_endpoint = meta.get("registration_endpoint")

    client_id = None
    if reg_endpoint:
        print("Registering dynamic client...")
        reg = await _register_client(reg_endpoint)
        client_id = reg["client_id"]
        print(f"  Client ID: {client_id}")
    else:
        client_id = "momo-assistant"

    code_verifier, code_challenge = _pkce_pair()

    auth_url = (
        f"{auth_endpoint}"
        f"?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&code_challenge={code_challenge}"
        f"&code_challenge_method=S256"
    )

    print(f"\nOpening browser for Granola sign-in...\n")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", REDIRECT_PORT), _CallbackHandler)
    print(f"Waiting for callback on http://localhost:{REDIRECT_PORT}/callback ...")
    while _auth_code is None and _auth_error is None:
        server.handle_request()
    server.server_close()

    if _auth_error:
        print(f"\nAuth failed with error: {_auth_error}")
        return

    print("Exchanging code for token...")
    tokens = await _exchange_code(token_endpoint, _auth_code, client_id, code_verifier)

    tokens["_token_endpoint"] = token_endpoint
    tokens["_client_id"] = client_id

    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)
    print(f"\nToken saved to {TOKEN_FILE}")

    # Also push to Firestore so Cloud Run has it immediately
    try:
        from granola_service import _write_token_to_firestore
        _write_token_to_firestore(tokens)
        print("Token synced to Firestore (Cloud Run will use this)")
    except Exception as exc:
        print(f"Firestore sync skipped: {exc}")
        print("  (Cloud Run won't have the token until you deploy with GRANOLA_TOKEN)")

    print("\nDone. Momo will auto-refresh the token — no need to re-authenticate.")


def main():
    asyncio.run(run_auth_flow())


if __name__ == "__main__":
    main()
