import os
import json
import threading
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import config

_cached_creds = None
_creds_lock = threading.Lock()


def get_credentials():
    """Get valid Google OAuth credentials, refreshing if needed.
    Caches credentials in memory to avoid re-parsing on every call."""
    global _cached_creds

    with _creds_lock:
        if _cached_creds and _cached_creds.valid:
            return _cached_creds

        creds = _cached_creds

        if creds is None:
            token_json = os.getenv("GOOGLE_TOKEN_JSON")
            if token_json:
                info = json.loads(token_json)
                creds = Credentials.from_authorized_user_info(info, config.GOOGLE_SCOPES)
            elif os.path.exists(config.GOOGLE_TOKEN_FILE):
                creds = Credentials.from_authorized_user_file(
                    config.GOOGLE_TOKEN_FILE, config.GOOGLE_SCOPES
                )

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _save_token(creds)

        if not creds or not creds.valid:
            raise RuntimeError(
                "No valid Google credentials found. Run auth_setup.py locally first."
            )

        _cached_creds = creds
        return creds


def warmup():
    """Pre-initialize credentials on app startup to avoid cold-start latency."""
    try:
        get_credentials()
        print("Google credentials pre-warmed successfully")
    except Exception as e:
        print(f"Credentials warmup failed (will retry on first request): {e}")


def _save_token(creds):
    """Save token to file (local dev only)."""
    if not os.getenv("GOOGLE_TOKEN_JSON"):
        with open(config.GOOGLE_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
