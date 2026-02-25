import os
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import config


def get_credentials():
    """Get valid Google OAuth credentials, refreshing if needed."""
    creds = None

    # Check for token in environment variable (for Cloud Run)
    token_json = os.getenv("GOOGLE_TOKEN_JSON")
    if token_json:
        info = json.loads(token_json)
        creds = Credentials.from_authorized_user_info(info, config.GOOGLE_SCOPES)

    # Check for token file (local development)
    elif os.path.exists(config.GOOGLE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            config.GOOGLE_TOKEN_FILE, config.GOOGLE_SCOPES
        )

    # Refresh if expired
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)

    if not creds or not creds.valid:
        raise RuntimeError(
            "No valid Google credentials found. Run auth_setup.py locally first."
        )

    return creds


def _save_token(creds):
    """Save token to file (local dev only)."""
    if not os.getenv("GOOGLE_TOKEN_JSON"):
        with open(config.GOOGLE_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
