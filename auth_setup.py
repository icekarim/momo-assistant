"""
Run this ONCE locally to authenticate with Google and generate token.json.

Usage:
    python auth_setup.py
"""

import json
import os
from google_auth_oauthlib.flow import InstalledAppFlow
import config


def main():
    if not os.path.exists(config.GOOGLE_CLIENT_SECRET_FILE):
        print(f"Missing {config.GOOGLE_CLIENT_SECRET_FILE}. Download from Google Cloud Console.")
        return

    flow = InstalledAppFlow.from_client_secrets_file(
        config.GOOGLE_CLIENT_SECRET_FILE,
        config.GOOGLE_SCOPES,
    )

    creds = flow.run_local_server(port=8080, prompt="consent")

    with open(config.GOOGLE_TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"\nCredentials saved to {config.GOOGLE_TOKEN_FILE}")
    print(f"Scopes: {config.GOOGLE_SCOPES}")


if __name__ == "__main__":
    main()
