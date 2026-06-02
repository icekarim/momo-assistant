import json
import os
import secrets
import threading
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
try:
    from google_auth_oauthlib.flow import Flow as OAuthFlow
except Exception:  # pragma: no cover - fallback for older library variants / mocks
    from google_auth_oauthlib.flow import InstalledAppFlow as OAuthFlow

import config

_cached_creds = None
_creds_lock = threading.RLock()
_reauth_required = False

_FIRESTORE_GOOGLE_AUTH_COLLECTION = "google_auth"
_FIRESTORE_GOOGLE_AUTH_DOC = "token"
_FIRESTORE_REAUTH_COLLECTION = "google_auth_pending"
_FIRESTORE_REAUTH_STATUS_DOC = "reauth_required"
_FIRESTORE_REAUTH_ALERT_DOC = "last_reauth_alert"
_REAUTH_ALERT_COOLDOWN_SECONDS = 12 * 60 * 60
_REAUTH_PENDING_TTL_SECONDS = 10 * 60
_DEFAULT_GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
_DEFAULT_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"


class ReauthRequiredError(RuntimeError):
    pass


def _get_db():
    from conversation_store import get_db

    return get_db()


def _credentials_from_serialized(serialized: str):
    data = json.loads(serialized)
    return Credentials.from_authorized_user_info(data, config.GOOGLE_SCOPES)


def _web_client_config_from_serialized(serialized: str):
    data = json.loads(serialized)
    client_info = data.get("web") or data.get("installed") or data
    if not isinstance(client_info, dict):
        return None

    client_id = client_info.get("client_id")
    client_secret = client_info.get("client_secret")
    auth_uri = client_info.get("auth_uri") or _DEFAULT_GOOGLE_AUTH_URI
    token_uri = client_info.get("token_uri") or _DEFAULT_GOOGLE_TOKEN_URI

    if not client_id or not client_secret:
        return None

    return {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": auth_uri,
            "token_uri": token_uri,
        }
    }


def _load_web_client_config_from_sources():
    token_json = os.getenv("GOOGLE_TOKEN_JSON")
    if token_json:
        try:
            client_config = _web_client_config_from_serialized(token_json)
            if client_config:
                return client_config
        except Exception:
            pass

    try:
        db = _get_db()
        doc = db.collection(_FIRESTORE_GOOGLE_AUTH_COLLECTION).document(
            _FIRESTORE_GOOGLE_AUTH_DOC
        ).get()
        if not doc.exists:
            return None

        data = doc.to_dict() or {}
        serialized = data.get("credentials_json") or data.get("token_json") or data.get("token")
        if not serialized:
            return None
        if isinstance(serialized, dict):
            serialized = json.dumps(serialized)
        return _web_client_config_from_serialized(serialized)
    except Exception as exc:
        print(f"Google auth reauth: failed to read derived web client config: {exc}")
        return None


def _read_credentials_from_firestore():
    try:
        db = _get_db()
        doc = db.collection(_FIRESTORE_GOOGLE_AUTH_COLLECTION).document(
            _FIRESTORE_GOOGLE_AUTH_DOC
        ).get()
        if not doc.exists:
            return None

        data = doc.to_dict() or {}
        serialized = data.get("credentials_json") or data.get("token_json") or data.get("token")
        if not serialized:
            return None
        if isinstance(serialized, dict):
            serialized = json.dumps(serialized)
        return _credentials_from_serialized(serialized)
    except Exception as exc:
        print(f"Google auth: Firestore credential read failed: {exc}")
        return None


def _write_credentials_to_firestore(credentials_json: str):
    try:
        db = _get_db()
        db.collection(_FIRESTORE_GOOGLE_AUTH_COLLECTION).document(
            _FIRESTORE_GOOGLE_AUTH_DOC
        ).set({
            "credentials_json": credentials_json,
            "updated_at": time.time(),
        })
    except Exception as exc:
        print(f"Google auth: Firestore credential write failed: {exc}")


def _create_reauth_ticket() -> str:
    ticket = secrets.token_urlsafe(32)
    try:
        db = _get_db()
        db.collection(_FIRESTORE_REAUTH_COLLECTION).document(f"ticket:{ticket}").set(
            {
                "used": False,
                "created_at": time.time(),
                "expires_at": time.time() + _REAUTH_PENDING_TTL_SECONDS,
            }
        )
    except Exception as exc:
        print(f"Google auth: failed to store reauth ticket: {exc}")
    return ticket


def _consume_reauth_ticket(ticket: str) -> bool:
    if not ticket:
        return False

    try:
        db = _get_db()
        doc_ref = db.collection(_FIRESTORE_REAUTH_COLLECTION).document(f"ticket:{ticket}")
        doc = doc_ref.get()
        if not doc.exists:
            return False

        data = doc.to_dict() or {}
        if data.get("used"):
            return False
        if time.time() > float(data.get("expires_at", 0)):
            return False

        doc_ref.set({"used": True})
        return True
    except Exception as exc:
        print(f"Google auth: failed to consume reauth ticket: {exc}")
        return False


def _write_credentials_to_file(credentials_json: str):
    try:
        if not isinstance(credentials_json, str):
            credentials_json = str(credentials_json)
        with open(config.GOOGLE_TOKEN_FILE, "w") as f:
            f.write(credentials_json)
    except OSError:
        pass


def _persist_credentials(creds):
    credentials_json = creds.to_json()
    _write_credentials_to_file(credentials_json)
    _write_credentials_to_firestore(credentials_json)


def _mark_reauth_required(reason: str, source: str):
    global _reauth_required
    _reauth_required = True

    try:
        db = _get_db()
        db.collection(_FIRESTORE_REAUTH_COLLECTION).document(_FIRESTORE_REAUTH_STATUS_DOC).set(
            {
                "reauth_required": True,
                "reason": reason,
                "source": source,
                "updated_at": time.time(),
            }
        )
    except Exception as exc:
        print(f"Google auth: failed to mark reauth required: {exc}")


def _clear_reauth_required():
    global _reauth_required
    _reauth_required = False

    try:
        db = _get_db()
        db.collection(_FIRESTORE_REAUTH_COLLECTION).document(_FIRESTORE_REAUTH_STATUS_DOC).set(
            {
                "reauth_required": False,
                "updated_at": time.time(),
            }
        )
    except Exception as exc:
        print(f"Google auth: failed to clear reauth required: {exc}")


def is_reauth_required() -> bool:
    global _reauth_required

    if _reauth_required:
        return True

    try:
        db = _get_db()
        doc = db.collection(_FIRESTORE_REAUTH_COLLECTION).document(_FIRESTORE_REAUTH_STATUS_DOC).get()
        if not doc.exists:
            return False
        data = doc.to_dict() or {}
        _reauth_required = bool(data.get("reauth_required"))
        return _reauth_required
    except Exception as exc:
        print(f"Google auth: failed to read reauth status: {exc}")
        return _reauth_required


def _build_reauth_alert_message(service_url, ticket):
    base_url = (service_url or config.MOMO_SERVICE_URL or "").rstrip("/")
    reauth_url = (
        f"{base_url}/google-auth/start?t={ticket}" if base_url else f"/google-auth/start?t={ticket}"
    )
    return (
        "🔴 *Google sign-in needs attention*\n\n"
        "momo needs you to reconnect google access.\n\n"
        f"👉 <{reauth_url}|*reconnect google access*>\n\n"
        "once you finish, i'll resume syncing workspace data automatically."
    )


def _should_send_throttled_reauth_alert(service_url=""):
    if not config.CHAT_SPACE_ID:
        return False

    url = service_url or config.MOMO_SERVICE_URL
    if not url:
        print("Google auth: no MOMO_SERVICE_URL configured, can't send reauth link")
        return False

    try:
        db = _get_db()
        alert_ref = db.collection(_FIRESTORE_REAUTH_COLLECTION).document(
            _FIRESTORE_REAUTH_ALERT_DOC
        )
        alert_doc = alert_ref.get()
        if alert_doc.exists:
            last_sent = alert_doc.to_dict().get("sent_at", 0)
            if time.time() - last_sent < _REAUTH_ALERT_COOLDOWN_SECONDS:
                return False
        return True
    except Exception as exc:
        print(f"Google auth: failed to check reauth alert cooldown: {exc}")
        return False


def _record_reauth_alert_sent():
    try:
        db = _get_db()
        db.collection(_FIRESTORE_REAUTH_COLLECTION).document(_FIRESTORE_REAUTH_ALERT_DOC).set(
            {"sent_at": time.time()}
        )
    except Exception as exc:
        print(f"Google auth: failed to record reauth alert send: {exc}")


def _send_throttled_reauth_alert(service_url=""):
    if not config.CHAT_SPACE_ID:
        return False

    url = service_url or config.MOMO_SERVICE_URL
    if not url:
        print("Google auth: no MOMO_SERVICE_URL configured, can't send reauth link")
        return False

    try:
        if not _should_send_throttled_reauth_alert(service_url=url):
            return False

        from chat_service import send_chat_message

        ticket = _create_reauth_ticket()
        message = _build_reauth_alert_message(service_url=url, ticket=ticket)
        send_chat_message(config.CHAT_SPACE_ID, message)
        _record_reauth_alert_sent()
        return True
    except Exception as exc:
        print(f"Google auth: failed to send reauth alert: {exc}")
        return False


def _load_credentials_from_sources():
    global _cached_creds

    if _cached_creds and _cached_creds.valid and not _cached_creds.expired:
        return _cached_creds

    creds = _cached_creds

    if creds is None:
        creds = _read_credentials_from_firestore()

    if creds is None:
        token_json = os.getenv("GOOGLE_TOKEN_JSON")
        if token_json:
            try:
                info = json.loads(token_json)
                creds = Credentials.from_authorized_user_info(info, config.GOOGLE_SCOPES)
                _write_credentials_to_firestore(creds.to_json())
            except Exception:
                creds = None

    if creds is None and os.path.exists(config.GOOGLE_TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(
                config.GOOGLE_TOKEN_FILE, config.GOOGLE_SCOPES
            )
        except Exception:
            creds = None

    if creds:
        _cached_creds = creds

    return creds


def _refresh_loaded_credentials(creds):
    global _cached_creds

    if not creds:
        return False, None

    if creds.valid and not creds.expired:
        _cached_creds = creds
        return True, creds

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _persist_credentials(creds)
            _clear_reauth_required()
            _cached_creds = creds
            return True, creds
        except Exception as exc:
            if "invalid_grant" in str(exc).lower():
                _mark_reauth_required(
                    reason="invalid_grant",
                    source="google_credentials_refresh",
                )
                _send_throttled_reauth_alert(service_url=config.MOMO_SERVICE_URL)
                _cached_creds = None
                raise ReauthRequiredError(
                    "Google credentials require re-authentication. Reconnect via /google-auth/start."
                )
            raise

    return False, creds


def refresh_google_credentials() -> bool:
    """Refresh the cached Google OAuth credentials if possible."""
    with _creds_lock:
        creds = _load_credentials_from_sources()
        try:
            refreshed, _ = _refresh_loaded_credentials(creds)
        except ReauthRequiredError:
            return False
        return refreshed


def get_credentials():
    """Get valid Google OAuth credentials, refreshing if needed.
    Caches credentials in memory to avoid re-parsing on every call."""
    global _cached_creds

    with _creds_lock:
        creds = _load_credentials_from_sources()
        if creds is None:
            raise RuntimeError(
                "No valid Google credentials found. Run auth_setup.py locally first."
            )

        refreshed, refreshed_creds = _refresh_loaded_credentials(creds)
        if refreshed:
            return refreshed_creds

        if refreshed_creds and refreshed_creds.valid and not refreshed_creds.expired:
            _cached_creds = refreshed_creds
            return refreshed_creds

        raise RuntimeError(
            "No valid Google credentials found. Run auth_setup.py locally first."
        )


async def start_web_reauth(redirect_uri: str, ticket: str) -> str | None:
    """Start a browser-based Google OAuth reauth flow."""
    if not ticket or not _consume_reauth_ticket(ticket):
        return None

    state = secrets.token_urlsafe(32)

    try:
        if os.path.exists(config.GOOGLE_CLIENT_SECRET_FILE):
            flow = OAuthFlow.from_client_secrets_file(
                config.GOOGLE_CLIENT_SECRET_FILE,
                scopes=config.GOOGLE_SCOPES,
                redirect_uri=redirect_uri,
            )
        else:
            client_config = _load_web_client_config_from_sources()
            if not client_config:
                print(f"Google auth reauth: missing {config.GOOGLE_CLIENT_SECRET_FILE}")
                return None
            flow = OAuthFlow.from_client_config(
                client_config,
                scopes=config.GOOGLE_SCOPES,
                redirect_uri=redirect_uri,
            )
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )
    except Exception as exc:
        print(f"Google auth reauth: failed to build authorization URL: {exc}")
        return None

    try:
        db = _get_db()
        db.collection(_FIRESTORE_REAUTH_COLLECTION).document(state).set(
            {
                "redirect_uri": redirect_uri,
                "created_at": time.time(),
                "expires_at": time.time() + _REAUTH_PENDING_TTL_SECONDS,
            }
        )
    except Exception as exc:
        print(f"Google auth reauth: failed to persist pending state: {exc}")
        return None

    return auth_url


async def complete_web_reauth(code: str, state: str) -> bool:
    """Complete the browser OAuth reauth flow and persist credentials."""
    global _cached_creds

    try:
        db = _get_db()
        doc_ref = db.collection(_FIRESTORE_REAUTH_COLLECTION).document(state)
        doc = doc_ref.get()
        if not doc.exists:
            print("Google auth reauth: invalid or expired state parameter")
            return False

        data = doc.to_dict() or {}
        if time.time() > float(data.get("expires_at", 0)):
            doc_ref.delete()
            print("Google auth reauth: state expired")
            return False

        redirect_uri = data.get("redirect_uri", "")
        if os.path.exists(config.GOOGLE_CLIENT_SECRET_FILE):
            flow = OAuthFlow.from_client_secrets_file(
                config.GOOGLE_CLIENT_SECRET_FILE,
                scopes=config.GOOGLE_SCOPES,
                redirect_uri=redirect_uri,
            )
        else:
            client_config = _load_web_client_config_from_sources()
            if not client_config:
                print(f"Google auth reauth: missing {config.GOOGLE_CLIENT_SECRET_FILE}")
                return False
            flow = OAuthFlow.from_client_config(
                client_config,
                scopes=config.GOOGLE_SCOPES,
                redirect_uri=redirect_uri,
            )
        flow.fetch_token(code=code)
        creds = flow.credentials
        _persist_credentials(creds)
        _clear_reauth_required()
        _cached_creds = creds
        doc_ref.delete()
        print("Google auth reauth: token acquired and stored successfully")
        return True
    except Exception as exc:
        print(f"Google auth reauth: token exchange failed: {exc}")
        return False


def warmup():
    """Pre-initialize credentials on app startup to avoid cold-start latency."""
    try:
        get_credentials()
        print("Google credentials pre-warmed successfully")
    except Exception as e:
        print(f"Credentials warmup failed (will retry on first request): {e}")
