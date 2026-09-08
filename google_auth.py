import json
import os
import secrets
import threading
import time
from urllib.parse import urlencode

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
try:
    from google_auth_oauthlib.flow import Flow as OAuthFlow
except Exception:  # pragma: no cover - fallback for older library variants / mocks
    from google_auth_oauthlib.flow import InstalledAppFlow as OAuthFlow

import config
from connection_errors import ExternalAuthError
from reauth_service import ReauthLinkError, get_service_url, send_auth_notification

_cached_creds = None
_creds_lock = threading.RLock()
_reauth_required = False
_reauth_alert_lock = threading.Lock()

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


class GoogleReauthError(RuntimeError):
    """A callback failure with a static, safe message, distinct from bad state."""

    _MESSAGES = {
        "insufficient_scope": (
            "Google did not grant all required permissions. Please request a fresh "
            "reconnect link and allow all required permissions."
        ),
        "exchange_failed": (
            "Google could not complete the token exchange. Please request a fresh "
            "reconnect link and try again."
        ),
        "config_unavailable": "Google sign-in configuration is unavailable. Please try again later.",
        "persistence_failed": (
            "Google authorization could not be saved. Please request a fresh "
            "reconnect link and try again."
        ),
    }

    def __init__(self, reason: str):
        self.reason = reason if reason in self._MESSAGES else "exchange_failed"
        super().__init__(self._MESSAGES[self.reason])


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
        print(f"Google auth reauth: config_unavailable ({type(exc).__name__})")
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


def _write_credentials_to_firestore(credentials_json: str) -> bool:
    try:
        db = _get_db()
        db.collection(_FIRESTORE_GOOGLE_AUTH_COLLECTION).document(
            _FIRESTORE_GOOGLE_AUTH_DOC
        ).set({
            "credentials_json": credentials_json,
            "updated_at": time.time(),
        })
        return True
    except Exception as exc:
        print(f"Google auth: persistence_failed ({type(exc).__name__})")
        return False


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
    except Exception:
        # A ticket that wasn't persisted can never pass the single-use gate.
        # Storage errors may contain the ticket document ID; don't log them.
        raise ReauthLinkError(
            "ticket_unavailable", "Could not create a Google reconnect link. Please request a fresh link later."
        ) from None
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


def _persist_credentials(creds, *, require_durable: bool = False):
    credentials_json = creds.to_json()
    if require_durable:
        # google-auth's to_json omits granted_scopes. Retain actual grants
        # separately without expanding the requested scopes used on refresh.
        data = json.loads(credentials_json)
        granted = getattr(creds, "granted_scopes", None)
        data["granted_scopes"] = sorted(_scope_set(
            config.GOOGLE_SCOPES if granted is None else granted
        ))
        credentials_json = json.dumps(data)
        if _write_credentials_to_firestore(credentials_json) is not True:
            raise GoogleReauthError("persistence_failed")
        _write_credentials_to_file(credentials_json)
        return

    # Refresh/legacy callers retain their best-effort persistence behavior.
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
        print(f"Google auth: reauth status cleanup failed ({type(exc).__name__})")


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


def _reauth_url(service_url: str, ticket: str) -> str:
    base_url = get_service_url(service_url)
    if not ticket:
        raise ReauthLinkError("ticket_unavailable", "Please request a fresh Google reconnect link.")
    return f"{base_url}/google-auth/start?{urlencode({'t': ticket})}"


def create_reauth_link(service_url: str = "") -> str:
    """Create a fresh, ten-minute, single-use link without sending an alert.

    Explicit requests never consult or update the automatic alert cooldown.
    Validate configuration before issuing a ticket.
    """
    base_url = get_service_url(service_url)
    return _reauth_url(base_url, _create_reauth_ticket())


def _build_reauth_alert_message(service_url, ticket):
    reauth_url = _reauth_url(service_url, ticket)
    return (
        "🔴 *Google sign-in needs attention*\n\n"
        "momo needs you to reconnect google access.\n\n"
        f"👉 <{reauth_url}|*reconnect google access*>\n\n"
        "this link is single-use and expires in 10 minutes. if it expires or you've already opened it, "
        "ask me for a fresh google reconnect link. after sign-in, i can check access again."
    )


def _should_send_throttled_reauth_alert(service_url=""):
    if not config.CHAT_SPACE_ID:
        return False

    try:
        get_service_url(service_url)
    except ReauthLinkError:
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


def _clear_reauth_alert_cooldown() -> None:
    """Allow a new failure to alert after successful reauth or confirmed recovery."""
    try:
        with _reauth_alert_lock:
            _get_db().collection(_FIRESTORE_REAUTH_COLLECTION).document(
                _FIRESTORE_REAUTH_ALERT_DOC
            ).delete()
    except Exception:
        print("Google auth: could not clear reauth alert cooldown")


def _record_reauth_alert_sent():
    try:
        db = _get_db()
        db.collection(_FIRESTORE_REAUTH_COLLECTION).document(_FIRESTORE_REAUTH_ALERT_DOC).set(
            {"sent_at": time.time()}
        )
    except Exception as exc:
        print(f"Google auth: failed to record reauth alert send: {exc}")


def _send_throttled_reauth_alert(service_url=""):
    with _reauth_alert_lock:
        return _send_reauth_alert_unlocked(service_url)


def _send_reauth_alert_unlocked(service_url=""):
    if not config.CHAT_SPACE_ID:
        return False

    try:
        url = get_service_url(service_url)
        if not _should_send_throttled_reauth_alert(service_url=url):
            return False

        ticket = _create_reauth_ticket()
        message = _build_reauth_alert_message(service_url=url, ticket=ticket)
        if not send_auth_notification(message):
            return False
        _record_reauth_alert_sent()
        return True
    except Exception:
        print("Google auth: could not deliver reauth alert")
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
                    "Google credentials require re-authentication. Ask momo for a fresh Google reconnect link."
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


def _scope_set(scopes) -> set[str]:
    if isinstance(scopes, str):
        return set(scopes.split())
    if scopes is None:
        return set()
    if not isinstance(scopes, (list, tuple, set, frozenset)) or not all(
        isinstance(scope, str) and not any(char.isspace() for char in scope)
        for scope in scopes
    ):
        raise ValueError("Invalid scope metadata")
    return {scope for scope in scopes if scope}


def _validate_reauth_token_scopes(token, required_scopes: set[str]) -> None:
    if not isinstance(token, dict) or not token.get("access_token") or "error" in token:
        raise GoogleReauthError("exchange_failed")
    # RFC 6749: an omitted scope means the requested scopes are unchanged.
    # An explicitly empty/null scope is NOT omission and must be rejected.
    granted = _scope_set(token["scope"]) if "scope" in token else required_scopes
    if not required_scopes.issubset(granted):
        raise GoogleReauthError("insufficient_scope")


def _fetch_reauth_credentials(flow, code: str, required_scopes: set[str]):
    from oauthlib.oauth2.rfc6749.tokens import OAuth2Token

    try:
        flow.fetch_token(code=code)
    except Warning as warning:
        token = getattr(warning, "token", None)
        # Only oauthlib's post-validation scope-change Warning is recoverable.
        # Provider errors/missing tokens are rejected by the parser before it.
        if not (
            type(warning) is Warning
            and isinstance(token, OAuth2Token)
            and warning.args
            and isinstance(warning.args[0], str)
            and warning.args[0].startswith("Scope has changed from ")
            and token.scope_changed
            and "scope" in token
            and getattr(warning, "old_scope", None) == token.old_scopes
            and getattr(warning, "new_scope", None) == token.scopes
            and _scope_set(token.old_scopes) == required_scopes
        ):
            raise
        _validate_reauth_token_scopes(token, required_scopes)
        # The fetch stopped before requests-oauthlib populated its client.
        # Use the public setter, preserving all returned token/scope metadata.
        flow.oauth2session.token = token

    _validate_reauth_token_scopes(flow.oauth2session.token, required_scopes)
    return flow.credentials


async def complete_web_reauth(code: str, state: str) -> bool:
    """Return False only for invalid state; raise safe errors for other failures."""
    global _cached_creds

    if not state:
        return False
    failure_reason = "persistence_failed"
    try:
        db = _get_db()
        doc_ref = db.collection(_FIRESTORE_REAUTH_COLLECTION).document(state)
        doc = doc_ref.get()
        if not doc.exists:
            print("Google auth reauth: invalid or expired state parameter")
            return False

        data = doc.to_dict() or {}
        try:
            expires_at = float(data.get("expires_at", 0))
            redirect_uri = data.get("redirect_uri")
            if not isinstance(redirect_uri, str) or not redirect_uri or not (0 < expires_at < float("inf")):
                return False
        except (AttributeError, TypeError, ValueError):
            return False
        if time.time() > expires_at:
            try:
                doc_ref.delete()
            except Exception as exc:
                print(f"Google auth reauth: expired state cleanup failed ({type(exc).__name__})")
            print("Google auth reauth: state expired")
            return False

        if not code:
            raise GoogleReauthError("exchange_failed")

        failure_reason = "config_unavailable"
        required_scopes = _scope_set(config.GOOGLE_SCOPES)
        if not required_scopes:
            raise GoogleReauthError("config_unavailable")
        if os.path.exists(config.GOOGLE_CLIENT_SECRET_FILE):
            flow = OAuthFlow.from_client_secrets_file(
                config.GOOGLE_CLIENT_SECRET_FILE,
                scopes=config.GOOGLE_SCOPES,
                redirect_uri=redirect_uri,
            )
        else:
            client_config = _load_web_client_config_from_sources()
            if not client_config:
                raise GoogleReauthError("config_unavailable")
            flow = OAuthFlow.from_client_config(
                client_config,
                scopes=config.GOOGLE_SCOPES,
                redirect_uri=redirect_uri,
            )
        failure_reason = "exchange_failed"
        creds = _fetch_reauth_credentials(flow, code, required_scopes)
        failure_reason = "persistence_failed"
        _persist_credentials(creds, require_durable=True)
        doc_ref.delete()
        _clear_reauth_required()
        _cached_creds = creds
        _clear_reauth_alert_cooldown()
        print("Google auth reauth: token acquired and stored successfully")
        return True
    except GoogleReauthError as exc:
        print(f"Google auth reauth: {exc.reason}")
        raise exc from None
    except Exception as exc:
        print(f"Google auth reauth: {failure_reason} ({type(exc).__name__})")
        raise GoogleReauthError(failure_reason) from None


def _http_error_status(exc: Exception) -> int | None:
    """Extract the HTTP status from a googleapiclient HttpError (any version)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "resp", None)
        status = getattr(resp, "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_google_auth_error(exc: Exception) -> bool:
    """True when exc is a googleapiclient HttpError with status 401/403 (or an
    already-classified Google Workspace auth error)."""
    if isinstance(exc, (ExternalAuthError, ReauthRequiredError)):
        return True
    return _http_error_status(exc) in (401, 403)


def classify_google_auth_error(exc: Exception, source: str = "google_workspace") -> ExternalAuthError | None:
    """Shared classifier for Google Workspace API failures.

    When exc is a 401/403 HttpError (or ReauthRequiredError), marks reauth
    required, fires the existing throttled Chat alert, and returns a typed
    ExternalAuthError for the caller to raise (or log-and-continue for
    per-item failures). Returns None for non-auth failures."""
    if not is_google_auth_error(exc):
        return None
    status = _http_error_status(exc)
    if not isinstance(exc, (ExternalAuthError, ReauthRequiredError)):
        try:
            _mark_reauth_required(reason=f"http_{status}", source=source)
            _send_throttled_reauth_alert(service_url=config.MOMO_SERVICE_URL)
        except Exception as mark_exc:
            print(f"Google auth: failed to mark/alert reauth from {source}: {mark_exc}")
    if isinstance(exc, ExternalAuthError):
        return exc
    return ExternalAuthError(
        "google_workspace",
        f"Google Workspace API auth failure ({source}) — {exc}",
        status=status,
        reconnect_hint="ask momo for a fresh Google Workspace reconnect link",
    )


def raise_if_google_auth_error(exc: Exception, source: str = "google_workspace") -> None:
    """Raise a typed ExternalAuthError when exc is an auth failure; no-op otherwise."""
    auth_err = classify_google_auth_error(exc, source=source)
    if auth_err is not None:
        raise auth_err from exc


def warmup():
    """Pre-initialize credentials on app startup to avoid cold-start latency."""
    try:
        get_credentials()
        print("Google credentials pre-warmed successfully")
    except Exception as e:
        print(f"Credentials warmup failed (will retry on first request): {e}")
