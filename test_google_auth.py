import asyncio
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class _FakeRefreshError(Exception):
    pass


_SENTINEL = object()
_MODULES_TO_ISOLATE = [
    "config",
    "google",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google.cloud",
    "google.cloud.firestore",
    "google.oauth2",
    "google.oauth2.credentials",
    "google_auth_oauthlib",
    "google_auth_oauthlib.flow",
    "chat_service",
    "google_auth",
]
_ORIGINAL_MODULES = {name: sys.modules.get(name, _SENTINEL) for name in _MODULES_TO_ISOLATE}
for name in _MODULES_TO_ISOLATE:
    sys.modules.pop(name, None)

config_mock = MagicMock()
config_mock.GOOGLE_SCOPES = ["scope-a", "scope-b"]
_TEST_TOKEN_DIR = tempfile.mkdtemp(prefix="momo-test-google-auth-")
config_mock.GOOGLE_TOKEN_FILE = os.path.join(_TEST_TOKEN_DIR, "token.json")
config_mock.CHAT_SPACE_ID = "spaces/test"
config_mock.MOMO_SERVICE_URL = "https://momo.example"
config_mock.FIRESTORE_DATABASE = "testing"
sys.modules["config"] = config_mock

google_module = MagicMock()
google_auth_module = MagicMock()
google_auth_transport_module = MagicMock()
google_auth_transport_requests_module = MagicMock()
google_cloud_module = MagicMock()
google_cloud_firestore_module = MagicMock()
google_oauth2_module = MagicMock()
google_oauth2_credentials_module = MagicMock()
google_auth_oauthlib_module = MagicMock()
google_auth_oauthlib_flow_module = MagicMock()
chat_service_module = MagicMock()

google_module.auth = google_auth_module
google_module.cloud = google_cloud_module
google_module.oauth2 = google_oauth2_module
google_auth_module.transport = google_auth_transport_module
google_auth_transport_module.requests = google_auth_transport_requests_module
google_cloud_module.firestore = google_cloud_firestore_module
google_oauth2_module.credentials = google_oauth2_credentials_module
google_auth_oauthlib_module.flow = google_auth_oauthlib_flow_module
chat_service_module.send_chat_message = MagicMock(name="send_chat_message")

google_auth_transport_requests_module.Request = MagicMock(name="Request")
google_oauth2_credentials_module.Credentials = MagicMock(name="Credentials")
google_auth_oauthlib_flow_module.InstalledAppFlow = MagicMock(name="InstalledAppFlow")
google_cloud_firestore_module.Client = MagicMock(name="Client")

sys.modules["google"] = google_module
sys.modules["google.auth"] = google_auth_module
sys.modules["google.auth.transport"] = google_auth_transport_module
sys.modules["google.auth.transport.requests"] = google_auth_transport_requests_module
sys.modules["google.cloud"] = google_cloud_module
sys.modules["google.cloud.firestore"] = google_cloud_firestore_module
sys.modules["google.oauth2"] = google_oauth2_module
sys.modules["google.oauth2.credentials"] = google_oauth2_credentials_module
sys.modules["google_auth_oauthlib"] = google_auth_oauthlib_module
sys.modules["google_auth_oauthlib.flow"] = google_auth_oauthlib_flow_module
sys.modules["chat_service"] = chat_service_module

import google_auth

sys.modules.pop("google_auth", None)
sys.modules["google_auth"] = google_auth


def tearDownModule():
    google_auth._cached_creds = None
    shutil.rmtree(_TEST_TOKEN_DIR, ignore_errors=True)
    for name in _MODULES_TO_ISOLATE:
        original = _ORIGINAL_MODULES[name]
        if original is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _make_creds(*, valid=True, expired=False, refresh_token="refresh-token", json_text="{}"): 
    creds = MagicMock(name="creds")
    creds.valid = valid
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.to_json.return_value = json_text
    return creds


class TestGoogleAuthSelfHealing(unittest.TestCase):
    def setUp(self):
        google_auth._cached_creds = None
        google_auth.Credentials.from_authorized_user_info.reset_mock()
        google_auth.Credentials.from_authorized_user_file.reset_mock()
        google_auth.Request.reset_mock()

    def test_write_credentials_to_file_never_touches_repo_token_json(self):
        repo_token = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "token.json"
        )
        self.assertNotEqual(
            os.path.abspath(google_auth.config.GOOGLE_TOKEN_FILE),
            repo_token,
            "tests must not point GOOGLE_TOKEN_FILE at the real repo token.json",
        )

        before = None
        if os.path.exists(repo_token):
            with open(repo_token, "rb") as handle:
                before = handle.read()

        google_auth._write_credentials_to_file("<MagicMock id='0xdeadbeef'>")

        if before is None:
            self.assertFalse(
                os.path.exists(repo_token),
                "tests must not create the real repo token.json",
            )
        else:
            with open(repo_token, "rb") as handle:
                after = handle.read()
            self.assertEqual(
                before, after, "tests must not modify the real repo token.json"
            )

    def test_firestore_token_is_preferred_over_stale_google_token_json(self):
        firestore_creds = _make_creds(json_text='{"access_token": "fs-token"}')
        env_creds = _make_creds(json_text='{"access_token": "env-token"}')

        with patch.dict(
            "os.environ",
            {"GOOGLE_TOKEN_JSON": '{"access_token": "env-token"}'},
            clear=True,
        ), patch.object(
            google_auth,
            "_read_credentials_from_firestore",
            create=True,
            return_value=firestore_creds,
        ) as mock_read_firestore:
            google_auth.Credentials.from_authorized_user_info.return_value = env_creds

            creds = google_auth.get_credentials()

        self.assertIs(
            creds,
            firestore_creds,
            "get_credentials() should prefer Firestore credentials over GOOGLE_TOKEN_JSON",
        )
        mock_read_firestore.assert_called_once_with()
        google_auth.Credentials.from_authorized_user_info.assert_not_called()

    def test_google_token_json_seeds_firestore_when_firestore_has_no_token(self):
        env_creds = _make_creds(json_text='{"access_token": "env-token"}')

        with patch.dict(
            "os.environ",
            {"GOOGLE_TOKEN_JSON": '{"access_token": "env-token"}'},
            clear=True,
        ), patch.object(
            google_auth,
            "_read_credentials_from_firestore",
            create=True,
            return_value=None,
        ) as mock_read_firestore, patch.object(
            google_auth,
            "_write_credentials_to_firestore",
            create=True,
        ) as mock_write_firestore:
            google_auth.Credentials.from_authorized_user_info.return_value = env_creds

            creds = google_auth.get_credentials()

        self.assertIs(
            creds,
            env_creds,
            "get_credentials() should still return the env-seeded credential object",
        )
        mock_read_firestore.assert_called_once_with()
        mock_write_firestore.assert_called_once_with(env_creds.to_json())

    def test_refreshing_expired_credentials_persists_serialized_json_to_firestore(self):
        expired_creds = _make_creds(valid=False, expired=True, refresh_token="refresh-token")

        def _refresh(_request):
            expired_creds.valid = True

        expired_creds.refresh.side_effect = _refresh

        with patch.dict(
            "os.environ",
            {"GOOGLE_TOKEN_JSON": '{"access_token": "env-token"}'},
            clear=True,
        ), patch.object(
            google_auth,
            "_read_credentials_from_firestore",
            create=True,
            return_value=expired_creds,
        ), patch.object(
            google_auth,
            "_write_credentials_to_firestore",
            create=True,
        ) as mock_write_firestore, patch("builtins.open", MagicMock()):
            google_auth.Credentials.from_authorized_user_info.return_value = expired_creds

            creds = google_auth.get_credentials()

        self.assertIs(creds, expired_creds)
        mock_write_firestore.assert_called_once_with(expired_creds.to_json())

    def test_invalid_grant_marks_reauth_required_and_triggers_throttled_alert(self):
        expired_creds = _make_creds(valid=False, expired=True, refresh_token="refresh-token")

        def _raise_invalid_grant(_request):
            raise _FakeRefreshError("invalid_grant: Token has been revoked")

        expired_creds.refresh.side_effect = _raise_invalid_grant

        with patch.dict(
            "os.environ",
            {"GOOGLE_TOKEN_JSON": '{"access_token": "env-token"}'},
            clear=True,
        ), patch.object(
            google_auth,
            "_read_credentials_from_firestore",
            create=True,
            return_value=expired_creds,
        ), patch.object(
            google_auth,
            "_mark_reauth_required",
            create=True,
        ) as mock_mark_reauth, patch.object(
            google_auth,
            "_send_throttled_reauth_alert",
            create=True,
        ) as mock_send_alert:
            google_auth.Credentials.from_authorized_user_info.return_value = expired_creds

            with self.assertRaisesRegex(RuntimeError, "re-auth"):
                google_auth.get_credentials()

        mock_mark_reauth.assert_called_once_with(
            reason="invalid_grant",
            source="google_credentials_refresh",
        )
        mock_send_alert.assert_called_once_with(
            service_url="https://momo.example",
        )

    def test_refresh_google_credentials_returns_false_and_marks_reauth_required_on_invalid_grant(self):
        expired_creds = _make_creds(valid=False, expired=True, refresh_token="refresh-token")

        def _raise_invalid_grant(_request):
            raise _FakeRefreshError("invalid_grant: Token has been revoked")

        expired_creds.refresh.side_effect = _raise_invalid_grant

        with patch.dict(
            "os.environ",
            {"GOOGLE_TOKEN_JSON": '{"access_token": "env-token"}'},
            clear=True,
        ), patch.object(
            google_auth,
            "_read_credentials_from_firestore",
            create=True,
            return_value=expired_creds,
        ), patch.object(
            google_auth,
            "_mark_reauth_required",
            create=True,
        ) as mock_mark_reauth, patch.object(
            google_auth,
            "_send_throttled_reauth_alert",
            create=True,
        ) as mock_send_alert:
            google_auth.Credentials.from_authorized_user_info.return_value = expired_creds

            refreshed = google_auth.refresh_google_credentials()

        self.assertFalse(refreshed)
        self.assertTrue(google_auth.is_reauth_required())
        mock_mark_reauth.assert_called_once_with(
            reason="invalid_grant",
            source="google_credentials_refresh",
        )
        mock_send_alert.assert_called_once_with(
            service_url="https://momo.example",
        )

    def test_reauth_alert_links_to_start_route_and_redacts_secret_material(self):
        alert_builder = getattr(google_auth, "_build_reauth_alert_message", None)
        self.assertIsNotNone(
            alert_builder,
            "google_auth should expose _build_reauth_alert_message(service_url, ticket) for redacted reauth alerts",
        )

        alert_message = alert_builder(service_url="https://momo.example", ticket="ticket-123")

        self.assertIn("/google-auth/start?t=ticket-123", alert_message)
        self.assertIn("?t=", alert_message)
        self.assertNotIn("refresh_token", alert_message)
        self.assertNotIn("client_secret", alert_message)
        self.assertNotIn("client_id", alert_message)
        self.assertNotIn("token", alert_message)

    def test_send_throttled_reauth_alert_creates_ticket_and_includes_it_in_chat_message(self):
        with patch.object(google_auth, "_should_send_throttled_reauth_alert", return_value=True), patch.object(
            google_auth,
            "_create_reauth_ticket",
            create=True,
            return_value="ticket-abc123",
        ) as mock_create_ticket, patch.object(
            google_auth,
            "_build_reauth_alert_message",
            side_effect=lambda service_url, ticket: f"{service_url}/google-auth/start?t={ticket}",
        ) as mock_build_message, patch.object(
            google_auth,
            "_record_reauth_alert_sent",
            create=True,
        ), patch("chat_service.send_chat_message") as mock_send_chat:
            result = google_auth._send_throttled_reauth_alert(service_url="https://momo.example")

        self.assertTrue(result)
        mock_create_ticket.assert_called_once_with()
        mock_build_message.assert_called_once_with(
            service_url="https://momo.example",
            ticket="ticket-abc123",
        )
        mock_send_chat.assert_called_once_with(
            config_mock.CHAT_SPACE_ID,
            "https://momo.example/google-auth/start?t=ticket-abc123",
        )

    def test_start_web_reauth_rejects_invalid_ticket_without_building_flow_or_pending_state(self):
        pending_state_db = MagicMock(name="pending_state_db")
        flow = MagicMock(name="oauth_flow")

        with patch.dict(
            "os.environ",
            {
                "GOOGLE_TOKEN_JSON": (
                    '{"client_id":"fake-client-id","client_secret":"fake-client-secret",'
                    '"token_uri":"https://oauth2.googleapis.com/token"}'
                )
            },
            clear=True,
        ), patch.object(google_auth.os.path, "exists", return_value=False), patch.object(
            google_auth,
            "_consume_reauth_ticket",
            create=True,
            return_value=False,
        ) as mock_consume_ticket, patch.object(
            google_auth,
            "_get_db",
            return_value=pending_state_db,
        ), patch.object(
            google_auth.OAuthFlow,
            "from_client_config",
            return_value=flow,
            create=True,
        ) as mock_from_client_config:
            auth_url = asyncio.run(google_auth.start_web_reauth("http://localhost/callback", "bad-ticket"))

        self.assertIsNone(auth_url)
        mock_consume_ticket.assert_called_once_with("bad-ticket")
        mock_from_client_config.assert_not_called()
        pending_state_db.collection.assert_not_called()

    def test_start_web_reauth_builds_oauth_flow_without_local_client_secret_file(self):
        pending_state_db = MagicMock(name="pending_state_db")
        pending_state_collection = MagicMock(name="pending_state_collection")
        pending_state_doc = MagicMock(name="pending_state_doc")
        pending_state_collection.document.return_value = pending_state_doc
        pending_state_db.collection.return_value = pending_state_collection

        flow = MagicMock(name="oauth_flow")
        flow.authorization_url.return_value = (
            "https://accounts.google.com/o/oauth2/auth?client_id=fake-client-id",
            None,
        )

        client_config = {
            "web": {
                "client_id": "fake-client-id",
                "client_secret": "fake-client-secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }

        with patch.dict(
            "os.environ",
            {
                "GOOGLE_TOKEN_JSON": (
                    '{"client_id":"fake-client-id","client_secret":"fake-client-secret",'
                    '"token_uri":"https://oauth2.googleapis.com/token"}'
                )
            },
            clear=True,
        ), patch.object(google_auth.os.path, "exists", return_value=False), patch.object(
            google_auth,
            "_consume_reauth_ticket",
            create=True,
            return_value=True,
        ) as mock_consume_ticket, patch.object(
            google_auth,
            "_get_db",
            return_value=pending_state_db,
        ), patch.object(
            google_auth.OAuthFlow,
            "from_client_config",
            return_value=flow,
            create=True,
        ) as mock_from_client_config:
            auth_url = asyncio.run(google_auth.start_web_reauth("http://localhost/callback", "valid-ticket"))

        self.assertEqual(auth_url, "https://accounts.google.com/o/oauth2/auth?client_id=fake-client-id")
        mock_consume_ticket.assert_called_once_with("valid-ticket")
        mock_from_client_config.assert_called_once_with(
            client_config,
            scopes=config_mock.GOOGLE_SCOPES,
            redirect_uri="http://localhost/callback",
        )
        pending_state_collection.document.assert_called_once()
        pending_state_doc.set.assert_called_once()


if __name__ == "__main__":
    unittest.main()
