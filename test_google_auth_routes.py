import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class DummyCache(dict):
    def __init__(self, maxsize, ttl):
        super().__init__()


_SENTINEL = object()
_MODULES_TO_ISOLATE = [
    "config",
    "briefing",
    "gmail_service",
    "calendar_service",
    "tasks_service",
    "gemini_service",
    "chat_service",
    "conversation_store",
    "agent",
    "granola_service",
    "google",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google.oauth2",
    "google.oauth2.credentials",
    "google_auth_oauthlib",
    "google_auth_oauthlib.flow",
    "google.cloud",
    "google.cloud.firestore",
    "googleapiclient",
    "googleapiclient.discovery",
    "cachetools",
    "fastapi",
    "fastapi.responses",
    "main",
    "google_auth",
]
_ORIGINAL_MODULES = {name: sys.modules.get(name, _SENTINEL) for name in _MODULES_TO_ISOLATE}
for name in _MODULES_TO_ISOLATE:
    sys.modules.pop(name, None)


sys.modules["google"] = MagicMock()
sys.modules["google.auth"] = MagicMock()
sys.modules["google.auth.transport"] = MagicMock()
sys.modules["google.auth.transport.requests"] = MagicMock(Request=MagicMock())
sys.modules["google.oauth2"] = MagicMock()
sys.modules["google.oauth2.credentials"] = MagicMock(Credentials=MagicMock())
sys.modules["google_auth_oauthlib"] = MagicMock()
sys.modules["google_auth_oauthlib.flow"] = MagicMock(InstalledAppFlow=MagicMock())
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.firestore"] = MagicMock()
sys.modules["googleapiclient"] = MagicMock()
sys.modules["googleapiclient.discovery"] = MagicMock()
sys.modules["cachetools"] = MagicMock(TTLCache=DummyCache)

config_mock = MagicMock()
config_mock.MOMO_API_SECRET = ""
config_mock.CHAT_SPACE_ID = "spaces/test_space"
config_mock.MOMO_SERVICE_URL = "https://momo.example"
config_mock.GOOGLE_SCOPES = ["scope-a"]
_TEST_TOKEN_DIR = tempfile.mkdtemp(prefix="momo-test-google-auth-routes-")
config_mock.GOOGLE_TOKEN_FILE = os.path.join(_TEST_TOKEN_DIR, "token.json")
config_mock.GRANOLA_ENABLED = True
config_mock.KNOWLEDGE_GRAPH_ENABLED = False
sys.modules["config"] = config_mock

sys.modules["briefing"] = MagicMock()
sys.modules["gmail_service"] = MagicMock()
sys.modules["calendar_service"] = MagicMock()
sys.modules["tasks_service"] = MagicMock()
sys.modules["gemini_service"] = MagicMock()
sys.modules["chat_service"] = MagicMock(
    format_for_google_chat=lambda text: text,
    send_chat_message=MagicMock(),
    download_attachment=MagicMock(),
    _SUPPORTED_AUDIO_TYPES=frozenset(["audio/mp3"]),
)
sys.modules["conversation_store"] = MagicMock(
    get_conversation=MagicMock(),
    add_turn=MagicMock(),
    clear_conversation=MagicMock(),
    conversation_scope=MagicMock(),
    get_pending_task_actions=MagicMock(),
    clear_pending_task_actions=MagicMock(),
    store_pending_task_actions=MagicMock(),
    store_pending_task_actions_if_empty=MagicMock(),
)
sys.modules["agent"] = MagicMock()
sys.modules["granola_service"] = MagicMock()

import google_auth
import main


def tearDownModule():
    shutil.rmtree(_TEST_TOKEN_DIR, ignore_errors=True)
    for name in _MODULES_TO_ISOLATE:
        original = _ORIGINAL_MODULES[name]
        if original is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original



def call_app(method, path, query_string="", headers=None, body=b""):
    response = {"status": None, "headers": [], "body": bytearray()}
    request_sent = False

    async def receive():
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response["body"].extend(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query_string.encode(),
        "root_path": "",
        "headers": [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    asyncio.run(main.app(scope, receive, send))

    header_map = {
        key.decode(): value.decode()
        for key, value in response["headers"]
    }
    return response["status"], header_map, bytes(response["body"])


class TestGoogleAuthRoutes(unittest.TestCase):
    def test_google_auth_paths_are_open(self):
        self.assertIn("/google-auth/start", main._OPEN_PATHS)
        self.assertIn("/google-auth/callback", main._OPEN_PATHS)

    @patch("google_auth.refresh_google_credentials", create=True)
    def test_google_token_refresh_returns_ok_on_success(self, mock_refresh):
        mock_refresh.return_value = True

        status_code, headers, body = call_app("POST", "/google-token-refresh")

        self.assertEqual(status_code, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    @patch("main.send_chat_message")
    @patch("google_auth._send_throttled_reauth_alert", create=True)
    @patch("google_auth.is_reauth_required", create=True)
    @patch("google_auth.refresh_google_credentials", create=True)
    def test_google_token_refresh_uses_throttled_reauth_helper_on_reauth_required(
        self,
        mock_refresh,
        mock_is_reauth,
        mock_send_alert,
        mock_send,
    ):
        mock_refresh.return_value = False
        mock_is_reauth.return_value = True

        first_status, _, first_body = call_app("POST", "/google-token-refresh")

        self.assertEqual(first_status, 200)
        self.assertEqual(json.loads(first_body)["status"], "reauth_required")
        self.assertEqual(json.loads(first_body)["message"], "Google credentials need re-authentication")
        self.assertFalse(mock_send_alert.called)
        self.assertFalse(mock_send.called)

    @patch("google_auth.start_web_reauth", new_callable=AsyncMock, create=True)
    def test_google_auth_start_redirects_to_authorization_url(self, mock_start):
        expected_url = "https://accounts.google.com/o/oauth2/auth?client_id=test"
        mock_start.return_value = expected_url

        status_code, headers, _ = call_app("GET", "/google-auth/start", "t=valid-ticket")

        self.assertEqual(status_code, 307)
        self.assertEqual(headers["location"], expected_url)
        mock_start.assert_awaited_once_with("http://testserver/google-auth/callback", "valid-ticket")

    @patch("google_auth.start_web_reauth", new_callable=AsyncMock, create=True)
    def test_google_auth_start_rejects_missing_ticket_without_starting_reauth(self, mock_start):
        status_code, _, _ = call_app("GET", "/google-auth/start")

        self.assertIn(status_code, {400, 403})
        mock_start.assert_not_called()

    def test_google_auth_callback_rejects_missing_code_or_state(self):
        missing_both_status, _, _ = call_app("GET", "/google-auth/callback")
        missing_code_status, _, _ = call_app("GET", "/google-auth/callback", "state=test-state")
        missing_state_status, _, _ = call_app("GET", "/google-auth/callback", "code=test-code")

        self.assertEqual(missing_both_status, 400)
        self.assertEqual(missing_code_status, 400)
        self.assertEqual(missing_state_status, 400)

    @patch("main.send_chat_message")
    @patch("google_auth.complete_web_reauth", new_callable=AsyncMock, create=True)
    def test_google_auth_callback_confirms_completion_without_secrets(self, mock_complete, mock_send):
        mock_complete.return_value = True

        status_code, _, _ = call_app("GET", "/google-auth/callback", "code=test-code&state=test-state")

        self.assertEqual(status_code, 200)
        mock_complete.assert_awaited_once_with("test-code", "test-state")
        mock_send.assert_called_once()
        sent_message = mock_send.call_args.args[1]
        self.assertIn("reconnected", sent_message.lower())
        self.assertNotIn("test-code", sent_message)
        self.assertNotIn("test-state", sent_message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
