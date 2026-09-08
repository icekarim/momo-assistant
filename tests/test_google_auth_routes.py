import asyncio
import html
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch
from urllib.parse import urlencode

from test_reauth_tools import _load_fresh_module, isolated_module_registry


class DummyCache(dict):
    def __init__(self, maxsize, ttl):
        super().__init__()


_TEST_MODULES = {
    "google": MagicMock(),
    "google.auth": MagicMock(),
    "google.auth.transport": MagicMock(),
    "google.auth.transport.requests": MagicMock(Request=MagicMock()),
    "google.oauth2": MagicMock(),
    "google.oauth2.credentials": MagicMock(Credentials=MagicMock()),
    "google_auth_oauthlib": MagicMock(),
    "google_auth_oauthlib.flow": MagicMock(InstalledAppFlow=MagicMock()),
    "google.cloud": MagicMock(),
    "google.cloud.firestore": MagicMock(),
    "googleapiclient": MagicMock(),
    "googleapiclient.discovery": MagicMock(),
    "cachetools": MagicMock(TTLCache=DummyCache),
}

config_mock = MagicMock()
config_mock.MOMO_API_SECRET = ""
config_mock.CHAT_SPACE_ID = "spaces/test_space"
config_mock.MOMO_SERVICE_URL = "https://momo.example"
config_mock.GOOGLE_SCOPES = ["scope-a"]
_TEST_TOKEN_DIR = tempfile.mkdtemp(prefix="momo-test-google-auth-routes-")
config_mock.GOOGLE_TOKEN_FILE = os.path.join(_TEST_TOKEN_DIR, "token.json")
config_mock.GRANOLA_ENABLED = True
config_mock.KNOWLEDGE_GRAPH_ENABLED = False
config_mock.LANGFUSE_TRACING_ENABLED = False
_TEST_MODULES["config"] = config_mock

_TEST_MODULES["briefing"] = MagicMock()
_TEST_MODULES["gmail_service"] = MagicMock()
_TEST_MODULES["calendar_service"] = MagicMock()
_TEST_MODULES["tasks_service"] = MagicMock()
_TEST_MODULES["gemini_service"] = MagicMock()
_TEST_MODULES["chat_service"] = MagicMock(
    format_for_google_chat=lambda text: text,
    send_chat_message=MagicMock(),
    download_attachment=MagicMock(),
    _SUPPORTED_AUDIO_TYPES=frozenset(["audio/mp3"]),
)
_TEST_MODULES["conversation_store"] = MagicMock(
    get_conversation=MagicMock(),
    add_turn=MagicMock(),
    clear_conversation=MagicMock(),
    conversation_scope=MagicMock(),
    get_pending_task_actions=MagicMock(),
    clear_pending_task_actions=MagicMock(),
    store_pending_task_actions=MagicMock(),
    store_pending_task_actions_if_empty=MagicMock(),
)
_TEST_MODULES["agent"] = MagicMock()
_TEST_MODULES["granola_service"] = MagicMock()


def setUpModule():
    """No production imports or sys.modules mutations during collection."""
    global _module_patch, google_auth, main
    _module_patch = isolated_module_registry(
        *(name.split(".")[0] for name in _TEST_MODULES),
        "fastapi", "connection_errors", "reauth_service", "observability", "cards", "google_auth", "main",
    )
    _module_patch.__enter__()
    sys.modules.update(_TEST_MODULES)
    try:
        # These must be real and consistently bound, not cached sibling stubs.
        for name in list(sys.modules):
            if name == "fastapi" or name.startswith("fastapi."):
                sys.modules.pop(name)
        for name in ("connection_errors", "reauth_service", "observability", "cards"):
            _load_fresh_module(name)
        google_auth = _load_fresh_module("google_auth")
        main = _load_fresh_module("main")
    except BaseException:
        _module_patch.__exit__(*sys.exc_info())
        raise


def tearDownModule():
    try:
        shutil.rmtree(_TEST_TOKEN_DIR, ignore_errors=True)
    finally:
        _module_patch.__exit__(None, None, None)



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


class OAuthRouteRegressions:
    """Exercise the shared route contract for both providers, without network I/O."""

    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(config_mock, "MOMO_SERVICE_URL", "https://momo.example"))
        self.enterContext(patch.object(config_mock, "CHAT_SPACE_ID", "spaces/test_space"))
        self.mock_send = self.enterContext(patch.object(main, "send_chat_message", return_value=True))
        self.mock_scope = self.enterContext(
            patch.object(main, "conversation_scope", return_value="space:spaces/test_space")
        )
        self.mock_add = self.enterContext(patch.object(main, "add_turn"))

    def start_request(self, headers=None):
        query = "t=valid-ticket" if self.provider == "google" else ""
        return call_app("GET", f"/{self.provider}-auth/start", query, headers=headers)

    def expected_start_args(self, origin):
        args = (f"{origin}/{self.provider}-auth/callback",)
        return args + ("valid-ticket",) if self.provider == "google" else args

    def test_https_callback_uses_canonical_origin_behind_http_and_ignores_headers(self):
        origin = "https://momo-ia4bhvubwa-uc.a.run.app"
        config_mock.MOMO_SERVICE_URL = origin + "/"
        auth_url = (
            "https://provider.example/authorize?state=opaque-state"
            "&code_challenge=opaque-challenge&code_challenge_method=S256"
        )
        for headers in (
            {"Host": "momo-ia4bhvubwa-uc.a.run.app"},
            {"Host": "attacker.example"},
            {
                "Host": "internal:8080",
                "Forwarded": 'for=192.0.2.1;proto=http;host="attacker.example"',
                "X-Forwarded-Host": "attacker.example",
                "X-Forwarded-Proto": "http",
                "X-Forwarded-Prefix": "/untrusted",
            },
        ):
            with self.subTest(headers=headers), patch.object(
                self.auth_module, "start_web_reauth", new_callable=AsyncMock, return_value=auth_url
            ) as mock_start:
                status, response_headers, _ = self.start_request(headers)

                self.assertEqual(status, 307)
                self.assertEqual(response_headers["location"], auth_url)
                mock_start.assert_awaited_once_with(*self.expected_start_args(origin))

    def test_invalid_canonical_configuration_fails_before_oauth(self):
        invalid_origins = (
            "momo.example", "/relative", "//momo.example", "https://", "https:/momo.example",
            "http://momo.example", "ftp://momo.example", "javascript:alert(1)",
            "https://user:password@momo.example", "https://@momo.example",
            "https://momo.example/path", "https://momo.example//",
            "https://momo.example?secret=value", "https://momo.example?",
            "https://momo.example#fragment", "https://momo.example#",
            "https://momo.example:bad", "https://momo.example:0",
            "https://momo.example:65536", "https://momo.example:",
            "https://momo.example\\@attacker.example", "https://momo%2eexample",
            " https://momo.example", "https://momo.example ",
            "https://momo.\nexample", "https://momo.\texample", "\x00https://momo.example",
            "https://momo.example\x7f", "https://bad_host.example", "https://-bad.example",
            "https://momo..example", "https://[::1", "https://[::1]attacker.example",
            "http://localhost.attacker.example", "http://127.0.0.1.attacker.example",
            "http://0.0.0.0", "http://192.168.1.1", "http://[::]", 123,
        )
        for origin in invalid_origins:
            with self.subTest(origin=origin), patch.object(
                config_mock, "MOMO_SERVICE_URL", origin
            ), patch.object(self.auth_module, "start_web_reauth", new_callable=AsyncMock) as mock_start:
                status, headers, body = self.start_request({"Host": "localhost:8000"})

                self.assertEqual(status, 500)
                self.assertNotIn("location", headers)
                self.assertIn(b"MOMO_SERVICE_URL", body)
                self.assertNotIn(b"secret=value", body)
                mock_start.assert_not_called()

    def test_missing_configuration_never_falls_back_to_request_or_forwarded_host(self):
        for origin in ("", None):
            for host in ("attacker.example", "localhost:8000"):
                with self.subTest(origin=origin, host=host), patch.object(
                    config_mock, "MOMO_SERVICE_URL", origin
                ), patch.object(self.auth_module, "start_web_reauth", new_callable=AsyncMock) as mock_start:
                    status, headers, _ = self.start_request({
                        "Host": host,
                        "X-Forwarded-Host": "momo.example",
                        "X-Forwarded-Proto": "https",
                    })

                    self.assertEqual(status, 500)
                    self.assertNotIn("location", headers)
                    mock_start.assert_not_called()

    def test_explicit_localhost_http_configuration_is_allowed(self):
        for origin in ("http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"):
            with self.subTest(origin=origin), patch.object(
                config_mock, "MOMO_SERVICE_URL", origin
            ), patch.object(
                self.auth_module, "start_web_reauth", new_callable=AsyncMock,
                return_value="https://provider.example/authorize",
            ) as mock_start:
                status, _, _ = self.start_request()

                self.assertEqual(status, 307)
                mock_start.assert_awaited_once_with(*self.expected_start_args(origin))

    def test_start_exception_does_not_notify_success(self):
        with patch.object(
            self.auth_module, "start_web_reauth", new_callable=AsyncMock,
            side_effect=RuntimeError("discovery failed"),
        ):
            status, _, _ = self.start_request()

        self.assertEqual(status, 500)
        self.mock_send.assert_not_called()
        self.mock_add.assert_not_called()

    def test_callback_records_only_delivered_notification_in_space_history(self):
        calls = MagicMock()
        calls.attach_mock(self.mock_send, "send")
        calls.attach_mock(self.mock_scope, "scope")
        calls.attach_mock(self.mock_add, "add")
        with patch.object(
            self.auth_module, "complete_web_reauth", new_callable=AsyncMock, return_value=True
        ) as mock_complete:
            status, _, body = call_app(
                "GET", f"/{self.provider}-auth/callback", "code=test-code&state=test-state"
            )

        self.assertEqual(status, 200)
        mock_complete.assert_awaited_once_with("test-code", "test-state")
        message = self.mock_send.call_args.args[1]
        self.assertEqual(calls.mock_calls, [
            call.send("spaces/test_space", message),
            call.scope(space="spaces/test_space"),
            call.add("space:spaces/test_space", "assistant", message),
        ])
        for text in (message, body.decode()):
            self.assertIn("authorization", text.lower())
            for forbidden in ("back online", "reconnected", "immediately", "instantly", "test-code", "test-state"):
                self.assertNotIn(forbidden, text.lower())

    def test_callback_success_without_chat_space_skips_notification_and_history(self):
        config_mock.CHAT_SPACE_ID = ""
        with patch.object(
            self.auth_module, "complete_web_reauth", new_callable=AsyncMock, return_value=True
        ):
            status, _, _ = call_app("GET", f"/{self.provider}-auth/callback", "code=code&state=state")

        self.assertEqual(status, 200)
        self.mock_send.assert_not_called()
        self.mock_scope.assert_not_called()
        self.mock_add.assert_not_called()

    def test_callback_chat_failure_preserves_oauth_success_without_history(self):
        for result in (False, None, RuntimeError("Chat unavailable")):
            with self.subTest(result=result), patch.object(
                self.auth_module, "complete_web_reauth", new_callable=AsyncMock, return_value=True
            ):
                self.mock_send.reset_mock(side_effect=True)
                if isinstance(result, Exception):
                    self.mock_send.side_effect = result
                else:
                    self.mock_send.return_value = result
                status, _, _ = call_app("GET", f"/{self.provider}-auth/callback", "code=code&state=state")

                self.assertEqual(status, 200)
                self.mock_send.assert_called_once()
                self.mock_scope.assert_not_called()
                self.mock_add.assert_not_called()

    def test_callback_history_failure_preserves_oauth_success(self):
        for failing_mock in (self.mock_scope, self.mock_add):
            with self.subTest(history_operation=failing_mock), patch.object(
                self.auth_module, "complete_web_reauth", new_callable=AsyncMock, return_value=True
            ):
                self.mock_send.reset_mock()
                failing_mock.side_effect = RuntimeError("History unavailable")
                status, _, _ = call_app("GET", f"/{self.provider}-auth/callback", "code=code&state=state")
                failing_mock.side_effect = None

                self.assertEqual(status, 200)
                self.mock_send.assert_called_once()

    def test_callback_oauth_failures_never_send_or_record_success(self):
        for query, outcome, expected_status, expected_calls in (
            ("error=access_denied&code=code&state=state", True, 400, 0),
            ("", True, 400, 0),
            ("code=code", True, 400, 0),
            ("state=state", True, 400, 0),
            ("code=code&state=state", False, 400, 1),
            ("code=code&state=state", RuntimeError("Token exchange failed"), 500, 1),
        ):
            with self.subTest(query=query, outcome=outcome), patch.object(
                self.auth_module, "complete_web_reauth", new_callable=AsyncMock
            ) as mock_complete:
                if isinstance(outcome, Exception):
                    mock_complete.side_effect = outcome
                else:
                    mock_complete.return_value = outcome
                status, _, _ = call_app("GET", f"/{self.provider}-auth/callback", query)

                self.assertEqual(status, expected_status)
                self.assertEqual(mock_complete.await_count, expected_calls)
                self.mock_send.assert_not_called()
                self.mock_scope.assert_not_called()
                self.mock_add.assert_not_called()


class TestGoogleAuthRoutes(OAuthRouteRegressions, unittest.TestCase):
    provider = "google"
    _PRIVATE_CODE = "private-oauth-code-784f"
    _PRIVATE_STATE = "private-oauth-state-861c"
    _PRIVATE_DETAIL = (
        "Scope has changed: access_token=private-access-token "
        "refresh_token=private-refresh-token client_secret=private-client-secret "
        "<script>private-script-payload</script><img src=x onerror=private-handler>"
    )

    @property
    def auth_module(self):
        return google_auth

    def _assert_callback_html_is_redacted(self, body):
        # Escaping arbitrary exception text is not redaction: credentials must
        # be absent even after the browser decodes HTML entities.
        text = html.unescape(body.decode()).lower()
        for forbidden in (
            self._PRIVATE_CODE, self._PRIVATE_STATE,
            "private-access-token", "private-refresh-token", "private-client-secret",
            "private-script-payload", "private-handler", "<script", "onerror=",
            "scope has changed",
        ):
            self.assertNotIn(forbidden.lower(), text)
        return text

    def _assert_google_callback_failure(self, outcome, expected_status, *, invalid_state=False):
        with patch.object(
            google_auth, "complete_web_reauth", new_callable=AsyncMock,
        ) as complete:
            if isinstance(outcome, Exception):
                complete.side_effect = outcome
            else:
                self.assertIs(outcome, False)
                complete.return_value = outcome
            status, headers, body = call_app(
                "GET", "/google-auth/callback",
                urlencode({"code": self._PRIVATE_CODE, "state": self._PRIVATE_STATE}),
            )

        self.assertEqual(status, expected_status)
        self.assertIn("text/html", headers.get("content-type", ""))
        self.assertNotIn("location", headers)
        complete.assert_awaited_once_with(self._PRIVATE_CODE, self._PRIVATE_STATE)
        text = self._assert_callback_html_is_redacted(body)
        if invalid_state:
            self.assertIn("state", text)
            self.assertRegex(text, r"\b(?:invalid|expired)\b")
        else:
            self.assertNotIn("expired", text)
            self.assertNotIn("invalid state", text)
        self.mock_send.assert_not_called()
        self.mock_scope.assert_not_called()
        self.mock_add.assert_not_called()
        return text

    def _assert_typed_google_callback_failure(self, reason, expected_status):
        # Use the real production exception and its fixed constructor contract,
        # not a lookalike or a boolean completion result. Its chained SDK error
        # deliberately contains material that must never reach the browser.
        failure = google_auth.GoogleReauthError(reason)
        self.assertIsInstance(failure, Exception)
        self.assertEqual(failure.reason, reason)
        public_message = str(failure)
        self.assertTrue(public_message)
        failure.__cause__ = RuntimeError(self._PRIVATE_DETAIL)
        self.assertEqual(str(failure), public_message)
        self._assert_callback_html_is_redacted(str(failure).encode())
        return self._assert_google_callback_failure(failure, expected_status)

    def test_google_callback_insufficient_scope_requires_permissions_not_new_state(self):
        text = self._assert_typed_google_callback_failure("insufficient_scope", 400)
        self.assertIn("required", text)
        self.assertRegex(text, r"\b(?:permissions?|access)\b")
        self.assertRegex(text, r"reconnect|sign[ -]?in|authoriz|grant|allow|approve")

    def test_google_callback_exchange_failure_is_502_not_expired_state(self):
        text = self._assert_typed_google_callback_failure("exchange_failed", 502)
        self.assertIn("exchange", text)

    def test_google_callback_configuration_failure_is_503_not_expired_state(self):
        text = self._assert_typed_google_callback_failure("config_unavailable", 503)
        self.assertRegex(text, r"configur|unavailable")

    def test_google_callback_persistence_failure_is_500_without_success_notice(self):
        text = self._assert_typed_google_callback_failure("persistence_failed", 500)
        self.assertRegex(text, r"\b(?:save|saved|saving|store|stored|storage|persist|persistence)\b")

    def test_google_callback_unexpected_exception_is_safe_generic_500(self):
        failure = RuntimeError(
            f"{self._PRIVATE_DETAIL} code={self._PRIVATE_CODE} state={self._PRIVATE_STATE}"
        )
        text = self._assert_google_callback_failure(failure, 500)
        self.assertRegex(text, r"failed|unavailable|unable|could not|couldn't|went wrong")

    def test_google_callback_false_retains_actual_invalid_or_expired_state_response(self):
        self._assert_google_callback_failure(False, 400, invalid_state=True)

    def test_google_callback_rejected_inputs_do_not_leak_code_or_state(self):
        for params in (
            {"code": self._PRIVATE_CODE},
            {"state": self._PRIVATE_STATE},
            {"error": "access_denied", "code": self._PRIVATE_CODE, "state": self._PRIVATE_STATE},
        ):
            with self.subTest(params=params), patch.object(
                google_auth, "complete_web_reauth", new_callable=AsyncMock,
            ) as complete:
                status, headers, body = call_app("GET", "/google-auth/callback", urlencode(params))
                self.assertEqual(status, 400)
                self.assertIn("text/html", headers.get("content-type", ""))
                self._assert_callback_html_is_redacted(body)
                complete.assert_not_called()
                self.mock_send.assert_not_called()
                self.mock_scope.assert_not_called()
                self.mock_add.assert_not_called()

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
        mock_start.assert_awaited_once_with("https://momo.example/google-auth/callback", "valid-ticket")

    @patch("google_auth.start_web_reauth", new_callable=AsyncMock, create=True)
    def test_google_auth_start_rejects_missing_ticket_without_starting_reauth(self, mock_start):
        status_code, _, _ = call_app("GET", "/google-auth/start")

        self.assertIn(status_code, {400, 403})
        mock_start.assert_not_called()

    @patch("google_auth.start_web_reauth", new_callable=AsyncMock)
    def test_google_auth_missing_ticket_is_rejected_even_without_configuration(self, mock_start):
        config_mock.MOMO_SERVICE_URL = ""

        status, _, _ = call_app("GET", "/google-auth/start")

        self.assertEqual(status, 403)
        mock_start.assert_not_called()

    @patch("google_auth.OAuthFlow")
    @patch("google_auth._consume_reauth_ticket", return_value=False)
    def test_google_auth_invalid_ticket_is_rejected_before_creating_flow(self, mock_consume, mock_flow):
        status, _, _ = call_app("GET", "/google-auth/start", "t=invalid-ticket")

        self.assertEqual(status, 403)
        mock_consume.assert_called_once_with("invalid-ticket")
        self.assertEqual(mock_flow.mock_calls, [])

    @patch("google_auth._get_db")
    @patch("google_auth.OAuthFlow")
    @patch("google_auth._load_web_client_config_from_sources", return_value={"web": {"client_id": "test"}})
    @patch("google_auth.os.path.exists", return_value=False)
    @patch("google_auth._consume_reauth_ticket", side_effect=[True, False])
    def test_google_auth_consumed_ticket_cannot_start_a_second_flow(
        self, mock_consume, mock_exists, mock_config, mock_flow, mock_db
    ):
        auth_url = "https://accounts.google.com/o/oauth2/auth?state=opaque-state"
        mock_flow.from_client_config.return_value.authorization_url.return_value = (auth_url, "opaque-state")

        first_status, headers, _ = call_app("GET", "/google-auth/start", "t=one-time-ticket")
        second_status, _, _ = call_app("GET", "/google-auth/start", "t=one-time-ticket")

        self.assertEqual(first_status, 307)
        self.assertEqual(headers["location"], auth_url)
        self.assertEqual(second_status, 403)
        self.assertEqual(mock_consume.call_args_list, [call("one-time-ticket"), call("one-time-ticket")])
        mock_flow.from_client_config.assert_called_once_with(
            {"web": {"client_id": "test"}}, scopes=["scope-a"],
            redirect_uri="https://momo.example/google-auth/callback",
        )
        pending_state = mock_db.return_value.collection.return_value.document.return_value.set
        pending_state.assert_called_once()
        self.assertEqual(pending_state.call_args.args[0]["redirect_uri"], "https://momo.example/google-auth/callback")

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
        self.assertIn("authorization updated", sent_message.lower())
        self.assertNotIn("test-code", sent_message)
        self.assertNotIn("test-state", sent_message)


class TestGranolaAuthRoutes(OAuthRouteRegressions, unittest.TestCase):
    provider = "granola"
    auth_module = _TEST_MODULES["granola_service"]

    def test_granola_auth_paths_are_open(self):
        self.assertIn("/granola-auth/start", main._OPEN_PATHS)
        self.assertIn("/granola-auth/callback", main._OPEN_PATHS)

    def test_granola_auth_discovery_failure_does_not_redirect_or_notify(self):
        with patch.object(self.auth_module, "start_web_reauth", new_callable=AsyncMock, return_value=None):
            status, headers, _ = self.start_request()

        self.assertEqual(status, 500)
        self.assertNotIn("location", headers)
        self.mock_send.assert_not_called()
        self.mock_add.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
