"""TDD tests for interactive-card send support in chat_service.py.

Stubs heavy imports via sys.modules before importing chat_service so no
real Google auth or config is needed.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Stub every heavy dependency BEFORE importing chat_service
# ---------------------------------------------------------------------------
_google_stub = MagicMock()
_auth_stub = MagicMock()
_transport_stub = MagicMock()
_requests_stub = MagicMock()
_config_stub = MagicMock()

sys.modules.setdefault("google", _google_stub)
sys.modules.setdefault("google.auth", _auth_stub)
sys.modules.setdefault("google.auth.transport", _transport_stub)
sys.modules.setdefault("google.auth.transport.requests", _requests_stub)
sys.modules.setdefault("config", _config_stub)

sys.modules.pop("chat_service", None)  # evict any leaked sibling mock; import REAL module
import chat_service  # noqa: E402


class TestCardsSend(unittest.TestCase):
    # ------------------------------------------------------------------
    # (a) test_sends_cardsv2_body
    # When cards= is supplied, POST body must be {"cardsV2": cards}
    # and _split_message must NOT be called.
    # ------------------------------------------------------------------
    def test_sends_cardsv2_body(self):
        session = MagicMock()
        session.post.return_value = MagicMock(status_code=200)

        with patch.object(chat_service, "_get_chat_session", return_value=session):
            with patch.object(chat_service, "_split_message") as mock_split:
                cards = [{"cardId": "c", "card": {"header": {"title": "T"}}}]
                chat_service.send_chat_message("spaces/s", cards=cards)

        expected_url = "https://chat.googleapis.com/v1/spaces/s/messages"
        session.post.assert_called_once_with(
            expected_url, json={"cardsV2": cards}
        )
        mock_split.assert_not_called()

    # ------------------------------------------------------------------
    # (b) test_text_path_unchanged
    # When text= is supplied, existing behavior must be preserved:
    # _split_message is called and each chunk is POSTed as {"text": chunk}.
    # ------------------------------------------------------------------
    def test_text_path_unchanged(self):
        session = MagicMock()
        session.post.return_value = MagicMock(status_code=200)

        with patch.object(chat_service, "_get_chat_session", return_value=session):
            chat_service.send_chat_message("spaces/s", text="hi")

        expected_url = "https://chat.googleapis.com/v1/spaces/s/messages"
        session.post.assert_called_once_with(expected_url, json={"text": "hi"})

    def test_text_path_splits_long_message(self):
        """Long text must be split and each chunk POSTed separately."""
        session = MagicMock()
        session.post.return_value = MagicMock(status_code=200)

        long_text = "x\n" * 3000  # well above 4000-char limit
        with patch.object(chat_service, "_get_chat_session", return_value=session):
            chat_service.send_chat_message("spaces/s", text=long_text)

        # Must have been called more than once (split occurred)
        self.assertGreater(session.post.call_count, 1)
        # Every call must use {"text": ...} body
        for c in session.post.call_args_list:
            body = c.kwargs.get("json") or c.args[1] if len(c.args) > 1 else c.kwargs["json"]
            self.assertIn("text", body)
            self.assertNotIn("cardsV2", body)

    # ------------------------------------------------------------------
    # (c) test_cards_path_retries
    # First post raises, second succeeds — retry logic is preserved.
    # ------------------------------------------------------------------
    def test_cards_path_retries(self):
        session = MagicMock()
        ok_resp = MagicMock(status_code=200)
        session.post.side_effect = [Exception("network blip"), ok_resp]

        cards = [{"cardId": "retry-card"}]
        with patch.object(chat_service, "_get_chat_session", return_value=session):
            with patch("time.sleep"):  # don't actually wait
                chat_service.send_chat_message("spaces/s", cards=cards)

        self.assertEqual(session.post.call_count, 2)
        # Both calls must use cardsV2 body
        for c in session.post.call_args_list:
            body = c.kwargs.get("json") or (c.args[1] if len(c.args) > 1 else None)
            self.assertIsNotNone(body)
            self.assertIn("cardsV2", body)
            self.assertNotIn("text", body)


if __name__ == "__main__":
    unittest.main()
