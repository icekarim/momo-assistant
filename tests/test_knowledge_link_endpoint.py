"""Tests for POST /knowledge-link endpoint (TDD — RED → GREEN).

Module isolation note
─────────────────────
main.py has import-time and startup side-effects (google_auth.warmup, Google
API discovery caching), so heavy modules must be MagicMock-stubbed in
sys.modules before importing main.

CRITICAL: the stubbing must happen at TEST-RUN time inside a fixture, NOT at
module import time. pytest imports every test module during collection before
running anything; module-level sys.modules surgery poisons every test module
collected after this one (observed: 70 cross-file failures).
"""
import importlib
import sys
from unittest.mock import MagicMock

import pytest

_SENTINEL = object()
_MODULES_TO_ISOLATE = [
    "config", "briefing", "gmail_service", "calendar_service", "tasks_service",
    "gemini_service", "chat_service", "conversation_store", "agent",
    "granola_service",
    "google", "google.auth", "google.auth.transport",
    "google.auth.transport.requests", "google.oauth2", "google.oauth2.credentials",
    "google_auth_oauthlib", "google_auth_oauthlib.flow",
    "google.cloud", "google.cloud.firestore",
    "googleapiclient", "googleapiclient.discovery",
    "cachetools", "main", "google_auth",
    "knowledge_graph", "knowledge_linking",
]


class _DummyCache(dict):
    def __init__(self, maxsize=128, ttl=0):
        super().__init__()


@pytest.fixture(scope="module")
def isolated_main():
    originals = {name: sys.modules.get(name, _SENTINEL) for name in _MODULES_TO_ISOLATE}
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
    sys.modules["cachetools"] = MagicMock(TTLCache=_DummyCache)

    config_mock = MagicMock()
    config_mock.MOMO_API_SECRET = ""
    config_mock.CHAT_SPACE_ID = "spaces/test"
    config_mock.MOMO_SERVICE_URL = "https://momo.example"
    config_mock.GOOGLE_SCOPES = []
    config_mock.GRANOLA_ENABLED = False
    config_mock.KNOWLEDGE_GRAPH_ENABLED = False
    config_mock.KG_RESOLUTION_ENABLED = False
    config_mock.KG_LINKING_ENABLED = False
    config_mock.COMMITMENT_FOLLOWUP_DAYS = 3
    config_mock.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION = "knowledge_graph"
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
    cs_mock = MagicMock(
        get_conversation=MagicMock(),
        add_turn=MagicMock(),
        clear_conversation=MagicMock(),
        conversation_scope=MagicMock(),
        get_pending_task_actions=MagicMock(),
        clear_pending_task_actions=MagicMock(),
        store_pending_task_actions=MagicMock(),
        store_pending_task_actions_if_empty=MagicMock(),
    )
    sys.modules["conversation_store"] = cs_mock
    sys.modules["agent"] = MagicMock()
    sys.modules["granola_service"] = MagicMock()
    sys.modules["google_auth"] = MagicMock()
    kg_mock = MagicMock()
    sys.modules["knowledge_graph"] = kg_mock
    kl_mock = MagicMock()
    sys.modules["knowledge_linking"] = kl_mock

    main = importlib.import_module("main")
    from fastapi.testclient import TestClient

    yield {
        "main": main,
        "client_factory": lambda: TestClient(main.app, raise_server_exceptions=True),
        "config": config_mock,
        "kl": kl_mock,
        "kg": kg_mock,
        "cs": cs_mock,
    }

    for name in _MODULES_TO_ISOLATE:
        original = originals[name]
        if original is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def test_disabled_returns_disabled_and_schedules_nothing(isolated_main):
    isolated_main["config"].KG_LINKING_ENABLED = False
    isolated_main["kl"].run_linking.reset_mock()

    response = isolated_main["client_factory"]().post("/knowledge-link")

    assert response.status_code == 200
    assert response.json() == {"status": "disabled"}
    isolated_main["kl"].run_linking.assert_not_called()


def test_enabled_returns_started_and_calls_run_linking(isolated_main):
    isolated_main["config"].KG_LINKING_ENABLED = True
    isolated_main["config"].COMMITMENT_FOLLOWUP_DAYS = 3

    fake_commitments = [{"id": "c1", "name": "x", "content": "y"}]
    isolated_main["kg"].query_open_by_age.return_value = fake_commitments

    fake_db = MagicMock()
    isolated_main["cs"].get_db.return_value = fake_db

    isolated_main["kl"].run_linking.reset_mock()
    isolated_main["kl"].run_linking.return_value = {"linked": 1}

    response = isolated_main["client_factory"]().post("/knowledge-link")

    assert response.status_code == 200
    assert response.json() == {"status": "started"}

    isolated_main["kl"].run_linking.assert_called_once()
    call_args = isolated_main["kl"].run_linking.call_args.args
    assert call_args[0] == fake_commitments
    assert call_args[1] is fake_db


def test_endpoint_is_post_only(isolated_main):
    response = isolated_main["client_factory"]().get("/knowledge-link")
    assert response.status_code == 405
