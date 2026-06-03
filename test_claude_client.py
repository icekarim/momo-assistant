import anthropic
import httpx
import pytest

import claude_client as cc
from claude_client import TaskComplexity as T


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def _fake_message(text):
    block = type("B", (), {"type": "text", "text": text})()
    return type("M", (), {"stop_reason": "end_turn", "content": [block]})()


def test_tier_model_mapping():
    assert cc.TASK_MODEL_MAP[T.LIGHT] == "claude-haiku-4-5-20251001"
    assert cc.TASK_MODEL_MAP[T.STANDARD] == "claude-sonnet-4-6"
    assert cc.TASK_MODEL_MAP[T.DEEP] == "claude-opus-4-8"


def test_max_tokens_always_set(monkeypatch):
    seen = {}

    class FakeMsgs:
        def create(self, **kw):
            seen.update(kw)
            return _fake_message("ok")

    monkeypatch.setattr(cc._client, "messages", FakeMsgs())
    cc.generate("hi", tier=T.LIGHT)
    assert seen["max_tokens"] == 1024
    assert seen["model"] == "claude-haiku-4-5-20251001"


def test_extract_text_joins_text_blocks():
    msg = type("M", (), {"content": [
        type("B", (), {"type": "text", "text": "foo "})(),
        type("B", (), {"type": "tool_use", "text": "ignored"})(),
        type("B", (), {"type": "text", "text": "bar"})(),
    ]})()
    assert cc.extract_text(msg) == "foo bar"


def test_deep_downshifts_on_transient(monkeypatch):
    calls = {"n": 0}

    class FakeMsgs:
        def create(self, **kw):
            calls["n"] += 1
            if kw["model"] == "claude-opus-4-8":
                raise anthropic.InternalServerError("overloaded", response=_resp(529), body=None)
            return _fake_message("recovered")

    monkeypatch.setattr(cc._client, "messages", FakeMsgs())
    msg = cc.generate("x", tier=T.DEEP)
    assert cc.extract_text(msg) == "recovered"
    assert calls["n"] == 2  # opus failed, sonnet succeeded — one downshift


def test_deep_hardfails_on_bad_request(monkeypatch):
    calls = {"n": 0}

    class FakeMsgs:
        def create(self, **kw):
            calls["n"] += 1
            raise anthropic.BadRequestError("bad", response=_resp(400), body=None)

    monkeypatch.setattr(cc._client, "messages", FakeMsgs())
    with pytest.raises(anthropic.BadRequestError):
        cc.generate("x", tier=T.DEEP)
    assert calls["n"] == 1  # NO downshift on 400


def test_fallback_cap_not_exceeded(monkeypatch):
    calls = {"n": 0}

    class FakeMsgs:
        def create(self, **kw):
            calls["n"] += 1
            raise anthropic.InternalServerError("overloaded", response=_resp(529), body=None)

    monkeypatch.setattr(cc._client, "messages", FakeMsgs())
    with pytest.raises(anthropic.InternalServerError):
        cc.generate("x", tier=T.DEEP)
    assert calls["n"] == 2  # 1 opus + 1 sonnet, capped
