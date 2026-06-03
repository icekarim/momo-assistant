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


def test_extract_json_three_shapes_identical():
    expected = [{"a": 1}, {"b": 2}]
    fenced = '```json\n[{"a": 1}, {"b": 2}]\n```'
    clean = '[{"a": 1}, {"b": 2}]'
    prose = 'Here you go:\n[{"a": 1}, {"b": 2}]'
    assert cc.extract_json(fenced) == expected
    assert cc.extract_json(clean) == expected
    assert cc.extract_json(prose) == expected


def test_extract_json_dict_normalized_to_list():
    assert cc.extract_json('{"x": 1}') == [{"x": 1}]


def test_extract_json_garbage_returns_none():
    assert cc.extract_json("the model rambled with no json") is None
    assert cc.extract_json("") is None
    assert cc.extract_json(None) is None


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


def _tool_use_block(tid, name, inp):
    return type("TU", (), {"type": "tool_use", "id": tid, "name": name, "input": inp})()


def _msg(content, stop_reason="end_turn"):
    return type("M", (), {"content": content, "stop_reason": stop_reason})()


def test_gemini_tool_to_claude_shape():
    decl = {"name": "get_x", "description": "d", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}
    out = cc.gemini_tool_to_claude(decl)
    assert out == {"name": "get_x", "description": "d", "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}


def test_tool_loop_full_cycle(monkeypatch):
    text_block = type("B", (), {"type": "text", "text": "the answer is 5"})()
    seq = [
        _msg([_tool_use_block("tu1", "add", {"a": 2, "b": 3})], stop_reason="tool_use"),
        _msg([text_block], stop_reason="end_turn"),
    ]
    calls = {"n": 0}

    class FakeMsgs:
        def create(self, **kw):
            return seq[calls["n"]] if calls["n"] < len(seq) else seq[-1]
    monkeypatch.setattr(cc._client, "messages", FakeMsgs())

    def advance(**kw):
        m = seq[calls["n"]]; calls["n"] += 1; return m
    monkeypatch.setattr(cc._client.messages, "create", advance)

    dispatched = []
    def dispatch(name, inp):
        dispatched.append((name, inp))
        return "5"
    final, stop = cc.run_tool_loop(messages=[{"role": "user", "content": "add 2 and 3"}],
                                   tools=[{"name": "add", "input_schema": {}}], system=None, dispatch=dispatch)
    assert dispatched == [("add", {"a": 2, "b": 3})]
    assert final == "the answer is 5" and stop == "end_turn"


def test_tool_loop_malformed_input(monkeypatch):
    bad = type("TU", (), {"type": "tool_use", "id": "t", "name": "add", "input": None})()
    text_block = type("B", (), {"type": "text", "text": "handled"})()
    seq = [_msg([bad], "tool_use"), _msg([text_block], "end_turn")]
    i = {"n": 0}
    def advance(**kw):
        m = seq[i["n"]]; i["n"] += 1; return m
    monkeypatch.setattr(cc._client, "messages", type("M", (), {"create": staticmethod(advance)})())
    errors = []
    final, stop = cc.run_tool_loop(messages=[{"role": "user", "content": "x"}], tools=[], system=None,
                                   dispatch=lambda n, i: "should not be called",
                                   on_tool=lambda n, r, e: errors.append(e))
    assert errors == [True] and final == "handled"


def test_tool_loop_max_tokens_truncation(monkeypatch):
    text_block = type("B", (), {"type": "text", "text": "partial"})()
    def advance(**kw):
        return _msg([text_block], "max_tokens")
    monkeypatch.setattr(cc._client, "messages", type("M", (), {"create": staticmethod(advance)})())
    final, stop = cc.run_tool_loop(messages=[{"role": "user", "content": "x"}], tools=[], system=None,
                                   dispatch=lambda n, i: "x")
    assert stop == "max_tokens" and "partial" in final


def test_tool_loop_max_iteration_guard(monkeypatch):
    def advance(**kw):
        return _msg([_tool_use_block("t", "loop", {})], "tool_use")  # always wants another tool
    monkeypatch.setattr(cc._client, "messages", type("M", (), {"create": staticmethod(advance)})())
    final, stop = cc.run_tool_loop(messages=[{"role": "user", "content": "x"}], tools=[], system=None,
                                   dispatch=lambda n, i: "again", max_iterations=3)
    assert final == "" and stop == "tool_use"


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
