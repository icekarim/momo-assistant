"""Safety tests: Jira write tools QUEUE for approval and never execute in-loop.

These import the real agent module (no heavy stubbing needed) and assert the
dispatch layer can never reach the REST write functions.
"""

import json

import config
import agent
import jira_service


def _boom(*_a, **_k):
    raise AssertionError("REST write was called during dispatch — must only queue!")


def test_create_jira_dispatch_queues_and_never_executes(monkeypatch):
    monkeypatch.setattr(jira_service, "create_jira_ticket", _boom)
    sink: list[dict] = []
    out = agent._dispatch(
        "create_jira_ticket",
        {"project": "OSD", "summary": "Test ticket", "issue_type": "Task"},
        pending_jira_actions=sink,
    )
    parsed = json.loads(out)
    assert parsed["status"] == "pending_approval"
    assert sink == [{"action": "create_jira", "project": "OSD", "summary": "Test ticket", "issue_type": "Task"}]


def test_comment_and_transition_dispatch_queue(monkeypatch):
    monkeypatch.setattr(jira_service, "add_jira_comment", _boom)
    monkeypatch.setattr(jira_service, "transition_jira_ticket", _boom)
    sink: list[dict] = []
    agent._dispatch("comment_jira_ticket", {"key": "OSD-1", "comment": "hi"}, pending_jira_actions=sink)
    agent._dispatch("transition_jira_ticket", {"key": "OSD-1", "transition": "Done"}, pending_jira_actions=sink)
    assert [a["action"] for a in sink] == ["comment_jira", "transition_jira"]
    assert sink[1]["transition"] == "Done"


def test_jira_actions_never_land_in_task_sink(monkeypatch):
    monkeypatch.setattr(jira_service, "create_jira_ticket", _boom)
    task_sink: list[dict] = []
    jira_sink: list[dict] = []
    agent._dispatch(
        "create_jira_ticket", {"project": "OSD", "summary": "X"},
        pending_task_actions=task_sink, pending_jira_actions=jira_sink,
    )
    assert task_sink == []            # never mixed into the task approval queue
    assert len(jira_sink) == 1


def test_write_tools_declared_only_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "JIRA_ENABLED", True)
    monkeypatch.setattr(config, "MCP_ENABLED", False)
    monkeypatch.setattr(config, "JIRA_WRITE_ENABLED", False)
    names_off = {t["name"] for t in agent._build_optional_tools()}
    monkeypatch.setattr(config, "JIRA_WRITE_ENABLED", True)
    names_on = {t["name"] for t in agent._build_optional_tools()}
    writes = {"create_jira_ticket", "comment_jira_ticket", "transition_jira_ticket"}
    assert not (writes & names_off), "write tools leaked while JIRA_WRITE_ENABLED=False"
    assert writes <= names_on
