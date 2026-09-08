"""Safety tests: Jira write tools QUEUE for approval and never execute in-loop.

These need the REAL agent/config/jira_service modules and assert the dispatch
layer can never reach the REST write functions.

Co-run-safe isolation (HANDOFF_addon_cards.md §8): sibling test files
(test_jira_approval_safety.py, test_handle_card_click.py,
test_conversational_create_card.py, ...) stub app modules into sys.modules at
import time (``sys.modules["agent"] = MagicMock()`` etc.). Drop any cached or
stubbed instance of the modules this file needs — plus their app-level
imports — and re-import fresh, so this file passes alone AND in any co-run
order with the stub-based files.
"""

import json
import sys

# Pop in dependency order so each fresh import below binds fresh real deps
# (agent -> claude_client/observability/config, jira_service -> config).
for _name in ("agent", "claude_client", "observability", "jira_service", "config"):
    sys.modules.pop(_name, None)

import config  # noqa: E402
import jira_service  # noqa: E402
import agent  # noqa: E402


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
