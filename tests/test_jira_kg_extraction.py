"""Unit tests for Jira -> knowledge-graph extraction.

Covers knowledge_graph.extract_from_jira_tickets (kwargs assembly + dedup id)
and jira_service ADF flattening / issue normalization. No network, no Firestore.
"""

import pytest


def _capture_extractions(monkeypatch):
    import knowledge_graph as kg

    calls = []
    monkeypatch.setattr(kg.config, "KNOWLEDGE_GRAPH_ENABLED", True)
    monkeypatch.setattr(
        kg, "extract_and_store_background", lambda **kw: calls.append(kw)
    )
    return kg, calls


def test_extract_from_jira_tickets_builds_expected_kwargs(monkeypatch):
    kg, calls = _capture_extractions(monkeypatch)

    kg.extract_from_jira_tickets([
        {
            "key": "OSD-117155",
            "summary": "DSW SDK+ Web Upgrade",
            "status": "Waiting for support",
            "priority": "High",
            "issue_type": "Task",
            "assignee": "Alex Rivera",
            "reporter": "Sam Brooks",
            "project": "Onboarding",
            "labels": ["sdk", "web"],
            "updated": "2026-06-20",
            "description": "Upgrade the DSW integration to SDK+.",
        }
    ])

    assert len(calls) == 1
    kw = calls[0]
    assert kw["source_type"] == "jira"
    assert kw["source_id"] == "jira-OSD-117155-2026-06-20"
    assert kw["source_date"] == "2026-06-20"
    assert "OSD-117155" in kw["source_title"]
    assert kw["attendees"] == ["Alex Rivera", "Sam Brooks"]
    assert "DSW SDK+ Web Upgrade" in kw["content"]
    assert "Status: Waiting for support" in kw["content"]
    assert "Upgrade the DSW integration to SDK+." in kw["content"]


def test_extract_from_jira_tickets_skips_keyless_and_respects_flag(monkeypatch):
    kg, calls = _capture_extractions(monkeypatch)

    kg.extract_from_jira_tickets([{"summary": "no key"}])
    assert calls == []

    monkeypatch.setattr(kg.config, "KNOWLEDGE_GRAPH_ENABLED", False)
    kg.extract_from_jira_tickets([{"key": "X-1", "summary": "s", "updated": "2026-01-01"}])
    assert calls == []


def test_adf_to_text_flattens_nested_document():
    import jira_service

    adf = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Line one."}]},
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Line "},
                {"type": "text", "text": "two."},
            ]},
        ],
    }
    text = jira_service._adf_to_text(adf)
    assert "Line one." in text
    assert "Line two." in text
    assert jira_service._adf_to_text(None) == ""
    assert jira_service._adf_to_text("plain") == "plain"


def test_normalize_issue_flattens_fields():
    import jira_service

    issue = {
        "key": "CP-6714",
        "fields": {
            "summary": "Zalando New Business",
            "status": {"name": "DISCOVERY"},
            "priority": {"name": "Medium"},
            "issuetype": {"name": "Opportunity"},
            "assignee": {"displayName": "Alex Rivera"},
            "reporter": None,
            "project": {"name": "Commercial Pipeline"},
            "labels": ["emea"],
            "updated": "2026-06-21T10:00:00.000+0000",
            "description": {"type": "doc", "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "New biz deal."}]}
            ]},
        },
    }
    rec = jira_service._normalize_issue(issue)
    assert rec["key"] == "CP-6714"
    assert rec["status"] == "DISCOVERY"
    assert rec["assignee"] == "Alex Rivera"
    assert rec["reporter"] == ""
    assert rec["project"] == "Commercial Pipeline"
    assert rec["updated"] == "2026-06-21"
    assert "New biz deal." in rec["description"]
