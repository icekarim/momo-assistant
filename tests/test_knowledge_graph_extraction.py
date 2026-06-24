"""Unit tests for knowledge_graph._run_extraction with Claude backend."""

import json
import types

import pytest


def _fake_message(text: str):
    """Build a minimal fake Anthropic message with one text block."""
    block = types.SimpleNamespace(type="text", text=text)
    return types.SimpleNamespace(content=[block])


def test_run_extraction_parses_json_fenced_entities(monkeypatch):
    """_run_extraction returns the parsed entity list from a ```json fenced response."""
    entities = [
        {
            "entity_type": "decision",
            "name": "Launch in Q3",
            "content": "Team decided to launch in Q3.",
            "status": "open",
            "owner": "Alice",
            "related_people": ["Alice", "Bob"],
            "related_projects": ["ProjectX"],
            "tags": ["launch", "q3"],
        }
    ]
    fenced = "```json\n" + json.dumps(entities) + "\n```"
    fake_msg = _fake_message(fenced)

    import knowledge_graph  # noqa: PLC0415 — import after monkeypatch setup

    monkeypatch.setattr(knowledge_graph, "generate", lambda *args, **kwargs: fake_msg)

    result = knowledge_graph._run_extraction(
        source_type="meeting",
        source_title="Q3 Planning",
        content="We decided to launch in Q3. Alice owns it.",
        attendees=["Alice", "Bob"],
    )

    assert result == entities


def test_run_extraction_returns_empty_on_prose_no_json(monkeypatch):
    """_run_extraction returns [] when Claude returns plain prose with no JSON."""
    fake_msg = _fake_message("I cannot extract any entities from this content.")

    import knowledge_graph  # noqa: PLC0415

    monkeypatch.setattr(knowledge_graph, "generate", lambda *args, **kwargs: fake_msg)

    result = knowledge_graph._run_extraction(
        source_type="chat",
        source_title="Random chat",
        content="Hello world, nothing to extract here.",
        attendees=[],
    )

    assert result == []
