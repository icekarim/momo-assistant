"""extract_and_store_via_bg_tasks must enqueue work onto the supplied
BackgroundTasks queue rather than spawning a raw thread."""
from fastapi import BackgroundTasks
import knowledge_graph


def test_via_bg_tasks_enqueues(monkeypatch):
    calls = []

    def fake_extract_and_store(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(knowledge_graph, "extract_and_store", fake_extract_and_store)

    bg = BackgroundTasks()
    knowledge_graph.extract_and_store_via_bg_tasks(
        bg,
        source_type="meeting",
        source_id="m1",
        source_title="Test",
        source_date="2026-05-11",
        content="some content",
        attendees=["alice"],
    )

    # Should have queued, not executed yet
    assert calls == []
    assert len(bg.tasks) == 1

    # Drain the queue
    import asyncio
    asyncio.run(bg())
    assert len(calls) == 1
    assert calls[0][1]["source_id"] == "m1"
