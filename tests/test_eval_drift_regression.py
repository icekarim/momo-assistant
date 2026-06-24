"""TDD regression harness for the drift-detection snapshot tool.

Tests inject FAKE entries — no Firestore, no network, no live data.

RED phase: all tests fail until scripts/eval_drift_regression.py exists.
GREEN phase: import compute_drift_snapshot + write_snapshot from the script.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# ── fixed reference point ────────────────────────────────────────────────────

FIXED_NOW = datetime(2026, 6, 9, 12, 0, 0)
THRESHOLD = 14
# cutoff = 2026-05-26; dates on/before this are stale
CUTOFF_STR = (FIXED_NOW - timedelta(days=THRESHOLD)).strftime("%Y-%m-%d")


# ── helpers ──────────────────────────────────────────────────────────────────

def _entry(
    id: str,
    name: str,
    source_type: str,
    source_date: str,
    related_projects: list | None = None,
    entity_type: str = "commitment",
    status: str = "open",
) -> dict:
    return {
        "id": id,
        "name": name,
        "source_type": source_type,
        "source_date": source_date,
        "related_projects": related_projects or [],
        "entity_type": entity_type,
        "status": status,
    }


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def inputs():
    """
    Minimal fake data that exercises every branch in compute_drift_snapshot.

    Projects:
      Alpha Project — last drift-activity 2026-05-20 ≤ cutoff 2026-05-26 → STALE
      Beta Project  — last drift-activity 2026-06-01 > cutoff              → fresh
      Gamma Project — only appears via source_type=calendar (non-drift)    → ignored

    Stale open items (stale_items list):
      id-001  "Follow up with Alice"  → no name/project match in recent_entries → STALE
      id-002  "Review proposal"       → name "Review proposal" matches r1        → filtered out

    recent_entries contains one entry whose name equals "Review proposal".
    """
    stale_items = [
        _entry("id-001", "Follow up with Alice", "email", "2026-05-10"),
        _entry("id-002", "Review proposal", "email", "2026-05-15"),
    ]

    all_entries = [
        _entry("e1", "Alpha meeting", "meeting",  "2026-05-20",
               related_projects=["Alpha Project"]),
        _entry("e2", "Beta email",    "email",    "2026-06-01",
               related_projects=["Beta Project"]),
        # calendar source → must NOT contribute to project_last_seen
        _entry("e3", "Gamma cal",     "calendar", "2026-05-01",
               related_projects=["Gamma Project"]),
    ]

    recent_entries = [
        # name matches stale_items[1] → filters id-002 out
        _entry("r1", "Review proposal", "meeting", "2026-06-05"),
    ]

    return stale_items, all_entries, recent_entries


# ── test_snapshot_shape ───────────────────────────────────────────────────────

class TestSnapshotShape:

    def test_required_keys_present(self, inputs):
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        assert "flagged_projects" in result
        assert "stale_open_items" in result
        assert "generated_at" in result

    def test_flagged_projects_sorted_strings(self, inputs):
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        fp = result["flagged_projects"]
        assert isinstance(fp, list), "flagged_projects must be a list"
        assert all(isinstance(s, str) for s in fp), "all elements must be strings"
        assert fp == sorted(fp), "flagged_projects must be sorted"

    def test_stale_open_items_sorted_strings(self, inputs):
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        soi = result["stale_open_items"]
        assert isinstance(soi, list), "stale_open_items must be a list"
        assert all(isinstance(s, str) for s in soi), "all elements must be strings"
        assert soi == sorted(soi), "stale_open_items must be sorted"

    def test_stale_project_flagged(self, inputs):
        """Alpha Project (last seen 2026-05-20 ≤ cutoff) must appear."""
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        assert "Alpha Project" in result["flagged_projects"]

    def test_fresh_project_not_flagged(self, inputs):
        """Beta Project (last seen 2026-06-01 > cutoff) must NOT appear."""
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        assert "Beta Project" not in result["flagged_projects"]

    def test_non_drift_source_type_ignored(self, inputs):
        """Gamma Project (source_type=calendar) must NOT appear in flagged_projects."""
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        assert "Gamma Project" not in result["flagged_projects"]

    def test_item_without_recent_activity_included(self, inputs):
        """id-001 has no name/project match in recent_entries → must appear."""
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        assert any("id-001" in s for s in result["stale_open_items"])

    def test_item_with_recent_activity_excluded(self, inputs):
        """id-002 name-matches r1 in recent_entries → must NOT appear."""
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        assert not any("id-002" in s for s in result["stale_open_items"])

    def test_stale_open_item_format_name_id(self, inputs):
        """Items must be formatted as 'name (id)'."""
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        assert "Follow up with Alice (id-001)" in result["stale_open_items"]

    def test_generated_at_equals_now_isoformat(self, inputs):
        from scripts.eval_drift_regression import compute_drift_snapshot

        stale, all_e, recent = inputs
        result = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )

        assert result["generated_at"] == FIXED_NOW.isoformat()


# ── test_deterministic ────────────────────────────────────────────────────────

class TestDeterministic:

    def test_byte_identical_files(self, inputs, tmp_path):
        """Two runs on identical inputs must produce byte-identical JSON files."""
        from scripts.eval_drift_regression import compute_drift_snapshot, write_snapshot

        stale, all_e, recent = inputs

        out1 = tmp_path / "snap1.json"
        out2 = tmp_path / "snap2.json"

        for out in (out1, out2):
            snap = compute_drift_snapshot(
                stale_items=stale,
                all_entries=all_e,
                recent_entries=recent,
                threshold_days=THRESHOLD,
                now=FIXED_NOW,
            )
            write_snapshot(snap, out)

        assert out1.read_bytes() == out2.read_bytes()

    def test_written_json_parseable(self, inputs, tmp_path):
        from scripts.eval_drift_regression import compute_drift_snapshot, write_snapshot

        stale, all_e, recent = inputs
        out = tmp_path / "snap.json"

        snap = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )
        write_snapshot(snap, out)

        loaded = json.loads(out.read_text())
        assert loaded["flagged_projects"] == snap["flagged_projects"]
        assert loaded["stale_open_items"] == snap["stale_open_items"]
        assert loaded["generated_at"] == snap["generated_at"]

    def test_json_uses_sort_keys_and_indent(self, inputs, tmp_path):
        """JSON must be written with sort_keys=True, indent=2 for diffability."""
        from scripts.eval_drift_regression import compute_drift_snapshot, write_snapshot

        stale, all_e, recent = inputs
        out = tmp_path / "snap.json"

        snap = compute_drift_snapshot(
            stale_items=stale,
            all_entries=all_e,
            recent_entries=recent,
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )
        write_snapshot(snap, out)

        text = out.read_text()
        # sort_keys=True means "flagged_projects" comes before "generated_at" and "stale_open_items"
        fp_pos = text.index('"flagged_projects"')
        ga_pos = text.index('"generated_at"')
        soi_pos = text.index('"stale_open_items"')
        assert fp_pos < ga_pos < soi_pos, "keys must be in sorted order"

        # indent=2 means every line of a non-empty list starts with at least 4 spaces
        lines = text.splitlines()
        assert any(line.startswith("  ") for line in lines), "indent=2 required"

    def test_empty_inputs_produce_empty_lists(self, tmp_path):
        """With no entries, both lists are empty — snapshot is still valid."""
        from scripts.eval_drift_regression import compute_drift_snapshot, write_snapshot

        snap = compute_drift_snapshot(
            stale_items=[],
            all_entries=[],
            recent_entries=[],
            threshold_days=THRESHOLD,
            now=FIXED_NOW,
        )
        assert snap["flagged_projects"] == []
        assert snap["stale_open_items"] == []
        assert snap["generated_at"] == FIXED_NOW.isoformat()

        out = tmp_path / "empty.json"
        write_snapshot(snap, out)
        assert out.exists()
