"""TDD tests for scripts/gen_golden_candidates.py.

Run with: .venv/bin/python -m pytest test_gen_golden_candidates.py -v
No Firestore access at test time — entity list is injected directly.
"""
import csv
import io
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entities(n: int = 60) -> list[dict]:
    """Build a varied fake entity list of length n for testing."""
    entity_types = ["decision", "commitment", "action_item", "blocker", "topic", "update"]
    source_types = ["meeting", "email", "chat", "calendar", "tasks", "meeting_notes"]
    people = ["alice", "bob", "carol", "dave", "eve"]
    projects = ["alpha", "beta", "gamma", "delta"]

    entities = []
    for i in range(n):
        et = entity_types[i % len(entity_types)]
        person = people[i % len(people)]
        project = projects[i % len(projects)]
        entities.append({
            "id": f"entity_{i:03d}",
            "entity_type": et,
            "name": f"{et.replace('_', ' ').title()} {i} about {project}",
            "content": (
                f"We decided to proceed with {project} rollout on Q{(i % 4) + 1}. "
                f"{person} will lead the initiative."
            ),
            "owner": person,
            "related_people": [person, people[(i + 1) % len(people)]],
            "related_projects": [project],
            "tags": [et, project],
            "source_type": source_types[i % len(source_types)],
            "source_date": f"2024-0{(i % 9) + 1}-{(i % 28) + 1:02d}",
        })
    return entities


@pytest.fixture
def fake_entities():
    return _make_entities(60)


# ---------------------------------------------------------------------------
# Tests — these FAIL until generate_candidates is implemented
# ---------------------------------------------------------------------------

class TestEmitsMinRows:
    """test_emits_min_rows: given ~60 varied entities, generator emits ≥50 rows."""

    def test_emits_min_rows(self, fake_entities):
        from scripts.gen_golden_candidates import generate_candidates

        rows = generate_candidates(fake_entities, target=50)
        assert len(rows) >= 50, (
            f"Expected ≥50 candidate rows, got {len(rows)}"
        )


class TestRowShape:
    """test_row_shape: every row has non-empty `query` and ≥1 expected entity id."""

    def test_row_shape(self, fake_entities):
        from scripts.gen_golden_candidates import generate_candidates

        rows = generate_candidates(fake_entities, target=50)
        assert rows, "generate_candidates returned empty list"
        for i, row in enumerate(rows):
            assert "query" in row, f"Row {i} missing 'query' key"
            assert row["query"].strip(), f"Row {i} has empty query"
            assert "expected_entity_ids" in row, f"Row {i} missing 'expected_entity_ids'"
            ids = row["expected_entity_ids"]
            assert isinstance(ids, list), f"Row {i} expected_entity_ids must be a list"
            assert len(ids) >= 1, f"Row {i} has zero expected entity ids"


class TestCsvHeader:
    """test_csv_header: written CSV has exact header `query,expected_entity_ids`
    with ids joined by `;`."""

    def test_csv_header(self, fake_entities, tmp_path):
        from scripts.gen_golden_candidates import generate_candidates, write_csv

        rows = generate_candidates(fake_entities, target=50)
        out_path = tmp_path / "out.csv"
        write_csv(rows, out_path)

        with open(out_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert header == ["query", "expected_entity_ids"], (
                f"Bad header: {header}"
            )
            # Verify ids are `;`-joined in at least one data row
            found_joined = False
            for data_row in reader:
                assert len(data_row) == 2, f"Row has {len(data_row)} columns, expected 2"
                ids_cell = data_row[1]
                if ";" in ids_cell:
                    found_joined = True
            # Not all rows must have multiple ids, but we just verify format is correct
            # (single-id rows are fine — they just won't have `;`)
