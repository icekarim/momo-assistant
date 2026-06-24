"""TDD eval harness for KG retrieval — recall@10 + MRR.

Tests inject a FAKE search function — no live Firestore, no network,
no `semantic_search` import at test time.
"""
import json
import pytest

# ---------------------------------------------------------------------------
# Import ONLY the pure-function helpers from the eval script.
# The script must NOT import knowledge_graph at module level.
# ---------------------------------------------------------------------------
from scripts.eval_kg_retrieval import (
    compute_recall_at_k,
    compute_mrr,
    run_eval,
)


# ---------------------------------------------------------------------------
# Fake search helpers
# ---------------------------------------------------------------------------

def _make_search_fn(result_ids):
    """Return a search function that always returns a fixed list of result dicts."""
    def _search(query, limit=10):
        return [{"id": rid, "name": rid, "content": ""} for rid in result_ids]
    return _search


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_recall_at_10_perfect():
    """Search returns the expected id → recall@10 == 1.0."""
    expected_ids = ["e1"]
    retrieved = [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]
    assert compute_recall_at_k(expected_ids, retrieved, k=10) == 1.0


def test_mrr_first_hit_rank3():
    """Expected id appears at rank 3 (0-indexed position 2) → MRR == 1/3."""
    expected_ids = ["e3"]
    retrieved = [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}, {"id": "e4"}]
    assert compute_mrr(expected_ids, retrieved) == pytest.approx(1 / 3)


def test_miss_recorded(tmp_path):
    """Search returns zero expected ids → that query appears in the run's `misses` list."""
    golden_rows = [{"query": "orphan query", "expected_entity_ids": ["missing_id"]}]
    search_fn = _make_search_fn(["other_id_1", "other_id_2"])
    out_path = tmp_path / "result.json"

    result = run_eval(golden_rows, out_path, search_fn)

    assert "orphan query" in result["misses"]


def test_writes_baseline_json(tmp_path):
    """run_eval writes a JSON file with the required top-level keys."""
    golden_rows = [{"query": "test query", "expected_entity_ids": ["e1"]}]
    search_fn = _make_search_fn(["e1", "e2"])
    out_path = tmp_path / "baseline.json"

    run_eval(golden_rows, out_path, search_fn)

    assert out_path.exists(), "output JSON was not written"
    data = json.loads(out_path.read_text())
    assert "recall_at_10" in data
    assert "mrr" in data
    assert "per_query" in data
