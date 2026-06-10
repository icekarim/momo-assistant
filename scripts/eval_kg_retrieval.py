"""READ-ONLY KG retrieval eval harness — recall@10 + MRR.

Does NOT write to Firestore. Loads a golden CSV, runs semantic_search
(optionally — only with --live), and writes a JSON report.

Usage:
    # Dry-run with a fake search stub (no Firestore):
    python scripts/eval_kg_retrieval.py --golden qa/golden.csv

    # Live run against real Firestore KG:
    python scripts/eval_kg_retrieval.py --golden qa/golden.csv --live
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Pure metric helpers
# ---------------------------------------------------------------------------

def compute_recall_at_k(expected_ids, retrieved, k=10):
    """Fraction of expected_ids found in the top-k retrieved results.

    Args:
        expected_ids: list of ground-truth entity id strings.
        retrieved:    list of result dicts, each with an ``id`` key.
        k:            cutoff rank (inclusive).

    Returns:
        float in [0.0, 1.0].
    """
    if not expected_ids:
        return 0.0
    top_k_ids = {r["id"] for r in retrieved[:k]}
    hits = sum(1 for eid in expected_ids if eid in top_k_ids)
    return hits / len(expected_ids)


def compute_mrr(expected_ids, retrieved):
    """Mean Reciprocal Rank of the first hit among expected_ids.

    Ranks are 1-based.  If none of the expected ids appear in the
    retrieved list, the reciprocal rank for this query is 0.

    Args:
        expected_ids: list of ground-truth entity id strings.
        retrieved:    ordered list of result dicts, each with an ``id`` key.

    Returns:
        float in [0.0, 1.0].
    """
    if not expected_ids:
        return 0.0
    expected_set = set(expected_ids)
    for rank, result in enumerate(retrieved, start=1):
        if result["id"] in expected_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_golden(path):
    """Load the golden CSV and return a list of row dicts.

    Expected CSV format::

        query,expected_entity_ids
        "some query","id1;id2;id3"

    Returns:
        list of dicts with keys ``query`` (str) and
        ``expected_entity_ids`` (list[str]).
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_ids = row.get("expected_entity_ids", "")
            ids = [eid.strip() for eid in raw_ids.split(";") if eid.strip()]
            rows.append({"query": row["query"].strip(), "expected_entity_ids": ids})
    return rows


# ---------------------------------------------------------------------------
# Core eval runner
# ---------------------------------------------------------------------------

def run_eval(golden_rows, out_path, search_fn):
    """Run recall@10 + MRR evaluation over golden_rows.

    Args:
        golden_rows: list of dicts from :func:`load_golden`.
        out_path:    pathlib.Path — destination for the JSON report.
        search_fn:   callable(query: str, limit: int) -> list[dict].
                     Each returned dict must have an ``id`` key.

    Returns:
        The result dict that was written to *out_path*.
    """
    per_query = []
    misses = []
    recall_scores = []
    mrr_scores = []

    for row in golden_rows:
        query = row["query"]
        expected = row["expected_entity_ids"]
        retrieved = search_fn(query, limit=10)

        recall = compute_recall_at_k(expected, retrieved, k=10)
        mrr = compute_mrr(expected, retrieved)

        recall_scores.append(recall)
        mrr_scores.append(mrr)

        # rank of first hit (1-based; None if miss)
        expected_set = set(expected)
        rank_of_first_hit = None
        for rank, r in enumerate(retrieved, start=1):
            if r["id"] in expected_set:
                rank_of_first_hit = rank
                break

        is_miss = rank_of_first_hit is None
        if is_miss:
            misses.append(query)

        per_query.append({
            "query": query,
            "hits": int(recall * len(expected)) if expected else 0,
            "rank_of_first_hit": rank_of_first_hit,
            "miss": is_miss,
        })

    macro_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    macro_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    result = {
        "recall_at_10": macro_recall,
        "mrr": macro_mrr,
        "per_query": per_query,
        "misses": misses,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate KG retrieval quality (recall@10, MRR) from a golden CSV."
    )
    ap.add_argument(
        "--golden",
        required=True,
        help="Path to the golden CSV (columns: query, expected_entity_ids).",
    )
    ap.add_argument(
        "--out",
        default="qa/phase0_baseline.json",
        help="Output JSON path (default: qa/phase0_baseline.json).",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help=(
            "Use the real semantic_search from knowledge_graph.py. "
            "Requires valid Firestore credentials. "
            "Without --live, the script exits after loading the golden CSV "
            "without touching Firestore."
        ),
    )
    args = ap.parse_args()

    golden_rows = load_golden(args.golden)
    print(f"Loaded {len(golden_rows)} golden rows from {args.golden}", flush=True)

    if args.live:
        # Lazy import — only when explicitly requested; keeps non-live mode
        # fully offline and import-clean.
        import knowledge_graph  # noqa: PLC0415

        def _live_search(query, limit=10):
            return knowledge_graph.semantic_search(query, limit=limit)

        search_fn = _live_search
    else:
        print(
            "No --live flag: running in stub mode (no Firestore calls). "
            "Pass --live to use the real semantic_search.",
            flush=True,
        )

        def _stub_search(query, limit=10):
            return []

        search_fn = _stub_search

    out_path = Path(args.out)
    result = run_eval(golden_rows, out_path, search_fn)

    print(f"recall@10 : {result['recall_at_10']:.4f}", flush=True)
    print(f"MRR       : {result['mrr']:.4f}", flush=True)
    print(f"Misses    : {len(result['misses'])}/{len(golden_rows)}", flush=True)
    print(f"Written   : {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
