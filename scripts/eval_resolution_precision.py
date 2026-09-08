"""Phase 1 merge-precision gate — scores the labeled set against the resolver.

Reads the local labeled set ``eval_kg_labels.json`` (repo root), scores every
MERGE/NO-MERGE pair with the kind-appropriate deterministic scorer from
``knowledge_resolution``, and writes ``qa/phase1_resolution.json`` with
precision/recall. A pair is predicted MERGE when its score is at or above the
queue threshold (queue and auto both count as predicted-merge).

GATE: precision >= 0.95 AND recall >= 0.70 on the MERGE/NO-MERGE pairs. Exits 1
when the gate fails. No Firestore, no LLM calls — pure local scoring.

Usage:
    .venv/bin/python scripts/eval_resolution_precision.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from knowledge_resolution import score_pair  # noqa: E402

GATE_PRECISION = 0.95
GATE_RECALL = 0.70


def evaluate(labels: list[dict]) -> dict:
    """Score every merge-type label and compute precision/recall on the MERGE
    class. Returns the full report dict (also written to disk)."""
    threshold = config.KG_MERGE_QUEUE_THRESHOLD
    pairs = []
    tp = fp = fn = 0

    for item in labels:
        if item.get("type") != "merge":
            continue
        a, b = item["pair"]
        kind = item["kind"]
        score = score_pair(a, b, kind)
        predicted = "MERGE" if score >= threshold else "NO-MERGE"
        label = item["suggested"]

        if predicted == "MERGE" and label == "MERGE":
            tp += 1
        elif predicted == "MERGE" and label == "NO-MERGE":
            fp += 1
        elif predicted == "NO-MERGE" and label == "MERGE":
            fn += 1

        pairs.append({
            "pair": [a, b],
            "kind": kind,
            "score": round(score, 4),
            "predicted": predicted,
            "label": label,
        })

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "threshold_auto": config.KG_MERGE_AUTO_THRESHOLD,
        "threshold_queue": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "pairs": pairs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 1 merge-precision gate over eval_kg_labels.json."
    )
    repo_root = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--labels",
        default=str(repo_root / "eval_kg_labels.json"),
        help="Path to the labeled set (default: repo-root eval_kg_labels.json).",
    )
    ap.add_argument(
        "--out",
        default=str(repo_root / "qa" / "phase1_resolution.json"),
        help="Output JSON path (default: qa/phase1_resolution.json).",
    )
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    report = evaluate(labels)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")

    merge_pairs = len(report["pairs"])
    print(f"Scored {merge_pairs} MERGE/NO-MERGE pairs from {args.labels}", flush=True)
    print(
        f"TP={report['true_positives']} FP={report['false_positives']} "
        f"FN={report['false_negatives']}",
        flush=True,
    )
    print(f"precision : {report['precision']:.4f} (gate >= {GATE_PRECISION})", flush=True)
    print(f"recall    : {report['recall']:.4f} (gate >= {GATE_RECALL})", flush=True)
    print(f"Written   : {out_path}", flush=True)

    passed = report["precision"] >= GATE_PRECISION and report["recall"] >= GATE_RECALL
    if not passed:
        print("GATE FAILED", flush=True)
        misclassified = [
            p for p in report["pairs"]
            if (p["predicted"] == "MERGE") != (p["label"] == "MERGE")
        ]
        for p in misclassified:
            print(
                f"  MISS {p['kind']}: {p['pair']} score={p['score']} "
                f"predicted={p['predicted']} label={p['label']}",
                flush=True,
            )
        return 1

    print("GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
