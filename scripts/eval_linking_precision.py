"""Phase 2 linking-precision gate — scores labeled link set with Claude judge.

Reads the local labeled set ``eval_kg_labels.json`` (repo root), filters to
``type=="link"`` items, judges each commitment/evidence pair with Claude LIGHT
via ``make_claude_judge`` from ``knowledge_linking``, and writes
``qa/phase2_linking.json`` with precision/recall.

GATE: precision > 0.90 on the LINK class. Exits 1 when the gate fails.

Usage:
    .venv/bin/python scripts/eval_linking_precision.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from knowledge_linking import make_claude_judge  # noqa: E402

GATE_PRECISION = 0.90
_ATTEMPT = 1  # bump when iterating the judge prompt


def evaluate(link_items: list[dict], judge, threshold: float) -> dict:
    """Score every link-type label and compute precision/recall on LINK class.

    Args:
        link_items: list of label dicts (already filtered or mixed — items with
                    type != 'link' are silently skipped).
        judge:      callable (commitment_desc, evidence_desc) -> dict | None.
                    dict must have 'match' (bool) and 'confidence' (float).
                    None return → conservative NO-LINK, flagged judge_failed=True.
        threshold:  minimum confidence required for a LINK prediction.

    Returns the full report dict.
    """
    pairs = []
    tp = fp = fn = 0

    for item in link_items:
        if item.get("type") != "link":
            continue

        commitment = item["commitment"]
        name = commitment.get("name", "") or ""
        owner = commitment.get("owner") or "unknown"
        commitment_desc = f"{name} (owner: {owner})"
        evidence_desc = item["evidence"]["desc"]
        label = item["suggested"]

        judgment = judge(commitment_desc, evidence_desc)

        judge_failed = False
        confidence = None
        excerpt = ""

        if judgment is None:
            judge_failed = True
            predicted = "NO-LINK"
        elif judgment.get("match") and judgment.get("confidence", 0.0) >= threshold:
            predicted = "LINK"
            confidence = judgment["confidence"]
            excerpt = judgment.get("excerpt", "")
        else:
            predicted = "NO-LINK"
            confidence = judgment.get("confidence")
            excerpt = judgment.get("excerpt", "")

        if predicted == "LINK" and label == "LINK":
            tp += 1
        elif predicted == "LINK" and label == "NO-LINK":
            fp += 1
        elif predicted == "NO-LINK" and label == "LINK":
            fn += 1
        # TN: predicted NO-LINK and label NO-LINK → not counted

        pair_record: dict = {
            "commitment_name": name,
            "evidence_desc": evidence_desc,
            "predicted": predicted,
            "label": label,
            "confidence": confidence,
            "excerpt": excerpt,
        }
        if judge_failed:
            pair_record["judge_failed"] = True

        pairs.append(pair_record)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    gate = "PASS" if precision > GATE_PRECISION else "FAIL"

    return {
        "attempt": _ATTEMPT,
        "fn": fn,
        "fp": fp,
        "gate": gate,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_tier": "LIGHT",
        "pairs": pairs,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "tp": tp,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2 linking-precision gate over eval_kg_labels.json."
    )
    repo_root = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--labels",
        default=str(repo_root / "eval_kg_labels.json"),
        help="Path to the labeled set (default: repo-root eval_kg_labels.json).",
    )
    ap.add_argument(
        "--out",
        default=str(repo_root / "qa" / "phase2_linking.json"),
        help="Output JSON path (default: qa/phase2_linking.json).",
    )
    args = ap.parse_args()

    all_labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    link_items = [item for item in all_labels if item.get("type") == "link"]

    judge = make_claude_judge()
    threshold = config.KG_LINK_MIN_CONFIDENCE
    report = evaluate(link_items, judge, threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")

    link_count = len(report["pairs"])
    print(f"Scored {link_count} LINK/NO-LINK pairs from {args.labels}", flush=True)
    print(
        f"TP={report['tp']} FP={report['fp']} FN={report['fn']}",
        flush=True,
    )
    print(f"precision : {report['precision']:.4f} (gate > {GATE_PRECISION})", flush=True)
    print(f"recall    : {report['recall']:.4f}", flush=True)
    print(f"attempt   : {report['attempt']}", flush=True)
    print(f"Written   : {out_path}", flush=True)

    if report["gate"] == "FAIL":
        print("GATE FAILED", flush=True)
        misclassified = [
            p for p in report["pairs"]
            if p["predicted"] != p["label"]
        ]
        for p in misclassified:
            print(
                f"  MISS commitment={p['commitment_name']!r} "
                f"predicted={p['predicted']} label={p['label']} "
                f"conf={p['confidence']}",
                flush=True,
            )
        return 1

    print("GATE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
