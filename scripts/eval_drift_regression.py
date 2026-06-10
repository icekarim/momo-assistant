"""Drift-detection regression snapshot tool.

Records which projects and open items the drift engine currently flags, so
later phases can prove they didn't change drift behaviour.

Pure read + compute — NO Firestore writes, NO nudge sending, NO Gemini calls.

Usage
-----
    # Write an empty baseline (useful as a structural sanity check):
    .venv/bin/python scripts/eval_drift_regression.py

    # Pull real data and write a live baseline:
    .venv/bin/python scripts/eval_drift_regression.py --live

    # Override output path:
    .venv/bin/python scripts/eval_drift_regression.py --live --out qa/drift_baseline.json

Design notes
------------
* This module intentionally does NOT import proactive_intelligence at module
  level.  proactive_intelligence chains into conversation_store → Firestore,
  calendar_service, claude_client, etc.  Following the established pattern in
  scripts/eval_kg_retrieval.py ("The script must NOT import knowledge_graph at
  module level"), we keep top-level imports stdlib-only so tests can import
  compute_drift_snapshot / write_snapshot without any network or credentials.

* DUPLICATION NOTE: _is_drift_activity_entry and _has_recent_activity are
  copied verbatim from proactive_intelligence.py (lines 61-94 as of this
  writing).  They are pure functions with zero side-effects.  If the originals
  change, update the copies here and re-run this test suite.  The alternative
  (lazy import inside the function) was rejected because it would still trigger
  the full Firestore import chain during tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Bootstrap sys.path so bare `import config` works (mirrors embedding_ab.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Duplicated pure helpers from proactive_intelligence ─────────────────────
# Source: proactive_intelligence.py lines 61-94
# Reason: avoids Firestore chain-import at module level (see design notes above)

_DRIFT_ACTIVITY_SOURCES = {"meeting", "meeting_notes", "email"}


def _is_drift_activity_entry(entry: dict) -> bool:
    """Return True if entry comes from a drift-relevant source type."""
    return entry.get("source_type") in _DRIFT_ACTIVITY_SOURCES


def _has_recent_activity(entry: dict, recent_entries: list[dict]) -> bool:
    """Return True if the item showed up in recent meeting/email activity.

    Matches by name or by shared related_projects (case-insensitive).
    """
    target_name = (entry.get("name") or "").strip().lower()
    target_projects = {
        project.strip().lower()
        for project in entry.get("related_projects", [])
        if project
    }
    if not target_name and not target_projects:
        return False

    for other in recent_entries:
        if other.get("id") == entry.get("id"):
            continue

        if target_name and (other.get("name") or "").strip().lower() == target_name:
            return True

        other_projects = {
            project.strip().lower()
            for project in other.get("related_projects", [])
            if project
        }
        if target_projects and target_projects & other_projects:
            return True

    return False


# ── Core pure function ───────────────────────────────────────────────────────


def compute_drift_snapshot(
    stale_items: list[dict],
    all_entries: list[dict],
    recent_entries: list[dict],
    threshold_days: int,
    now: datetime,
) -> dict:
    """Replicate _run_drift_engine's flagging logic — no writes, no nudges.

    Parameters
    ----------
    stale_items:
        Open commitments / action_items older than threshold_days.
        Mirrors the output of knowledge_graph.query_open_by_age().
    all_entries:
        Full KG entry set used to build per-project last-seen dates.
        Mirrors knowledge_graph.query_all_entries().
    recent_entries:
        KG entries within the threshold window (any source_type; filtered
        internally to drift-activity sources).
        Mirrors knowledge_graph.query_recent().
    threshold_days:
        Days of inactivity before an item/project is considered stale.
    now:
        Reference "now"; inject a fixed value in tests for determinism.

    Returns
    -------
    dict with keys:
        flagged_projects  – sorted list[str] of stale project names
        stale_open_items  – sorted list[str] of "name (id)" strings
        generated_at      – ISO-8601 string of `now`
    """
    cutoff = (now - timedelta(days=threshold_days)).strftime("%Y-%m-%d")

    # Filter recent entries to drift-activity sources (mirrors _run_drift_engine)
    recent_activity = [e for e in recent_entries if _is_drift_activity_entry(e)]

    # ── Build project → (canonical_name, last_seen_date) map ────────────────
    # Mirrors proactive_intelligence._run_drift_engine lines 463-478:
    #   for each drift-activity entry, track the most-recent source_date per project
    project_last_seen: dict[str, tuple[str, str]] = {}
    for e in all_entries:
        if not _is_drift_activity_entry(e):
            continue
        source_date = e.get("source_date", "")
        if not source_date:
            continue
        for proj in e.get("related_projects", []):
            normalized = proj.strip().lower()
            if not normalized:
                continue
            _, existing_date = project_last_seen.get(normalized, (proj, ""))
            if source_date > existing_date:
                project_last_seen[normalized] = (proj, source_date)

    # ── Stale projects: last activity on or before cutoff ───────────────────
    flagged_projects = sorted(
        proj
        for proj, last_date in project_last_seen.values()
        if last_date <= cutoff
    )

    # ── Stale open items: no name/project match in recent activity ───────────
    stale_open_items = sorted(
        f"{entry.get('name', 'Unnamed item')} ({entry.get('id', '')})"
        for entry in stale_items
        if not _has_recent_activity(entry, recent_activity)
    )

    return {
        "flagged_projects": flagged_projects,
        "stale_open_items": stale_open_items,
        "generated_at": now.isoformat(),
    }


# ── I/O helper ───────────────────────────────────────────────────────────────


def write_snapshot(snapshot: dict, out_path: Path) -> None:
    """Write snapshot as deterministic, diffable JSON.

    Uses sort_keys=True and indent=2 so the file is stable across runs
    (useful as a regression gate: `git diff qa/drift_baseline.json`).
    Trailing newline ensures POSIX-clean files and avoids false diffs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, sort_keys=True, indent=2) + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record which projects / open items the drift engine currently flags. "
            "Output is a deterministic JSON snapshot used as a regression gate."
        )
    )
    parser.add_argument(
        "--out",
        default="qa/drift_baseline.json",
        help="Output path for the snapshot (default: qa/drift_baseline.json)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Pull real data via knowledge_graph query functions (READ-ONLY). "
            "Requires valid GCP credentials and a reachable Firestore instance."
        ),
    )
    args = parser.parse_args()
    out_path = Path(args.out)

    if args.live:
        # Lazy imports keep module-level footprint stdlib-only (see design notes).
        # All three query functions are READ-ONLY; nothing is written to Firestore.
        from knowledge_graph import (  # noqa: PLC0415
            query_open_by_age,
            query_recent,
            query_all_entries,
        )
        import config  # noqa: PLC0415

        threshold_days: int = config.DRIFT_THRESHOLD_DAYS
        now = datetime.now()

        print("Fetching stale open items …")
        stale_items = query_open_by_age(min_days=threshold_days, limit=30)

        print("Fetching recent activity …")
        recent_entries = query_recent(days=threshold_days, limit=500)

        print("Fetching all entries for project last-seen …")
        all_entries = query_all_entries(limit=5000)
    else:
        print(
            "No --live flag: writing an empty baseline snapshot.\n"
            "Re-run with --live to capture real drift data."
        )
        threshold_days = 14
        now = datetime.now()
        stale_items, recent_entries, all_entries = [], [], []

    snapshot = compute_drift_snapshot(
        stale_items=stale_items,
        all_entries=all_entries,
        recent_entries=recent_entries,
        threshold_days=threshold_days,
        now=now,
    )

    write_snapshot(snapshot, out_path)

    print(f"\nSnapshot written → {out_path}")
    print(f"  flagged_projects : {len(snapshot['flagged_projects'])}")
    print(f"  stale_open_items : {len(snapshot['stale_open_items'])}")
    print(f"  generated_at     : {snapshot['generated_at']}")


if __name__ == "__main__":
    main()
