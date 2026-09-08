"""Label candidate generator for merge/link labeling.

Mines the knowledge graph for:
- MERGE candidates: likely person/project name-variant pairs
- LINK candidates: commitment<->evidence pairs for human review

No LLM calls. No Firestore writes. Deterministic mining only.

Usage:
    # Dry-run (empty dataset, tests the pipeline):
    .venv/bin/python scripts/gen_label_candidates.py

    # Live Firestore read (READ-ONLY):
    .venv/bin/python scripts/gen_label_candidates.py --live --out qa/candidates/eval_kg_labels.json

    # Override caps:
    .venv/bin/python scripts/gen_label_candidates.py --live --max-merge 30 --max-link 15
"""

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

# Bootstrap sys.path so bare `import config` works (mirrors embedding_ab.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _get_person_tokens(value: str | None) -> list[str]:
    """Delegate to knowledge_graph._person_tokens (lazy import, no reimplementation)."""
    from knowledge_graph import _person_tokens  # noqa: PLC0415

    return _person_tokens(value)


def _person_matches_local(a: str, b: str) -> bool:
    """Mirror knowledge_graph._person_matches using the canonical tokeniser."""
    q_tokens = _get_person_tokens(a)
    c_tokens = _get_person_tokens(b)

    if not q_tokens or not c_tokens:
        return False
    if q_tokens == c_tokens:
        return True

    q_set = set(q_tokens)
    c_set = set(c_tokens)

    if len(q_tokens) >= 2 and len(c_tokens) >= 2:
        if q_tokens[0] == c_tokens[0] and q_tokens[-1] == c_tokens[-1]:
            return True
        if q_set.issubset(c_set) or c_set.issubset(q_set):
            return True

    if len(q_tokens) == 1:
        return q_tokens[0] in c_set
    if len(c_tokens) == 1:
        return c_tokens[0] in q_set

    return False


def _project_token_set(value: str) -> frozenset:
    """Simple alphanumeric token set for project name comparison."""
    return frozenset(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _projects_overlap(a: str, b: str) -> bool:
    """True when two project strings share a meaningful token overlap (≥50% of smaller)."""
    ta = _project_token_set(a)
    tb = _project_token_set(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    shared = ta & tb
    if not shared:
        return False
    return len(shared) / min(len(ta), len(tb)) >= 0.5


def _jaccard(sa: frozenset, sb: frozenset) -> float:
    """Jaccard similarity between two sets; 1.0 = identical."""
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _person_cluster_key(a: str, b: str) -> frozenset:
    """Cluster key for a person pair = shared token set (or union when disjoint)."""
    ta = frozenset(_get_person_tokens(a))
    tb = frozenset(_get_person_tokens(b))
    shared = ta & tb
    return shared if shared else ta | tb


def _proj_cluster_key(a: str, b: str) -> frozenset:
    """Cluster key for a project pair = shared token set (or union when disjoint)."""
    ta = _project_token_set(a)
    tb = _project_token_set(b)
    shared = ta & tb
    return shared if shared else ta | tb


def _apply_cluster_limit(
    scored_pairs: list[tuple[float, str, str]],
    budget: int,
    cluster_key_fn,
    max_per_cluster: int = 2,
) -> list[tuple[str, str]]:
    """Select up to *budget* pairs; at most *max_per_cluster* pairs per cluster.

    *scored_pairs* must already be sorted in descending priority order.
    Returns a list of (a, b) tuples.
    """
    cluster_counts: dict[frozenset, int] = {}
    selected: list[tuple[str, str]] = []
    for _score, a, b in scored_pairs:
        if len(selected) >= budget:
            break
        key = cluster_key_fn(a, b)
        cnt = cluster_counts.get(key, 0)
        if cnt < max_per_cluster:
            selected.append((a, b))
            cluster_counts[key] = cnt + 1
    return selected


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _unique_ordered(values: list[str]) -> list[str]:
    """Deduplicate *values* while preserving first-seen order."""
    return list(dict.fromkeys(v for v in values if v))


def _collect_people(entities: list[dict]) -> list[str]:
    """Collect unique person strings from related_people + owner across all entities."""
    raw: list[str] = []
    for e in entities:
        raw.extend(e.get("related_people") or [])
        if e.get("owner"):
            raw.append(str(e["owner"]))
    return _unique_ordered(raw)


def _collect_projects(entities: list[dict]) -> list[str]:
    """Collect unique project strings from related_projects across all entities."""
    raw: list[str] = []
    for e in entities:
        raw.extend(e.get("related_projects") or [])
    return _unique_ordered(raw)


# ---------------------------------------------------------------------------
# Public API (called by tests and by main)
# ---------------------------------------------------------------------------


def person_overlap_pairs(values: list[str]) -> list[tuple[str, str]]:
    """Return (a, b) pairs whose person tokens overlap (MERGE signal).

    Uses knowledge_graph._person_tokens so matching logic is identical to what
    the graph itself uses for deduplication.
    """
    pairs = []
    for a, b in combinations(values, 2):
        if _person_matches_local(a, b):
            pairs.append((a, b))
    return pairs


def _select_merge_candidates(entities: list[dict], max_merge: int = 30) -> list[dict]:
    """Build a ranked, diverse, capped list of MERGE / NO-MERGE candidates.

    Budget allocation (deterministic, ~70/30 person/project, ~25% NO-MERGE):
      no_merge_budget  = round(max_merge * 0.25)   # distractors
      positive_budget  = max_merge - no_merge_budget
      person_budget    = round(positive_budget * 0.70)
      proj_budget      = positive_budget - person_budget
    Any unused project budget is redistributed to persons.
    Per-cluster limit = 2 (same canonical token root doesn't eat the whole budget).
    """
    people = _collect_people(entities)
    projects = _collect_projects(entities)

    # ── Score + sort all positive person pairs ──────────────────────────────
    person_scored: list[tuple[float, str, str]] = []
    for a, b in combinations(people, 2):
        if _person_matches_local(a, b):
            ta = frozenset(_get_person_tokens(a))
            tb = frozenset(_get_person_tokens(b))
            score = _jaccard(ta, tb)
            person_scored.append((score, a, b))
    person_scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    # ── Score + sort all positive project pairs ──────────────────────────────
    proj_scored: list[tuple[float, str, str]] = []
    for a, b in combinations(projects, 2):
        if _projects_overlap(a, b):
            ta = _project_token_set(a)
            tb = _project_token_set(b)
            score = _jaccard(ta, tb)
            proj_scored.append((score, a, b))
    proj_scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    # ── Budget allocation (round-half-up via int(x + 0.5)) ──────────────────
    no_merge_budget = max(1, int(max_merge * 0.25 + 0.5))
    positive_budget = max_merge - no_merge_budget
    person_budget = int(positive_budget * 0.70 + 0.5)
    proj_budget = positive_budget - person_budget

    # ── Cluster-limited selection ────────────────────────────────────────────
    selected_proj = _apply_cluster_limit(proj_scored, proj_budget, _proj_cluster_key)
    proj_slack = proj_budget - len(selected_proj)
    selected_person = _apply_cluster_limit(
        person_scored, person_budget + proj_slack, _person_cluster_key
    )

    candidates: list[dict] = []
    for a, b in selected_person:
        candidates.append(
            {
                "kind": "person",
                "pair": [a, b],
                "reason": "token overlap / subset match via _person_tokens",
                "suggested": "MERGE",
                "type": "merge",
            }
        )
    for a, b in selected_proj:
        candidates.append(
            {
                "kind": "project",
                "pair": [a, b],
                "reason": "alphanumeric token overlap",
                "suggested": "MERGE",
                "type": "merge",
            }
        )

    # ── NO-MERGE distractors (deterministic: first non-matching pairs in lex order) ─
    distractor_pairs: list[tuple[str, str, str]] = []
    for a, b in combinations(people, 2):
        if len(distractor_pairs) >= no_merge_budget:
            break
        if not _person_matches_local(a, b):
            distractor_pairs.append(("person", a, b))
    for a, b in combinations(projects, 2):
        if len(distractor_pairs) >= no_merge_budget:
            break
        if not _projects_overlap(a, b):
            distractor_pairs.append(("project", a, b))

    for kind, a, b in distractor_pairs[:no_merge_budget]:
        candidates.append(
            {
                "kind": kind,
                "pair": [a, b],
                "reason": "no token overlap detected",
                "suggested": "NO-MERGE",
                "type": "merge",
            }
        )

    return candidates


def _select_link_candidates(entities: list[dict], max_link: int = 15) -> list[dict]:
    """Build a ranked, capped list of LINK / NO-LINK candidates.

    Selection:
    - Prefers the most recent open commitments (source_date desc, id asc for ties).
    - Max 1 pair per commitment (each commitment appears exactly once).
    - ~70% LINK / ~30% NO-LINK mix.
    """
    open_items = [
        e
        for e in entities
        if e.get("entity_type") in ("commitment", "action_item")
        and e.get("status") == "open"
    ]
    if not open_items:
        return []

    # Stable sort: id asc first, then source_date desc (Python sort is stable)
    by_id = sorted(open_items, key=lambda e: e.get("id", ""))
    sorted_items = sorted(
        by_id,
        key=lambda e: e.get("source_date") or "0000-00-00",
        reverse=True,
    )

    link_budget = min(max_link, len(sorted_items))
    selected = sorted_items[:link_budget]
    link_count = int(link_budget * 0.70)  # floor → ~70% LINK, ensures ≥0

    candidates: list[dict] = []
    for i, item in enumerate(selected):
        name = item.get("name") or ""
        comm = {
            "id": item.get("id", ""),
            "name": name,
            "owner": item.get("owner") or "",
        }
        if i < link_count:
            evidence_desc = (
                f"email/task referencing '{name}'" if name else "related activity in source"
            )
            candidates.append(
                {
                    "commitment": comm,
                    "evidence": {"desc": evidence_desc},
                    "reason": "evidence description derived from commitment name/content",
                    "suggested": "LINK",
                    "type": "link",
                }
            )
        else:
            # Evidence sourced from a different LINK item (or another NO-LINK when
            # link_count==0 so there are no LINK items to borrow from).
            if link_count > 0:
                other = selected[i % link_count]
            else:
                other = selected[(i + 1) % len(selected)]
            other_name = other.get("name") or ""
            mismatched = (
                f"email/task referencing '{other_name}'"
                if other_name
                else "unrelated activity"
            )
            candidates.append(
                {
                    "commitment": comm,
                    "evidence": {"desc": mismatched},
                    "reason": "evidence belongs to a different commitment",
                    "suggested": "NO-LINK",
                    "type": "link",
                }
            )

    return candidates


def generate_candidates(
    entities: list[dict],
    max_merge: int = 30,
    max_link: int = 15,
) -> list[dict]:
    """Generate ranked, capped merge + link candidates from *entities*."""
    return _select_merge_candidates(entities, max_merge) + _select_link_candidates(
        entities, max_link
    )


def write_candidates(
    entities: list[dict],
    out_path,
    max_merge: int = 30,
    max_link: int = 15,
) -> None:
    """Generate candidates and write JSON to *out_path*."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates = generate_candidates(entities, max_merge=max_merge, max_link=max_link)
    out.write_text(json.dumps(candidates, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate merge/link label candidates from the knowledge graph.",
    )
    parser.add_argument(
        "--out",
        default="qa/candidates/eval_kg_labels.json",
        help="Output JSON file path (default: qa/candidates/eval_kg_labels.json)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Read entities from live Firestore (READ-ONLY). Default: dry-run with empty dataset.",
    )
    parser.add_argument(
        "--max-merge",
        type=int,
        default=30,
        dest="max_merge",
        help="Maximum merge candidates to emit (default: 30)",
    )
    parser.add_argument(
        "--max-link",
        type=int,
        default=15,
        dest="max_link",
        help="Maximum link candidates to emit (default: 15)",
    )
    args = parser.parse_args()

    if args.live:
        # Lazy imports — avoid Firestore connection in non-live mode
        import config as _cfg  # noqa: PLC0415
        from conversation_store import get_db  # noqa: PLC0415

        db = get_db()
        docs = db.collection(_cfg.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).stream()
        entities: list[dict] = []
        for doc in docs:
            d = doc.to_dict() or {}
            d.setdefault("id", doc.id)
            entities.append(d)
        print(f"[live] loaded {len(entities)} entities from Firestore", file=sys.stderr)
    else:
        entities = []
        print(
            "[dry-run] no entities loaded; pass --live to read from Firestore",
            file=sys.stderr,
        )

    write_candidates(entities, args.out, max_merge=args.max_merge, max_link=args.max_link)
    count = len(json.loads(Path(args.out).read_text()))
    print(f"[done] wrote {count} candidates → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
