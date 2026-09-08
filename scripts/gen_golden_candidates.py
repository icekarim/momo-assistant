"""Generate golden-set CANDIDATE queries for knowledge-graph retrieval eval.

Output is a DRAFT for human review — not ground truth.  The human corrects
entity associations before the CSV is used as an eval dataset.

Usage (dry-run, no Firestore):
    python scripts/gen_golden_candidates.py --target 50

Usage (live, reads real KG READ-ONLY):
    python scripts/gen_golden_candidates.py --live --target 100

CSV contract: query,expected_entity_ids  (ids `;`-joined)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# Bootstrap sys.path so bare `import config` works (mirrors embedding_ab.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Pure / deterministic core — NO Firestore, NO LLM calls
# ---------------------------------------------------------------------------

_MAX_IDS_PER_ROW = 5


def _sorted_by_date(entities: list[dict]) -> list[dict]:
    """Return entities sorted newest-first by source_date (string sort is fine for ISO dates)."""
    return sorted(entities, key=lambda e: e.get("source_date") or "", reverse=True)


def _extract_keyphrases(content: str) -> list[str]:
    """Extract short noun-phrase-like keyphrases from content (heuristic, no NLP deps)."""
    # Strip very short words and stop-words; keep 2-5 word runs of capitalised or
    # domain-flavoured tokens.
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "will", "we", "it", "that",
        "this", "are", "be", "have", "has", "had", "not", "as", "so", "up",
        "do", "did", "its", "our", "their", "they", "he", "she", "him", "her",
        "his", "my", "your", "all", "also", "about", "per", "via", "would",
        "could", "should", "then", "than", "into", "onto", "over", "under",
        "just", "been", "being", "more", "no", "yes", "if", "when", "which",
        "who", "whom", "what", "how", "why", "where", "each", "every", "both",
        "any", "some", "such", "other", "new", "old", "one", "two",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", content)
    phrases = []
    window: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in STOPWORDS or len(tok) < 3:
            if window:
                phrase = " ".join(window)
                if len(phrase) > 8:
                    phrases.append(phrase.lower())
                window = []
        else:
            window.append(tok)
            if len(window) == 4:
                phrases.append(" ".join(window).lower())
                window = window[1:]
    if window:
        phrase = " ".join(window)
        if len(phrase) > 8:
            phrases.append(phrase.lower())
    # Dedupe while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for p in phrases:
        if p not in seen:
            seen |= {p}
            result.append(p)
    return result


def generate_candidates(entities: list[dict], target: int = 50) -> list[dict]:
    """Mine `entities` and return candidate rows for retrieval eval.

    Each row: {"query": str, "expected_entity_ids": list[str]}

    Strategies (applied in order until `target` rows reached):
        (a) entity name → own id
        (b) "what happened with {project}?" → ids of that project's entities
        (c) "what did {person} commit to?" → commitment ids for that person
        (d) content-keyphrase queries for recent decisions/updates
    """
    rows: list[dict] = []
    seen_queries: set[str] = set()

    def _add(query: str, ids: list[str]) -> None:
        nonlocal seen_queries
        q = query.strip()
        if not q or q in seen_queries:
            return
        if not ids:
            return
        seen_queries |= {q}
        # Cap ids at _MAX_IDS_PER_ROW, prefer most-recent (entities already sorted)
        rows.append({"query": q, "expected_entity_ids": ids[:_MAX_IDS_PER_ROW]})

    sorted_ents = _sorted_by_date(entities)

    # ── (a) Entity name as query → own id ────────────────────────────────────
    for ent in sorted_ents:
        if len(rows) >= target:
            break
        name = (ent.get("name") or "").strip()
        eid = ent.get("id", "")
        if name and eid:
            _add(name, [eid])

    # ── (b) "what happened with {project}?" ──────────────────────────────────
    project_entities: dict[str, list[str]] = defaultdict(list)
    for ent in sorted_ents:
        eid = ent.get("id", "")
        if not eid:
            continue
        for proj in ent.get("related_projects") or []:
            proj = proj.strip()
            if proj:
                project_entities[proj].append(eid)

    for proj, ids in sorted(project_entities.items()):
        if len(rows) >= target:
            break
        _add(f"what happened with {proj}?", ids)

    # ── (c) "what did {person} commit to?" ───────────────────────────────────
    person_commitments: dict[str, list[str]] = defaultdict(list)
    for ent in sorted_ents:
        eid = ent.get("id", "")
        if not eid:
            continue
        if ent.get("entity_type") == "commitment":
            for person in ent.get("related_people") or []:
                person = person.strip()
                if person:
                    person_commitments[person].append(eid)
            owner = (ent.get("owner") or "").strip()
            if owner:
                person_commitments[owner].append(eid)

    for person, ids in sorted(person_commitments.items()):
        if len(rows) >= target:
            break
        # Dedupe ids while preserving order
        unique_ids: list[str] = list(dict.fromkeys(ids))
        _add(f"what did {person} commit to?", unique_ids)

    # ── (d) Content-keyphrase queries for recent decisions / updates ──────────
    target_types = {"decision", "update", "action_item", "blocker"}
    for ent in sorted_ents:
        if len(rows) >= target:
            break
        if ent.get("entity_type") not in target_types:
            continue
        eid = ent.get("id", "")
        content = (ent.get("content") or "").strip()
        if not (eid and content):
            continue
        keyphrases = _extract_keyphrases(content)
        for phrase in keyphrases[:3]:
            if len(rows) >= target:
                break
            _add(phrase, [eid])

    return rows


# ---------------------------------------------------------------------------
# CSV writer (shared by tests and main)
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], out_path: Path) -> None:
    """Write rows to a CSV with header `query,expected_entity_ids` (`;`-joined ids)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query", "expected_entity_ids"])
        for row in rows:
            ids_cell = ";".join(row.get("expected_entity_ids") or [])
            writer.writerow([row["query"], ids_cell])


# ---------------------------------------------------------------------------
# main() — argparse entry point; Firestore only touched with --live
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate golden-set CANDIDATE queries for KG retrieval eval."
    )
    ap.add_argument(
        "--out",
        default="qa/candidates/eval_kg_golden.csv",
        help="Output CSV path (default: qa/candidates/eval_kg_golden.csv)",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=50,
        help="Target number of candidate rows (default: 50)",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Stream real entities from Firestore knowledge_graph collection (READ-ONLY)",
    )
    args = ap.parse_args()

    if args.live:
        # Lazy imports so non-live mode never touches credentials or Firestore.
        import config  # noqa: PLC0415
        from conversation_store import get_db  # noqa: PLC0415

        db = get_db()
        docs = list(
            db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).stream()
        )
        entities = []
        for doc in docs:
            e = doc.to_dict()
            e.setdefault("id", doc.id)
            entities.append(e)
        print(f"[gen_golden_candidates] Loaded {len(entities)} entities from Firestore.")
    else:
        entities = []
        print("[gen_golden_candidates] Dry-run mode — no entities loaded (use --live for real data).")

    rows = generate_candidates(entities, target=args.target)
    out_path = Path(args.out)
    write_csv(rows, out_path)
    print(f"[gen_golden_candidates] Wrote {len(rows)} candidate rows → {out_path}")


if __name__ == "__main__":
    main()
