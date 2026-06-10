"""KG v2 Phase 1 — entity resolution (canonical people & projects).

OVERLAY INVARIANT (LOCKED — see .omo/plans/kg-v2-memory-consolidation.md):
    The raw ``knowledge_graph`` collection is NEVER mutated by this module. All
    consolidation state is written to the overlay collections ``kg_canonical``
    and ``kg_merge_queue`` ONLY. There is no ``.update``/``.set`` against the
    knowledge_graph collection anywhere in this file. Rollback for the whole
    phase is ``KG_RESOLUTION_ENABLED=false`` (plus optionally deleting overlay
    docs); raw data is untouched in all cases.

Pipeline (``run_resolution``): mine candidate name pairs → score pair confidence
with deterministic heuristics → apply the hybrid policy:

    score >= KG_MERGE_AUTO_THRESHOLD (0.90)             → kg_canonical (auto)
    KG_MERGE_QUEUE_THRESHOLD (0.75) <= score < auto     → kg_merge_queue (pending)
    score < KG_MERGE_QUEUE_THRESHOLD                    → dropped

Person scoring reuses ``knowledge_graph._person_tokens`` / ``_normalize_text`` /
``_EMAIL_RE`` so matching stays identical to the graph's own dedup logic.
Projects use a simple alphanumeric token-overlap scorer (NOT person tokens). No
LLM calls live here; ``judge_fn`` is an optional extension point for the
ambiguous zone, defaulting to None.
"""

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone

import config
from knowledge_graph import _EMAIL_RE, _normalize_text, _person_tokens

# Single-token alias that is a strict subset of a multi-token name
# (e.g. "Karim" ⊂ "Karim Tounkara"): kept inside the [queue, auto) band so it is
# QUEUED for review, never auto-merged.
_SUBSET_SINGLE_TOKEN_CONFIDENCE = 0.80
# Multi-token alias that is a strict subset of a longer name
# (e.g. "Alex Smith" ⊂ "Alex B Smith").
_SUBSET_MULTI_TOKEN_CONFIDENCE = 0.90
# Email whose local-part tokens equal the other side's name tokens
# (e.g. "Sarah Chen" vs "sarah.chen@x.com"): high but slightly discounted, since
# an email local part is not a guaranteed display name.
_EMAIL_LOCAL_MATCH_CONFIDENCE = 0.95
# Weak signal: pairs sharing a token (same first OR same surname) but neither
# equal nor subset. Below the queue threshold → dropped. This is the ambiguous
# zone an optional judge_fn may override.
_SHARED_TOKEN_CONFIDENCE = 0.5


def _project_tokens(value: str | None) -> list[str]:
    """Alphanumeric tokens for project-name comparison.

    Deliberately NOT ``_person_tokens`` — projects carry no email/first-last
    semantics. Mirrors the tokenisation used by ``knowledge_graph.query_by_project``.
    """
    return re.findall(r"[a-z0-9]+", (value or "").lower())


# ── Confidence scoring ───────────────────────────────────────


def score_person_pair(a: str, b: str) -> float:
    a_tokens = _person_tokens(a)
    b_tokens = _person_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0

    a_set, b_set = set(a_tokens), set(b_tokens)
    one_is_email = bool(_EMAIL_RE.search((a or "").lower())) != bool(
        _EMAIL_RE.search((b or "").lower())
    )

    if a_set == b_set:
        return _EMAIL_LOCAL_MATCH_CONFIDENCE if one_is_email else 1.0

    if a_set < b_set or b_set < a_set:
        smaller = a_set if len(a_set) < len(b_set) else b_set
        if len(smaller) == 1:
            return _SUBSET_SINGLE_TOKEN_CONFIDENCE
        return _SUBSET_MULTI_TOKEN_CONFIDENCE

    if a_set & b_set:
        return _SHARED_TOKEN_CONFIDENCE

    return 0.0


def score_project_pair(a: str, b: str) -> float:
    a_set = set(_project_tokens(a))
    b_set = set(_project_tokens(b))
    if not a_set or not b_set:
        return 0.0
    if a_set == b_set:
        return 1.0
    union = a_set | b_set
    return len(a_set & b_set) / len(union)


def score_pair(a: str, b: str, kind: str, judge_fn=None) -> float:
    """Score a candidate pair with the kind-appropriate heuristic.

    ``judge_fn`` is an optional ``callable(pair_context) -> float | None``
    extension point for the ambiguous band ([_SHARED_TOKEN_CONFIDENCE, auto)).
    It is never invoked for definitive scores (exact matches at/above auto, or
    disjoint pairs below the band). Returns the heuristic when ``judge_fn`` is
    None or returns None.
    """
    raw = score_project_pair(a, b) if kind == "project" else score_person_pair(a, b)
    if judge_fn is not None and _SHARED_TOKEN_CONFIDENCE <= raw < config.KG_MERGE_AUTO_THRESHOLD:
        judged = judge_fn({"a": a, "b": b, "kind": kind, "heuristic_score": raw})
        if judged is not None:
            return float(judged)
    return raw


# ── Candidate mining ─────────────────────────────────────────


def _collect_names(entities: list[dict], kind: str) -> set[str]:
    names: set[str] = set()
    for entity in entities:
        if kind == "project":
            for project in entity.get("related_projects") or []:
                if project and str(project).strip():
                    names.add(str(project))
        else:
            for person in entity.get("related_people") or []:
                if person and str(person).strip():
                    names.add(str(person))
            owner = entity.get("owner")
            if owner and str(owner).strip():
                names.add(str(owner))
    return names


def mine_candidate_pairs(entities: list[dict], kind: str) -> list[tuple[str, str]]:
    """Blocking-based candidate generation: only names sharing >=1 token are
    compared, keeping this ~linear instead of O(n^2) over the full name set."""
    tokenizer = _project_tokens if kind == "project" else _person_tokens
    blocks: dict[str, set[str]] = defaultdict(set)
    for name in _collect_names(entities, kind):
        for token in set(tokenizer(name)):
            blocks[token].add(name)

    pairs: set[tuple[str, str]] = set()
    for group in blocks.values():
        ordered = sorted(group)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs.add((ordered[i], ordered[j]))
    return sorted(pairs)


# ── Overlay doc construction ─────────────────────────────────


def _canonical_id(aliases: list[str]) -> str:
    """Deterministic 16-char id from the normalized alias set, so reruns over
    the same cluster overwrite a single doc instead of duplicating."""
    normalized = sorted(_normalize_text(alias) for alias in aliases)
    return hashlib.sha1("|".join(normalized).encode()).hexdigest()[:16]


def _merge_queue_id(a: str, b: str, kind: str) -> str:
    normalized = sorted([_normalize_text(a), _normalize_text(b)])
    return hashlib.sha1("|".join(normalized + [kind]).encode()).hexdigest()[:16]


def _alias_tokens(aliases: list[str], kind: str) -> list[str]:
    tokenizer = _project_tokens if kind == "project" else _person_tokens
    tokens: set[str] = set()
    for alias in aliases:
        tokens.update(tokenizer(alias))
    return sorted(tokens)


def _select_display_name(aliases: list[str], kind: str) -> str:
    """Prefer the longest multi-token non-email alias (e.g. "Sarah Chen" over
    "sarah.chen@x.com")."""
    non_email = [a for a in aliases if not _EMAIL_RE.search((a or "").lower())]
    candidates = non_email or list(aliases)
    tokenizer = _project_tokens if kind == "project" else _person_tokens
    return max(candidates, key=lambda alias: (len(tokenizer(alias)), len(alias), alias))


def _canonical_doc(aliases: list[str], kind: str, confidence: float,
                   now: datetime, source: str) -> dict:
    return {
        "canonical_id": _canonical_id(aliases),
        "kind": kind,
        "display_name": _select_display_name(aliases, kind),
        "aliases": aliases,
        "alias_tokens": _alias_tokens(aliases, kind),
        "confidence": confidence,
        "created_at": now.isoformat(),
        "source": source,
    }


# ── Apply policy ─────────────────────────────────────────────


def apply_resolution(pair: tuple[str, str], kind: str, score: float, db,
                     now: datetime, source: str = "auto") -> str:
    """Apply the hybrid merge policy for one scored pair.

    Writes ONLY to the overlay collections and returns the action taken:
    "auto", "queue", "preserved", or "drop". "preserved" means the queue doc
    already carries a user decision ("approved"/"rejected") and is left
    untouched, so nightly reruns never resurrect decided pairs to "pending".
    """
    a, b = pair
    if score >= config.KG_MERGE_AUTO_THRESHOLD:
        aliases = sorted({a, b})
        doc = _canonical_doc(aliases, kind, score, now, source)
        db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).document(
            doc["canonical_id"]
        ).set(doc)
        return "auto"

    if score >= config.KG_MERGE_QUEUE_THRESHOLD:
        doc_ref = db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).document(
            _merge_queue_id(a, b, kind)
        )
        existing = doc_ref.get()
        if existing.exists and (existing.to_dict() or {}).get("status") in (
            "approved", "rejected",
        ):
            return "preserved"
        doc_ref.set({
            "pair": [a, b],
            "kind": kind,
            "confidence": score,
            "status": "pending",
            "proposed_at": now.isoformat(),
        })
        return "queue"

    return "drop"


def run_resolution(entities: list[dict], db, now: datetime | None = None,
                   judge_fn=None) -> dict:
    """Mine → score → apply over all person and project candidate pairs.

    Writes canonical/queue overlay docs via the injected ``db``. Idempotent:
    canonical and queue docs use deterministic ids, so reruns overwrite in place
    rather than duplicate. Returns a summary count dict.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    summary = {"candidates": 0, "auto": 0, "queued": 0, "preserved": 0, "dropped": 0}
    for kind in ("person", "project"):
        for a, b in mine_candidate_pairs(entities, kind):
            score = score_pair(a, b, kind, judge_fn=judge_fn)
            action = apply_resolution((a, b), kind, score, db, now)
            summary["candidates"] += 1
            if action == "auto":
                summary["auto"] += 1
            elif action == "queue":
                summary["queued"] += 1
            elif action == "preserved":
                summary["preserved"] += 1
            else:
                summary["dropped"] += 1
    return summary


# ── Approval flow (read pending queue + apply one-tap decisions) ─


def get_pending_merge_suggestions(limit: int = 3, db=None) -> list[dict]:
    """Return up to ``limit`` pending merge-queue docs, highest confidence first.

    Returns an empty list when KG_RESOLUTION_ENABLED is false, so the briefing
    and agent surfaces stay unchanged while the flag is off. Pure read against
    the kg_merge_queue overlay — NEVER touches the raw knowledge_graph collection.
    ``db`` is injectable for tests; defaults to the shared Firestore client.
    """
    if not config.KG_RESOLUTION_ENABLED:
        return []
    if db is None:
        from conversation_store import get_db
        db = get_db()

    pending = []
    for doc in db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).stream():
        data = doc.to_dict() or {}
        if data.get("status") != "pending":
            continue
        data["id"] = doc.id
        pending.append(data)

    pending.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
    return pending[:limit]


def apply_merge(queue_doc: dict, db, now: datetime | None = None) -> dict:
    """Approve a queued merge: write the kg_canonical doc (``source="approved"``)
    and flip the queue doc to ``status="approved"``.

    Writes ONLY to the overlay collections (kg_canonical, kg_merge_queue) — the
    raw knowledge_graph collection is never mutated. The canonical/queue ids are
    deterministic, so this overwrites the existing pending doc in place. Returns
    the written canonical doc.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    pair = list(queue_doc.get("pair", []))
    kind = queue_doc.get("kind", "person")
    confidence = queue_doc.get("confidence", config.KG_MERGE_QUEUE_THRESHOLD)

    aliases = sorted({a for a in pair if a})
    doc = _canonical_doc(aliases, kind, confidence, now, source="approved")
    db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).document(
        doc["canonical_id"]
    ).set(doc)

    db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).document(
        _merge_queue_id(pair[0], pair[1], kind)
    ).set({
        "pair": pair,
        "kind": kind,
        "confidence": confidence,
        "status": "approved",
        "proposed_at": queue_doc.get("proposed_at"),
        "decided_at": now.isoformat(),
    })
    return doc


def reject_merge(queue_doc: dict, db, now: datetime | None = None) -> None:
    """Reject a queued merge: flip ONLY the queue doc to ``status="rejected"``.

    No canonical doc is written. Writes ONLY to kg_merge_queue — the raw
    knowledge_graph collection is never mutated.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    pair = list(queue_doc.get("pair", []))
    kind = queue_doc.get("kind", "person")
    db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).document(
        _merge_queue_id(pair[0], pair[1], kind)
    ).set({
        "pair": pair,
        "kind": kind,
        "confidence": queue_doc.get("confidence", 0.0),
        "status": "rejected",
        "proposed_at": queue_doc.get("proposed_at"),
        "decided_at": now.isoformat(),
    })
