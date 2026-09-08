"""KG v2 Phase 2 — commitment-evidence linking.

OVERLAY INVARIANT (LOCKED — see .omo/plans/kg-v2-memory-consolidation.md):
    The raw ``knowledge_graph`` collection is NEVER mutated by this module. All
    linking state is written to the overlay collection ``kg_links`` ONLY. There
    is no ``.update``/``.set`` against the knowledge_graph collection anywhere
    in this file. Rollback for the whole phase is ``KG_LINKING_ENABLED=false``
    (plus optionally deleting overlay docs); raw data is untouched in all cases.

Pipeline (``run_linking``): for each open commitment, search Gmail and Google
Tasks for evidence of completion → judge each candidate with Claude (LIGHT tier)
via ``_JUDGE_PROMPT`` → write a link doc to ``kg_links`` when judgment confidence
>= ``KG_LINK_MIN_CONFIDENCE`` (0.85). Uses the same search-term heuristic as
``proactive_intelligence._check_commitment_evidence`` (name if len>3, else
content[:50]). Boundary functions are fully injectable so the hermetic test suite
never touches network or Firestore.
"""

import hashlib
from datetime import datetime, timezone

import config
from claude_client import TaskComplexity, extract_json, extract_text, generate
from knowledge_graph import stable_key


_JUDGE_PROMPT = """\
Evaluate whether the evidence demonstrates that the commitment was fulfilled.

COMMITMENT:
{commitment}

EVIDENCE:
{evidence}

Respond with a JSON array containing exactly one object with these fields:
- "match": true if the evidence clearly shows the commitment was completed, false otherwise
- "confidence": a float 0.0-1.0 indicating how confident you are
- "excerpt": a short quote or phrase from the evidence supporting your verdict (empty string if match is false)

Output ONLY the JSON array, no other text. Example:
[{{"match": true, "confidence": 0.91, "excerpt": "sent the proposal on Monday"}}]"""


def _link_id(commitment_id: str, source_type: str, source_ref: str) -> str:
    """Deterministic 16-char id from commitment_id + source coords, so reruns
    over the same evidence overwrite a single doc instead of duplicating."""
    return hashlib.sha1(
        f"{commitment_id}|{source_type}|{source_ref}".encode()
    ).hexdigest()[:16]


def parse_judgment(text: str) -> dict | None:
    """Extract and validate a judgment dict from Claude's JSON array response.

    Returns None on: non-JSON, empty array, missing required keys, non-bool
    match, non-numeric confidence (e.g. "high"). Clamps confidence to [0, 1].
    """
    result = extract_json(text)
    if not result or not isinstance(result, list) or len(result) == 0:
        return None
    item = result[0]
    if not isinstance(item, dict):
        return None
    if "match" not in item or "confidence" not in item:
        return None
    if not isinstance(item["match"], bool):
        return None
    conf = item["confidence"]
    # Explicitly exclude bool: isinstance(True, int) is True in Python
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return None
    return {**item, "confidence": max(0.0, min(1.0, float(conf)))}


def make_claude_judge():
    """Return a judge callable: (commitment_desc, evidence_desc) -> dict | None.

    Each call fires exactly one LIGHT-tier generate call, then parses with
    parse_judgment. Returns None on any generation or parse failure so callers
    can treat it as no-match without crashing.
    """
    def judge(commitment_desc: str, evidence_desc: str) -> dict | None:
        prompt = _JUDGE_PROMPT.format(
            commitment=commitment_desc,
            evidence=evidence_desc,
        )
        try:
            msg = generate(prompt=prompt, tier=TaskComplexity.LIGHT)
            return parse_judgment(extract_text(msg))
        except Exception:
            return None
    return judge


def _link_doc(commitment: dict, source_type: str, source_ref: str,
              excerpt: str, confidence: float, now: datetime) -> dict:
    """Build the kg_links overlay document for one matched evidence item.

    Uses deterministic link_id so rerunning the same commitment+evidence pair
    overwrites the existing doc rather than creating a duplicate.
    """
    lid = _link_id(commitment["id"], source_type, source_ref)
    return {
        "link_id": lid,
        "link_type": "evidence_of",
        "commitment_entity_id": commitment["id"],
        "commitment_stable_key": stable_key(commitment),
        "evidence": {
            "source_type": source_type,
            "source_ref": source_ref,
            "excerpt": excerpt,
        },
        "confidence": confidence,
        "created_at": now.isoformat(),
    }


def run_linking(
    commitments: list[dict],
    db,
    now: datetime | None = None,
    *,
    judge=None,
    search_email_fn=None,
    find_task_fn=None,
    aliases_fn=None,
) -> dict:
    """Mine Gmail + Tasks candidates for each commitment, judge, and write kg_links docs.

    Writes ONLY to the overlay collection ``kg_links`` — the raw
    ``knowledge_graph`` collection is never mutated. Idempotent: link docs use
    deterministic ids so reruns overwrite rather than duplicate. Returns a
    summary count dict.

    Boundary injection: judge, search_email_fn, find_task_fn, and aliases_fn
    default to their production implementations via lazy imports, so hermetic
    tests can pass stubs without any sys.modules surgery.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if judge is None:
        judge = make_claude_judge()

    summary = {
        "commitments": 0,
        "candidates": 0,
        "linked": 0,
        "skipped_low_confidence": 0,
        "no_match": 0,
        "errors": 0,
    }

    for commitment in commitments:
        summary["commitments"] += 1
        try:
            # Resolve injectable defaults lazily so google service deps are
            # never imported at module level (hermetic test requirement).
            _search_email = search_email_fn
            if _search_email is None:
                from gmail_service import search_emails as _search_email  # noqa: PLC0415

            _find_task = find_task_fn
            if _find_task is None:
                from tasks_service import find_completed_task as _find_task  # noqa: PLC0415

            _aliases = aliases_fn
            if _aliases is None:
                from knowledge_graph import get_canonical_aliases as _aliases  # noqa: PLC0415

            name = commitment.get("name", "") or ""
            content = commitment.get("content", "") or ""
            search_terms = name if len(name) > 3 else content[:50]

            owner = commitment.get("owner", "") or ""
            owner_aliases = _aliases(owner)

            # Build commitment description (mirrors _check_commitment_evidence)
            commitment_desc = f"{name}: {content}\nOwner: {owner}"
            if owner_aliases:
                commitment_desc += f"\nAliases: {', '.join(owner_aliases)}"

            # ── Email candidates ───────────────────────────────────────────
            emails = _search_email(search_terms, days_back=30, max_results=3)
            if emails:
                for email in emails:
                    summary["candidates"] += 1
                    evidence_desc = (
                        f"From: {email.get('from', '?')}\n"
                        f"Subject: {email.get('subject', '?')}\n"
                        f"Body: {(email.get('body', '') or '')[:500]}"
                    )
                    judgment = judge(commitment_desc, evidence_desc)
                    if judgment is None or not judgment.get("match"):
                        summary["no_match"] += 1
                    elif judgment["confidence"] < config.KG_LINK_MIN_CONFIDENCE:
                        summary["skipped_low_confidence"] += 1
                    else:
                        doc = _link_doc(
                            commitment, "email", email["id"],
                            judgment.get("excerpt", ""),
                            judgment["confidence"], now,
                        )
                        db.collection(config.FIRESTORE_KG_LINKS_COLLECTION).document(
                            doc["link_id"]
                        ).set(doc)
                        summary["linked"] += 1

            # ── Task candidate ─────────────────────────────────────────────
            result = _find_task(name, days_back=30)
            if result:
                summary["candidates"] += 1
                evidence_desc = (
                    f"Task: {result.get('title', '?')}\n"
                    f"List: {result.get('list_name', '?')}"
                )
                judgment = judge(commitment_desc, evidence_desc)
                if judgment is None or not judgment.get("match"):
                    summary["no_match"] += 1
                elif judgment["confidence"] < config.KG_LINK_MIN_CONFIDENCE:
                    summary["skipped_low_confidence"] += 1
                else:
                    doc = _link_doc(
                        commitment, "task", result["id"],
                        judgment.get("excerpt", ""),
                        judgment["confidence"], now,
                    )
                    db.collection(config.FIRESTORE_KG_LINKS_COLLECTION).document(
                        doc["link_id"]
                    ).set(doc)
                    summary["linked"] += 1

        except Exception:
            summary["errors"] += 1
            continue

    return summary
