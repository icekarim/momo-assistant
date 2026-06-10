"""TDD tests for scripts/gen_label_candidates.py — merge/link label candidate generator.

Tests inject fake data; no Firestore, no LLM, no network calls.
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake entities
# ---------------------------------------------------------------------------

def _entity(
    eid: str,
    entity_type: str = "commitment",
    name: str = "Do something",
    content: str = "Some content.",
    owner: str | None = None,
    related_people: list[str] | None = None,
    related_projects: list[str] | None = None,
    status: str | None = "open",
    source_type: str = "meeting",
) -> dict:
    return {
        "id": eid,
        "entity_type": entity_type,
        "name": name,
        "content": content,
        "owner": owner,
        "related_people": related_people or [],
        "related_projects": related_projects or [],
        "status": status,
        "source_type": source_type,
        "source_date": "2026-01-01",
    }


# ---------------------------------------------------------------------------
# Lazy import helper — avoids module-level import of gen_label_candidates
# which would fail before the file exists (RED phase).
# ---------------------------------------------------------------------------

def _import_module():
    import importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_label_candidates",
        Path(__file__).parent / "scripts" / "gen_label_candidates.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# test_token_overlap_pairs
# ===========================================================================

def test_token_overlap_pairs():
    """From values ['Sarah Chen', 'sarah.chen@x.com', 'Bob'], a Sarah-pair is
    proposed and no Bob-Sarah pair is proposed."""
    mod = _import_module()

    values = ["Sarah Chen", "sarah.chen@x.com", "Bob"]
    pairs = mod.person_overlap_pairs(values)

    # Flatten all pairs into sets for easy membership check
    pair_sets = [frozenset(p) for p in pairs]

    # A Sarah-pair should be present
    sarah_pair = frozenset({"Sarah Chen", "sarah.chen@x.com"})
    assert sarah_pair in pair_sets, (
        f"Expected Sarah-Sarah pair in results, got: {pairs}"
    )

    # No Bob-Sarah pair should appear
    bob_sarah_pair = frozenset({"Bob", "Sarah Chen"})
    assert bob_sarah_pair not in pair_sets, (
        f"Bob-Sarah pair must NOT appear: {pairs}"
    )
    bob_email_pair = frozenset({"Bob", "sarah.chen@x.com"})
    assert bob_email_pair not in pair_sets, (
        f"Bob-email pair must NOT appear: {pairs}"
    )


# ===========================================================================
# test_min_counts
# ===========================================================================

def _build_rich_entities() -> list[dict]:
    """Build a dataset large enough to produce ≥25 merge + ≥10 link candidates."""
    entities = []

    # Many person name-variant pairs (feeds merge candidates)
    person_groups = [
        ("Alice Smith", "alice.smith@corp.com", "alice smith"),
        ("Bob Jones", "bob.jones@corp.com", "B. Jones"),
        ("Carol White", "carol.white@corp.com", "Carol W."),
        ("David Lee", "david.lee@corp.com", "david lee"),
        ("Emma Brown", "emma.brown@corp.com", "E. Brown"),
        ("Frank Miller", "frank.miller@corp.com", "frank miller"),
        ("Grace Wilson", "grace.wilson@corp.com", "Grace W."),
        ("Henry Moore", "henry.moore@corp.com", "henry moore"),
        ("Iris Taylor", "iris.taylor@corp.com", "iris taylor"),
        ("Jack Anderson", "jack.anderson@corp.com", "jack anderson"),
        ("Karen Thomas", "karen.thomas@corp.com", "karen thomas"),
        ("Leo Jackson", "leo.jackson@corp.com", "leo jackson"),
        ("Maria White", "maria.white@corp.com", "maria white"),
        ("Nina Harris", "nina.harris@corp.com", "nina harris"),
    ]
    project_groups = [
        ("ProjectAlpha", "project alpha", "proj-alpha"),
        ("ProjectBeta", "project beta", "proj-beta"),
        ("ProjectGamma", "project gamma", "proj-gamma"),
    ]

    all_people = [v for grp in person_groups for v in grp]
    all_projects = [v for grp in project_groups for v in grp]

    # Commitments / action_items with open status — feed link candidates
    for i in range(20):
        grp = person_groups[i % len(person_groups)]
        proj_grp = project_groups[i % len(project_groups)]
        entities.append(_entity(
            eid=f"commit-{i}",
            entity_type="commitment" if i % 2 == 0 else "action_item",
            name=f"Follow up on item {i}",
            content=f"Owner needs to complete task number {i} for the project.",
            owner=grp[0],
            related_people=list(grp),
            related_projects=[proj_grp[0]],
            status="open",
        ))

    # Decisions/topics (also carry related_people variants)
    for i in range(15):
        grp = person_groups[i % len(person_groups)]
        proj_grp = project_groups[i % len(project_groups)]
        entities.append(_entity(
            eid=f"decision-{i}",
            entity_type="decision",
            name=f"Decided on approach {i}",
            content=f"Team decided approach {i} is best.",
            owner=grp[1],  # email variant
            related_people=[grp[0], grp[2]],  # different variants of same person
            related_projects=[proj_grp[1]],
            status=None,
        ))

    return entities


def test_min_counts():
    """With rich fake dataset and defaults (max_merge=30, max_link=15),
    output is CAPPED: 25–35 merge and 10–20 link (not thousands)."""
    mod = _import_module()

    entities = _build_rich_entities()
    candidates = mod.generate_candidates(entities)  # uses defaults: max_merge=30, max_link=15

    merge_items = [c for c in candidates if c["type"] == "merge"]
    link_items = [c for c in candidates if c["type"] == "link"]

    assert 25 <= len(merge_items) <= 35, (
        f"Expected 25–35 merge candidates (capped), got {len(merge_items)}"
    )
    assert 10 <= len(link_items) <= 20, (
        f"Expected 10–20 link candidates (capped), got {len(link_items)}"
    )


def test_determinism():
    """Same input twice produces byte-identical output (no randomness)."""
    mod = _import_module()

    entities = _build_rich_entities()
    c1 = mod.generate_candidates(entities)
    c2 = mod.generate_candidates(entities)

    assert c1 == c2, (
        f"generate_candidates must be deterministic; got differing results on second call"
    )


# ===========================================================================
# test_json_shape
# ===========================================================================

def test_json_shape(tmp_path):
    """Merge items have correct keys; link items have correct keys."""
    mod = _import_module()

    entities = _build_rich_entities()
    out_file = tmp_path / "candidates.json"
    mod.write_candidates(entities, out_file)

    data = json.loads(out_file.read_text())
    assert isinstance(data, list), "Output must be a JSON list"
    assert len(data) > 0, "Output must not be empty"

    merge_items = [c for c in data if c["type"] == "merge"]
    link_items = [c for c in data if c["type"] == "link"]

    assert merge_items, "Must have at least one merge item"
    assert link_items, "Must have at least one link item"

    # Check merge item shape
    for item in merge_items:
        assert set(item.keys()) >= {"type", "kind", "pair", "suggested", "reason"}, (
            f"Merge item missing keys: {item}"
        )
        assert item["type"] == "merge"
        assert item["kind"] in ("person", "project"), f"Bad kind: {item['kind']}"
        assert isinstance(item["pair"], list) and len(item["pair"]) == 2, (
            f"pair must be [a,b]: {item['pair']}"
        )
        assert item["suggested"] in ("MERGE", "NO-MERGE"), (
            f"suggested must be MERGE or NO-MERGE: {item['suggested']}"
        )
        assert isinstance(item["reason"], str) and item["reason"], (
            f"reason must be a non-empty string: {item}"
        )

    # Check link item shape
    for item in link_items:
        assert set(item.keys()) >= {"type", "commitment", "evidence", "suggested", "reason"}, (
            f"Link item missing keys: {item}"
        )
        assert item["type"] == "link"
        assert isinstance(item["commitment"], dict), "commitment must be a dict"
        assert set(item["commitment"].keys()) >= {"id", "name", "owner"}, (
            f"commitment missing keys: {item['commitment']}"
        )
        assert isinstance(item["evidence"], dict), "evidence must be a dict"
        assert "desc" in item["evidence"], f"evidence must have 'desc': {item['evidence']}"
        assert item["suggested"] in ("LINK", "NO-LINK"), (
            f"suggested must be LINK or NO-LINK: {item['suggested']}"
        )
        assert isinstance(item["reason"], str) and item["reason"], (
            f"reason must be a non-empty string: {item}"
        )
