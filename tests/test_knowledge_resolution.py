"""Tests for knowledge_resolution.py — KG v2 Phase 1 entity-resolution core.

Covers the deterministic confidence scorer (person + project), the hybrid merge
policy (auto / queue / drop), the kg_canonical / kg_merge_queue overlay doc
shapes, deterministic-id idempotency, display-name selection, and the
mine->score->apply batch runner.

The overlay invariant is asserted directly: run_resolution must NEVER touch the
raw knowledge_graph collection. Firestore is faked in-memory, so nothing here
hits a live backend and no LLM calls are made.
"""

from datetime import datetime, timezone

import pytest

import config
import knowledge_resolution as kr


# ── In-memory Firestore fake ─────────────────────────────────


class _FakeDocSnapshot:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _FakeDocRef:
    def __init__(self, collection, doc_id):
        self._collection = collection
        self._id = doc_id

    def get(self):
        return _FakeDocSnapshot(self._id, self._collection.docs.get(self._id))

    def set(self, data):
        self._collection.docs[self._id] = dict(data)


class _FakeCollection:
    def __init__(self, name):
        self.name = name
        self.docs = {}
        self.stream_calls = 0

    def document(self, doc_id):
        return _FakeDocRef(self, doc_id)

    def stream(self):
        self.stream_calls += 1
        return [_FakeDocSnapshot(i, d) for i, d in self.docs.items()]


class FakeDB:
    def __init__(self):
        self.collections = {}

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = _FakeCollection(name)
        return self.collections[name]


_NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)


# ── Person scoring ───────────────────────────────────────────


class TestPersonScoring:
    def test_exact_token_set_case_variant(self):
        """Case-only difference is an exact token-set match → 1.0."""
        assert kr.score_person_pair("Aidan McWeeney", "Aidan Mcweeney") == 1.0

    def test_exact_token_set_order_variant(self):
        """Token order is irrelevant for a set match → 1.0."""
        assert kr.score_person_pair("Sarah Chen", "Chen Sarah") == 1.0

    def test_single_token_case_variant(self):
        """Single-token aliases that match case-insensitively → 1.0."""
        assert kr.score_person_pair("BJ", "Bj") == 1.0

    def test_email_local_part_equals_name(self):
        """Email local-part tokens equal to name tokens → 0.95."""
        score = kr.score_person_pair("Sarah Chen", "sarah.chen@example.com")
        assert score == pytest.approx(0.95)

    def test_email_single_token(self):
        """Single-token name vs its email → email-discounted high confidence."""
        score = kr.score_person_pair("Sam", "sam@example.com")
        assert score == pytest.approx(0.95)

    def test_last_comma_first_reorder(self):
        """'Last, First' reorder is a set match → >= 0.9 (auto-merge band)."""
        score = kr.score_person_pair("Harsh Nigam", "Nigam, Harsh")
        assert score >= 0.9

    def test_single_token_subset_is_queue_band(self):
        """Single token ⊂ multi-token name (Alex ⊂ Alex Rivera) → QUEUE,
        not auto: confidence in [queue, auto)."""
        score = kr.score_person_pair("Alex", "Alex Rivera")
        assert config.KG_MERGE_QUEUE_THRESHOLD <= score < config.KG_MERGE_AUTO_THRESHOLD

    def test_shared_first_name_different_surname_dropped(self):
        """Shared first name, different surname → below queue threshold."""
        score = kr.score_person_pair("Alex Frick", "Alex Gurevich")
        assert score < config.KG_MERGE_QUEUE_THRESHOLD

    def test_shared_surname_different_first_dropped(self):
        """Shared surname, different first name → below queue threshold."""
        score = kr.score_person_pair("Aniyah Smith", "Julia Smith")
        assert score < config.KG_MERGE_QUEUE_THRESHOLD

    def test_name_position_crossover_dropped(self):
        """Crossover (Acel Joseph vs Joseph Karlin) → below queue threshold."""
        score = kr.score_person_pair("Acel Joseph", "Joseph Karlin")
        assert score < config.KG_MERGE_QUEUE_THRESHOLD

    def test_hyphenated_shared_surname_dropped(self):
        """Hyphenated shared surname, different first → below queue threshold."""
        score = kr.score_person_pair("Aidan McWeeney", "Thomas Bugas-McWeeney")
        assert score < config.KG_MERGE_QUEUE_THRESHOLD

    def test_disjoint_names_zero(self):
        """No shared tokens → 0.0."""
        assert kr.score_person_pair("Bob Jones", "Alice Walker") == 0.0

    def test_empty_returns_zero(self):
        assert kr.score_person_pair("", "Sarah Chen") == 0.0
        assert kr.score_person_pair("Sarah Chen", "") == 0.0


# ── Project scoring ──────────────────────────────────────────


class TestProjectScoring:
    def test_case_variant_is_one(self):
        assert kr.score_project_pair("Ads Team", "ads team") == 1.0

    def test_case_variant_no_space_diff(self):
        assert kr.score_project_pair("Aftersell", "AfterSell") == 1.0

    def test_multi_word_case_variant(self):
        assert kr.score_project_pair("App Layout Experiment", "app layout experiment") == 1.0

    def test_partial_overlap_below_threshold(self):
        """Token overlap that isn't a full set match scores below queue."""
        score = kr.score_project_pair("Android", "Android Integration")
        assert score < config.KG_MERGE_QUEUE_THRESHOLD

    def test_disjoint_projects_zero(self):
        assert kr.score_project_pair("Aftersales", "Pricing") == 0.0

    def test_uses_project_tokens_not_person_tokens(self):
        """Projects must NOT route through _person_tokens (no first/last or
        email semantics). A 3-word project with one differing word stays a
        token-overlap score, never a person 'first+last' style match."""
        score = kr.score_project_pair("Q2 Launch Plan", "Q3 Launch Plan")
        # 2 of 4 distinct tokens shared → jaccard 0.5, below queue.
        assert score == pytest.approx(2 / 4)


# ── score_pair dispatcher + judge_fn extension point ─────────


class TestScorePairDispatcher:
    def test_person_dispatch(self):
        assert kr.score_pair("Aidan McWeeney", "Aidan Mcweeney", "person") == 1.0

    def test_project_dispatch(self):
        assert kr.score_pair("Ads Team", "ads team", "project") == 1.0

    def test_judge_fn_defaults_none(self):
        """Default judge_fn=None → pure heuristic, hard negative stays dropped."""
        score = kr.score_pair("Alex Frick", "Alex Gurevich", "person")
        assert score < config.KG_MERGE_QUEUE_THRESHOLD

    def test_judge_fn_overrides_ambiguous_zone(self):
        """A judge_fn lifts an ambiguous shared-token pair above the threshold."""
        score = kr.score_pair(
            "Alex Frick", "Alex Gurevich", "person", judge_fn=lambda ctx: 0.99
        )
        assert score == pytest.approx(0.99)

    def test_judge_fn_can_override_queue_band(self):
        """A judge_fn also applies to the [queue, auto) subset band."""
        score = kr.score_pair(
            "Alex", "Alex Rivera", "person", judge_fn=lambda ctx: 0.1
        )
        assert score == pytest.approx(0.1)

    def test_judge_fn_not_called_for_exact_match(self):
        """Exact matches (1.0) are outside the ambiguous band → judge ignored."""
        score = kr.score_pair("BJ", "Bj", "person", judge_fn=lambda ctx: 0.0)
        assert score == 1.0

    def test_judge_fn_not_called_for_disjoint(self):
        """Totally unrelated pairs (0.0) are below the band → judge ignored."""
        score = kr.score_pair(
            "Bob Jones", "Alice Walker", "person", judge_fn=lambda ctx: 0.99
        )
        assert score == 0.0

    def test_judge_fn_receives_context(self):
        captured = {}

        def _judge(ctx):
            captured.update(ctx)
            return None  # None → keep heuristic

        score = kr.score_pair("Alex Frick", "Alex Gurevich", "person", judge_fn=_judge)
        assert captured.get("a") == "Alex Frick"
        assert captured.get("b") == "Alex Gurevich"
        assert captured.get("kind") == "person"
        assert "heuristic_score" in captured
        # judge returned None → heuristic preserved
        assert score < config.KG_MERGE_QUEUE_THRESHOLD


# ── Apply policy ─────────────────────────────────────────────


class TestApplyPolicy:
    def test_auto_writes_canonical(self):
        db = FakeDB()
        action = kr.apply_resolution(
            ("Sarah Chen", "sarah.chen@example.com"), "person", 0.95, db, _NOW
        )
        assert action == "auto"
        canonical = db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs
        assert len(canonical) == 1
        doc = next(iter(canonical.values()))
        assert doc["source"] == "auto"
        # nothing queued
        assert db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs == {}

    def test_queue_writes_pending(self):
        db = FakeDB()
        action = kr.apply_resolution(
            ("Alex", "Alex Rivera"), "person", 0.80, db, _NOW
        )
        assert action == "queue"
        queue = db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs
        assert len(queue) == 1
        doc = next(iter(queue.values()))
        assert doc["status"] == "pending"
        assert doc["pair"] == ["Alex", "Alex Rivera"]
        assert doc["kind"] == "person"
        assert doc["confidence"] == pytest.approx(0.80)
        assert "proposed_at" in doc
        # nothing auto-merged
        assert db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs == {}

    def test_below_threshold_writes_nothing(self):
        db = FakeDB()
        action = kr.apply_resolution(
            ("Alex Frick", "Alex Gurevich"), "person", 0.5, db, _NOW
        )
        assert action == "drop"
        assert db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs == {}
        assert db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs == {}

    def test_auto_threshold_boundary_is_auto(self):
        """A score exactly at the auto threshold auto-merges."""
        db = FakeDB()
        action = kr.apply_resolution(
            ("Ads Team", "ads team"), "project", config.KG_MERGE_AUTO_THRESHOLD, db, _NOW
        )
        assert action == "auto"

    def test_queue_threshold_boundary_is_queue(self):
        """A score exactly at the queue threshold queues (not dropped)."""
        db = FakeDB()
        action = kr.apply_resolution(
            ("Alex", "Alex Rivera"), "person", config.KG_MERGE_QUEUE_THRESHOLD, db, _NOW
        )
        assert action == "queue"


# ── Canonical doc shape ──────────────────────────────────────


class TestCanonicalDocShape:
    def _make_doc(self):
        db = FakeDB()
        kr.apply_resolution(
            ("Sarah Chen", "sarah.chen@example.com"), "person", 0.95, db, _NOW
        )
        return next(iter(db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs.values()))

    def test_has_required_fields(self):
        doc = self._make_doc()
        for field in (
            "canonical_id",
            "kind",
            "display_name",
            "aliases",
            "alias_tokens",
            "confidence",
            "created_at",
            "source",
        ):
            assert field in doc, f"canonical doc missing {field}"

    def test_kind_and_source(self):
        doc = self._make_doc()
        assert doc["kind"] == "person"
        assert doc["source"] == "auto"

    def test_aliases_sorted_unique(self):
        doc = self._make_doc()
        assert doc["aliases"] == sorted(set(doc["aliases"]))
        assert "Sarah Chen" in doc["aliases"]
        assert "sarah.chen@example.com" in doc["aliases"]

    def test_alias_tokens_exclude_email_domain(self):
        """alias_tokens use person tokenisation → email domain stripped."""
        doc = self._make_doc()
        assert doc["alias_tokens"] == ["chen", "sarah"]

    def test_created_at_is_now_iso(self):
        doc = self._make_doc()
        assert doc["created_at"] == _NOW.isoformat()

    def test_canonical_id_deterministic(self):
        """Same alias set → identical canonical_id across calls."""
        id1 = kr._canonical_id(["Sarah Chen", "sarah.chen@example.com"])
        id2 = kr._canonical_id(["sarah.chen@example.com", "Sarah Chen"])
        assert id1 == id2
        assert len(id1) == 16
        assert all(c in "0123456789abcdef" for c in id1)


# ── Display name selection ───────────────────────────────────


class TestDisplayNameSelection:
    def test_prefers_multi_token_non_email(self):
        """Longest multi-token non-email alias wins (Sarah Chen over email)."""
        name = kr._select_display_name(["sarah.chen@x.com", "Sarah Chen"], "person")
        assert name == "Sarah Chen"

    def test_prefers_longer_full_name(self):
        name = kr._select_display_name(["Alex", "Alex Rivera"], "person")
        assert name == "Alex Rivera"

    def test_falls_back_to_email_when_only_email(self):
        name = kr._select_display_name(["sarah.chen@x.com"], "person")
        assert name == "sarah.chen@x.com"

    def test_project_display_name(self):
        name = kr._select_display_name(["ads team", "Ads Team"], "project")
        assert name in ("Ads Team", "ads team")


# ── run_resolution batch runner ──────────────────────────────


class TestRunResolution:
    def _entities(self):
        return [
            {
                "related_people": ["Sarah Chen", "sarah.chen@example.com"],
                "owner": "Alex Rivera",
                "related_projects": [],
            },
            {
                "related_people": ["Alex"],
                "related_projects": ["Ads Team", "ads team"],
            },
        ]

    def test_summary_counts(self):
        db = FakeDB()
        summary = kr.run_resolution(self._entities(), db, _NOW)
        # Sarah↔email (auto), Alex↔Alex Rivera (queue), Ads Team↔ads team (auto)
        assert summary["candidates"] == 3
        assert summary["auto"] == 2
        assert summary["queued"] == 1
        assert summary["dropped"] == 0

    def test_writes_only_overlay_collections(self):
        db = FakeDB()
        kr.run_resolution(self._entities(), db, _NOW)
        assert config.FIRESTORE_KG_CANONICAL_COLLECTION in db.collections
        assert config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION in db.collections

    def test_never_touches_raw_knowledge_graph(self):
        """OVERLAY INVARIANT: the raw knowledge_graph collection is never
        accessed (read or write) by the resolver."""
        db = FakeDB()
        kr.run_resolution(self._entities(), db, _NOW)
        assert config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION not in db.collections

    def test_idempotent_no_duplicate_docs(self):
        """Deterministic ids → running twice does not duplicate overlay docs."""
        db = FakeDB()
        kr.run_resolution(self._entities(), db, _NOW)
        canonical_after_1 = len(db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs)
        queue_after_1 = len(db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs)

        kr.run_resolution(self._entities(), db, _NOW)
        canonical_after_2 = len(db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs)
        queue_after_2 = len(db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs)

        assert canonical_after_1 == canonical_after_2 == 2
        assert queue_after_1 == queue_after_2 == 1
