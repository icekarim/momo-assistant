"""Tests for knowledge_linking (KG v2 Phase 2). Hermetic — no Firestore, no network."""
import hashlib
import importlib
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


class TestPhase2Config:
    def test_linking_flag_defaults_false(self, monkeypatch):
        monkeypatch.delenv("KG_LINKING_ENABLED", raising=False)
        import config
        importlib.reload(config)
        assert config.KG_LINKING_ENABLED is False

    def test_min_confidence_default(self, monkeypatch):
        monkeypatch.delenv("KG_LINK_MIN_CONFIDENCE", raising=False)
        import config
        importlib.reload(config)
        assert config.KG_LINK_MIN_CONFIDENCE == 0.85

    def test_links_collection_name(self):
        import config
        importlib.reload(config)
        assert config.FIRESTORE_KG_LINKS_COLLECTION == "kg_links"


# ── Scenario 1: Happy email link ──────────────────────────────

class TestScenario01HappyEmailLink:
    def test_happy_email_link(self):
        from knowledge_linking import run_linking

        commitment = {
            "id": "c1",
            "name": "Send proposal to client",
            "content": "Need to send the project proposal to the client",
            "owner": "Sarah",
        }
        email = {
            "id": "email1",
            "from": "sarah@example.com",
            "subject": "Project Proposal",
            "body": "Please find attached the project proposal",
        }
        judgment = {"match": True, "confidence": 0.92, "excerpt": "project proposal sent"}

        mock_judge = MagicMock(return_value=judgment)
        mock_search_email = MagicMock(return_value=[email])
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        mock_doc_ref = MagicMock()
        db.collection.return_value.document.return_value = mock_doc_ref

        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = run_linking(
            [commitment], db, now=now,
            judge=mock_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        # Exactly one .set() on kg_links
        assert mock_doc_ref.set.call_count == 1
        db.collection.assert_called_with("kg_links")

        # Full 7-field doc shape
        doc = mock_doc_ref.set.call_args[0][0]
        assert "link_id" in doc
        assert doc["link_type"] == "evidence_of"
        assert "commitment_entity_id" in doc
        assert "commitment_stable_key" in doc
        assert "evidence" in doc
        assert "confidence" in doc
        assert "created_at" in doc

        assert summary["linked"] == 1
        assert summary["commitments"] == 1


# ── Scenario 2: Task evidence ─────────────────────────────────

class TestScenario02TaskEvidence:
    def test_task_evidence_doc_shape(self):
        from knowledge_linking import run_linking

        commitment = {
            "id": "c1",
            "name": "Complete design review",
            "content": "Need to complete the design review",
            "owner": "",
        }
        task = {"id": "task_abc123", "title": "Complete design review", "list_name": "Work"}

        mock_judge = MagicMock(return_value={"match": True, "confidence": 0.92, "excerpt": "review completed"})
        mock_search_email = MagicMock(return_value=[])
        mock_find_task = MagicMock(return_value=task)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        mock_doc_ref = MagicMock()
        db.collection.return_value.document.return_value = mock_doc_ref

        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        run_linking(
            [commitment], db, now=now,
            judge=mock_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        assert mock_doc_ref.set.call_count == 1
        doc = mock_doc_ref.set.call_args[0][0]
        assert doc["evidence"]["source_type"] == "task"
        assert doc["evidence"]["source_ref"] == "task_abc123"


# ── Scenario 3: Canonical-aware aliases ───────────────────────

class TestScenario03CanonicalAware:
    def test_aliases_all_in_commitment_desc(self):
        from knowledge_linking import run_linking

        commitment = {
            "id": "c1",
            "name": "Send report to Sarah",
            "content": "Need to send weekly report",
            "owner": "Sarah Chen",
        }
        email = {"id": "email1", "from": "x@x.com", "subject": "report", "body": "weekly report attached"}
        aliases = ["Sarah", "Sarah Chen", "sarah.chen@x.com"]

        captured_descs = []

        def capturing_judge(commitment_desc, evidence_desc):
            captured_descs.append(commitment_desc)
            return {"match": False, "confidence": 0.0, "excerpt": ""}

        mock_aliases = MagicMock(return_value=aliases)
        mock_search_email = MagicMock(return_value=[email])
        mock_find_task = MagicMock(return_value=None)

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        run_linking(
            [commitment], db, now=now,
            judge=capturing_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        assert len(captured_descs) == 1
        commitment_desc = captured_descs[0]
        for alias in aliases:
            assert alias in commitment_desc, f"Expected alias '{alias}' in: {commitment_desc!r}"


# ── Scenario 4: Deterministic ID ─────────────────────────────

class TestScenario04DeterministicId:
    def test_link_id_formula(self):
        from knowledge_linking import _link_id

        expected = hashlib.sha1(b"c1|email|m1").hexdigest()[:16]
        assert _link_id("c1", "email", "m1") == expected

    def test_two_runs_same_document_path(self):
        from knowledge_linking import run_linking

        commitment = {"id": "c1", "name": "Test commitment", "content": "do it", "owner": ""}
        email = {"id": "m1", "from": "x@x.com", "subject": "test", "body": "test body"}

        mock_judge = MagicMock(return_value={"match": True, "confidence": 0.92, "excerpt": ""})
        mock_search_email = MagicMock(return_value=[email])
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db1 = MagicMock()
        db2 = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        run_linking([commitment], db1, now=now, judge=mock_judge,
                    search_email_fn=mock_search_email, find_task_fn=mock_find_task,
                    aliases_fn=mock_aliases)
        run_linking([commitment], db2, now=now, judge=mock_judge,
                    search_email_fn=mock_search_email, find_task_fn=mock_find_task,
                    aliases_fn=mock_aliases)

        path1 = db1.collection.return_value.document.call_args[0][0]
        path2 = db2.collection.return_value.document.call_args[0][0]
        assert path1 == path2


# ── Scenario 5: Low confidence skipped ───────────────────────

class TestScenario05LowConfidence:
    def test_low_confidence_no_write(self):
        from knowledge_linking import run_linking

        commitment = {"id": "c1", "name": "Send report", "content": "monthly report", "owner": ""}
        email = {"id": "email1", "from": "x@x.com", "subject": "report", "body": "here is the report"}

        mock_judge = MagicMock(return_value={"match": True, "confidence": 0.70, "excerpt": ""})
        mock_search_email = MagicMock(return_value=[email])
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = run_linking(
            [commitment], db, now=now,
            judge=mock_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        db.collection.return_value.document.return_value.set.assert_not_called()
        assert summary["skipped_low_confidence"] == 1
        assert summary["linked"] == 0


# ── Scenario 6: Match False ───────────────────────────────────

class TestScenario06MatchFalse:
    def test_match_false_no_write(self):
        from knowledge_linking import run_linking

        commitment = {"id": "c1", "name": "Follow up with team", "content": "follow up on project", "owner": ""}
        email = {"id": "email1", "from": "x@x.com", "subject": "hello", "body": "unrelated email content"}

        mock_judge = MagicMock(return_value={"match": False, "confidence": 0.95, "excerpt": ""})
        mock_search_email = MagicMock(return_value=[email])
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = run_linking(
            [commitment], db, now=now,
            judge=mock_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        db.collection.return_value.document.return_value.set.assert_not_called()
        assert summary["no_match"] >= 1
        assert summary["linked"] == 0


# ── Scenario 7: Parse judgment garbage ───────────────────────

class TestScenario07ParseJudgmentGarbage:
    def test_non_json_returns_none(self):
        from knowledge_linking import parse_judgment
        assert parse_judgment("not json at all") is None

    def test_empty_array_returns_none(self):
        from knowledge_linking import parse_judgment
        assert parse_judgment("[]") is None

    def test_missing_keys_returns_none(self):
        from knowledge_linking import parse_judgment
        # Missing confidence
        assert parse_judgment('[{"match": true}]') is None
        # Missing match
        assert parse_judgment('[{"confidence": 0.9, "excerpt": ""}]') is None

    def test_string_confidence_returns_none(self):
        from knowledge_linking import parse_judgment
        assert parse_judgment('[{"match": true, "confidence": "high", "excerpt": ""}]') is None

    def test_run_linking_none_judgment_no_crash(self):
        from knowledge_linking import run_linking

        commitment = {"id": "c1", "name": "Test garbage", "content": "test content", "owner": ""}
        email = {"id": "email1", "from": "x@x.com", "subject": "test", "body": "test"}

        garbage_judge = MagicMock(return_value=None)
        mock_search_email = MagicMock(return_value=[email])
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = run_linking(
            [commitment], db, now=now,
            judge=garbage_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        # No crash, no write, treated as no-match
        db.collection.return_value.document.return_value.set.assert_not_called()
        assert summary["no_match"] >= 1


# ── Scenario 8: Zero candidates ──────────────────────────────

class TestScenario08ZeroCandidates:
    def test_zero_candidates_judge_never_called(self):
        from knowledge_linking import run_linking

        commitment = {"id": "c1", "name": "Do something", "content": "needs to be done", "owner": ""}

        mock_judge = MagicMock()
        mock_search_email = MagicMock(return_value=[])
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = run_linking(
            [commitment], db, now=now,
            judge=mock_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        mock_judge.assert_not_called()
        assert summary["candidates"] == 0


# ── Scenario 9: Short name uses content ──────────────────────

class TestScenario09ShortName:
    def test_short_name_uses_content_for_search(self):
        from knowledge_linking import run_linking

        commitment = {
            "id": "c1",
            "name": "Act",   # len("Act") == 3, not > 3
            "content": "This is a long content that needs to be done and is very important",
            "owner": "",
        }

        mock_judge = MagicMock(return_value={"match": False, "confidence": 0.0, "excerpt": ""})
        mock_search_email = MagicMock(return_value=[])
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        run_linking(
            [commitment], db, now=now,
            judge=mock_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        expected_terms = commitment["content"][:50]
        mock_search_email.assert_called_once_with(expected_terms, days_back=30, max_results=3)


# ── Scenario 10: search_email_fn raises ──────────────────────

class TestScenario10SearchEmailRaises:
    def test_email_error_increments_errors_next_commitment_processed(self):
        from knowledge_linking import run_linking

        commitments = [
            {"id": "c1", "name": "First commitment", "content": "first thing to do", "owner": ""},
            {"id": "c2", "name": "Second commitment", "content": "second thing to do", "owner": ""},
        ]

        call_count = [0]

        def mock_search_email(terms, *, days_back, max_results):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Network error")
            return []

        mock_judge = MagicMock()
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = run_linking(
            commitments, db, now=now,
            judge=mock_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        assert summary["errors"] == 1
        assert summary["commitments"] == 2
        assert call_count[0] == 2   # search called for both commitments


# ── Scenario 11: Empty commitments ───────────────────────────

class TestScenario11EmptyCommitments:
    def test_empty_commitments_all_zero_no_db(self):
        from knowledge_linking import run_linking

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = run_linking(
            [], db, now=now,
            judge=MagicMock(),
            search_email_fn=MagicMock(return_value=[]),
            find_task_fn=MagicMock(return_value=None),
            aliases_fn=MagicMock(return_value=[]),
        )

        assert summary["commitments"] == 0
        assert summary["candidates"] == 0
        assert summary["linked"] == 0
        assert summary["skipped_low_confidence"] == 0
        assert summary["no_match"] == 0
        assert summary["errors"] == 0
        db.collection.assert_not_called()


# ── Scenario 12: Overlay invariant ───────────────────────────

class TestScenario12OverlayInvariant:
    def test_never_writes_to_knowledge_graph_collection(self):
        from knowledge_linking import run_linking
        import config

        commitment = {"id": "c1", "name": "Test overlay", "content": "test overlay content", "owner": ""}
        email = {"id": "email1", "from": "x@x.com", "subject": "test", "body": "test body"}

        mock_judge = MagicMock(return_value={"match": True, "confidence": 0.92, "excerpt": ""})
        mock_search_email = MagicMock(return_value=[email])
        mock_find_task = MagicMock(return_value=None)
        mock_aliases = MagicMock(return_value=[])

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        run_linking(
            [commitment], db, now=now,
            judge=mock_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        # collection() NEVER called with the raw knowledge_graph collection
        for call_args in db.collection.call_args_list:
            assert call_args[0][0] != config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION, (
                f"Unexpected write to {config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION}"
            )

        # Only kg_links was used
        db.collection.assert_called_with(config.FIRESTORE_KG_LINKS_COLLECTION)


# ── Scenario 13: Regression guard ────────────────────────────

class TestScenario13RegressionGuard:
    def test_check_commitment_evidence_still_importable(self):
        from proactive_intelligence import _check_commitment_evidence
        assert callable(_check_commitment_evidence)


# ── Scenario 14: Aliases default empty ───────────────────────

class TestScenario14AliasesDefaultEmpty:
    def test_empty_aliases_no_crash_no_none_in_prompt(self):
        from knowledge_linking import run_linking

        commitment = {
            "id": "c1",
            "name": "Test commitment with owner",
            "content": "this needs to be done now",
            "owner": "Bob Smith",
        }
        email = {"id": "email1", "from": "bob@example.com", "subject": "done", "body": "finished it"}

        captured_descs = []

        def capturing_judge(commitment_desc, evidence_desc):
            captured_descs.append(commitment_desc)
            return {"match": False, "confidence": 0.0, "excerpt": ""}

        mock_aliases = MagicMock(return_value=[])   # empty aliases
        mock_search_email = MagicMock(return_value=[email])
        mock_find_task = MagicMock(return_value=None)

        db = MagicMock()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Must not crash
        run_linking(
            [commitment], db, now=now,
            judge=capturing_judge,
            search_email_fn=mock_search_email,
            find_task_fn=mock_find_task,
            aliases_fn=mock_aliases,
        )

        assert len(captured_descs) == 1
        commitment_desc = captured_descs[0]
        assert "None" not in commitment_desc
        assert isinstance(commitment_desc, str)
        assert len(commitment_desc) > 0
