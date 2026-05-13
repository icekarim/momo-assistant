import unittest
from unittest.mock import MagicMock, patch

from meeting_prep_accuracy import (
    PrepEvidence,
    build_prep_diagnostics,
    is_generic_meeting_title,
    plan_prep_queries,
    select_prep_evidence,
    format_prep_evidence_context,
)


class TestMeetingPrepAccuracyUtilities(unittest.TestCase):
    def test_generic_titles_are_suppressed(self):
        generic = [
            "last sync part 2",
            "weekly sync",
            "touchbase",
            "catch up",
            "1:1",
            "follow up sync",
        ]
        for title in generic:
            with self.subTest(title=title):
                self.assertTrue(is_generic_meeting_title(title))

    def test_specific_titles_are_not_generic(self):
        specific = [
            "PacSun integration launch review",
            "Saks Global decision makers",
            "Paymenttype escalation readout",
        ]
        for title in specific:
            with self.subTest(title=title):
                self.assertFalse(is_generic_meeting_title(title))

    def test_diagnostics_include_included_and_excluded_reasons(self):
        included = [
            PrepEvidence(
                evidence_id="E1",
                entry={"id": "kg1", "source_title": "last sync part 2", "source_date": "2026-05-13"},
                score=90,
                reasons=["exact source title"],
            )
        ]
        excluded = [
            PrepEvidence(
                evidence_id="X1",
                entry={"id": "kg2", "source_title": "QVC sync", "source_date": "2026-03-24"},
                score=-20,
                reasons=["generic title search"],
            )
        ]

        text = build_prep_diagnostics(
            meeting={"title": "last sync part 2", "attendees": [{"name": "a@example.com"}]},
            included=included,
            excluded=excluded,
            query_labels=["person:a@example.com", "title:last sync part 2"],
        )

        self.assertIn("meeting='last sync part 2'", text)
        self.assertIn("included E1 score=90", text)
        self.assertIn("excluded X1 score=-20", text)
        self.assertIn("generic title search", text)


class TestMeetingPrepRetrievalPlanning(unittest.TestCase):
    def test_large_generic_meeting_skips_title_search_and_limits_people(self):
        meeting = {
            "title": "last sync part 2",
            "description": "Agenda: final handovers for Agnes Jang",
            "organizer": "user@example.com",
            "attendees": [
                {"name": "alex.rivera@example.com"},
                {"name": "pat.miller@example.com"},
                {"name": "jamie.fox@example.com"},
                {"name": "taylor.singh@example.com"},
                {"name": "morgan.reed@example.com"},
                {"name": "riley.chen@example.com"},
                {"name": "daniel.price@example.com"},
                {"name": "casey.morgan@example.com"},
                {"name": "person5@example.com"},
            ],
        }

        plan = plan_prep_queries(meeting)

        self.assertFalse(plan["include_title_semantic_search"])
        self.assertLessEqual(len(plan["people"]), 4)
        self.assertIn("alex.rivera@example.com", plan["people"])
        self.assertIn("user@example.com", plan["people"])

    def test_specific_meeting_allows_title_search(self):
        meeting = {
            "title": "PacSun integration launch review",
            "description": "",
            "organizer": "user@example.com",
            "attendees": [{"name": "jamie.fox@example.com"}],
        }

        plan = plan_prep_queries(meeting)

        self.assertTrue(plan["include_title_semantic_search"])
        self.assertIn("jamie.fox@example.com", plan["people"])


class TestMeetingPrepEvidenceScoring(unittest.TestCase):
    def test_selects_evidence_ids_stably_independent_of_input_order(self):
        meeting = {
            "title": "PacSun launch",
            "description": "",
            "attendees": [{"name": "alex.rivera@example.com"}],
        }
        first = {
            "id": "a-entry",
            "entity_type": "update",
            "name": "Alpha update",
            "content": "PacSun launch alpha detail.",
            "source_title": "PacSun launch",
            "source_date": "2026-05-13",
        }
        second = {
            "id": "b-entry",
            "entity_type": "update",
            "name": "Beta update",
            "content": "PacSun launch beta detail.",
            "source_title": "PacSun launch",
            "source_date": "2026-05-13",
        }

        included_forward, _ = select_prep_evidence(meeting, [first, second], max_items=5)
        included_reverse, _ = select_prep_evidence(meeting, [second, first], max_items=5)

        forward_ids = {item.entry["id"]: item.evidence_id for item in included_forward}
        reverse_ids = {item.entry["id"]: item.evidence_id for item in included_reverse}
        self.assertEqual(forward_ids, reverse_ids)
        self.assertEqual(forward_ids, {"a-entry": "E1", "b-entry": "E2"})

    def test_scores_all_query_labels_for_duplicate_entries(self):
        meeting = {
            "title": "weekly sync",
            "description": "Agenda: final handovers for Agnes Jang",
            "attendees": [{"name": "alex.rivera@example.com"}],
        }
        entries = [
            {
                "id": "merged",
                "entity_type": "update",
                "name": "Agnes handover",
                "content": "Agnes Jang has handover items to finalize.",
                "source_title": "weekly sync",
                "source_date": "2026-05-13",
                "mentioned_people": ["Agnes Jang"],
                "_query_labels": ["person:alex.rivera@example.com", "title:weekly sync"],
            }
        ]

        included, _ = select_prep_evidence(meeting, entries, max_items=5)

        self.assertEqual([item.entry["id"] for item in included], ["merged"])
        self.assertIn("person query match", included[0].reasons)
        self.assertIn("generic title search", included[0].reasons)

    def test_selects_exact_source_and_rejects_generic_sync_noise(self):
        meeting = {
            "title": "last sync part 2",
            "description": "Agenda: final handovers for Agnes Jang",
            "attendees": [{"name": "alex.rivera@example.com"}],
        }
        entries = [
            {
                "id": "good",
                "entity_type": "update",
                "name": "Agnes handover",
                "content": "Agnes Jang has handover items to finalize.",
                "source_title": "last sync part 2",
                "source_date": "2026-05-13",
                "mentioned_people": ["Agnes Jang"],
                "related_people": ["Agnes Jang"],
                "related_projects": [],
            },
            {
                "id": "bad",
                "entity_type": "topic",
                "name": "QVC sync review",
                "content": "QVC follow-up sync needs rescheduling.",
                "source_title": "Re: QVC x Rokt follow up",
                "source_date": "2026-03-24",
                "mentioned_people": ["Stephanie"],
                "related_people": ["Stephanie"],
                "related_projects": ["QVC"],
                "_query_label": "title:last sync part 2",
            },
        ]

        included, excluded = select_prep_evidence(meeting, entries, max_items=5)

        self.assertEqual([item.entry["id"] for item in included], ["good"])
        self.assertEqual([item.entry["id"] for item in excluded], ["bad"])
        self.assertIn("exact source title", included[0].reasons)
        self.assertIn("generic title search", excluded[0].reasons)

    def test_formats_evidence_context_with_ids(self):
        evidence = [
            PrepEvidence(
                evidence_id="E1",
                entry={
                    "source_date": "2026-05-13",
                    "entity_type": "commitment",
                    "source_title": "last sync part 2",
                    "name": "Agnes handover",
                    "content": "Agnes Jang needs to finalize mapping handover.",
                },
                score=90,
                reasons=["exact source title"],
            )
        ]

        context = format_prep_evidence_context(evidence)

        self.assertIn("[E1]", context)
        self.assertIn("date=2026-05-13", context)
        self.assertIn("source=last sync part 2", context)
        self.assertIn("Agnes handover", context)

    def test_empty_retrieval_uses_evidence_formatter_context(self):
        import proactive_intelligence

        captured_prompts = []

        def fake_generate(_model, prompt, model_name):
            captured_prompts.append(prompt)
            return MagicMock(text="brief")

        meeting = {
            "title": "weekly sync",
            "attendees": [{"name": "alex.rivera@example.com"}],
            "start_time": "2026-05-13T10:00:00Z",
        }

        with (
            patch.object(proactive_intelligence, "query_by_person", return_value=[]),
            patch.object(proactive_intelligence.genai, "GenerativeModel", return_value=object()),
            patch.object(proactive_intelligence, "traced_generate_content", side_effect=fake_generate),
        ):
            brief = proactive_intelligence._build_meeting_prep(meeting)

        self.assertEqual(brief, "brief")
        self.assertTrue(captured_prompts)
        self.assertIn(
            "(No strong prior context found for this meeting.)",
            captured_prompts[0],
        )
        self.assertNotIn(
            "(No prior context found for these attendees or topics.)",
            captured_prompts[0],
        )

    def test_build_meeting_prep_merges_duplicate_query_labels(self):
        import knowledge_graph
        import meeting_prep_accuracy
        import proactive_intelligence

        captured_entries = []
        duplicate_entry = {
            "id": "dup",
            "entity_type": "update",
            "name": "PacSun mapping",
            "content": "PacSun launch mapping needs review.",
            "source_title": "PacSun launch",
            "source_date": "2026-05-13",
            "related_projects": [],
        }
        original_select = meeting_prep_accuracy.select_prep_evidence

        def capture_select(meeting, entries, *args, **kwargs):
            captured_entries.extend(entries)
            return original_select(meeting, entries, *args, **kwargs)

        meeting = {
            "title": "PacSun launch",
            "description": "",
            "attendees": [{"name": "alex.rivera@example.com"}],
            "start_time": "2026-05-13T10:00:00Z",
        }

        with (
            patch.object(proactive_intelligence, "query_by_person", return_value=[duplicate_entry]),
            patch.object(knowledge_graph, "semantic_search", return_value=[duplicate_entry]),
            patch.object(meeting_prep_accuracy, "select_prep_evidence", side_effect=capture_select),
            patch.object(proactive_intelligence.genai, "GenerativeModel", return_value=object()),
            patch.object(
                proactive_intelligence,
                "traced_generate_content",
                return_value=MagicMock(text="brief"),
            ),
        ):
            proactive_intelligence._build_meeting_prep(meeting)

        self.assertEqual(len(captured_entries), 1)
        self.assertEqual(
            captured_entries[0]["_query_labels"],
            ["person:alex.rivera@example.com", "title:PacSun launch"],
        )
