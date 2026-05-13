import unittest
from unittest.mock import MagicMock, patch

from meeting_prep_accuracy import (
    PrepEvidence,
    build_prep_diagnostics,
    finalize_evidence_gated_prep,
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

    def test_finalize_evidence_gated_prep_keeps_cited_bullets_and_drops_uncited(self):
        evidence = [
            PrepEvidence(
                evidence_id="E1",
                entry={"source_title": "last sync part 2", "source_date": "2026-05-13"},
                score=90,
                reasons=["exact source title"],
            )
        ]
        raw_text = "\n".join(
            [
                "📋 *meeting prep — last sync part 2*",
                "- *Agnes* needs to finalize the handover mapping. [E1]",
                "- PacSun launch has an open blocker.",
            ]
        )

        brief = finalize_evidence_gated_prep("last sync part 2", raw_text, evidence)

        self.assertIn("📋 *meeting prep — last sync part 2*", brief)
        self.assertIn("*Agnes* needs to finalize the handover mapping.", brief)
        self.assertIn("_(source: last sync part 2, 2026-05-13)_", brief)
        self.assertNotIn("[E1]", brief)
        self.assertNotIn("PacSun", brief)

    def test_finalize_evidence_gated_prep_uses_low_context_message_without_evidence(self):
        brief = finalize_evidence_gated_prep(
            "last sync part 2",
            "- PacSun launch has an open blocker.",
            [],
        )

        self.assertIn("📋 *meeting prep — last sync part 2*", brief)
        self.assertIn("I don't have strong prep context for this one yet.", brief)
        self.assertNotIn("PacSun", brief)

    def test_finalize_evidence_gated_prep_normalizes_retained_lines_to_bullets(self):
        evidence = [
            PrepEvidence(
                evidence_id="E1",
                entry={"source_title": "last sync part 2", "source_date": "2026-05-13"},
                score=90,
                reasons=["exact source title"],
            )
        ]
        raw_text = "\n".join(
            [
                "### Agnes handover [E1]",
                "1. Mapping checklist needs review. [E1]",
                "- Already a bullet stays a single bullet. [E1]",
            ]
        )

        brief = finalize_evidence_gated_prep("last sync part 2", raw_text, evidence)
        lines = brief.splitlines()

        self.assertEqual(lines[1], "- Agnes handover _(source: last sync part 2, 2026-05-13)_")
        self.assertEqual(lines[2], "- Mapping checklist needs review. _(source: last sync part 2, 2026-05-13)_")
        self.assertEqual(lines[3], "- Already a bullet stays a single bullet. _(source: last sync part 2, 2026-05-13)_")

    def test_finalize_evidence_gated_prep_dedupes_duplicate_sources(self):
        evidence = [
            PrepEvidence(
                evidence_id="E1",
                entry={"source_title": "last sync part 2", "source_date": "2026-05-13"},
                score=90,
                reasons=["exact source title"],
            ),
            PrepEvidence(
                evidence_id="E2",
                entry={"source_title": "last sync part 2", "source_date": "2026-05-13"},
                score=80,
                reasons=["person query match"],
            ),
        ]

        brief = finalize_evidence_gated_prep(
            "last sync part 2",
            "- Agnes handover has two supporting entries. [E1] [E2] [E1]",
            evidence,
        )

        self.assertEqual(
            brief.splitlines()[1],
            "- Agnes handover has two supporting entries. _(source: last sync part 2, 2026-05-13)_",
        )

    def test_finalize_evidence_gated_prep_caps_output_at_six_bullets(self):
        evidence = [
            PrepEvidence(
                evidence_id="E1",
                entry={"source_title": "last sync part 2", "source_date": "2026-05-13"},
                score=90,
                reasons=["exact source title"],
            )
        ]
        raw_text = "\n".join(f"- Item {idx} [E1]" for idx in range(1, 9))

        brief = finalize_evidence_gated_prep("last sync part 2", raw_text, evidence)
        bullet_lines = [line for line in brief.splitlines() if line.startswith("- ")]

        self.assertEqual(len(bullet_lines), 6)
        self.assertIn("Item 6", bullet_lines[-1])
        self.assertNotIn("Item 7", brief)

    def test_finalize_evidence_gated_prep_keeps_normal_source_phrases_before_citation(self):
        evidence = [
            PrepEvidence(
                evidence_id="E1",
                entry={"source_title": "last sync part 2", "source_date": "2026-05-13"},
                score=90,
                reasons=["exact source title"],
            )
        ]
        raw_text = "\n".join(
            [
                "- Ask Agnes to share source: deployment notes before Friday. [E1]",
                "- Strip generated suffixes only. [E1] _(source: hallucinated)_",
            ]
        )

        brief = finalize_evidence_gated_prep("last sync part 2", raw_text, evidence)

        self.assertIn(
            "- Ask Agnes to share source: deployment notes before Friday. "
            "_(source: last sync part 2, 2026-05-13)_",
            brief,
        )
        self.assertIn(
            "- Strip generated suffixes only. _(source: last sync part 2, 2026-05-13)_",
            brief,
        )
        self.assertNotIn("hallucinated", brief)


class TestMeetingPrepRetrievalPlanning(unittest.TestCase):
    def test_large_generic_meeting_skips_title_search_and_limits_people(self):
        meeting = {
            "title": "last sync part 2",
            "description": "Agenda: final handovers for Agnes Jang",
            "organizer": "karim@rokt.com",
            "attendees": [
                {"name": "agnes.jang@rokt.com"},
                {"name": "parth.merchant@rokt.com"},
                {"name": "jessica.francis@rokt.com"},
                {"name": "patrick.tsui@rokt.com"},
                {"name": "matthew.monjarrez@rokt.com"},
                {"name": "raghav.chawla@rokt.com"},
                {"name": "daniel.piet@rokt.com"},
                {"name": "caroline.kar-lai.tan@rokt.com"},
                {"name": "ellie.park@rokt.com"},
            ],
        }

        plan = plan_prep_queries(meeting)

        self.assertFalse(plan["include_title_semantic_search"])
        self.assertLessEqual(len(plan["people"]), 4)
        self.assertIn("agnes.jang@rokt.com", plan["people"])
        self.assertIn("karim@rokt.com", plan["people"])

    def test_specific_meeting_allows_title_search(self):
        meeting = {
            "title": "PacSun integration launch review",
            "description": "",
            "organizer": "karim@rokt.com",
            "attendees": [{"name": "jessica.francis@rokt.com"}],
        }

        plan = plan_prep_queries(meeting)

        self.assertTrue(plan["include_title_semantic_search"])
        self.assertIn("jessica.francis@rokt.com", plan["people"])


class TestMeetingPrepEvidenceScoring(unittest.TestCase):
    def test_selects_evidence_ids_stably_independent_of_input_order(self):
        meeting = {
            "title": "PacSun launch",
            "description": "",
            "attendees": [{"name": "agnes.jang@rokt.com"}],
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
            "attendees": [{"name": "agnes.jang@rokt.com"}],
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
                "_query_labels": ["person:agnes.jang@rokt.com", "title:weekly sync"],
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
            "attendees": [{"name": "agnes.jang@rokt.com"}],
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
            "attendees": [{"name": "agnes.jang@rokt.com"}],
            "start_time": "2026-05-13T10:00:00Z",
        }

        with (
            patch.object(proactive_intelligence, "query_by_person", return_value=[]),
            patch.object(proactive_intelligence.genai, "GenerativeModel", return_value=object()),
            patch.object(proactive_intelligence, "traced_generate_content", side_effect=fake_generate),
        ):
            brief = proactive_intelligence._build_meeting_prep(meeting)

        self.assertIn("📋 *meeting prep — weekly sync*", brief)
        self.assertIn("I don't have strong prep context for this one yet.", brief)
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
            "attendees": [{"name": "agnes.jang@rokt.com"}],
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
            ["person:agnes.jang@rokt.com", "title:PacSun launch"],
        )

    def test_build_meeting_prep_drops_uncited_generated_bullets(self):
        import proactive_intelligence

        evidence_entry = {
            "id": "evidence-1",
            "entity_type": "update",
            "name": "Agnes handover",
            "content": "Agnes Jang needs to finalize mapping handover.",
            "source_title": "last sync part 2",
            "source_date": "2026-05-13",
            "mentioned_people": ["Agnes Jang"],
            "related_people": ["Agnes Jang"],
        }
        meeting = {
            "title": "last sync part 2",
            "description": "Agenda: final handovers for Agnes Jang",
            "attendees": [{"name": "agnes.jang@rokt.com"}],
            "start_time": "2026-05-13T10:00:00Z",
        }
        raw_brief = "\n".join(
            [
                "📋 *meeting prep — last sync part 2*",
                "- *Agnes* needs to finalize mapping handover. [E1]",
                "- PacSun launch has an unrelated open blocker.",
            ]
        )

        with (
            patch.object(proactive_intelligence, "query_by_person", return_value=[evidence_entry]),
            patch.object(proactive_intelligence.genai, "GenerativeModel", return_value=object()),
            patch.object(
                proactive_intelligence,
                "traced_generate_content",
                return_value=MagicMock(text=raw_brief),
            ),
        ):
            brief = proactive_intelligence._build_meeting_prep(meeting)

        self.assertIn("*Agnes* needs to finalize mapping handover.", brief)
        self.assertIn("_(source: last sync part 2, 2026-05-13)_", brief)
        self.assertNotIn("PacSun", brief)


class TestLastSyncRegression(unittest.TestCase):
    def test_last_sync_context_excludes_unrelated_sync_and_social_items(self):
        meeting = {
            "title": "last sync part 2",
            "description": "Agenda: final handovers for Agnes Jang",
            "attendees": [{"name": "agnes.jang@rokt.com"}, {"name": "matthew.monjarrez@rokt.com"}],
        }
        entries = [
            {
                "id": "handover",
                "entity_type": "commitment",
                "name": "Agnes handover",
                "content": "Agnes Jang needs to finalize data mapping table handover.",
                "source_title": "last sync part 2",
                "source_date": "2026-05-13",
                "mentioned_people": ["Agnes Jang"],
                "related_projects": ["Carbon"],
            },
            {
                "id": "pacsun",
                "entity_type": "blocker",
                "name": "PacSun latency",
                "content": "Jessica Francis is dealing with 2-second latency spikes.",
                "source_title": "Internal Sync: Aftersell 1P for PacSun",
                "source_date": "2026-04-21",
                "mentioned_people": ["Jessica Francis"],
                "related_projects": ["PacSun"],
                "_query_label": "title:last sync part 2",
            },
            {
                "id": "tekken",
                "entity_type": "topic",
                "name": "Tekken 3 Exhibition Matches",
                "content": "Matthew Monjarrez and Daniel Piet played Tekken 3.",
                "source_title": "Tekken 3 Exhibition Matches",
                "source_date": "2026-05-11",
                "mentioned_people": ["Matthew Monjarrez", "Daniel Piet"],
                "related_projects": ["Tekken 3"],
                "_query_label": "person:matthew.monjarrez@rokt.com",
            },
        ]

        included, excluded = select_prep_evidence(meeting, entries, max_items=10)
        context = format_prep_evidence_context(included)

        self.assertIn("Agnes handover", context)
        self.assertNotIn("PacSun latency", context)
        self.assertNotIn("Tekken", context)
        self.assertEqual({item.entry["id"] for item in excluded}, {"pacsun", "tekken"})
