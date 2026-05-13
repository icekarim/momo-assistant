import unittest

from meeting_prep_accuracy import (
    PrepEvidence,
    build_prep_diagnostics,
    is_generic_meeting_title,
    plan_prep_queries,
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
