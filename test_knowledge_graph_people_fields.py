import unittest
from unittest.mock import MagicMock, patch

import knowledge_graph


class TestKnowledgeGraphPeopleFields(unittest.TestCase):
    def test_prepare_entry_document_separates_attendees_and_mentions(self):
        entry = {
            "entity_type": "commitment",
            "name": "Finalize mapping",
            "content": "Agnes Jang will finalize mapping tables.",
            "owner": "Agnes Jang",
            "mentioned_people": ["Agnes Jang", "Scott"],
            "related_people": ["Agnes Jang", "Scott"],
            "related_projects": ["Carbon"],
            "tags": ["mapping"],
        }

        with patch.object(knowledge_graph, "_get_embedding", return_value=[0.1, 0.2]):
            doc = knowledge_graph._prepare_entry_document(
                entry,
                source_type="meeting",
                source_title="last sync part 2",
                source_date="2026-05-13",
                attendees=["Jessica Francis", "Patrick Tsui"],
            )

        self.assertEqual(doc["mentioned_people"], ["Agnes Jang", "Scott"])
        self.assertEqual(doc["attendees"], ["Jessica Francis", "Patrick Tsui"])
        self.assertIn("agnes", doc["_search_mentioned_people"])
        self.assertIn("jessica", doc["_search_attendees"])

    def test_store_entries_keeps_people_fields_and_source_metadata_on_embedding_auth_failure(self):
        entry = {
            "name": "Finalize mapping",
            "mentioned_people": ["Agnes Jang"],
            "related_people": ["Agnes Jang"],
        }
        db = MagicMock()
        failure = knowledge_graph.ExternalAuthError("gemini", "Expired credentials")
        with (
            patch.object(knowledge_graph, "get_db", return_value=db),
            patch.object(knowledge_graph, "_get_embedding", side_effect=failure),
            patch("builtins.print") as logged,
        ):
            knowledge_graph._store_entries(
                [entry], "meeting", "event-52", "last sync part 2", "2026-05-13",
                attendees=["Jessica Francis"],
            )

        db.collection.return_value.add.assert_called_once()
        doc = db.collection.return_value.add.call_args.args[0]
        self.assertEqual(doc["source_id"], "event-52")
        self.assertEqual(doc["source_date"], "2026-05-13")
        self.assertTrue(doc["extracted_at"])
        self.assertEqual(doc["mentioned_people"], ["Agnes Jang"])
        self.assertEqual(doc["attendees"], ["Jessica Francis"])
        self.assertIn("agnes", doc["_search_mentioned_people"])
        self.assertIn("jessica", doc["_search_attendees"])
        self.assertNotIn("embedding", doc)
        self.assertNotIn("embedding_model", doc)
        self.assertIn("EMBEDDING AUTH FAILURE", logged.call_args.args[0])
        self.assertNotIn("embedding generation failed", logged.call_args.args[0])
