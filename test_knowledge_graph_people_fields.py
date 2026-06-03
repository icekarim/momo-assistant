import sys
import unittest
from unittest.mock import MagicMock, patch

_MOCKED_MODULES = (
    "google",
    "google.cloud",
    "google.cloud.firestore",
    "google.cloud.firestore_v1",
    # Submodules referenced via `from ... import ...` in knowledge_graph
    "google.cloud.firestore_v1.base_query",
    "google.cloud.firestore_v1.base_vector_query",
    "google.cloud.firestore_v1.vector",
    "google.generativeai",
)

_saved_modules = {name: sys.modules.get(name) for name in _MOCKED_MODULES}
for name in _MOCKED_MODULES:
    sys.modules[name] = MagicMock()

try:
    import knowledge_graph
finally:
    for name in _MOCKED_MODULES:
        original = _saved_modules[name]
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


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
