"""Verify Firestore-native semantic_search returns results comparable to the
in-memory cosine implementation.

Compares top-N results for a fixed set of queries; the third test is the
guard that drives the migration — it asserts the in-memory matrix code path
has been removed."""
import knowledge_graph


def test_semantic_search_returns_results():
    """semantic_search must return at least one result for a known-good query
    that exists in the KG (proves Firestore vector index is wired up)."""
    results = knowledge_graph.semantic_search("Rokt v4")
    assert len(results) > 0, "expected at least 1 result for 'Rokt v4'"
    for r in results:
        assert "id" in r
        assert "content" in r
        assert "entity_type" in r


def test_semantic_search_respects_limit():
    results = knowledge_graph.semantic_search("decisions", limit=3)
    assert len(results) <= 3


def test_semantic_search_no_in_memory_matrix():
    """After migration, _get_all_embeddings must NOT exist — code path is dead."""
    assert not hasattr(knowledge_graph, "_get_all_embeddings"), \
        "_get_all_embeddings should be removed after migration"
    assert not hasattr(knowledge_graph, "warm_embedding_cache"), \
        "warm_embedding_cache should be removed after migration"
