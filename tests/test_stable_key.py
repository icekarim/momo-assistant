"""Tests for stable_key helper function in knowledge_graph.py.

Tests verify that stable_key generates deterministic, normalized keys for entity dedup
in the nudge system (KG v2 Phase 0.5).
"""

import pytest
from knowledge_graph import stable_key


class TestStableKeyDeterministic:
    """Test that stable_key produces identical output for identical inputs."""

    def test_deterministic_same_dict_twice(self):
        """Same entity dict twice should produce identical keys."""
        entity = {"source_id": "s1", "name": "Sarah Chen"}
        key1 = stable_key(entity)
        key2 = stable_key(entity)
        assert key1 == key2, "Same entity should produce same key"

    def test_deterministic_dict_order_irrelevant(self):
        """Dict key order should not affect the stable key."""
        entity1 = {"source_id": "s1", "name": "Sarah Chen"}
        entity2 = {"name": "Sarah Chen", "source_id": "s1"}
        assert stable_key(entity1) == stable_key(entity2)


class TestStableKeyNormalization:
    """Test that stable_key normalizes name and source_id correctly."""

    def test_normalizes_name_whitespace(self):
        """Extra whitespace in name should be collapsed."""
        entity1 = {"source_id": "s1", "name": "Sarah  Chen"}
        entity2 = {"source_id": "s1", "name": "Sarah Chen"}
        assert stable_key(entity1) == stable_key(entity2)

    def test_normalizes_name_case_insensitive(self):
        """Name should be case-insensitive."""
        entity1 = {"source_id": "s1", "name": "Sarah Chen"}
        entity2 = {"source_id": "s1", "name": "sarah chen"}
        assert stable_key(entity1) == stable_key(entity2)

    def test_normalizes_name_mixed_case_and_whitespace(self):
        """Name normalization should handle both case and whitespace."""
        entity1 = {"source_id": "s1", "name": "Sarah  Chen"}
        entity2 = {"source_id": "s1", "name": "SARAH CHEN"}
        assert stable_key(entity1) == stable_key(entity2)


class TestStableKeyDistinctSources:
    """Test that different source_ids produce different keys."""

    def test_distinct_sources_same_name(self):
        """Same name, different source_id should produce different keys."""
        entity1 = {"source_id": "s1", "name": "Sarah Chen"}
        entity2 = {"source_id": "s2", "name": "Sarah Chen"}
        assert stable_key(entity1) != stable_key(entity2)


class TestStableKeyFormat:
    """Test that stable_key produces correctly formatted output."""

    def test_length_16_hex_chars(self):
        """Key should be exactly 16 lowercase hex characters."""
        entity = {"source_id": "s1", "name": "Sarah Chen"}
        key = stable_key(entity)
        assert len(key) == 16, f"Key should be 16 chars, got {len(key)}"
        assert all(c in "0123456789abcdef" for c in key), f"Key should be hex, got {key}"

    def test_lowercase_hex(self):
        """Key should use lowercase hex digits."""
        entity = {"source_id": "s1", "name": "Sarah Chen"}
        key = stable_key(entity)
        assert key == key.lower(), "Key should be lowercase"


class TestStableKeyMissingFields:
    """Test that stable_key handles missing or None fields gracefully."""

    def test_empty_dict(self):
        """Empty dict should not raise and should return a 16-char key."""
        key = stable_key({})
        assert isinstance(key, str)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_none_source_id(self):
        """None source_id should not raise."""
        entity = {"source_id": None, "name": "Sarah Chen"}
        key = stable_key(entity)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_none_name(self):
        """None name should not raise."""
        entity = {"source_id": "s1", "name": None}
        key = stable_key(entity)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_both_none(self):
        """Both None should not raise."""
        entity = {"source_id": None, "name": None}
        key = stable_key(entity)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_missing_source_id_key(self):
        """Missing source_id key should not raise."""
        entity = {"name": "Sarah Chen"}
        key = stable_key(entity)
        assert isinstance(key, str)
        assert len(key) == 16

    def test_missing_name_key(self):
        """Missing name key should not raise."""
        entity = {"source_id": "s1"}
        key = stable_key(entity)
        assert isinstance(key, str)
        assert len(key) == 16
