"""Tests for user_memory fuzzy matching logic."""

import pytest
from unittest.mock import MagicMock, patch

import user_memory


def test_fuzzy_match_with_claude():
    """Test that Claude is called for fuzzy matching and returns correct memory."""
    memories = [
        {"id": "1", "content": "I like coffee", "memory_type": "preference", "created_at": "2024-01-01"},
        {"id": "2", "content": "Meeting with Sarah on Fridays", "memory_type": "fact", "created_at": "2024-01-02"},
        {"id": "3", "content": "Deadline is next Monday", "memory_type": "preference", "created_at": "2024-01-03"},
    ]
    
    # Mock the Claude message with text "2"
    fake_message = MagicMock()
    fake_message.content = [MagicMock(type="text", text="2")]
    
    with patch("user_memory.generate", return_value=fake_message) as mock_generate:
        result = user_memory._find_best_match(memories, "Sarah meeting")
        
        # Assert generate was called with LIGHT tier
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["tier"].value == "light"
        
        # Assert the correct memory was returned (index 1, which is memory 2)
        assert result == memories[1]
        assert result["content"] == "Meeting with Sarah on Fridays"


def test_substring_match_skips_claude():
    """Test that substring match returns early without calling Claude."""
    memories = [
        {"id": "1", "content": "I like coffee", "memory_type": "preference", "created_at": "2024-01-01"},
        {"id": "2", "content": "Meeting with Sarah on Fridays", "memory_type": "fact", "created_at": "2024-01-02"},
        {"id": "3", "content": "Deadline is next Monday", "memory_type": "preference", "created_at": "2024-01-03"},
    ]
    
    with patch("user_memory.generate") as mock_generate:
        # Hint contains substring of memory 2
        result = user_memory._find_best_match(memories, "Sarah")
        
        # Assert generate was NOT called
        mock_generate.assert_not_called()
        
        # Assert the correct memory was returned via substring match
        assert result == memories[1]
        assert result["content"] == "Meeting with Sarah on Fridays"


def test_single_memory_returns_directly():
    """Test that single memory is returned without any matching logic."""
    memories = [
        {"id": "1", "content": "I like coffee", "memory_type": "preference", "created_at": "2024-01-01"},
    ]
    
    with patch("user_memory.generate") as mock_generate:
        result = user_memory._find_best_match(memories, "anything")
        
        # Assert generate was NOT called
        mock_generate.assert_not_called()
        
        # Assert the single memory was returned
        assert result == memories[0]


def test_claude_exception_returns_none():
    """Test that Claude exceptions are caught and None is returned."""
    memories = [
        {"id": "1", "content": "I like coffee", "memory_type": "preference", "created_at": "2024-01-01"},
        {"id": "2", "content": "Meeting with Sarah on Fridays", "memory_type": "fact", "created_at": "2024-01-02"},
    ]
    
    with patch("user_memory.generate", side_effect=Exception("API error")):
        with patch("builtins.print") as mock_print:
            result = user_memory._find_best_match(memories, "unknown hint")
            
            # Assert None was returned
            assert result is None
            
            # Assert error was logged
            mock_print.assert_called_once()
            assert "Claude fuzzy match failed" in mock_print.call_args[0][0]


def test_claude_invalid_response_returns_none():
    """Test that invalid Claude responses (no digits) return None."""
    memories = [
        {"id": "1", "content": "I like coffee", "memory_type": "preference", "created_at": "2024-01-01"},
        {"id": "2", "content": "Meeting with Sarah on Fridays", "memory_type": "fact", "created_at": "2024-01-02"},
    ]
    
    # Mock Claude response with no digits
    fake_message = MagicMock()
    fake_message.content = [MagicMock(type="text", text="I'm not sure")]
    
    with patch("user_memory.generate", return_value=fake_message):
        result = user_memory._find_best_match(memories, "unknown hint")
        
        # Assert None was returned (no digits to extract)
        assert result is None


def test_claude_out_of_range_index_returns_none():
    """Test that out-of-range indices return None."""
    memories = [
        {"id": "1", "content": "I like coffee", "memory_type": "preference", "created_at": "2024-01-01"},
        {"id": "2", "content": "Meeting with Sarah on Fridays", "memory_type": "fact", "created_at": "2024-01-02"},
    ]
    
    # Mock Claude response with index 5 (out of range)
    fake_message = MagicMock()
    fake_message.content = [MagicMock(type="text", text="5")]
    
    with patch("user_memory.generate", return_value=fake_message):
        result = user_memory._find_best_match(memories, "unknown hint")
        
        # Assert None was returned (index out of range)
        assert result is None
