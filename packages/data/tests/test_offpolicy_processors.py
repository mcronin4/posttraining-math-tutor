"""
Tests for off-policy dataset processors.

Tests MathDial and SocraticMATH processing, including edge cases and validation.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
import sys
from pathlib import Path

test_dir = Path(__file__).resolve().parent
data_dir = test_dir.parent
if str(data_dir) not in sys.path:
    sys.path.insert(0, str(data_dir))

from data.offpolicy_types import ConversationMessage, ConversationTrajectory
from data.scripts.process_mathdial_offpolicy import (
    remove_strategy_tags,
    parse_mathdial_conversation,
    process_mathdial_row,
)
from data.scripts.process_socraticmath_offpolicy import (
    extract_metadata,
    parse_socraticmath_conversation,
    process_socraticmath_row,
)
from data.scripts.convert_to_offpolicy_format import (
    validate_trajectory,
    load_trajectory_from_dict,
)


class TestMathDialProcessor:
    """Tests for MathDial processing functions."""
    
    def test_remove_strategy_tags(self):
        """Test strategy tag removal."""
        assert remove_strategy_tags("(probing)Hello") == "Hello"
        assert remove_strategy_tags("(generic)Exactly correct!") == "Exactly correct!"
        assert remove_strategy_tags("(hint)What do you think?") == "What do you think?"
        assert remove_strategy_tags("No tag here") == "No tag here"
        assert remove_strategy_tags("(multiple words)Test") == "Test"
        assert remove_strategy_tags("(probing)  Extra spaces") == "Extra spaces"
    
    def test_parse_mathdial_conversation_valid(self):
        """Test parsing valid MathDial conversation."""
        conv_str = "Teacher: (probing)Hello|EOM|Student: Hi|EOM|Teacher: (generic)Good!"
        messages = parse_mathdial_conversation(conv_str)
        
        assert len(messages) == 3
        assert messages[0].role == "tutor"
        assert messages[0].content == "Hello"  # Strategy tag removed
        assert messages[0].turn == 0
        assert messages[1].role == "student"
        assert messages[1].content == "Hi"
        assert messages[1].turn == 1
        assert messages[2].role == "tutor"
        assert messages[2].content == "Good!"  # Strategy tag removed
    
    def test_parse_mathdial_conversation_empty(self):
        """Test parsing empty conversation."""
        assert parse_mathdial_conversation("") == []
        assert parse_mathdial_conversation("   ") == []
    
    def test_parse_mathdial_conversation_malformed(self):
        """Test parsing malformed conversation."""
        # Missing EOM delimiter
        messages = parse_mathdial_conversation("Teacher: Hello")
        assert len(messages) == 1
        assert messages[0].role == "tutor"
        
        # Unknown role prefix
        messages = parse_mathdial_conversation("Unknown: Message|EOM|")
        assert len(messages) == 0
    
    def test_process_mathdial_row_valid(self):
        """Test processing valid MathDial row."""
        row = {
            "qid": 5000012,
            "question": "What is 2+2?",
            "ground_truth": "4",
            "conversation": "Teacher: (probing)What do you think?|EOM|Student: 4|EOM|Teacher: (generic)Correct!",
            "student_profile": "Test profile",
        }
        traj = process_mathdial_row(row, 0)
        
        assert traj is not None
        assert traj.id == "mathdial_5000012"
        assert traj.question == "What is 2+2?"
        assert traj.expected_answer == "4"
        assert traj.source == "mathdial"
        assert len(traj.messages) == 3
        assert traj.messages[0].content == "What do you think?"  # Strategy removed
    
    def test_process_mathdial_row_missing_fields(self):
        """Test processing row with missing required fields."""
        # Missing question
        row = {"qid": 1, "ground_truth": "4", "conversation": "Teacher: Hi|EOM|"}
        assert process_mathdial_row(row, 0) is None
        
        # Missing ground_truth
        row = {"qid": 1, "question": "What is 2+2?", "conversation": "Teacher: Hi|EOM|"}
        assert process_mathdial_row(row, 0) is None
        
        # Missing conversation
        row = {"qid": 1, "question": "What is 2+2?", "ground_truth": "4"}
        assert process_mathdial_row(row, 0) is None


class TestSocraticMATHProcessor:
    """Tests for SocraticMATH processing functions."""
    
    def test_extract_metadata(self):
        """Test metadata extraction from SocraticMATH array."""
        text_array = [
            "Teacher: Hello",
            "Student: Hi",
            "【ProblemID】:17578",
            "【解析】:Solution: The answer is 9 and 10.",
        ]
        messages, metadata = extract_metadata(text_array)
        
        assert len(messages) == 2
        assert messages[0] == "Teacher: Hello"
        assert messages[1] == "Student: Hi"
        assert metadata["problem_id"] == "17578"
        assert "9 and 10" in metadata["analysis"]
    
    def test_extract_metadata_no_metadata(self):
        """Test extraction when no metadata present."""
        text_array = ["Teacher: Hello", "Student: Hi"]
        messages, metadata = extract_metadata(text_array)
        
        assert len(messages) == 2
        assert metadata == {}
    
    def test_parse_socraticmath_conversation_valid(self):
        """Test parsing valid SocraticMATH conversation."""
        text_array = [
            "Teacher: First message",
            "Student: Second message",
            "Teacher: Third message",
            "Student: Fourth message",
        ]
        messages, question, answer = parse_socraticmath_conversation(text_array)
        
        assert len(messages) == 4
        assert messages[0].role == "tutor"
        assert messages[0].content == "Teacher: First message"
        assert messages[1].role == "student"
        assert messages[1].content == "Student: Second message"
        assert messages[2].role == "tutor"
        assert messages[3].role == "student"
    
    def test_parse_socraticmath_conversation_with_metadata(self):
        """Test parsing conversation with metadata."""
        text_array = [
            "Teacher: Question?",
            "Student: How to solve?",
            "Teacher: Let's think",
            "【ProblemID】:123",
            "【解析】:Answer: 42",
        ]
        messages, question, answer = parse_socraticmath_conversation(text_array)
        
        assert len(messages) == 3  # Metadata removed
        assert messages[0].role == "tutor"
        assert messages[1].role == "student"
        assert messages[2].role == "tutor"
    
    def test_process_socraticmath_row_valid(self):
        """Test processing valid SocraticMATH row."""
        row = {
            "text": [
                "Teacher: First",
                "Student: Second",
                "Teacher: Third",
            ]
        }
        traj = process_socraticmath_row(row, 0)
        
        assert traj is not None
        assert traj.source == "socraticmath"
        assert len(traj.messages) == 3
        assert traj.messages[0].role == "tutor"
        assert traj.messages[1].role == "student"
        assert traj.messages[2].role == "tutor"
    
    def test_process_socraticmath_row_invalid(self):
        """Test processing invalid rows."""
        # Empty text array
        assert process_socraticmath_row({"text": []}, 0) is None
        
        # Single message
        assert process_socraticmath_row({"text": ["Teacher: Only"]}, 0) is None
        
        # Missing text field
        assert process_socraticmath_row({}, 0) is None


class TestValidation:
    """Tests for trajectory validation."""
    
    def test_validate_trajectory_valid(self):
        """Test validation of valid trajectory."""
        traj = ConversationTrajectory(
            id="test_1",
            question="What is 2+2?",
            expected_answer="4",
            messages=[
                ConversationMessage(role="tutor", content="Hello", turn=0),
                ConversationMessage(role="student", content="Hi", turn=1),
                ConversationMessage(role="tutor", content="Good", turn=2),
            ],
            source="mathdial",
        )
        is_valid, error = validate_trajectory(traj)
        assert is_valid
        assert error is None
    
    def test_validate_trajectory_missing_id(self):
        """Test validation fails for missing ID."""
        traj = ConversationTrajectory(
            id="",
            question="Test",
            expected_answer="",
            messages=[ConversationMessage(role="tutor", content="Hi", turn=0)],
        )
        is_valid, error = validate_trajectory(traj)
        assert not is_valid
        assert "id" in error.lower()
    
    def test_validate_trajectory_no_messages(self):
        """Test validation fails for no messages."""
        traj = ConversationTrajectory(
            id="test",
            question="Test",
            expected_answer="",
            messages=[],
        )
        is_valid, error = validate_trajectory(traj)
        assert not is_valid
        assert "messages" in error.lower()
    
    def test_validate_trajectory_single_message(self):
        """Test validation fails for single message."""
        traj = ConversationTrajectory(
            id="test",
            question="Test",
            expected_answer="",
            messages=[ConversationMessage(role="tutor", content="Hi", turn=0)],
        )
        is_valid, error = validate_trajectory(traj)
        assert not is_valid
        assert "at least 2" in error.lower()
    
    def test_validate_trajectory_invalid_role(self):
        """Test validation fails for invalid role."""
        traj = ConversationTrajectory(
            id="test",
            question="Test",
            expected_answer="",
            messages=[
                ConversationMessage(role="invalid", content="Hi", turn=0),
                ConversationMessage(role="student", content="Hello", turn=1),
            ],
        )
        is_valid, error = validate_trajectory(traj)
        assert not is_valid
        assert "role" in error.lower()
    
    def test_load_trajectory_from_dict(self):
        """Test loading trajectory from dictionary."""
        data = {
            "id": "test_1",
            "question": "What is 2+2?",
            "expected_answer": "4",
            "messages": [
                {"role": "tutor", "content": "Hello", "turn": 0},
                {"role": "student", "content": "Hi", "turn": 1},
            ],
            "source": "mathdial",
            "metadata": {},
        }
        traj = load_trajectory_from_dict(data)
        
        assert traj.id == "test_1"
        assert traj.question == "What is 2+2?"
        assert len(traj.messages) == 2
        assert traj.messages[0].role == "tutor"
        assert traj.messages[1].role == "student"


class TestIntegration:
    """Integration tests for full processing pipeline."""
    
    def test_mathdial_jsonl_roundtrip(self):
        """Test MathDial processing and JSONL roundtrip."""
        # Create sample MathDial row
        row = {
            "qid": 5000012,
            "question": "What is 2+2?",
            "ground_truth": "4",
            "conversation": "Teacher: (probing)What do you think?|EOM|Student: 4|EOM|Teacher: (generic)Correct!",
        }
        
        # Process row
        traj = process_mathdial_row(row, 0)
        assert traj is not None
        
        # Convert to dict and back
        traj_dict = {
            "id": traj.id,
            "question": traj.question,
            "expected_answer": traj.expected_answer,
            "messages": [
                {"role": msg.role, "content": msg.content, "turn": msg.turn}
                for msg in traj.messages
            ],
            "source": traj.source,
            "metadata": traj.metadata,
        }
        
        # Load back
        traj_loaded = load_trajectory_from_dict(traj_dict)
        assert traj_loaded.id == traj.id
        assert len(traj_loaded.messages) == len(traj.messages)
        assert traj_loaded.messages[0].content == "What do you think?"  # Strategy removed
    
    def test_socraticmath_jsonl_roundtrip(self):
        """Test SocraticMATH processing and JSONL roundtrip."""
        row = {
            "text": [
                "Teacher: First message",
                "Student: Second message",
                "Teacher: Third message",
            ]
        }
        
        traj = process_socraticmath_row(row, 0)
        assert traj is not None
        
        # Validate
        is_valid, error = validate_trajectory(traj)
        assert is_valid, f"Validation failed: {error}"
