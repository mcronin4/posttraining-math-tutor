"""
Data schemas for training examples.

This module defines the JSONL format used for training data.
Each line in a JSONL file should be a valid TrainingExample.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    """Problem difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TutoringMode(str, Enum):
    """Tutoring modes for training examples."""

    HINT = "hint"
    CHECK_STEP = "check_step"
    EXPLAIN = "explain"


class GradeLevel(str, Enum):
    """Ontario K-12 grade levels."""

    K = "K"
    G1 = "1"
    G2 = "2"
    G3 = "3"
    G4 = "4"
    G5 = "5"
    G6 = "6"
    G7 = "7"
    G8 = "8"
    G9 = "9"
    G10 = "10"
    G11 = "11"
    G12 = "12"


class ExampleMetadata(BaseModel):
    """Metadata for a training example."""

    source: str = Field(..., description="Source of the example (e.g., 'synthetic', 'textbook')")
    difficulty: Optional[Difficulty] = Field(None, description="Problem difficulty")
    mode: Optional[TutoringMode] = Field(None, description="Intended tutoring mode")
    curriculum_topic_id: Optional[str] = Field(
        None, description="ID from Ontario math taxonomy"
    )
    annotator: Optional[str] = Field(None, description="Who created/reviewed this example")
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Quality score from review"
    )


class TrainingExample(BaseModel):
    """
    A single training example in JSONL format.

    JSONL Format:
        Each line should be a JSON object matching this schema.
        Example:
        {"id": "ex_001", "grade": "6", "topic_tags": ["fractions", "addition"], ...}

    Fields:
        id: Unique identifier for the example
        grade: Target grade level
        topic_tags: List of curriculum topic tags
        prompt: The input to the model (question + context)
        response: The expected model output
        metadata: Additional metadata about the example
    """

    id: str = Field(..., description="Unique identifier")
    grade: GradeLevel = Field(..., description="Target grade level")
    topic_tags: list[str] = Field(default_factory=list, description="Curriculum topic tags")
    prompt: str = Field(..., description="Input prompt (question + context)")
    response: str = Field(..., description="Expected response")
    metadata: ExampleMetadata = Field(..., description="Example metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "ex_001",
                    "grade": "6",
                    "topic_tags": ["fractions", "addition"],
                    "prompt": "Student: How do I add 1/2 + 1/4?\nMode: hint",
                    "response": "Great question! When adding fractions, we need a common denominator. What do you notice about 2 and 4? Is one a multiple of the other?",
                    "metadata": {
                        "source": "synthetic",
                        "difficulty": "easy",
                        "mode": "hint",
                    },
                }
            ]
        }
    }


class MathProblem(BaseModel):
    """
    A math problem for evaluation (question + expected answer).

    Used in eval_math.py for accuracy testing.
    """

    id: str
    question: str
    answer: str  # Expected answer (can be numeric or expression)
    grade: Optional[GradeLevel] = None
    topic_tags: Optional[list[str]] = None


class TutoringPrompt(BaseModel):
    """
    A tutoring prompt for rubric evaluation.

    Used in eval_tutor.py for tutoring quality testing.
    """

    id: str
    question: str
    attempt: Optional[str] = None
    mode: TutoringMode
    grade: GradeLevel
    expected_behavior: Optional[dict] = None  # Expected rubric scores


class SafetyPrompt(BaseModel):
    """
    A safety test prompt.

    Used in eval_safety.py for refusal testing.
    """

    id: str
    prompt: str
    should_refuse: bool  # Whether the model should refuse this prompt
    category: str  # e.g., "off_topic", "inappropriate", "academic_integrity"

