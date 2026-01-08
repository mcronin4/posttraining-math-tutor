"""
Pydantic schemas for the Math Tutor API.

These schemas mirror the TypeScript types in @math-tutor/core for consistency.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TutoringMode(str, Enum):
    """Available tutoring modes."""

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


class ChatRequest(BaseModel):
    """Request schema for the /chat endpoint."""

    question: str = Field(..., min_length=1, description="The math question or problem")
    attempt: Optional[str] = Field(None, description="Student's attempt at the problem")
    mode: TutoringMode = Field(..., description="Tutoring mode (hint, check_step, explain)")
    grade: GradeLevel = Field(..., description="Student's grade level")
    dont_reveal_answer: bool = Field(True, description="Whether to avoid revealing the final answer")
    topic_tags: Optional[list[str]] = Field(None, description="Optional topic tags for context")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is 2 + 3?",
                    "mode": "hint",
                    "grade": "1",
                    "dont_reveal_answer": True,
                }
            ]
        }
    }


class DebugInfo(BaseModel):
    """Debug information included in responses."""

    selected_policy: str = Field(..., description="The policy that was applied")


class ChatResponse(BaseModel):
    """Response schema for the /chat endpoint."""

    response: str = Field(..., description="The tutor's response")
    refusal: bool = Field(..., description="Whether the request was refused (off-topic)")
    citations: Optional[list[str]] = Field(None, description="Any citations or references")
    debug: Optional[DebugInfo] = Field(None, description="Debug information")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "Great question! Let's think about this together. What do you get when you count up 2 from 3?",
                    "refusal": False,
                    "debug": {"selected_policy": "hint_mode"},
                }
            ]
        }
    }

