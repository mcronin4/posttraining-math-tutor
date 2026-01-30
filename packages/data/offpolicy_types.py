"""
Type definitions for off-policy distillation data processing.

This module defines the shared data structures used for processing
MathDial and SocraticMATH datasets into a unified format for off-policy distillation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ConversationMessage:
    """
    Represents a single message in a conversation trajectory.
    
    Attributes:
        role: "tutor" or "student"
        content: Message content (strategy tags removed for MathDial)
        turn: Turn number (0-indexed)
    """
    role: str  # "tutor" or "student"
    content: str
    turn: int


@dataclass
class ConversationTrajectory:
    """
    Represents a complete conversation trajectory for off-policy distillation.
    
    This format is used to store processed conversations from MathDial and
    SocraticMATH datasets before converting them to training examples.
    
    Attributes:
        id: Unique identifier for this conversation
        question: Original problem/question
        expected_answer: Ground truth answer
        messages: Full conversation history (list of ConversationMessage)
        source: Dataset source ("mathdial" or "socraticmath")
        metadata: Additional information (student_profile, etc.)
    """
    id: str
    question: str
    expected_answer: str
    messages: List[ConversationMessage] = field(default_factory=list)
    source: str = ""  # "mathdial" or "socraticmath"
    metadata: Dict[str, Any] = field(default_factory=dict)
