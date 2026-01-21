"""
Type definitions for LLM-as-a-Judge evaluation.

This module contains all dataclass definitions used in the socratic evaluation system.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import tinker
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer


@dataclass
class ModelClients:
    """Container for initialized model clients."""
    tutor_client: tinker.SamplingClient
    tutor_tokenizer: Tokenizer
    tutor_renderer: Renderer
    tutor_model_name: str
    student_judge_client: tinker.SamplingClient # Shared client for student and judge
    student_judge_tokenizer: Tokenizer
    student_judge_renderer: Renderer
    student_judge_model_name: str


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    role: str  # "tutor" or "student"
    content: str
    turn: int
    thinking: Optional[str] = None  # Reasoning/thinking content from <think> blocks


@dataclass
class ConversationResult:
    """Result of a single tutoring conversation."""
    problem_id: str
    problem: str
    expected_answer: str
    student_profile: str
    messages: List['ConversationMessage'] = field(default_factory=list)
    judge_evaluation: Optional[str] = None
    judge_scores: Optional[dict] = None
    final_turn: int = 0
    student_solved: bool = False
