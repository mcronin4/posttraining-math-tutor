"""
Base Model Adapter Interface

This module defines the abstract interface that all model adapters must implement.
This allows easy swapping between stub models, different LLM providers, and
fine-tuned models.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..schemas import GradeLevel, TutoringMode


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    All model implementations (stub, OpenAI, Tinker, etc.) should
    inherit from this class and implement the required methods.
    """

    @abstractmethod
    async def generate_response(
        self,
        question: str,
        attempt: Optional[str],
        mode: TutoringMode,
        grade: GradeLevel,
        dont_reveal_answer: bool,
        topic_tags: Optional[list[str]] = None,
    ) -> tuple[str, str]:
        """
        Generate a tutoring response.

        Args:
            question: The math question or problem
            attempt: Optional student attempt at the problem
            mode: The tutoring mode (hint, check_step, explain)
            grade: The student's grade level
            dont_reveal_answer: Whether to avoid revealing the final answer
            topic_tags: Optional topic tags for context

        Returns:
            A tuple of (response_text, policy_used)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get information about the model.

        Returns:
            A dict with model metadata (name, version, etc.)
        """
        pass

