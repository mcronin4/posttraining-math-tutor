"""
Math Tutor Evaluation Package

Evaluation harness for math accuracy, tutoring quality, and safety.
"""

from .utils import parse_llm_json_response, ParseResult

__all__ = ["parse_llm_json_response", "ParseResult"]