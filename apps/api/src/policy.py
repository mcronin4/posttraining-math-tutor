"""
Tutor Policy Module

This module implements content filtering and policy rules for the tutor.
It determines when to refuse requests, redirect users, or apply special handling.
"""

import re
from typing import Optional

from .schemas import TutoringMode


class TutorPolicy:
    """
    Policy wrapper for tutoring requests.

    Handles:
    - Off-topic detection and refusal
    - Direct answer detection and redirection
    - Content filtering
    """

    # Keywords that suggest off-topic content
    OFF_TOPIC_PATTERNS = [
        r"\b(tell me a joke|tell me a story|play a game)\b",
        r"\b(who are you|what are you|are you real|are you human)\b",
        r"\b(your name|your age|where do you live)\b",
        r"\b(write me a|write a poem|write a song)\b",
        r"\b(what's the weather|news|politics|sports)\b",
        r"\b(hello|hi|hey)\s*$",  # Just greetings with nothing else
    ]

    # Keywords that suggest math content
    MATH_KEYWORDS = [
        r"\b\d+\b",  # Numbers
        r"\b(add|subtract|multiply|divide|plus|minus|times)\b",
        r"\b(equation|solve|calculate|find|prove|simplify)\b",
        r"\b(fraction|decimal|percent|ratio)\b",
        r"\b(algebra|geometry|calculus|trigonometry)\b",
        r"\b(triangle|circle|square|rectangle|angle)\b",
        r"\b(graph|slope|intercept|function)\b",
        r"\b(derivative|integral|limit)\b",
        r"\b(polynomial|exponent|logarithm)\b",
        r"[+\-*/=<>≤≥²³√∫∑]",  # Math symbols
        r"\b(x|y|n)\s*[+\-*/=]",  # Variables in equations
    ]

    # Patterns suggesting user wants direct answer
    ANSWER_REQUEST_PATTERNS = [
        r"\b(what is the answer|give me the answer|tell me the answer)\b",
        r"\b(just tell me|just give me)\b",
        r"\b(what's|what is)\s+\d+\s*[+\-*/]\s*\d+\s*\??\s*$",  # "What's 2+2?"
        r"\b(solve this for me|do this for me)\b",
    ]

    def __init__(self):
        """Initialize policy with compiled regex patterns."""
        self._off_topic_re = [re.compile(p, re.IGNORECASE) for p in self.OFF_TOPIC_PATTERNS]
        self._math_re = [re.compile(p, re.IGNORECASE) for p in self.MATH_KEYWORDS]
        self._answer_re = [re.compile(p, re.IGNORECASE) for p in self.ANSWER_REQUEST_PATTERNS]

    def is_off_topic(self, question: str) -> bool:
        """
        Determine if the question is off-topic (not math-related).

        Returns True if the question appears to be off-topic.
        """
        question = question.strip()

        # Check for math content first - if it has math, it's on-topic
        has_math = any(pattern.search(question) for pattern in self._math_re)
        if has_math:
            return False

        # Check for explicit off-topic patterns
        has_off_topic = any(pattern.search(question) for pattern in self._off_topic_re)
        if has_off_topic:
            return True

        # If very short and no math content, might be off-topic
        if len(question) < 10 and not has_math:
            # Allow short math questions like "2+2?" or "x=5?"
            if not re.search(r"[\d+\-*/=x]", question):
                return True

        return False

    def is_asking_for_answer(self, question: str) -> bool:
        """
        Determine if the user is explicitly asking for a direct answer.

        Returns True if the user wants the answer revealed.
        """
        return any(pattern.search(question) for pattern in self._answer_re)

    def get_refusal_message(self) -> str:
        """Get a friendly refusal message for off-topic requests."""
        return (
            "I'm here to help you with math! 📚 "
            "If you have a math question or problem you're working on, "
            "I'd be happy to help you understand it. What math topic can I help you with today?"
        )

    def get_no_reveal_message(self, mode: TutoringMode) -> str:
        """Get a message redirecting from direct answer requests."""
        if mode == TutoringMode.HINT:
            return (
                "I want to help you discover the answer yourself! 🌟 "
                "Let me give you a hint instead. "
                "What have you tried so far? Can you tell me what you know about this problem?"
            )
        elif mode == TutoringMode.CHECK_STEP:
            return (
                "I'd love to check your work! ✏️ "
                "Can you show me your attempt or the steps you've taken? "
                "That way I can help you see if you're on the right track."
            )
        else:  # EXPLAIN
            return (
                "Let me explain the concept instead of giving you the direct answer. 💡 "
                "Understanding the 'why' will help you solve this and similar problems. "
                "What part of this problem is most confusing?"
            )

