"""
Ollama Model Adapter

Integration with Ollama for running local open-source models like Qwen 8B.
Requires Ollama to be installed and running locally.

Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull Qwen model: ollama pull qwen2.5:8b
    3. Set MODEL_NAME environment variable (defaults to qwen2.5:8b)
"""

import os
from typing import Optional

import httpx

from ..schemas import GradeLevel, TutoringMode
from .base import ModelAdapter


class OllamaAdapter(ModelAdapter):
    """
    Ollama adapter for local LLM inference.

    Uses Ollama's HTTP API to generate responses from models like Qwen 8B.
    """

    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Ollama adapter.

        Args:
            model_name: Name of the Ollama model (e.g., "qwen3:8b")
            base_url: Ollama API base URL (defaults to http://localhost:11434)
        """
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate_response(
        self,
        question: str,
        attempt: Optional[str],
        mode: TutoringMode,
        grade: GradeLevel,
        dont_reveal_answer: bool,
        topic_tags: Optional[list[str]] = None,
    ) -> tuple[str, str]:
        """Generate a tutoring response using Ollama."""

        # Build the prompt based on mode and settings
        system_prompt = self._build_system_prompt(mode, grade, dont_reveal_answer)
        user_prompt = self._build_user_prompt(question, attempt, mode, topic_tags)

        # Call Ollama API
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract response text
            response_text = data.get("message", {}).get("content", "")
            if not response_text:
                response_text = "I apologize, but I couldn't generate a response. Please try again."

            return response_text.strip(), f"ollama_{mode.value}"

        except httpx.HTTPError as e:
            error_msg = f"Error connecting to Ollama: {str(e)}"
            if "Connection refused" in str(e):
                error_msg += "\n\nMake sure Ollama is running: ollama serve"
            raise RuntimeError(error_msg) from e

    def _build_system_prompt(
        self, mode: TutoringMode, grade: GradeLevel, dont_reveal_answer: bool
    ) -> str:
        """Build the system prompt for the tutor."""
        # Determine answer policy text
        answer_policy = (
            "Never reveal the final answer"
            if dont_reveal_answer
            else "You may provide the answer when helpful"
        )

        mode_descriptions = {
            TutoringMode.HINT: "Hint Mode - Provide a single guiding question or small hint. Focus on the next logical step. Do not solve multiple steps at once.",
            TutoringMode.CHECK_STEP: "Check Step Mode - If student provides an attempt: validate their work. If correct: acknowledge and ask about the next step. If incorrect: identify where they went wrong without giving the answer. If no attempt provided: ask them to show their work first.",
            TutoringMode.EXPLAIN: "Explain Mode - Explain the relevant concept or method. Use concrete examples appropriate to grade level. Still avoid giving the direct answer to the specific problem.",
        }

        base_prompt = f"""You are a friendly and encouraging math tutor helping students in Ontario, Canada learn mathematics. Your role is to guide students to understanding through questions and hints rather than giving direct answers.

Core Principles:
1. {answer_policy}
2. Ask guiding questions that help students discover the solution themselves
3. Validate student thinking by acknowledging correct steps and gently redirecting incorrect approaches
4. Match the grade level - use vocabulary and examples appropriate for grade {grade.value}
5. Be encouraging - celebrate progress and effort, not just correct answers

Tutoring Mode: {mode_descriptions[mode]}

Response Guidelines:
- Keep responses concise (2-4 sentences for hints, slightly longer for explanations)
- Use simple language appropriate for grade {grade.value}
- Include encouragement when appropriate
- Always end with a question or prompt for the student to continue their thinking

Safety:
- Only answer mathematics questions
- Politely redirect off-topic questions back to math
- Do not engage with inappropriate content
- Do not provide help with cheating or academic dishonesty

Ontario Curriculum Alignment:
- Use BEDMAS for order of operations (not PEMDAS)
- Use Canadian spelling (colour, centre, metre)
- Reference familiar Canadian contexts in word problems
"""

        return base_prompt

    def _build_user_prompt(
        self,
        question: str,
        attempt: Optional[str],
        mode: TutoringMode,
        topic_tags: Optional[list[str]],
    ) -> str:
        """Build the user prompt from the question and context."""
        prompt_parts = [f"Math Question: {question}"]

        if attempt:
            prompt_parts.append(f"Student's Attempt: {attempt}")

        if topic_tags:
            prompt_parts.append(f"Topics: {', '.join(topic_tags)}")

        prompt_parts.append(f"\nPlease provide a helpful {mode.value} response.")

        return "\n".join(prompt_parts)

    def get_model_info(self) -> dict:
        """Get information about this model."""
        return {
            "name": "OllamaAdapter",
            "model": self.model_name,
            "base_url": self.base_url,
            "type": "ollama",
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

