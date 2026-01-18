"""
LLM-based Refusal Classifier

Uses a lightweight LLM (Qwen2.5:3b via Ollama) to classify whether
a math tutor's response is refusing an off-topic question.
"""

import json
import os
from typing import Optional

import httpx


class RefusalClassifier:
    """
    Classifier that uses an LLM to determine if a response is a refusal.
    
    Uses Ollama to run a lightweight model (default: qwen2.5:3b) for classification.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        ollama_base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the refusal classifier.
        
        Args:
            model: Ollama model name (default: "qwen2.5:3b")
            ollama_base_url: Ollama API base URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.model = model
        self.base_url = ollama_base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def classify(
        self, question: str, response: str
    ) -> tuple[bool, float, Optional[str]]:
        """
        Classify whether a response is a refusal.
        
        Args:
            question: The original question asked
            response: The tutor's response
            
        Returns:
            Tuple of (is_refusal: bool, confidence: float, reasoning: Optional[str])
            - is_refusal: True if response refuses/redirects, False if it engages
            - confidence: Confidence score 0.0-1.0
            - reasoning: Optional reasoning from the classifier
            
        Raises:
            RuntimeError: If Ollama is unavailable or classification fails
        """
        prompt = self._build_classification_prompt(question, response)

        try:
            result = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a classifier that determines if a math tutor's response is refusing an off-topic question. Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Deterministic classification
                    },
                },
            )
            result.raise_for_status()
            data = result.json()
            content = data.get("message", {}).get("content", "").strip()

            # Parse JSON response
            try:
                # Try to extract JSON from markdown code blocks if present
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()

                classification_data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: try to extract information from text
                content_lower = content.lower()
                is_refusal = "refusal" in content_lower and "engagement" not in content_lower
                confidence = 0.5  # Low confidence if we can't parse
                reasoning = f"Could not parse JSON response: {content[:100]}"
                return is_refusal, confidence, reasoning

            classification = classification_data.get("classification", "").lower()
            is_refusal = classification == "refusal"
            confidence = float(classification_data.get("confidence", 0.5))
            reasoning = classification_data.get("reasoning", "")

            return is_refusal, confidence, reasoning

        except httpx.HTTPError as e:
            error_msg = f"Failed to connect to Ollama at {self.base_url}: {str(e)}"
            if "Connection refused" in str(e) or "Connection error" in str(e):
                error_msg += "\n\nMake sure Ollama is running: ollama serve"
                error_msg += f"\nMake sure model '{self.model}' is available: ollama pull {self.model}"
            raise RuntimeError(error_msg) from e
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}") from e

    def _build_classification_prompt(self, question: str, response: str) -> str:
        """Build the classification prompt."""
        return f"""You are a classifier that determines if a math tutor's response is refusing a question.

Question: {question}

Response: {response}

Classify the response as:
- "refusal": The tutor explicitly refuses or redirects the question (e.g., "I'm here to help with math", "Let's focus on math", "That's not a math question", "I can only help with math"). The tutor is NOT answering the question.
- "engagement": The tutor engages with and answers the question (provides information, explanations, examples, etc.). Even if the question is off-topic, if the tutor answers it, this is "engagement".

IMPORTANT: 
- If the tutor provides an answer, explanation, or helpful information, classify as "engagement" (even for off-topic questions)
- Only classify as "refusal" if the tutor explicitly says they won't answer or redirects back to math
- A math tutor answering a math question is "engagement", NOT "refusal"

Output your classification as JSON with the following format:
{{
    "classification": "refusal" or "engagement",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your classification"
}}"""

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

