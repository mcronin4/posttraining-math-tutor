"""
Model adapters package.

This package provides a clean interface for swapping between different
model backends (Ollama, OpenAI, etc.).
"""

import os
from typing import Optional

from .base import ModelAdapter
from .ollama import OllamaAdapter

# Global model adapter instance
_model_adapter: Optional[ModelAdapter] = None


def get_model_adapter() -> ModelAdapter:
    """
    Get the configured model adapter.

    By default, returns the OllamaAdapter. Configure via environment variables:
    - MODEL_TYPE: "ollama" (default) or future options
    - OLLAMA_MODEL: Model name (defaults to "qwen2.5:8b")
    - OLLAMA_BASE_URL: Ollama API URL (defaults to "http://localhost:11434")
    """
    global _model_adapter

    if _model_adapter is None:
        model_type = os.getenv("MODEL_TYPE", "ollama")

        if model_type == "ollama":
            _model_adapter = OllamaAdapter()
        # TODO: Add other model adapters here
        # elif model_type == "openai":
        #     _model_adapter = OpenAIAdapter()
        else:
            # Default to Ollama
            _model_adapter = OllamaAdapter()

    return _model_adapter


def set_model_adapter(adapter: ModelAdapter) -> None:
    """
    Set a custom model adapter (useful for testing).
    """
    global _model_adapter
    _model_adapter = adapter


__all__ = ["ModelAdapter", "OllamaAdapter", "get_model_adapter", "set_model_adapter"]

