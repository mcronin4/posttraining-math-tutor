"""
Custom renderers for tutor/student role mapping.

This module provides custom renderer wrappers that map "tutor"/"student" roles
to the underlying model's expected role format (e.g., "user"/"assistant" for Kimi).
"""

from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import Message,RenderContext, RenderedMessage
import tinker


def _get_kimi_k2_renderer_class():
    """Get KimiK2Renderer class using get_renderer.
    
    Since we need the class for inheritance (not an instance), we use get_renderer
    with a dummy tokenizer to get an instance, then extract its class type.
    This follows the pattern of using get_renderer instead of direct imports.
    """
    # Import tokenizer_utils to create a dummy tokenizer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    # Use a common model to get a tokenizer (any model will work for getting the class)
    dummy_tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
    # Use get_renderer to get the renderer instance
    renderer_instance = renderers.get_renderer("kimi_k2", dummy_tokenizer)
    # Return the class type for inheritance
    return type(renderer_instance)


# Get the base class using get_renderer (as requested, instead of direct import)
KimiK2Renderer = _get_kimi_k2_renderer_class()


class TutorStudentKimiRenderer(KimiK2Renderer):
    """
    Custom renderer for Kimi K2 that maps "tutor"/"student" roles to
    "user"/"assistant" while preserving thinking token handling.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # Map custom roles to Kimi's expected roles
        self.role_map = {
            "tutor": "user",
            "student": "assistant"
        }

    def _map_role(self, role: str) -> str:
        """Map custom role to Kimi's expected role."""
        return self.role_map.get(role, role)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Override to map roles before rendering."""
        # Create a copy with mapped role
        mapped_message = {**message, "role": self._map_role(message["role"])}

        # Use parent's render_message with the mapped role and context
        return super().render_message(mapped_message, ctx)

    def build_generation_prompt(
        self, messages: list[Message], role: str = "student", prefill: str | None = None
    ) -> tinker.ModelInput:
        """Override to map the generation role."""
        mapped_role = self._map_role(role)
        return super().build_generation_prompt(messages, mapped_role, prefill)
