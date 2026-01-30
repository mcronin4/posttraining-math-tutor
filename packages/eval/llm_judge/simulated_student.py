"""Simulated student implementation for tutoring conversations."""

import sys
from pathlib import Path
from typing import List, Optional

try:
    from .prompts import build_student_system_prompt
    from .response_generation import (
        generate_response,
        check_if_solved,
        clean_student_response,
    )
    from .llm_judge_types import ConversationMessage
except ImportError:
    # Fallback for when script is run directly (not as module)
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.llm_judge.prompts import build_student_system_prompt
    from eval.llm_judge.response_generation import (
        generate_response,
        check_if_solved,
        clean_student_response,
    )
    from eval.llm_judge.llm_judge_types import ConversationMessage

try:
    import tinker
except ImportError:
    print("❌ Error: Tinker SDK not found.")
    sys.exit(1)


class SimulatedStudent:
    """Simulated student that wraps the Kimi-K2 client for generating student responses."""
    
    def __init__(
        self,
        client: tinker.SamplingClient,
        renderer: any,
        tokenizer: any,
        student_profile: str,
    ):
        """Initialize the simulated student.
        
        Args:
            client: Tinker sampling client (Kimi-K2)
            renderer: Renderer instance for the student model
            tokenizer: Tokenizer instance for the student model
            student_profile: The student persona/profile description
        """
        self.client = client
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.student_profile = student_profile
    
    def _build_system_prompt(self, problem: str, seed_error: Optional[str] = None) -> str:
        """Build the system prompt for the student, optionally injecting a seed error.
        
        Args:
            problem: The math problem text
            seed_error: Optional misconception/error to inject into the first message
            
        Returns:
            The student system prompt as a string
        """
        base_prompt = build_student_system_prompt(
            student_profile=self.student_profile,
            problem=problem
        )
        
        # If seed_error is provided, inject it into the prompt
        if seed_error:
            error_instruction = f"""

# Initial Misconception
You have a specific misconception about this problem: {seed_error}
Your first message should reflect this misconception. Start by expressing confusion or making an error related to: {seed_error}
After your first message, you can respond naturally based on the tutor's guidance, but your initial response should be based on this misconception.
"""
            return base_prompt + error_instruction
        
        return base_prompt
    
    async def generate_student_turn(
        self,
        problem: str,
        history: List[ConversationMessage],
        seed_error: Optional[str] = None,
        temperature: float = 0.9,
        max_tokens: int = 8192,
        debug: bool = False,
        log_buffer: Optional[List[str]] = None,
    ) -> tuple[str, Optional[str], bool]:
        """Generate a student turn response.
        
        Args:
            problem: The math problem text
            history: Previous conversation messages
            seed_error: Optional misconception/error to inject into the first message
            temperature: Sampling temperature (default: 0.9)
            max_tokens: Maximum tokens to generate (default: 8192)
            debug: If True, print debug information (default: False)
            log_buffer: Optional list to append log messages to
            
        Returns:
            Tuple of (content, thinking, solved) where:
            - content: The cleaned student response (with SOLVED token removed)
            - thinking: The reasoning content from <think> blocks (None if no blocks found)
            - solved: Whether the student has solved the problem
        """
        # DEBUG: Log what history we're receiving
        if debug:
            debug_msg = f"\n🔍 DEBUG (generate_student_turn): Received history with {len(history)} messages"
            if log_buffer is not None:
                log_buffer.append(debug_msg)
            else:
                print(debug_msg)
            for i, msg in enumerate(history):
                debug_msg = f"  History[{i}]: role={msg.role}, turn={msg.turn}, content_len={len(msg.content)}"
                if log_buffer is not None:
                    log_buffer.append(debug_msg)
                else:
                    print(debug_msg)
        
        # Build system prompt with optional seed error injection
        system_prompt = self._build_system_prompt(problem, seed_error)
        
        # Generate response
        student_content, student_thinking = await generate_response(
            client=self.client,
            renderer=self.renderer,
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            conversation_history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            role="student",
            debug=debug,
            log_buffer=log_buffer,
        )
        
        # Check if student solved (using special token) before cleaning
        student_solved = check_if_solved(student_content)
        
        # Clean the response (remove tokens) before storing
        cleaned_response = clean_student_response(student_content)
        
        return cleaned_response, student_thinking, student_solved
