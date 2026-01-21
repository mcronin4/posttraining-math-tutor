"""Response generation and related utilities."""

import sys
from pathlib import Path
from typing import List, Optional

try:
    from ..utils import parse_reasoning_blocks
    from .llm_judge_types import ConversationMessage
except ImportError:
    # Fallback for when script is run directly (not as module)
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.utils import parse_reasoning_blocks
    from eval.llm_judge.llm_judge_types import ConversationMessage

try:
    import tinker
    from tinker.types import SamplingParams
except ImportError:
    print("❌ Error: Tinker SDK not found.")
    sys.exit(1)

from .prompts import STUDENT_SOLVED_TOKEN

# Track which roles have already printed their system prompt (for logging)
_roles_with_printed_system_prompt = set()


def print_debug_info(
    role: str,
    system_prompt: str,
    messages: list,
    conversation_history: Optional[List[ConversationMessage]],
    formatted_prompt: any,
) -> None:
    """Print debug information about the messages being sent to the model.
    
    Args:
        role: Role for generation ("tutor" or "student")
        system_prompt: System prompt being used
        messages: List of message dicts (after role mapping, including system)
        conversation_history: Raw conversation history (before role mapping)
        formatted_prompt: The formatted prompt object
    """
    print("\n" + "="*80)
    print(f"🔍 DEBUG: generate_response called for role='{role}'")
    print("="*80)
    
    # Only print system prompt on turn 1 for each role
    is_turn_1 = role not in _roles_with_printed_system_prompt
    if is_turn_1:
        _roles_with_printed_system_prompt.add(role)
        print(f"\n📋 SYSTEM PROMPT ({len(system_prompt)} chars):")
        print("-"*80)
        print(system_prompt)
        print("-"*80)
    else:
        print(f"\n📋 SYSTEM PROMPT: (omitted - already shown for role '{role}' on turn 1)")
    
    conversation_messages = [msg for msg in messages if msg["role"] != "system"]
    print(f"\n💬 CONVERSATION HISTORY ({len(conversation_messages)} messages, excluding system):")
    print("-"*80)
    for i, msg in enumerate(conversation_messages):
        print(f"\nMessage {i} (role='{msg['role']}'):")
        print(f"  Content ({len(msg['content'])} chars): {repr(msg['content'][:200])}")
        if len(msg['content']) > 200:
            print(f"  ... (truncated, full length: {len(msg['content'])})")
    print("-"*80)
    
    print("="*80 + "\n")


def has_incomplete_thinking_tag(raw_output: str) -> bool:
    """
    Check if the raw output contains an incomplete thinking tag.
    
    An incomplete thinking tag means there's an opening <think> tag
    but no matching closing </think> tag after it.
    
    Args:
        raw_output: Raw model output to check
        
    Returns:
        True if there's an incomplete thinking tag, False otherwise
    """
    # Find the last occurrence of opening tag
    last_open_idx = raw_output.rfind("<think>")
    
    # If no opening tag, it's not incomplete
    if last_open_idx == -1:
        return False
    
    # Check if there's a closing tag after the last opening tag
    after_open = raw_output[last_open_idx:]
    closing_idx = after_open.find("</think>")
    
    # If no closing tag found after the opening tag, it's incomplete
    return closing_idx == -1


def check_if_solved(response: str) -> bool:
    """Check if the student response contains the SOLVED token indicating they solved the problem."""
    # Check if the response contains the special solved token
    # The student model is instructed to output this token when it has determined
    # the correct answer based on sufficient tutor guidance
    return STUDENT_SOLVED_TOKEN in response


def clean_student_response(response: str) -> str:
    """Remove the SOLVED token from the student response before storing it."""
    # Remove the special token so it doesn't appear in conversation logs
    cleaned = response.replace(STUDENT_SOLVED_TOKEN, "")
    return cleaned.strip()


def reset_debug_tracking():
    """Reset the debug tracking for system prompts. Called at the start of each conversation."""
    global _roles_with_printed_system_prompt
    _roles_with_printed_system_prompt.clear()


async def generate_response(
    client: tinker.SamplingClient,
    renderer: any,
    tokenizer: any,
    system_prompt: str,
    conversation_history: Optional[List[ConversationMessage]] = None,
    temperature: float = 0.6,
    max_tokens: int = 8192, # Need longer max tokens because even when the 'response' is short, the thinking takes quite a few tokens
    role: str = "student",
    max_retries: int = 1,
    debug: bool = False,
    log_buffer: Optional[List[str]] = None,
) -> tuple[str, Optional[str]]:
    """Generate a response using the Tinker sampling client.
    
    Args:
        client: Tinker sampling client
        renderer: Renderer instance (supports custom tutor/student roles)
        tokenizer: Tokenizer instance
        system_prompt: System prompt (required)
        conversation_history: Previous conversation messages (ONLY content is used, not thinking)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        role: Role for generation ("tutor" or "student")
        max_retries: Maximum number of retries if incomplete thinking tag is detected (default: 1)
        debug: If True, print detailed debug information about messages being sent (default: False)
        
    Returns:
        Tuple of (content, thinking) where:
        - content: The response text with reasoning blocks removed (safe to pass to next generation)
        - thinking: The reasoning content from <think> blocks (None if no blocks found)
    """
    current_max_tokens = max_tokens
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            messages = []

            # Add system prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history if provided
            # CRITICAL: Only pass content, NOT thinking, to avoid leaking reasoning to next generation
            # Use tutor/student roles directly - renderers will handle mapping
            if conversation_history:
                for msg in conversation_history:
                    messages.append({"role": msg.role, "content": msg.content})

            # need to use a renderer to apply chat template and think tags, also tokenizes the query
            # Pass the role parameter so renderers can map it appropriately
            formatted_prompt = renderer.build_generation_prompt(messages, role=role)
            
            # Print debug information if requested
            if debug:
                print_debug_info(
                    role=role,
                    system_prompt=system_prompt,
                    messages=messages,
                    conversation_history=conversation_history,
                    formatted_prompt=formatted_prompt,
                )
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=current_max_tokens,
                stop=renderer.get_stop_sequences(),
            )
            
            # Generate response
            result = await client.sample_async(prompt=formatted_prompt, sampling_params=sampling_params, num_samples=1)
            # Extract the generated text
            raw_output = ""
            for sequence in result.sequences:
                raw_output = tokenizer.decode(sequence.tokens)
            
            
            # Check for incomplete thinking tag
            if has_incomplete_thinking_tag(raw_output):
                if retry_count < max_retries:
                    # Double the max_tokens for retry
                    current_max_tokens = current_max_tokens * 2
                    retry_count += 1
                    msg = f"   ⚠️  Detected incomplete thinking tag. Retrying with max_tokens={current_max_tokens} (retry {retry_count}/{max_retries})"
                    if log_buffer is not None:
                        log_buffer.append(msg)
                    else:
                        print(msg)
                    continue
                else:
                    msg = f"   ⚠️  Detected incomplete thinking tag but max retries ({max_retries}) reached. Returning partial response."
                    if log_buffer is not None:
                        log_buffer.append(msg)
                    else:
                        print(msg)
            
            # Parse reasoning blocks and separate content from thinking
            content, thinking = parse_reasoning_blocks(raw_output)
            
            return content, thinking
                
        except Exception as e:
            msg = f"   ⚠️  Error generating response: {e}"
            if log_buffer is not None:
                log_buffer.append(msg)
            else:
                print(msg)
            import traceback
            if log_buffer is not None:
                import io
                traceback_str = io.StringIO()
                traceback.print_exc(file=traceback_str)
                log_buffer.append(traceback_str.getvalue())
            else:
                traceback.print_exc()
            return f"[ERROR: {str(e)}]", None
    
    # Should never reach here, but just in case
    return "[ERROR: Max retries exceeded]", None
