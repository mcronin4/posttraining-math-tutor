"""Conversation execution logic for tutoring conversations and judge evaluation."""

import sys
from pathlib import Path
from typing import List, Optional

try:
    from ..utils import parse_llm_json_response
    from .llm_judge_types import ModelClients, ConversationMessage, ConversationResult
except ImportError:
    # Fallback for when script is run directly (not as module)
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.utils import parse_llm_json_response
    from eval.llm_judge.llm_judge_types import ModelClients, ConversationMessage, ConversationResult

from .prompts import build_tutor_system_prompt
from .response_generation import (
    generate_response,
    reset_debug_tracking,
)
from .simulated_student import SimulatedStudent


async def run_conversation(
    model_clients: ModelClients,
    problem: dict,
    student_profile: str,
    max_turns: int = 10,
    prompt_type: str = "slim",
    debug: bool = False,
    log_buffer: Optional[List[str]] = None,
    seed_error: Optional[str] = None,
) -> ConversationResult:
    """Run a single tutoring conversation between tutor and student models.
    
    Args:
        model_clients: Initialized model clients
        problem: Problem dictionary with 'id', 'question', and 'answer' keys
        student_profile: The student persona/profile description
        max_turns: Maximum number of conversation turns
        prompt_type: Type of tutor prompt to use
        debug: If True, print debug information
        log_buffer: Optional list to append log messages to
        seed_error: Optional misconception/error to inject into the student's first message
    """
    # Reset system prompt tracking for this conversation
    reset_debug_tracking()
    
    problem_id = problem.get("id", "unknown")
    problem_text = problem.get("question", "")
    expected_answer = problem.get("answer", "")
    
    messages: List[ConversationMessage] = []
    # Build system prompt for tutor with problem included (persistent across all turns)
    tutor_system_prompt = build_tutor_system_prompt(prompt_type=prompt_type, problem=problem_text)
    
    # Create simulated student instance
    simulated_student = SimulatedStudent(
        client=model_clients.student_judge_client,
        renderer=model_clients.student_judge_renderer,
        tokenizer=model_clients.student_judge_tokenizer,
        student_profile=student_profile,
    )
    
    # Track conversation outcomes
    student_solved = False
    
    # Conversation loop
    for turn in range(1, max_turns + 1):
        msg = f"STUDENT TURN {turn} STARTING"
        if log_buffer is not None:
            log_buffer.append(msg)
        else:
            print(msg)
        # Student turn (student goes first)
        # Use seed_error only on the first turn
        current_seed_error = seed_error if turn == 1 else None
        cleaned_response, student_thinking, student_solved = await simulated_student.generate_student_turn(
            problem=problem_text,
            history=messages,
            seed_error=current_seed_error,
            temperature=0.9,  # Increased for more varied student responses
            max_tokens=8192, # Need longer max tokens because even when the 'response' is short, the thinking takes quite a few tokens
            debug=debug,
            log_buffer=log_buffer,
        )
        msg = f"STUDENT TURN {turn} COMPLETE"
        if log_buffer is not None:
            log_buffer.append(msg)
        else:
            print(msg)
        
        messages.append(ConversationMessage(role="student", content=cleaned_response, thinking=student_thinking, turn=turn))
        
        # Break if student solved
        if student_solved:
            msg = f"Breaking conversation due to student solved"
            if log_buffer is not None:
                log_buffer.append(msg)
            else:
                print(msg)
            break

        msg = f"TUTOR TURN {turn} STARTING"
        if log_buffer is not None:
            log_buffer.append(msg)
        else:
            print(msg)
        # Tutor turn (tutor responds to student)
        # System prompt includes problem, so no additional prompt needed
        tutor_content, tutor_thinking = await generate_response(
            client=model_clients.tutor_client,
            renderer=model_clients.tutor_renderer,
            tokenizer=model_clients.tutor_tokenizer,
            system_prompt=tutor_system_prompt,
            conversation_history=messages,  # Only content is passed, not thinking
            temperature=0.7,
            max_tokens=8192, # Need longer max tokens because even when the 'response' is short, the thinking takes quite a few tokens
            role="tutor",
            debug=debug,
            log_buffer=log_buffer,
        )
        msg = f"TUTOR TURN {turn} COMPLETE"
        if log_buffer is not None:
            log_buffer.append(msg)
        else:
            print(msg)
        messages.append(ConversationMessage(role="tutor", content=tutor_content, thinking=tutor_thinking, turn=turn))
    
    return ConversationResult(
        problem_id=problem_id,
        problem=problem_text,
        expected_answer=expected_answer,
        student_profile=student_profile,
        messages=messages,
        final_turn=len([m for m in messages if m.role == "tutor"]),
        student_solved=student_solved,
    )


async def judge_conversation(
    model_clients: ModelClients,
    conversation_result: ConversationResult,
    debug: bool = False,
    log_buffer: Optional[List[str]] = None,
) -> tuple[str, dict]:
    """Use the judge model to evaluate the tutoring conversation.
    
    CRITICAL: Only passes content (not thinking) to the judge to avoid leaking reasoning.
    The judge should only see what a human evaluator would see - the actual conversation
    content, not the internal reasoning/thinking of the models.
    
    Note: msg.content is already cleaned by parse_reasoning_blocks() which removes all
    <think>...</think> blocks, so thinking is never included in content.
    """
    # Build judge prompt
    # CRITICAL: Only use content, NOT thinking, to avoid leaking reasoning to judge
    # The msg.content field has already had all thinking blocks removed by parse_reasoning_blocks()
    messages_text = "\n".join([
        f"[{msg.role.upper()} - Turn {msg.turn}]: {msg.content}"
        for msg in conversation_result.messages
    ])
    
    judge_prompt = f"""
You are a Senior Pedagogical Evaluator specializing in Socratic Math Instruction.
Your goal is to evaluate a transcript between an AI Tutor and a Student.

### Context
Problem: {conversation_result.problem}
Expected Answer: {conversation_result.expected_answer}

### Conversation Transcript
{messages_text}

### Evaluation Rubric
Rate each category from 1 (Poor) to 10 (Exemplary):

1. **Information Bottleneck (Safety):** Did the tutor withhold the final answer and intermediate solutions? 
   - *1-2: Directly gave the answer or made it trivial.*
   - *3-4: Gave heavy hints that made the calculation obvious.*
   - *5-6: Provided some hints that partially revealed the solution.*
   - *7-8: Withheld solutions well but had minor hints.*
   - *9-10: Perfectly withheld all solutions while guiding effectively.*

2. **Diagnostic Accuracy:** Did the tutor correctly identify the student's specific misconception or calculation error?
   - *1-2: Ignored student errors or gave completely generic hints.*
   - *3-4: Noticed errors but addressed them superficially.*
   - *5-6: Identified general areas of confusion.*
   - *7-8: Pinpointed specific logic gaps with good accuracy.*
   - *9-10: Perfectly identified exact misconceptions and addressed them precisely.*

3. **Socratic Depth:** Did the tutor ask probing questions that forced the student to reflect on their own logic?
   - *1-2: Purely instructional/telling, no questions.*
   - *3-4: Asked surface-level questions that didn't probe deeply.*
   - *5-6: Asked some helpful questions but mixed with telling.*
   - *7-8: Mostly used questions that guided student thinking well.*
   - *9-10: Questions brilliantly led the student to their own 'aha!' moment.*

4. **Mathematical Soundness:** Was every hint or statement made by the tutor mathematically correct?
   - *1-2: The tutor provided or validated incorrect math.*
   - *3-4: Had significant mathematical errors.*
   - *5-6: Mostly correct but with minor inaccuracies.*
   - *7-8: Correct with occasional imprecise language.*
   - *9-10: Flawless mathematical logic throughout.*

5. **Pedagogical Tone:** Was the tutor supportive, encouraging, and age-appropriate (Grade School)?
   - *1-2: Inappropriate tone, discouraging, or too advanced/complex.*
   - *3-4: Tone was acceptable but lacked warmth or was somewhat inappropriate.*
   - *5-6: Generally appropriate but could be more encouraging or age-appropriate.*
   - *7-8: Supportive and mostly age-appropriate with minor issues.*
   - *9-10: Exemplary tone - perfectly supportive, encouraging, and age-appropriate.*

### Instructions
First, perform a step-by-step analysis of the conversation. Note specific turns where the tutor succeeded or failed. Then, provide the final scores using the 1-10 scale.

Format your response as a JSON object with the following structure:
{{
  "thought_process": "Detailed analysis of the tutoring trajectory...",
  "scores": {{
    "information_bottleneck": int,
    "diagnostic_accuracy": int,
    "socratic_depth": int,
    "math_soundness": int,
    "tone": int
  }},
  "overall_evaluation": "A 2-3 sentence summary of effectiveness.",
  "answer_leaked": bool
}}
"""
    
    judge_content, judge_thinking = await generate_response(
        client=model_clients.student_judge_client,
        renderer=model_clients.student_judge_renderer,
        tokenizer=model_clients.student_judge_tokenizer,
        system_prompt=judge_prompt,
        temperature=0.3,  # Lower temperature for more consistent judging
        max_tokens=8192, # Need longer max tokens because even when the 'response' is short, the thinking takes quite a few tokens
        role="student",  # Judge uses student/judge client, default to student role
        debug=debug,
        log_buffer=log_buffer,
    )
    
    # Parse JSON from response using robust utility function
    # Use only content, not thinking, for JSON parsing
    parse_result = parse_llm_json_response(
        judge_content,
        allow_sloppy=True,
        expected_keys=["scores"],  # Ensure we have the scores key
        warn_on_failure=True,
    )
    
    if parse_result.success and parse_result.data:
        parsed = parse_result.data
        judge_scores = parsed.get("scores", {})
        evaluation_text = parsed.get("overall_evaluation", judge_content)
    else:
        # Fallback: use the raw response if parsing failed
        evaluation_text = judge_content
        judge_scores = {}
    
    return evaluation_text, judge_scores
