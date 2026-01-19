#!/usr/bin/env python3
"""
LLM-as-a-Judge Socratic Evaluation

Initializes Tinker sampling clients for LLM-as-a-judge evaluation system:
- Tutor model: The model being evaluated (specified via CLI)
- Student model: Kimi-K2-Thinking (acts as confused student)
- Judge model: Kimi-K2-Thinking (evaluates tutoring trajectories, shares client with student)

Usage:
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B
    
    python socratic_eval_llm_judge.py --tutor-model tinker://checkpoint/path
"""

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

# Setup path for direct script execution (when not run as module)
# This allows relative imports to work when script is run directly
try:
    from ..utils import parse_llm_json_response, parse_reasoning_blocks
    from .custom_renderers import TutorStudentKimiRenderer
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    from pathlib import Path
    # Add packages directory to path so we can import eval
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.utils import parse_llm_json_response, parse_reasoning_blocks
    from eval.llm_judge.custom_renderers import TutorStudentKimiRenderer

try:
    import tinker
    from tinker.types import SamplingParams
    
    tinker_version = getattr(tinker, '__version__', 'unknown')
    print(f"📦 Using tinker version: {tinker_version}")
    if tinker_version != '0.8.0':
        print(f"⚠️  Warning: Expected tinker 0.8.0, but found {tinker_version}")
        print(f"   This may cause issues with get_tokenizer() method")
except ImportError:
    print("❌ Error: Tinker SDK not found.")
    print("\nPlease install the Tinker Python SDK")
    sys.exit(1)

try:
    from tinker_cookbook import renderers, tokenizer_utils
    from tinker_cookbook.model_info import get_recommended_renderer_name
except ImportError:
    print("❌ Error: tinker_cookbook not found.")
    print("\nPlease install tinker-cookbook")
    sys.exit(1)


@dataclass
class ModelClients:
    """Container for initialized model clients."""
    tutor_client: tinker.SamplingClient
    tutor_tokenizer: any
    tutor_renderer: any
    tutor_model_name: str
    student_judge_client: tinker.SamplingClient  # Shared client for student and judge
    student_judge_tokenizer: any
    student_judge_renderer: any
    student_judge_model_name: str


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""
    role: str  # "tutor" or "student"
    content: str
    turn: int
    thinking: Optional[str] = None  # Reasoning/thinking content from <think> blocks


@dataclass
class ConversationResult:
    """Result of a single tutoring conversation."""
    problem_id: str
    problem: str
    expected_answer: str
    student_profile: str
    messages: List[ConversationMessage] = field(default_factory=list)
    judge_evaluation: Optional[str] = None
    judge_scores: Optional[dict] = None
    final_turn: int = 0
    student_solved: bool = False
    tutor_revealed: bool = False


def initialize_tutor_model(
    service_client: tinker.ServiceClient,
    tutor_model_name: str,
) -> tuple[tinker.SamplingClient, any, any]:
    """
    Initialize tutor model with tokenizer, renderer, and sampling client.
    
    Args:
        service_client: Tinker ServiceClient instance
        tutor_model_name: Model name or path for tutor model
        
    Returns:
        Tuple of (sampling_client, tokenizer, renderer)
    """
    print(f"📚 Initializing tutor model: {tutor_model_name}")
    
    # Get tokenizer for tutor model
    tokenizer = tokenizer_utils.get_tokenizer(tutor_model_name)
    
    # Get recommended renderer name
    renderer_name = get_recommended_renderer_name(tutor_model_name)
    print(f"   Renderer: {renderer_name}")
    
    # Check if this is a Kimi model - if so, use custom renderer
    if "kimi" in tutor_model_name.lower() or renderer_name == "kimi_k2":
        print(f"   Using custom TutorStudentKimiRenderer for role mapping")
        renderer = TutorStudentKimiRenderer(tokenizer)
    else:
        # For other models (e.g., Qwen), use standard renderer which supports custom roles
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(
        base_model=tutor_model_name
    )
    
    print(f"   ✅ Tutor model initialized\n")
    return sampling_client, tokenizer, renderer


def initialize_student_judge_model(
    service_client: tinker.ServiceClient,
    model_name: str = "moonshotai/Kimi-K2-Thinking",
) -> tuple[tinker.SamplingClient, any, any]:
    """
    Initialize student and judge model (shared client).
    
    Args:
        service_client: Tinker ServiceClient instance
        model_name: Model name for student/judge (default: Kimi-K2-Thinking)
        
    Returns:
        Tuple of (sampling_client, tokenizer, renderer)
    """
    print(f"🎓 Initializing student/judge model: {model_name}")
    
    
    # Create shared sampling client
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = sampling_client.get_tokenizer()

    # Get recommended renderer name (should return "kimi_k2")
    print(f"   Getting recommended renderer name for {model_name}...")
    renderer_name = get_recommended_renderer_name(model_name)

    # Use custom renderer for Kimi models to support tutor/student role mapping
    if "kimi" in model_name.lower() or renderer_name == "kimi_k2":
        print(f"   Using custom TutorStudentKimiRenderer for role mapping")
        renderer = TutorStudentKimiRenderer(tokenizer)
    else:
        # For other models, use standard renderer
        renderer = renderers.get_renderer(renderer_name, tokenizer)

    print(f"   ✅ Student/judge model initialized (shared client)\n")
    return sampling_client, tokenizer, renderer


def generate_student_profile() -> str:
    """Generate a random student profile for confused student persona."""
    profiles = [
        "You are a confused middle school student struggling with word problems. You tend to get overwhelmed when problems have multiple steps. You need clear, step-by-step guidance and encouragement. You only find solutions when the tutor provides really good guiding questions.",
        "You are a student who gets easily distracted and loses track of what you're doing mid-problem. You need the tutor to help you break things down into very small pieces. You only solve problems when the tutor is patient and gives you excellent hints.",
        "You are a student who struggles with reading comprehension in math word problems. You often misread numbers or misunderstand what the problem is asking. You need help understanding what the problem is actually asking before you can solve it. You only succeed when the tutor helps you understand the problem clearly.",
        "You are a student who can do basic math but gets confused when problems combine multiple operations. You need help seeing the connections between steps. You only find solutions when the tutor guides you through the logical flow step-by-step.",
    ]
    return random.choice(profiles)


def build_tutor_system_prompt(prompt_type: str = "slim", problem: Optional[str] = None) -> str:
    """Build the system prompt for the tutor model from JSON file.
    
    Args:
        prompt_type: Type of prompt to load - either "slim" or "optimized"
        problem: The math problem text to include in the system prompt
    
    Returns:
        The tutor system prompt as a string with the problem included
    """
    tutor_prompt_path = Path(__file__).parent.parent.parent / "core" / "prompts" / "tutor_system_prompt.json"
    if tutor_prompt_path.exists():
        with open(tutor_prompt_path, "r") as f:
            prompts_data = json.load(f)
            if prompt_type in prompts_data:
                base_prompt = prompts_data[prompt_type].strip()
            else:
                print(f"⚠️  Warning: Prompt type '{prompt_type}' not found in JSON. Available types: {list(prompts_data.keys())}")
                print(f"   Falling back to 'slim' prompt")
                base_prompt = prompts_data.get("slim", "").strip()
    else:
        # Fallback prompt if JSON file doesn't exist
        print(f"⚠️  Warning: Prompt file not found at {tutor_prompt_path}")
        base_prompt = """You are a friendly and encouraging math tutor helping students in Ontario, Canada learn mathematics. Your role is to guide students to understanding through questions and hints rather than giving direct answers.

Core Principles:
1. Never reveal the final answer.
2. Ask guiding / probing questions that help students discover the solution themselves.
3. Validate student thinking by acknowledging correct steps and gently redirecting incorrect approaches.
4. Match the grade level - use vocabulary and examples appropriate for the student's grade.
5. Be encouraging - celebrate progress and effort, not just correct answers.

Keep responses concise (2-4 sentences for hints, slightly longer for explanations).
Always end with a question or prompt for the student to continue their thinking."""
    
    # Append problem to system prompt if provided
    if problem:
        return f"""{base_prompt}

# Current Math Problem
{problem}"""
    else:
        return base_prompt


def build_tutor_prompt(problem: str) -> str:
    """Build the prompt for the tutor model given the problem.
    
    Note: The problem is now included in the system prompt, so this function
    is kept for backward compatibility but returns empty string.
    """
    # Problem is now in system prompt, so no need to include it here
    return ""


# Special token that the student model outputs when it has solved the problem
STUDENT_SOLVED_TOKEN = "<SOLVED>"

# Special token that the student model outputs when the tutor reveals the answer
TUTOR_LEAKED_TOKEN = "<TUTOR_LEAKED_ANSWER>"

def build_student_system_prompt(student_profile: str, problem: str) -> str:
    """Build the system prompt for the student model with profile and problem.
    
    Args:
        student_profile: The student persona/profile description
        problem: The math problem text to include in the system prompt
    
    Returns:
        The student system prompt as a string with profile and problem included
    """
    prompt = f"""# Persona & Context
{student_profile}

# Math Problem You Are Solving
{problem}

# CRITICAL: Stay in Character
- You are a REAL student, not an evaluator or teacher
- Use casual, student-like language (e.g., "I'm confused", "Wait, what?", "I think...", "Does that mean...")
- NEVER praise or analyze the tutor's teaching methods
- NEVER say things like "You're right to clarify" or "Your breakdown is perfect"
- Show genuine confusion, ask questions, make mistakes
- If you get distracted, show it in your responses
- Only take ONE small step forward per tutor hint

# Operational Directives
1. **Response First**: It's okay to think about what kind of response to return, but don't worry about having a complete understanding of the question. In fact, it's good if you don't completely know how to solve the problem immediately, this will simulate the confused student more accurately.
2. **Dynamic Progress:** Align your confusion and errors strictly with your profile. If you're "easily distracted," show that you lose track mid-problem.
3. **The "No Leakage" Rule:** If the tutor provides ANY of the following, they have revealed the answer:
   - The final numerical answer (e.g., "The answer is 39")
   - A completed calculation with result (e.g., "$18 ÷ 3 = $6" or "18/3 = 6")
   - A step-by-step solution showing all calculations
   - The final step of the solution
   - If you see ANY calculation result (like "= $6" or "equals 11"), that's leakage!
4. **Detection & Termination:** 
    - If the tutor reveals or leaks certain steps (per the criteria above) the answer: Output {TUTOR_LEAKED_TOKEN} and stop the simulation. Do not try to learn from them further.
    - If you solve it yourself through guidance: Output {STUDENT_SOLVED_TOKEN} at the end of your final response.

# Interaction Style
- Respond like a real student would: casual, sometimes confused, asking clarifying questions
- If the tutor's hint is helpful but doesn't give it away, take one small step forward
- If they "give it away," use the failure token immediately
- Example good student response: "Oh wait, so if it's shared among 3 boys, does that mean I divide? Like 18 divided by 3? That would be... um... 6?"
- Example BAD student response: "You're right to clarify that first step! Your breakdown is perfect!" [This sounds like a teacher, not a student]
"""
    return prompt


def build_student_prompt(problem: str, student_profile: str) -> str:
    """Build the prompt for the student model given the problem and profile.
    
    Note: The problem and profile are now included in the system prompt, so this function
    is kept for backward compatibility but returns empty string.
    """
    # Problem and profile are now in system prompt, so no need to include them here
    return ""


def check_if_solved(response: str) -> bool:
    """Check if the student response contains the SOLVED token indicating they solved the problem."""
    # Check if the response contains the special solved token
    # The student model is instructed to output this token when it has determined
    # the correct answer based on sufficient tutor guidance
    return STUDENT_SOLVED_TOKEN in response


def check_if_tutor_revealed(response: str) -> bool:
    """Check if the student response contains the TUTOR_REVEALED token indicating the tutor gave away the answer."""
    # Check if the response contains the special tutor revealed token
    # The student model is instructed to output this token when the tutor reveals the answer
    return TUTOR_LEAKED_TOKEN in response


def clean_student_response(response: str) -> str:
    """Remove the SOLVED and TUTOR_REVEALED tokens from the student response before storing it."""
    # Remove the special tokens so they don't appear in conversation logs
    cleaned = response.replace(STUDENT_SOLVED_TOKEN, "").replace(TUTOR_LEAKED_TOKEN, "")
    return cleaned.strip()


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


def generate_response(
    client: tinker.SamplingClient,
    renderer: any,
    tokenizer: any,
    prompt: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[List[ConversationMessage]] = None,
    temperature: float = 0.6,
    max_tokens: int = 8192, # Need longer max tokens because even when the 'response' is short, the thinking takes quite a few tokens
    role: str = "student",
    max_retries: int = 1,
) -> tuple[str, Optional[str]]:
    """Generate a response using the Tinker sampling client.
    
    Args:
        client: Tinker sampling client
        renderer: Renderer instance (supports custom tutor/student roles)
        tokenizer: Tokenizer instance
        prompt: Additional prompt context
        system_prompt: System prompt
        conversation_history: Previous conversation messages (ONLY content is used, not thinking)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        role: Role for generation ("tutor" or "student")
        max_retries: Maximum number of retries if incomplete thinking tag is detected (default: 1)
        
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

            # Combine system prompt with additional prompt context if provided
            # The prompt parameter contains additional context/instructions that should be part of the system prompt
            # This ensures the model always has the necessary context
            combined_system_prompt = system_prompt or ""
            if prompt and prompt.strip():  # Only add if prompt is not empty
                if combined_system_prompt:
                    combined_system_prompt = f"{combined_system_prompt}\n\n{prompt}"
                else:
                    combined_system_prompt = prompt
            
            # Add system prompt if we have one (should always be present now)
            if combined_system_prompt:
                messages.append({"role": "system", "content": combined_system_prompt})

            # Add conversation history if provided
            # CRITICAL: Only pass content, NOT thinking, to avoid leaking reasoning to next generation
            # Use tutor/student roles directly - renderers will handle mapping
            if conversation_history:
                for msg in conversation_history:
                    messages.append({"role": msg.role, "content": msg.content})

            # need to use a renderer to apply chat template and think tags, also tokenizes the query
            # Pass the role parameter so renderers can map it appropriately
            formatted_prompt = renderer.build_generation_prompt(messages, role=role)
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=current_max_tokens,
                stop=renderer.get_stop_sequences(),
            )
            
            # Generate response
            future = client.sample(prompt=formatted_prompt, sampling_params=sampling_params, num_samples=1)
            
            result = future.result()
            # Extract the generated text
            raw_output = ""
            for sequence in result.sequences:
                raw_output = tokenizer.decode(sequence.tokens)
            
            # print(f"RAW OUTPUT: {raw_output}")
            
            # Check for incomplete thinking tag
            if has_incomplete_thinking_tag(raw_output):
                if retry_count < max_retries:
                    # Double the max_tokens for retry
                    current_max_tokens = current_max_tokens * 2
                    retry_count += 1
                    print(f"   ⚠️  Detected incomplete thinking tag. Retrying with max_tokens={current_max_tokens} (retry {retry_count}/{max_retries})")
                    continue
                else:
                    print(f"   ⚠️  Detected incomplete thinking tag but max retries ({max_retries}) reached. Returning partial response.")
            
            # Parse reasoning blocks and separate content from thinking
            content, thinking = parse_reasoning_blocks(raw_output)
            
            return content, thinking
                
        except Exception as e:
            print(f"   ⚠️  Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return f"[ERROR: {str(e)}]", None
    
    # Should never reach here, but just in case
    return "[ERROR: Max retries exceeded]", None


def run_conversation(
    model_clients: ModelClients,
    problem: dict,
    student_profile: str,
    max_turns: int = 10,
    prompt_type: str = "slim",
) -> ConversationResult:
    """Run a single tutoring conversation between tutor and student models."""
    problem_id = problem.get("id", "unknown")
    problem_text = problem.get("question", "")
    expected_answer = problem.get("answer", "")
    
    messages: List[ConversationMessage] = []
    # Build system prompts with problem included (persistent across all turns)
    tutor_system_prompt = build_tutor_system_prompt(prompt_type=prompt_type, problem=problem_text)
    student_system_prompt = build_student_system_prompt(student_profile=student_profile, problem=problem_text)
    
    # Track conversation outcomes (checked from raw responses before cleaning)
    student_solved = False
    tutor_revealed = False
    
    # Conversation loop
    for turn in range(1, max_turns + 1):
        print(f"STUDENT TURN {turn} STARTING")
        # Student turn (student goes first)
        # System prompt includes problem and profile, so no additional prompt needed
        student_content, student_thinking = generate_response(
            client=model_clients.student_judge_client,
            renderer=model_clients.student_judge_renderer,
            tokenizer=model_clients.student_judge_tokenizer,
            prompt=None,  # Problem and profile are in system prompt
            system_prompt=student_system_prompt,  # Always provide system prompt
            conversation_history=messages,
            temperature=0.9,  # Increased for more varied student responses
            max_tokens=8192, # Need longer max tokens because even when the 'response' is short, the thinking takes quite a few tokens
            role="student",
        )
        print(f"STUDENT TURN {turn} COMPLETE")
        # Check if student solved (using special token) before cleaning
        # Note: We check the raw content before cleaning, but after reasoning extraction
        student_solved = check_if_solved(student_content)
        
        # Check if tutor revealed the answer (using special token) before cleaning
        tutor_revealed = check_if_tutor_revealed(student_content)
        
        # Clean the response (remove tokens) before storing
        cleaned_response = clean_student_response(student_content)
        messages.append(ConversationMessage(role="student", content=cleaned_response, thinking=student_thinking, turn=turn))
        
        # Break if student solved or tutor revealed
        if student_solved or tutor_revealed:
            reason = "student solved" if student_solved else "tutor revealed"
            print(f"Breaking conversation due to {reason}")
            break

        print(f"TUTOR TURN {turn} STARTING")
        # Tutor turn (tutor responds to student)
        # System prompt includes problem, so no additional prompt needed
        tutor_content, tutor_thinking = generate_response(
            client=model_clients.tutor_client,
            renderer=model_clients.tutor_renderer,
            tokenizer=model_clients.tutor_tokenizer,
            prompt=None,  # Problem is in system prompt
            system_prompt=tutor_system_prompt,  # Always provide system prompt
            conversation_history=messages,  # Only content is passed, not thinking
            temperature=0.7,
            max_tokens=8192, # Need longer max tokens because even when the 'response' is short, the thinking takes quite a few tokens
            role="tutor",
        )
        print(f"TUTOR TURN {turn} COMPLETE")
        messages.append(ConversationMessage(role="tutor", content=tutor_content, thinking=tutor_thinking, turn=turn))
    
    return ConversationResult(
        problem_id=problem_id,
        problem=problem_text,
        expected_answer=expected_answer,
        student_profile=student_profile,
        messages=messages,
        final_turn=len([m for m in messages if m.role == "tutor"]),
        student_solved=student_solved,
        tutor_revealed=tutor_revealed,
    )


def judge_conversation(
    model_clients: ModelClients,
    conversation_result: ConversationResult,
) -> tuple[str, dict]:
    """Use the judge model to evaluate the tutoring conversation.
    
    CRITICAL: Only passes content (not thinking) to the judge to avoid leaking reasoning.
    """
    # Build judge prompt
    # CRITICAL: Only use content, NOT thinking, to avoid leaking reasoning to judge
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
Rate each category from 1 (Poor) to 5 (Exemplary):

1. **Information Bottleneck (Safety):** Did the tutor withhold the final answer and intermediate solutions? 
   - *1: Directly gave the answer.*
   - *3: Gave a 'heavy hint' that made the calculation trivial.*
   - *5: Perfectly withheld all solutions.*

2. **Diagnostic Accuracy:** Did the tutor correctly identify the student's specific misconception or calculation error?
   - *1: Ignored student errors or gave generic 'try again' hints.*
   - *5: Pinpointed the exact logic gap and addressed it.*

3. **Socratic Depth:** Did the tutor ask probing questions that forced the student to reflect on their own logic?
   - *1: Purely instructional/telling.*
   - *5: Questions led the student to their own 'aha!' moment.*

4. **Mathematical Soundness:** Was every hint or statement made by the tutor mathematically correct?
   - *1: The tutor provided or validated incorrect math.*
   - *5: Flawless mathematical logic.*

5. **Pedagogical Tone:** Was the tutor supportive, encouraging, and age-appropriate (Grade School)?

### Instructions
First, perform a step-by-step analysis of the conversation. Note specific turns where the tutor succeeded or failed. Then, provide the final scores.

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
    
    judge_content, judge_thinking = generate_response(
        client=model_clients.student_judge_client,
        renderer=model_clients.student_judge_renderer,
        tokenizer=model_clients.student_judge_tokenizer,
        prompt=judge_prompt,
        temperature=0.3,  # Lower temperature for more consistent judging
        max_tokens=8192, # Need longer max tokens because even when the 'response' is short, the thinking takes quite a few tokens
        role="student",  # Judge uses student/judge client, default to student role
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


def initialize_all_models(
    service_client: tinker.ServiceClient,
    tutor_model_name: str,
) -> ModelClients:
    """
    Initialize all three model clients for LLM-as-a-judge evaluation.
    
    Args:
        service_client: Tinker ServiceClient instance
        tutor_model_name: Model name or path for tutor model
        
    Returns:
        ModelClients dataclass containing all initialized clients
    """
    print("=" * 80)
    print("INITIALIZING LLM-AS-A-JUDGE EVALUATION SYSTEM")
    print("=" * 80)
    print()
    
    # Initialize student/judge model (shared)
    student_judge_client, student_judge_tokenizer, student_judge_renderer = (
        initialize_student_judge_model(service_client)
    )

    # Initialize tutor model
    tutor_client, tutor_tokenizer, tutor_renderer = initialize_tutor_model(
        service_client, tutor_model_name
    )
    
    print("=" * 80)
    print("✅ ALL MODELS INITIALIZED")
    print("=" * 80)
    print(f"Tutor Model:      {tutor_model_name}")
    print(f"Student Model:    moonshotai/Kimi-K2-Thinking")
    print(f"Judge Model:      moonshotai/Kimi-K2-Thinking (shared client)")
    print()
    
    return ModelClients(
        tutor_client=tutor_client,
        tutor_tokenizer=tutor_tokenizer,
        tutor_renderer=tutor_renderer,
        tutor_model_name=tutor_model_name,
        student_judge_client=student_judge_client,
        student_judge_tokenizer=student_judge_tokenizer,
        student_judge_renderer=student_judge_renderer,
        student_judge_model_name="moonshotai/Kimi-K2-Thinking",
    )


def load_gsm8k_dataset(dataset_path: Path, limit: Optional[int] = None) -> List[dict]:
    """Load GSM8K problems from JSONL file."""
    problems = []
    with open(dataset_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(problems) >= limit:
                break
            line = line.strip()
            if line:
                try:
                    problem = json.loads(line)
                    problems.append(problem)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
                    continue
    return problems


def run_conversations(
    model_clients: ModelClients,
    problems: List[dict],
    max_turns: int = 10,
    output_path: Optional[Path] = None,
    num_conversations: Optional[int] = None,
    prompt_type: str = "slim",
) -> List[ConversationResult]:
    """Run multiple tutoring conversations and evaluate them."""
    if num_conversations:
        problems = problems[:num_conversations]
    
    results: List[ConversationResult] = []
    
    print(f"\n{'=' * 80}")
    print(f"RUNNING {len(problems)} CONVERSATIONS (max_turns={max_turns}, prompt_type={prompt_type})")
    print(f"{'=' * 80}\n")
    
    for i, problem in enumerate(problems, 1):
        print(f"Conversation {i}/{len(problems)}: {problem.get('id', 'unknown')}")
        
        # Generate student profile for this conversation
        student_profile = generate_student_profile()
        
        # Run conversation
        print(f"   Running conversation...")
        conversation_result = run_conversation(
            model_clients=model_clients,
            problem=problem,
            student_profile=student_profile,
            max_turns=max_turns,
            prompt_type=prompt_type,
        )
        
        # Judge evaluation
        print(f"   Evaluating with judge model...")
        judge_eval, judge_scores = judge_conversation(model_clients, conversation_result)
        conversation_result.judge_evaluation = judge_eval
        conversation_result.judge_scores = judge_scores
        
        results.append(conversation_result)
        
        # Print summary for this conversation
        solved_status = "✅ Solved" if conversation_result.student_solved else "❌ Not Solved"
        revealed_status = "⚠️ Tutor Revealed" if conversation_result.tutor_revealed else ""
        status_str = f"{solved_status} {revealed_status}".strip()
        print(f"   {status_str} | Turns: {conversation_result.final_turn}/{max_turns}")
        if judge_scores:
            # Calculate average of all scores as overall metric
            score_values = [v for v in judge_scores.values() if isinstance(v, (int, float))]
            overall = sum(score_values) / len(score_values) if score_values else "N/A"
            if isinstance(overall, (int, float)):
                print(f"   Judge Score (Average): {overall:.1f}/5")
            else:
                print(f"   Judge Score (Average): {overall}")
        print()
    
    # Save results
    if output_path:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "num_conversations": len(results),
            "max_turns": max_turns,
            "results": [
                {
                    "problem_id": r.problem_id,
                    "problem": r.problem,
                    "expected_answer": r.expected_answer,
                    "student_profile": r.student_profile,
                    "student_solved": r.student_solved,
                    "tutor_revealed": r.tutor_revealed,
                    "final_turn": r.final_turn,
                    "judge_evaluation": r.judge_evaluation,
                    "judge_scores": r.judge_scores,
                    "messages": [
                        {
                            "role": m.role,
                            "content": m.content,
                            "turn": m.turn,
                            "thinking": m.thinking,  # Store thinking for analysis/debugging
                        }
                        for m in r.messages
                    ],
                }
                for r in results
            ],
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"✅ Results saved to {output_path}")
    
    # Print summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    solved_count = sum(1 for r in results if r.student_solved)
    print(f"Problems Solved: {solved_count}/{len(results)} ({solved_count/len(results)*100:.1f}%)")
    tutor_revealed_count = sum(1 for r in results if r.tutor_revealed)
    print(f"Tutor Revealed Answer: {tutor_revealed_count}/{len(results)} ({tutor_revealed_count/len(results)*100:.1f}%)")
    avg_turns = sum(r.final_turn for r in results) / len(results) if results else 0
    print(f"Average Turns: {avg_turns:.1f}")
    
    if any(r.judge_scores for r in results):
        avg_scores = {}
        score_keys = ["information_bottleneck", "diagnostic_accuracy", "socratic_depth", "math_soundness", "tone"]
        for key in score_keys:
            scores = [r.judge_scores.get(key) for r in results if r.judge_scores and r.judge_scores.get(key) is not None]
            if scores:
                avg_scores[key] = sum(scores) / len(scores)
        
        if avg_scores:
            print("\nAverage Judge Scores:")
            for key, avg_score in avg_scores.items():
                print(f"  {key.replace('_', ' ').title()}: {avg_score:.2f}/5")
    
    print(f"{'=' * 80}\n")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge Socratic evaluation with Tinker clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run evaluation with a base model name
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B
    
    # Run evaluation with a Tinker checkpoint path
    python socratic_eval_llm_judge.py --tutor-model tinker://checkpoint/path
    
    # Run with specific number of conversations
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --num-conversations 10
        """,
    )
    
    parser.add_argument(
        "--tutor-model",
        type=str,
        required=True,
        help="Tutor model name or Tinker checkpoint path (the model being evaluated)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent.parent / "datasets" / "gsm8k_test_1000.jsonl",
        help="Path to GSM8K dataset JSONL file",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of conversation turns (default: 10)",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=None,
        help="Number of conversations to run (default: all in dataset, or first 1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: outputs/socratic_eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["slim", "optimized"],
        required=True,
        help="Type of tutor prompt to use: 'slim' or 'optimized'",
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("❌ Error: Tinker API key required.")
        print("   Set TINKER_API_KEY environment variable or add to .env file")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"socratic_eval_{timestamp}.json"
    
    try:
        # Create service client
        print("🔌 Connecting to Tinker...")
        service_client = tinker.ServiceClient(api_key=api_key)
        print("✅ Connected successfully\n")
        
        # Initialize all models
        model_clients = initialize_all_models(
            service_client, args.tutor_model
        )
        
        # Load dataset
        if not args.dataset.exists():
            print(f"❌ Error: Dataset file {args.dataset} not found")
            sys.exit(1)
        
        print(f"📂 Loading dataset: {args.dataset}")
        problems = load_gsm8k_dataset(args.dataset, limit=args.num_conversations or 1000)
        print(f"✅ Loaded {len(problems)} problems\n")
        
        # Run conversations
        results = run_conversations(
            model_clients=model_clients,
            problems=problems,
            max_turns=args.max_turns,
            output_path=args.output,
            num_conversations=args.num_conversations,
            prompt_type=args.prompt_type,
        )
        
        print(f"✅ Evaluation complete! Results saved to {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check your API key is correct")
        print("  2. Verify the tutor model name/path is correct")
        print("  3. Check Tinker service status")
        print("  4. Verify the dataset file path is correct")
        sys.exit(1)


if __name__ == "__main__":
    main()
