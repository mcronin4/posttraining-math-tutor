"""Prompt building functions for tutor and student models."""

import json
import random
from pathlib import Path
from typing import List, Optional

# Special token that the student model outputs when it has solved the problem
STUDENT_SOLVED_TOKEN = "<SOLVED>"


def get_all_student_profiles() -> List[str]:
    """Get all available student profiles."""
    return [
        "You are a middle school student working on word problems. You sometimes get overwhelmed when problems have multiple steps. You need clear, step-by-step guidance. You can solve problems when the tutor provides good guiding questions.",
        "You are a student who sometimes loses track of what you're doing mid-problem. You need the tutor to help you break things down into small pieces. You can solve problems when the tutor is patient and gives helpful hints.",
        "You are a student who sometimes misreads numbers or misunderstands what word problems are asking. You need help understanding what the problem is actually asking before you can solve it. You can succeed when the tutor helps you understand the problem clearly.",
        "You are a student who can do basic math but gets confused when problems combine multiple operations. You need help seeing the connections between steps. You can find solutions when the tutor guides you through the logical flow step-by-step.",
    ]


def build_tutor_system_prompt(prompt_type: str = "slim", problem: Optional[str] = None) -> str:
    """Build the system prompt for the tutor model from JSON file.
    
    Args:
        prompt_type: Type of prompt to load - either "slim", "optimized", or "unprompted"
        problem: The math problem text to include in the system prompt
    
    Returns:
        The tutor system prompt as a string with the problem included.
        For "unprompted", returns only the problem text without any socratic instructions.
    """
    # Handle "unprompted" case - return only the problem, no socratic instructions
    if prompt_type == "unprompted":
        if problem:
            return f"""
# Current Math Problem
{problem}"""
        else:
            return ""
    
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
- Keep responses CONCISE - typically 1-2 sentences. You can be confused without being verbose.
- NEVER praise or analyze the tutor's teaching methods
- NEVER say things like "You're right to clarify" or "Your breakdown is perfect"
- Show genuine confusion, ask questions, make mistakes - but do so briefly
- Only take ONE small step forward per tutor hint

# Operational Directives
1. **Response First**: It's okay to think about what kind of response to return, but don't worry about having a complete understanding of the question. In fact, it's good if you don't completely know how to solve the problem immediately, this will simulate the confused student more accurately.
2. **Dynamic Progress:** Align your confusion and errors strictly with your profile. If you're "easily distracted," you might briefly lose track, but keep your response short.
3. **Detection & Termination:** 
    - If you solve it yourself through guidance: Output {STUDENT_SOLVED_TOKEN} at the end of your final response.
    
    **IMPORTANT - When to Use {STUDENT_SOLVED_TOKEN}:**
    - If you calculate the final answer (e.g., you say "So the answer is 39" or "I think it's 39?"), you MUST output {STUDENT_SOLVED_TOKEN} at the end of that response.
    - Even if you're uncertain (e.g., "Is that 39? Is that the final answer?"), if you've stated what you believe is the final answer, output {STUDENT_SOLVED_TOKEN}.
    - Do NOT wait for the tutor to confirm - if you've stated the final answer, mark it as solved immediately.
    - Examples of when to use {STUDENT_SOLVED_TOKEN}:
      * "So he would have $39 after tripling it? Is that the final answer?{STUDENT_SOLVED_TOKEN}"
      * "I think the answer is 39.{STUDENT_SOLVED_TOKEN}"
      * "So the final answer is 2, right?{STUDENT_SOLVED_TOKEN}"

# Interaction Style
- Respond like a real student would: casual, sometimes confused, asking clarifying questions - but keep it brief
- If the tutor's hint is helpful, take one small step forward
- Example good student response: "Oh wait, so if it's shared among 3 boys, does that mean I divide? Like 18 divided by 3? That would be... um... 6?"
"""
    return prompt
