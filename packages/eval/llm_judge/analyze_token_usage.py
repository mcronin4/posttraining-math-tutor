#!/usr/bin/env python3
"""
Analyze token usage from LLM judge evaluation results.

Computes token usage statistics for tutor and student messages,
including both thinking and content tokens.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tinker_cookbook import tokenizer_utils
except ImportError:
    print("❌ Error: tinker_cookbook not found.")
    print("Please ensure tinker_cookbook is installed.")
    sys.exit(1)


def get_tokenizer_for_model(model_name: str):
    """Get tokenizer for a given model name."""
    # Try to infer model name from directory name
    # e.g., "Qwen_Qwen3-8B_slim" -> "Qwen/Qwen3-8B"
    if "Qwen" in model_name or "qwen" in model_name:
        # Try common Qwen model names
        if "8B" in model_name:
            return tokenizer_utils.get_tokenizer("Qwen/Qwen3-8B")
        elif "235B" in model_name:
            return tokenizer_utils.get_tokenizer("Qwen/Qwen3-235B-A22B-Instruct")
        else:
            return tokenizer_utils.get_tokenizer("Qwen/Qwen3-8B")  # Default fallback
    else:
        # Try the model name as-is
        return tokenizer_utils.get_tokenizer(model_name)


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Warning: Error counting tokens: {e}")
        return 0


def analyze_token_usage(results_path: Path, tutor_model_name: str = None, student_model_name: str = None):
    """Analyze token usage from results.json file."""
    
    # Load results
    print(f"Loading results from: {results_path}")
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Infer model names from directory if not provided
    if not tutor_model_name or not student_model_name:
        dir_name = results_path.parent.name
        # Extract model name from directory (e.g., "socratic_eval_Qwen_Qwen3-8B_slim_20260120_203056")
        if "Qwen" in dir_name or "qwen" in dir_name:
            if "8B" in dir_name:
                tutor_model_name = tutor_model_name or "Qwen/Qwen3-8B"
                student_model_name = student_model_name or "Qwen/Qwen3-8B"
            elif "235B" in dir_name:
                tutor_model_name = tutor_model_name or "Qwen/Qwen3-235B-A22B-Instruct"
                student_model_name = student_model_name or "Qwen/Qwen3-235B-A22B-Instruct"
    
    # Default fallback
    tutor_model_name = tutor_model_name or "Qwen/Qwen3-8B"
    student_model_name = student_model_name or "Qwen/Qwen3-8B"
    
    print(f"Using tutor tokenizer: {tutor_model_name}")
    print(f"Using student tokenizer: {student_model_name}")
    
    # Initialize tokenizers
    try:
        tutor_tokenizer = get_tokenizer_for_model(tutor_model_name)
        student_tokenizer = get_tokenizer_for_model(student_model_name)
    except Exception as e:
        print(f"❌ Error initializing tokenizers: {e}")
        sys.exit(1)
    
    # Statistics
    tutor_total_tokens = 0
    student_total_tokens = 0
    tutor_turns = 0
    student_turns = 0
    total_conversations = len(data.get("results", []))
    
    # Process each conversation
    for result in data.get("results", []):
        messages = result.get("messages", [])
        
        for message in messages:
            role = message.get("role", "")
            
            # Skip judge messages
            if role == "judge":
                continue
            
            # Get content and thinking
            content = message.get("content", "")
            thinking = message.get("thinking", "")
            
            # Count tokens
            if role == "tutor":
                content_tokens = count_tokens(content, tutor_tokenizer)
                thinking_tokens = count_tokens(thinking, tutor_tokenizer)
                total_message_tokens = content_tokens + thinking_tokens
                tutor_total_tokens += total_message_tokens
                tutor_turns += 1
            elif role == "student":
                content_tokens = count_tokens(content, student_tokenizer)
                thinking_tokens = count_tokens(thinking, student_tokenizer)
                total_message_tokens = content_tokens + thinking_tokens
                student_total_tokens += total_message_tokens
                student_turns += 1
    
    # Calculate averages
    tutor_avg_tokens_per_turn = tutor_total_tokens / tutor_turns if tutor_turns > 0 else 0
    student_avg_tokens_per_turn = student_total_tokens / student_turns if student_turns > 0 else 0
    
    # Calculate average turns per conversation
    # Each conversation has alternating tutor/student turns
    # We can approximate by dividing total turns by number of conversations
    tutor_avg_turns_per_conv = tutor_turns / total_conversations if total_conversations > 0 else 0
    student_avg_turns_per_conv = student_turns / total_conversations if total_conversations > 0 else 0
    
    # Display results
    print("\n" + "=" * 80)
    print("TOKEN USAGE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal Conversations: {total_conversations}")
    print(f"\n{'─' * 80}")
    print("TUTOR STATISTICS")
    print(f"{'─' * 80}")
    print(f"  Total Tokens:           {tutor_total_tokens:,}")
    print(f"  Total Turns:            {tutor_turns:,}")
    print(f"  Average Tokens/Turn:    {tutor_avg_tokens_per_turn:.2f}")
    print(f"  Average Turns/Conv:     {tutor_avg_turns_per_conv:.2f}")
    
    print(f"\n{'─' * 80}")
    print("STUDENT STATISTICS")
    print(f"{'─' * 80}")
    print(f"  Total Tokens:           {student_total_tokens:,}")
    print(f"  Total Turns:             {student_turns:,}")
    print(f"  Average Tokens/Turn:    {student_avg_tokens_per_turn:.2f}")
    print(f"  Average Turns/Conv:     {student_avg_turns_per_conv:.2f}")
    
    print(f"\n{'─' * 80}")
    print("COMBINED STATISTICS")
    print(f"{'─' * 80}")
    print(f"  Total Tokens (Tutor + Student): {tutor_total_tokens + student_total_tokens:,}")
    print(f"  Total Turns (Tutor + Student):  {tutor_turns + student_turns:,}")
    print(f"  Average Tokens/Turn (Combined): {(tutor_total_tokens + student_total_tokens) / (tutor_turns + student_turns) if (tutor_turns + student_turns) > 0 else 0:.2f}")
    print("=" * 80 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_token_usage.py <results.json> [tutor_model] [student_model]")
        print("\nExample:")
        print("  python analyze_token_usage.py llm_judge_outputs/socratic_eval_Qwen_Qwen3-8B_slim_20260120_203056/results.json")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"❌ Error: File not found: {results_path}")
        sys.exit(1)
    
    tutor_model = sys.argv[2] if len(sys.argv) > 2 else None
    student_model = sys.argv[3] if len(sys.argv) > 3 else None
    
    analyze_token_usage(results_path, tutor_model, student_model)


if __name__ == "__main__":
    main()
