#!/usr/bin/env python3
"""
Test script to verify the parsing logic for gsm8k_synthetic_errors.py
Tests the parsing logic with sample outputs that simulate what the model might generate.
"""

import re
import sys
from pathlib import Path

# Add parent directory to path to import parse_reasoning_blocks
script_dir = Path(__file__).resolve().parent
packages_dir = script_dir.parent.parent
eval_dir = packages_dir / "eval"
if str(eval_dir) not in sys.path:
    sys.path.insert(0, str(eval_dir))

from utils import parse_reasoning_blocks


def is_prompt_text(text: str) -> bool:
    """Check if text is just repeating the prompt instructions."""
    if not text or len(text) < 20:
        return False
    text_lower = text.lower()
    prompt_phrases = [
        "the user wants me to",
        "act as a struggling grade school student",
        "provide a wrong first step",
        "give a one-sentence wrong first step",
        "i need to act as",
        "i should act as",
    ]
    # Check if text starts with or heavily contains prompt phrases
    return any(
        text_lower.startswith(phrase) or 
        (phrase in text_lower and len(text_lower) < 200)  # Short text with prompt = likely just prompt
        for phrase in prompt_phrases
    )


def parse_error_output(raw_output: str) -> str:
    """Parse the raw model output to extract the actual error."""
    # Check if output was truncated (no closing </think> tag means likely truncated)
    is_truncated = "<think>" in raw_output.lower() and "</think>" not in raw_output.lower()
    
    # Use parse_reasoning_blocks to remove thinking tags and extract content
    content, thinking = parse_reasoning_blocks(raw_output)
    
    # The content should be what comes after </think> tags
    output = content.strip()
    
    # If no content after </think>, check if thinking contains actual error (not just prompt)
    if not output and thinking:
        thinking_text = thinking.strip()
        
        # If truncated, the thinking block might contain the actual error after the prompt
        # Look for content that comes after common prompt endings
        if is_truncated or is_prompt_text(thinking_text):
            # Try to find where the prompt ends and actual content begins
            # Look for patterns that indicate the end of prompt repetition
            prompt_end_markers = [
                r"act as a struggling grade school student[^.]*\.\s*",
                r"provide a wrong first step[^.]*\.\s*",
                r"give a one-sentence wrong first step[^.]*\.\s*",
                r"the user wants me to[^.]*\.\s*",
                r"problem:\s*[^.]*\.\s*",
            ]
            
            # Try splitting on prompt patterns to find content after
            for pattern in prompt_end_markers:
                parts = re.split(pattern, thinking_text, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    extracted = parts[1].strip()
                    # Remove any remaining prompt-like prefixes
                    extracted = re.sub(r'^(wrong first step|problem|answer|step|i should|i need to):\s*', '', extracted, flags=re.IGNORECASE)
                    if extracted and len(extracted) > 15 and not is_prompt_text(extracted):
                        output = extracted
                        break
            
            # If still no output, try to find the last sentence that doesn't look like a prompt
            if not output:
                sentences = re.split(r'[.!?]\s+', thinking_text)
                # Look for sentences that are substantial and don't look like prompts
                for sentence in reversed(sentences):
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 15 and not is_prompt_text(sentence):
                        # Check if it's actually an error (not just prompt continuation)
                        if not any(phrase in sentence.lower()[:50] for phrase in ["act as", "provide a", "give a", "the user"]):
                            output = sentence
                            break
        elif not is_prompt_text(thinking_text) and len(thinking_text) > 15:
            # Use thinking if it looks like actual content (not prompt)
            output = thinking_text
    
    # Also check if content itself is prompt-like
    if output and is_prompt_text(output):
        # Try to clean it up
        prompt_patterns = [
            r"^(act as a struggling grade school student|provide a wrong first step|give a one-sentence wrong first step)[^.]*\.\s*",
        ]
        for pattern in prompt_patterns:
            cleaned = re.sub(pattern, '', output, flags=re.IGNORECASE)
            if cleaned.strip() and len(cleaned.strip()) > 15:
                output = cleaned.strip()
                break
        # If still looks like prompt, reject it
        if is_prompt_text(output):
            output = ""
    
    # Clean up output: remove any remaining prompt artifacts
    if output:
        # Remove common prefixes that might be left over
        output = re.sub(r'^(wrong first step|problem|answer|step):\s*', '', output, flags=re.IGNORECASE)
        output = output.strip()
    
    # Take only the first sentence
    if output:
        sentences = re.split(r'[.!?]\s+', output)
        if sentences:
            first_sentence = sentences[0].strip()
            if first_sentence and len(first_sentence) > 10 and not is_prompt_text(first_sentence):
                # Ensure it ends with punctuation
                if not first_sentence[-1] in '.!?':
                    return first_sentence + "."
                return first_sentence
    
    # Fallback: return cleaned output (truncated to first sentence if needed)
    if output and not is_prompt_text(output) and len(output) > 10:
        # Extract first sentence
        sentences = re.split(r'[.!?]\s+', output)
        if sentences:
            first = sentences[0].strip()
            if first and not first[-1] in '.!?':
                return first + "."
            return first
        return output.split('.')[0].strip() + "." if output else "I'm not sure where to start."
    else:
        return "I'm not sure where to start."


def test_cases():
    """Test various cases that might occur."""
    test_cases = [
        # Case 1: Truncated output with prompt in thinking
        {
            "input": "<think>The user wants me to act as a struggling grade school student and provide a wrong first step for the given math problem.",
            "expected_contains": "I'm not sure",
            "description": "Truncated output with only prompt"
        },
        # Case 2: Truncated output with prompt + partial error
        {
            "input": "<think>The user wants me to act as a struggling grade school student and provide a wrong first step for the given math problem. I should multiply 20 by 50",
            "expected_contains": "multiply",
            "description": "Truncated output with prompt + partial error"
        },
        # Case 3: Complete output with prompt in thinking, error after
        {
            "input": "<think>The user wants me to act as a struggling grade school student and provide a wrong first step.</think>I should add 20 and 50 together.",
            "expected_contains": "add 20 and 50",
            "description": "Complete output with error after </think>"
        },
        # Case 4: Error inside thinking block (truncated)
        {
            "input": "<think>The user wants me to act as a struggling grade school student. I think I should divide 100 by 2",
            "expected_contains": "divide",
            "description": "Error in thinking block after prompt"
        },
        # Case 5: Just prompt, no error
        {
            "input": "<think>Act as a struggling grade school student and provide a wrong first step for this problem.</think>",
            "expected_contains": "I'm not sure",
            "description": "Only prompt, no actual error"
        },
    ]
    
    print("Testing parsing logic...\n")
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        result = parse_error_output(test["input"])
        passed = test["expected_contains"].lower() in result.lower()
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"Test {i}: {test['description']}")
        print(f"  Input: {test['input'][:100]}...")
        print(f"  Expected to contain: '{test['expected_contains']}'")
        print(f"  Got: '{result}'")
        print(f"  {status}\n")
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed


if __name__ == "__main__":
    success = test_cases()
    sys.exit(0 if success else 1)
