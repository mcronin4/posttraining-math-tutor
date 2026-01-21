#!/usr/bin/env python3
"""
Helper script to view tutor system prompts in a readable format.
This makes it easier to review the prompts without dealing with \n escape sequences.
"""

import json
from pathlib import Path

def view_prompts():
    """Load and display prompts in a readable format."""
    prompt_file = Path(__file__).parent / "tutor_system_prompt.json"
    
    if not prompt_file.exists():
        print(f"Error: {prompt_file} not found")
        return
    
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    
    print("=" * 80)
    for prompt_type, content in prompts.items():
        print(f"\n# {prompt_type.upper()} PROMPT")
        print("=" * 80)
        print(content)
        print("\n" + "=" * 80)

if __name__ == "__main__":
    view_prompts()
