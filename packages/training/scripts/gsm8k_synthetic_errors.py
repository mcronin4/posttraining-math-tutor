#!/usr/bin/env python3
"""
Generate Synthetic Errors for GSM8K Training Set

Loads the GSM8K training set with 3000 unique questions, then for each question,
calls a Tinker SamplingClient to generate a one-sentence wrong first step.
Saves the output as JSONL of Seed Contexts, similar to the MathDial format.

Usage:
    uv run python scripts/gsm8k_synthetic_errors.py
    uv run python scripts/gsm8k_synthetic_errors.py --model Qwen/Qwen3-8B --max-tokens 256
    uv run python scripts/gsm8k_synthetic_errors.py --output gsm8k_synthetic_errors.jsonl --limit 3000
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path to import from eval package
script_dir = Path(__file__).resolve().parent
packages_dir = script_dir.parent.parent.parent
if str(packages_dir) not in sys.path:
    sys.path.insert(0, str(packages_dir))

# Import Tinker SDK
try:
    import tinker
    from tinker.types import SamplingParams
    from tinker_cookbook import renderers, tokenizer_utils
    from tinker_cookbook.model_info import get_recommended_renderer_name
except ImportError:
    print("❌ Error: Tinker SDK or tinker_cookbook not found.")
    print("   Install with: uv add tinker tinker-cookbook")
    sys.exit(1)

# Try to import custom renderer for Kimi models
try:
    from eval.llm_judge.custom_renderers import TutorStudentKimiRenderer
except ImportError:
    # Fallback for when script is run directly
    eval_dir = packages_dir / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    try:
        from llm_judge.custom_renderers import TutorStudentKimiRenderer
    except ImportError:
        TutorStudentKimiRenderer = None



def load_gsm8k_questions(limit: int = 3000, split: str = "train") -> List[dict]:
    """
    Load GSM8K training set questions.
    
    Reuses the same loading pattern as download_datasets.py for consistency.
    
    Args:
        limit: Maximum number of unique questions to load
        split: Dataset split to use (default: "train")
        
    Returns:
        List of question dictionaries with id, question, and answer fields
    """
    print(f"📥 Loading GSM8K {split} set (limit: {limit})...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: HuggingFace datasets not installed.")
        print("   Install with: uv add datasets")
        sys.exit(1)
    
    # Use the same revision pinning as download_datasets.py for reproducibility
    revision = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"
    if revision:
        print(f"   📌 Pinned to revision: {revision[:8]}...")
    
    # Load dataset with revision pinning
    load_kwargs = {"split": split}
    if revision:
        load_kwargs["revision"] = revision
    dataset = load_dataset("gsm8k", "main", **load_kwargs)
    
    # Convert to list and sort deterministically by question text
    items = list(dataset)
    items.sort(key=lambda x: x['question'])
    
    # Limit if specified
    if limit is not None and limit < len(items):
        items = items[:limit]
        print(f"📋 Limited to first {limit} examples (sorted by question)")
    
    # Convert to our format
    # GSM8K answer format: "full solution text #### final_answer"
    questions = []
    for i, item in enumerate(items):
        answer_parts = item["answer"].split("####")
        full_solution = answer_parts[0].strip() if len(answer_parts) > 1 else item["answer"].strip()
        final_answer = answer_parts[-1].strip()
        
        questions.append({
            "id": f"gsm8k_{i:05d}",
            "question": item["question"],
            "full_solution": full_solution,
            "final_answer": final_answer,
        })
    
    print(f"✅ Loaded {len(questions)} questions\n")
    return questions


def initialize_client(model_name: str, api_key: Optional[str] = None) -> tuple:
    """
    Initialize SamplingClient for the specified model.
    
    Args:
        model_name: Name of the model to use (e.g., "moonshotai/Kimi-K2-Thinking", "Qwen/Qwen3-8B")
        api_key: Tinker API key (or set TINKER_API_KEY env var)
        
    Returns:
        Tuple of (sampling_client, tokenizer, renderer)
    """
    print(f"🔌 Initializing client for model: {model_name}")
    
    # Get API key
    if not api_key:
        api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("❌ Error: Tinker API key required.")
        print("   Set TINKER_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create service client
    service_client = tinker.ServiceClient(api_key=api_key)
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    
    # Get tokenizer
    tokenizer = sampling_client.get_tokenizer()
    
    # Get recommended renderer name
    renderer_name = get_recommended_renderer_name(model_name)
    print(f"   Renderer: {renderer_name}")
    
    # Use custom renderer for Kimi models if available
    if TutorStudentKimiRenderer and ("kimi" in model_name.lower() or renderer_name == "kimi_k2"):
        print(f"   Using custom TutorStudentKimiRenderer")
        renderer = TutorStudentKimiRenderer(tokenizer)
    else:
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    print(f"   ✅ Client initialized\n")
    return sampling_client, tokenizer, renderer


async def generate_wrong_first_step(
    client: tinker.SamplingClient,
    renderer: any,
    tokenizer: any,
    question: str,
    temperature: float = 0.9,
    max_tokens: int = 256,
) -> str:
    """
    Generate a one-sentence wrong first step for a math problem.
    
    Args:
        client: Tinker SamplingClient
        renderer: Renderer instance
        tokenizer: Tokenizer instance
        question: The math problem question
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated wrong first step as a string
    """
    prompt = f"Act as a struggling grade school student. Give a one-sentence wrong first step for this problem.\n\nProblem: {question}\n\nWrong first step:"
    
    # Build messages
    messages = [{"role": "user", "content": prompt}]
    
    # Format prompt using renderer
    formatted_prompt = renderer.build_generation_prompt(messages, role="student")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    
    # Generate response
    result = await client.sample_async(
        prompt=formatted_prompt,
        sampling_params=sampling_params,
        num_samples=1
    )
    
    # Extract the generated text
    raw_output = ""
    for sequence in result.sequences:
        raw_output = tokenizer.decode(sequence.tokens)
    
    # Simple think tag removal logic
    has_opening_tag = re.search(r'<think>', raw_output, re.IGNORECASE)
    has_closing_tag = re.search(r'</think>', raw_output, re.IGNORECASE)
    
    if has_opening_tag and has_closing_tag:
        # Both tags present: remove all think blocks (including tags)
        output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE)
    elif has_closing_tag:
        # Only closing tag: remove everything before the last closing tag
        output = raw_output[raw_output.lower().rfind('</think>') + len('</think>'):]
    elif has_opening_tag:
        # Only opening tag: log warning and return content with tag still in it
        logging.warning(f"Truncated output detected (opening <think> tag without closing tag). Raw output: {raw_output[:200]}...")
        output = raw_output
    else:
        # No think tags: use output as-is
        output = raw_output
    
    # Clean up whitespace
    output = output.strip()
    
    # Take only the first sentence
    if output:
        sentences = re.split(r'[.!?]\s+', output)
        if sentences:
            first_sentence = sentences[0].strip()
            if first_sentence:
                # Ensure it ends with punctuation
                if not first_sentence[-1] in '.!?':
                    return first_sentence + "."
                return first_sentence
    
    # Fallback
    return output if output else "I'm not sure where to start."


async def process_questions(
    questions: List[dict],
    client: tinker.SamplingClient,
    renderer: any,
    tokenizer: any,
    output_file: Path,
    batch_size: int = 10,
    temperature: float = 0.9,
    max_tokens: int = 256,
) -> None:
    """
    Process all questions and generate synthetic errors.
    
    Args:
        questions: List of question dictionaries
        client: Tinker SamplingClient
        renderer: Renderer instance
        tokenizer: Tokenizer instance
        output_file: Path to output JSONL file
        batch_size: Number of questions to process concurrently
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    print(f"🔄 Processing {len(questions)} questions...")
    print(f"   Batch size: {batch_size}")
    print(f"   Temperature: {temperature}")
    print(f"   Max tokens: {max_tokens}\n")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(questions) + batch_size - 1) // batch_size
            
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} questions)...")
            
            # Process batch concurrently
            tasks = [
                generate_wrong_first_step(
                    client, renderer, tokenizer, q["question"],
                    temperature=temperature, max_tokens=max_tokens
                )
                for q in batch
            ]
            errors = await asyncio.gather(*tasks)
            
            # Write results
            for q, error in zip(batch, errors):
                seed_context = {
                    "id": q["id"],
                    "question": q["question"],
                    "expected_answer": q["full_solution"],  # Full solution for MathDial format
                    "initial_error": error,
                }
                f.write(json.dumps(seed_context, ensure_ascii=False) + "\n")
                f.flush()
            
            print(f"      ✅ Completed batch {batch_num}/{total_batches}")
    
    print(f"\n✅ Generated {len(questions)} synthetic errors")
    print(f"📁 Saved to: {output_file}")


async def main_async():
    """Async main function."""
    # Default output path: packages/data/gsm8k_synthetic_errors.jsonl
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent  # packages/
    default_output = packages_dir / "data" / "gsm8k_synthetic_errors.jsonl"
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic errors for GSM8K training set"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"Output JSONL path (default: {default_output})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Number of unique questions to process (default: 3000)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions to process concurrently (default: 10)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Tinker API key (or set TINKER_API_KEY env var)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model to use for generation (default: Qwen/Qwen3-8B). "
             "Examples: 'Qwen/Qwen3-8B', 'moonshotai/Kimi-K2-Thinking', 'Qwen/Qwen3-235B-A22B-Instruct-2507'",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256). Increase if using thinking models.",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("GSM8K SYNTHETIC ERROR GENERATION")
    print("=" * 80)
    print()
    
    # Load questions
    questions = load_gsm8k_questions(limit=args.limit, split=args.split)
    
    if not questions:
        print("❌ Error: No questions loaded")
        sys.exit(1)
    
    # Initialize client for specified model
    client, tokenizer, renderer = initialize_client(model_name=args.model, api_key=args.api_key)
    
    # Process questions and generate errors
    await process_questions(
        questions=questions,
        client=client,
        renderer=renderer,
        tokenizer=tokenizer,
        output_file=args.output,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"\nOutput file: {args.output}")
    print(f"Total seed contexts: {len(questions)}")


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
