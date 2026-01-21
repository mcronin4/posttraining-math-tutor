#!/usr/bin/env python3
"""
Batch evaluation script - runs socratic_eval_llm_judge.py for multiple models sequentially.

This script allows you to "set and forget" by running multiple model evaluations
one after another without manual intervention. Runs both models with both prompt types
(slim and optimized) for a total of 4 evaluations.

Usage:
    uv run python run_batch_eval.py
    uv run python run_batch_eval.py --num-conversations 100
    uv run python run_batch_eval.py --debug
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_evaluation(tutor_model: str, prompt_type: str, num_conversations: int = None, 
                   checkpoint_interval: int = None, debug: bool = False):
    """Run socratic_eval_llm_judge.py for a given tutor model."""
    
    script_path = Path(__file__).parent / "socratic_eval_llm_judge.py"
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--tutor-model", tutor_model,
        "--prompt-type", prompt_type,
    ]
    
    if num_conversations:
        cmd.extend(["--num-conversations", str(num_conversations)])
    
    if checkpoint_interval:
        cmd.extend(["--checkpoint-interval", str(checkpoint_interval)])
    
    if debug:
        cmd.append("--debug")
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting evaluation for: {tutor_model}")
    print(f"   Prompt type: {prompt_type}")
    if num_conversations:
        print(f"   Number of conversations: {num_conversations}")
    print(f"{'='*80}\n")
    
    # Run the evaluation
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Completed evaluation for: {tutor_model} ({prompt_type})\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running evaluation for {tutor_model} ({prompt_type}): {e}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation for {tutor_model} ({prompt_type}) interrupted by user")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch evaluations for multiple models sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (both models, both prompt types)
    uv run python run_batch_eval.py
    
    # Run with custom number of conversations
    uv run python run_batch_eval.py --num-conversations 100
    
    # Run with debug output
    uv run python run_batch_eval.py --debug
        """,
    )
    
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=None,
        help="Number of conversations to run for each model/prompt combination (default: all in dataset, or first 1000)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save checkpoint every N conversations (default: 50)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug information",
    )
    
    args = parser.parse_args()
    
    # Models to evaluate (in order)
    models = [
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "moonshotai/Kimi-K2-Thinking",
    ]
    
    # Prompt types to use
    prompt_types = ["slim", "optimized"]
    
    # Generate all combinations
    evaluations = []
    for model in models:
        for prompt_type in prompt_types:
            evaluations.append((model, prompt_type))
    
    print(f"\n{'#'*80}")
    print(f"# Batch Evaluation Script")
    print(f"# Running {len(models)} models × {len(prompt_types)} prompt types = {len(evaluations)} evaluations")
    print(f"# Models: {', '.join(models)}")
    print(f"# Prompt types: {', '.join(prompt_types)}")
    if args.num_conversations:
        print(f"# Number of conversations per evaluation: {args.num_conversations}")
    print(f"{'#'*80}\n")
    
    # Track results
    results = {}
    
    # Run each evaluation sequentially
    for i, (model, prompt_type) in enumerate(evaluations, 1):
        key = f"{model} ({prompt_type})"
        print(f"\n[{i}/{len(evaluations)}] Processing: {key}")
        
        success = run_evaluation(
            tutor_model=model,
            prompt_type=prompt_type,
            num_conversations=args.num_conversations,
            checkpoint_interval=args.checkpoint_interval,
            debug=args.debug,
        )
        
        results[key] = success
        
        if not success:
            print(f"\n⚠️  Evaluation failed for {key}")
            response = input("Continue with next evaluation? (y/n): ")
            if response.lower() != 'y':
                print("\n❌ Stopping batch evaluation")
                break
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 Batch Evaluation Summary")
    print(f"{'='*80}")
    for key, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {status}: {key}")
    print(f"{'='*80}\n")
    
    # Exit with error code if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
