#!/usr/bin/env python3
"""
Tutoring Rubric Evaluation

Evaluates tutoring quality based on pedagogical rubrics:
- Does the response avoid revealing the final answer (when configured)?
- Does it follow the Socratic method (asking questions, suggesting steps)?
- Is the response going in the correct direction?

Usage:
    python eval_tutor.py --dataset samples/tutor_samples.jsonl \
                         --endpoint http://localhost:8000/chat \
                         --output outputs/tutor_results.json

Rubric Criteria:
    - no_answer_reveal: Response doesn't directly give the answer
    - socratic_step: Includes guiding question or suggests next step
    - correct_direction: Response is mathematically sound (placeholder)
    - appropriate_level: Uses grade-appropriate language
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from tqdm import tqdm


# =============================================================================
# Rubric Heuristics
# =============================================================================

# Common patterns that indicate a direct answer
ANSWER_PATTERNS = [
    r"the answer is\s+\d",
    r"equals\s+\d+\s*$",
    r"=\s*\d+\s*$",
    r"the result is\s+\d",
    r"that gives us\s+\d+\s*$",
    r"so\s+\d+\s+is\s+(the\s+)?answer",
]

# Patterns indicating Socratic approach
SOCRATIC_PATTERNS = [
    r"\?$",  # Ends with question
    r"\?[\s\n]*$",  # Ends with question (with trailing whitespace)
    r"what do you (think|notice)",
    r"can you (try|tell|explain|think)",
    r"what would happen if",
    r"how would you",
    r"why do you think",
    r"what's (your|the) next step",
    r"let's (think|consider|look)",
    r"try to",
    r"consider",
    r"think about",
]

# Grade-level vocabulary indicators
ELEMENTARY_VOCAB = ["count", "add", "take away", "groups", "share", "same as"]
MIDDLE_VOCAB = ["equation", "variable", "fraction", "ratio", "solve"]
HIGH_VOCAB = ["function", "derivative", "polynomial", "theorem", "proof"]


def check_no_answer_reveal(response: str, expected_answer: Optional[str] = None) -> bool:
    """
    Check if the response avoids revealing the direct answer.

    Returns True if no direct answer is revealed.
    """
    response_lower = response.lower()

    # Check for explicit answer patterns
    for pattern in ANSWER_PATTERNS:
        if re.search(pattern, response_lower):
            return False

    # If we know the expected answer, check if it appears explicitly
    if expected_answer:
        # Normalize the expected answer
        normalized = expected_answer.strip().lower()
        # Check for exact answer in response
        if f"answer is {normalized}" in response_lower:
            return False
        if f"= {normalized}" in response_lower:
            return False

    return True


def check_socratic_step(response: str) -> bool:
    """
    Check if the response follows Socratic method.

    Returns True if the response includes guiding questions or suggests steps.
    """
    response_lower = response.lower()

    for pattern in SOCRATIC_PATTERNS:
        if re.search(pattern, response_lower):
            return True

    return False


def check_correct_direction(response: str, topic: Optional[str] = None) -> Optional[bool]:
    """
    Check if the response is mathematically correct in direction.

    Returns None to indicate "not evaluated" - this metric requires
    mathematical reasoning validation and is not yet implemented.

    Future implementation could use:
    - Mathematical reasoning validation
    - Topic-specific fact checking
    - Separate verification model
    """
    return None


def check_appropriate_level(response: str, grade: str) -> bool:
    """
    Check if the response uses grade-appropriate vocabulary.

    This is a simple heuristic based on vocabulary presence.
    """
    response_lower = response.lower()

    # Parse grade
    if grade == "K":
        grade_num = 0
    else:
        grade_num = int(grade)

    # Check vocabulary appropriateness
    if grade_num <= 3:
        # Elementary: should use simple vocab, shouldn't use high-level
        uses_simple = any(word in response_lower for word in ELEMENTARY_VOCAB)
        uses_high = any(word in response_lower for word in HIGH_VOCAB)
        return uses_simple or not uses_high

    elif grade_num <= 8:
        # Middle: can use middle vocab
        uses_high = any(word in response_lower for word in HIGH_VOCAB)
        return not uses_high

    else:
        # High school: any vocabulary is fine
        return True


async def evaluate_prompt(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: dict,
) -> dict:
    """Evaluate a single tutoring prompt."""
    try:
        # Handle null/None grade values - default to "6" if missing or null
        grade = prompt.get("grade") or "6"
        
        response = await client.post(
            endpoint,
            json={
                "question": prompt["question"],
                "attempt": prompt.get("attempt"),
                "mode": prompt.get("mode", "hint"),
                "grade": grade,
                "dont_reveal_answer": True,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        model_response = data["response"]

    except Exception as e:
        return {
            "id": prompt.get("id", "unknown"),
            "question": prompt["question"],
            "response": f"ERROR: {str(e)}",
            "rubric": {
                "no_answer_reveal": False,
                "socratic_step": False,
                "correct_direction": None,
                "appropriate_level": False,
                "error": True,
            },
        }

    # Evaluate rubric
    grade = prompt.get("grade", "6")
    expected = prompt.get("expected_answer")

    rubric = {
        "no_answer_reveal": check_no_answer_reveal(model_response, expected),
        "socratic_step": check_socratic_step(model_response),
        "correct_direction": check_correct_direction(model_response),
        "appropriate_level": check_appropriate_level(model_response, grade),
        "error": False,
    }

    return {
        "id": prompt.get("id", "unknown"),
        "question": prompt["question"],
        "mode": prompt.get("mode", "hint"),
        "grade": grade,
        "response": model_response,
        "rubric": rubric,
    }


async def run_evaluation(
    dataset_path: Path,
    endpoint: str,
    output_path: Optional[Path] = None,
    batch_size: int = 10,
) -> dict:
    """
    Run tutoring rubric evaluation on a dataset.
    
    Args:
        dataset_path: Path to JSONL dataset file
        endpoint: API endpoint URL
        output_path: Optional path to save results JSON
        batch_size: Number of concurrent requests per batch (default: 10)
    """
    # Load dataset
    prompts = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    print(f"Loaded {len(prompts)} prompts from {dataset_path}")
    print(f"Using batch size: {batch_size}")

    # Run evaluations in batches
    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(prompts) + batch_size - 1) // batch_size
            
            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[evaluate_prompt(client, endpoint, prompt) for prompt in batch],
                return_exceptions=True
            )
            
            # Handle exceptions in batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    prompt = batch[j]
                    results.append({
                        "id": prompt.get("id", "unknown"),
                        "prompt": prompt,
                        "response": f"ERROR: {str(result)}",
                        "rubric": {
                            "no_answer_reveal": False,
                            "socratic_step": False,
                            "appropriate_level": False,
                            "correct_direction": None,
                            "error": True,
                        },
                    })
                else:
                    results.append(result)
            
            # Progress update
            tqdm.write(f"Batch {batch_num}/{total_batches} complete ({len(results)}/{len(prompts)} prompts)")
            
            # Rate limiting between batches
            if i + batch_size < len(prompts):
                await asyncio.sleep(0.5)

    # Calculate aggregate metrics
    total = len(results)
    errors = sum(1 for r in results if r["rubric"]["error"])
    valid = total - errors

    if valid > 0:
        no_reveal_rate = sum(1 for r in results if r["rubric"]["no_answer_reveal"]) / valid
        socratic_rate = sum(1 for r in results if r["rubric"]["socratic_step"]) / valid
        appropriate_rate = sum(1 for r in results if r["rubric"]["appropriate_level"]) / valid
    else:
        no_reveal_rate = socratic_rate = appropriate_rate = 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "endpoint": endpoint,
        "total_prompts": total,
        "errors": errors,
        "metrics": {
            "no_answer_reveal_rate": round(no_reveal_rate, 4),
            "socratic_step_rate": round(socratic_rate, 4),
            "appropriate_level_rate": round(appropriate_rate, 4),
            "correct_direction_rate": None,  # Not evaluated
        },
        "results": results,
    }

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {output_path}")

    return summary


def print_summary(summary: dict) -> None:
    """Print evaluation summary to stdout."""
    print("\n" + "=" * 60)
    print("TUTORING RUBRIC EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {summary['dataset']}")
    print(f"Total Prompts: {summary['total_prompts']}")
    print(f"Errors: {summary['errors']}")
    print()
    print("Rubric Scores:")
    print(f"  No Answer Reveal Rate:   {summary['metrics']['no_answer_reveal_rate']:.1%}")
    print(f"  Socratic Step Rate:      {summary['metrics']['socratic_step_rate']:.1%}")
    print(f"  Appropriate Level Rate:  {summary['metrics']['appropriate_level_rate']:.1%}")
    print(f"  Correct Direction Rate:  {summary['metrics']['correct_direction_rate'] or 'N/A'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tutoring rubric")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/chat",
        help="Model endpoint URL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent requests per batch (default: 10)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset file {args.dataset} not found")
        sys.exit(1)

    summary = asyncio.run(
        run_evaluation(args.dataset, args.endpoint, args.output, batch_size=args.batch_size)
    )
    print_summary(summary)

    # Exit with warning if Socratic rate is low
    if summary["metrics"]["socratic_step_rate"] < 0.5:
        print("\n⚠️  Warning: Low Socratic step rate")


if __name__ == "__main__":
    main()

