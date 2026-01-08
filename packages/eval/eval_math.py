#!/usr/bin/env python3
"""
Math Accuracy Evaluation

Evaluates the mathematical accuracy of model responses by comparing
computed answers against expected answers.

Usage:
    python eval_math.py --dataset samples/math_samples.jsonl \
                        --endpoint http://localhost:8000/chat \
                        --output outputs/math_results.json

Metrics:
    - exact_match: Exact string match of the answer
    - numeric_match: Numeric equivalence (handles formatting differences)
    - partial_match: Answer appears somewhere in response
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


def extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text."""
    # Match integers, decimals, and fractions
    patterns = [
        r"-?\d+\.?\d*",  # Integers and decimals
        r"-?\d+/\d+",  # Fractions
    ]

    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                if "/" in m:
                    num, denom = m.split("/")
                    numbers.append(float(num) / float(denom))
                else:
                    numbers.append(float(m))
            except (ValueError, ZeroDivisionError):
                continue

    return numbers


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    # Remove whitespace and convert to lowercase
    answer = answer.strip().lower()
    # Remove common units and labels
    answer = re.sub(r"\s*(units?|cm|m|kg|g|%)\s*$", "", answer)
    # Remove dollar signs and commas
    answer = answer.replace("$", "").replace(",", "")
    return answer


def exact_match(expected: str, actual: str) -> bool:
    """Check for exact string match."""
    return normalize_answer(expected) == normalize_answer(actual)


def numeric_match(expected: str, actual: str, tolerance: float = 0.001) -> bool:
    """Check for numeric equivalence with tolerance."""
    expected_nums = extract_numbers(expected)
    actual_nums = extract_numbers(actual)

    if not expected_nums:
        return False

    # Check if any expected number appears in actual
    for exp in expected_nums:
        for act in actual_nums:
            if abs(exp - act) < tolerance:
                return True

    return False


def partial_match(expected: str, actual: str) -> bool:
    """Check if expected answer appears anywhere in actual response."""
    normalized_expected = normalize_answer(expected)
    normalized_actual = normalize_answer(actual)
    return normalized_expected in normalized_actual


async def evaluate_problem(
    client: httpx.AsyncClient,
    endpoint: str,
    problem: dict,
) -> dict:
    """Evaluate a single math problem."""
    try:
        # Handle null/None grade values - default to "6" if missing or null
        grade = problem.get("grade") or "6"
        
        response = await client.post(
            endpoint,
            json={
                "question": problem["question"],
                "mode": "explain",  # Use explain mode for answer generation
                "grade": grade,
                "dont_reveal_answer": False,  # Allow answer for evaluation
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        model_response = data["response"]

    except httpx.HTTPStatusError as e:
        # Capture validation errors (422) with details
        error_detail = "Unknown error"
        if e.response.status_code == 422:
            try:
                error_data = e.response.json()
                error_detail = f"Validation error: {error_data.get('detail', 'Unknown validation error')}"
            except:
                error_detail = f"HTTP 422: {e.response.text[:200]}"
        return {
            "id": problem.get("id", "unknown"),
            "question": problem["question"],
            "expected": problem["answer"],
            "actual": f"ERROR: {error_detail}",
            "metrics": {
                "exact_match": False,
                "numeric_match": False,
                "partial_match": False,
                "error": True,
            },
        }
    except Exception as e:
        return {
            "id": problem.get("id", "unknown"),
            "question": problem["question"],
            "expected": problem["answer"],
            "actual": f"ERROR: {str(e)}",
            "metrics": {
                "exact_match": False,
                "numeric_match": False,
                "partial_match": False,
                "error": True,
            },
        }

    expected = problem["answer"]
    return {
        "id": problem.get("id", "unknown"),
        "question": problem["question"],
        "expected": expected,
        "actual": model_response,
        "metrics": {
            "exact_match": exact_match(expected, model_response),
            "numeric_match": numeric_match(expected, model_response),
            "partial_match": partial_match(expected, model_response),
            "error": False,
        },
    }


async def run_evaluation(
    dataset_path: Path,
    endpoint: str,
    output_path: Optional[Path] = None,
    batch_size: int = 10,
) -> dict:
    """
    Run evaluation on a dataset.
    
    Args:
        dataset_path: Path to JSONL dataset file
        endpoint: API endpoint URL
        output_path: Optional path to save results JSON
        batch_size: Number of concurrent requests per batch (default: 10)
    """
    # Load dataset
    problems = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    print(f"Loaded {len(problems)} problems from {dataset_path}")
    print(f"Using batch size: {batch_size}")

    # Run evaluations in batches
    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Process in batches
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(problems) + batch_size - 1) // batch_size
            
            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[evaluate_problem(client, endpoint, problem) for problem in batch],
                return_exceptions=True
            )
            
            # Handle exceptions in batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    problem = batch[j]
                    results.append({
                        "id": problem.get("id", "unknown"),
                        "question": problem.get("question", ""),
                        "expected": problem.get("answer", ""),
                        "actual": f"ERROR: {str(result)}",
                        "metrics": {
                            "exact_match": False,
                            "numeric_match": False,
                            "partial_match": False,
                            "error": True,
                        },
                    })
                else:
                    results.append(result)
            
            # Progress update
            tqdm.write(f"Batch {batch_num}/{total_batches} complete ({len(results)}/{len(problems)} problems)")
            
            # Rate limiting between batches (small delay to avoid overwhelming server)
            if i + batch_size < len(problems):
                await asyncio.sleep(0.5)

    # Calculate aggregate metrics
    total = len(results)
    errors = sum(1 for r in results if r["metrics"]["error"])
    valid = total - errors

    if valid > 0:
        exact_acc = sum(1 for r in results if r["metrics"]["exact_match"]) / valid
        numeric_acc = sum(1 for r in results if r["metrics"]["numeric_match"]) / valid
        partial_acc = sum(1 for r in results if r["metrics"]["partial_match"]) / valid
    else:
        exact_acc = numeric_acc = partial_acc = 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "endpoint": endpoint,
        "total_problems": total,
        "errors": errors,
        "metrics": {
            "exact_match_accuracy": round(exact_acc, 4),
            "numeric_match_accuracy": round(numeric_acc, 4),
            "partial_match_accuracy": round(partial_acc, 4),
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
    print("MATH ACCURACY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {summary['dataset']}")
    print(f"Total Problems: {summary['total_problems']}")
    print(f"Errors: {summary['errors']}")
    print()
    print("Metrics:")
    print(f"  Exact Match Accuracy:   {summary['metrics']['exact_match_accuracy']:.1%}")
    print(f"  Numeric Match Accuracy: {summary['metrics']['numeric_match_accuracy']:.1%}")
    print(f"  Partial Match Accuracy: {summary['metrics']['partial_match_accuracy']:.1%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate math accuracy")
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

    # Exit with error if accuracy is very low
    if summary["metrics"]["numeric_match_accuracy"] < 0.1:
        sys.exit(1)


if __name__ == "__main__":
    main()

