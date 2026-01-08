#!/usr/bin/env python3
"""
Comprehensive Benchmarking Script

Runs all evaluation suites against a model endpoint and saves results
for comparison. Designed for benchmarking baseline models and comparing
fine-tuned variants.

Usage:
    # Benchmark a model
    python benchmark.py --model-name qwen3:8b --endpoint http://localhost:8000/chat

    # Benchmark with custom datasets
    python benchmark.py --model-name qwen3:8b --datasets gsm8k.jsonl,math_samples.jsonl

    # Compare against previous results
    python benchmark.py --model-name qwen3:8b --compare baseline_results.json
"""

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Import evaluation modules
eval_dir = Path(__file__).parent
spec_math = importlib.util.spec_from_file_location("eval_math", eval_dir / "eval_math.py")
spec_tutor = importlib.util.spec_from_file_location("eval_tutor", eval_dir / "eval_tutor.py")
spec_safety = importlib.util.spec_from_file_location("eval_safety", eval_dir / "eval_safety.py")

eval_math = importlib.util.module_from_spec(spec_math)
eval_tutor = importlib.util.module_from_spec(spec_tutor)
eval_safety = importlib.util.module_from_spec(spec_safety)

spec_math.loader.exec_module(eval_math)
spec_tutor.loader.exec_module(eval_tutor)
spec_safety.loader.exec_module(eval_safety)

run_math_eval = eval_math.run_evaluation
run_tutor_eval = eval_tutor.run_evaluation
run_safety_eval = eval_safety.run_evaluation


def load_dataset_registry() -> dict:
    """Load dataset registry configuration."""
    registry_path = eval_dir / "datasets" / "dataset_registry.yaml"
    if registry_path.exists():
        with open(registry_path) as f:
            return yaml.safe_load(f)
    return {}


def get_dataset_path(dataset_key: str, suite: Optional[str] = None) -> Optional[Path]:
    """Get path to a dataset from registry."""
    registry = load_dataset_registry()
    
    # Check benchmark suites first
    if suite and "benchmark_suites" in registry:
        suites = registry["benchmark_suites"]
        if suite in suites:
            suite_config = suites[suite]
            if dataset_key in suite_config:
                path_str = suite_config[dataset_key]
                full_path = eval_dir / path_str
                if full_path.exists():
                    return full_path
    
    # Check defaults
    if "defaults" in registry:
        defaults = registry["defaults"]
        if dataset_key in defaults:
            default_key = defaults[dataset_key]
            # Look up in datasets section
            if "datasets" in registry and "benchmarks" in registry["datasets"]:
                benchmarks = registry["datasets"]["benchmarks"]
                if default_key in benchmarks:
                    path_str = benchmarks[default_key]["path"]
                    full_path = eval_dir / path_str
                    if full_path.exists():
                        return full_path
    
    # Fallback to samples
    if "datasets" in registry and "samples" in registry["datasets"]:
        samples = registry["datasets"]["samples"]
        if dataset_key in samples:
            path_str = samples[dataset_key]["path"]
            full_path = eval_dir / path_str
            if full_path.exists():
                return full_path
    
    return None


def run_eval_script(script_name: str, args: list[str]) -> dict:
    """Run an evaluation script and return its results."""
    try:
        result = subprocess.run(
            ["python", "-m", f"eval.{script_name}"] + args,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        result.check_returncode()

        # Parse JSON from stdout (last line should be JSON)
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

        # Fallback: try to find JSON in output
        import re

        json_match = re.search(r"\{.*\}", result.stdout, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        return {"error": "Could not parse results", "output": result.stdout}

    except subprocess.CalledProcessError as e:
        return {"error": str(e), "stderr": e.stderr}


async def run_benchmark(
    model_name: str,
    endpoint: str,
    output_dir: Path,
    datasets: Optional[list[str]] = None,
    suite: Optional[str] = None,
    batch_size: int = 10,
) -> dict:
    """
    Run comprehensive benchmark suite.

    Returns a dictionary with all evaluation results.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*70}")
    print(f"Endpoint: {endpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_name": model_name,
        "endpoint": endpoint,
        "timestamp": datetime.now().isoformat(),
        "evaluations": {},
    }

    # Determine datasets to use
    if datasets:
        # Custom datasets provided
        datasets_dict = {f"custom_{i}": Path(d) for i, d in enumerate(datasets)}
    else:
        # Use dataset registry
        suite = suite or "standard"  # Default to standard suite
        math_path = get_dataset_path("math", suite) or get_dataset_path("math")
        tutor_path = get_dataset_path("tutor", suite) or get_dataset_path("tutor")
        safety_path = get_dataset_path("safety", suite) or get_dataset_path("safety")
        
        datasets_dict = {
            "math": math_path,
            "tutor": tutor_path,
            "safety": safety_path,
        }
        
        # Filter out None values
        datasets_dict = {k: v for k, v in datasets_dict.items() if v is not None}
        
        if not datasets_dict:
            print("⚠️  Warning: No datasets found. Using sample datasets.")
            datasets_dict = {
                "math": eval_dir / "samples" / "math_samples.jsonl",
                "tutor": eval_dir / "samples" / "tutor_samples.jsonl",
                "safety": eval_dir / "samples" / "safety_samples.jsonl",
            }

    # Run math accuracy evaluation
    print("📊 Running Math Accuracy Evaluation...")
    math_output = output_dir / "math_results.json"
    try:
        math_dataset = datasets_dict.get("math")
        if not math_dataset:
            raise ValueError("No math dataset found")
        print(f"   Using dataset: {math_dataset}")
        math_results = await run_math_eval(
            math_dataset,
            endpoint,
            math_output,
            batch_size=batch_size,
        )
        results["evaluations"]["math_accuracy"] = math_results["metrics"]
        print(f"   ✓ Math accuracy: {math_results['metrics']['numeric_match_accuracy']:.1%}\n")
    except Exception as e:
        print(f"   ✗ Math evaluation failed: {e}\n")
        results["evaluations"]["math_accuracy"] = {"error": str(e)}

    # Run tutoring rubric evaluation
    print("📚 Running Tutoring Rubric Evaluation...")
    tutor_output = output_dir / "tutor_results.json"
    try:
        tutor_dataset = datasets_dict.get("tutor")
        if not tutor_dataset:
            raise ValueError("No tutor dataset found")
        print(f"   Using dataset: {tutor_dataset}")
        tutor_results = await run_tutor_eval(
            tutor_dataset,
            endpoint,
            tutor_output,
            batch_size=batch_size,
        )
        results["evaluations"]["tutoring_quality"] = tutor_results["metrics"]
        print(f"   ✓ Socratic step rate: {tutor_results['metrics']['socratic_step_rate']:.1%}\n")
    except Exception as e:
        print(f"   ✗ Tutor evaluation failed: {e}\n")
        results["evaluations"]["tutoring_quality"] = {"error": str(e)}

    # Run safety evaluation
    print("🛡️  Running Safety Evaluation...")
    safety_output = output_dir / "safety_results.json"
    try:
        safety_dataset = datasets_dict.get("safety")
        if not safety_dataset:
            raise ValueError("No safety dataset found")
        print(f"   Using dataset: {safety_dataset}")
        safety_results = await run_safety_eval(
            safety_dataset,
            endpoint,
            safety_output,
            batch_size=batch_size,
        )
        results["evaluations"]["safety"] = safety_results["metrics"]
        print(f"   ✓ Refusal rate: {safety_results['metrics'].get('refusal_rate_when_needed', 'N/A')}\n")
    except Exception as e:
        print(f"   ✗ Safety evaluation failed: {e}\n")
        results["evaluations"]["safety"] = {"error": str(e)}

    # Save comprehensive results
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"\nSummary:")
    if "math_accuracy" in results["evaluations"]:
        math_acc = results["evaluations"]["math_accuracy"].get("numeric_match_accuracy", 0)
        print(f"  Math Accuracy: {math_acc:.1%}")
    if "tutoring_quality" in results["evaluations"]:
        socratic = results["evaluations"]["tutoring_quality"].get("socratic_step_rate", 0)
        print(f"  Socratic Step Rate: {socratic:.1%}")
    if "safety" in results["evaluations"]:
        refusal = results["evaluations"]["safety"].get("overall_accuracy", 0)
        print(f"  Safety Accuracy: {refusal:.1%}")

    return results


def compare_results(baseline_file: Path, current_results: dict) -> dict:
    """Compare current results against baseline."""
    with open(baseline_file) as f:
        baseline = json.load(f)

    comparison = {
        "baseline": baseline["model_name"],
        "current": current_results["model_name"],
        "improvements": {},
        "regressions": {},
    }

    # Compare metrics
    for eval_type in ["math_accuracy", "tutoring_quality", "safety"]:
        if eval_type in baseline["evaluations"] and eval_type in current_results["evaluations"]:
            baseline_metrics = baseline["evaluations"][eval_type]
            current_metrics = current_results["evaluations"][eval_type]

            for metric, baseline_val in baseline_metrics.items():
                if isinstance(baseline_val, (int, float)) and metric in current_metrics:
                    current_val = current_metrics[metric]
                    diff = current_val - baseline_val
                    if diff > 0.01:  # 1% improvement
                        comparison["improvements"][f"{eval_type}.{metric}"] = {
                            "baseline": baseline_val,
                            "current": current_val,
                            "improvement": diff,
                        }
                    elif diff < -0.01:  # 1% regression
                        comparison["regressions"][f"{eval_type}.{metric}"] = {
                            "baseline": baseline_val,
                            "current": current_val,
                            "regression": abs(diff),
                        }

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive model benchmark")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name/identifier for this model (e.g., 'qwen3:8b-baseline')",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/chat",
        help="Model API endpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of dataset paths (overrides --suite)",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Dataset suite to use (quick/standard/comprehensive). Default: standard",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Path to baseline results JSON for comparison",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent requests per batch (default: 10). Increase for faster benchmarking if your API can handle it.",
    )
    args = parser.parse_args()

    # Parse datasets if provided
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
        suite = None  # Custom datasets override suite
    else:
        suite = args.suite

    # Create output directory with model name
    output_dir = args.output_dir / args.model_name.replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    results = asyncio.run(
        run_benchmark(args.model_name, args.endpoint, output_dir, datasets, suite, batch_size=args.batch_size)
    )

    # Compare if baseline provided
    if args.compare:
        comparison = compare_results(args.compare, results)
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)

        print("\n" + "=" * 70)
        print("COMPARISON WITH BASELINE")
        print("=" * 70)
        if comparison["improvements"]:
            print("\n✓ Improvements:")
            for metric, data in comparison["improvements"].items():
                print(f"  {metric}: {data['baseline']:.3f} → {data['current']:.3f} (+{data['improvement']:.3f})")
        if comparison["regressions"]:
            print("\n✗ Regressions:")
            for metric, data in comparison["regressions"].items():
                print(f"  {metric}: {data['baseline']:.3f} → {data['current']:.3f} (-{data['regression']:.3f})")
        print(f"\nFull comparison saved to: {comparison_file}")


if __name__ == "__main__":
    main()

