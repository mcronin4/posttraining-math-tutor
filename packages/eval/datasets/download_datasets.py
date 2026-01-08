#!/usr/bin/env python3
"""
Download Benchmark Datasets

Downloads established math datasets for comprehensive benchmarking:
- MathBench (if available)
- GSM8K (HuggingFace)
- MATH (HuggingFace)
- AQuA-RAT (HuggingFace)

Usage:
    python download_datasets.py --all
    python download_datasets.py --dataset gsm8k
    python download_datasets.py --dataset math --split test
"""

import argparse
import json
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("⚠️  HuggingFace datasets not installed.")
    print("Install with: pip install datasets")
    print("Or: python -m uv pip install datasets")
    exit(1)


DATASET_CONFIGS = {
    "gsm8k": {
        "name": "gsm8k",
        "splits": ["test", "train"],
        "description": "Grade School Math 8K",
    },
    "math": {
        "name": "hendrycks/competition_math",
        "splits": ["test", "train"],
        "description": "Competition Math Problems",
    },
    "aqua": {
        "name": "allenai/ai2_arc",
        "splits": ["ARC-Challenge", "ARC-Easy"],
        "description": "AI2 ARC (similar to AQuA-RAT)",
        "note": "Using ARC as AQuA-RAT alternative",
    },
}


def download_gsm8k(output_dir: Path, split: str = "test") -> Path:
    """Download GSM8K dataset."""
    print(f"📥 Downloading GSM8K ({split} split)...")
    dataset = load_dataset("gsm8k", "main", split=split)

    output_file = output_dir / f"gsm8k_{split}.jsonl"
    with open(output_file, "w") as f:
        for item in dataset:
            # Convert GSM8K format to our format
            f.write(
                json.dumps(
                    {
                        "id": f"gsm8k_{item['question'][:20]}",
                        "question": item["question"],
                        "answer": item["answer"].split("####")[-1].strip(),
                        "grade": None,  # GSM8K doesn't have grade info
                        "topic_tags": [],
                    }
                )
                + "\n"
            )

    print(f"✅ Saved {len(dataset)} examples to {output_file}")
    return output_file


def download_math(output_dir: Path, split: str = "test") -> Path:
    """Download MATH dataset."""
    print(f"📥 Downloading MATH ({split} split)...")
    # Try different possible dataset names
    dataset_names = [
        "lighteval/MATH",
        "hendrycks/competition_math",
        "MATH",
    ]
    
    dataset = None
    for name in dataset_names:
        try:
            print(f"   Trying: {name}")
            dataset = load_dataset(name, split=split)
            break
        except Exception as e:
            print(f"   Failed: {e}")
            continue
    
    if dataset is None:
        raise ValueError(f"Could not download MATH dataset. Tried: {dataset_names}")

    output_file = output_dir / f"math_{split}.jsonl"
    with open(output_file, "w") as f:
        for i, item in enumerate(dataset):
            # Extract answer from solution
            solution = item["solution"]
            # Try to extract final answer (usually at the end)
            answer = solution.split("\\boxed{")[-1].split("}")[0] if "\\boxed{" in solution else solution[-50:]

            f.write(
                json.dumps(
                    {
                        "id": f"math_{split}_{i}",
                        "question": item["problem"],
                        "answer": answer,
                        "grade": None,  # MATH has level instead
                        "topic_tags": [item.get("type", "unknown")],
                        "metadata": {"level": item.get("level", "unknown")},
                    }
                )
                + "\n"
            )

    print(f"✅ Saved {len(dataset)} examples to {output_file}")
    return output_file


def download_all(output_dir: Path):
    """Download all supported datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOWNLOADING BENCHMARK DATASETS")
    print("=" * 70)

    try:
        # Download GSM8K test set 
        download_gsm8k(output_dir, "test")
        download_math(output_dir, "test")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not download dataset: {e}")

    print("\n" + "=" * 70)
    print("✅ Dataset download complete!")
    print(f"📁 Location: {output_dir}")
    print("\nNext steps:")
    print("1. Review datasets: cat datasets/gsm8k_test.jsonl | head")
    print("2. Tag with taxonomy: python packages/data/scripts/tag_problem.py --input datasets/gsm8k_test.jsonl --output datasets/gsm8k_test_tagged.jsonl")
    print("3. Benchmark: python benchmark.py --model-name qwen3:8b --suite standard")


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "all"],
        default="all",
        help="Dataset to download",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (test/train)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for datasets",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        download_all(args.output_dir)
    elif args.dataset == "gsm8k":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        download_gsm8k(args.output_dir, args.split)
    elif args.dataset == "math":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        download_math(args.output_dir, args.split)


if __name__ == "__main__":
    main()

