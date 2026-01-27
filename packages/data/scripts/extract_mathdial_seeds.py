#!/usr/bin/env python3
"""
Extract MathDial seed contexts.

Loads eth-nlped/mathdial, extracts 2,000 unique questions with ground_truth and
student_incorrect_solution, and writes a JSONL of Seed Context records.

Usage:
    uv run python scripts/extract_mathdial_seeds.py
    uv run python scripts/extract_mathdial_seeds.py --output seeds.jsonl --max 2000
    uv run python scripts/extract_mathdial_seeds.py --by-row --split train --max 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


def extract_seeds(
    max_seeds: int = 2000,
    *,
    unique_questions: bool = True,
    split: str | None = None,
) -> list[dict]:
    """
    Load mathdial and produce Seed Context records.

    Each record has: id, question, expected_answer (ground_truth), initial_error
    (student_incorrect_solution).

    If unique_questions=True (default), deduplicate by question text so each
    question appears at most once. If False, emit one seed per row (same
    question can appear multiple times with different initial_error).

    split: "train" | "test" | None. If set, use only that split; else train+test.
    """
    ds = load_dataset("eth-nlped/mathdial")
    if split and split in ds:
        combined = ds[split].to_list()
    elif "train" in ds and "test" in ds:
        combined = ds["train"].to_list() + ds["test"].to_list()
    else:
        combined = list(ds[list(ds.keys())[0]])

    seen: set[str] | None = set() if unique_questions else None
    seeds: list[dict] = []
    for row in combined:
        if len(seeds) >= max_seeds:
            break
        q = (row.get("question") or "").strip()
        if not q:
            continue
        if unique_questions and seen is not None:
            if q in seen:
                continue
            seen.add(q)
        gt = (row.get("ground_truth") or "").strip()
        err = (row.get("student_incorrect_solution") or "").strip()
        seed_id = f"mathdial_seed_{len(seeds):04d}"
        seeds.append({
            "id": seed_id,
            "question": q,
            "expected_answer": gt,
            "initial_error": err,
        })
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MathDial seed contexts to JSONL"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("mathdial_seeds.jsonl"),
        help="Output JSONL path (default: mathdial_seeds.jsonl)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=2000,
        help="Max seeds to extract (default: 2000)",
    )
    parser.add_argument(
        "--by-row",
        action="store_true",
        help="One seed per row (no dedup). Use --split train to take only train.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default=None,
        help="Use only this split (default: train+test). Useful with --by-row.",
    )
    args = parser.parse_args()

    unique = not args.by_row
    print("Loading eth-nlped/mathdial...")
    seeds = extract_seeds(
        max_seeds=args.max,
        unique_questions=unique,
        split=args.split,
    )
    mode = "unique questions" if unique else "rows"
    print(f"Extracted {len(seeds)} seed contexts ({mode}).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")

    if len(seeds) < args.max:
        print(
            f"Warning: only {len(seeds)} unique questions found (requested {args.max}).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
