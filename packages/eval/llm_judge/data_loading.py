"""Dataset loading utilities."""

import json
from pathlib import Path
from typing import List, Optional


def load_gsm8k_dataset(dataset_path: Path, limit: Optional[int] = None) -> List[dict]:
    """Load GSM8K problems from JSONL file."""
    problems = []
    with open(dataset_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(problems) >= limit:
                break
            line = line.strip()
            if line:
                try:
                    problem = json.loads(line)
                    problems.append(problem)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
                    continue
    return problems
