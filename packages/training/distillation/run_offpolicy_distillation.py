"""
Run off-policy distillation with pre-existing conversation trajectories.

This script provides a command-line interface for running off-policy distillation
training using conversation trajectories from MathDial and SocraticMATH datasets.

Usage:
    # Single conversation file
    uv run python run_offpolicy_distillation.py \
        --tutor-model Qwen3-8B \
        --teacher-model Qwen3-35B-A22B-Instruct \
        --conversations-file ../../data/unified_offpolicy.jsonl \
        --log-path ./logs/offpolicy_distillation_run_001 \
        --learning-rate 1e-5 \
        --max-tokens 2048

    # Multiple conversation files with limits
    uv run python run_offpolicy_distillation.py \
        --tutor-model Qwen3-8B \
        --teacher-model Qwen3-35B-A22B-Instruct \
        --conversations-files ../../data/mathdial_offpolicy.jsonl 2000 ../../data/socraticmath_offpolicy.jsonl 3000 \
        --log-path ./logs/offpolicy_distillation_run_002 \
        --learning-rate 1e-5 \
        --max-tokens 2048
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tinker_cookbook.distillation.train_on_policy import main

from training.distillation.distillation import build_offpolicy_socratic_config


def parse_conversations_files_args(args_list: List[str]) -> List[tuple[Path, Optional[int]]]:
    """
    Parse conversations_files argument list into list of (file_path, limit) tuples.
    
    Expected format: [file1, limit1, file2, limit2, ...]
    If limit is missing for a file, it defaults to None (no limit).
    """
    if len(args_list) % 2 == 1:
        # Odd number of args - last one is a file without limit
        result = []
        for i in range(0, len(args_list) - 1, 2):
            file_path = Path(args_list[i])
            limit = int(args_list[i + 1]) if args_list[i + 1] != "None" else None
            result.append((file_path, limit))
        # Add last file without limit
        result.append((Path(args_list[-1]), None))
        return result
    else:
        # Even number of args - pairs of (file, limit)
        result = []
        for i in range(0, len(args_list), 2):
            file_path = Path(args_list[i])
            limit = int(args_list[i + 1]) if args_list[i + 1] != "None" else None
            result.append((file_path, limit))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run off-policy distillation with pre-existing conversation trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single conversation file
  uv run python run_offpolicy_distillation.py \\
      --tutor-model Qwen3-8B \\
      --teacher-model Qwen3-35B-A22B-Instruct \\
      --conversations-file ../../data/unified_offpolicy.jsonl \\
      --log-path ./logs/offpolicy_distillation_run_001 \\
      --learning-rate 1e-5 \\
      --max-tokens 2048

  # Multiple conversation files with limits
  uv run python run_offpolicy_distillation.py \\
      --tutor-model Qwen3-8B \\
      --teacher-model Qwen3-35B-A22B-Instruct \\
      --conversations-files ../../data/mathdial_offpolicy.jsonl 2000 ../../data/socraticmath_offpolicy.jsonl 3000 \\
      --log-path ./logs/offpolicy_distillation_run_002 \\
      --learning-rate 1e-5 \\
      --max-tokens 2048
        """,
    )
    
    # Model arguments
    parser.add_argument(
        "--tutor-model",
        type=str,
        required=True,
        help="Name of the tutor model (student being trained, e.g., Qwen3-8B)",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
        help="Name of the teacher model for KL penalty (e.g., Qwen3-35B-A22B-Instruct)",
    )
    
    # Data arguments
    conversations_group = parser.add_mutually_exclusive_group(required=True)
    conversations_group.add_argument(
        "--conversations-file",
        type=Path,
        help="Path to single JSONL file with conversation trajectories",
    )
    conversations_group.add_argument(
        "--conversations-files",
        nargs="+",
        help="List of conversation files with optional limits: file1 limit1 file2 limit2 ...",
    )
    parser.add_argument(
        "--conversations-limit",
        type=int,
        default=None,
        help="Optional limit on number of conversations to load (only used with --conversations-file)",
    )
    
    # Training arguments
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Maximum tokens per generation",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path for logging and checkpoints (default: ./offpolicy_distillation_results/{date}_{tutor_model})",
    )
    
    # Optional arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for training (default: 2e-4)",
    )
    parser.add_argument(
        "--groups-per-batch",
        type=int,
        default=10,
        help="Number of environment groups per batch",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        help="Base URL for Tinker service (default: https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1)",
    )
    
    args = parser.parse_args()
    
    # Generate default log path if not provided
    if args.log_path is None:
        # Create a sanitized tutor model name for the path
        tutor_model_safe = args.tutor_model.replace("/", "_").replace(":", "_").replace("-", "_")
        # Format: YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_path = f"./offpolicy_distillation_results/{timestamp}_{tutor_model_safe}"
    
    # Parse conversations_files if provided
    conversations_files = None
    conversations_file = None
    conversations_limit = None
    
    if args.conversations_files:
        conversations_files = parse_conversations_files_args(args.conversations_files)
        print(f"📂 Using multiple conversation files:")
        for file_path, limit in conversations_files:
            print(f"   - {file_path} (limit: {limit if limit is not None else 'unlimited'})")
    else:
        conversations_file = args.conversations_file
        conversations_limit = args.conversations_limit
        print(f"📂 Using single conversation file: {conversations_file}")
        if conversations_limit:
            print(f"   Limit: {conversations_limit}")
    
    # Build config
    print("\n🔧 Building off-policy distillation config...")
    config = build_offpolicy_socratic_config(
        tutor_model_name=args.tutor_model,
        teacher_model_name=args.teacher_model,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        log_path=args.log_path,
        groups_per_batch=args.groups_per_batch,
        conversations_file=conversations_file,
        conversations_limit=conversations_limit,
        conversations_files=conversations_files,
        base_url=args.base_url,
    )
    
    print(f"\n✅ Config built successfully!")
    print(f"   Tutor model: {args.tutor_model}")
    print(f"   Teacher model: {args.teacher_model}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Log path: {args.log_path}")
    print(f"\n🚀 Starting off-policy distillation training...\n")
    
    # Run training
    asyncio.run(main(config))
