#!/usr/bin/env python3
"""
Convert processed MathDial and SocraticMATH files to unified off-policy format.

This script validates and combines processed conversation trajectories from
multiple sources into a single unified JSONL file ready for off-policy distillation.

Usage:
    uv run python scripts/convert_to_offpolicy_format.py \
        --input mathdial_offpolicy.jsonl socraticmath_offpolicy.jsonl \
        --output unified_offpolicy.jsonl
    
    uv run python scripts/convert_to_offpolicy_format.py \
        --input mathdial_offpolicy.jsonl \
        --output unified_offpolicy.jsonl \
        --validate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent
if str(data_dir) not in sys.path:
    sys.path.insert(0, str(data_dir))

from data.offpolicy_types import ConversationTrajectory, ConversationMessage


def validate_trajectory(traj: ConversationTrajectory) -> tuple[bool, Optional[str]]:
    """
    Validate a ConversationTrajectory.
    
    Checks:
    - At least one tutor-student exchange
    - Proper role alternation (tutor/student/tutor/student...)
    - Non-empty messages
    - Required fields present
    
    Args:
        traj: ConversationTrajectory to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    if not traj.id:
        return False, "Missing id"
    if not traj.question:
        return False, "Missing question"
    if not traj.messages:
        return False, "No messages"
    
    # Check message validity
    if len(traj.messages) < 2:
        return False, "Need at least 2 messages (one tutor-student exchange)"
    
    # Check role alternation
    expected_role = None
    for msg in traj.messages:
        if not msg.content or not msg.content.strip():
            return False, f"Empty message content at turn {msg.turn}"
        
        if msg.role not in ("tutor", "student"):
            return False, f"Invalid role '{msg.role}' at turn {msg.turn}"
        
        # Check alternation (first message can be either, but should alternate after)
        if expected_role is not None and msg.role == expected_role:
            # Allow same role if it's a continuation, but warn
            pass
        
        expected_role = "student" if msg.role == "tutor" else "tutor"
    
    # Check that we have at least one tutor message
    tutor_count = sum(1 for msg in traj.messages if msg.role == "tutor")
    if tutor_count == 0:
        return False, "No tutor messages found"
    
    # Check that we have at least one student message
    student_count = sum(1 for msg in traj.messages if msg.role == "student")
    if student_count == 0:
        return False, "No student messages found"
    
    return True, None


def load_trajectory_from_dict(data: dict) -> ConversationTrajectory:
    """
    Load ConversationTrajectory from dictionary.
    
    Args:
        data: Dictionary with trajectory data
        
    Returns:
        ConversationTrajectory object
    """
    messages = [
        ConversationMessage(
            role=msg_data["role"],
            content=msg_data["content"],
            turn=msg_data["turn"],
        )
        for msg_data in data.get("messages", [])
    ]
    
    return ConversationTrajectory(
        id=data["id"],
        question=data["question"],
        expected_answer=data.get("expected_answer", ""),
        messages=messages,
        source=data.get("source", ""),
        metadata=data.get("metadata", {}),
    )


def load_trajectories_from_jsonl(file_path: Path) -> List[ConversationTrajectory]:
    """
    Load trajectories from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of ConversationTrajectory objects
    """
    trajectories = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                traj = load_trajectory_from_dict(data)
                trajectories.append(traj)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse line {line_num} in {file_path}: {e}")
                continue
            except KeyError as e:
                print(f"⚠️  Warning: Missing required field at line {line_num} in {file_path}: {e}")
                continue
    
    return trajectories


def convert_and_validate(
    input_files: List[Path],
    output_file: Optional[Path] = None,
    validate_only: bool = False,
) -> tuple[List[ConversationTrajectory], List[ConversationTrajectory]]:
    """
    Load, validate, and optionally write unified trajectories.
    
    Args:
        input_files: List of input JSONL files
        output_file: Output file path (None if validate_only)
        validate_only: If True, only validate without writing
        
    Returns:
        Tuple of (valid_trajectories, invalid_trajectories)
    """
    all_trajectories = []
    
    # Load from all input files
    for input_file in input_files:
        if not input_file.exists():
            print(f"❌ Error: Input file not found: {input_file}")
            sys.exit(1)
        
        print(f"📂 Loading {input_file.name}...")
        trajectories = load_trajectories_from_jsonl(input_file)
        print(f"   Loaded {len(trajectories)} trajectories")
        all_trajectories.extend(trajectories)
    
    print(f"\n📊 Total trajectories loaded: {len(all_trajectories)}")
    
    # Validate trajectories
    print("\n🔍 Validating trajectories...")
    valid_trajectories = []
    invalid_trajectories = []
    
    for traj in all_trajectories:
        is_valid, error = validate_trajectory(traj)
        if is_valid:
            valid_trajectories.append(traj)
        else:
            invalid_trajectories.append((traj, error))
            print(f"   ⚠️  Invalid trajectory {traj.id}: {error}")
    
    print(f"\n✅ Valid: {len(valid_trajectories)}")
    print(f"❌ Invalid: {len(invalid_trajectories)}")
    
    # Write output if not validate-only
    if not validate_only and output_file:
        print(f"\n💾 Writing to {output_file}...")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for traj in valid_trajectories:
                traj_dict = {
                    "id": traj.id,
                    "question": traj.question,
                    "expected_answer": traj.expected_answer,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "turn": msg.turn,
                        }
                        for msg in traj.messages
                    ],
                    "source": traj.source,
                    "metadata": traj.metadata,
                }
                f.write(json.dumps(traj_dict, ensure_ascii=False) + "\n")
        
        print(f"✅ Wrote {len(valid_trajectories)} valid trajectories to {output_file}")
    
    return valid_trajectories, invalid_trajectories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert processed datasets to unified off-policy format"
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        type=Path,
        required=True,
        help="Input JSONL files (processed MathDial and/or SocraticMATH)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSONL path (required unless --validate-only)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate trajectories without writing output",
    )
    args = parser.parse_args()
    
    if not args.validate_only and not args.output:
        parser.error("--output is required unless --validate-only is set")
    
    convert_and_validate(
        input_files=args.input,
        output_file=args.output,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
