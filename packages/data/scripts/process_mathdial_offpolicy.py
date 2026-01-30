#!/usr/bin/env python3
"""
Process MathDial dataset for off-policy distillation.

Downloads eth-nlped/mathdial from HuggingFace and converts conversations
to the shared ConversationTrajectory format.

MathDial format:
- conversation: "Teacher: (strategy)message|EOM|Student: message|EOM|..."
- Strategy tags like (probing), (generic) are removed
- Messages are separated by |EOM| delimiter

Usage:
    uv run python scripts/process_mathdial_offpolicy.py
    uv run python scripts/process_mathdial_offpolicy.py --output mathdial_offpolicy.jsonl --max 1000
    uv run python scripts/process_mathdial_offpolicy.py --split train --max 500
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset
except ImportError:
    print("❌ Error: datasets library not found. Install with: uv add datasets")
    sys.exit(1)

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent
if str(data_dir) not in sys.path:
    sys.path.insert(0, str(data_dir))

from data.offpolicy_types import ConversationTrajectory, ConversationMessage


def remove_strategy_tags(content: str) -> str:
    """
    Remove strategy tags from MathDial teacher messages.
    
    Examples:
        "(probing)Hello" -> "Hello"
        "(generic)Exactly correct!" -> "Exactly correct!"
        "No tag here" -> "No tag here"
    """
    # Pattern matches (strategy) at the start of the string
    pattern = r'^\([^)]+\)\s*'
    return re.sub(pattern, '', content).strip()


def parse_mathdial_conversation(conversation_str: str) -> List[ConversationMessage]:
    """
    Parse MathDial conversation string into list of ConversationMessage objects.
    
    Format: "Teacher: (strategy)message|EOM|Student: message|EOM|..."
    
    Args:
        conversation_str: Raw conversation string from MathDial
        
    Returns:
        List of ConversationMessage objects with strategy tags removed
    """
    if not conversation_str or not conversation_str.strip():
        return []
    
    messages = []
    # Split by |EOM| delimiter
    parts = conversation_str.split('|EOM|')
    
    turn = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Parse role and content
        if part.startswith('Teacher:'):
            role = 'tutor'
            content = part[len('Teacher:'):].strip()
            # Remove strategy tags
            content = remove_strategy_tags(content)
        elif part.startswith('Student:'):
            role = 'student'
            content = part[len('Student:'):].strip()
        else:
            # Skip malformed messages
            continue
        
        if content:  # Only add non-empty messages
            messages.append(ConversationMessage(
                role=role,
                content=content,
                turn=turn
            ))
            turn += 1
    
    return messages


def process_mathdial_row(row: dict, row_index: int) -> Optional[ConversationTrajectory]:
    """
    Process a single MathDial row into ConversationTrajectory.
    
    Args:
        row: Raw row from MathDial dataset
        row_index: Index of row (for generating IDs)
        
    Returns:
        ConversationTrajectory or None if row is invalid
    """
    # Extract required fields
    qid = row.get('qid', f'mathdial_{row_index}')
    question = (row.get('question') or '').strip()
    ground_truth = (row.get('ground_truth') or '').strip()
    conversation_str = (row.get('conversation') or '').strip()
    
    # Validate required fields
    if not question:
        return None
    if not ground_truth:
        return None
    if not conversation_str:
        return None
    
    # Parse conversation
    messages = parse_mathdial_conversation(conversation_str)
    if not messages:
        return None
    
    # Extract metadata
    metadata = {
        'qid': qid,
        'scenario': row.get('scenario'),
        'student_profile': row.get('student_profile'),
        'student_incorrect_solution': row.get('student_incorrect_solution'),
        'teacher_described_confusion': row.get('teacher_described_confusion'),
        'self_correctness': row.get('self-correctness'),
        'self_typical_confusion': row.get('self-typical-confusion'),
        'self_typical_interactions': row.get('self-typical-interactions'),
    }
    # Remove None values from metadata
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return ConversationTrajectory(
        id=f"mathdial_{qid}",
        question=question,
        expected_answer=ground_truth,
        messages=messages,
        source="mathdial",
        metadata=metadata
    )


def process_mathdial_dataset(
    max_examples: Optional[int] = None,
    split: Optional[str] = None,
) -> List[ConversationTrajectory]:
    """
    Download and process MathDial dataset.
    
    Args:
        max_examples: Maximum number of examples to process (None for all)
        split: Dataset split to use ("train", "test", or None for both)
        
    Returns:
        List of ConversationTrajectory objects
    """
    print("📥 Downloading eth-nlped/mathdial from HuggingFace...")
    try:
        ds = load_dataset("eth-nlped/mathdial")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise
    
    # Determine which splits to use
    if split and split in ds:
        splits_to_process = [split]
    elif "train" in ds and "test" in ds:
        splits_to_process = ["train", "test"]
    else:
        splits_to_process = [list(ds.keys())[0]]
    
    print(f"📂 Processing splits: {', '.join(splits_to_process)}")
    
    trajectories = []
    skipped = 0
    
    for split_name in splits_to_process:
        print(f"   Processing {split_name} split...")
        split_data = ds[split_name].to_list()
        
        for idx, row in enumerate(split_data):
            if max_examples and len(trajectories) >= max_examples:
                break
            
            trajectory = process_mathdial_row(row, len(trajectories))
            if trajectory:
                trajectories.append(trajectory)
            else:
                skipped += 1
            
            if (idx + 1) % 100 == 0:
                print(f"      Processed {idx + 1} rows, {len(trajectories)} valid trajectories...")
        
        if max_examples and len(trajectories) >= max_examples:
            break
    
    print(f"✅ Processed {len(trajectories)} trajectories (skipped {skipped} invalid rows)")
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process MathDial dataset for off-policy distillation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("mathdial_offpolicy.jsonl"),
        help="Output JSONL path (default: mathdial_offpolicy.jsonl)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of examples to process (default: all)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default=None,
        help="Dataset split to use (default: train+test)",
    )
    args = parser.parse_args()
    
    # Process dataset
    trajectories = process_mathdial_dataset(
        max_examples=args.max,
        split=args.split,
    )
    
    if not trajectories:
        print("❌ Error: No valid trajectories processed")
        sys.exit(1)
    
    # Print example output
    if trajectories:
        print("\n" + "="*80)
        print("📋 Example Post-Processed MathDial Trajectory:")
        print("="*80)
        example = trajectories[0]
        example_dict = {
            "id": example.id,
            "question": example.question,
            "expected_answer": example.expected_answer,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "turn": msg.turn,
                }
                for msg in example.messages
            ],
            "source": example.source,
            "metadata": example.metadata,
        }
        print(json.dumps(example_dict, indent=2, ensure_ascii=False))
        print("="*80)
        print(f"\n📊 Summary:")
        print(f"   - Total messages: {len(example.messages)}")
        print(f"   - Tutor messages: {sum(1 for m in example.messages if m.role == 'tutor')}")
        print(f"   - Student messages: {sum(1 for m in example.messages if m.role == 'student')}")
        print(f"   - Strategy tags removed: ✓")
        print()
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for traj in trajectories:
            # Convert to JSON-serializable format
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
    
    print(f"✅ Wrote {len(trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
