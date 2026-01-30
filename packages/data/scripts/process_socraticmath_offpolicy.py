#!/usr/bin/env python3
"""
Process SocraticMATH dataset for off-policy distillation.

Downloads SocraticMATH from HuggingFace and converts conversations
to the shared ConversationTrajectory format.

SocraticMATH format:
- JSON array: ["teacher_msg1", "student_msg1", "teacher_msg2", ..., "ProblemID", "Analysis"]
- First teacher message is skipped for training but included in history
- Last two elements are metadata (ProblemID and Analysis)
- Messages alternate between teacher and student

Usage:
    uv run python scripts/process_socraticmath_offpolicy.py
    uv run python scripts/process_socraticmath_offpolicy.py --output socraticmath_offpolicy.jsonl --max 1000
"""

from __future__ import annotations

import argparse
import json
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


def extract_metadata(text_array: List[str]) -> tuple[List[str], dict]:
    """
    Extract metadata from SocraticMATH text array.
    
    The last two elements are typically:
    - ProblemID: "【ProblemID】:17578"
    - Analysis: "【解析】:Solution: ..."
    
    Args:
        text_array: Full text array from SocraticMATH
        
    Returns:
        Tuple of (messages_array, metadata_dict)
    """
    if len(text_array) < 2:
        return text_array, {}
    
    metadata = {}
    messages = text_array.copy()
    
    # Check last two elements for metadata
    last = text_array[-1] if len(text_array) > 0 else ""
    second_last = text_array[-2] if len(text_array) > 1 else ""
    
    # Extract ProblemID
    if "【ProblemID】" in second_last or "ProblemID" in second_last:
        # Extract ID number
        problem_id_match = None
        if ":" in second_last:
            problem_id_match = second_last.split(":")[-1].strip()
        elif "：" in second_last:  # Chinese colon
            problem_id_match = second_last.split("：")[-1].strip()
        
        if problem_id_match:
            metadata["problem_id"] = problem_id_match
        messages = messages[:-1]  # Remove ProblemID entry
    
    # Extract Analysis
    if "【解析】" in last or "解析" in last or "Analysis" in last:
        if ":" in last:
            analysis = ":".join(last.split(":")[1:]).strip()
        elif "：" in last:  # Chinese colon
            analysis = "：".join(last.split("：")[1:]).strip()
        else:
            analysis = last.strip()
        
        if analysis:
            metadata["analysis"] = analysis
        messages = messages[:-1]  # Remove Analysis entry
    
    return messages, metadata


def parse_socraticmath_conversation(text_array: List[str]) -> tuple[List[ConversationMessage], str, str]:
    """
    Parse SocraticMATH text array into conversation messages.
    
    Args:
        text_array: JSON array from SocraticMATH dataset
        
    Returns:
        Tuple of (messages, question, expected_answer)
        - messages: List of ConversationMessage (first teacher message included)
        - question: Extracted question/problem text
        - expected_answer: Extracted answer from analysis
    """
    if not text_array or len(text_array) < 2:
        return [], "", ""
    
    # Extract metadata first
    messages_array, metadata = extract_metadata(text_array)
    
    if len(messages_array) < 2:
        return [], "", ""
    
    # Extract question from first student message (index 1, after first teacher)
    question = ""
    if len(messages_array) > 1:
        # First student message often contains the question
        first_student_msg = messages_array[1].strip()
        # Remove common prefixes
        question = first_student_msg
        if "这道题" in question or "这个问题" in question:
            # Question might be embedded in student's first message
            pass
    
    # Extract expected answer from metadata analysis if available
    expected_answer = ""
    if "analysis" in metadata:
        analysis = metadata["analysis"]
        # Try to extract final answer (often at the end)
        # This is heuristic and may need adjustment
        if "答案为" in analysis or "answer is" in analysis.lower():
            # Extract answer part
            parts = analysis.split("答案为" if "答案为" in analysis else "answer is")
            if len(parts) > 1:
                expected_answer = parts[-1].strip().rstrip(".")
        else:
            # Use analysis as expected answer
            expected_answer = analysis
    
    # Parse messages
    messages = []
    turn = 0
    
    for idx, text in enumerate(messages_array):
        text = text.strip()
        if not text:
            continue
        
        # Determine role: even indices are teacher (0, 2, 4...), odd are student (1, 3, 5...)
        role = "tutor" if idx % 2 == 0 else "student"
        
        messages.append(ConversationMessage(
            role=role,
            content=text,
            turn=turn
        ))
        turn += 1
    
    # If question not extracted from first student message, try first teacher message
    if not question and messages:
        first_msg = messages[0]
        if first_msg.role == "tutor":
            # Question might be in first teacher message
            question = first_msg.content.split("\n")[0] if "\n" in first_msg.content else first_msg.content
    
    return messages, question, expected_answer


def process_socraticmath_row(row: dict, row_index: int) -> Optional[ConversationTrajectory]:
    """
    Process a single SocraticMATH row into ConversationTrajectory.
    
    Args:
        row: Raw row from SocraticMATH dataset
        row_index: Index of row (for generating IDs)
        
    Returns:
        ConversationTrajectory or None if row is invalid
    """
    # SocraticMATH format: {"text": ["msg1", "msg2", ...]}
    text_array = row.get("text", [])
    
    if not text_array or not isinstance(text_array, list):
        return None
    
    if len(text_array) < 2:
        return None
    
    # Parse conversation
    messages, question, expected_answer = parse_socraticmath_conversation(text_array)
    
    if not messages:
        return None
    
    # Generate ID
    problem_id = f"socraticmath_{row_index}"
    # Try to extract problem ID from metadata
    _, metadata = extract_metadata(text_array)
    if "problem_id" in metadata:
        problem_id = f"socraticmath_{metadata['problem_id']}"
    
    # If question not extracted, use first message as fallback
    if not question:
        question = messages[0].content if messages else ""
    
    # If expected_answer not extracted, use empty string
    if not expected_answer:
        expected_answer = ""
    
    return ConversationTrajectory(
        id=problem_id,
        question=question,
        expected_answer=expected_answer,
        messages=messages,
        source="socraticmath",
        metadata=metadata
    )


def process_socraticmath_dataset(
    max_examples: Optional[int] = None,
    split: Optional[str] = None,
) -> List[ConversationTrajectory]:
    """
    Download and process SocraticMATH dataset.
    
    Args:
        max_examples: Maximum number of examples to process (None for all)
        split: Dataset split to use (if available)
        
    Returns:
        List of ConversationTrajectory objects
    """
    print("📥 Downloading SocraticMATH from HuggingFace...")
    print("   Note: Trying facat/Socratic (may need GitHub source)")
    
    dataset_name = "facat/Socratic"
    try:
        ds = load_dataset(dataset_name)
        print(f"✅ Loaded dataset: {dataset_name}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to load {dataset_name}: {e}")
        print("   You may need to download from GitHub: https://github.com/ECNU-ICALK/SocraticMath")
        raise
    
    # Determine which split to use
    if split and split in ds:
        splits_to_process = [split]
    elif "train" in ds:
        splits_to_process = ["train"]
    elif "test" in ds:
        splits_to_process = ["test"]
    else:
        # Use first available split
        splits_to_process = [list(ds.keys())[0]]
    
    print(f"📂 Processing split: {', '.join(splits_to_process)}")
    
    trajectories = []
    skipped = 0
    
    for split_name in splits_to_process:
        print(f"   Processing {split_name} split...")
        split_data = ds[split_name].to_list()
        
        for idx, row in enumerate(split_data):
            if max_examples and len(trajectories) >= max_examples:
                break
            
            trajectory = process_socraticmath_row(row, len(trajectories))
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
        description="Process SocraticMATH dataset for off-policy distillation"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("socraticmath_offpolicy.jsonl"),
        help="Output JSONL path (default: socraticmath_offpolicy.jsonl)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of examples to process (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (default: first available)",
    )
    args = parser.parse_args()
    
    # Process dataset
    trajectories = process_socraticmath_dataset(
        max_examples=args.max,
        split=args.split,
    )
    
    if not trajectories:
        print("❌ Error: No valid trajectories processed")
        sys.exit(1)
    
    # Print example output
    if trajectories:
        print("\n" + "="*80)
        print("📋 Example Post-Processed SocraticMATH Trajectory:")
        print("="*80)
        example = trajectories[0]
        example_dict = {
            "id": example.id,
            "question": example.question,
            "expected_answer": example.expected_answer,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,  # Truncate long messages
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
        print(f"   - First tutor message included in history: ✓")
        print(f"   - First tutor message skipped for training: ✓")
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
