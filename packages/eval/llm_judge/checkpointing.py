"""
Checkpointing utilities for LLM-as-a-Judge evaluation.

This module provides functions for saving and loading checkpoint files
during evaluation runs, allowing for resumable evaluation sessions.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Try relative import first, fallback to absolute import
try:
    from .llm_judge_types import ConversationResult, ConversationMessage
except ImportError:
    import sys
    from pathlib import Path
    # Add packages directory to path so we can import eval
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.llm_judge.llm_judge_types import ConversationResult, ConversationMessage


def load_checkpoint(checkpoint_path: Path) -> tuple[List[ConversationResult], dict]:
    """Load conversation results and metadata from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint JSON file
        
    Returns:
        Tuple of (results, metadata) where:
        - results: List of ConversationResult objects loaded from checkpoint
        - metadata: Dictionary containing run configuration (total_problems, prompt_type, etc.)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with open(checkpoint_path, "r") as f:
        checkpoint_data = json.load(f)
    
    # Reconstruct ConversationResult objects from JSON
    results = []
    for r_data in checkpoint_data.get("results", []):
        messages = [
            ConversationMessage(
                role=m["role"],
                content=m["content"],
                turn=m["turn"],
                thinking=m.get("thinking"),
            )
            for m in r_data.get("messages", [])
        ]
        
        result = ConversationResult(
            problem_id=r_data["problem_id"],
            problem=r_data["problem"],
            expected_answer=r_data["expected_answer"],
            student_profile=r_data["student_profile"],
            messages=messages,
            final_turn=r_data.get("final_turn", 0),
            student_solved=r_data.get("student_solved", False),
            judge_evaluation=r_data.get("judge_evaluation"),
            judge_scores=r_data.get("judge_scores"),
        )
        results.append(result)
    
    # Extract metadata
    metadata = {
        "total_problems": checkpoint_data.get("total_problems"),
        "conversations_completed": checkpoint_data.get("conversations_completed"),
        "max_turns": checkpoint_data.get("max_turns"),
        "prompt_type": checkpoint_data.get("prompt_type"),
        "tutor_model_name": checkpoint_data.get("tutor_model_name"),
        "dataset_path": checkpoint_data.get("dataset_path"),
        "checkpoint_interval": checkpoint_data.get("checkpoint_interval"),
    }
    
    return results, metadata


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint file in a run directory.
    
    Args:
        run_dir: Directory containing checkpoint files
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    
    # Find all checkpoint files
    checkpoint_files = list(run_dir.glob("checkpoint_*.json"))
    if not checkpoint_files:
        return None
    
    # Extract checkpoint numbers and find the highest
    def get_checkpoint_number(path: Path) -> int:
        # Extract number from "checkpoint_N.json"
        match = re.search(r"checkpoint_(\d+)\.json", path.name)
        return int(match.group(1)) if match else 0
    
    latest_checkpoint = max(checkpoint_files, key=get_checkpoint_number)
    return latest_checkpoint


def save_checkpoint(
    results: List[ConversationResult],
    checkpoint_number: int,
    output_path: Path,
    max_turns: int,
    total_problems: int,
    prompt_type: str,
    tutor_model_name: str,
    dataset_path: Optional[Path] = None,
    checkpoint_interval: int = 50,
) -> Path:
    """Save a checkpoint file with all results accumulated so far.
    
    Checkpoints are saved in a subdirectory named after the output file (without extension).
    For example, if output_path is "llm_judge_outputs/socratic_eval_20240101_120000.json",
    checkpoints will be saved in "llm_judge_outputs/socratic_eval_20240101_120000/".
    The final results are also saved in the same directory as "results.json".
    
    Args:
        results: List of conversation results completed so far
        checkpoint_number: The conversation number at which this checkpoint is saved
        output_path: Full path to the output JSON file (used to determine checkpoint subdirectory)
        max_turns: Maximum turns per conversation
        total_problems: Total number of problems in the run (num_conversations or dataset size)
        prompt_type: Type of tutor prompt ("slim" or "optimized")
        tutor_model_name: Name or path of the tutor model being evaluated
        dataset_path: Path to the dataset file used
        checkpoint_interval: Checkpoint interval used
        
    Returns:
        Path to the saved checkpoint file
    """
    # Create subdirectory based on output filename (without extension)
    checkpoint_dir = output_path.parent / output_path.stem
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "checkpoint_number": checkpoint_number,
        "conversations_completed": len(results),
        "timestamp": datetime.now().isoformat(),
        "max_turns": max_turns,
        "total_problems": total_problems,
        "prompt_type": prompt_type,
        "tutor_model_name": tutor_model_name,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "checkpoint_interval": checkpoint_interval,
        "results": [
            {
                "problem_id": r.problem_id,
                "problem": r.problem,
                "expected_answer": r.expected_answer,
                "student_profile": r.student_profile,
                "student_solved": r.student_solved,
                "final_turn": r.final_turn,
                "judge_evaluation": r.judge_evaluation,
                "judge_scores": r.judge_scores,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "turn": m.turn,
                        "thinking": m.thinking,
                    }
                    for m in r.messages
                ],
            }
            for r in results
        ],
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_number}.json"
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)
    
    return checkpoint_path
