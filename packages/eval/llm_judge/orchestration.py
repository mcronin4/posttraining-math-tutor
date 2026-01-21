"""Batch processing, checkpointing, and orchestration for running multiple conversations."""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    from .llm_judge_types import ModelClients, ConversationResult
    from .checkpointing import load_checkpoint, find_latest_checkpoint, save_checkpoint
except ImportError:
    # Fallback for when script is run directly (not as module)
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.llm_judge.llm_judge_types import ModelClients, ConversationResult
    from eval.llm_judge.checkpointing import load_checkpoint, find_latest_checkpoint, save_checkpoint

from tqdm import tqdm

from .prompts import get_all_student_profiles
from .conversation import run_conversation, judge_conversation


async def run_single_conversation_with_judge(
    model_clients: ModelClients,
    problem: dict,
    problem_index: int,
    total_problems: int,
    max_turns: int,
    prompt_type: str,
    debug: bool,
    student_profile: str,
) -> ConversationResult:
    """Run a single conversation with judge evaluation. Used as a task for concurrent execution.
    
    Args:
        student_profile: The student persona profile to use for this conversation
    """
    # Create log buffer for this conversation
    log_buffer: List[str] = []
    
    # Add conversation header to log buffer
    problem_id = problem.get("id", "unknown")
    log_buffer.append(f"\n{'=' * 80}")
    log_buffer.append(f"Conversation {problem_index}/{total_problems}: {problem_id}")
    log_buffer.append(f"{'=' * 80}")
    log_buffer.append("Running conversation...")
    
    # Run conversation
    conversation_result = await run_conversation(
        model_clients=model_clients,
        problem=problem,
        student_profile=student_profile,
        max_turns=max_turns,
        prompt_type=prompt_type,
        debug=debug,
        log_buffer=log_buffer,
    )
    
    # Judge evaluation
    log_buffer.append("Evaluating with judge model...")
    judge_eval, judge_scores = await judge_conversation(
        model_clients, conversation_result, debug=debug, log_buffer=log_buffer
    )
    conversation_result.judge_evaluation = judge_eval
    conversation_result.judge_scores = judge_scores
    
    # Add summary to log buffer
    solved_status = "✅ Solved" if conversation_result.student_solved else "❌ Not Solved"
    log_buffer.append(f"{solved_status} | Turns: {conversation_result.final_turn}/{max_turns}")
    if judge_scores:
        score_values = [v for v in judge_scores.values() if isinstance(v, (int, float))]
        overall = sum(score_values) / len(score_values) if score_values else "N/A"
        if isinstance(overall, (int, float)):
            log_buffer.append(f"Judge Score (Average): {overall:.1f}/10")
        else:
            log_buffer.append(f"Judge Score (Average): {overall}")
    log_buffer.append(f"{'=' * 80}\n")
    
    # Store log buffer in conversation result for printing
    conversation_result._log_buffer = log_buffer
    
    return conversation_result


async def run_conversations(
    model_clients: ModelClients,
    problems: List[dict],
    max_turns: Optional[int] = None,
    output_path: Optional[Path] = None,
    num_conversations: Optional[int] = None,
    prompt_type: Optional[str] = None,
    tutor_model_name: Optional[str] = None,
    debug: bool = False,
    checkpoint_interval: Optional[int] = None,
    resume_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
) -> List[ConversationResult]:
    """Run multiple tutoring conversations concurrently and evaluate them.
    
    Args:
        model_clients: Initialized model clients
        problems: List of problem dictionaries
        max_turns: Maximum number of conversation turns
        output_path: Path to save final results
        num_conversations: Limit number of conversations (None for all)
        prompt_type: Type of tutor prompt ("slim" or "optimized")
        tutor_model_name: Name or path of the tutor model being evaluated (required for checkpointing)
        debug: Enable debug logging
        checkpoint_interval: Save checkpoint every N conversations (default: 50)
        resume_path: Path to run subdirectory to resume from (loads latest checkpoint)
        dataset_path: Path to the dataset file used
    """
    # Handle resuming from checkpoint
    results: List[ConversationResult] = []
    completed_problem_ids: set[str] = set()
    start_index = 0
    checkpoint_metadata: Optional[dict] = None
    
    if resume_path:
        # resume_path should already be a full Path object
        if not resume_path.exists() or not resume_path.is_dir():
            raise ValueError(f"Resume path does not exist or is not a directory: {resume_path}")
        
        # Find and load latest checkpoint
        latest_checkpoint = find_latest_checkpoint(resume_path)
        if latest_checkpoint:
            print(f"\n{'=' * 80}")
            print(f"RESUMING FROM CHECKPOINT: {latest_checkpoint.name}")
            print(f"{'=' * 80}\n")
            
            results, checkpoint_metadata = load_checkpoint(latest_checkpoint)
            completed_problem_ids = {r.problem_id for r in results}
            start_index = len(results)
            
            print(f"Loaded {len(results)} completed conversations from checkpoint")
            print(f"Resuming from conversation {start_index + 1}\n")
            
            # Use metadata from checkpoint if not explicitly provided
            if checkpoint_metadata:
                # Load and validate tutor_model_name
                checkpoint_tutor_model = checkpoint_metadata.get("tutor_model_name")
                if checkpoint_tutor_model:
                    if tutor_model_name is None:
                        tutor_model_name = checkpoint_tutor_model
                        print(f"📋 Using tutor_model_name from checkpoint: {tutor_model_name}")
                    elif tutor_model_name != checkpoint_tutor_model:
                        print(f"⚠️  Warning: Tutor model mismatch!")
                        print(f"   Checkpoint has: {checkpoint_tutor_model}")
                        print(f"   Command line has: {tutor_model_name}")
                        print(f"   Using checkpoint value: {checkpoint_tutor_model}")
                        tutor_model_name = checkpoint_tutor_model
                
                # Load total_problems (required)
                if num_conversations is None:
                    if checkpoint_metadata.get("total_problems"):
                        num_conversations = checkpoint_metadata["total_problems"]
                        print(f"📋 Using num_conversations from checkpoint: {num_conversations}")
                    else:
                        raise ValueError(
                            "Checkpoint missing 'total_problems' metadata. "
                            "Cannot resume without knowing the original target number of conversations. "
                            "Please create a new checkpoint with the updated checkpointing code."
                        )
                
                # Load other parameters if not explicitly provided
                if max_turns is None and checkpoint_metadata.get("max_turns"):
                    max_turns = checkpoint_metadata["max_turns"]
                    print(f"📋 Using max_turns from checkpoint: {max_turns}")
                if prompt_type is None and checkpoint_metadata.get("prompt_type"):
                    prompt_type = checkpoint_metadata["prompt_type"]
                    print(f"📋 Using prompt_type from checkpoint: {prompt_type}")
                if dataset_path is None and checkpoint_metadata.get("dataset_path"):
                    dataset_path = Path(checkpoint_metadata["dataset_path"])
                    print(f"📋 Using dataset_path from checkpoint: {dataset_path}")
                if checkpoint_interval is None and checkpoint_metadata.get("checkpoint_interval"):
                    checkpoint_interval = checkpoint_metadata["checkpoint_interval"]
                    print(f"📋 Using checkpoint_interval from checkpoint: {checkpoint_interval}")
        else:
            print(f"⚠️  Warning: No checkpoint found in {resume_path}, starting from beginning\n")
    
    # Set defaults if not provided (and not loaded from checkpoint)
    if max_turns is None:
        max_turns = 10
    if prompt_type is None:
        prompt_type = "slim"
    if checkpoint_interval is None:
        checkpoint_interval = 50
    if tutor_model_name is None:
        raise ValueError("tutor_model_name must be provided either via argument or checkpoint metadata")
    
    if num_conversations:
        problems = problems[:num_conversations]
    
    # Filter out already-completed problems
    remaining_problems = [p for p in problems if p.get("id", "unknown") not in completed_problem_ids]
    total_problems = len(problems)
    remaining_count = len(remaining_problems)
    
    BATCH_SIZE = 395  # Process conversations in batches of 200
    
    print(f"\n{'=' * 80}")
    if resume_path and start_index > 0:
        print(f"RESUMING: {remaining_count} remaining conversations (already completed {start_index})")
    else:
        print(f"RUNNING {total_problems} CONVERSATIONS (max_turns={max_turns}, prompt_type={prompt_type})")
    print(f"Concurrency: {BATCH_SIZE} conversations per batch")
    print(f"Checkpoint interval: Every {checkpoint_interval} conversations")
    print(f"{'=' * 80}\n")
    
    if remaining_count == 0:
        print("✅ All conversations already completed!")
        return results
    
    # Create balanced persona assignments to ensure equal distribution
    # This prevents bias from uneven persona difficulty in small runs
    all_profiles = get_all_student_profiles()
    num_profiles = len(all_profiles)
    balanced_profiles = []
    for i in range(len(remaining_problems)):
        # Round-robin assignment: cycles through personas evenly
        profile = all_profiles[i % num_profiles]
        balanced_profiles.append(profile)
    
    # Create conversation tasks for remaining problems
    all_tasks = []
    for i, problem in enumerate(remaining_problems, 1):
        # Calculate the actual problem index in the original list
        actual_index = start_index + i
        # Assign persona from balanced list
        student_profile = balanced_profiles[i - 1]
        task = asyncio.create_task(
            run_single_conversation_with_judge(
                model_clients=model_clients,
                problem=problem,
                problem_index=actual_index,
                total_problems=total_problems,
                max_turns=max_turns,
                prompt_type=prompt_type,
                debug=debug,
                student_profile=student_profile,
            )
        )
        all_tasks.append(task)
    
    # Process conversations in batches
    completed_count = start_index  # Start from already-completed count
    start_time = time.time()
    previous_checkpoint_path: Optional[Path] = None  # Track previous checkpoint for deletion
    
    # Use tqdm for progress tracking (set initial value if resuming)
    with tqdm(total=total_problems, desc="Conversations", unit="conv", initial=start_index) as pbar:
        for i in range(0, len(all_tasks), BATCH_SIZE):
            batch = all_tasks[i:i + BATCH_SIZE]
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Handle exception
                    problem = remaining_problems[i + j]
                    problem_id = problem.get("id", "unknown")
                    actual_conversation_num = start_index + i + j + 1
                    tqdm.write(f"\n{'=' * 80}")
                    tqdm.write(f"Conversation {actual_conversation_num}/{total_problems}: {problem_id}")
                    tqdm.write(f"{'=' * 80}")
                    tqdm.write(f"ERROR: {str(result)}")
                    tqdm.write(f"{'=' * 80}\n")
                    
                    # Create error result
                    error_result = ConversationResult(
                        problem_id=problem_id,
                        problem=problem.get("question", ""),
                        expected_answer=problem.get("answer", ""),
                        student_profile="",
                        messages=[],
                        final_turn=0,
                        student_solved=False,
                    )
                    error_result.judge_evaluation = f"ERROR: {str(result)}"
                    error_result.judge_scores = {}
                    results.append(error_result)
                else:
                    # Print buffered logs for this conversation
                    if hasattr(result, '_log_buffer') and result._log_buffer:
                        for log_line in result._log_buffer:
                            tqdm.write(log_line)
                    
                    results.append(result)
                
                completed_count += 1
                pbar.update(1)
                
                # Update progress bar with additional metrics
                elapsed = time.time() - start_time
                if completed_count > start_index:  # Only calculate ETA for new conversations
                    new_conversations = completed_count - start_index
                    avg_time = elapsed / new_conversations if new_conversations > 0 else 0
                    remaining = total_problems - completed_count
                    eta_seconds = avg_time * remaining
                    pbar.set_postfix({
                        'avg_time': f"{avg_time:.1f}s",
                        'ETA': f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                    })
                
                # Save checkpoint if needed
                if completed_count % checkpoint_interval == 0 and output_path:
                    checkpoint_path = save_checkpoint(
                        results=results,
                        checkpoint_number=completed_count,
                        output_path=output_path,
                        max_turns=max_turns,
                        total_problems=total_problems,
                        prompt_type=prompt_type,
                        tutor_model_name=tutor_model_name,
                        dataset_path=dataset_path,
                        checkpoint_interval=checkpoint_interval,
                    )
                    tqdm.write(f"✓ Checkpoint saved: {checkpoint_path.name} ({completed_count}/{total_problems} conversations)")
                    
                    # Delete previous checkpoint after successfully saving new one
                    if previous_checkpoint_path and previous_checkpoint_path.exists():
                        try:
                            previous_checkpoint_path.unlink()
                            tqdm.write(f"  → Deleted previous checkpoint: {previous_checkpoint_path.name}")
                        except Exception as e:
                            tqdm.write(f"  ⚠️  Warning: Could not delete previous checkpoint {previous_checkpoint_path.name}: {e}")
                    
                    previous_checkpoint_path = checkpoint_path
    
    # Calculate summary statistics (used for both saving and printing)
    avg_scores = {}
    overall_average = None
    if any(r.judge_scores for r in results):
        score_keys = ["information_bottleneck", "diagnostic_accuracy", "socratic_depth", "math_soundness", "tone"]
        for key in score_keys:
            scores = [r.judge_scores.get(key) for r in results if r.judge_scores and r.judge_scores.get(key) is not None]
            if scores:
                avg_scores[key] = sum(scores) / len(scores)
        
        # Calculate overall average across all categories
        if avg_scores:
            all_scores = [r.judge_scores.get(key) for r in results for key in score_keys if r.judge_scores and r.judge_scores.get(key) is not None]
            if all_scores:
                overall_average = sum(all_scores) / len(all_scores)
    
    # Save final results
    if output_path:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "num_conversations": len(results),
            "max_turns": max_turns,
            "summary_statistics": {
                "problems_solved": sum(1 for r in results if r.student_solved),
                "problems_solved_percentage": sum(1 for r in results if r.student_solved) / len(results) * 100 if results else 0,
                "average_turns": sum(r.final_turn for r in results) / len(results) if results else 0,
                "average_judge_scores": avg_scores,
                "overall_average_score": overall_average,
            },
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
        
        # Save final results in the same directory as checkpoints
        # This keeps all run outputs together in one directory
        checkpoint_dir = output_path.parent / output_path.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_results_path = checkpoint_dir / "results.json"
        
        with open(final_results_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✅ Results saved to {final_results_path}")
        
        # Calculate runtime
        total_runtime = time.time() - start_time
        hours = int(total_runtime // 3600)
        minutes = int((total_runtime % 3600) // 60)
        seconds = int(total_runtime % 60)
        runtime_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        avg_time_per_conversation = total_runtime / len(results) if results else 0
        
        # Generate and save summary statistics as a text file
        summary_stats_path = checkpoint_dir / "summary_statistics.txt"
        with open(summary_stats_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("LLM-AS-A-JUDGE SOCRATIC EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic information
            f.write("EVALUATION SETTINGS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tutor Model: {tutor_model_name}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"Max Turns: {max_turns}\n")
            f.write(f"Total Problems: {len(results)}\n")
            if dataset_path:
                f.write(f"Dataset: {dataset_path}\n")
            f.write("\n")
            
            # Overall statistics
            solved_count = sum(1 for r in results if r.student_solved)
            solved_percentage = solved_count / len(results) * 100 if results else 0
            avg_turns = sum(r.final_turn for r in results) / len(results) if results else 0
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Problems Solved: {solved_count}/{len(results)} ({solved_percentage:.1f}%)\n")
            f.write(f"Problems Not Solved: {len(results) - solved_count}/{len(results)} ({(100 - solved_percentage):.1f}%)\n")
            f.write(f"Average Turns per Conversation: {avg_turns:.2f}\n")
            f.write(f"\nRUNTIME STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Runtime: {runtime_str} ({total_runtime:.2f} seconds)\n")
            f.write(f"Average Time per Conversation: {avg_time_per_conversation:.2f} seconds\n")
            f.write("\n")
            
            # Judge scores
            if avg_scores:
                f.write("AVERAGE JUDGE SCORES (out of 10)\n")
                f.write("-" * 80 + "\n")
                for key, avg_score in avg_scores.items():
                    # Format key nicely (e.g., "information_bottleneck" -> "Information Bottleneck")
                    formatted_key = key.replace("_", " ").title()
                    f.write(f"{formatted_key}: {avg_score:.2f}\n")
                if overall_average is not None:
                    f.write(f"\nOverall Average Score: {overall_average:.2f}\n")
                f.write("\n")
            
            # Turn distribution
            turn_distribution = {}
            for r in results:
                turns = r.final_turn
                turn_distribution[turns] = turn_distribution.get(turns, 0) + 1
            
            if turn_distribution:
                f.write("TURN DISTRIBUTION\n")
                f.write("-" * 80 + "\n")
                for turns in sorted(turn_distribution.keys()):
                    count = turn_distribution[turns]
                    percentage = count / len(results) * 100
                    f.write(f"  {turns} turn(s): {count} conversations ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Score distribution (if available)
            if avg_scores:
                # Calculate score distribution for each category
                score_keys = ["information_bottleneck", "diagnostic_accuracy", "socratic_depth", "math_soundness", "tone"]
                f.write("SCORE DISTRIBUTION (by category)\n")
                f.write("-" * 80 + "\n")
                for key in score_keys:
                    scores = [r.judge_scores.get(key) for r in results if r.judge_scores and r.judge_scores.get(key) is not None]
                    if scores:
                        min_score = min(scores)
                        max_score = max(scores)
                        median_score = sorted(scores)[len(scores) // 2]
                        formatted_key = key.replace("_", " ").title()
                        f.write(f"{formatted_key}:\n")
                        f.write(f"  Min: {min_score:.2f}  Max: {max_score:.2f}  Median: {median_score:.2f}  Mean: {avg_scores[key]:.2f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Summary\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Summary statistics saved to {summary_stats_path}")
        
        # Delete all remaining checkpoints after final results are saved
        checkpoint_dir = output_path.parent / output_path.stem
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.json"))
            if checkpoint_files:
                for checkpoint_file in checkpoint_files:
                    try:
                        checkpoint_file.unlink()
                    except Exception as e:
                        print(f"⚠️  Warning: Could not delete checkpoint {checkpoint_file.name}: {e}")
                if len(checkpoint_files) > 0:
                    print(f"✓ Deleted {len(checkpoint_files)} checkpoint file(s) (final results saved)")
    
    # Print summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    solved_count = sum(1 for r in results if r.student_solved)
    print(f"Problems Solved: {solved_count}/{len(results)} ({solved_count/len(results)*100:.1f}%)")
    avg_turns = sum(r.final_turn for r in results) / len(results) if results else 0
    print(f"Average Turns: {avg_turns:.1f}")
    
    if avg_scores:
        print("\nAverage Judge Scores:")
        for key, avg_score in avg_scores.items():
            print(f"  {key.replace('_', ' ').title()}: {avg_score:.2f}/10")
        if overall_average is not None:
            print(f"\n  Overall Average: {overall_average:.2f}/10")
    
    print(f"{'=' * 80}\n")
    
    return results
