#!/usr/bin/env python3
"""
LLM-as-a-Judge Socratic Evaluation

Initializes Tinker sampling clients for LLM-as-a-judge evaluation system:
- Tutor model: The model being evaluated (specified via CLI)
- Student model: Kimi-K2-Thinking (acts as confused student)
- Judge model: Kimi-K2-Thinking (evaluates tutoring trajectories, shares client with student)

Usage:
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B
    
    python socratic_eval_llm_judge.py --tutor-model tinker://checkpoint/path
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Setup path for direct script execution (when not run as module)
# This allows relative imports to work when script is run directly
try:
    from .checkpointing import load_checkpoint, find_latest_checkpoint
    from .model_initialization import initialize_all_models
    from .data_loading import load_gsm8k_dataset
    from .orchestration import run_conversations
except ImportError:
    # Fallback for when script is run directly (not as module)
    import sys
    from pathlib import Path
    # Add packages directory to path so we can import eval
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.llm_judge.checkpointing import load_checkpoint, find_latest_checkpoint
    from eval.llm_judge.model_initialization import initialize_all_models
    from eval.llm_judge.data_loading import load_gsm8k_dataset
    from eval.llm_judge.orchestration import run_conversations

try:
    import tinker
    
    tinker_version = getattr(tinker, '__version__', 'unknown')
    print(f"📦 Using tinker version: {tinker_version}")
    if tinker_version != '0.8.0':
        print(f"⚠️  Warning: Expected tinker 0.8.0, but found {tinker_version}")
        print(f"   This may cause issues with get_tokenizer() method")
except ImportError:
    print("❌ Error: Tinker SDK not found.")
    print("\nPlease install the Tinker Python SDK")
    sys.exit(1)


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for filesystem use by replacing special characters."""
    # Replace slashes with underscores
    sanitized = model_name.replace("/", "_")
    # Replace other filesystem-unsafe characters
    sanitized = sanitized.replace("\\", "_")
    sanitized = sanitized.replace(":", "_")
    sanitized = sanitized.replace("*", "_")
    sanitized = sanitized.replace("?", "_")
    sanitized = sanitized.replace('"', "_")
    sanitized = sanitized.replace("<", "_")
    sanitized = sanitized.replace(">", "_")
    sanitized = sanitized.replace("|", "_")
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Remove multiple consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge Socratic evaluation with Tinker clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run evaluation with a base model name
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim
    
    # Run evaluation with a Tinker checkpoint path
    python socratic_eval_llm_judge.py --tutor-model tinker://checkpoint/path --prompt-type slim
    
    # Run with specific number of conversations
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --num-conversations 10
    
    # Resume from a previous run (subdirectory name within llm_judge/llm_judge_outputs)
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --resume socratic_eval_20240101_120000
    
    # Specify custom output filename or subdirectory/filename
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --output my_results.json
    python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --output experiments/run1.json
        """,
    )
    
    parser.add_argument(
        "--tutor-model",
        type=str,
        required=True,
        help="Tutor model name or Tinker checkpoint path (the model being evaluated)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent.parent / "datasets" / "gsm8k_test_1000.jsonl",
        help="Path to GSM8K dataset JSONL file",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of conversation turns (default: 10)",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=None,
        help="Number of conversations to run (default: all in dataset, or first 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename or subdirectory/filename within llm_judge/llm_judge_outputs (default: socratic_eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["slim", "optimized"],
        required=True,
        help="Type of tutor prompt to use: 'slim' or 'optimized'",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug information about messages being sent to models",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N conversations (default: 50). For 1000 conversations, this creates 20 checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Subdirectory name within llm_judge/llm_judge_outputs to resume from (e.g., socratic_eval_20240101_120000). Will automatically load the latest checkpoint.",
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("❌ Error: Tinker API key required.")
        print("   Set TINKER_API_KEY environment variable or add to .env file")
        sys.exit(1)
    
    # Base output directory is always llm_judge/llm_judge_outputs
    base_output_dir = Path(__file__).parent / "llm_judge_outputs"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle resume mode
    resume_dir = None
    original_num_conversations = args.num_conversations  # Track original value to detect if user provided it
    checkpoint_metadata_for_main = None
    
    if args.resume:
        # Resume path is a subdirectory name within llm_judge/llm_judge_outputs
        resume_dir = base_output_dir / args.resume
        if not resume_dir.exists() or not resume_dir.is_dir():
            print(f"❌ Error: Resume path does not exist or is not a directory: {resume_dir}")
            sys.exit(1)
        
        # Derive output path from resume directory name (used to determine checkpoint directory location)
        output_filename = f"{resume_dir.name}.json"
        args.output = resume_dir.parent / output_filename
        
        print(f"📂 Resuming from: {resume_dir}")
        print(f"📄 Final results will be saved to: {resume_dir / 'results.json'}\n")
        
        # Load checkpoint metadata early to determine correct values for dataset loading
        latest_checkpoint = find_latest_checkpoint(resume_dir)
        if latest_checkpoint:
            _, checkpoint_metadata_for_main = load_checkpoint(latest_checkpoint)
        
        # When resuming, if arguments match defaults, pass None so checkpoint metadata is used
        # This allows checkpoint values to be used when user doesn't explicitly override
        # Defaults: max_turns=10, checkpoint_interval=50, prompt_type="slim"
        if args.max_turns == 10:
            args.max_turns = None
        if args.checkpoint_interval == 50:
            args.checkpoint_interval = None
        # For prompt_type, since it's required, we check if it equals the default "slim"
        # If so, treat it as "not explicitly provided" and let checkpoint metadata override it
        if args.prompt_type == "slim":
            args.prompt_type = None
    else:
        # Set default output path for new runs
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize model name for filesystem use
            sanitized_model = sanitize_model_name(args.tutor_model)
            # Get prompt type (should be available since it's required)
            prompt_type = args.prompt_type if args.prompt_type else "slim"
            args.output = base_output_dir / f"socratic_eval_{sanitized_model}_{prompt_type}_{timestamp}.json"
        else:
            # User provided output filename or subdirectory/filename
            # Construct full path relative to base_output_dir
            output_path = Path(args.output)
            if output_path.is_absolute():
                # If absolute path provided, use it as-is but warn
                print(f"⚠️  Warning: Absolute path provided for --output. Using as-is: {output_path}")
                args.output = output_path
            else:
                # Relative path - construct full path within base_output_dir
                args.output = base_output_dir / output_path
    
    try:
        # Create service client
        print("🔌 Connecting to Tinker...")
        service_client = tinker.ServiceClient(api_key=api_key)
        print("✅ Connected successfully\n")
        
        # Initialize all models
        model_clients = initialize_all_models(
            service_client, args.tutor_model
        )
        
        # Load dataset
        if not args.dataset.exists():
            print(f"❌ Error: Dataset file {args.dataset} not found")
            sys.exit(1)
        
        print(f"📂 Loading dataset: {args.dataset}")
        # Determine dataset limit: use checkpoint's total_problems when resuming if user didn't provide num_conversations
        if args.resume and original_num_conversations is None and checkpoint_metadata_for_main:
            if checkpoint_metadata_for_main.get("total_problems"):
                dataset_limit = checkpoint_metadata_for_main["total_problems"]
                print(f"📋 Using num_conversations from checkpoint for dataset loading: {dataset_limit}")
            else:
                raise ValueError(
                    "Checkpoint missing 'total_problems' metadata. "
                    "Cannot resume without knowing the original target number of conversations. "
                    "Please create a new checkpoint with the updated checkpointing code."
                )
        else:
            dataset_limit = args.num_conversations if args.num_conversations is not None else 1000
        problems = load_gsm8k_dataset(args.dataset, limit=dataset_limit)
        print(f"✅ Loaded {len(problems)} problems\n")
        
        # Run conversations (async)
        results = asyncio.run(
            run_conversations(
                model_clients=model_clients,
                problems=problems,
                max_turns=args.max_turns,
                output_path=args.output,
                num_conversations=args.num_conversations,
                prompt_type=args.prompt_type,
                tutor_model_name=args.tutor_model,
                debug=args.debug,
                checkpoint_interval=args.checkpoint_interval,
                resume_path=resume_dir,
                dataset_path=args.dataset,
            )
        )
        
        # Calculate the actual results path (where orchestration saves results.json)
        checkpoint_dir = args.output.parent / args.output.stem
        final_results_path = checkpoint_dir / "results.json"
        print(f"✅ Evaluation complete! Results saved to {final_results_path}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check your API key is correct")
        print("  2. Verify the tutor model name/path is correct")
        print("  3. Check Tinker service status")
        print("  4. Verify the dataset file path is correct")
        sys.exit(1)


if __name__ == "__main__":
    main()
