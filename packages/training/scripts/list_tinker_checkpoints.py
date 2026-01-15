#!/usr/bin/env python3
"""
List Tinker Checkpoints and Available Models

This script lists:
1. Available base models from Tinker server capabilities
2. Available checkpoints from Tinker, including base models and fine-tuned checkpoints

Useful for finding the base Qwen3-8b model path for use with MathTutorBench.

Usage:
    python -m uv run python scripts/list_tinker_checkpoints.py
    
    # List checkpoints for a specific training run
    python -m uv run python scripts/list_tinker_checkpoints.py --run-id <run-id>
    
    # Filter by model name
    python -m uv run python scripts/list_tinker_checkpoints.py --filter qwen
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory (packages/training/)
load_dotenv()

# Add parent directory to path to import from training package
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Try importing Tinker SDK - adjust import based on actual package name
    # Common names: tinker, tinker-ai, tinker_sdk
    try:
        from tinker import ServiceClient
    except ImportError:
        try:
            from tinker_ai import ServiceClient
        except ImportError:
            try:
                from tinker_sdk import ServiceClient
            except ImportError:
                # If none work, try direct import
                import tinker
                ServiceClient = tinker.ServiceClient
except ImportError as e:
    print("❌ Error: Tinker SDK not found.")
    print(f"   Import error: {e}")
    print("\nPlease install the Tinker Python SDK:")
    print("   uv add tinker")
    print("   # or")
    print("   pip install tinker")
    sys.exit(1)


def format_checkpoint_path(checkpoint) -> str:
    """Format checkpoint path for display."""
    # Try to extract path from checkpoint object
    if hasattr(checkpoint, 'tinker_path'):
        return checkpoint.tinker_path
    elif hasattr(checkpoint, 'path'):
        return checkpoint.path
    elif hasattr(checkpoint, 'checkpoint_id'):
        # Construct path from checkpoint_id if needed
        return f"tinker://{checkpoint.checkpoint_id}"
    else:
        return str(checkpoint)


def list_available_models(service_client):
    """List all available base models from Tinker server capabilities."""
    print("\n" + "=" * 80)
    print("AVAILABLE BASE MODELS")
    print("=" * 80)
    
    try:
        capabilities = service_client.get_server_capabilities()
        supported_models = capabilities.supported_models
        
        if not supported_models:
            print("\n   No models found.")
            return
        
        # Handle different return types
        models = supported_models
        if hasattr(supported_models, '__iter__') and not isinstance(supported_models, str):
            models = list(supported_models)
        else:
            models = [supported_models]
        
        print(f"\n✅ Found {len(models)} available model(s):\n")
        
        # Display models
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        # Highlight Qwen models
        qwen_models = [m for m in models if 'qwen' in str(m).lower() or 'Qwen' in str(m)]
        if qwen_models:
            print(f"\n⭐ Qwen models found ({len(qwen_models)}):")
            for model in qwen_models:
                print(f"     {model}")
        
    except Exception as e:
        print(f"\n❌ Error listing available models: {e}")
        print("   This feature may not be available or the API may have changed")


def format_tinker_path_for_inference(checkpoint) -> str:
    """Format checkpoint as Tinker inference path (tinker://...)."""
    # Extract the path that can be used with OpenAI-compatible API
    if hasattr(checkpoint, 'tinker_path'):
        path = checkpoint.tinker_path
    elif hasattr(checkpoint, 'sampler_path'):
        path = checkpoint.sampler_path
    elif hasattr(checkpoint, 'path'):
        path = checkpoint.path
    else:
        return None
    
    # Ensure it starts with tinker://
    if path and not path.startswith('tinker://'):
        if path.startswith('/'):
            path = 'tinker://' + path.lstrip('/')
        else:
            path = 'tinker://' + path
    
    return path


def list_checkpoints(rest_client, run_id: str = None, filter_term: str = None):
    """List and display checkpoints."""
    print("\n" + "=" * 80)
    print("TINKER CHECKPOINTS")
    print("=" * 80)
    
    all_checkpoints = []
    
    if run_id:
        # List checkpoints for a specific training run
        print(f"\n📋 Listing checkpoints for training run: {run_id}")
        try:
            result = rest_client.list_checkpoints(run_id).result()
            if hasattr(result, 'checkpoints'):
                all_checkpoints = result.checkpoints
            elif isinstance(result, list):
                all_checkpoints = result
            else:
                all_checkpoints = [result]
        except Exception as e:
            print(f"❌ Error listing checkpoints for run {run_id}: {e}")
            return
    else:
        # List all user checkpoints
        print("\n📋 Listing all user checkpoints...")
        try:
            result = rest_client.list_user_checkpoints().result()
            if hasattr(result, 'checkpoints'):
                all_checkpoints = result.checkpoints
            elif isinstance(result, list):
                all_checkpoints = result
            else:
                all_checkpoints = [result]
        except Exception as e:
            print(f"❌ Error listing user checkpoints: {e}")
            return
    
    if not all_checkpoints:
        print("   No checkpoints found.")
        return
    
    # Filter if requested
    if filter_term:
        filter_lower = filter_term.lower()
        all_checkpoints = [
            cp for cp in all_checkpoints
            if filter_lower in str(cp).lower() or 
               (hasattr(cp, 'model_name') and filter_lower in cp.model_name.lower()) or
               (hasattr(cp, 'base_model') and filter_lower in str(cp.base_model).lower())
        ]
        print(f"   Filtered to {len(all_checkpoints)} checkpoints matching '{filter_term}'")
    
    print(f"\n✅ Found {len(all_checkpoints)} checkpoint(s)\n")
    
    # Group by training run or model
    checkpoints_by_run = {}
    for checkpoint in all_checkpoints:
        # Try to extract training run ID
        run_id_key = "unknown"
        if hasattr(checkpoint, 'training_run_id'):
            run_id_key = checkpoint.training_run_id
        elif hasattr(checkpoint, 'run_id'):
            run_id_key = checkpoint.run_id
        elif hasattr(checkpoint, 'id'):
            run_id_key = str(checkpoint.id)[:20]  # Truncate if too long
        
        if run_id_key not in checkpoints_by_run:
            checkpoints_by_run[run_id_key] = []
        checkpoints_by_run[run_id_key].append(checkpoint)
    
    # Display checkpoints
    for run_id_key, checkpoints in checkpoints_by_run.items():
        print(f"\n{'─' * 80}")
        print(f"Training Run: {run_id_key}")
        print(f"{'─' * 80}")
        
        # Try to get training run info
        try:
            training_run = rest_client.get_training_run(run_id_key).result()
            if hasattr(training_run, 'base_model'):
                print(f"  Base Model: {training_run.base_model}")
            if hasattr(training_run, 'is_lora'):
                print(f"  LoRA: {training_run.is_lora}")
        except:
            pass
        
        for i, checkpoint in enumerate(checkpoints, 1):
            print(f"\n  Checkpoint {i}:")
            
            # Display checkpoint type
            if hasattr(checkpoint, 'checkpoint_type'):
                print(f"    Type: {checkpoint.checkpoint_type}")
            
            # Display checkpoint ID
            if hasattr(checkpoint, 'checkpoint_id'):
                print(f"    ID: {checkpoint.checkpoint_id}")
            
            # Display model name if available
            if hasattr(checkpoint, 'model_name'):
                print(f"    Model: {checkpoint.model_name}")
            
            # Display Tinker path for inference
            inference_path = format_tinker_path_for_inference(checkpoint)
            if inference_path:
                print(f"    Inference Path: {inference_path}")
                print(f"    ✅ Use this with MathTutorBench:")
                print(f"       model={inference_path}")
            else:
                # Fallback: try to construct path
                path = format_checkpoint_path(checkpoint)
                print(f"    Path: {path}")
    
    # Summary: Find Qwen3-8b base model
    print(f"\n{'=' * 80}")
    print("BASE MODEL SEARCH")
    print("=" * 80)
    
    qwen_checkpoints = [
        cp for cp in all_checkpoints
        if 'qwen' in str(cp).lower() or
           (hasattr(cp, 'model_name') and 'qwen' in cp.model_name.lower()) or
           (hasattr(cp, 'base_model') and 'qwen' in str(cp.base_model).lower())
    ]
    
    if qwen_checkpoints:
        print(f"\n✅ Found {len(qwen_checkpoints)} Qwen-related checkpoint(s):\n")
        for cp in qwen_checkpoints:
            inference_path = format_tinker_path_for_inference(cp)
            if inference_path:
                print(f"  {inference_path}")
                if hasattr(cp, 'checkpoint_type') and cp.checkpoint_type == 'sampler':
                    print(f"    ⭐ This looks like a sampler checkpoint (ready for inference)")
            else:
                print(f"  {format_checkpoint_path(cp)}")
    else:
        print("\n⚠️  No Qwen-related checkpoints found.")
        print("   Try listing checkpoints for a specific training run:")
        print("   python scripts/list_tinker_checkpoints.py --run-id <run-id>")


def main():
    parser = argparse.ArgumentParser(
        description="List Tinker checkpoints for finding base model paths"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="List checkpoints for a specific training run ID",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter checkpoints by model name (e.g., 'qwen')",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Tinker API key (or set TINKER_API_KEY env var)",
    )
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("TINKER_API_KEY")
    if not api_key:
        print("❌ Error: Tinker API key required.")
        print("   Set TINKER_API_KEY environment variable or use --api-key")
        print("\n   You can also create a .env file in packages/training/ with:")
        print("   TINKER_API_KEY=your-key-here")
        sys.exit(1)
    
    # Create service client
    try:
        print("🔌 Connecting to Tinker...")
        service_client = ServiceClient(api_key=api_key)
        rest_client = service_client.create_rest_client()
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Error connecting to Tinker: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your API key is correct")
        print("  2. Verify you have network access")
        print("  3. Check Tinker service status")
        sys.exit(1)
    
    # List available models
    list_available_models(service_client)
    
    # List checkpoints
    list_checkpoints(rest_client, run_id=args.run_id, filter_term=args.filter)
    
    print(f"\n{'=' * 80}")
    print("USAGE WITH MATHTUTORBENCH")
    print("=" * 80)
    print("\nOnce you have a checkpoint path, use it like this:")
    print("\n  python main.py --tasks problem_solving.yaml \\")
    print("    --provider completion_api \\")
    print("    --model_args \\")
    print("      base_url=https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1,\\")
    print("      model=tinker://YOUR_CHECKPOINT_PATH,\\")
    print("      is_chat=True,\\")
    print("      api_key=YOUR_TINKER_API_KEY")
    print()


if __name__ == "__main__":
    main()
