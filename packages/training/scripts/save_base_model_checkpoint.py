#!/usr/bin/env python3
"""
Save Base Model Checkpoint for Inference

This script creates a sampler checkpoint from a base model (e.g., Qwen3-8b)
without any fine-tuning. This gives you a checkpoint path that can be used
with MathTutorBench or other inference tools.

Usage:
    python -m uv run python scripts/save_base_model_checkpoint.py
    
    # Specify a different base model
    python -m uv run python scripts/save_base_model_checkpoint.py --base-model qwen/qwen3-8b
    
    # Custom checkpoint name
    python -m uv run python scripts/save_base_model_checkpoint.py --name my_checkpoint
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


# Try importing Tinker SDK - adjust import based on actual package name
try:
    import tinker
except ImportError:
    print("❌ Error: Tinker SDK not found.")
    print("\nPlease install the Tinker Python SDK")
    sys.exit(1)


def save_base_checkpoint(
    base_model: str = "Qwen/Qwen3-8B",
    checkpoint_name: str = "qwen3_8b_base_checkpoint",
    rank: int = 16,
) -> str:
    """
    Save base model weights as a sampler checkpoint.
    
    Args:
        base_model: Base model identifier (e.g., "Qwen/Qwen3-8B")
        checkpoint_name: Name for the checkpoint
        rank: LoRA rank (required even for base model)
    
    Returns:
        Sampler checkpoint path (tinker://...)
    """
    print("=" * 80)
    print("SAVING BASE MODEL CHECKPOINT")
    print("=" * 80)
    print(f"Base Model: {base_model}")
    print(f"Checkpoint Name: {checkpoint_name}")
    print(f"LoRA Rank: {rank}")
    print()
    
    # Get API key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("❌ Error: Tinker API key required.")
        print("   Set TINKER_API_KEY environment variable or add to .env file")
        sys.exit(1)
    
    try:
        # Create service client
        print("🔌 Connecting to Tinker...")
        service_client = tinker.ServiceClient(api_key=api_key)
        print("✅ Connected successfully\n")
        
        # Create training client from base model
        print(f"📦 Creating training client for {base_model}...")
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=rank  # LoRA rank (required even for base model)
        )
        print("✅ Training client created\n")
        
        # Save initial weights to get sampler checkpoint path
        print(f"💾 Saving weights as '{checkpoint_name}'...")
        result = training_client.save_weights_for_sampler(name=checkpoint_name).result()
        
        if hasattr(result, 'path'):
            sampling_path = result.path
        elif hasattr(result, 'sampler_path'):
            sampling_path = result.sampler_path
        elif hasattr(result, 'checkpoint_path'):
            sampling_path = result.checkpoint_path
        else:
            # Try to extract from result object
            sampling_path = str(result)
        
        # Ensure path starts with tinker://
        if not sampling_path.startswith('tinker://'):
            if sampling_path.startswith('/'):
                sampling_path = 'tinker://' + sampling_path.lstrip('/')
            else:
                sampling_path = 'tinker://' + sampling_path
        
        print(f"✅ Checkpoint saved successfully!\n")
        print(f"📋 Sampler Path: {sampling_path}\n")
        
        return sampling_path
        
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your API key is correct")
        print("  2. Verify the base model name is correct")
        print("  3. Check Tinker service status")
        print("  4. Ensure you have permissions to create checkpoints")
        sys.exit(1)


def validate_checkpoint(service_client, checkpoint_path: str) -> bool:
    """
    Validate that a checkpoint exists by trying to list it.
    
    Args:
        service_client: Tinker ServiceClient instance
        checkpoint_path: Checkpoint path to validate (tinker://...)
    
    Returns:
        True if checkpoint exists and is accessible
    """
    print("=" * 80)
    print("VALIDATING CHECKPOINT")
    print("=" * 80)
    print(f"Checkpoint Path: {checkpoint_path}\n")
    
    try:
        # Create REST client
        rest_client = service_client.create_rest_client()
        
        # Try to list user checkpoints and find this one
        print("🔍 Searching for checkpoint...")
        result = rest_client.list_user_checkpoints().result()
        
        if hasattr(result, 'checkpoints'):
            checkpoints = result.checkpoints
        elif isinstance(result, list):
            checkpoints = result
        else:
            checkpoints = [result]
        
        # Check if our checkpoint is in the list
        found = False
        for checkpoint in checkpoints:
            # Try different ways to get the path
            cp_path = None
            if hasattr(checkpoint, 'tinker_path'):
                cp_path = checkpoint.tinker_path
            elif hasattr(checkpoint, 'sampler_path'):
                cp_path = checkpoint.sampler_path
            elif hasattr(checkpoint, 'path'):
                cp_path = checkpoint.path
            
            # Normalize paths for comparison
            if cp_path:
                if not cp_path.startswith('tinker://'):
                    cp_path = 'tinker://' + cp_path.lstrip('/')
                
                if cp_path == checkpoint_path:
                    found = True
                    print(f"✅ Found checkpoint!")
                    if hasattr(checkpoint, 'checkpoint_type'):
                        print(f"   Type: {checkpoint.checkpoint_type}")
                    if hasattr(checkpoint, 'checkpoint_id'):
                        print(f"   ID: {checkpoint.checkpoint_id}")
                    break
        
        if not found:
            print("⚠️  Checkpoint not found in user checkpoints list")
            print("   This might be normal - it may take a moment to appear")
            print("   Or the checkpoint might be accessible via a different method")
            return False
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error validating checkpoint: {e}")
        print("   The checkpoint may still be valid - validation is best-effort")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Save base model checkpoint for inference use"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model identifier (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="qwen3_8b_base_checkpoint",
        help="Name for the checkpoint (default: qwen3_8b_base_checkpoint)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16, required even for base model)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip checkpoint validation step",
    )
    args = parser.parse_args()
    
    # Save checkpoint
    checkpoint_path = save_base_checkpoint(
        base_model=args.base_model,
        checkpoint_name=args.name,
        rank=args.rank,
    )
    
    # Validate checkpoint
    if not args.skip_validation:
        try:
            api_key = os.getenv("TINKER_API_KEY")
            service_client = tinker.ServiceClient(api_key=api_key)
            validate_checkpoint(service_client, checkpoint_path)
        except Exception as e:
            print(f"\n⚠️  Validation failed: {e}")
            print("   Checkpoint was saved but validation encountered an error")
    
    # Display usage information
    print("\n" + "=" * 80)
    print("USAGE WITH MATHTUTORBENCH")
    print("=" * 80)
    print("\nUse this checkpoint path with MathTutorBench:\n")
    print("  python main.py --tasks problem_solving.yaml \\")
    print("    --provider completion_api \\")
    print("    --model_args \\")
    print(f"      base_url=https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1,\\")
    print(f"      model={checkpoint_path},\\")
    print("      is_chat=True,\\")
    print("      api_key=$TINKER_API_KEY")
    print()
    print("Or with your existing evaluation suite:\n")
    print("  cd ../eval")
    print("  python -m uv run python benchmark.py \\")
    print(f"    --model-name {args.name} \\")
    print(f"    --endpoint https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1")
    print()


if __name__ == "__main__":
    main()
