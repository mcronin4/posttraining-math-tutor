#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Script

Skeleton script for fine-tuning a base model on math tutoring data.
This script provides the structure for a production training pipeline.

Usage:
    python sft.py --config configs/sft.yaml

Supported backends (TODO):
    - Local: PyTorch + Transformers + PEFT
    - Cloud: Tinker API, OpenAI fine-tuning, Together AI
    - Distributed: Accelerate, DeepSpeed

Example workflow:
    1. Prepare dataset using packages/data scripts
    2. Configure training in configs/sft.yaml
    3. Run training: python sft.py --config configs/sft.yaml
    4. Evaluate model using packages/eval scripts
    5. Deploy to apps/api
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ModelConfig:
    """Model configuration."""

    base_model: str = "meta-llama/Llama-2-7b-hf"
    model_type: str = "causal_lm"
    use_peft: bool = True
    peft_config: dict = field(
        default_factory=lambda: {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        }
    )


@dataclass
class DataConfig:
    """Dataset configuration."""

    train_file: str = ""
    eval_file: Optional[str] = None
    max_seq_length: int = 2048
    prompt_template: str = "default"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    output_dir: str = "outputs/sft"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    seed: int = 42


@dataclass
class SFTConfig:
    """Complete SFT configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backend: str = "local"  # local, tinker, openai, together


def load_config(config_path: Path) -> SFTConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Parse nested configs
    model_config = ModelConfig(**raw_config.get("model", {}))
    data_config = DataConfig(**raw_config.get("data", {}))
    training_config = TrainingConfig(**raw_config.get("training", {}))

    return SFTConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        backend=raw_config.get("backend", "local"),
    )


# =============================================================================
# Dataset Loading
# =============================================================================


def load_dataset(config: DataConfig) -> tuple[list[dict], Optional[list[dict]]]:
    """
    Load training and evaluation datasets from JSONL files.

    Expected format: JSONL with {prompt, response, grade?, topic_tags?} fields.
    
    Future enhancements:
        - HuggingFace datasets integration
        - Streaming for large datasets
        - Data validation and filtering
    """
    train_data = []
    eval_data = None

    if config.train_file:
        train_path = Path(config.train_file)
        if train_path.exists():
            with open(train_path) as f:
                for line in f:
                    train_data.append(json.loads(line.strip()))
            print(f"Loaded {len(train_data)} training examples")
        else:
            print(f"Warning: Training file {train_path} not found")

    if config.eval_file:
        eval_path = Path(config.eval_file)
        if eval_path.exists():
            eval_data = []
            with open(eval_path) as f:
                for line in f:
                    eval_data.append(json.loads(line.strip()))
            print(f"Loaded {len(eval_data)} evaluation examples")

    return train_data, eval_data


def format_training_example(example: dict, template: str = "default") -> str:
    """
    Format a training example into the model's expected input format.

    TODO: Implement multiple template formats:
        - default: Simple prompt/response
        - chat: Chat-style with system prompt
        - instruct: Instruction-following format
    """
    prompt = example.get("prompt", "")
    response = example.get("response", "")

    if template == "chat":
        return f"<|system|>You are a helpful math tutor.<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>{response}<|end|>"
    elif template == "instruct":
        return f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    else:
        return f"### Human: {prompt}\n\n### Assistant: {response}"


# =============================================================================
# Training Backends
# =============================================================================


class TrainingBackend:
    """Base class for training backends."""

    def __init__(self, config: SFTConfig):
        self.config = config

    def train(
        self, train_data: list[dict], eval_data: Optional[list[dict]] = None
    ) -> dict:
        """Run training. Returns metrics dict."""
        raise NotImplementedError

    def save(self, output_path: Path) -> None:
        """Save the trained model."""
        raise NotImplementedError


class LocalBackend(TrainingBackend):
    """
    Local training backend using PyTorch + Transformers.
    
    Note: This backend is not currently implemented. Use TinkerBackend for training.
    To implement local training, add PyTorch/Transformers integration here.
    """

    def train(
        self, train_data: list[dict], eval_data: Optional[list[dict]] = None
    ) -> dict:
        raise NotImplementedError(
            "Local training backend not implemented. "
            "Use TinkerBackend (set backend: tinker in config) for cloud training."
        )

    def save(self, output_path: Path) -> None:
        raise NotImplementedError("Local backend not implemented.")


class TinkerBackend(TrainingBackend):
    """
    Tinker API backend for cloud-based fine-tuning.

    Uses Tinker's cloud infrastructure for training.
    Requires TINKER_API_KEY environment variable.
    """

    def __init__(self, config: SFTConfig):
        super().__init__(config)
        try:
            from .tinker_client import TinkerClient

            self.tinker = TinkerClient()
        except ImportError:
            raise ImportError("tinker_client module not found")
        except ValueError as e:
            raise ValueError(f"Tinker API key not configured: {e}")

    def train(
        self, train_data: list[dict], eval_data: Optional[list[dict]] = None
    ) -> dict:
        """Train model using Tinker API."""
        print("\n" + "=" * 60)
        print("TINKER TRAINING BACKEND")
        print("=" * 60)
        print(f"Base model: {self.config.model.base_model}")
        print(f"Training examples: {len(train_data)}")
        if eval_data:
            print(f"Eval examples: {len(eval_data)}")
        print()

        # Save datasets to temporary files
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = Path(tmpdir) / "train.jsonl"
            eval_file = Path(tmpdir) / "eval.jsonl" if eval_data else None

            # Write training data
            with open(train_file, "w") as f:
                for example in train_data:
                    f.write(json.dumps(example) + "\n")

            # Write eval data if provided
            eval_file_id = None
            if eval_file and eval_data:
                with open(eval_file, "w") as f:
                    for example in eval_data:
                        f.write(json.dumps(example) + "\n")
                print("📤 Uploading evaluation dataset...")
                eval_dataset = self.tinker.upload_dataset(eval_file)
                eval_file_id = eval_dataset["id"]

            # Upload training dataset
            print("📤 Uploading training dataset...")
            train_dataset = self.tinker.upload_dataset(train_file)
            train_file_id = train_dataset["id"]

            # Create fine-tuning job
            print("🚀 Creating fine-tuning job...")
            job = self.tinker.create_finetuning_job(
                base_model=self.config.model.base_model,
                training_file_id=train_file_id,
                method="sft",
                hyperparameters={
                    "epochs": self.config.training.num_epochs,
                    "batch_size": self.config.training.batch_size,
                    "learning_rate": self.config.training.learning_rate,
                },
                eval_file_id=eval_file_id,
            )

            job_id = job["id"]
            print(f"✅ Job created: {job_id}")

            # Wait for completion
            try:
                final_status = self.tinker.wait_for_completion(job_id)
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "model_endpoint": self.tinker.get_model_endpoint(job_id),
                    "metrics": final_status.get("metrics", {}),
                }
            except Exception as e:
                return {
                    "status": "error",
                    "job_id": job_id,
                    "error": str(e),
                }

    def save(self, output_path: Path) -> None:
        """Save job metadata (model is hosted on Tinker)."""
        output_path.mkdir(parents=True, exist_ok=True)
        config_file = output_path / "tinker_job_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(
                {
                    "backend": "tinker",
                    "model": vars(self.config.model),
                    "training": vars(self.config.training),
                },
                f,
            )
        print(f"Job config saved to {config_file}")


def get_backend(config: SFTConfig) -> TrainingBackend:
    """Get the appropriate training backend."""
    backends = {
        "local": LocalBackend,
        "tinker": TinkerBackend,
    }

    backend_class = backends.get(config.backend, LocalBackend)
    return backend_class(config)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running training",
    )
    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file {args.config} not found")
        sys.exit(1)

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Validate
    print("\nConfiguration:")
    print(f"  Backend: {config.backend}")
    print(f"  Model: {config.model.base_model}")
    print(f"  Data: {config.data.train_file}")
    print(f"  Output: {config.training.output_dir}")

    if args.dry_run:
        print("\n✓ Config validation passed (dry run)")
        return

    # Load data
    print("\nLoading datasets...")
    train_data, eval_data = load_dataset(config.data)

    if not train_data:
        print("Error: No training data found")
        sys.exit(1)

    # Run training
    print(f"\nStarting training with {config.backend} backend...")
    backend = get_backend(config)

    start_time = datetime.now()
    metrics = backend.train(train_data, eval_data)
    duration = datetime.now() - start_time

    # Save
    output_path = Path(config.training.output_dir)
    backend.save(output_path)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration}")
    print(f"Output: {output_path}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()

