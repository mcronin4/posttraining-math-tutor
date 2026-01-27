"""
Run on-policy distillation with Socratic tutoring environment.
"""

import math
from pathlib import Path
from typing import List, Optional, Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.distillation.train_on_policy import Config, main
from tinker_cookbook.distillation.datasets import DistillationDatasetConfig, TeacherConfig
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

from .socratic_env import (
    SocraticEnvGroupBuilder,
    SeedContext,
    load_seeds_from_jsonl,
    create_socratic_env_group_builder,
)


class SocraticDataset(RLDataset):
    """Dataset that provides SocraticEnvGroupBuilder instances from seeds."""

    def __init__(
        self,
        seeds: List[SeedContext],
        batch_size: int,
        env_group_builder_factory: callable,
    ):
        self.seeds = seeds
        self.batch_size = batch_size
        self.env_group_builder_factory = env_group_builder_factory

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.seeds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            self.env_group_builder_factory(seed)
            for seed in self.seeds[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.seeds) / self.batch_size)


class SocraticDatasetBuilder(RLDatasetBuilder):
    """Builder for Socratic tutoring datasets."""

    seeds_file: Path
    tutor_model_name: str
    student_model_name: str
    student_profile: str
    groups_per_batch: int
    num_envs: int = 1
    max_turns: int = 20
    limit: Optional[int] = None
    base_url: Optional[str] = None
    # Support for multiple seed files with sequential limits
    # Format: [(file_path, limit), ...] - loads limit from each file sequentially
    seeds_files: Optional[List[tuple[Path, Optional[int]]]] = None

    async def __call__(self) -> tuple[SocraticDataset, SocraticDataset | None]:
        # Load seeds - support both single file and multiple files
        if self.seeds_files is not None:
            # Load sequentially from multiple files
            seeds = []
            for seeds_file, file_limit in self.seeds_files:
                # Use filename stem as ID prefix to avoid conflicts
                file_seeds = load_seeds_from_jsonl(
                    seeds_file, 
                    limit=file_limit,
                    id_prefix=seeds_file.stem
                )
                seeds.extend(file_seeds)
                print(f"Loaded {len(file_seeds)} seeds from {seeds_file.name} (total: {len(seeds)})")
        else:
            # Single file mode (backward compatible)
            seeds = load_seeds_from_jsonl(self.seeds_file, limit=self.limit)

        # Create ServiceClient for student (simulated student)
        service_client = tinker.ServiceClient(base_url=self.base_url)
        student_client = service_client.create_sampling_client(base_model=self.student_model_name)

        # Get student renderer and tokenizer
        student_tokenizer = get_tokenizer(self.student_model_name)
        student_renderer_name = get_recommended_renderer_name(self.student_model_name)
        student_renderer = renderers.get_renderer(student_renderer_name, tokenizer=student_tokenizer)

        # Create factory function for SocraticEnvGroupBuilder
        def env_group_builder_factory(seed: SeedContext) -> SocraticEnvGroupBuilder:
            return create_socratic_env_group_builder(
                seed=seed,
                tutor_model_name=self.tutor_model_name,
                student_client=student_client,
                student_renderer=student_renderer,
                student_tokenizer=student_tokenizer,
                student_profile=self.student_profile,
                num_envs=self.num_envs,
                max_turns=self.max_turns,
            )

        # Create dataset
        train_dataset = SocraticDataset(
            seeds=seeds,
            batch_size=self.groups_per_batch,
            env_group_builder_factory=env_group_builder_factory,
        )

        # No test dataset for now
        test_dataset = None

        return train_dataset, test_dataset


def build_socratic_config(
    tutor_model_name: str,
    student_model_name: str,
    student_profile: str,
    teacher_model_name: str,
    learning_rate: float,
    max_tokens: int,
    log_path: str,
    groups_per_batch: int = 1,
    num_envs: int = 1,
    max_turns: int = 20,
    seeds_file: Optional[Path] = None,
    seeds_limit: Optional[int] = None,
    seeds_files: Optional[List[tuple[Path, Optional[int]]]] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> Config:
    """
    Build a Config for on-policy distillation using SocraticEnvGroupBuilder.

    The seeds are used to initialize the SimulatedStudent (Kimi-K2) environment.
    Each seed provides a problem and optional initial error that seeds the first
    Kimi output, then the conversation plays out dynamically.

    Note on Socratic prompting:
    - The tutor (student being trained, e.g., Qwen3-8B) is NOT prompted to be Socratic.
    - The teacher (e.g., Qwen3-35B) should be configured with Socratic instructions
      so it can distill this behavior into the unprompted tutor through KL penalty.
    - Teacher prompt configuration is handled separately (not in this function).

    Args:
        tutor_model_name: Name of the tutor model (student being trained)
        student_model_name: Name of the student model (simulated student, e.g., Kimi-K2)
        student_profile: Profile description for simulated student
        teacher_model_name: Name of the teacher model for KL penalty
        learning_rate: Learning rate for training
        max_tokens: Maximum tokens per generation
        log_path: Path for logging and checkpoints
        groups_per_batch: Number of environment groups per batch
        num_envs: Number of environments per group
        max_turns: Maximum conversation turns
        seeds_file: Path to single JSONL file (for backward compatibility)
        seeds_limit: Optional limit on number of seeds to load (when using seeds_file)
        seeds_files: List of (file_path, limit) tuples for sequential loading.
                     Example: [(mathdial_file, 2000), (gsm8k_file, 3000)]
                     Loads limit from each file sequentially.
        base_url: Optional base URL for Tinker service (defaults to None for local)
        **kwargs: Additional config parameters

    Returns:
        Config object configured for Socratic tutoring distillation with
        kl_penalty_coef=1.0 and lora_rank=32
    """
    # Validate arguments
    if seeds_file is None and seeds_files is None:
        raise ValueError("Must provide either seeds_file or seeds_files")
    if seeds_file is not None and seeds_files is not None:
        raise ValueError("Cannot provide both seeds_file and seeds_files")

    # Create dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=SocraticDatasetBuilder(
            seeds_file=seeds_file or Path(""),  # Dummy if using seeds_files
            tutor_model_name=tutor_model_name,
            student_model_name=student_model_name,
            student_profile=student_profile,
            groups_per_batch=groups_per_batch,
            num_envs=num_envs,
            max_turns=max_turns,
            limit=seeds_limit,
            seeds_files=seeds_files,
            base_url=base_url,
        ),
        teacher_config=TeacherConfig(base_model=teacher_model_name),
        groups_per_batch=groups_per_batch,
    )

    # Create config with kl_penalty_coef=1.0 and lora_rank=32
    config = Config(
        learning_rate=learning_rate,
        dataset_configs=[dataset_config],
        model_name=tutor_model_name,
        max_tokens=max_tokens,
        kl_penalty_coef=1.0,
        lora_rank=32,
        log_path=log_path,
        base_url=base_url,
        **kwargs,
    )

    return config
