"""
Off-policy distillation dataset builder for Socratic tutoring.

This module provides dataset builders that load pre-existing conversation trajectories
from MathDial and SocraticMATH datasets for off-policy distillation training.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

from training.distillation.offpolicy_env import (
    OffPolicySocraticEnvGroupBuilder,
    ConversationTrajectory,
    load_trajectories_from_jsonl,
    create_offpolicy_env_group_builder,
)

# Re-export for convenience
__all__ = [
    "OffPolicySocraticDataset",
    "OffPolicySocraticDatasetBuilder",
    "ConversationTrajectory",
]


class OffPolicySocraticDataset(RLDataset):
    """Dataset that provides OffPolicySocraticEnvGroupBuilder instances from conversation trajectories."""

    def __init__(
        self,
        trajectories: List[ConversationTrajectory],
        batch_size: int,
        env_group_builder_factory: callable,
    ):
        self.trajectories = trajectories
        self.batch_size = batch_size
        self.env_group_builder_factory = env_group_builder_factory

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.trajectories))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            self.env_group_builder_factory(traj)
            for traj in self.trajectories[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.trajectories) / self.batch_size)


class OffPolicySocraticDatasetBuilder(RLDatasetBuilder):
    """Builder for off-policy Socratic tutoring datasets."""

    conversations_file: Path
    tutor_model_name: str
    groups_per_batch: int
    limit: Optional[int] = None
    base_url: Optional[str] = None
    # Support for multiple conversation files with sequential limits
    # Format: [(file_path, limit), ...] - loads limit from each file sequentially
    conversations_files: Optional[List[tuple[Path, Optional[int]]]] = None

    async def __call__(self) -> tuple[OffPolicySocraticDataset, OffPolicySocraticDataset | None]:
        # Load trajectories - support both single file and multiple files
        if self.conversations_files is not None:
            # Load sequentially from multiple files
            trajectories = []
            for conv_file, file_limit in self.conversations_files:
                file_trajectories = load_trajectories_from_jsonl(
                    conv_file,
                    limit=file_limit,
                )
                trajectories.extend(file_trajectories)
                print(f"Loaded {len(file_trajectories)} trajectories from {conv_file.name} (total: {len(trajectories)})")
        else:
            # Single file mode (backward compatible)
            trajectories = load_trajectories_from_jsonl(
                self.conversations_file,
                limit=self.limit,
            )

        # Get tutor renderer and tokenizer (no student model needed for off-policy)
        tutor_tokenizer = get_tokenizer(self.tutor_model_name)
        tutor_renderer_name = get_recommended_renderer_name(self.tutor_model_name)
        tutor_renderer = renderers.get_renderer(tutor_renderer_name, tokenizer=tutor_tokenizer)

        # Create factory function for OffPolicySocraticEnvGroupBuilder
        def env_group_builder_factory(traj: ConversationTrajectory) -> OffPolicySocraticEnvGroupBuilder:
            return create_offpolicy_env_group_builder(
                trajectory=traj,
                tutor_model_name=self.tutor_model_name,
                tutor_renderer=tutor_renderer,
                tutor_tokenizer=tutor_tokenizer,
            )

        # Create dataset
        train_dataset = OffPolicySocraticDataset(
            trajectories=trajectories,
            batch_size=self.groups_per_batch,
            env_group_builder_factory=env_group_builder_factory,
        )

        # No test dataset for now
        test_dataset = None

        return train_dataset, test_dataset
