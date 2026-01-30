"""
Off-policy Socratic Environment for replaying pre-existing conversations.

This module implements an environment that replays conversation trajectories
from MathDial and SocraticMATH datasets for off-policy distillation training.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import tinker
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

# Import ConversationTrajectory from data package
try:
    from data.offpolicy_types import ConversationTrajectory, ConversationMessage
except ImportError:
    # Fallback for when script is run directly
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from data.offpolicy_types import ConversationTrajectory, ConversationMessage


@dataclass
class OffPolicySocraticEnv(Env):
    """
    Environment that replays pre-existing conversation trajectories.
    
    For off-policy distillation, this environment replays conversations from
    MathDial and SocraticMATH datasets. Each tutor response becomes a training
    example with full conversation history up to that point.
    
    For SocraticMATH: First teacher message is skipped as training target but
    included in history for all subsequent training examples.
    """
    
    trajectory: ConversationTrajectory
    tutor_renderer: renderers.Renderer
    tutor_convo_prefix: Optional[List[renderers.Message]] = None
    
    # Internal state
    _current_turn: int = 0
    _tutor_turn_index: int = 0  # Index of current tutor turn in trajectory
    _tutor_turns: List[int] = None  # List of indices where tutor speaks
    _episode_done: bool = False
    
    def __post_init__(self):
        """Initialize tutor turn indices."""
        # Find all tutor turn indices
        # For SocraticMATH: skip first tutor message (index 0) for training
        # For MathDial: include all tutor messages
        self._tutor_turns = []
        skip_first_tutor = (self.trajectory.source == "socraticmath" and 
                           len(self.trajectory.messages) > 0 and
                           self.trajectory.messages[0].role == "tutor")
        
        for idx, msg in enumerate(self.trajectory.messages):
            if msg.role == "tutor":
                if skip_first_tutor and idx == 0:
                    # Skip first tutor message for training, but include in history
                    continue
                self._tutor_turns.append(idx)
        
        if not self._tutor_turns:
            # No tutor turns to train on
            self._episode_done = True
    
    @property
    def stop_condition(self) -> StopCondition:
        """Return stop condition for tutor generation."""
        return self.tutor_renderer.get_stop_sequences()
    
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """
        Initialize the environment with the first observation.
        
        Returns the conversation history up to the first tutor training target.
        """
        if self._episode_done or not self._tutor_turns:
            return (
                tinker.ModelInput.empty(),
                self.stop_condition,
            )
        
        # Get the first tutor turn index
        first_tutor_idx = self._tutor_turns[0]
        
        # Build conversation history up to (but not including) this tutor message
        # Include all messages before this tutor turn
        convo = self._build_conversation_up_to(first_tutor_idx - 1)
        
        observation = self.tutor_renderer.build_generation_prompt(convo)
        return observation, self.stop_condition
    
    def _build_conversation_up_to(self, message_idx: int) -> List[renderers.Message]:
        """
        Build conversation history up to (and including) message_idx.
        
        Args:
            message_idx: Index of the last message to include
            
        Returns:
            List of messages in renderer format
        """
        convo = []
        
        # Add tutor conversation prefix if available
        if self.tutor_convo_prefix:
            convo.extend(self.tutor_convo_prefix)
        else:
            # Default prefix: system prompt with problem
            system_prompt = """You are a friendly and encouraging math tutor helping students learn mathematics. Your role is to guide students to understanding through questions and hints rather than giving direct answers.

Core Principles:
1. Never reveal the final answer directly - guide students to discover it themselves
2. Ask guiding questions that help students discover the solution themselves
3. Validate student thinking by acknowledging correct steps and gently redirecting incorrect approaches
4. Be encouraging - celebrate progress and effort, not just correct answers

Response Guidelines:
- Keep responses concise (2-4 sentences for hints, slightly longer for explanations)
- Use simple language appropriate for middle school students
- Include encouragement when appropriate
- Always end with a question or prompt for the student to continue their thinking

Safety:
- Only answer mathematics questions
- Politely redirect off-topic questions back to math
- Do not engage with inappropriate content"""
            
            convo.append({
                "role": "system",
                "content": system_prompt,
            })
            convo.append({
                "role": "user",
                "content": f"Math Problem: {self.trajectory.question}\n\nPlease help the student work through this problem step by step.",
            })
        
        # Add all messages up to message_idx
        for i in range(message_idx + 1):
            msg = self.trajectory.messages[i]
            # Convert to renderer message format
            role = "user" if msg.role == "student" else "assistant"
            convo.append({
                "role": role,
                "content": msg.content,
            })
        
        return convo
    
    async def step(self, action: Action) -> StepResult:
        """
        Take a step in the environment.
        
        Args:
            action: Tutor's response tokens
            
        Returns:
            StepResult with reward, episode_done, next_observation, metrics
        """
        if self._episode_done:
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "turn": self._current_turn,
                    "episode_length": self._current_turn,
                },
            )
        
        # Parse tutor's response
        tutor_message, parse_success = self.tutor_renderer.parse_response(action)
        tutor_content = renderers.get_text_content(tutor_message)
        
        # Get current tutor turn index in trajectory
        tutor_idx = self._tutor_turns[self._tutor_turn_index]
        
        # Move to next tutor turn
        self._tutor_turn_index += 1
        self._current_turn += 1
        
        # Check if we've processed all tutor turns
        if self._tutor_turn_index >= len(self._tutor_turns):
            self._episode_done = True
            return StepResult(
                reward=0.0,  # Rewards computed by teacher model
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "turn": self._current_turn,
                    "parse_success": float(parse_success),
                    "episode_length": self._current_turn,
                    "tutor_turns_processed": len(self._tutor_turns),
                },
            )
        
        # Build next observation (conversation up to next tutor turn, excluding it)
        next_tutor_idx = self._tutor_turns[self._tutor_turn_index]
        
        # Build conversation history: prefix + all messages up to (but not including) tutor_idx
        # Then add the tutor's generated response, then add student responses up to next tutor
        convo = self._build_conversation_up_to(tutor_idx - 1)
        
        # Add the tutor's generated response (not the trajectory's, to allow exploration)
        convo.append({
            "role": "assistant",
            "content": tutor_content,
        })
        
        # Add any messages between current tutor turn and next tutor turn
        # (typically student responses from the trajectory)
        for i in range(tutor_idx + 1, next_tutor_idx):
            msg = self.trajectory.messages[i]
            role = "user" if msg.role == "student" else "assistant"
            convo.append({
                "role": role,
                "content": msg.content,
            })
        
        next_observation = self.tutor_renderer.build_generation_prompt(convo)
        
        return StepResult(
            reward=0.0,  # Rewards computed by teacher model
            episode_done=False,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={
                "turn": self._current_turn,
                "parse_success": float(parse_success),
                "episode_length": self._current_turn,
            },
        )


@dataclass(frozen=True)
class OffPolicySocraticEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for groups of off-policy Socratic tutoring environments.
    
    Creates environments that replay pre-existing conversation trajectories.
    """
    
    trajectory: ConversationTrajectory
    tutor_renderer: renderers.Renderer
    num_envs: int = 1
    dataset_name: str = "offpolicy_socratic"
    tutor_convo_prefix: Optional[List[renderers.Message]] = None
    
    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments from the trajectory."""
        return [
            OffPolicySocraticEnv(
                trajectory=self.trajectory,
                tutor_renderer=self.tutor_renderer,
                tutor_convo_prefix=self.tutor_convo_prefix,
            )
            for _ in range(self.num_envs)
        ]
    
    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """
        Compute group-level rewards for trajectories.
        
        By default, returns zero rewards. Rewards should be computed
        by a teacher model that evaluates the tutoring quality.
        """
        group_metrics = []
        for traj, env in zip(trajectory_group, env_group):
            episode_metrics = {}
            
            # Cast env to OffPolicySocraticEnv to access its state
            offpolicy_env = env  # env is already OffPolicySocraticEnv
            
            # Extract episode-level metrics
            if len(traj.transitions) == 0:
                episode_metrics["contributed_to_training"] = 0.0
            else:
                episode_metrics["contributed_to_training"] = 1.0
                episode_metrics["episode_length"] = len(traj.transitions)
                
                # Check if episode completed all tutor turns
                final_metrics = traj.transitions[-1].metrics if traj.transitions else {}
                episode_metrics["tutor_turns_processed"] = final_metrics.get("tutor_turns_processed", 0)
            
            group_metrics.append((0.0, episode_metrics))
        
        return group_metrics
    
    def logging_tags(self) -> list[str]:
        """Return tags for logging and aggregation."""
        return [self.dataset_name, "offpolicy", "socratic", "tutoring"]


def load_trajectories_from_jsonl(
    file_path: Path,
    limit: Optional[int] = None,
) -> List[ConversationTrajectory]:
    """
    Load conversation trajectories from a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        limit: Optional limit on number of trajectories to load
        
    Returns:
        List of ConversationTrajectory objects
    """
    trajectories = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(trajectories) >= limit:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Convert to ConversationTrajectory
                messages = [
                    ConversationMessage(
                        role=msg_data["role"],
                        content=msg_data["content"],
                        turn=msg_data["turn"],
                    )
                    for msg_data in data.get("messages", [])
                ]
                
                trajectory = ConversationTrajectory(
                    id=data["id"],
                    question=data["question"],
                    expected_answer=data.get("expected_answer", ""),
                    messages=messages,
                    source=data.get("source", ""),
                    metadata=data.get("metadata", {}),
                )
                
                trajectories.append(trajectory)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Warning: Failed to parse line {line_num} in {file_path}: {e}")
                continue
    
    return trajectories


def create_offpolicy_env_group_builder(
    trajectory: ConversationTrajectory,
    tutor_model_name: str,
    tutor_renderer: renderers.Renderer,
    tutor_tokenizer,
    tutor_convo_prefix: Optional[List[renderers.Message]] = None,
    num_envs: int = 1,
) -> OffPolicySocraticEnvGroupBuilder:
    """
    Factory function to create an OffPolicySocraticEnvGroupBuilder.
    
    Args:
        trajectory: Conversation trajectory to replay
        tutor_model_name: Name of the tutor model (for logging)
        tutor_renderer: Renderer for tutor model
        tutor_tokenizer: Tokenizer for tutor model
        tutor_convo_prefix: Optional conversation prefix for tutor
        num_envs: Number of environments in the group
        
    Returns:
        Configured OffPolicySocraticEnvGroupBuilder instance
    """
    return OffPolicySocraticEnvGroupBuilder(
        trajectory=trajectory,
        tutor_renderer=tutor_renderer,
        num_envs=num_envs,
        tutor_convo_prefix=tutor_convo_prefix,
    )
