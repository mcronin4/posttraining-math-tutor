"""
Socratic Environment for Tinker RL Training

Implements a Tinker EnvGroupBuilder that uses SimulatedStudent (Kimi-K2) to simulate
student responses. The step() function takes the 8B tutor's output, passes it to Kimi,
and returns Kimi's response as the next observation.

Uses MathDial/GSM8K seeds to initialize the first turn.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
from dotenv import load_dotenv
load_dotenv()

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

# Import SimulatedStudent and related utilities
try:
    from eval.llm_judge.simulated_student import SimulatedStudent
    from eval.llm_judge.llm_judge_types import ConversationMessage
except ImportError:
    # Fallback for when script is run directly (not as module)
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.llm_judge.simulated_student import SimulatedStudent
    from eval.llm_judge.llm_judge_types import ConversationMessage


@dataclass
class SeedContext:
    """Represents a seed context for initializing a tutoring conversation."""
    id: str
    question: str
    expected_answer: str
    initial_error: Optional[str] = None


class SocraticEnv(Env):
    """
    Environment for Socratic tutoring conversations.
    
    The tutor (8B model) interacts with a simulated student (Kimi-K2).
    Each step:
    1. Takes tutor's action (response)
    2. Passes it to SimulatedStudent
    3. Returns student's response as the next observation
    """
    
    def __init__(
        self,
        seed: SeedContext,
        tutor_renderer: renderers.Renderer,
        simulated_student: SimulatedStudent,
        student_profile: str = "You are a middle school student working on word problems. You sometimes get overwhelmed when problems have multiple steps. You need clear, step-by-step guidance.",
        max_turns: int = 20,
        tutor_convo_prefix: Optional[List[renderers.Message]] = None,
    ):
        """
        Initialize the Socratic environment.
        
        Args:
            seed: Seed context with problem, expected answer, and optional initial error
            tutor_renderer: Renderer for the tutor model (8B)
            simulated_student: SimulatedStudent instance (Kimi-K2)
            student_profile: Profile description for the simulated student
            max_turns: Maximum number of conversation turns
            tutor_convo_prefix: Optional conversation prefix for tutor (e.g., system prompt with problem)
        """
        self.seed = seed
        self.tutor_renderer = tutor_renderer
        self.simulated_student = simulated_student
        self.student_profile = student_profile
        self.max_turns = max_turns
        self.tutor_convo_prefix = tutor_convo_prefix or []
        
        # Conversation state
        self.conversation_history: List[ConversationMessage] = []
        self.turn_count = 0
        self.episode_done = False
        self.student_solved = False
    
    @property
    def stop_condition(self) -> StopCondition:
        """Return stop condition for tutor generation."""
        return self.tutor_renderer.get_stop_sequences()
    
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """
        Initialize the environment with the first student message.
        
        Uses the seed's initial_error if available, otherwise generates
        the first student turn naturally.
        """
        # Generate the first student turn using SimulatedStudent
        student_content, _, student_solved = await self.simulated_student.generate_student_turn(
            problem=self.seed.question,
            history=[],  # Empty history for first turn
            seed_error=self.seed.initial_error,
            temperature=0.9,
            max_tokens=8192,
        )
        
        # Record the first student message
        self.conversation_history.append(
            ConversationMessage(
                role="student",
                content=student_content,
                turn=0,
            )
        )
        self.turn_count = 0
        self.student_solved = student_solved
        
        # Check if student solved on first turn - episode ends immediately
        if self.student_solved:
            self.episode_done = True
            # Note: This episode didn't contribute to training (no tutor response)
            # Metrics will be tracked when step() is called with episode_done=True
            return (
                tinker.ModelInput.empty(),
                self.stop_condition,
            )
        
        # Build the observation for the tutor
        # The tutor sees: convo_prefix (system prompt with problem) + student's first message
        convo = self.tutor_convo_prefix + [
            {"role": "user", "content": student_content},
        ]
        observation = self.tutor_renderer.build_generation_prompt(convo)
        
        return observation, self.stop_condition
    
    async def step(self, action: Action) -> StepResult:
        """
        Take a step in the environment.
        
        Args:
            action: Tutor's response tokens
            
        Returns:
            StepResult with:
            - reward: 0 (rewards computed elsewhere, e.g., by judge)
            - episode_done: True if max turns reached or student solved
            - next_observation: Student's response as next observation
            - metrics: Episode metrics
        """
        if self.episode_done:
            # Episode already ended, return empty observation
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "turn": self.turn_count,
                    "solved": self.student_solved,
                    "episode_length": self.turn_count,
                },
            )
        
        # Parse tutor's response
        tutor_message, parse_success = self.tutor_renderer.parse_response(action)
        tutor_content = renderers.get_text_content(tutor_message)
        
        # Record tutor's message
        self.turn_count += 1
        self.conversation_history.append(
            ConversationMessage(
                role="tutor",
                content=tutor_content,
                turn=self.turn_count,
            )
        )
        
        # Check if we've reached max turns (after tutor's turn)
        if self.turn_count >= self.max_turns:
            self.episode_done = True
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "turn": self.turn_count,
                    "solved": False,
                    "max_turns_reached": True,
                    "episode_length": self.turn_count,
                },
            )
        
        # Generate student's response using SimulatedStudent
        student_content, _, student_solved = await self.simulated_student.generate_student_turn(
            problem=self.seed.question,
            history=self.conversation_history,
            seed_error=None,  # Only use seed_error on first turn
            temperature=0.9,
            max_tokens=8192,
        )
        
        # Record student's message
        self.conversation_history.append(
            ConversationMessage(
                role="student",
                content=student_content,
                turn=self.turn_count,
            )
        )
        
        # Check if student solved
        self.student_solved = student_solved
        if self.student_solved:
            self.episode_done = True
            # Return empty observation immediately when solved
            return StepResult(
                reward=0.0,  # Rewards computed by judge/reward model
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "turn": self.turn_count,
                    "solved": True,
                    "parse_success": float(parse_success),
                    "episode_length": self.turn_count,
                },
            )
        
        # Build next observation for tutor
        # The tutor sees: convo_prefix + conversation history including the new student message
        convo = self.tutor_convo_prefix + [
            {"role": "user" if msg.role == "student" else "assistant", "content": msg.content}
            for msg in self.conversation_history
        ]
        next_observation = self.tutor_renderer.build_generation_prompt(convo)
        
        return StepResult(
            reward=0.0,  # Rewards computed by judge/reward model
            episode_done=False,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={
                "turn": self.turn_count,
                "solved": False,
                "parse_success": float(parse_success),
                "episode_length": self.turn_count,
            },
        )


@dataclass(frozen=True)
class SocraticEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for groups of Socratic tutoring environments.
    
    Creates multiple environments from seeds, each using SimulatedStudent (Kimi-K2)
    to simulate student responses.
    """
    
    seed: SeedContext
    tutor_renderer: renderers.Renderer
    simulated_student: SimulatedStudent
    student_profile: str
    num_envs: int = 1
    max_turns: int = 20
    dataset_name: str = "socratic"
    tutor_convo_prefix: Optional[List[renderers.Message]] = None
    
    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments from the seed."""
        return [
            SocraticEnv(
                seed=self.seed,
                tutor_renderer=self.tutor_renderer,
                simulated_student=self.simulated_student,
                student_profile=self.student_profile,
                max_turns=self.max_turns,
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
        by a judge/reward model that evaluates the tutoring quality.
        
        Also aggregates episode-level metrics for tracking training statistics.
        """
        group_metrics = []
        for traj, env in zip(trajectory_group, env_group):
            # Extract episode-level metrics from the trajectory
            episode_metrics = {}
            
            # Cast env to SocraticEnv to access its state
            socratic_env = env  # env is already SocraticEnv
            
            # Check if episode ended on first turn (no tutor response)
            if len(traj.transitions) == 0:
                # Episode ended before any tutor response
                # Check if it was because student solved on first turn
                if socratic_env.student_solved and socratic_env.episode_done:
                    episode_metrics["solved_on_first_turn"] = 1.0
                    episode_metrics["contributed_to_training"] = 0.0
                else:
                    # Some other reason (shouldn't happen, but handle gracefully)
                    episode_metrics["solved_on_first_turn"] = 0.0
                    episode_metrics["contributed_to_training"] = 0.0
            else:
                episode_metrics["solved_on_first_turn"] = 0.0
                episode_metrics["contributed_to_training"] = 1.0
                
                # Check if episode hit max turns
                final_metrics = traj.transitions[-1].metrics if traj.transitions else {}
                if final_metrics.get("max_turns_reached", False):
                    episode_metrics["hit_max_turns"] = 1.0
                else:
                    episode_metrics["hit_max_turns"] = 0.0
                
                # Check if episode ended with solution
                if final_metrics.get("solved", False):
                    episode_metrics["ended_with_solution"] = 1.0
                else:
                    episode_metrics["ended_with_solution"] = 0.0
                
                # Episode length (number of tutor turns)
                episode_metrics["episode_length"] = len(traj.transitions)
            
            group_metrics.append((0.0, episode_metrics))
        
        return group_metrics
    
    def logging_tags(self) -> list[str]:
        """Return tags for logging and aggregation."""
        return [self.dataset_name, "socratic", "tutoring"]


def load_seeds_from_jsonl(
    file_path: Path, 
    limit: Optional[int] = None,
    id_prefix: Optional[str] = None
) -> List[SeedContext]:
    """
    Load seed contexts from a JSONL file.
    
    Supports both MathDial format (question, expected_answer, initial_error)
    and GSM8K format (question, expected_answer, initial_error).
    
    Args:
        file_path: Path to JSONL file
        limit: Optional limit on number of seeds to load
        id_prefix: Optional prefix for seed IDs (useful when loading from multiple files)
        
    Returns:
        List of SeedContext objects
    """
    seeds = []
    # Use filename as default prefix if not provided
    if id_prefix is None:
        id_prefix = file_path.stem
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(seeds) >= limit:
                break
            
            try:
                data = json.loads(line.strip())
                
                # Handle different field names
                question = data.get("question", data.get("problem", ""))
                expected_answer = data.get("expected_answer", data.get("answer", data.get("ground_truth", "")))
                initial_error = data.get("initial_error", data.get("student_incorrect_solution", None))
                
                # Use existing ID if present, otherwise generate with prefix
                seed_id = data.get("id")
                if not seed_id:
                    seed_id = f"{id_prefix}_{line_num:05d}"
                
                # Validate required fields
                if not question:
                    print(f"⚠️  Warning: Skipping line {line_num} in {file_path}: missing 'question' field")
                    continue
                
                if not expected_answer:
                    print(f"⚠️  Warning: Skipping line {line_num} in {file_path}: missing 'expected_answer' field (id: {seed_id})")
                    continue
                
                seeds.append(
                    SeedContext(
                        id=seed_id,
                        question=question,
                        expected_answer=expected_answer,
                        initial_error=initial_error,
                    )
                )
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse line {line_num} in {file_path}: {e}")
                continue
    
    return seeds


def create_socratic_env_group_builder(
    seed: SeedContext,
    tutor_model_name: str,
    student_client: tinker.SamplingClient,
    student_renderer: renderers.Renderer,
    student_tokenizer,
    student_profile: str,
    num_envs: int = 1,
    max_turns: int = 20,
    tutor_convo_prefix: Optional[List[renderers.Message]] = None,
) -> SocraticEnvGroupBuilder:
    """
    Factory function to create a SocraticEnvGroupBuilder.
    
    Args:
        seed: Seed context for the environment
        tutor_model_name: Name of the tutor model (for tokenizer and renderer)
        student_client: Tinker SamplingClient for student (Kimi-K2)
        student_renderer: Renderer for student model
        student_tokenizer: Tokenizer for student model
        student_profile: Profile description for simulated student
        num_envs: Number of environments in the group
        max_turns: Maximum conversation turns
        
    Returns:
        Configured SocraticEnvGroupBuilder instance
    """
    # Get tutor renderer
    tutor_tokenizer = get_tokenizer(tutor_model_name)
    tutor_renderer_name = get_recommended_renderer_name(tutor_model_name)
    tutor_renderer = renderers.get_renderer(tutor_renderer_name, tokenizer=tutor_tokenizer)
    
    # Create SimulatedStudent
    simulated_student = SimulatedStudent(
        client=student_client,
        renderer=student_renderer,
        tokenizer=student_tokenizer,
        student_profile=student_profile,
    )
    
    return SocraticEnvGroupBuilder(
        seed=seed,
        tutor_renderer=tutor_renderer,
        simulated_student=simulated_student,
        student_profile=student_profile,
        num_envs=num_envs,
        max_turns=max_turns,
        tutor_convo_prefix=tutor_convo_prefix,
    )
