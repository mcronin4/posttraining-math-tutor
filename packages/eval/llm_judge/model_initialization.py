"""Model initialization for LLM-as-a-Judge evaluation system."""

import sys
from pathlib import Path

try:
    from .custom_renderers import TutorStudentKimiRenderer
except ImportError:
    # Fallback for when script is run directly (not as module)
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.llm_judge.custom_renderers import TutorStudentKimiRenderer

try:
    import tinker
    from tinker_cookbook import renderers, tokenizer_utils
    from tinker_cookbook.model_info import get_recommended_renderer_name
except ImportError:
    print("❌ Error: Tinker SDK or tinker_cookbook not found.")
    sys.exit(1)

from .llm_judge_types import ModelClients


def initialize_tutor_model(
    service_client: tinker.ServiceClient,
    tutor_model_name: str,
) -> tuple[tinker.SamplingClient, any, any]:
    """
    Initialize tutor model with tokenizer, renderer, and sampling client.
    
    Args:
        service_client: Tinker ServiceClient instance
        tutor_model_name: Model name or path for tutor model
        
    Returns:
        Tuple of (sampling_client, tokenizer, renderer)
    """
    print(f"📚 Initializing tutor model: {tutor_model_name}")
    
    # Get tokenizer for tutor model
    tokenizer = tokenizer_utils.get_tokenizer(tutor_model_name)
    
    # Get recommended renderer name
    renderer_name = get_recommended_renderer_name(tutor_model_name)
    print(f"   Renderer: {renderer_name}")
    
    # Check if this is a Kimi model - if so, use custom renderer
    if "kimi" in tutor_model_name.lower() or renderer_name == "kimi_k2":
        print(f"   Using custom TutorStudentKimiRenderer for role mapping")
        renderer = TutorStudentKimiRenderer(tokenizer)
    else:
        # For other models (e.g., Qwen), use standard renderer which supports custom roles
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(
        base_model=tutor_model_name
    )
    
    print(f"   ✅ Tutor model initialized\n")
    return sampling_client, tokenizer, renderer


def initialize_student_judge_model(
    service_client: tinker.ServiceClient,
    model_name: str = "moonshotai/Kimi-K2-Thinking",
) -> tuple[tinker.SamplingClient, any, any]:
    """
    Initialize student and judge model (shared client).
    
    Args:
        service_client: Tinker ServiceClient instance
        model_name: Model name for student/judge (default: Kimi-K2-Thinking)
        
    Returns:
        Tuple of (sampling_client, tokenizer, renderer)
    """
    print(f"🎓 Initializing student/judge model: {model_name}")
    
    
    # Create shared sampling client
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = sampling_client.get_tokenizer()

    # Get recommended renderer name (should return "kimi_k2")
    print(f"   Getting recommended renderer name for {model_name}...")
    renderer_name = get_recommended_renderer_name(model_name)

    # Use custom renderer for Kimi models to support tutor/student role mapping
    if "kimi" in model_name.lower() or renderer_name == "kimi_k2":
        print(f"   Using custom TutorStudentKimiRenderer for role mapping")
        renderer = TutorStudentKimiRenderer(tokenizer)
    else:
        # For other models, use standard renderer
        renderer = renderers.get_renderer(renderer_name, tokenizer)

    print(f"   ✅ Student/judge model initialized (shared client)\n")
    return sampling_client, tokenizer, renderer


def initialize_all_models(
    service_client: tinker.ServiceClient,
    tutor_model_name: str,
) -> ModelClients:
    """
    Initialize all three model clients for LLM-as-a-judge evaluation.
    
    Args:
        service_client: Tinker ServiceClient instance
        tutor_model_name: Model name or path for tutor model
        
    Returns:
        ModelClients dataclass containing all initialized clients
    """
    print("=" * 80)
    print("INITIALIZING LLM-AS-A-JUDGE EVALUATION SYSTEM")
    print("=" * 80)
    print()
    
    # Initialize student/judge model (shared)
    student_judge_client, student_judge_tokenizer, student_judge_renderer = (
        initialize_student_judge_model(service_client)
    )

    # Initialize tutor model
    tutor_client, tutor_tokenizer, tutor_renderer = initialize_tutor_model(
        service_client, tutor_model_name
    )
    
    print("=" * 80)
    print("✅ ALL MODELS INITIALIZED")
    print("=" * 80)
    print(f"Tutor Model:      {tutor_model_name}")
    print(f"Student Model:    moonshotai/Kimi-K2-Thinking")
    print(f"Judge Model:      moonshotai/Kimi-K2-Thinking (shared client)")
    print()
    
    return ModelClients(
        tutor_client=tutor_client,
        tutor_tokenizer=tutor_tokenizer,
        tutor_renderer=tutor_renderer,
        tutor_model_name=tutor_model_name,
        student_judge_client=student_judge_client,
        student_judge_tokenizer=student_judge_tokenizer,
        student_judge_renderer=student_judge_renderer,
        student_judge_model_name="moonshotai/Kimi-K2-Thinking",
    )
