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
        # For other models, use standard renderer which supports custom roles
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(
        base_model=tutor_model_name
    )
    
    print(f"   ✅ Tutor model initialized\n")
    return sampling_client, tokenizer, renderer


def initialize_student_model(
    service_client: tinker.ServiceClient,
    model_name: str,
) -> tuple[tinker.SamplingClient, any, any]:
    """
    Initialize student model.
    
    Args:
        service_client: Tinker ServiceClient instance
        model_name: Model name for student
        
    Returns:
        Tuple of (sampling_client, tokenizer, renderer)
    """
    print(f"🎓 Initializing student model: {model_name}")
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = sampling_client.get_tokenizer()

    # Get recommended renderer name
    print(f"   Getting recommended renderer name for {model_name}...")
    renderer_name = get_recommended_renderer_name(model_name)

    # Use custom renderer for Kimi models to support tutor/student role mapping
    if "kimi" in model_name.lower() or renderer_name == "kimi_k2":
        print(f"   Using custom TutorStudentKimiRenderer for role mapping")
        renderer = TutorStudentKimiRenderer(tokenizer)
    else:
        # For other models, use standard renderer which supports custom roles
        renderer = renderers.get_renderer(renderer_name, tokenizer)

    print(f"   ✅ Student model initialized\n")
    return sampling_client, tokenizer, renderer


def initialize_judge_model(
    service_client: tinker.ServiceClient,
    model_name: str,
) -> tuple[tinker.SamplingClient, any, any]:
    """
    Initialize judge model.
    
    Args:
        service_client: Tinker ServiceClient instance
        model_name: Model name for judge
        
    Returns:
        Tuple of (sampling_client, tokenizer, renderer)
    """
    print(f"⚖️  Initializing judge model: {model_name}")
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = sampling_client.get_tokenizer()

    # Get recommended renderer name
    print(f"   Getting recommended renderer name for {model_name}...")
    renderer_name = get_recommended_renderer_name(model_name)

    # Use custom renderer for Kimi models to support tutor/student role mapping
    if "kimi" in model_name.lower() or renderer_name == "kimi_k2":
        print(f"   Using custom TutorStudentKimiRenderer for role mapping")
        renderer = TutorStudentKimiRenderer(tokenizer)
    else:
        # For other models, use standard renderer which supports custom roles
        renderer = renderers.get_renderer(renderer_name, tokenizer)

    print(f"   ✅ Judge model initialized\n")
    return sampling_client, tokenizer, renderer


def initialize_all_models(
    service_client: tinker.ServiceClient,
    tutor_model_name: str,
    student_model_name: str = "moonshotai/Kimi-K2-Thinking",
    judge_model_name: str = "moonshotai/Kimi-K2-Thinking",
) -> ModelClients:
    """
    Initialize all model clients for LLM-as-a-judge evaluation.
    
    Args:
        service_client: Tinker ServiceClient instance
        tutor_model_name: Model name or path for tutor model
        student_model_name: Model name for student (default: Kimi-K2-Thinking)
        judge_model_name: Model name for judge (default: Kimi-K2-Thinking)
        
    Returns:
        ModelClients dataclass containing all initialized clients
    """
    print("=" * 80)
    print("INITIALIZING LLM-AS-A-JUDGE EVALUATION SYSTEM")
    print("=" * 80)
    print()
    
    # Initialize student model
    student_client, student_tokenizer, student_renderer = (
        initialize_student_model(service_client, student_model_name)
    )

    # Initialize judge model
    judge_client, judge_tokenizer, judge_renderer = (
        initialize_judge_model(service_client, judge_model_name)
    )

    # Initialize tutor model
    tutor_client, tutor_tokenizer, tutor_renderer = initialize_tutor_model(
        service_client, tutor_model_name
    )
    
    print("=" * 80)
    print("✅ ALL MODELS INITIALIZED")
    print("=" * 80)
    print(f"Tutor Model:      {tutor_model_name}")
    print(f"Student Model:   {student_model_name}")
    print(f"Judge Model:     {judge_model_name}")
    print()
    
    return ModelClients(
        tutor_client=tutor_client,
        tutor_tokenizer=tutor_tokenizer,
        tutor_renderer=tutor_renderer,
        tutor_model_name=tutor_model_name,
        student_client=student_client,
        student_tokenizer=student_tokenizer,
        student_renderer=student_renderer,
        student_model_name=student_model_name,
        judge_client=judge_client,
        judge_tokenizer=judge_tokenizer,
        judge_renderer=judge_renderer,
        judge_model_name=judge_model_name,
    )
