# Distillation Training Guide

## Checkpointing and Resuming

### Automatic Checkpointing

Tinker automatically handles checkpointing during distillation training:

- **Sampler weights**: Saved after every update
- **Training weights**: Saved every 20 batches

These checkpoints are stored in the `log_path` directory specified when starting training.

### Resuming Training

#### Resume from Same Run (Automatic)

If you use the **same `log_path`**, Tinker will automatically resume from the most recent checkpoint:

```python
from pathlib import Path

# Get repo root (adjust as needed)
REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Same log_path → automatically resumes
log_path = str(REPO_ROOT / "experiments" / "distillation" / "run-001")
```

#### Start Fresh Training

Use a **different `log_path`** to start training from scratch:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Different log_path → starts fresh
log_path = str(REPO_ROOT / "experiments" / "distillation" / "run-002")
```

#### Resume from Specific Checkpoint

To build on a previous checkpoint (e.g., from SFT), use a different `log_path` and set `load_checkpoint_path`:

```python
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Different log_path for new experiment
log_path = str(REPO_ROOT / "experiments" / "distillation" / "run-from-sft")

# Load from previous checkpoint
load_checkpoint_path = "tinker://<model_id>/000100"  # or path from SFT's checkpoints.jsonl
```

Pass `load_checkpoint_path` via `**kwargs` to `build_socratic_config()`:

```python
config = build_socratic_config(
    tutor_model_name="Qwen3-8B",
    student_model_name="Kimi-K2-Thinking",
    teacher_model_name="Qwen3-35B-A22B-Instruct",
    learning_rate=1e-5,
    max_tokens=2048,
    log_path=log_path,
    load_checkpoint_path=load_checkpoint_path,  # Resume from this checkpoint
    # ... other args
)
```

### Summary

| Scenario | `log_path` | `load_checkpoint_path` | Result |
|----------|-----------|------------------------|--------|
| Resume same run | Same | Not set | Resumes from latest checkpoint |
| Start fresh | Different | Not set | Starts from scratch |
| Build on checkpoint | Different | Set to checkpoint path | Starts from specified checkpoint |
