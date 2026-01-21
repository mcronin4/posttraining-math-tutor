# LLM-as-a-Judge Evaluation

This directory contains the LLM-as-a-Judge evaluation system for Socratic tutoring evaluation. The system runs tutoring conversations between a tutor model and a student model, then uses a judge model to evaluate the quality of the tutoring.

## Running Benchmarks

The main entry point for running evaluations is `socratic_eval_llm_judge.py`.

### Basic Usage

```bash
# Run evaluation with a base model name
uv run socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim

# Run evaluation with a Tinker checkpoint path
uv run socratic_eval_llm_judge.py --tutor-model tinker://checkpoint/path --prompt-type slim

# Run with specific number of conversations
uv run socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --num-conversations 10

# Run with optimized prompt
uv run socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type optimized
```

### Command Line Arguments

- `--tutor-model` (required): Tutor model name or Tinker checkpoint path (the model being evaluated)
- `--prompt-type` (required): Type of tutor prompt to use: `"slim"` or `"optimized"`
- `--dataset`: Path to GSM8K dataset JSONL file (default: `datasets/gsm8k_test_1000.jsonl`)
- `--max-turns`: Maximum number of conversation turns (default: 10)
- `--num-conversations`: Number of conversations to run (default: all in dataset, or first 1000)
- `--output`: Output filename or subdirectory/filename within `llm_judge/llm_judge_outputs` (default: `socratic_eval_<model>_<prompt-type>_<timestamp>.json`)
- `--checkpoint-interval`: Save checkpoint every N conversations (default: 50)
- `--resume`: Subdirectory name within `llm_judge/llm_judge_outputs` to resume from (e.g., `socratic_eval_Qwen_Qwen3-8B_slim_20240101_120000`)
- `--debug`: Print detailed debug information about messages being sent to models

### Examples

```bash
# Get full description of command line args
uv run python socratic_eval_llm_judge.py --help

# Custom output filename
uv run python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --output my_results.json

# Save to subdirectory
uv run python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --output experiments/run1.json

# Custom checkpoint interval (every 100 conversations)
uv run python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --checkpoint-interval 100
```

## Output Directory Naming

The output directory name is automatically generated from the output filename (which defaults to including the model and prompt type). The checkpoint directory uses the same name as the output filename (without the `.json` extension).

### Default Output Filename Format

When `--output` is not specified, the default filename is:
```
socratic_eval_{sanitized_model}_{prompt_type}_{timestamp}.json
```

**Examples:**
- Model: `Qwen/Qwen3-8B`, Prompt: `slim` → Creates directory `socratic_eval_Qwen_Qwen3-8B_slim_20240101_120000/`
- Model: `tinker://checkpoint/path`, Prompt: `optimized` → Creates directory `socratic_eval_tinker___checkpoint_path_optimized_20240101_120000/`

**Model name sanitization:** Special characters in model names (slashes, colons, etc.) are replaced with underscores to ensure filesystem compatibility.

Each run directory contains:
- `results.json` - Final complete results file
- `checkpoint_N.json` - Checkpoint files saved during the run

This naming scheme makes it easy to identify:
- Which model was evaluated
- Which prompt type was used
- When the run was started

### Custom Output Names

You can still provide a custom output filename/path using `--output`:
```bash
# Custom filename (checkpoint directory will be "my_results/")
uv run python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --output my_results.json

# Custom path in subdirectory (checkpoint directory will be "experiments/run1/")
uv run python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --output experiments/run1.json
```

## Concurrency Controls (BATCH_SIZE)

The system processes conversations in batches for concurrent execution. The batch size is controlled by the `BATCH_SIZE` constant in `orchestration.py` (line 203).

**Current setting:** `BATCH_SIZE = 200`

This means:
- Up to 200 conversations are processed concurrently in each batch
- All tasks in a batch run in parallel using `asyncio.gather()`
- After a batch completes, the next batch starts

### Modifying BATCH_SIZE

To change the concurrency level, edit `orchestration.py`:

```python
# In orchestration.py, line 203
BATCH_SIZE = 200  # Change this value to adjust concurrency
```

**Considerations:**
- Tinker actually supports up to 400 concurrent requests. If we are only running one benchmark from one device, we can use 400, but if we are running multiple / running
  things on multiple devices, we should use fewer.

The batch size is displayed when starting a run:
```
Concurrency: 200 conversations per batch
```

## Checkpointing and Resume

The system automatically saves checkpoints during evaluation runs, allowing you to resume from where you left off if a run is interrupted.

### How Checkpointing Works

1. **Checkpoint Location**: Checkpoints are saved in a subdirectory named after the output file (without extension). The final results are also saved in the same directory.
   - Final results: `llm_judge_outputs/socratic_eval_Qwen_Qwen3-8B_slim_20240101_120000/results.json`
   - Checkpoints: `llm_judge_outputs/socratic_eval_Qwen_Qwen3-8B_slim_20240101_120000/checkpoint_N.json`
   
   **Note**: The default output filename includes the tutor model name (sanitized for filesystem) and prompt type for easy identification. All outputs (final results and checkpoints) are kept together in one directory for easy organization.

2. **Checkpoint Interval**: By default, checkpoints are saved every 50 conversations (configurable via `--checkpoint-interval`)

3. **Checkpoint Contents**: Each checkpoint file contains:
   - All completed conversation results up to that point
   - Run metadata (total_problems, max_turns, prompt_type, tutor_model_name, etc.)
   - Timestamp information

4. **Checkpoint Naming**: Checkpoints are named `checkpoint_N.json` where N is the conversation number at which the checkpoint was saved

### Resuming from Checkpoint

To resume a previous run:

```bash
# Resume from a previous run directory
uv run python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type slim --resume socratic_eval_Qwen_Qwen3-8B_slim_20240101_120000
```

**How Resume Works:**

1. **Automatic Checkpoint Detection**: The system automatically finds the latest checkpoint in the specified run directory

2. **Metadata Restoration**: When resuming, the system loads all run configuration from the checkpoint metadata:
   - `total_problems` (required for proper resumption)
   - `max_turns`
   - `prompt_type`
   - `tutor_model_name`
   - `dataset_path`
   - `checkpoint_interval`

3. **Problem Deduplication**: The system tracks which problems have already been completed and skips them

4. **Seamless Continuation**: Only remaining conversations are run, and results are merged with previously completed ones

5. **Parameter Override**: You can override checkpoint parameters by explicitly providing them:
   ```bash
   # Resume but override prompt type
   uv run python socratic_eval_llm_judge.py --tutor-model Qwen/Qwen3-8B --prompt-type optimized --resume socratic_eval_Qwen_Qwen3-8B_slim_20240101_120000
   ```

**Example Resume Flow:**

```
Initial run (interrupted at conversation 250):
  llm_judge_outputs/socratic_eval_Qwen_Qwen3-8B_slim_20240101_120000/
    checkpoint_50.json   (conversations 1-50)
    checkpoint_100.json  (conversations 1-100)
    checkpoint_150.json  (conversations 1-150)
    checkpoint_200.json  (conversations 1-200)
    checkpoint_250.json  (conversations 1-250)  ← latest

Resume command:
  uv run ... --resume socratic_eval_Qwen_Qwen3-8B_slim_20240101_120000

System behavior:
  - Loads checkpoint_250.json (latest)
  - Restores 250 completed conversations
  - Identifies remaining problems (251-1000)
  - Continues from conversation 251
```

### Checkpoint File Structure

Each checkpoint file is a JSON file with the following structure:

```json
{
  "checkpoint_number": 250,
  "conversations_completed": 250,
  "timestamp": "2024-01-01T12:00:00",
  "max_turns": 10,
  "total_problems": 1000,
  "prompt_type": "slim",
  "tutor_model_name": "Qwen/Qwen3-8B",
  "dataset_path": "/path/to/dataset.jsonl",
  "checkpoint_interval": 50,
  "results": [
    {
      "problem_id": "...",
      "problem": "...",
      "expected_answer": "...",
      "student_profile": "...",
      "student_solved": true,
      "final_turn": 5,
      "judge_evaluation": "...",
      "judge_scores": {...},
      "messages": [...]
    },
    ...
  ]
}
```

## Output Files

### Final Results File

The final results are saved as `results.json` in the checkpoint directory (same directory as checkpoint files). The file contains:

- Summary statistics (problems solved, average turns, judge scores)
- Complete conversation results with all messages
- Timestamp and run metadata

### Checkpoint Directory

Each run creates a subdirectory containing all checkpoint files. This allows you to:
- Resume from any checkpoint
- Inspect intermediate results
- Debug issues at specific points in the run

## Troubleshooting

### Checkpoint Not Found

If you get an error about checkpoint not found:
- Verify the resume directory exists: `llm_judge_outputs/<resume_name>/`
- Check that checkpoint files exist in that directory
- Ensure you're using the correct subdirectory name (not the full path)

### Model Mismatch Warning

If you see a tutor model mismatch warning when resuming:
- The checkpoint's tutor model will be used (this is intentional to maintain consistency)
- If you need to evaluate a different model, start a new run

### Missing Metadata

If a checkpoint is missing required metadata (like `total_problems`):
- This indicates an older checkpoint format
- You'll need to start a new run or manually specify `--num-conversations`
