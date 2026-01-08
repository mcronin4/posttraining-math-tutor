# Contributing Guide

This guide helps teammates understand the codebase structure and how to contribute effectively.

## Project Structure

```
posttraining-math-tutor/
├── apps/
│   ├── web/              # Next.js frontend (unchanged - UI only)
│   └── api/              # FastAPI backend (unchanged - inference only)
├── packages/
│   ├── core/             # Shared types, schemas, curriculum taxonomy
│   ├── data/             # Data processing and tagging tools
│   ├── eval/             # Evaluation and benchmarking suite
│   └── training/         # Training scripts (SFT, Distillation, RLVR)
└── WORKFLOW.md           # Complete training/benchmarking workflow
```

## Core Workflows

### 1. Benchmarking Models

**Purpose:** Evaluate model performance on standard datasets (GSM8K, MATH)

```bash
# Download benchmark datasets (one-time)
make download-datasets

# Benchmark a model
make benchmark-baseline
# Or with custom suite:
cd packages/eval
python -m uv run python benchmark.py \
    --model-name qwen3:8b-baseline \
    --suite standard \
    --endpoint http://localhost:8000/chat
```

**Key Files:**
- `packages/eval/benchmark.py` - Main benchmarking script
- `packages/eval/eval_math.py` - Math accuracy evaluation
- `packages/eval/eval_tutor.py` - Tutoring quality evaluation
- `packages/eval/eval_safety.py` - Safety/refusal evaluation

### 2. Training Models

**Purpose:** Fine-tune models using Tinker cloud infrastructure

```bash
# Set up Tinker API key
cd packages/training
cp env.example .env
# Edit .env and add TINKER_API_KEY

# Train a model
python -m uv run python sft.py --config configs/sft.yaml
```

**Key Files:**
- `packages/training/sft.py` - Supervised fine-tuning script
- `packages/training/tinker_client.py` - Tinker API integration
- `packages/training/configs/sft.yaml` - Training configuration

### 3. Data Processing

**Purpose:** Tag and prepare datasets for training

```bash
# Tag problems with curriculum taxonomy
cd packages/data
python -m uv run python scripts/tag_problem.py \
    --input ../eval/datasets/gsm8k_test.jsonl \
    --output data/tagged_gsm8k.jsonl \
    --grade 6
```

**Key Files:**
- `packages/data/scripts/tag_problem.py` - Automatic tagging
- `packages/data/scripts/build_taxonomy.py` - Taxonomy builder
- `packages/core/src/curriculum.ts` - Curriculum definitions

## Code Organization

### Evaluation Package (`packages/eval/`)

- **`benchmark.py`** - Orchestrates all evaluations, saves results, compares models
- **`eval_math.py`** - Math accuracy metrics (exact/numeric/partial match)
- **`eval_tutor.py`** - Tutoring rubric (Socratic method, answer reveal, grade appropriateness)
- **`eval_safety.py`** - Safety checks (refusal rates, inappropriate engagement)
- **`datasets/`** - Dataset downloaders and registry

### Training Package (`packages/training/`)

- **`sft.py`** - Main training script, supports Tinker and local backends
- **`tinker_client.py`** - Tinker API client (update with actual endpoints)
- **`configs/`** - Training configurations (YAML)
- **`experiments/`** - Experiment tracking directory

### Data Package (`packages/data/`)

- **`schemas.py`** - Data schemas (TrainingExample, MathProblem, etc.)
- **`scripts/tag_problem.py`** - Heuristic tagger for curriculum alignment
- **`scripts/build_taxonomy.py`** - Taxonomy builder (placeholder for future expansion)

## Adding New Features

### Adding a New Evaluation Metric

1. Create `packages/eval/eval_<metric>.py`
2. Implement `run_evaluation(dataset_path, endpoint, output_path)` function
3. Add to `benchmark.py` to include in benchmark suite
4. Update `dataset_registry.yaml` if new dataset needed

### Adding a New Training Method

1. Create `packages/training/<method>.py` (e.g., `distillation.py`)
2. Inherit from `TrainingBackend` base class
3. Implement `train()` and `save()` methods
4. Add config file in `configs/`
5. Update `WORKFLOW.md` with usage instructions

### Adding a New Dataset

1. Add downloader to `packages/eval/datasets/download_datasets.py`
2. Register in `packages/eval/datasets/dataset_registry.yaml`
3. Update benchmark suites if needed

## Experiment Tracking

All experiments should be saved to `packages/training/experiments/`:

```
experiments/
├── baseline/
│   └── qwen3_8b-baseline/
│       ├── benchmark_results.json
│       └── README.md
├── sft/
│   └── sft_001/
│       ├── benchmark_results.json
│       ├── training_config.yaml
│       └── comparison.json
```

**Naming Convention:**
- Baseline: `baseline/<model-name>/`
- SFT: `sft/sft_<number>_<description>/`
- Distillation: `distillation/dist_<number>_<description>/`
- RLVR: `rlvr/rlvr_<number>_<description>/`

## Code Style

- **Python:** Follow PEP 8, use type hints, docstrings for public functions
- **TypeScript:** Use TypeScript types, avoid `any`
- **Comments:** Explain "why", not "what" - code should be self-documenting
- **Error Handling:** Always handle errors gracefully with helpful messages

## Testing Your Changes

```bash
# Run smoke tests
make eval-smoke

# Test specific evaluation
cd packages/eval
python -m uv run python eval_math.py \
    --dataset samples/math_samples.jsonl \
    --endpoint http://localhost:8000/chat
```

## Common Tasks

### Benchmark a New Model

1. Update `apps/api/.env` with new model name (if using Ollama)
2. Restart API: `make api-dev`
3. Run benchmark: `make benchmark-baseline`
4. Save results to `experiments/`

### Compare Two Models

```bash
cd packages/eval
python -m uv run python benchmark.py \
    --model-name new-model \
    --compare outputs/benchmarks/qwen3_8b-baseline/benchmark_results.json
```

### Prepare Training Data

1. Download/obtain dataset
2. Tag with taxonomy: `python scripts/tag_problem.py --input dataset.jsonl --output tagged.jsonl`
3. Format as JSONL with `{prompt, response, grade, topic_tags}`
4. Use in training config

## Questions?

- See `WORKFLOW.md` for detailed training/benchmarking workflow
- See `README.md` for setup instructions
- Check experiment READMEs in `packages/training/experiments/` for examples

