# Training Package

Training scripts for fine-tuning math tutoring models using Tinker cloud infrastructure.

## Quick Start

```bash
# Set up Tinker API key
cd packages/training
cp env.example .env
# Edit .env and add: TINKER_API_KEY=your-key-here

# Train a model
python -m uv run python sft.py --config configs/sft.yaml
```

## Training Methods

### Supervised Fine-Tuning (SFT)

Fine-tune a base model on math tutoring examples.

**Configuration:** `configs/sft.yaml`

**Usage:**
```bash
python -m uv run python sft.py --config configs/sft.yaml
```

**Training Data Format:**
JSONL file with examples:
```json
{
  "prompt": "Student question or problem",
  "response": "Tutor response (hint, explanation, etc.)",
  "grade": "6",
  "topic_tags": ["algebra", "linear-equations"]
}
```

### Distillation (Coming Soon)

Knowledge distillation from a teacher model to a student model.

**Status:** Skeleton implementation - see `distillation.py`

### RLVR (Coming Soon)

Reinforcement Learning from Verifier Feedback.

**Status:** Skeleton implementation - see `rlvr.py`

## Configuration

Training configurations are YAML files in `configs/`:

```yaml
backend: tinker  # or "local" (not implemented)

model:
  base_model: "qwen3:8b"
  
data:
  train_file: "data/train.jsonl"
  eval_file: "data/eval.jsonl"
  
training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 2e-5
```

## Tinker Integration

The `TinkerClient` handles all communication with Tinker's API:

- **Dataset Upload**: Upload training/eval datasets
- **Job Creation**: Create fine-tuning jobs
- **Status Monitoring**: Check job progress
- **Model Endpoint**: Get inference endpoint for trained model

**Note:** Update `tinker_client.py` with actual Tinker API endpoints when available.

## Experiment Tracking

Save all experiments to `experiments/`:

```
experiments/
├── baseline/
│   └── qwen3_8b-baseline/
│       └── benchmark_results.json
├── sft/
│   └── sft_001/
│       ├── training_config.yaml
│       ├── benchmark_results.json
│       └── comparison.json
```

**Naming Convention:**
- Baseline: `baseline/<model-name>/`
- SFT: `sft/sft_<number>_<description>/`
- Distillation: `distillation/dist_<number>_<description>/`
- RLVR: `rlvr/rlvr_<number>_<description>/`

## Workflow

1. **Prepare Data**: Tag problems with curriculum taxonomy
   ```bash
   cd ../data
   python -m uv run python scripts/tag_problem.py \
       --input ../eval/datasets/gsm8k_test.jsonl \
       --output data/tagged_train.jsonl
   ```

2. **Create Config**: Copy and edit `configs/sft.yaml`

3. **Train Model**: Run training script
   ```bash
   python -m uv run python sft.py --config configs/sft.yaml
   ```

4. **Benchmark**: Evaluate trained model
   ```bash
   cd ../eval
   python -m uv run python benchmark.py \
       --model-name sft_001 \
       --endpoint <tinker-endpoint-url>
   ```

5. **Compare**: Compare with baseline
   ```bash
   python -m uv run python benchmark.py \
       --model-name sft_001 \
       --compare outputs/benchmarks/qwen3_8b-baseline/benchmark_results.json
   ```

## Architecture

```
training/
├── sft.py              # Main SFT training script
├── distillation.py     # Distillation training (skeleton)
├── rlvr.py            # RLVR training (skeleton)
├── tinker_client.py   # Tinker API client
├── scripts/           # Utility scripts
├── configs/           # Training configurations
│   └── sft.yaml
└── experiments/       # Experiment results
```

## Troubleshooting

**Tinker API Key Not Found:**
- Set `TINKER_API_KEY` in `.env` file or environment variable
- Check `env.example` for format

**Dataset Format Errors:**
- Ensure JSONL format (one JSON object per line)
- Required fields: `prompt`, `response`
- Optional: `grade`, `topic_tags`

**Training Job Fails:**
- Check Tinker API status
- Verify dataset format and size
- Review Tinker API logs/errors

**Local Backend Not Working:**
- Local backend is not implemented - use Tinker backend
- Set `backend: tinker` in config

## Next Steps

1. **Update Tinker Client**: Add actual Tinker API endpoints to `tinker_client.py`
2. **Implement Distillation**: Complete `distillation.py` implementation
3. **Implement RLVR**: Complete `rlvr.py` implementation
4. **Add Evaluation**: Integrate benchmark evaluation into training pipeline

See `WORKFLOW.md` in project root for complete training workflow.
