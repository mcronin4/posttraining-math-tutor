# Evaluation Package

Comprehensive evaluation suite for math tutoring models. Evaluates models across three dimensions: **math accuracy**, **tutoring quality**, and **safety**.

## Quick Start

```bash
# Download benchmark datasets (one-time setup)
make download-datasets

# Run full benchmark suite
make benchmark-baseline

# Or run individual evaluations
cd packages/eval
python -m uv run python eval_math.py --dataset datasets/gsm8k_test.jsonl
python -m uv run python eval_tutor.py --dataset datasets/gsm8k_test.jsonl
python -m uv run python eval_safety.py --dataset samples/safety_samples.jsonl
```

## Evaluation Metrics

### 1. Math Accuracy (`eval_math.py`)

Measures whether the model produces correct mathematical answers:

- **Exact Match**: Response exactly matches expected answer
- **Numeric Match**: Extracts numbers from response, matches expected numeric value
- **Partial Match**: Response contains correct answer but with extra text

**Usage:**
```bash
python -m uv run python eval_math.py \
    --dataset datasets/gsm8k_test.jsonl \
    --endpoint http://localhost:8000/chat \
    --output outputs/math_results.json \
    --batch-size 20  # Optional: concurrent requests per batch (default: 10)
```

**Expected Dataset Format:**
```json
{"id": "problem_1", "question": "What is 2+2?", "answer": "4", "grade": "1"}
```

### 2. Tutoring Quality (`eval_tutor.py`)

Evaluates pedagogical effectiveness:

- **No Answer Reveal**: Model doesn't give away the final answer (when `dont_reveal_answer=true`)
- **Socratic Step**: Model asks guiding questions or provides hints
- **Appropriate Level**: Language and concepts match the grade level

**Usage:**
```bash
python -m uv run python eval_tutor.py \
    --dataset datasets/gsm8k_test.jsonl \
    --endpoint http://localhost:8000/chat \
    --output outputs/tutor_results.json \
    --batch-size 20  # Optional: concurrent requests per batch (default: 10)
```

**Expected Dataset Format:**
```json
{
  "id": "problem_1",
  "question": "What is 2+2?",
  "answer": "4",
  "grade": "1",
  "prompt": {"mode": "hint", "dont_reveal_answer": true}
}
```

### 3. Safety (`eval_safety.py`)

Checks refusal behavior and inappropriate engagement:

- **Refusal Rate**: Model refuses inappropriate requests
- **Math Focus**: Model redirects off-topic questions back to math

**Usage:**
```bash
python -m uv run python eval_safety.py \
    --dataset samples/safety_samples.jsonl \
    --endpoint http://localhost:8000/chat \
    --output outputs/safety_results.json \
    --batch-size 20  # Optional: concurrent requests per batch (default: 10)
```

## Benchmark Suite (`benchmark.py`)

Orchestrates all evaluations and compares results across models.

**Usage:**
```bash
# Run standard benchmark suite
python -m uv run python benchmark.py \
    --model-name qwen3_8b-baseline \
    --suite standard \
    --endpoint http://localhost:8000/chat \
    --batch-size 20  # Optional: concurrent requests per batch (default: 10)

# Compare with previous results
python -m uv run python benchmark.py \
    --model-name new-model \
    --compare outputs/benchmarks/qwen3_8b-baseline/benchmark_results.json \
    --batch-size 20
```

**Benchmark Suites** (defined in `datasets/dataset_registry.yaml`):
- `quick`: Small sample for fast testing
- `standard`: GSM8K test set (standard benchmark)
- `comprehensive`: GSM8K + MATH datasets (full evaluation)

**Output:**
- Results saved to `outputs/benchmarks/<model-name>/benchmark_results.json`
- Includes metrics from all three evaluation dimensions
- Comparison JSON if `--compare` used

## Dataset Management

### Downloading Standard Datasets

```bash
# Download all datasets
make download-datasets

# Or manually
cd packages/eval
python -m uv run python datasets/download_datasets.py --dataset all
```

**Supported Datasets:**
- **GSM8K**: Grade school math problems (8.5K problems)
- **MATH**: Competition math problems (12.5K problems)

Datasets are downloaded to `packages/eval/datasets/` in JSONL format.

### Dataset Format

All datasets should be JSONL (one JSON object per line):

```json
{"id": "unique_id", "question": "Math problem text", "answer": "Expected answer", "grade": "6"}
```

Optional fields:
- `grade`: Grade level ("K", "1", "2", ..., "12")
- `topic_tags`: Array of curriculum topics
- `prompt`: Object with `mode` and `dont_reveal_answer` for tutoring evaluation

## Architecture

```
eval/
├── benchmark.py          # Main orchestrator
├── eval_math.py         # Math accuracy evaluation
├── eval_tutor.py        # Tutoring quality evaluation
├── eval_safety.py       # Safety evaluation
├── datasets/
│   ├── download_datasets.py    # Dataset downloader
│   ├── dataset_registry.yaml   # Suite definitions
│   └── README.md               # Dataset documentation
└── outputs/             # Results directory
    └── benchmarks/      # Benchmark results by model
```

## Adding New Metrics

1. Create `eval_<metric>.py` with `run_evaluation()` function
2. Add to `benchmark.py` imports and evaluation loop
3. Update `dataset_registry.yaml` if new dataset needed
4. Document in this README

## Performance: Batched Requests

All evaluation scripts support concurrent request batching to speed up benchmarking:

**Default:** `--batch-size 10` (10 concurrent requests per batch)

**Increase batch size for faster benchmarking:**
```bash
# Faster benchmarking (if your API can handle it)
python -m uv run python benchmark.py \
    --model-name qwen3_8b-baseline \
    --batch-size 50 \
    --suite standard

# Or via Makefile
make benchmark-baseline BATCH_SIZE=50
```

**Performance Tips:**
- **Small datasets (<100 problems)**: `--batch-size 10-20` is fine
- **Medium datasets (100-1000 problems)**: `--batch-size 20-50` speeds things up significantly
- **Large datasets (>1000 problems)**: `--batch-size 50-100` for maximum speed
- **If you see errors/timeouts**: Reduce batch size - your API may be rate-limited
- **GPU inference**: Can typically handle larger batches (50-100)
- **CPU inference**: Use smaller batches (10-20) to avoid overwhelming the server

**How it works:**
- Requests are processed in batches concurrently using `asyncio.gather()`
- Small delay (0.5s) between batches prevents overwhelming the server
- Errors in individual requests don't stop the batch - they're captured and reported

## Troubleshooting

**API Connection Errors:**
- Ensure API is running: `make api-dev`
- Check endpoint URL matches API port (default: `http://localhost:8000/chat`)

**Dataset Not Found:**
- Run `make download-datasets` first
- Check `datasets/` directory exists

**Validation Errors (422):**
- Ensure dataset has required fields (`id`, `question`, `answer`)
- Grade should be valid: "K", "1", "2", ..., "12" or null (defaults to "6")

**Slow Evaluation:**
- Use `--suite quick` for faster testing
- Reduce dataset size with `head -n 100 dataset.jsonl > small.jsonl`
