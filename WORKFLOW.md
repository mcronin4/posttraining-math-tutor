# Training and Benchmarking Workflow

This document outlines the complete workflow for benchmarking baseline models and training improved variants.

## Overview

1. **Benchmark Baseline** → Get baseline performance metrics
2. **Prepare Training Data** → Create/format datasets for training
3. **Train Model** → Fine-tune using Tinker (SFT, Distillation, RLVR)
4. **Benchmark Trained Model** → Evaluate improvements
5. **Compare Results** → Analyze what worked

---

## Step 1: Benchmark Baseline Model

**Goal:** Establish baseline performance metrics for `qwen3:8b` against established datasets

### 1.1 Download Benchmark Datasets

**Important:** We benchmark against established datasets like GSM8K and MATH, not just small samples.

```bash
cd packages/eval

# Install dataset dependencies
python -m uv pip install datasets huggingface-hub

# Download standard benchmark datasets
python -m uv run python datasets/download_datasets.py --all

# This downloads:
# - GSM8K test set (~1,319 problems)
# - MATH test set (~5,000 problems)
```

**Datasets are saved to:** `packages/eval/datasets/`

### 1.2 Run Baseline Benchmark

```bash
# Make sure API is running with baseline model
cd apps/api
python -m uv run uvicorn src.main:app --reload --port 8000

# In another terminal, run benchmark with standard suite
cd packages/eval
python -m uv run python benchmark.py \
    --model-name qwen3:8b-baseline \
    --endpoint http://localhost:8000/chat \
    --suite standard \
    --output-dir outputs/benchmarks
```

**Dataset Suites:**
- `--suite quick`: Small samples (10 examples each) - for fast testing
- `--suite standard`: GSM8K test set + samples - **recommended for baseline**
- `--suite comprehensive`: GSM8K + MATH + all datasets - for thorough evaluation

**Results saved to:** `packages/eval/outputs/benchmarks/qwen3_8b-baseline/benchmark_results.json`

**Metrics captured:**
- Math accuracy on GSM8K (exact, numeric, partial match)
- Tutoring quality (Socratic step rate, no answer reveal)
- Safety (refusal rate, inappropriate engagement)

**Expected baseline performance:**
- GSM8K numeric match: ~60-80% (varies by model)
- This gives you a reference point for improvements

---

## Step 2: Prepare Training Data

**Goal:** Create training datasets from various sources

### Option A: Use existing datasets

```bash
# Tag a dataset with taxonomy
cd packages/data
python -m uv run python scripts/tag_problem.py \
    --input ../eval/samples/math_samples.jsonl \
    --output data/tagged_math.jsonl \
    --grade 6
```

### Option B: Create synthetic tutoring data

```python
# Use packages/data scripts to generate tutoring examples
# Format: JSONL with {prompt, response, grade, topic_tags}
```

### Format Requirements

Training data should be JSONL format:
```json
{"prompt": "Student: How do I add fractions?\nMode: hint", "response": "Great question! When adding fractions...", "grade": "6", "topic_tags": ["fractions", "addition"]}
{"prompt": "Student: What is 2+3?\nMode: explain", "response": "Addition is combining quantities...", "grade": "1", "topic_tags": ["addition"]}
```

---

## Step 3: Train Model with Tinker

**Goal:** Fine-tune the model using Tinker's cloud infrastructure

### Setup Tinker API Key

**Option 1: Using .env file (Recommended)**

```bash
cd packages/training
cp env.example .env
# Edit .env and add your API key:
# TINKER_API_KEY=your-api-key-here
```

**Option 2: Environment variable**

```bash
export TINKER_API_KEY="your-api-key-here"
# Or on Windows PowerShell:
# $env:TINKER_API_KEY="your-api-key-here"
```

The training scripts will automatically load the `.env` file from `packages/training/`.

### Supervised Fine-Tuning (SFT)

```bash
cd packages/training

# Update configs/sft.yaml with your settings
# Set backend: tinker
# Set base_model: qwen3:8b
# Set train_file: path/to/your/training_data.jsonl

python -m uv run python sft.py --config configs/sft.yaml
```

**Config example:**
```yaml
backend: tinker
model:
  base_model: qwen3:8b
data:
  train_file: data/training_data.jsonl
  eval_file: data/eval_data.jsonl
training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
```

### Knowledge Distillation

```bash
# TODO: Create distillation script
python -m uv run python distillation.py --config configs/distillation.yaml
```

### RLHF/RLVR

```bash
# TODO: Create RLVR script
python -m uv run python rlvr.py --config configs/rlvr.yaml
```

**After training:**
- Tinker provides a model endpoint URL
- Save this endpoint for benchmarking
- Model is hosted on Tinker (no local download needed)

---

## Step 4: Benchmark Trained Model

**Goal:** Evaluate the fine-tuned model's performance

### Option A: If model is deployed to Tinker endpoint

```bash
cd packages/eval

python -m uv run python benchmark.py \
    --model-name qwen3:8b-sft-001 \
    --endpoint https://api.tinker.ai/models/your-model-id \
    --output-dir outputs/benchmarks \
    --compare outputs/benchmarks/qwen3_8b-baseline/benchmark_results.json
```

### Option B: If model is running locally (via Ollama)

```bash
# Pull fine-tuned model to Ollama (if Tinker supports export)
# ollama pull qwen3:8b-sft-001

# Update apps/api/.env:
# OLLAMA_MODEL=qwen3:8b-sft-001

# Restart API, then benchmark
python -m uv run python benchmark.py \
    --model-name qwen3:8b-sft-001 \
    --endpoint http://localhost:8000/chat \
    --output-dir outputs/benchmarks \
    --compare outputs/benchmarks/qwen3_8b-baseline/benchmark_results.json
```

---

## Step 5: Compare Results

**Goal:** Analyze improvements and regressions

The benchmark script automatically compares if you use `--compare`:

```bash
python -m uv run python benchmark.py \
    --model-name qwen3:8b-sft-001 \
    --compare outputs/benchmarks/qwen3_8b-baseline/benchmark_results.json
```

**Output includes:**
- Improvements: Metrics that got better
- Regressions: Metrics that got worse
- Comparison JSON saved for detailed analysis

---

## Experiment Tracking

**Save experiment metadata:**

```bash
# Create experiment directory
mkdir -p packages/training/experiments/sft/sft_001

# Copy results
cp packages/eval/outputs/benchmarks/qwen3_8b-sft-001/* packages/training/experiments/sft/sft_001/

# Document experiment
cat > packages/training/experiments/sft/sft_001/README.md << EOF
# SFT Experiment 001

**Date:** $(date)
**Base Model:** qwen3:8b
**Training Data:** data/training_data.jsonl (1000 examples)
**Method:** Supervised Fine-Tuning
**Hyperparameters:**
- Epochs: 3
- Batch Size: 4
- Learning Rate: 2.0e-5

**Results:**
- Math Accuracy: X%
- Socratic Step Rate: Y%
- Safety Accuracy: Z%

**Key Findings:**
- ...
EOF
```

---

## Complete Example Workflow

```bash
# 1. Benchmark baseline
cd packages/eval
python -m uv run python benchmark.py \
    --model-name qwen3:8b-baseline \
    --endpoint http://localhost:8000/chat

# 2. Prepare training data
cd ../data
python -m uv run python scripts/tag_problem.py \
    --input ../datasets/gsm8k.jsonl \
    --output data/tagged_gsm8k.jsonl

# 3. Train with Tinker
cd ../training
export TINKER_API_KEY="your-key"
python -m uv run python sft.py --config configs/sft.yaml

# 4. Get model endpoint from Tinker output
# Model endpoint: https://api.tinker.ai/models/abc123

# 5. Benchmark trained model
cd ../eval
python -m uv run python benchmark.py \
    --model-name qwen3:8b-sft-001 \
    --endpoint https://api.tinker.ai/models/abc123 \
    --compare outputs/benchmarks/qwen3_8b-baseline/benchmark_results.json

# 6. Review comparison results
cat outputs/benchmarks/qwen3_8b-sft-001/comparison.json
```

---

## Tips

1. **Always benchmark baseline first** - You need a reference point
2. **Use consistent datasets** - Compare apples to apples
3. **Save experiment configs** - Reproducibility is key
4. **Track costs** - Monitor Tinker credit usage
5. **Iterate small** - Start with small experiments before big training runs

---

## Next Steps

- [ ] Implement Tinker API integration (update `tinker_client.py`)
- [ ] Create distillation training script
- [ ] Create RLVR training script
- [ ] Add experiment tracking dashboard
- [ ] Set up automated benchmarking pipeline

