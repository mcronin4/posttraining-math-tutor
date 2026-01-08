# Benchmark Datasets

This directory contains scripts and configurations for downloading and preparing established math datasets for benchmarking.

## Supported Datasets

### MathBench
- **Description:** Comprehensive math evaluation covering K-12 topics
- **Size:** ~10,000 problems
- **Format:** JSONL with {question, answer, grade, topic}
- **Download:** Use `download_datasets.py`

### GSM8K
- **Description:** Grade School Math 8K problems
- **Size:** ~8,000 problems
- **Format:** JSONL with {question, answer}
- **Source:** HuggingFace `gsm8k`

### MATH
- **Description:** Competition-level math problems
- **Size:** ~12,500 problems
- **Format:** JSONL with {problem, solution, level}
- **Source:** HuggingFace `hendrycks/competition_math`

### AQuA-RAT
- **Description:** Algebraic word problems with reasoning
- **Size:** ~100,000 problems
- **Format:** JSONL with {question, options, correct, rationale}

## Usage

```bash
# Download all datasets
python download_datasets.py --all

# Download specific dataset
python download_datasets.py --dataset gsm8k

# Prepare dataset for benchmarking
python prepare_dataset.py --input gsm8k_raw.jsonl --output gsm8k_benchmark.jsonl
```

