.PHONY: help install web-dev api-dev eval-smoke eval-math eval-tutor eval-safety benchmark-baseline download-datasets train-sft clean

# Batch size for concurrent requests (increase for faster benchmarking if API can handle it)
BATCH_SIZE ?= 10

# Default target
help:
	@echo "LLM Math Tutor - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install all dependencies"
	@echo "  make web-dev      - Run Next.js frontend (port 3000)"
	@echo "  make api-dev      - Run FastAPI backend (port 8000)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-smoke   - Run all smoke tests"
	@echo "  make eval-math    - Run math accuracy evaluation"
	@echo "  make eval-tutor   - Run tutoring rubric evaluation"
	@echo "  make eval-safety  - Run safety/refusal evaluation"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make download-datasets   - Download benchmark datasets (GSM8K, MATH)"
	@echo "  make benchmark-baseline  - Benchmark baseline model (qwen3:8b)"
	@echo ""
	@echo "Training:"
	@echo "  make train-sft    - Run SFT training via Tinker"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make lint         - Run linters"
	@echo ""
	@echo "See WORKFLOW.md for complete training and benchmarking workflow"

# =============================================================================
# Installation
# =============================================================================

install:
	npm install
	cd apps/api && python -m uv sync
	cd packages/data && python -m uv sync
	cd packages/eval && python -m uv sync
	cd packages/training && python -m uv sync

# =============================================================================
# Development
# =============================================================================

web-dev:
	cd apps/web && npm run dev

api-dev:
	cd apps/api && python -m uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# =============================================================================
# Evaluation
# =============================================================================

eval-smoke: eval-math eval-tutor eval-safety
	@echo ""
	@echo "✓ All smoke tests completed"

eval-math:
	@echo "Running math accuracy evaluation..."
	cd packages/eval && python -m uv run python eval_math.py \
		--dataset samples/math_samples.jsonl \
		--endpoint http://localhost:8000/chat \
		--output outputs/math_results.json

eval-tutor:
	@echo "Running tutor rubric evaluation..."
	cd packages/eval && python -m uv run python eval_tutor.py \
		--dataset samples/tutor_samples.jsonl \
		--endpoint http://localhost:8000/chat \
		--output outputs/tutor_results.json

eval-safety:
	@echo "Running safety/refusal evaluation..."
	cd packages/eval && python -m uv run python eval_safety.py \
		--dataset samples/safety_samples.jsonl \
		--endpoint http://localhost:8000/chat \
		--output outputs/safety_results.json

# =============================================================================
# Benchmarking
# =============================================================================

benchmark-baseline:
	@echo "Running comprehensive benchmark on baseline model..."
	@echo "Using batch size: $(BATCH_SIZE)"
	cd packages/eval && python -m uv run python benchmark.py \
		--model-name qwen3:8b-baseline \
		--endpoint http://localhost:8000/chat \
		--suite standard \
		--output-dir outputs/benchmarks \
		--batch-size $(BATCH_SIZE)

download-datasets:
	@echo "Downloading benchmark datasets (GSM8K, MATH)..."
	cd packages/eval && python -m uv sync --extra datasets && \
		python -m uv run python datasets/download_datasets.py --dataset all

# =============================================================================
# Training
# =============================================================================

train-sft:
	@echo "Running SFT training..."
	cd packages/training && python -m uv run python sft.py --config configs/sft.yaml

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".next" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true
	rm -rf packages/eval/outputs/* 2>/dev/null || true

lint:
	cd apps/api && python -m uv run ruff check src/
	cd apps/web && npm run lint

