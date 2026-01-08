# Experiment Tracking

This directory tracks all training experiments and their results.

## Structure

```
experiments/
├── baseline/              # Baseline model benchmarks
│   └── qwen3:8b/
│       ├── benchmark_results.json
│       └── training_config.yaml
├── sft/                   # Supervised fine-tuning experiments
│   └── sft_001/
│       ├── benchmark_results.json
│       ├── training_config.yaml
│       └── comparison.json
├── distillation/          # Knowledge distillation experiments
└── rlvr/                 # RLHF/RLVR experiments
```

## Workflow

1. **Benchmark baseline**: Run `benchmark.py` on base model
2. **Train model**: Run training script, save to experiments/
3. **Benchmark trained model**: Run `benchmark.py` on new model
4. **Compare**: Use `compare_results()` to see improvements

## Naming Convention

- Baseline: `baseline/<model-name>/`
- SFT: `sft/sft_<number>_<description>/`
- Distillation: `distillation/dist_<number>_<description>/`
- RLVR: `rlvr/rlvr_<number>_<description>/`

