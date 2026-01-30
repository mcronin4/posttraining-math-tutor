# Data Package

Data processing and curriculum tagging tools for preparing training datasets.

## Quick Start

```bash
# Tag a dataset with curriculum taxonomy
cd packages/data
python -m uv run python scripts/tag_problem.py \
    --input ../eval/datasets/gsm8k_test.jsonl \
    --output data/tagged_gsm8k.jsonl \
    --grade 6
```

## Scripts

### `tag_problem.py`

Automatically tags math problems with Ontario K-12 curriculum taxonomy using keyword matching.

**Usage:**
```bash
python -m uv run python scripts/tag_problem.py \
    --input input.jsonl \
    --output output.jsonl \
    --grade 6 \
    --min-confidence 0.3
```

**Input Format:**
```json
{"id": "problem_1", "question": "Solve 2x + 3 = 7", "answer": "2"}
```

**Output Format:**
```json
{
  "id": "problem_1",
  "question": "Solve 2x + 3 = 7",
  "answer": "2",
  "grade": "6",
  "topic_tags": ["algebra", "linear-equations"],
  "strand": "Algebra",
  "confidence": 0.85
}
```

**Options:**
- `--input`: Input JSONL file path
- `--output`: Output JSONL file path
- `--grade`: Default grade level if not in dataset
- `--min-confidence`: Minimum confidence threshold for tags (0.0-1.0)

### `build_taxonomy.py`

Build or update the curriculum taxonomy from sources.

**Status:** Placeholder - future implementation for expanding taxonomy

### Off-Policy Dataset Processing

Scripts for processing MathDial and SocraticMATH datasets into a unified format for off-policy distillation.

#### `process_mathdial_offpolicy.py`

Downloads and processes MathDial dataset from HuggingFace (`eth-nlped/mathdial`).

**Usage:**
```bash
cd packages/data
uv run python scripts/process_mathdial_offpolicy.py \
    --output mathdial_offpolicy.jsonl \
    --max 1000 \
    --split train
```

**Features:**
- Removes strategy tags like `(probing)`, `(generic)` from teacher messages
- Parses conversation format: `"Teacher: (strategy)message|EOM|Student: message|EOM|..."`
- Extracts question, ground truth, and conversation history
- Preserves metadata (student_profile, etc.)

#### `process_socraticmath_offpolicy.py`

Downloads and processes SocraticMATH dataset from HuggingFace (`facat/Socratic`).

**Usage:**
```bash
cd packages/data
uv run python scripts/process_socraticmath_offpolicy.py \
    --output socraticmath_offpolicy.jsonl \
    --max 1000
```

**Features:**
- Handles Chinese text (preserved as-is)
- Parses JSON array format: `["teacher_msg1", "student_msg1", ..., "ProblemID", "Analysis"]`
- Extracts metadata (ProblemID, Analysis)
- Includes first teacher message in history (skipped for training targets later)

#### `convert_to_offpolicy_format.py`

Validates and combines processed datasets into unified format.

**Usage:**
```bash
cd packages/data
uv run python scripts/convert_to_offpolicy_format.py \
    --input mathdial_offpolicy.jsonl socraticmath_offpolicy.jsonl \
    --output unified_offpolicy.jsonl
```

**Features:**
- Validates conversation trajectories
- Checks role alternation, message validity
- Combines multiple datasets
- Outputs unified JSONL format

**Output Format:**
```json
{
  "id": "mathdial_5000012",
  "question": "What is 2+2?",
  "expected_answer": "4",
  "messages": [
    {"role": "tutor", "content": "What do you think?", "turn": 0},
    {"role": "student", "content": "4", "turn": 1},
    {"role": "tutor", "content": "Correct!", "turn": 2}
  ],
  "source": "mathdial",
  "metadata": {"student_profile": "..."}
}
```

## Curriculum Taxonomy

The taxonomy is defined in `packages/core/src/curriculum.ts` and `packages/core/curriculum/ontario_math_taxonomy.json`.

**Structure:**
- **Strands**: Major math areas (Number, Algebra, Geometry, Data, etc.)
- **Topics**: Specific concepts within strands
- **Keywords**: Terms used for matching problems to topics

**Example:**
```json
{
  "grade": "6",
  "strands": [
    {
      "name": "Algebra",
      "topics": [
        {
          "name": "linear-equations",
          "keywords": ["solve", "equation", "x", "variable", "linear"]
        }
      ]
    }
  ]
}
```

## Data Schemas

See `packages/core/src/schemas.ts` for TypeScript type definitions:
- `TrainingExample`: Format for training data
- `MathProblem`: Format for math problems
- `GradeLevel`: Valid grade levels ("K", "1", "2", ..., "12")

## Workflow

1. **Download Dataset**: Use `packages/eval/datasets/download_datasets.py`
2. **Tag Problems**: Run `tag_problem.py` to add curriculum tags
3. **Format for Training**: Ensure `prompt` and `response` fields exist
4. **Use in Training**: Reference tagged dataset in training config

## Architecture

```
data/
├── scripts/
│   ├── tag_problem.py      # Automatic tagging tool
│   └── build_taxonomy.py   # Taxonomy builder (placeholder)
└── data/                    # Processed datasets (gitignored)
```

## Troubleshooting

**No Tags Found:**
- Check if keywords match problem text
- Lower `--min-confidence` threshold
- Verify grade level matches taxonomy

**Tagging Too Slow:**
- Process in batches
- Use `head -n 1000 input.jsonl > sample.jsonl` for testing

**Missing Curriculum Topics:**
- Update `ontario_math_taxonomy.json` with new topics/keywords
- Run `build_taxonomy.py` (when implemented) to rebuild
