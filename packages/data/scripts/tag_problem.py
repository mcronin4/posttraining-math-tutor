#!/usr/bin/env python3
"""
Problem Tagger Script

Heuristic-based tagger that assigns curriculum tags to math problems
based on keyword matching against the Ontario math taxonomy.

Usage:
    # Tag a single problem
    python tag_problem.py --question "What is 2 + 3?"

    # Tag problems from a JSONL file
    python tag_problem.py --input problems.jsonl --output tagged_problems.jsonl

    # Tag with specific grade level hint
    python tag_problem.py --question "Solve x + 5 = 12" --grade 9
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# =============================================================================
# Taxonomy Keywords (embedded for standalone use)
# =============================================================================

TAXONOMY_KEYWORDS = {
    # Basic Operations
    "addition": {
        "keywords": ["add", "plus", "sum", "total", "combine", "+"],
        "grades": ["K", "1", "2", "3", "4", "5", "6"],
    },
    "subtraction": {
        "keywords": ["subtract", "minus", "difference", "take away", "less", "-"],
        "grades": ["K", "1", "2", "3", "4", "5", "6"],
    },
    "multiplication": {
        "keywords": ["multiply", "times", "product", "×", "*", "groups of"],
        "grades": ["2", "3", "4", "5", "6", "7", "8"],
    },
    "division": {
        "keywords": ["divide", "quotient", "÷", "/", "split", "share equally"],
        "grades": ["3", "4", "5", "6", "7", "8"],
    },
    # Number Concepts
    "fractions": {
        "keywords": [
            "fraction",
            "numerator",
            "denominator",
            "half",
            "quarter",
            "third",
            "/",
        ],
        "grades": ["3", "4", "5", "6", "7", "8"],
    },
    "decimals": {
        "keywords": ["decimal", "point", "tenth", "hundredth", "."],
        "grades": ["4", "5", "6", "7", "8"],
    },
    "percent": {
        "keywords": ["percent", "percentage", "%"],
        "grades": ["6", "7", "8", "9"],
    },
    "integers": {
        "keywords": ["integer", "negative", "positive", "opposite", "absolute"],
        "grades": ["6", "7", "8", "9"],
    },
    # Algebra
    "algebra": {
        "keywords": ["variable", "equation", "solve", "x", "y", "unknown"],
        "grades": ["6", "7", "8", "9", "10", "11", "12"],
    },
    "linear_equations": {
        "keywords": ["linear", "slope", "y-intercept", "mx + b", "rate of change"],
        "grades": ["8", "9", "10"],
    },
    "polynomials": {
        "keywords": [
            "polynomial",
            "term",
            "coefficient",
            "like terms",
            "expand",
            "factor",
        ],
        "grades": ["9", "10", "11"],
    },
    "quadratics": {
        "keywords": [
            "quadratic",
            "parabola",
            "vertex",
            "x²",
            "ax² + bx + c",
            "discriminant",
        ],
        "grades": ["10", "11"],
    },
    # Geometry
    "geometry_2d": {
        "keywords": [
            "triangle",
            "rectangle",
            "circle",
            "square",
            "polygon",
            "angle",
        ],
        "grades": ["1", "2", "3", "4", "5", "6", "7", "8"],
    },
    "area_perimeter": {
        "keywords": ["area", "perimeter", "square units", "length", "width"],
        "grades": ["3", "4", "5", "6", "7", "8"],
    },
    "geometry_3d": {
        "keywords": ["volume", "surface area", "cube", "sphere", "cylinder", "prism"],
        "grades": ["5", "6", "7", "8", "9"],
    },
    "trigonometry": {
        "keywords": [
            "sine",
            "cosine",
            "tangent",
            "sin",
            "cos",
            "tan",
            "angle",
            "radian",
        ],
        "grades": ["10", "11", "12"],
    },
    # Calculus
    "limits": {
        "keywords": ["limit", "approaches", "infinity", "continuous", "lim"],
        "grades": ["12"],
    },
    "derivatives": {
        "keywords": [
            "derivative",
            "differentiate",
            "rate of change",
            "d/dx",
            "tangent",
        ],
        "grades": ["12"],
    },
    "integrals": {
        "keywords": ["integral", "integrate", "area under", "antiderivative", "∫"],
        "grades": ["12"],
    },
    # Other
    "patterns": {
        "keywords": ["pattern", "sequence", "next", "rule", "term"],
        "grades": ["K", "1", "2", "3", "4", "5"],
    },
    "measurement": {
        "keywords": [
            "measure",
            "length",
            "weight",
            "capacity",
            "time",
            "metre",
            "gram",
        ],
        "grades": ["K", "1", "2", "3", "4", "5"],
    },
    "data_probability": {
        "keywords": [
            "graph",
            "chart",
            "probability",
            "chance",
            "data",
            "mean",
            "median",
        ],
        "grades": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
    },
}


def tag_problem(
    question: str, grade_hint: Optional[str] = None
) -> dict[str, list[str] | float]:
    """
    Tag a math problem with curriculum topics.

    Args:
        question: The math question/problem text
        grade_hint: Optional grade level to prioritize matching topics

    Returns:
        Dictionary with:
            - tags: List of matched topic tags
            - confidence: Overall confidence score (0-1)
            - grade_suggestions: Suggested grade levels based on content
    """
    question_lower = question.lower()
    matched_tags = []
    grade_votes: dict[str, int] = {}

    for topic, info in TAXONOMY_KEYWORDS.items():
        keywords = info["keywords"]
        topic_grades = info["grades"]

        # Check each keyword
        for keyword in keywords:
            # Handle special regex patterns for operators
            if keyword in ["+", "-", "*", "/", "×", "÷"]:
                pattern = re.escape(keyword)
            else:
                pattern = r"\b" + re.escape(keyword) + r"\b"

            if re.search(pattern, question_lower):
                if topic not in matched_tags:
                    matched_tags.append(topic)

                # Vote for grade levels
                for g in topic_grades:
                    grade_votes[g] = grade_votes.get(g, 0) + 1
                break  # One match per topic is enough

    # Calculate confidence based on number of matches
    confidence = min(len(matched_tags) / 3.0, 1.0)  # Max confidence at 3+ tags

    # If grade hint provided, boost tags that match
    if grade_hint and grade_hint in grade_votes:
        confidence = min(confidence + 0.2, 1.0)

    # Sort grade suggestions by vote count
    sorted_grades = sorted(grade_votes.keys(), key=lambda g: -grade_votes[g])
    grade_suggestions = sorted_grades[:3] if sorted_grades else []

    return {
        "tags": matched_tags,
        "confidence": round(confidence, 2),
        "grade_suggestions": grade_suggestions,
    }


def process_file(input_path: Path, output_path: Path) -> None:
    """Process a JSONL file and add tags to each problem."""
    processed = 0
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                problem = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:50]}...")
                continue

            # Get question text
            question = problem.get("question", problem.get("prompt", ""))
            if not question:
                print(f"Warning: No question found in {problem.get('id', 'unknown')}")
                continue

            # Tag the problem
            grade_hint = problem.get("grade")
            result = tag_problem(question, grade_hint)

            # Update problem with tags
            if "topic_tags" not in problem:
                problem["topic_tags"] = result["tags"]
            problem["tagging_metadata"] = {
                "auto_tags": result["tags"],
                "confidence": result["confidence"],
                "grade_suggestions": result["grade_suggestions"],
            }

            f_out.write(json.dumps(problem) + "\n")
            processed += 1

    print(f"✓ Processed {processed} problems")


def main():
    parser = argparse.ArgumentParser(description="Tag math problems with curriculum topics")
    parser.add_argument("--question", type=str, help="Single question to tag")
    parser.add_argument("--input", type=Path, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output JSONL file")
    parser.add_argument("--grade", type=str, help="Grade level hint")
    args = parser.parse_args()

    if args.question:
        # Tag single question
        result = tag_problem(args.question, args.grade)
        print("\nTagging Results:")
        print("-" * 40)
        print(f"Question: {args.question}")
        print(f"Tags: {', '.join(result['tags']) or 'none'}")
        print(f"Confidence: {result['confidence']}")
        print(f"Grade suggestions: {', '.join(result['grade_suggestions']) or 'unknown'}")

    elif args.input and args.output:
        # Process file
        if not args.input.exists():
            print(f"Error: Input file {args.input} not found")
            sys.exit(1)

        print(f"Processing {args.input}...")
        process_file(args.input, args.output)
        print(f"Output written to {args.output}")

    else:
        parser.print_help()
        print("\nExamples:")
        print('  python tag_problem.py --question "What is 2 + 3?"')
        print("  python tag_problem.py --input problems.jsonl --output tagged.jsonl")


if __name__ == "__main__":
    main()

