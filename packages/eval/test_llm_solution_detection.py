#!/usr/bin/env python3
"""
Test script for LLM-based solution detection.

Tests lightweight LLM models (Llama-3.2-3B and Llama-3.2-1B) for their ability
to detect when a student has stated the solution, especially in edge cases
where the correct answer appears as an intermediate step.

Tracks accuracy rate and latency for both models.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Setup path for imports
try:
    from utils import check_if_solution_found
except ImportError:
    # Fallback for when script is run directly
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    packages_dir = script_dir.parent.parent
    if str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    from eval.utils import check_if_solution_found

try:
    import tinker
    from tinker.types import SamplingParams
except ImportError:
    print("❌ Error: Tinker SDK not found.")
    print("\nPlease install the Tinker Python SDK")
    sys.exit(1)

try:
    from tinker_cookbook import renderers, tokenizer_utils
    from tinker_cookbook.model_info import get_recommended_renderer_name
except ImportError:
    print("❌ Error: tinker_cookbook not found.")
    print("\nPlease install tinker-cookbook")
    sys.exit(1)


@dataclass
class TestCase:
    """Test case for solution detection."""
    question: str
    expected_answer: str
    student_responses: List[str]  # Multiple steps/responses
    should_detect: bool  # Whether solution should be detected
    description: str  # Description of the test case


@dataclass
class TestResult:
    """Result of a single test case."""
    test_case: TestCase
    model_name: str
    detected: bool
    llm_response: str
    latency_ms: float
    correct: bool


def create_test_cases() -> List[TestCase]:
    """Create test cases including edge cases."""
    test_cases = [
        # Edge case: answer appears as intermediate step
        TestCase(
            question="If I have $6 and I double it twice then lose half the remaining money, how much do I have?",
            expected_answer="12",
            student_responses=[
                "If you double it once you get 12, then again you get 24. Now if you lose half you divide by 2 and end up with 12 as the final answer."
            ],
            should_detect=True,
            description="Answer appears as intermediate step, then as final answer"
        ),
        
        # Edge case: answer appears only as intermediate step (should NOT detect)
        TestCase(
            question="If I have $6 and I double it twice then lose half the remaining money, how much do I have?",
            expected_answer="12",
            student_responses=[
                "If you double it once you get 12, then again you get 24",
                "Now if you lose half you divide by 2 and end up with 10."
            ],
            should_detect=False,
            description="Answer appears only as intermediate step, not final"
        ),
        
        # Simple positive case
        TestCase(
            question="If I triple $13, how much do I have?",
            expected_answer="39",
            student_responses=[
                "13 times 3 is 39. So the answer is 39."
            ],
            should_detect=True,
            description="Simple positive case with explicit answer"
        ),
        
        # Single digit answer - should match
        TestCase(
            question="What is 1 + 1?",
            expected_answer="2",
            student_responses=[
                "The final answer is 2."
            ],
            should_detect=True,
            description="Single digit answer"
        ),
        
        # Single digit - should NOT match when part of larger number
        TestCase(
            question="If I have $6 then double it and finally spend $10, how much money do I have?",
            expected_answer="2",
            student_responses=[
                "I got 12 as the answer."
            ],
            should_detect=False,
            description="Single digit answer appears in larger number (false positive check)"
        ),
        
        # Calculation step with answer - should NOT match
        TestCase(
            question="What is ((6 + 6) * 2)/2?",
            expected_answer="12",
            student_responses=[
                "I know 6 + 6 is 12... but what do these brackets mean?"
            ],
            should_detect=False,
            description="Answer appears only in calculation step, not as final answer"
        ),
        
        # Answer with currency
        TestCase(
            question="If I triple $13, how much do I have?",
            expected_answer="$39",
            student_responses=[
                "He would have $39 after tripling it? Is that the final answer?"
            ],
            should_detect=False,
            description="Student is uncertain of the answer"
        ),
        
        # Medium confidence pattern
        TestCase(
            question="What is 13 * 3?",
            expected_answer="39",
            student_responses=[
                "I think it's 39 but I'm not sure... Is that right?"
            ],
            should_detect=False,
            description="Uncertainty with question mark"
        ),
        
        # Answer at end with multiple mentions
        TestCase(
            question="What is 13 * 3?",
            expected_answer="39",
            student_responses=[
                "I calculated 39, then checked again and got 39. So the answer is 39."
            ],
            should_detect=True,
            description="Answer mentioned multiple times, final mention is explicit"
        ),
        
        # Negative case - partial match
        TestCase(
            question="What is 13 * 3?",
            expected_answer="39",
            student_responses=[
                "I got 239 as my result."
            ],
            should_detect=False,
            description="Partial number match (false positive check)"
        ),

        # --- CORRECTION & AMBIGUITY ---
        TestCase(
            question="What is 15 minus 7?",
            expected_answer="8",
            student_responses=["First I thought it was 10, but then I realized 15 - 7 is actually 8."],
            should_detect=True,
            description="Student corrects themselves to the right answer"
        ),
        TestCase(
            question="What is 15 minus 7?",
            expected_answer="8",
            student_responses=["The answer is 8... wait, no, I think it's 9."],
            should_detect=False,
            description="Student corrects themselves AWAY from the right answer"
        ),
        TestCase(
            question="Calculate 5 * 4.",
            expected_answer="20",
            student_responses=["Is the answer 20?"],
            should_detect=False,
            description="Student asks a question rather than stating a solution"
        ),

        # --- FORMATTING & UNITS ---
        TestCase(
            question="How many apples if you have 5 and buy 7 more?",
            expected_answer="12",
            student_responses=["I have exactly twelve apples now."],
            should_detect=True,
            description="Answer written as a word (Twelve vs 12)"
        ),
        TestCase(
            question="What is 5 + 5?",
            expected_answer="10",
            student_responses=["Final Answer: \n\n\n 10"],
            should_detect=True,
            description="Answer hidden behind heavy whitespace/newlines"
        ),
        TestCase(
            question="What is 1/2 as a decimal?",
            expected_answer="0.5",
            student_responses=["It is .5"],
            should_detect=True,
            description="Decimal leading zero missing"
        ),
        TestCase(
            question="If the area is 12.0, what is the value?",
            expected_answer="12",
            student_responses=["The result is 12.00"],
            should_detect=True,
            description="Trailing zeros in decimal"
        ),

        # --- THE "PREMATURE STEP" TRAPS ---
        TestCase(
            question="I have $10. I double it, then add $5. How much?",
            expected_answer="25",
            student_responses=["If you double 10 you get 20."],
            should_detect=False,
            description="Student stops after step 1 of a 2-step problem"
        ),
        TestCase(
            question="Take 100, divide by 2, then divide by 2 again.",
            expected_answer="25",
            student_responses=["100 divided by 2 is 50. Now I need to divide 50 by 2."],
            should_detect=False,
            description="Student explains the next step but hasn't performed it"
        ),
        TestCase(
            question="Double 6, then add 0.",
            expected_answer="12",
            student_responses=["6 doubled is 12."],
            should_detect=True,
            description="Mathematically finished even if they didn't mention the 'add 0' step"
        ),

        # --- DISTRACTOR NUMBERS ---
        TestCase(
            question="What is 10 + 10?",
            expected_answer="20",
            student_responses=["I have 20 dollars in my pocket, but the answer to the math is 18."],
            should_detect=False,
            description="Right number mentioned as a distractor, wrong number as final"
        ),
        TestCase(
            question="What is 10 + 10?",
            expected_answer="20",
            student_responses=["I thought the answer was 20, but now I'm not sure."],
            should_detect=False,
            description="Right number mentioned but then doubted/retracted"
        ),
        TestCase(
            question="Find x: 2x = 24",
            expected_answer="12",
            student_responses=["If we divide both sides by 2, we get x = 12"],
            should_detect=True,
            description="Algebraic statement of the final answer"
        ),

        # --- CONVERSATIONAL FILLER ---
        TestCase(
            question="What is 6 + 6?",
            expected_answer="12",
            student_responses=["I'm quite confident that the final result you are looking for is 12."],
            should_detect=True,
            description="Heavy conversational filler surrounding the answer"
        ),
        TestCase(
            question="What is 6 + 6?",
            expected_answer="12",
            student_responses=["My teacher told me the answer is 12, but I think it's 13."],
            should_detect=False,
            description="Attributing correct answer to someone else, then disagreeing"
        ),

        # --- MULTI-TURN LOGIC ---
        TestCase(
            question="I have $6. Double it twice, then lose half.",
            expected_answer="12",
            student_responses=[
                "6 doubled is 12.",
                "12 doubled is 24.",
                "24 minus half is 12. Done!"
            ],
            should_detect=True,
            description="Final answer provided after multiple turns of logic"
        ),
        TestCase(
            question="What is 5 * 4?",
            expected_answer="20",
            student_responses=["It's definitely not 19, and it's not 21. It's 20."],
            should_detect=True,
            description="Process of elimination ending in the correct answer"
        ),

        # --- NEGATION & SKEPTICISM ---
        TestCase(
            question="What is 10 + 2?",
            expected_answer="12",
            student_responses=["I don't think it is 12."],
            should_detect=False,
            description="Explicit negation of the correct answer"
        ),
        TestCase(
            question="What is 10 + 2?",
            expected_answer="12",
            student_responses=["Is it 12? No, that can't be right."],
            should_detect=False,
            description="Self-doubt/Retraction"
        ),
        TestCase(
            question="If I have 12 apples and give away 0, how many do I have?",
            expected_answer="12",
            student_responses=["I still have 12."],
            should_detect=True,
            description="Correct final answer using 'still have' phrasing"
        )
    ]
    return test_cases


def initialize_model(
    service_client: tinker.ServiceClient,
    model_name: str,
) -> tuple[tinker.SamplingClient, any, any]:
    """
    Initialize model with tokenizer, renderer, and sampling client.
    
    Args:
        service_client: Tinker ServiceClient instance
        model_name: Model name for solution detection
        
    Returns:
        Tuple of (sampling_client, tokenizer, renderer)
    """
    print(f"📚 Initializing model: {model_name}")
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    
    # Get tokenizer
    tokenizer = sampling_client.get_tokenizer()
    
    # Get recommended renderer name
    renderer_name = get_recommended_renderer_name(model_name)
    print(f"   Renderer: {renderer_name}")
    
    # Get renderer
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    
    print(f"   ✅ Model initialized\n")
    return sampling_client, tokenizer, renderer


def build_detection_prompt(question: str, expected_answer: str, student_responses: List[str]) -> str:
    """Build prompt for solution detection."""
    responses_text = "\n".join([f"Step {i+1}: {resp}" for i, resp in enumerate(student_responses)])
    
    prompt = f"""You are a math solution classifier. Your task is to determine if a student has concluded the problem with the correct final answer.

### Evaluation Rules:
1. **The Assertion Rule:** The student must state the answer as a definitive conclusion. 
2. **The Socratic 'No-Trigger':** Phrasings of uncertainty, questions (e.g., "Is it 12?"), or guesses (e.g., "Maybe it's 12?") must be marked as NO. 
3. **The 'Work Your Way' Rule:** If the student provides the answer but admits they don't know why, or asks for validation, the result is NO.
5. **Target Match:** Does the student provide the number {expected_answer} as their conclusion?
6. **The Intermediate Rule:** If {expected_answer} is mentioned only as a step (e.g., "Step 1 is {expected_answer}"), but the problem is not finished, the result is NO.
7. **Finality:** The student must present the answer as a result or a conclusion, not just a passing calculation.
8. **Context Matters:** Compare the student's progress against the complexity of the Question.

### Examples:

# Case 1: Full Success (Correct steps and final conclusion)
Question: I have $6. Double it twice, then lose half.
Expected Answer: 12
Student: "6 doubled is 12, then doubled again is 24. Half of 24 is 12. So I have 12."
Result: YES

# Case 2: Simple Success (Direct answer)
Question: What is 5 plus 7?
Expected Answer: 12
Student: "The answer is 12."
Result: YES

# Case 3: Premature Step (Matches answer, but isn't finished)
Question: I have $6. Double it twice, then lose half.
Expected Answer: 12
Student: "If you double 6 you get 12."
Result: NO

# Case 4: Wrong Conclusion (Correct steps, but fails at the end)
Question: Add 5 to 10, then subtract 2.
Expected Answer: 13
Student: "10 plus 5 is 15. 15 minus 2 is 14."
Result: NO

# Case 5: Calculation Distractor (Answer appears in a string, not as a conclusion)
Question: What is 10 plus 10?
Expected Answer: 20
Student: "If I add 20 + 5 I get 25, so the answer to 10+10 is probably 18."
Result: NO

### Current Task:
Question: {question}
Expected Answer: {expected_answer}
Student Responses: {responses_text}

Does the student's response conclude the problem with the correct final answer ({expected_answer})?
Respond ONLY with YES or NO.

Result:"""
    
    return prompt


def remove_thinking_tags(response: str) -> str:
    """
    Remove all thinking/reasoning tags from response (case-insensitive).
    
    Handles various tag formats like:
    - <think>...</think>
    - <THINK>...</THINK>
    - <thinking>...</thinking>
    - <think>...</think>
    - etc.
    
    This ensures thinking content doesn't affect solution detection evaluation.
    
    Args:
        response: Raw model output that may contain thinking tags
        
    Returns:
        Response text with all thinking tags removed
    """
    # Case-insensitive pattern to match any thinking tag format
    # Matches opening tags like <think>, <THINK>, <think>, etc.
    # and their corresponding closing tags
    # Using non-greedy matching and DOTALL to handle multiline content
    
    # Pattern matches:
    # - Opening tag: < followed by any chars containing think/reasoning/redacted, then >
    # - Content: any characters (non-greedy)
    # - Closing tag: </ followed by any chars containing think/reasoning/redacted, then >
    thinking_pattern = re.compile(
        r'<[^>]*(?:think|reasoning|redacted)[^>]*>.*?</[^>]*(?:think|reasoning|redacted)[^>]*>',
        re.IGNORECASE | re.DOTALL
    )
    
    # Remove all thinking blocks
    cleaned = thinking_pattern.sub('', response).strip()
    
    # Also handle any remaining unmatched opening tags (incomplete blocks)
    # This catches cases where only opening tag exists
    incomplete_pattern = re.compile(
        r'<[^>]*(?:think|reasoning|redacted)[^>]*>',
        re.IGNORECASE
    )
    cleaned = incomplete_pattern.sub('', cleaned).strip()
    
    return cleaned


def detect_solution_llm(
    client: tinker.SamplingClient,
    renderer: any,
    tokenizer: any,
    question: str,
    expected_answer: str,
    student_responses: List[str],
) -> tuple[bool, str, float]:
    """
    Use LLM to detect if solution has been stated.
    
    Returns:
        Tuple of (detected, llm_response, latency_ms)
    """
    prompt = build_detection_prompt(question, expected_answer, student_responses)
    
    # Format as a simple user message
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = renderer.build_generation_prompt(messages)
    
    # Use low temperature for deterministic responses
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=1024,  # Keep it short - just YES or NO
        stop=renderer.get_stop_sequences(),  # Stop early to keep response short
    )
    
    # Generate response and measure latency
    start_time = time.time()
    future = client.sample(prompt=formatted_prompt, sampling_params=sampling_params, num_samples=1)
    result = future.result()
    
    # Extract the generated text
    raw_output = ""
    for sequence in result.sequences:
        raw_output = tokenizer.decode(sequence.tokens)
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Remove thinking tags before evaluation (case-insensitive)
    raw_output = remove_thinking_tags(raw_output)
    
    # Clean up response - look for YES or NO
    raw_output = raw_output.strip().upper()
    if "YES" in raw_output:
        detected = True
    elif "NO" in raw_output:
        detected = False
    else:
        # Fallback: if we can't parse, default to False
        detected = False
        print(f"   ⚠️  Warning: Could not parse LLM response: {raw_output[:50]}")
    
    return detected, raw_output, latency_ms


def run_tests(
    service_client: tinker.ServiceClient,
    model_name: str,
    test_cases: List[TestCase],
) -> List[TestResult]:
    """Run all test cases for a given model."""
    print(f"\n{'='*80}")
    print(f"Testing model: {model_name}")
    print(f"{'='*80}\n")
    
    # Initialize model
    client, tokenizer, renderer = initialize_model(service_client, model_name)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test_case.description}")
        print(f"  Question: {test_case.question}")
        print(f"  Student's Answer: {test_case.student_responses}")
        print(f"  Expected Answer: {test_case.expected_answer}")
        print(f"  Should Detect: {test_case.should_detect}")
        
        # Run LLM detection
        detected, llm_response, latency_ms = detect_solution_llm(
            client=client,
            renderer=renderer,
            tokenizer=tokenizer,
            question=test_case.question,
            expected_answer=test_case.expected_answer,
            student_responses=test_case.student_responses,
        )
        
        correct = (detected == test_case.should_detect)
        
        result = TestResult(
            test_case=test_case,
            model_name=model_name,
            detected=detected,
            llm_response=llm_response,
            latency_ms=latency_ms,
            correct=correct,
        )
        results.append(result)
        
        status = "✅ CORRECT" if correct else "❌ WRONG"
        print(f"  LLM Response Start\n")
        print(llm_response)
        print(f"\n  LLM Response End")
        print(f"  Detected: {detected}")
        print(f"  Latency: {latency_ms:.1f}ms")
        print(f"  {status}\n")
    
    return results


def print_summary(results: List[TestResult], model_name: str):
    """Print summary statistics for a model."""
    correct_count = sum(1 for r in results if r.correct)
    accuracy = correct_count / len(results) * 100 if results else 0
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*80}")
    print(f"Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    print(f"Average Latency: {avg_latency:.1f}ms")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test LLM-based solution detection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: outputs/llm_solution_detection_test.json)",
    )
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("❌ Error: Tinker API key required.")
        print("   Set TINKER_API_KEY environment variable or add to .env file")
        sys.exit(1)
    
    # Set default output path
    if args.output is None:
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / "llm_solution_detection_test.json"
    
    # Create test cases
    test_cases = create_test_cases()
    print(f"📋 Created {len(test_cases)} test cases\n")
    
    try:
        # Create service client
        print("🔌 Connecting to Tinker...")
        service_client = tinker.ServiceClient(api_key=api_key)
        print("✅ Connected successfully\n")
        
        # Test both models
        # models = ["Qwen/Qwen3-8B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-1B"]
        models = ["meta-llama/Llama-3.1-8B-Instruct"]

        all_results = []
        
        for model_name in models:
            results = run_tests(service_client, model_name, test_cases)
            all_results.extend(results)
            print_summary(results, model_name)
        
        # Save results
        output_data = {
            "test_cases": [
                {
                    "question": tc.question,
                    "expected_answer": tc.expected_answer,
                    "student_responses": tc.student_responses,
                    "should_detect": tc.should_detect,
                    "description": tc.description,
                }
                for tc in test_cases
            ],
            "results": [
                {
                    "model_name": r.model_name,
                    "test_description": r.test_case.description,
                    "detected": r.detected,
                    "should_detect": r.test_case.should_detect,
                    "correct": r.correct,
                    "llm_response": r.llm_response,
                    "latency_ms": r.latency_ms,
                }
                for r in all_results
            ],
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Test complete! Results saved to {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check your API key is correct")
        print("  2. Check Tinker service status")
        print("  3. Verify model names are correct")
        sys.exit(1)


if __name__ == "__main__":
    main()
