#!/usr/bin/env python3
"""
Utility functions for evaluation scripts.

Provides robust JSON parsing utilities for handling LLM responses
and reasoning block parsing for thinking models.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Any
import warnings

try:
    import json_repair
except ImportError:
    json_repair = None


@dataclass
class ParseResult:
    """Result of parsing JSON from LLM response."""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    method: Optional[str] = None  # Which parsing method succeeded
    raw_text: Optional[str] = None  # Raw text used for parsing


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extract JSON from markdown code fences.
    
    Looks for patterns like:
    - ```json\n{...}\n```
    - ```\n{...}\n```
    - `{...}`
    
    Returns the extracted JSON string, or None if not found.
    """
    # Find markdown code fences first
    code_fence_pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(code_fence_pattern, text, re.DOTALL)
    if match:
        code_content = match.group(1).strip()
        # Check if it looks like JSON (starts with { or [)
        if code_content.startswith(('{', '[')):
            return code_content
    
    # Try inline code fences
    inline_pattern = r'`(\{.*?\})`'
    match = re.search(inline_pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    return None


def extract_json_with_balanced_braces(text: str) -> Optional[str]:
    """
    Extract JSON object by finding balanced braces.
    
    Finds the first `{` and matches it with the corresponding closing `}`.
    This handles nested JSON better than simple regex.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Count braces to find the matching closing brace
    brace_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start_idx:i+1]
    
    return None


def parse_llm_json_response(
    text: str,
    *,
    allow_sloppy: bool = True,
    expected_keys: Optional[list[str]] = None,
    warn_on_failure: bool = True,
) -> ParseResult:
    """
    Parse JSON from LLM response with multiple fallback strategies.
    
    Attempts multiple strategies in order:
    1. Direct json.loads() on the full text
    2. Extract from markdown code fences and parse
    3. Extract using balanced brace matching and parse
    4. Use json_repair to repair malformed JSON (if available and allow_sloppy=True)
    5. Try to find any JSON-like object with regex
    
    Args:
        text: The LLM response text to parse
        allow_sloppy: If True, use lenient parsing with json_repair
        expected_keys: Optional list of keys that should be present in the parsed JSON.
                      If provided and keys are missing, parsing is considered failed.
        warn_on_failure: If True, emit warnings when parsing fails
    
    Returns:
        ParseResult with success status, parsed data, error message, and method used
    """
    original_text = text
    
    # Strategy 1: Direct parse of the full text
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            if expected_keys is None or all(key in parsed for key in expected_keys):
                return ParseResult(
                    success=True,
                    data=parsed,
                    method="direct_parse",
                    raw_text=text.strip()
                )
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 2: Extract from markdown code fences
    markdown_json = extract_json_from_markdown(text)
    if markdown_json:
        try:
            parsed = json.loads(markdown_json)
            if isinstance(parsed, dict):
                if expected_keys is None or all(key in parsed for key in expected_keys):
                    return ParseResult(
                        success=True,
                        data=parsed,
                        method="markdown_extraction",
                        raw_text=markdown_json
                    )
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Strategy 3: Extract using balanced brace matching
    balanced_json = extract_json_with_balanced_braces(text)
    if balanced_json:
        try:
            parsed = json.loads(balanced_json)
            if isinstance(parsed, dict):
                if expected_keys is None or all(key in parsed for key in expected_keys):
                    return ParseResult(
                        success=True,
                        data=parsed,
                        method="balanced_brace_extraction",
                        raw_text=balanced_json
                    )
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Strategy 4: Try json_repair if available and allowed
    if allow_sloppy and json_repair is not None:
        # Try repairing the full text
        try:
            repaired = json_repair.repair_json(text)
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                if expected_keys is None or all(key in parsed for key in expected_keys):
                    return ParseResult(
                        success=True,
                        data=parsed,
                        method="json_repair_full",
                        raw_text=repaired
                    )
        except Exception:
            pass
        
        # Try repairing extracted JSON if we found any
        for extracted in [markdown_json, balanced_json]:
            if extracted:
                try:
                    repaired = json_repair.repair_json(extracted)
                    parsed = json.loads(repaired)
                    if isinstance(parsed, dict):
                        if expected_keys is None or all(key in parsed for key in expected_keys):
                            return ParseResult(
                                success=True,
                                data=parsed,
                                method="json_repair_extracted",
                                raw_text=repaired
                            )
                except Exception:
                    pass
    
    # Strategy 5: Last resort - try to find any JSON-like object with broader regex
    # This is similar to the original regex but handles nested structures better
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    # Try the largest match (likely the full JSON object)
    if matches:
        # Sort by length (descending) and try to parse
        matches.sort(key=len, reverse=True)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    if expected_keys is None or all(key in parsed for key in expected_keys):
                        return ParseResult(
                            success=True,
                            data=parsed,
                            method="regex_fallback",
                            raw_text=match
                        )
            except (json.JSONDecodeError, ValueError):
                continue
    
    # All strategies failed
    error_msg = "Failed to parse JSON from response"
    if expected_keys:
        error_msg += f" (missing expected keys: {expected_keys})"
    
    if warn_on_failure:
        warnings.warn(
            f"{error_msg}. Response text (first 200 chars): {text[:200]}",
            category=UserWarning
        )
    
    return ParseResult(
        success=False,
        error=error_msg,
        method=None,
        raw_text=text
    )


def parse_reasoning_blocks(response: str) -> tuple[str, Optional[str]]:
    """
    Parse <think></think> blocks from model output.
    
    Both Qwen and Kimi models return reasoning in <think></think> blocks.
    This function extracts the thinking content and returns the remaining content separately.
    
    This is useful for separating reasoning from the actual response content, ensuring
    that reasoning doesn't leak into conversation history or evaluation contexts.
    
    This implementation is robust to malformed outputs where:
    - Opening <think> tags may be missing or truncated
    - Multiple </think> tags may appear without matching opening tags
    - Content may appear before the first </think> tag
    
    Args:
        response: Raw model output that may contain reasoning blocks
        
    Returns:
        Tuple of (content, thinking) where:
        - content: The response text with reasoning blocks removed (safe to use in conversations)
        - thinking: The content from reasoning blocks (None if no blocks found)
    
    Example:
        >>> response = "<think>I need to solve this step by step</think>The answer is 42"
        >>> content, thinking = parse_reasoning_blocks(response)
        >>> content
        'The answer is 42'
        >>> thinking
        'I need to solve this step by step'
    """
    # Find all closing tags (case-insensitive) - this is more robust than requiring matching pairs
    closing_tag_pattern = re.compile(r'</think>', re.IGNORECASE)
    closing_tag_matches = list(closing_tag_pattern.finditer(response))
    
    # Find all opening tags (case-insensitive)
    opening_tag_pattern = re.compile(r'<think>', re.IGNORECASE)
    opening_tag_matches = list(opening_tag_pattern.finditer(response))
    
    # If no closing tags found, check for incomplete thinking tags
    if not closing_tag_matches:
        # If there's an opening tag but no closing tag, remove everything from the opening tag onwards
        if opening_tag_matches:
            # Find the first opening tag
            first_open_match = opening_tag_matches[0]
            first_open_pos = first_open_match.start()
            # Return content before the opening tag, and the thinking content (incomplete)
            content_before = response[:first_open_pos].strip()
            thinking_content = response[first_open_pos + len(first_open_match.group()):].strip()
            # Return empty content if everything was in the thinking tag, None for thinking since it's incomplete
            return content_before, None
        # No tags at all - return response as-is
        return response.strip(), None
    
    # Find the last closing tag position and its actual length
    last_closing_match = closing_tag_matches[-1]
    last_closing_pos = last_closing_match.start()
    last_closing_tag_length = len(last_closing_match.group())
    
    # Check if there's an incomplete thinking tag after the last closing tag
    # (i.e., an opening tag without a matching closing tag)
    after_last_closing = response[last_closing_pos + last_closing_tag_length:]
    incomplete_open_match = opening_tag_pattern.search(after_last_closing)
    
    if incomplete_open_match:
        # There's an incomplete tag - remove everything from that opening tag onwards
        incomplete_open_pos = last_closing_pos + last_closing_tag_length + incomplete_open_match.start()
        content = response[last_closing_pos + last_closing_tag_length:incomplete_open_pos].strip()
    else:
        # Everything after the last </think> (case-insensitive) is the final content
        content = after_last_closing.strip()
    
    # Now extract thinking blocks - try to find matching pairs first (preferred, case-insensitive)
    reasoning_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    reasoning_matches = reasoning_pattern.findall(response)
    
    if reasoning_matches:
        # We found properly matched pairs - use them
        thinking = "\n".join(reasoning_matches).strip()
        if thinking == "":
            thinking = None
    else:
        # No properly matched pairs found - check if there's content before first </think>
        first_closing_match = closing_tag_matches[0]
        first_closing_pos = first_closing_match.start()
        before_first_closing = response[:first_closing_pos].strip()
        
        # Check if there's a <think> tag before the first </think> (case-insensitive)
        # If so, extract everything between <think> and </think>
        think_open_pattern = re.compile(r'<think>', re.IGNORECASE)
        think_open_matches = list(think_open_pattern.finditer(before_first_closing))
        
        if think_open_matches:
            # Get the last opening tag before the closing tag
            last_open_match = think_open_matches[-1]
            think_open_pos = last_open_match.start()
            think_open_tag_length = len(last_open_match.group())
            # Extract from after <think> to before </think>
            thinking = before_first_closing[think_open_pos + think_open_tag_length:].strip()
            if thinking == "":
                thinking = None
        else:
            # No opening tag found, but there's a closing tag
            # This might be truncated output - treat everything before </think> as thinking
            # but only if it's not empty and looks like it could be thinking content
            if before_first_closing:
                thinking = before_first_closing.strip()
                if thinking == "":
                    thinking = None
            else:
                thinking = None
    
    return content, thinking
