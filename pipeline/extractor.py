"""
extractor.py — LLM-based scientific knowledge extraction via Groq.

Uses the Groq API (llama-3.1-8b-instruct) to extract structured entities
and findings from scientific paper text.

Schema returned:
    {
        "title": str,
        "materials": list[str],
        "properties": list[str],
        "key_findings": list[str],
        "methods": list[str],
        "numerical_results": list[str],
        "applications": list[str]
    }
"""

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT_TEMPLATE = """You are a scientific knowledge extraction system.
Extract from the following paper text and return ONLY valid JSON, no other text:
{{
  "title": "paper title if found",
  "authors": ["list of author names if mentioned"],
  "publication_date": "publication date or year if mentioned, else empty string",
  "materials": ["list of materials/compounds mentioned"],
  "properties": ["physical/chemical properties discussed"],
  "key_findings": ["2-4 most important findings as clear sentences"],
  "methods": ["experimental or computational methods used"],
  "numerical_results": ["important numbers with units e.g. '30% higher conductivity'"],
  "applications": ["potential applications mentioned"],
  "limitations": ["any limitations or constraints of the study mentioned"],
  "future_work": ["suggested future research directions mentioned"],
  "datasets": ["names of datasets or databases used"]
}}

Paper text:
{text}
"""

EMPTY_SCHEMA = {
    "title": "",
    "authors": [],
    "publication_date": "",
    "materials": [],
    "properties": [],
    "key_findings": [],
    "methods": [],
    "numerical_results": [],
    "applications": [],
    "limitations": [],
    "future_work": [],
    "datasets": [],
}


class ExtractionError(Exception):
    """Raised when the LLM extraction step fails irrecoverably."""


def extract_knowledge(text: str, api_key: str, timeout: int = 60, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Extract structured knowledge from scientific paper text using the Groq LLM.

    Sends a structured prompt to Groq and parses the JSON response.
    Limits input text to 3000 characters to stay within token limits.

    Args:
        text: Full or partial paper text to extract knowledge from.
        api_key: Groq API key string.
        timeout: Maximum seconds to wait for the Groq response.
        model: Groq model identifier to use.

    Returns:
        dict conforming to the extraction schema above.

    Raises:
        ExtractionError: If Groq returns an error or the response is unparseable.
    """
    try:
        from groq import Groq
    except ImportError:
        raise ExtractionError("groq library is required. Install with: pip install groq")

    if not api_key:
        raise ExtractionError("Groq API key is missing. Set GROQ_API_KEY in your .env file.")

    if len(text) > 15000:
        truncated_text = text[:10000] + "\n\n... [TEXT TRUNCATED] ...\n\n" + text[-5000:]
    else:
        truncated_text = text
        
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except Exception as exc:
        logger.error("Failed to initialize Groq client: %s", exc)
        raise ExtractionError(f"Groq Client Init Failed: {exc}")

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                timeout=timeout,
            )

            elapsed = time.time() - start_time
            logger.info("Groq extraction completed in %.2f seconds (Attempt %d)", elapsed, attempt + 1)

            raw_content = completion.choices[0].message.content
            if not raw_content:
                raise ExtractionError("Groq returned an empty response.")

            extracted = _parse_llm_json(raw_content)
            return extracted

        except Exception as exc:
            last_error = exc
            logger.warning("Groq attempt %d failed: %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait 2 seconds before retry
            continue

    logger.error("All %d Groq extraction attempts failed. Last error: %s", max_retries, last_error)
    raise ExtractionError(f"Groq API call failed after {max_retries} attempts: {last_error}")


def _parse_llm_json(raw_content: str) -> dict:
    """
    Robustly parse JSON from an LLM response string.

    Handles cases where the LLM adds markdown code fences or extra text
    before/after the JSON block.

    Args:
        raw_content: Raw string returned by the LLM.

    Returns:
        Parsed dict conforming to the extraction schema.

    Raises:
        ExtractionError: If no valid JSON can be extracted.
    """
    # Strip markdown code fences if present
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first and last fence lines
        cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    # Try direct parse first
    try:
        parsed = json.loads(cleaned)
        return _validate_and_normalise_schema(parsed)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    json_match = _extract_json_block(cleaned)
    if json_match:
        try:
            parsed = json.loads(json_match)
            return _validate_and_normalise_schema(parsed)
        except json.JSONDecodeError:
            pass

    raise ExtractionError(f"Could not parse valid JSON from LLM response: {raw_content[:200]}")


def _extract_json_block(text: str) -> Optional[str]:
    """Find the first JSON object block within a string using brace matching."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for index, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            return text[start: index + 1]
    return None


def _validate_and_normalise_schema(data: dict) -> dict:
    """
    Ensure the extracted dict conforms to the expected schema.

    Fills in missing keys with empty defaults and normalises all
    list fields to actually be lists of strings.

    Args:
        data: Raw parsed dict from LLM JSON.

    Returns:
        Normalised dict with all expected keys present.
    """
    result = dict(EMPTY_SCHEMA)

    for key in EMPTY_SCHEMA:
        if key in data:
            value = data[key]
            if isinstance(EMPTY_SCHEMA[key], str):
                result[key] = str(value) if value is not None else ""
            elif isinstance(value, list):
                result[key] = [str(item) for item in value if item]
            elif isinstance(value, str):
                # Single string — wrap in list for list fields
                result[key] = [value] if value else []
            else:
                result[key] = []

    return result
