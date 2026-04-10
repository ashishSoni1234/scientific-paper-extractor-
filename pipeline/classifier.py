"""
classifier.py — Scientific paper type classification.

Classifies papers into one of the following categories:
    - experimental
    - review
    - survey
    - theoretical
    - unknown

Uses a two-tier approach:
    1. Primary: Zero-shot classification via Groq LLM (when available)
    2. Fallback: Keyword-based rules (always available, no API needed)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = """You are a scientific paper classifier.
Classify this paper into EXACTLY ONE of these categories:
- experimental (lab experiments, synthesis, fabrication, measurements)
- review (literature review or systematic review)
- survey (broad survey of existing work)
- theoretical (mathematical models, simulations, DFT, computational)

Return ONLY a JSON object with no other text:
{{"type": "category_name", "confidence": 0.95, "reasoning": "brief reason"}}

Paper text (first 1000 chars):
{text}
"""

# Keyword sets for rule-based classification
REVIEW_KEYWORDS = [
    "review", "literature review", "systematic review", "we reviewed",
    "in this review", "this review covers", "overview of", "survey of",
]

SURVEY_KEYWORDS = [
    "survey", "comprehensive survey", "we surveyed", "this survey",
    "broad overview", "state of the art", "state-of-the-art",
]

EXPERIMENTAL_KEYWORDS = [
    "we synthesized", "we fabricated", "we prepared", "we measured",
    "we characterized", "experiment", "experimental", "sample preparation",
    "we observed", "the specimen", "tensile test", "XRD analysis",
    "SEM images", "TEM images", "we performed", "our experiments",
]

THEORETICAL_KEYWORDS = [
    "DFT", "density functional theory", "molecular dynamics",
    "Monte Carlo simulation", "first principles", "ab initio",
    "theoretical model", "numerical simulation", "finite element",
    "we simulated", "computational study", "simulation results",
    "mathematical model", "theoretical framework",
]


def classify_paper(text: str, groq_client=None, api_key: Optional[str] = None) -> dict:
    """
    Classify the paper type using LLM or keyword rules.

    Tries the Groq LLM classifier first if an api_key is supplied.
    Falls back to keyword-based classification if LLM fails or is unavailable.

    Args:
        text: Paper text to classify (full or partial).
        groq_client: Optional pre-initialised Groq client instance.
        api_key: Groq API key string (used if groq_client is not provided).

    Returns:
        dict with keys:
            - type (str): Classification label.
            - confidence (float): Confidence score 0-1.
            - method (str): "llm" or "keyword_rules".
    """
    if api_key or groq_client:
        try:
            result = _classify_with_llm(text, groq_client, api_key)
            logger.info("LLM classification: %s (%.2f)", result["type"], result["confidence"])
            return result
        except Exception as exc:
            logger.warning("LLM classification failed, using keyword rules: %s", exc)

    result = _classify_with_keywords(text)
    logger.info("Keyword classification: %s (%.2f)", result["type"], result["confidence"])
    return result


def _classify_with_llm(text: str, groq_client=None, api_key: Optional[str] = None) -> dict:
    """
    Use the Groq LLM to zero-shot classify the paper type.

    Args:
        text: Paper text for classification.
        groq_client: Optional pre-built Groq client.
        api_key: Groq API key for building a new client.

    Returns:
        dict with type, confidence, and method="llm".

    Raises:
        Exception: If Groq call fails or response is unparseable.
    """
    import json

    if groq_client is None:
        from groq import Groq
        groq_client = Groq(api_key=api_key)

    max_retries = 3
    last_error = None
    prompt = CLASSIFICATION_PROMPT.format(text=text[:1000])

    for attempt in range(max_retries):
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                timeout=30,
            )

            raw_content = completion.choices[0].message.content.strip()

            # Strip markdown fences
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                raw_content = "\n".join(lines[1:-1])

            parsed = json.loads(raw_content)
            paper_type = parsed.get("type", "unknown").lower()
            confidence = float(parsed.get("confidence", 0.75))

            # Validate the type is one of the allowed values
            valid_types = {"experimental", "review", "survey", "theoretical", "unknown"}
            if paper_type not in valid_types:
                paper_type = "unknown"
                confidence = 0.5

            return {"type": paper_type, "confidence": confidence, "method": "llm"}

        except Exception as exc:
            last_error = exc
            logger.warning("LLM classification attempt %d failed: %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
            continue

    raise Exception(f"LLM classification failed after {max_retries} attempts: {last_error}")


def _classify_with_keywords(text: str) -> dict:
    """
    Classify paper type using keyword scoring heuristics.

    Counts keyword matches for each category and returns the highest-scoring
    one with an estimated confidence based on relative score strength.

    Args:
        text: Paper text to classify.

    Returns:
        dict with type, confidence (0.5-0.85), and method="keyword_rules".
    """
    text_lower = text.lower()

    scores = {
        "review": _count_keyword_matches(text_lower, REVIEW_KEYWORDS),
        "survey": _count_keyword_matches(text_lower, SURVEY_KEYWORDS),
        "experimental": _count_keyword_matches(text_lower, EXPERIMENTAL_KEYWORDS),
        "theoretical": _count_keyword_matches(text_lower, THEORETICAL_KEYWORDS),
    }

    total_matches = sum(scores.values())

    if total_matches == 0:
        return {"type": "unknown", "confidence": 0.3, "method": "keyword_rules"}

    best_type = max(scores, key=scores.__getitem__)
    best_score = scores[best_type]
    confidence = min(0.85, 0.5 + (best_score / max(total_matches, 1)) * 0.35)

    return {
        "type": best_type,
        "confidence": round(confidence, 2),
        "method": "keyword_rules",
    }


def _count_keyword_matches(text_lower: str, keywords: list) -> int:
    """Count how many keywords from the list appear in the text."""
    return sum(1 for keyword in keywords if keyword.lower() in text_lower)
