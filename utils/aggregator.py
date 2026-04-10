"""
aggregator.py — Merges all pipeline stage outputs into a single clean result.

Takes the outputs from parser, extractor/fallback_extractor, classifier,
and similarity ranker and produces the final structured result dict.
"""

import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


def aggregate_results(
    parsed: dict,
    extracted: dict,
    classified: dict,
    similar_ranked: List[dict],
    extraction_method: str = "llm",
    start_time: Optional[float] = None,
) -> dict:
    """
    Merge all pipeline outputs into the final structured knowledge dict.

    Args:
        parsed: Output from parser (title, abstract, body, source).
        extracted: Output from extractor or fallback_extractor.
        classified: Output from classifier (type, confidence, method).
        similar_ranked: Ranked list from embedder with similarity_score.
        extraction_method: "llm" or "fallback" — which extractor was used.
        start_time: Unix timestamp from time.time() at pipeline start.

    Returns:
        Final aggregated dict conforming to the output schema.
    """
    processing_time = round(time.time() - start_time, 2) if start_time else 0.0

    # Determine the best title: prefer extracted, fall back to parsed
    title = (
        extracted.get("title")
        or parsed.get("title")
        or "Unknown Title"
    )

    # Clean up similar papers to only include what we need
    similar_papers_clean = []
    for paper in similar_ranked[:5]:  # Top 5 only
        clean_entry = {
            "title": paper.get("title", ""),
            "similarity_score": paper.get("similarity_score", 0.0),
            "paper_id": paper.get("paperId", ""),
            "year": paper.get("year", 0),
        }
        similar_papers_clean.append(clean_entry)

    result = {
        "title": title.strip(),
        "paper_type": classified.get("type", "unknown"),
        "paper_type_confidence": classified.get("confidence", 0.0),
        "entities": {
            "materials": _deduplicate_list(extracted.get("materials", [])),
            "properties": _deduplicate_list(extracted.get("properties", [])),
            "methods": _deduplicate_list(extracted.get("methods", [])),
        },
        "key_findings": _deduplicate_list(extracted.get("key_findings", [])),
        "numerical_results": _deduplicate_list(extracted.get("numerical_results", [])),
        "applications": _deduplicate_list(extracted.get("applications", [])),
        "authors": extracted.get("authors", []),
        "publication_date": extracted.get("publication_date", ""),
        "limitations": _deduplicate_list(extracted.get("limitations", [])),
        "future_work": _deduplicate_list(extracted.get("future_work", [])),
        "datasets": _deduplicate_list(extracted.get("datasets", [])),
        "similar_papers": similar_papers_clean,
        "extraction_method": extraction_method,
        "processing_time_seconds": processing_time,
        "source": parsed.get("source", "unknown"),
    }

    logger.info(
        "Aggregated result: type=%s, entities=%d materials/%d properties/%d methods, "
        "%d findings, %d similar papers",
        result["paper_type"],
        len(result["entities"]["materials"]),
        len(result["entities"]["properties"]),
        len(result["entities"]["methods"]),
        len(result["key_findings"]),
        len(result["similar_papers"]),
    )

    return result


def _deduplicate_list(items: list) -> list:
    """Remove duplicates from a list while preserving insertion order."""
    seen = set()
    deduped = []
    for item in items:
        normalised = str(item).strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            deduped.append(item)
    return deduped
