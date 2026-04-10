"""
similar_papers.py — Semantic Scholar API integration for finding related papers.

Uses the public Semantic Scholar Graph API (no API key required) to fetch
similar papers based on ArXiv ID, title, or abstract.

Rate limit handling: 1-second sleep between API calls with full try/except
wrapping to prevent crashes.

Returns: List of dicts: [{"title": str, "abstract": str, "year": int, "paperId": str}]
"""

import logging
import time
from typing import List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
REQUEST_TIMEOUT = 15  # seconds
SLEEP_BETWEEN_CALLS = 1  # second — respect rate limits


def fetch_similar_papers(
    arxiv_id: Optional[str] = None,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    limit: int = 15,
) -> List[dict]:
    """
    Fetch candidate similar papers from the Semantic Scholar API.

    Strategy:
    1. If arxiv_id is provided, fetch the paper's own references.
    2. If not, use the search endpoint with the title as the query.
    3. On any failure, log a warning and return an empty list (never crash).

    Args:
        arxiv_id: ArXiv paper ID (e.g. "2401.00001"). Optional.
        title: Paper title used as search query fallback. Optional.
        abstract: Paper abstract (unused for API call; kept for embedder use). Optional.
        limit: Maximum number of candidate papers to return.

    Returns:
        List of dicts with keys: title, abstract, year, paperId.
        Returns empty list on failure.
    """
    try:
        if arxiv_id:
            return _fetch_by_arxiv_id(arxiv_id, limit)
        elif title:
            return _search_by_title(title, limit)
        else:
            logger.warning("fetch_similar_papers: no arxiv_id or title provided.")
            return []
    except Exception as exc:
        logger.warning("Semantic Scholar fetch failed entirely: %s", exc)
        return []


def _fetch_by_arxiv_id(arxiv_id: str, limit: int) -> List[dict]:
    """
    Fetch a paper's references from Semantic Scholar using its ArXiv ID.

    Args:
        arxiv_id: ArXiv paper ID string.
        limit: Max number of results to return.

    Returns:
        List of paper dicts, or empty list on failure.
    """
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/arXiv:{arxiv_id}"
    params = {
        "fields": "title,abstract,references.title,references.abstract,references.year,references.paperId"
    }

    try:
        time.sleep(SLEEP_BETWEEN_CALLS)
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        references = data.get("references", [])
        papers = []
        for ref in references[:limit]:
            if ref.get("title"):
                papers.append({
                    "title": ref.get("title", ""),
                    "abstract": ref.get("abstract", "") or "",
                    "year": ref.get("year") or 0,
                    "paperId": ref.get("paperId", ""),
                })

        if not papers:
            # The paper itself may not have references in the DB — fall back to search
            logger.info("No references found for ArXiv:%s, falling back to title search.", arxiv_id)
            title = data.get("title", "")
            if title:
                return _search_by_title(title, limit)

        return papers

    except requests.exceptions.HTTPError as http_err:
        logger.warning("Semantic Scholar HTTP error for ArXiv:%s — %s", arxiv_id, http_err)
        return []
    except Exception as exc:
        logger.warning("Failed to fetch ArXiv:%s from Semantic Scholar: %s", arxiv_id, exc)
        return []


def _search_by_title(title: str, limit: int) -> List[dict]:
    """
    Search Semantic Scholar for papers matching a title query.

    Args:
        title: Search query string (paper title or keywords).
        limit: Max number of results.

    Returns:
        List of paper dicts, or empty list on failure.
    """
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
    params = {
        "query": title[:200],  # API has query length limits
        "fields": "title,abstract,year,paperId",
        "limit": min(limit, 15),
    }

    try:
        time.sleep(SLEEP_BETWEEN_CALLS)
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        results = data.get("data", [])
        papers = []
        for item in results:
            if item.get("title"):
                papers.append({
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", "") or "",
                    "year": item.get("year") or 0,
                    "paperId": item.get("paperId", ""),
                })
        
        if not papers and len(title) > 10:
            # Try cleaning title and taking only the first few words as a fallback
            import re
            clean_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title).strip()
            clean_title = re.sub(r'\s+', ' ', clean_title)
            
            words = clean_title.split()
            if len(words) > 5:
                short_title = " ".join(words[:6])
            else:
                short_title = clean_title

            if short_title and short_title != title:
                logger.info("Retrying search with truncated/cleaned title: %s", short_title[:50])
                time.sleep(SLEEP_BETWEEN_CALLS)
                params["query"] = short_title[:200]
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if response.ok:
                    results = response.json().get("data", [])
                    for item in results:
                        if item.get("title"):
                            papers.append({
                                "title": item.get("title", ""),
                                "abstract": item.get("abstract", "") or "",
                                "year": item.get("year") or 0,
                                "paperId": item.get("paperId", ""),
                            })

        return papers

    except requests.exceptions.HTTPError as http_err:
        logger.warning("Semantic Scholar search HTTP error: %s", http_err)
        return []
    except Exception as exc:
        logger.warning("Semantic Scholar search failed: %s", exc)
        return []
