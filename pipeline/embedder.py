"""
embedder.py — Sentence-transformer embedding generation and similarity ranking.

Uses the all-MiniLM-L6-v2 model (384-dimensional embeddings) to convert
text into dense vector representations for semantic similarity calculations.

Model is cached in Streamlit session state to avoid reloading on every call.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level model cache (for non-Streamlit usage, e.g. tests)
_model_cache: Optional[object] = None

try:
    import streamlit as st
    @st.cache_resource(show_spinner=False)
    def _load_st_model():
        logger.info("Loading sentence-transformer model all-MiniLM-L6-v2 (st.cache_resource)...")
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-MiniLM-L3-V2")
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def _get_model():
    """
    Load and cache the sentence-transformer model.

    Attempts to use Streamlit `@st.cache_resource` when running
    inside a Streamlit app; falls back to a module-level cache otherwise.

    Returns:
        SentenceTransformer model instance.
    """
    global _model_cache

    if HAS_STREAMLIT:
        try:
            from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
            if get_script_run_ctx() is not None:
                return _load_st_model()
        except ImportError:
            pass

    # Fall back to module-level cache
    if _model_cache is None:
        logger.info("Loading sentence-transformer model all-MiniLM-L6-v2 (module cache)...")
        from sentence_transformers import SentenceTransformer
        _model_cache = SentenceTransformer("paraphrase-MiniLM-L3-V2")

    return _model_cache


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a 384-dimensional dense embedding for the input text.

    Uses the all-MiniLM-L6-v2 sentence transformer model. The model
    is cached to avoid repeated disk I/O on subsequent calls.

    Args:
        text: Input text to embed (any length; model handles truncation).

    Returns:
        numpy ndarray of shape (384,) with float32 values.
    """
    if not text or not text.strip():
        logger.warning("generate_embedding received empty text; returning zero vector.")
        return np.zeros(384, dtype=np.float32)
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(1)
    model = _get_model()
    # Truncate to ~512 tokens worth of characters to stay within model limits
    truncated = text[:2048]
    embedding = model.encode(truncated, normalize_embeddings=True)
    return embedding.astype(np.float32)


def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Because generate_embedding normalises embeddings to unit length,
    cosine similarity is equivalent to the dot product, making this
    both fast and numerically stable.

    Args:
        embedding1: First embedding vector (numpy ndarray).
        embedding2: Second embedding vector (numpy ndarray).

    Returns:
        Float in range [0.0, 1.0] representing cosine similarity.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    similarity = float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    # Clamp to [0, 1] to handle floating-point edge cases
    return max(0.0, min(1.0, similarity))


def rank_by_similarity(query_embedding: np.ndarray, candidates_list: List[dict]) -> List[dict]:
    """
    Rank a list of candidate paper dicts by semantic similarity to the query.

    Each candidate dict must contain an 'embedding' key with a numpy array
    or an 'abstract' key that will be embedded on the fly.

    Args:
        query_embedding: Embedding vector for the query paper.
        candidates_list: List of candidate dicts. Each must have at minimum
            a 'title' key, plus either 'embedding' or 'abstract'.

    Returns:
        List of candidate dicts sorted by descending similarity score,
        each augmented with a 'similarity_score' float key.
    """
    if not candidates_list:
        return []

    scored = []
    for candidate in candidates_list:
        if "embedding" in candidate and candidate["embedding"] is not None:
            candidate_embedding = candidate["embedding"]
        elif "abstract" in candidate and candidate["abstract"]:
            candidate_embedding = generate_embedding(candidate["abstract"])
        elif "title" in candidate and candidate["title"]:
            candidate_embedding = generate_embedding(candidate["title"])
        else:
            scored.append({**candidate, "similarity_score": 0.0})
            continue

        score = compute_cosine_similarity(query_embedding, candidate_embedding)
        scored.append({**candidate, "similarity_score": round(score, 4)})

    scored.sort(key=lambda item: item["similarity_score"], reverse=True)
    return scored
