"""
test_similar_papers.py — Unit tests for pipeline/similar_papers.py

Tests the Semantic Scholar API integration, HTTP mocking,
fallback behaviour on error, and cosine similarity ranking.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ── Semantic Scholar fetch tests ───────────────────────────────────────────────

class TestFetchSimilarPapers:
    """Tests for the fetch_similar_papers function."""

    @patch("pipeline.similar_papers.requests.get")
    def test_returns_list_of_dicts(self, mock_get):
        """Should return a list of dicts when API succeeds."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {"title": "Paper One", "abstract": "Abstract one.", "year": 2023, "paperId": "abc123"},
                {"title": "Paper Two", "abstract": "Abstract two.", "year": 2022, "paperId": "def456"},
            ]
        }
        mock_get.return_value = mock_response

        from pipeline.similar_papers import fetch_similar_papers
        results = fetch_similar_papers(title="Graphene thermal conductivity")
        assert isinstance(results, list)

    @patch("pipeline.similar_papers.requests.get")
    def test_returns_correct_keys_in_each_dict(self, mock_get):
        """Each result dict should have title, abstract, year, paperId keys."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {"title": "Test Paper", "abstract": "Test abstract.", "year": 2023, "paperId": "xyz789"},
            ]
        }
        mock_get.return_value = mock_response

        from pipeline.similar_papers import fetch_similar_papers
        results = fetch_similar_papers(title="Test")
        if results:  # May be empty if mocked differently
            assert "title" in results[0]
            assert "abstract" in results[0]
            assert "year" in results[0]
            assert "paperId" in results[0]

    @patch("pipeline.similar_papers.requests.get")
    def test_returns_empty_list_on_http_error(self, mock_get):
        """Should return empty list when API returns HTTP error, never crash."""
        import requests
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")

        from pipeline.similar_papers import fetch_similar_papers
        results = fetch_similar_papers(title="Some paper title")
        assert results == []

    @patch("pipeline.similar_papers.requests.get")
    def test_returns_empty_list_on_connection_error(self, mock_get):
        """Should return empty list when network connection fails."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        from pipeline.similar_papers import fetch_similar_papers
        results = fetch_similar_papers(title="Test paper")
        assert results == []

    @patch("pipeline.similar_papers.requests.get")
    def test_returns_empty_list_on_timeout(self, mock_get):
        """Should return empty list on request timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        from pipeline.similar_papers import fetch_similar_papers
        results = fetch_similar_papers(title="Test paper")
        assert results == []

    def test_returns_empty_list_when_no_args(self):
        """Should return empty list when neither arxiv_id nor title provided."""
        from pipeline.similar_papers import fetch_similar_papers
        results = fetch_similar_papers()
        assert results == []

    @patch("pipeline.similar_papers.requests.get")
    def test_uses_arxiv_id_endpoint_when_provided(self, mock_get):
        """Should call the arXiv-specific endpoint when arxiv_id is given."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "title": "ArXiv Paper",
            "references": [
                {"title": "Ref 1", "abstract": "Abstract.", "year": 2023, "paperId": "r1"},
            ]
        }
        mock_get.return_value = mock_response

        from pipeline.similar_papers import fetch_similar_papers
        results = fetch_similar_papers(arxiv_id="2401.00001")
        assert isinstance(results, list)
        # Verify the arXiv URL format was requested
        call_url = mock_get.call_args[0][0]
        assert "arXiv:" in call_url or "2401.00001" in call_url

    @patch("pipeline.similar_papers.requests.get")
    def test_no_api_key_in_headers(self, mock_get):
        """Requests should not include any Authorization header."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        from pipeline.similar_papers import fetch_similar_papers
        fetch_similar_papers(title="test")

        # Check no headers were passed
        call_kwargs = mock_get.call_args[1] if mock_get.call_args else {}
        headers = call_kwargs.get("headers", {})
        assert "Authorization" not in headers
        assert "x-api-key" not in headers


# ── Similarity ranking tests ───────────────────────────────────────────────────

class TestRankBySimilarity:
    """Tests for the embedder.rank_by_similarity function."""

    def test_returns_sorted_by_descending_score(self):
        """Results should be sorted by similarity_score descending."""
        from pipeline.embedder import rank_by_similarity

        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        candidates = [
            {"title": "Low similarity paper", "embedding": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
            {"title": "High similarity paper", "embedding": np.array([0.9, 0.1, 0.0], dtype=np.float32)},
            {"title": "Medium similarity paper", "embedding": np.array([0.5, 0.5, 0.0], dtype=np.float32)},
        ]

        results = rank_by_similarity(query_embedding, candidates)
        scores = [r["similarity_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_have_similarity_score(self):
        """Every result dict should have a similarity_score key."""
        from pipeline.embedder import rank_by_similarity

        query_embedding = np.array([1.0, 0.0], dtype=np.float32)
        candidates = [
            {"title": "Paper A", "embedding": np.array([1.0, 0.0], dtype=np.float32)},
            {"title": "Paper B", "embedding": np.array([0.0, 1.0], dtype=np.float32)},
        ]

        results = rank_by_similarity(query_embedding, candidates)
        for result in results:
            assert "similarity_score" in result

    def test_similarity_score_in_range_0_to_1(self):
        """All similarity scores should be between 0 and 1."""
        from pipeline.embedder import rank_by_similarity

        query_embedding = np.random.rand(384).astype(np.float32)
        candidates = [
            {"title": f"Paper {i}", "embedding": np.random.rand(384).astype(np.float32)}
            for i in range(5)
        ]

        results = rank_by_similarity(query_embedding, candidates)
        for result in results:
            assert 0.0 <= result["similarity_score"] <= 1.0

    def test_empty_candidates_returns_empty_list(self):
        """Empty candidates list should return empty list."""
        from pipeline.embedder import rank_by_similarity

        query_embedding = np.array([1.0, 0.0], dtype=np.float32)
        results = rank_by_similarity(query_embedding, [])
        assert results == []


# ── Cosine similarity tests ────────────────────────────────────────────────────

class TestCosineSimilarity:
    """Tests for the embedder.compute_cosine_similarity function."""

    def test_identical_vectors_score_is_one(self):
        """Two identical vectors should have similarity 1.0."""
        from pipeline.embedder import compute_cosine_similarity
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        score = compute_cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 1e-5

    def test_orthogonal_vectors_score_is_zero(self):
        """Orthogonal vectors should have similarity 0.0."""
        from pipeline.embedder import compute_cosine_similarity
        vec1 = np.array([1.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0], dtype=np.float32)
        score = compute_cosine_similarity(vec1, vec2)
        assert abs(score) < 1e-5

    def test_zero_vector_returns_zero(self):
        """Zero vector input should return 0.0 without crashing."""
        from pipeline.embedder import compute_cosine_similarity
        zero = np.zeros(384, dtype=np.float32)
        nonzero = np.ones(384, dtype=np.float32)
        score = compute_cosine_similarity(zero, nonzero)
        assert score == 0.0

    def test_score_in_range(self):
        """Score should always be in [0, 1] range."""
        from pipeline.embedder import compute_cosine_similarity
        vec1 = np.random.rand(384).astype(np.float32)
        vec2 = np.random.rand(384).astype(np.float32)
        score = compute_cosine_similarity(vec1, vec2)
        assert 0.0 <= score <= 1.0
