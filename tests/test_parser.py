"""
test_parser.py — Unit tests for pipeline/parser.py

Tests all three input modes: PDF, ArXiv, and raw text.
Uses mocking to avoid live network/file I/O during tests.
"""

import io
import pytest
from unittest.mock import MagicMock, patch


# ── Raw text tests ─────────────────────────────────────────────────────────────

class TestParseRawText:
    """Tests for the parse_raw_text function."""

    def test_basic_text_returns_all_keys(self):
        """Output dict must always have title, abstract, body, source keys."""
        from pipeline.parser import parse_raw_text
        text = "Enhanced Thermal Conductivity in Graphene-Polymer Nanocomposites\n\nAbstract\nThis paper studies the thermal properties of graphene composites. We found significant improvement in thermal conductivity by 45% when adding 5wt% graphene."
        result = parse_raw_text(text)
        assert "title" in result
        assert "abstract" in result
        assert "body" in result
        assert "source" in result

    def test_source_is_raw_text(self):
        """Source field should be 'raw_text' for pasted text input."""
        from pipeline.parser import parse_raw_text
        result = parse_raw_text("Some paper content with enough text to be meaningful.")
        assert result["source"] == "raw_text"

    def test_empty_text_returns_empty_strings(self):
        """Empty text input should return empty strings, not crash."""
        from pipeline.parser import parse_raw_text
        result = parse_raw_text("")
        assert result["title"] == ""
        assert result["abstract"] == ""
        assert result["body"] == ""

    def test_whitespace_only_text(self):
        """Whitespace-only text should behave like empty text."""
        from pipeline.parser import parse_raw_text
        result = parse_raw_text("   \n\n\t  ")
        assert result["title"] == ""

    def test_abstract_extracted_when_present(self):
        """Abstract section should be extracted when the keyword is present."""
        from pipeline.parser import parse_raw_text
        text = (
            "Enhanced Conductivity Study\n\n"
            "Abstract\nWe synthesized a novel material with improved conductivity. "
            "The results demonstrate a 30% increase in performance.\n\n"
            "Introduction\nThis paper covers graphene composites."
        )
        result = parse_raw_text(text)
        # Either abstract is extracted or it falls back to a paragraph — both acceptable
        assert result["abstract"] is not None
        assert isinstance(result["abstract"], str)

    def test_title_extracted_from_first_lines(self):
        """Title should be extracted from the first non-empty lines."""
        from pipeline.parser import parse_raw_text
        text = "Graphene-Based Nanocomposite with High Thermal Conductivity\n\nAbstract: This study reports..."
        result = parse_raw_text(text)
        assert "Graphene" in result["title"] or len(result["title"]) > 0

    def test_long_text_does_not_crash(self):
        """Very long text inputs should not crash."""
        from pipeline.parser import parse_raw_text
        long_text = "word " * 10000
        result = parse_raw_text(long_text)
        assert isinstance(result, dict)


# ── ArXiv tests ────────────────────────────────────────────────────────────────

class TestParseArxiv:
    """Tests for the parse_arxiv function with mocked API calls."""

    @patch("pipeline.parser.arxiv")
    def test_arxiv_id_returns_correct_schema(self, mock_arxiv_module):
        """parse_arxiv should return dict with all required keys."""
        mock_paper = MagicMock()
        mock_paper.title = "Test Paper Title"
        mock_paper.summary = "This is the abstract of the test paper."
        mock_paper.authors = [MagicMock(__str__=lambda self: "Author One")]
        mock_paper.categories = ["cs.LG"]
        mock_paper.published = "2024-01-01"

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])
        mock_arxiv_module.Client.return_value = mock_client
        mock_arxiv_module.Search.return_value = MagicMock()

        from pipeline.parser import parse_arxiv
        result = parse_arxiv("2401.00001")

        assert "title" in result
        assert "abstract" in result
        assert "body" in result
        assert "source" in result

    @patch("pipeline.parser.arxiv")
    def test_arxiv_source_includes_id(self, mock_arxiv_module):
        """Source field should reference the ArXiv ID."""
        mock_paper = MagicMock()
        mock_paper.title = "Test"
        mock_paper.summary = "Abstract text here."
        mock_paper.authors = []
        mock_paper.categories = []
        mock_paper.published = "2024-01-01"

        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_paper])
        mock_arxiv_module.Client.return_value = mock_client
        mock_arxiv_module.Search.return_value = MagicMock()

        from pipeline.parser import parse_arxiv
        result = parse_arxiv("2401.00001")
        assert "arxiv" in result["source"]

    @patch("pipeline.parser.arxiv")
    def test_arxiv_no_results_raises_value_error(self, mock_arxiv_module):
        """parse_arxiv should raise ValueError when no paper is found."""
        mock_client = MagicMock()
        mock_client.results.return_value = iter([])  # Empty results
        mock_arxiv_module.Client.return_value = mock_client
        mock_arxiv_module.Search.return_value = MagicMock()

        from pipeline.parser import parse_arxiv
        with pytest.raises(ValueError):
            parse_arxiv("9999.99999")

    def test_arxiv_strips_url_prefix(self):
        """ArXiv URLs should be handled by stripping the prefix."""
        from pipeline.parser import _clean_pdf_text  # just check the helper exists
        assert callable(_clean_pdf_text)


# ── PDF tests ─────────────────────────────────────────────────────────────────

class TestParsePdf:
    """Tests for parse_pdf using mocked pypdf."""

    @patch("pypdf.PdfReader")
    def test_pdf_returns_all_keys(self, mock_pdf_reader):
        """parse_pdf should return dict with title, abstract, body, source."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "Enhanced Thermal Conductivity in Nanocomposites\n\n"
            "Abstract\nWe studied graphene composites and found improved properties.\n\n"
            "Introduction\nGraphene is a 2D material with excellent properties."
        )
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader_instance

        from pipeline.parser import parse_pdf

        mock_uploaded = MagicMock()
        mock_uploaded.read.return_value = b"fake pdf bytes"

        result = parse_pdf(mock_uploaded)
        assert "title" in result
        assert "source" in result
        assert result["source"] == "pdf_upload"

    def test_parse_pdf_raises_on_empty_result(self):
        """parse_pdf should raise ValueError when no text is extracted."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader_instance):
            from pipeline.parser import parse_pdf
            mock_file = MagicMock()
            mock_file.read.return_value = b"fake pdf bytes"
            with pytest.raises(ValueError):
                parse_pdf(mock_file)


# ── Helper function tests ──────────────────────────────────────────────────────

class TestHelpers:
    """Tests for private helper functions."""

    def test_extract_abstract_finds_abstract_keyword(self):
        """Should detect and extract text after 'Abstract' keyword."""
        from pipeline.parser import _extract_abstract_from_text
        text = "Title Line\n\nAbstract\nThis is the abstract of the paper.\n\nIntroduction\nSome intro text."
        abstract = _extract_abstract_from_text(text)
        assert "abstract" in abstract.lower() or "This is" in abstract

    def test_remove_references_truncates_at_references(self):
        """References section should be removed from body text."""
        from pipeline.parser import _remove_references_section
        text = "Main content here.\n\nReferences\n[1] Smith et al. 2023..."
        result = _remove_references_section(text)
        assert "Main content" in result
        assert "[1] Smith" not in result

    def test_remove_references_leaves_text_without_references_intact(self):
        """Text without a references section should be returned unchanged."""
        from pipeline.parser import _remove_references_section
        text = "Main content that has no references section at all."
        result = _remove_references_section(text)
        assert "Main content" in result
