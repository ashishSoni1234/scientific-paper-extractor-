"""
test_extractor.py — Unit tests for pipeline/extractor.py

Tests Groq LLM extraction with mocked API responses,
JSON parsing robustness, and schema validation.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


VALID_JSON_RESPONSE = json.dumps({
    "title": "Enhanced Thermal Conductivity in Graphene Nanocomposites",
    "materials": ["graphene", "PMMA", "epoxy resin"],
    "properties": ["thermal conductivity", "tensile strength"],
    "key_findings": [
        "Adding 5wt% graphene increased thermal conductivity by 45%",
        "Optimal processing temperature is 80°C"
    ],
    "methods": ["SEM", "Raman spectroscopy", "hot disk method"],
    "numerical_results": ["45% increase", "5 wt%", "80°C"],
    "applications": ["thermal management", "electronics packaging"]
})

MALFORMED_JSON_WITH_TEXT = """Here is the extracted information:
```json
{
  "title": "Test Paper",
  "materials": ["graphene"],
  "properties": ["conductivity"],
  "key_findings": ["We found improvement"],
  "methods": ["XRD"],
  "numerical_results": ["30%"],
  "applications": ["electronics"]
}
```
I hope this helps!"""

GARBAGE_RESPONSE = "Sorry, I cannot extract from this text. The content is unclear."


class TestExtractKnowledge:
    """Tests for the main extract_knowledge function."""

    @patch("pipeline.extractor.Groq")
    def test_returns_correct_schema_keys(self, mock_groq_class):
        """Extracted dict must have all required schema keys."""
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = VALID_JSON_RESPONSE
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_client

        from pipeline.extractor import extract_knowledge
        result = extract_knowledge("Sample paper text about graphene composites.", api_key="test-key")

        assert "title" in result
        assert "materials" in result
        assert "properties" in result
        assert "key_findings" in result
        assert "methods" in result
        assert "numerical_results" in result
        assert "applications" in result

    @patch("pipeline.extractor.Groq")
    def test_materials_is_list(self, mock_groq_class):
        """Materials field should be a list of strings."""
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = VALID_JSON_RESPONSE
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_client

        from pipeline.extractor import extract_knowledge
        result = extract_knowledge("Some text.", api_key="test-key")
        assert isinstance(result["materials"], list)

    @patch("pipeline.extractor.Groq")
    def test_handles_malformed_json_with_markdown_fences(self, mock_groq_class):
        """Should successfully parse JSON even when wrapped in markdown code fences."""
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = MALFORMED_JSON_WITH_TEXT
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_client

        from pipeline.extractor import extract_knowledge
        result = extract_knowledge("Sample text.", api_key="test-key")
        assert isinstance(result, dict)
        assert "materials" in result

    @patch("pipeline.extractor.Groq")
    def test_raises_extraction_error_on_garbage_response(self, mock_groq_class):
        """Should raise ExtractionError when LLM returns non-JSON text."""
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = GARBAGE_RESPONSE
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_client

        from pipeline.extractor import extract_knowledge, ExtractionError
        with pytest.raises(ExtractionError):
            extract_knowledge("Some text.", api_key="test-key")

    def test_raises_extraction_error_with_no_api_key(self):
        """Should raise ExtractionError when no API key is provided."""
        from pipeline.extractor import extract_knowledge, ExtractionError
        with pytest.raises(ExtractionError):
            extract_knowledge("Some text.", api_key="")

    @patch("pipeline.extractor.Groq")
    def test_api_failure_raises_extraction_error(self, mock_groq_class):
        """Should wrap API exceptions in ExtractionError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("Network error")
        mock_groq_class.return_value = mock_client

        from pipeline.extractor import extract_knowledge, ExtractionError
        with pytest.raises(ExtractionError):
            extract_knowledge("Some text.", api_key="test-key")


class TestParseLlmJson:
    """Tests for the _parse_llm_json helper function."""

    def test_parses_clean_json(self):
        """Clean JSON string should parse correctly."""
        from pipeline.extractor import _parse_llm_json
        result = _parse_llm_json(VALID_JSON_RESPONSE)
        assert result["title"] == "Enhanced Thermal Conductivity in Graphene Nanocomposites"
        assert isinstance(result["materials"], list)

    def test_parses_json_with_markdown_fences(self):
        """JSON wrapped in ```json ... ``` fences should still parse."""
        from pipeline.extractor import _parse_llm_json
        fenced = "```json\n" + VALID_JSON_RESPONSE + "\n```"
        result = _parse_llm_json(fenced)
        assert "title" in result

    def test_parses_json_embedded_in_text(self):
        """JSON embedded within text should be extracted via brace matching."""
        from pipeline.extractor import _parse_llm_json
        text = 'Here is the result: {"title": "Test", "materials": [], "properties": [], "key_findings": [], "methods": [], "numerical_results": [], "applications": []} Done!'
        result = _parse_llm_json(text)
        assert result["title"] == "Test"

    def test_raises_on_no_json(self):
        """Should raise ExtractionError when no JSON is found."""
        from pipeline.extractor import _parse_llm_json, ExtractionError
        with pytest.raises(ExtractionError):
            _parse_llm_json("This is just plain text with no JSON at all.")


class TestValidateAndNormaliseSchema:
    """Tests for the _validate_and_normalise_schema helper."""

    def test_fills_missing_keys_with_defaults(self):
        """Missing keys should be filled with empty string/list defaults."""
        from pipeline.extractor import _validate_and_normalise_schema
        result = _validate_and_normalise_schema({"title": "Test"})
        assert result["materials"] == []
        assert result["key_findings"] == []
        assert result["title"] == "Test"

    def test_converts_single_string_list_fields_to_list(self):
        """String values for list fields should be wrapped in a list."""
        from pipeline.extractor import _validate_and_normalise_schema
        result = _validate_and_normalise_schema({
            "title": "T",
            "materials": "graphene",  # Should become ["graphene"]
            "properties": [],
            "key_findings": [],
            "methods": [],
            "numerical_results": [],
            "applications": [],
        })
        assert isinstance(result["materials"], list)
        assert "graphene" in result["materials"]

    def test_all_list_fields_are_strings(self):
        """All items in list fields should be converted to strings."""
        from pipeline.extractor import _validate_and_normalise_schema
        result = _validate_and_normalise_schema({
            "title": "Test",
            "materials": [123, "graphene", None],
            "properties": [],
            "key_findings": [],
            "methods": [],
            "numerical_results": [],
            "applications": [],
        })
        # None items should be filtered out
        for item in result["materials"]:
            assert isinstance(item, str)
