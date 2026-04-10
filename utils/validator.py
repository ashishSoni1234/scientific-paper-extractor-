"""
validator.py — Input validation helpers for the Scientific Paper Knowledge Extractor.

Validates user-provided inputs before passing them into the pipeline,
providing friendly error messages for invalid inputs.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ArXiv IDs: old format (YYMM.NNNNN) and new format (YYMM.NNNNN)
ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ARXIV_URL_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)")

MIN_TEXT_LENGTH = 50  # Minimum characters for meaningful extraction
MAX_TEXT_LENGTH = 500_000  # ~500KB upper limit


def validate_arxiv_id(arxiv_id: str) -> tuple:
    """
    Validate and normalise an ArXiv ID or URL provided by the user.

    Accepts formats:
    - "2401.00001"
    - "2401.00001v2"
    - "https://arxiv.org/abs/2401.00001"
    - "https://arxiv.org/pdf/2401.00001.pdf"

    Args:
        arxiv_id: Raw string from user input.

    Returns:
        Tuple of (is_valid: bool, cleaned_id_or_error: str).
        On success, cleaned_id_or_error is the normalised ID.
        On failure, cleaned_id_or_error is the error message.
    """
    if not arxiv_id or not arxiv_id.strip():
        return False, "Please enter an ArXiv ID."

    cleaned = arxiv_id.strip()

    # Check if it's a URL and extract the ID
    url_match = ARXIV_URL_PATTERN.search(cleaned)
    if url_match:
        cleaned = url_match.group(1)

    # Remove .pdf suffix if present
    cleaned = cleaned.replace(".pdf", "").strip()

    if ARXIV_ID_PATTERN.match(cleaned):
        return True, cleaned

    return False, (
        f"'{arxiv_id}' does not look like a valid ArXiv ID. "
        "Expected format: 2401.00001 or 2401.00001v2"
    )


def validate_raw_text(text: str) -> tuple:
    """
    Validate raw text pasted by the user.

    Args:
        text: Raw text string from user.

    Returns:
        Tuple of (is_valid: bool, cleaned_text_or_error: str).
    """
    if not text or not text.strip():
        return False, "Please paste some text before processing."

    cleaned = text.strip()

    if len(cleaned) < MIN_TEXT_LENGTH:
        return False, (
            f"Text is too short ({len(cleaned)} characters). "
            f"Please provide at least {MIN_TEXT_LENGTH} characters."
        )

    if len(cleaned) > MAX_TEXT_LENGTH:
        logger.warning("Text too long (%d chars), truncating to %d.", len(cleaned), MAX_TEXT_LENGTH)
        cleaned = cleaned[:MAX_TEXT_LENGTH]

    return True, cleaned


def validate_api_key(api_key: str, key_name: str = "API key") -> tuple:
    """
    Validate that an API key string is present and non-empty.

    Args:
        api_key: The API key string to validate.
        key_name: Human-readable name of the key (for error messages).

    Returns:
        Tuple of (is_valid: bool, key_or_error: str).
    """
    if not api_key or not api_key.strip():
        return False, f"{key_name} is missing. Please set it in your .env file."
    return True, api_key.strip()
