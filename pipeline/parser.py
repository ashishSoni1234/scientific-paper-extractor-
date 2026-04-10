"""
parser.py — Input handling for PDF, ArXiv, and raw text.

Supports three input modes:
1. PDF upload via pypdf
2. ArXiv ID lookup via the public ArXiv REST API (no library needed)
3. Raw text paste with basic cleaning

All parsers return a consistent dict:
    {
        "title": str,
        "abstract": str,
        "body": str,
        "source": str
    }
"""

import re
import io
import logging
import requests
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def parse_pdf(uploaded_file) -> dict:
    """
    Parse a PDF file uploaded via Streamlit and extract clean text.

    Strips common PDF artifacts like headers, footers, page numbers,
    and reference sections. Attempts to detect the paper title from
    the first page.

    Args:
        uploaded_file: A Streamlit UploadedFile object (BytesIO-like).

    Returns:
        dict with keys: title, abstract, body, source.

    Raises:
        ValueError: If parsing completely fails and no text is extracted.
    """
    try:
        import pypdf
    except ImportError as e:
        raise ImportError(f"pypdf import failed ({e}). Please run: pip install pypdf")

    try:
        raw_bytes = uploaded_file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(raw_bytes))

        pages_text = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        if not pages_text:
            raise ValueError("No text could be extracted from the PDF.")

        full_text = "\n".join(pages_text)
        full_text = _clean_pdf_text(full_text)

        title = _extract_title_from_text(full_text)
        abstract = _extract_abstract_from_text(full_text)
        body = _remove_references_section(full_text)

        return {
            "title": title,
            "abstract": abstract,
            "body": body,
            "source": "pdf_upload",
        }

    except Exception as exc:
        logger.error("PDF parsing failed: %s", exc)
        raise ValueError(f"Failed to parse PDF: {exc}") from exc


def parse_arxiv(arxiv_id: str) -> dict:
    """
    Fetch a paper from ArXiv using the public Atom/REST API directly.

    Uses `requests` + `xml.etree.ElementTree` — no external arxiv library needed.
    Calls: http://export.arxiv.org/api/query?id_list={arxiv_id}

    Args:
        arxiv_id: ArXiv paper ID string, e.g. "2401.00001" or "2304.01852".

    Returns:
        dict with keys: title, abstract, body, source.

    Raises:
        ValueError: If the paper is not found or the HTTP request fails.
    """
    # Normalise the ID — strip URL prefix if user pasted a full URL
    clean_id = arxiv_id.strip()
    clean_id = re.sub(r"https?://arxiv\.org/(abs|pdf)/", "", clean_id)
    clean_id = clean_id.replace(".pdf", "").strip()

    api_url = "http://export.arxiv.org/api/query"
    params = {"id_list": clean_id, "max_results": 1}

    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError(f"ArXiv API timed out for ID: {clean_id}")
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"ArXiv API request failed: {exc}") from exc

    # Parse the Atom XML response
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse ArXiv API response: {exc}") from exc

    # ArXiv Atom XML namespace
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    }

    entries = root.findall("atom:entry", ns)
    if not entries:
        raise ValueError(f"No paper found for ArXiv ID: {clean_id}")

    entry = entries[0]

    # Extract fields safely
    title_el = entry.find("atom:title", ns)
    summary_el = entry.find("atom:summary", ns)
    published_el = entry.find("atom:published", ns)
    authors = entry.findall("atom:author", ns)

    title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
    abstract = (summary_el.text or "").strip() if summary_el is not None else ""
    published = (published_el.text or "")[:10] if published_el is not None else ""

    authors_str = ", ".join(
        (a.find("atom:name", ns).text or "").strip()
        for a in authors
        if a.find("atom:name", ns) is not None
    )

    categories = [
        tag.get("term", "")
        for tag in entry.findall("atom:category", ns)
        if tag.get("term")
    ]
    categories_str = ", ".join(categories)

    body = (
        f"Title: {title}\n\n"
        f"Authors: {authors_str}\n\n"
        f"Categories: {categories_str}\n\n"
        f"Abstract:\n{abstract}\n\n"
        f"Published: {published}\n"
    )

    return {
        "title": title,
        "abstract": abstract,
        "body": body.strip(),
        "source": f"arxiv:{clean_id}",
    }


def parse_raw_text(text: str) -> dict:
    """
    Clean and structure a raw pasted text string.

    Extracts title, abstract (if present), and body from unstructured text.
    Applies basic normalisation: whitespace collapsing, encoding fixes.

    Args:
        text: Raw string pasted by the user.

    Returns:
        dict with keys: title, abstract, body, source.
    """
    if not text or not text.strip():
        return {"title": "", "abstract": "", "body": "", "source": "raw_text"}

    cleaned = _basic_text_clean(text)
    title = _extract_title_from_text(cleaned)
    abstract = _extract_abstract_from_text(cleaned)
    body = _remove_references_section(cleaned)

    return {
        "title": title,
        "abstract": abstract,
        "body": body,
        "source": "raw_text",
    }


# ── Private helpers ────────────────────────────────────────────────────────────


def _clean_pdf_text(text: str) -> str:
    """Remove common PDF extraction noise from raw text."""
    # Remove form feeds
    text = text.replace("\x0c", "\n")
    # Collapse excessive whitespace lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that look like lone page numbers (digits only)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove lines shorter than 4 chars (often artefacts)
    lines = [line for line in text.splitlines() if len(line.strip()) > 3 or line.strip() == ""]
    text = "\n".join(lines)
    return _basic_text_clean(text)


def _basic_text_clean(text: str) -> str:
    """Apply encoding normalisation and whitespace collapsing."""
    # Fix common PDF encoding issues
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    # Normalise unicode dashes
    text = re.sub(r"[\u2013\u2014]", "-", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def _extract_title_from_text(text: str) -> str:
    """Heuristically extract the paper title from the first few lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    # Title is usually in the first 5 non-empty lines, often the longest
    candidates = lines[:5]
    # Pick the longest candidate as most likely to be the title
    title = max(candidates, key=len, default="")

    # If the title looks like boilerplate (all caps short), fall back to first line
    if len(title) < 15:
        title = lines[0]

    # Truncate overly long "titles" — real titles are rarely > 200 chars
    return title[:200].strip()


def _extract_abstract_from_text(text: str) -> str:
    """Extract the abstract section from the text if present."""
    abstract_pattern = re.compile(
        r"(?:abstract|summary)[:\s]*\n?(.*?)(?=\n\s*(?:introduction|keywords|1\.|background|i\.))",
        re.IGNORECASE | re.DOTALL,
    )
    match = abstract_pattern.search(text)
    if match:
        return match.group(1).strip()[:2000]

    # Fallback: return first paragraph that is longer than 100 chars
    paragraphs = re.split(r"\n{2,}", text)
    for paragraph in paragraphs[1:4]:  # skip title paragraph
        if len(paragraph.strip()) > 100:
            return paragraph.strip()[:2000]

    return ""


def _remove_references_section(text: str) -> str:
    """Remove the References / Bibliography section and everything after it."""
    # Match common reference section headers
    ref_pattern = re.compile(
        r"\n\s*(?:references|bibliography|works cited)\s*\n",
        re.IGNORECASE,
    )
    match = ref_pattern.search(text)
    if match:
        return text[: match.start()].strip()
    return text.strip()
