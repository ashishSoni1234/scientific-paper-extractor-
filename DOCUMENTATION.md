# Documentation: Scientific Paper Knowledge Extractor

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Input Processing Stage](#input-processing-stage)
3. [Language Model Extraction Stage](#language-model-extraction-stage)
4. [Semantic Similarity & Embeddings Stage](#semantic-similarity--embeddings-stage)
5. [Aggregator & Data Schema](#aggregator--data-schema)
6. [Supported Data Fields](#supported-data-fields)
7. [Environment Configuration](#environment-configuration)

---

## 1. System Architecture
The application runs on a sequential data pipeline managed primarily by `app.py` via an interactive Streamlit UI. The system ensures robust scientific data parsing using PyTorch bindings without GIL lock contention on Windows environments.

**Stages of Execution:**
- **Parser**: Extracts raw string text from uploaded PDFs, ArXiv IDs, or manual input.
- **LLM Extractor**: Ingests up to 15,000 characters to process head and tail paragraphs using Groq API.
- **Classifier**: Zero-shot categorization using either the LLM or a rule-based algorithm.
- **Similarity Generator**: Creates document embeddings using Sentence-Transformers and queries Semantic Scholar.
- **Aggregator**: Deduplicates elements and securely structures the final dictionary for JSON, CSV, or Markdown download.

---

## 2. Input Processing Stage
**Location**: `pipeline/parser.py`

This module intercepts three input options:
1. **ArXiv ID or URL**: Calls the public ArXiv API using `urllib` and parses the XML feed to harvest the `title`, `abstract`, and publication `source`.
2. **PDF Verification**: Uses `pypdf` to run through physical PDF pages and compile them into unstructured raw text.
3. **Pasted Text**: Validates string lengths strictly before advancing it.

---

## 3. Language Model Extraction Stage
**Location**: `pipeline/extractor.py` & `pipeline/fallback_extractor.py`

When Groq's (`llama-3.1-8b-instant`) LLM API Key is provided, the extractor slices the target content to capture the very core sections explicitly bypassing the 8k token limit:
- Slicing Method: `text[:10000] + "\n\n... [TEXT TRUNCATED] ...\n\n" + text[-5000:]`
- This ensures that fields often tucked around "Conclusion" or "Data Availability" bounds are perfectly captured.

If the LLM fails or is disabled via the "Demo Mode", the `fallback_extractor.py` kicks in, leveraging RegEx arrays against keyword catalogs to formulate fallback JSONs.

---

## 4. Semantic Similarity & Embeddings Stage
**Location**: `pipeline/embedder.py` & `pipeline/similar_papers.py`

- **Local Embedder**: Utilizing `sentence-transformers/all-MiniLM-L6-v2`, it distills the merged `title + abstract` block into a high-dimensional vector space (NumPy matrix operations natively bypassed in single-process mode on Streamlit).
- **Semantic Scholar API Search**: Attempts exact matches using the *LLM Clean Title*. If API fails via timeout or `0 Results` on strict match bounds, the system auto-truncates the extracted title up to its *first 6 relevant words* as an architectural retry net.

---

## 5. Aggregator & Data Schema
**Location**: `utils/aggregator.py`

Assembles individual threads and parses responses into the ultimate schema. Deduplication runs natively here using Python strict `set()` logic to ensure items matching case-insensitive configurations inside `limitations`, `numerical_results`, `applications`, etc. don't repeat unnecessarily.

---

## 6. Supported Data Fields
The final JSON emitted by this software supports identically tracking:
- `title` - Processed string
- `paper_type` - Classified paper type (experimental, theoretical, survey, review, unknown)
- `paper_type_confidence` - Decimal value between 0.0-1.0
- `authors` - List of string names directly parsed from text
- `publication_date` - Timeframe explicitly detected within the context block
- `entities`: Nested object encapsulating lists of `materials`, `properties`, and `methods`
- `key_findings` - Main logical statements
- `numerical_results` - Specifically harvested metrics mapping exact unit measurements
- `applications` - Real-world utilities
- `limitations` - Stated drawbacks or boundary constraints (generally appended from the paper Tail)
- `future_work` - Next step expansions (generally appended from the paper Tail)
- `datasets` - External data/model sources utilized indicating replicability scopes
- `similar_papers` - Graph API JSON subset dictating Title, Score, ArXiv Year and external DB Paper ID.
- `extraction_method` - "llm" or "fallback"
- `processing_time_seconds` - Total compute execution delta
- `source` - Initial execution point ("pdf_upload" | "arxiv_id" | "raw_text")

---

## 7. Environment Configuration
See `.env.example` to review dependencies prior to `pip install -r requirements.txt`. You must generate your API keys at `https://console.groq.com/keys` inside `.env` (`GROQ_API_KEY=YOUR_KEY`). No API keys or external authentication structures are required for Semantic Scholar API routes nor ArXiv feed connections.
