# 🔬 Scientific Paper Knowledge Extractor

> Inspired by [Mimir Systems](https://mimirsystems.ai) — extract structured knowledge from scientific literature at scale using LLMs, NLP, and semantic search.

A production-ready Python + Streamlit application that takes any scientific paper (PDF, ArXiv ID, or pasted text) and extracts structured entities, key findings, methods, and semantically similar papers — all exportable as clean JSON.

---

## ✨ Features

| Feature | Description |
|---|---|
| **3 Input Modes** | Upload PDF · Enter ArXiv ID · Paste text |
| **LLM Context Expansion** | Uses a "Head + Tail" chunking method (first 10K & last 5K chars) to successfully extract conclusion data bypassing token limits. |
| **Comprehensive Parsing** | Precisely grabs Authors, Publication Dates, Future Work, Limitations, Datasets, and Numerical Results. |
| **Fallback Pipeline** | Rule-based regex extraction when Groq fails — never crashes |
| **Paper Classification** | Experimental / Review / Survey / Theoretical |
| **Semantic Similarity** | Finds top-5 similar papers via Semantic Scholar + sentence-transformers |
| **Demo Mode** | Toggle in sidebar to simulate API failure and test fallback |
| **Rich Exports** | Download fully structured knowledge as JSON, CSV, or Markdown |

> 📚 Check out the full **[DOCUMENTATION.md](./DOCUMENTATION.md)** for a deep dive into the system architecture and pipeline stages!

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ashishSoni1234/scientific-paper-extractor-.git
cd scientific-paper-extractor-

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Edit .env and replace the placeholder with your Groq API key

# 5. Run the app
streamlit run app.py
```

App opens at **http://localhost:8501**

### Get a free Groq API key
1. Visit [console.groq.com/keys](https://console.groq.com/keys)
2. Create an account and generate a key
3. Add it to your `.env` file as `GROQ_API_KEY=gsk_...`

> **Semantic Scholar** — No API key needed. The public API supports 100 requests/5 min, more than enough for interactive use.

---

## 🏗️ Architecture

```
User Input (PDF / ArXiv ID / Pasted Text)
         │
         ▼
┌─────────────────┐
│   Parser        │  PyMuPDF · arxiv library · text cleaner
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│   LLM Extractor (Groq llama-3.1-8b)    │  ← tries first
│   ↓ [if Groq fails for ANY reason]     │
│   Rule-based Fallback Extractor        │  ← always works
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Classifier    │  zero-shot (Groq) → keyword rules fallback
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedder      │  sentence-transformers all-MiniLM-L6-v2 (local)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────────┐
│ Similar Papers  │────▶│ Semantic Scholar API      │
└────────┬────────┘     │ (no API key required)     │
         │              └──────────────────────────┘
         ▼
┌─────────────────┐
│   Aggregator    │  merges all outputs into final dict
└────────┬────────┘
         │
         ▼
    JSON Output  →  Streamlit UI  →  Download
```

---

## 📁 Project Structure

```
scientific-paper-extractor/
├── app.py                     # Main Streamlit UI
├── requirements.txt           # Pinned dependencies
├── .env.example               # API key template
├── .gitignore
├── README.md
│
├── pipeline/
│   ├── parser.py              # PDF + ArXiv + text input handling
│   ├── extractor.py           # Groq LLM knowledge extraction
│   ├── fallback_extractor.py  # Rule-based regex fallback
│   ├── embedder.py            # sentence-transformers embeddings
│   ├── classifier.py          # Paper type classification
│   └── similar_papers.py     # Semantic Scholar API integration
│
├── utils/
│   ├── aggregator.py          # Merge all pipeline outputs
│   ├── validator.py           # Input validation helpers
│   └── json_formatter.py     # JSON sanitisation & export
│
└── tests/
    ├── test_parser.py
    ├── test_extractor.py
    ├── test_fallback.py
    └── test_similar_papers.py
```

---

## 📊 Example Output JSON

```json
{
  "title": "Enhanced Thermal Conductivity in Graphene-Polymer Nanocomposites",
  "paper_type": "experimental",
  "paper_type_confidence": 0.92,
  "entities": {
    "materials": ["graphene", "PMMA", "epoxy resin", "graphene oxide"],
    "properties": ["thermal conductivity", "tensile strength", "viscosity"],
    "methods": ["SEM", "Raman spectroscopy", "hot disk method", "melt mixing"]
  },
  "key_findings": [
    "Adding 5wt% graphene increased thermal conductivity by 45%",
    "Optimal dispersion achieved at 80°C processing temperature",
    "Mechanical properties improved without sacrificing flexibility"
  ],
  "numerical_results": ["45%", "5 wt%", "80°C", "3.2 W/mK"],
  "applications": ["thermal management", "electronics packaging", "aerospace"],
  "similar_papers": [
    {
      "title": "Graphene nanoplatelet composites for thermal applications",
      "similarity_score": 0.91,
      "paper_id": "abc123",
      "year": 2023
    }
  ],
  "extraction_method": "llm",
  "processing_time_seconds": 4.2
}
```

---

## 🔄 Fallback System

The app **never crashes**. Every stage has a fallback:

| Stage | Primary | Fallback |
|---|---|---|
| **Parsing** | PyMuPDF / arxiv library | Raw text cleaning |
| **Extraction** | Groq LLM (10s timeout) | Rule-based regex pipeline |
| **Classification** | Groq zero-shot | Keyword scoring heuristics |
| **Embeddings** | Local (always works) | Zero vector |
| **Similar Papers** | Semantic Scholar API | Empty list with warning |

In the UI:
- 🟢 **Extracted via LLM** — Groq succeeded
- 🟡 **Extracted via fallback** — Rule-based pipeline ran

---

## 🧪 Running Tests

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_fallback.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=pipeline --cov=utils --cov-report=term-missing
```

---

## 🛠️ Tech Stack

| Component | Tool | Version |
|---|---|---|
| PDF parsing | PyMuPDF (fitz) | 1.24.0 |
| ArXiv fetching | arxiv | 2.1.0 |
| LLM extraction | Groq API | 0.9.0 |
| Embeddings | sentence-transformers | 3.0.1 |
| ML utilities | scikit-learn, numpy | 1.5.0, 1.26.4 |
| Similar papers | Semantic Scholar API | — (no key) |
| UI | Streamlit | 1.36.0 |
| Config | python-dotenv | 1.0.1 |
| Testing | pytest | 8.2.2 |

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
