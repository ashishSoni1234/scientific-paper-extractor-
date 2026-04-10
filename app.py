"""
app.py — Main Streamlit UI for the Scientific Paper Knowledge Extractor.

A production-ready web application that extracts structured knowledge from
scientific papers using LLMs and NLP. Inspired by Mimir Systems' approach
to scaling scientific knowledge extraction.

Usage:
    streamlit run app.py
"""

import json
import logging
import os
import time
from typing import Optional

# Prevent OpenMP PyTorch/NumPy severe crash on Windows at embedding generation
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page Configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Scientific Paper Knowledge Extractor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #0a0e1a 100%);
    }

    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 12px;
    }

    /* Paper type badges */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-experimental { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-review       { background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-survey       { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-theoretical  { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .badge-unknown      { background: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }

    /* Entity tags */
    .entity-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 13px;
        margin: 3px;
    }
    .entity-tag-material {
        background: rgba(34, 197, 94, 0.1);
        color: #86efac;
        border-color: rgba(34, 197, 94, 0.25);
    }
    .entity-tag-property {
        background: rgba(251, 191, 36, 0.1);
        color: #fde68a;
        border-color: rgba(251, 191, 36, 0.25);
    }
    .entity-tag-method {
        background: rgba(251, 113, 133, 0.1);
        color: #fda4af;
        border-color: rgba(251, 113, 133, 0.25);
    }

    /* Extraction method badges */
    .method-llm      { background: rgba(34, 197, 94, 0.1);   color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.25);  border-radius: 8px; padding: 4px 12px; font-size: 12px; font-weight: 600; }
    .method-fallback { background: rgba(251, 191, 36, 0.1);  color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.25); border-radius: 8px; padding: 4px 12px; font-size: 12px; font-weight: 600; }

    /* Finding items */
    .finding-item {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 14px;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Similar paper rows */
    .similar-paper-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .similar-paper-title { font-size: 14px; font-weight: 500; color: #e2e8f0; margin-bottom: 4px; }
    .similar-paper-meta  { font-size: 12px; color: #64748b; }

    /* Status indicators */
    .status-ok  { color: #22c55e; }
    .status-err { color: #f87171; }

    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #a5b4fc;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(groq_api_key: str) -> bool:
    """
    Render the sidebar with API status indicators and demo mode toggle.

    Args:
        groq_api_key: The Groq API key from environment.

    Returns:
        demo_mode (bool): Whether demo mode (force fallback) is active.
    """
    with st.sidebar:
        st.markdown("### 🔬 Knowledge Extractor")
        st.markdown("*Mimir-inspired scientific NLP*")
        st.divider()

        st.markdown("**⚡ API Status**")
        groq_ok = bool(groq_api_key)
        groq_icon = "✅" if groq_ok else "❌"
        st.markdown(f"{groq_icon} &nbsp; **Groq LLM** {'Connected' if groq_ok else 'No API Key'}", unsafe_allow_html=True)
        st.markdown("✅ &nbsp; **Semantic Scholar** &nbsp; Public (no key needed)", unsafe_allow_html=True)
        st.markdown("✅ &nbsp; **Embeddings** &nbsp; Local (all-MiniLM-L6-v2)", unsafe_allow_html=True)

        st.divider()

        demo_mode = st.toggle(
            "🧪 Demo Mode",
            value=False,
            help="Simulates Groq API failure to test the rule-based fallback extractor.",
        )
        if demo_mode:
            st.warning("⚠️ Demo Mode ON — using rule-based fallback only.")

        st.divider()
        st.markdown("**📖 How it works**")
        st.markdown("""
1. Paste text / upload PDF / enter ArXiv ID
2. LLM extracts entities & findings
3. Embeddings find similar papers
4. Download structured JSON
        """)

        st.divider()
        st.markdown(
            "<small style='color:#475569'>Built with Groq · Semantic Scholar · sentence-transformers</small>",
            unsafe_allow_html=True,
        )

    return demo_mode


# ── Progress helpers ───────────────────────────────────────────────────────────

PIPELINE_STAGES = [
    "📄 Parsing input",
    "🧠 Extracting knowledge",
    "📐 Generating embeddings",
    "🔍 Finding similar papers",
    "✅ Finalising result",
]


def run_pipeline(
    input_mode: str,
    uploaded_file=None,
    arxiv_id: str = "",
    raw_text: str = "",
    groq_api_key: str = "",
    demo_mode: bool = False,
) -> Optional[dict]:
    """
    Execute the full extraction pipeline with live progress updates.

    Implements the fallback chain:
        Parser → LLM Extractor (or Fallback) → Classifier → Embedder → Similar Papers → Aggregator

    Args:
        input_mode: "pdf", "arxiv", or "text"
        uploaded_file: Streamlit UploadedFile (for pdf mode)
        arxiv_id: ArXiv paper ID string (for arxiv mode)
        raw_text: Pasted text (for text mode)
        groq_api_key: Groq API key
        demo_mode: If True, skip LLM and use fallback directly

    Returns:
        dict: Final aggregated result, or None if parsing fails.
    """
    pipeline_start = time.time()

    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    def update_progress(stage_index: int, message: str):
        progress = int((stage_index / len(PIPELINE_STAGES)) * 100)
        progress_bar.progress(progress)
        status_placeholder.info(f"**{PIPELINE_STAGES[stage_index]}** — {message}")

    # ── Stage 1: Parse ─────────────────────────────────────────────────────────
    update_progress(0, "Reading your input...")

    from pipeline.parser import parse_pdf, parse_arxiv, parse_raw_text
    from utils.validator import validate_arxiv_id, validate_raw_text

    parsed = None
    parse_error = None

    try:
        if input_mode == "pdf":
            parsed = parse_pdf(uploaded_file)
        elif input_mode == "arxiv":
            is_valid, cleaned_id = validate_arxiv_id(arxiv_id)
            if not is_valid:
                st.error(f"❌ {cleaned_id}")
                progress_bar.empty()
                status_placeholder.empty()
                return None
            parsed = parse_arxiv(cleaned_id)
        elif input_mode == "text":
            is_valid, cleaned_text = validate_raw_text(raw_text)
            if not is_valid:
                st.error(f"❌ {cleaned_text}")
                progress_bar.empty()
                status_placeholder.empty()
                return None
            parsed = parse_raw_text(cleaned_text)
    except Exception as exc:
        st.error(f"❌ **Parsing failed:** {exc}")
        progress_bar.empty()
        status_placeholder.empty()
        return None

    # Combine abstract + body for processing
    full_text = f"{parsed.get('abstract', '')} {parsed.get('body', '')}".strip()
    if not full_text:
        st.error("❌ Could not extract any text from the input. Please try a different source.")
        progress_bar.empty()
        status_placeholder.empty()
        return None

    # ── Stage 2 to 4: Concurrent Execution ─────────────────────────────────────
    update_progress(1, "Processing with LLM and APIs concurrently...")

    from pipeline.extractor import extract_knowledge, ExtractionError
    from pipeline.fallback_extractor import extract_with_rules
    from pipeline.classifier import classify_paper
    from pipeline.embedder import generate_embedding, rank_by_similarity
    from pipeline.similar_papers import fetch_similar_papers
    import concurrent.futures

    extraction_method = "fallback"

    def fetch_extract():
        nonlocal extraction_method
        if not demo_mode and groq_api_key:
            try:
                ext = extract_knowledge(full_text, groq_api_key)
                extraction_method = "llm"
                return ext
            except ExtractionError as exc:
                logger.warning("LLM extraction failed, using fallback: %s", exc)
        return extract_with_rules(full_text)

    def fetch_classify():
        return classify_paper(
            full_text,
            api_key=groq_api_key if (not demo_mode and groq_api_key) else None,
        )

    def fetch_similar(ext_dict: dict):
        arxiv_id_for_search = None
        if input_mode == "arxiv":
            is_val, clean = validate_arxiv_id(arxiv_id)
            if is_val: arxiv_id_for_search = clean
            
        search_title = ext_dict.get("title") or parsed.get("title")
        return fetch_similar_papers(
            arxiv_id=arxiv_id_for_search,
            title=search_title,
            abstract=parsed.get("abstract"),
        )

    def do_embedding():
        query_text = f"{parsed.get('title', '')} {parsed.get('abstract', '')}"
        return generate_embedding(query_text)

    update_progress(1, "Extracting knowledge via LLM...")
    extracted = fetch_extract()
    
    update_progress(1, "Classifying paper type...")
    classified = fetch_classify()
    
    update_progress(2, "Generating semantic embeddings (Local CPU)...")
    query_embedding = do_embedding()
    
    update_progress(3, "Searching Semantic Scholar...")
    candidates = fetch_similar(extracted)

    if not extracted.get("title") and parsed.get("title"):
        extracted["title"] = parsed["title"]

    if not candidates:
        st.info("ℹ️ Semantic Scholar returned no results — similar papers section will be empty.")

    similar_ranked = rank_by_similarity(query_embedding, candidates)

    # ── Stage 5: Aggregate ─────────────────────────────────────────────────────
    update_progress(4, "Assembling final output...")

    from utils.aggregator import aggregate_results

    result = aggregate_results(
        parsed=parsed,
        extracted=extracted,
        classified=classified,
        similar_ranked=similar_ranked,
        extraction_method=extraction_method,
        start_time=pipeline_start,
    )

    progress_bar.progress(100)
    status_placeholder.success(f"✅ **Complete!** Processed in {result['processing_time_seconds']}s")

    return result


# ── Result Rendering ───────────────────────────────────────────────────────────

def render_results(result: dict):
    """
    Render the extraction result in the Streamlit UI.

    Displays paper type badge, entities, findings, numerical results,
    applications, similar papers, and a JSON download button.

    Args:
        result: Aggregated result dict from the pipeline.
    """
    from utils.json_formatter import format_output_json

    st.divider()

    # ── Header row ─────────────────────────────────────────────────────────────
    col_title, col_badges = st.columns([3, 1])
    with col_title:
        st.markdown(f"## 📄 {result.get('title', 'Untitled Paper')}")
        
        authors = result.get('authors', [])
        authors_text = ", ".join(authors) if authors else "Unknown Authors"
        pub_date = result.get('publication_date', '')
        date_text = f" • Published: {pub_date}" if pub_date else ""
        
        st.markdown(f"<p style='color:#94a3b8; font-size: 1.1rem;'>👥 {authors_text}{date_text}</p>", unsafe_allow_html=True)


    with col_badges:
        paper_type = result.get("paper_type", "unknown")
        confidence = result.get("paper_type_confidence", 0.0)
        badge_class = f"badge badge-{paper_type}"
        st.markdown(
            f"<br><span class='{badge_class}'>{paper_type.upper()}</span> "
            f"<small style='color:#64748b'>{confidence:.0%}</small>",
            unsafe_allow_html=True,
        )

        extraction_method = result.get("extraction_method", "fallback")
        if extraction_method == "llm":
            st.markdown("<span class='method-llm'>🤖 Extracted via LLM</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='method-fallback'>⚙️ Extracted via fallback (rule-based)</span>", unsafe_allow_html=True)

    # ── Entities section ───────────────────────────────────────────────────────
    st.markdown("---")
    entities = result.get("entities", {})
    materials = entities.get("materials", [])
    properties = entities.get("properties", [])
    methods = entities.get("methods", [])

    col_mat, col_prop, col_meth = st.columns(3)

    with col_mat:
        st.markdown("<div class='section-header'>🧪 Materials</div>", unsafe_allow_html=True)
        if materials:
            tags = " ".join(f"<span class='entity-tag entity-tag-material'>{m}</span>" for m in materials[:12])
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<small style='color:#475569'>None detected</small>", unsafe_allow_html=True)

    with col_prop:
        st.markdown("<div class='section-header'>📊 Properties</div>", unsafe_allow_html=True)
        if properties:
            tags = " ".join(f"<span class='entity-tag entity-tag-property'>{p}</span>" for p in properties[:10])
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<small style='color:#475569'>None detected</small>", unsafe_allow_html=True)

    with col_meth:
        st.markdown("<div class='section-header'>🔬 Methods</div>", unsafe_allow_html=True)
        if methods:
            tags = " ".join(f"<span class='entity-tag entity-tag-method'>{m}</span>" for m in methods[:10])
            st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<small style='color:#475569'>None detected</small>", unsafe_allow_html=True)

    # ── Key findings ───────────────────────────────────────────────────────────
    st.markdown("---")
    col_findings, col_numbers = st.columns([2, 1])

    with col_findings:
        st.markdown("<div class='section-header'>💡 Key Findings</div>", unsafe_allow_html=True)
        findings = result.get("key_findings", [])
        if findings:
            for finding in findings:
                st.markdown(f"<div class='finding-item'>• {finding}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<small style='color:#475569'>No findings extracted</small>", unsafe_allow_html=True)

    with col_numbers:
        st.markdown("<div class='section-header'>📏 Numerical Results</div>", unsafe_allow_html=True)
        numbers = result.get("numerical_results", [])
        if numbers:
            for num in numbers[:8]:
                st.markdown(f"<div class='finding-item' style='border-color:#f59e0b'>⬤ {num}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<small style='color:#475569'>None found</small>", unsafe_allow_html=True)

    # ── Applications & Additional Insights ─────────────────────────────────────
    applications = result.get("applications", [])
    limitations = result.get("limitations", [])
    future_work = result.get("future_work", [])
    datasets = result.get("datasets", [])

    if applications or datasets:
        st.markdown("---")
        c_app, c_data = st.columns(2)
        with c_app:
            st.markdown("<div class='section-header'>🚀 Potential Applications</div>", unsafe_allow_html=True)
            if applications:
                tags = " ".join(f"<span class='entity-tag'>{a}</span>" for a in applications)
                st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#475569'>None provided</small>", unsafe_allow_html=True)
        with c_data:
            st.markdown("<div class='section-header'>📂 Datasets Used</div>", unsafe_allow_html=True)
            if datasets:
                tags = " ".join(f"<span class='entity-tag entity-tag-method'>{d}</span>" for d in datasets)
                st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#475569'>None detected</small>", unsafe_allow_html=True)

    if limitations or future_work:
        st.markdown("---")
        c_lim, c_fut = st.columns(2)
        with c_lim:
            st.markdown("<div class='section-header'>⚠️ Limitations</div>", unsafe_allow_html=True)
            if limitations:
                for lim in limitations:
                    st.markdown(f"<div class='finding-item' style='border-color:#ef4444'>• {lim}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#475569'>None noted</small>", unsafe_allow_html=True)
        with c_fut:
            st.markdown("<div class='section-header'>🔮 Future Work</div>", unsafe_allow_html=True)
            if future_work:
                for fw in future_work:
                    st.markdown(f"<div class='finding-item' style='border-color:#10b981'>• {fw}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#475569'>None noted</small>", unsafe_allow_html=True)

    # ── Similar papers ─────────────────────────────────────────────────────────
    similar = result.get("similar_papers", [])
    if similar:
        st.markdown("---")
        st.markdown("<div class='section-header'>📚 Similar Papers</div>", unsafe_allow_html=True)
        for idx, paper in enumerate(similar[:5]):
            title = paper.get("title", "Untitled")
            score = paper.get("similarity_score", 0.0)
            year = paper.get("year", "")
            paper_id = paper.get("paper_id", "")

            col_info, col_score = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f"<div class='similar-paper-card'>"
                    f"<div class='similar-paper-title'>{idx + 1}. {title}</div>"
                    f"<div class='similar-paper-meta'>Year: {year or 'N/A'} &nbsp;·&nbsp; ID: {paper_id[:12] or 'N/A'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_score:
                st.metric("Similarity", f"{score:.0%}")
                st.progress(score)

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2, col_dl3, col_meta = st.columns([1, 1, 1, 2])

    with col_dl1:
        try:
            json_string = format_output_json(result)
            st.download_button(
                label="⬇️ JSON",
                data=json_string,
                file_name="extracted_knowledge.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Error: {exc}")
            
    with col_dl2:
        try:
            # Create a simple markdown representation
            md_lines = [f"# {result.get('title', 'Paper')}"]
            if result.get("authors"): md_lines.append(f"**Authors:** {', '.join(result['authors'])}")
            if result.get("publication_date"): md_lines.append(f"**Date:** {result['publication_date']}")
            md_lines.append("\n## Key Findings")
            for f in result.get("key_findings", []): md_lines.append(f"- {f}")
            md_string = "\n".join(md_lines)
            st.download_button(
                label="⬇️ Markdown",
                data=md_string,
                file_name="extracted_knowledge.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except Exception:
            pass

    with col_dl3:
        try:
            # Simple CSV conversion
            import csv
            import io
            csv_output = io.StringIO()
            writer = csv.writer(csv_output)
            writer.writerow(["Field", "Value"])
            for k, v in result.items():
                if isinstance(v, list): v = " | ".join(map(str, v))
                if k not in ["similar_papers"]:
                    writer.writerow([k, str(v)])
            st.download_button(
                label="⬇️ CSV",
                data=csv_output.getvalue(),
                file_name="extracted_knowledge.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception:
            pass

    with col_meta:
        st.markdown(
            f"<small style='color:#64748b'>⏱ Processing time: **{result.get('processing_time_seconds', 0)}s** &nbsp;·&nbsp; "
            f"Source: **{result.get('source', 'unknown')}** &nbsp;·&nbsp; "
            f"Extractor: **{result.get('extraction_method', 'unknown')}**</small>",
            unsafe_allow_html=True,
        )


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    # Try st.secrets first (Streamlit Cloud), fall back to os.getenv (local)
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    except:
        groq_api_key = os.getenv("GROQ_API_KEY", "")
    
    demo_mode = render_sidebar(groq_api_key)

    # ── Hero header ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; padding: 32px 0 16px 0;'>
            <div style='font-size: 48px; margin-bottom: 8px;'>🔬</div>
            <h1 style='font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;'>
                Scientific Paper Knowledge Extractor
            </h1>
            <p style='color: #64748b; margin-top: 8px; font-size: 1rem;'>
                Extract structured entities, findings & similar papers from any scientific paper
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input tabs ─────────────────────────────────────────────────────────────
    tab_pdf, tab_arxiv, tab_text = st.tabs(["📤 Upload PDF", "🔗 ArXiv ID", "📝 Paste Text"])

    result = None

    with tab_pdf:
        st.markdown("**Upload a scientific paper PDF**")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a scientific paper in PDF format.",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.info(f"📄 File loaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
            if st.button("🚀 Extract Knowledge", key="btn_pdf", use_container_width=True):
                st.session_state.result = run_pipeline(
                    input_mode="pdf",
                    uploaded_file=uploaded_file,
                    groq_api_key=groq_api_key,
                    demo_mode=demo_mode,
                )

    with tab_arxiv:
        st.markdown("**Enter an ArXiv paper ID or URL**")
        arxiv_input = st.text_input(
            "ArXiv ID",
            placeholder="e.g.  2604.08376  or  https://arxiv.org/abs/2604.08376",
            label_visibility="collapsed",
            key="arxiv_input",
        )
        st.markdown(
            "<small style='color:#64748b'>💡 Try: <code>2604.08376</code></small>",
            unsafe_allow_html=True,
        )
        if st.button("🚀 Extract Knowledge", key="btn_arxiv", use_container_width=True):
            if arxiv_input.strip():
                st.session_state.result = run_pipeline(
                    input_mode="arxiv",
                    arxiv_id=arxiv_input.strip(),
                    groq_api_key=groq_api_key,
                    demo_mode=demo_mode,
                )
            else:
                st.error("❌ Please enter an ArXiv ID.")

    with tab_text:
        st.markdown("**Paste your paper text below**")
        pasted_text = st.text_area(
            "Paper text",
            height=280,
            placeholder="Paste the abstract, introduction, or full text of a scientific paper here...",
            label_visibility="collapsed",
            key="text_input",
        )
        char_count = len(pasted_text)
        st.markdown(
            f"<small style='color:#475569'>{char_count:,} characters</small>",
            unsafe_allow_html=True,
        )
        if st.button("🚀 Extract Knowledge", key="btn_text", use_container_width=True):
            if pasted_text.strip():
                st.session_state.result = run_pipeline(
                    input_mode="text",
                    raw_text=pasted_text,
                    groq_api_key=groq_api_key,
                    demo_mode=demo_mode,
                )
            else:
                st.error("❌ Please paste some text before extracting.")

    # ── Show results ───────────────────────────────────────────────────────────
    if getattr(st.session_state, 'result', None) is not None:
        render_results(st.session_state.result)


if __name__ == "__main__":
    main()
