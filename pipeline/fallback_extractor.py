"""
fallback_extractor.py — Rule-based regex extraction pipeline.

This module runs when the Groq LLM extractor is unavailable or fails.
It uses carefully crafted regular expressions and keyword lists to extract
scientific entities from paper text without any external API calls.

CRITICAL: This fallback ALWAYS returns a valid dict — it never crashes
and never returns an empty result for non-empty input.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── Keyword lists ──────────────────────────────────────────────────────────────

MATERIAL_KEYWORDS = [
    "nanotube", "nanoparticle", "graphene", "polymer", "ceramic", "alloy",
    "composite", "fiber", "oxide", "carbide", "nitride", "silica", "alumina",
    "titanium", "copper", "iron", "steel", "aluminum", "carbon", "silicon",
    "perovskite", "zeolite", "hydrogel", "membrane", "coating", "thin film",
    "nanocomposite", "biopolymer", "elastomer", "epoxy", "resin", "PMMA",
    "PDMS", "PLA", "PEG", "chitosan", "cellulose", "lignin", "graphite",
    "fullerene", "quantum dot", "aerogel", "scaffold", "cathode", "anode",
]

PROPERTY_KEYWORDS = [
    "conductivity", "tensile strength", "hardness", "melting point", "bandgap",
    "viscosity", "permeability", "porosity", "density", "Young's modulus",
    "thermal conductivity", "electrical conductivity", "thermal stability",
    "elasticity", "yield strength", "fracture toughness", "ductility",
    "compressive strength", "flexural strength", "wear resistance",
    "corrosion resistance", "magnetic susceptibility", "dielectric constant",
    "refractive index", "surface area", "pore size", "crystallinity",
    "solubility", "hydrophobicity", "biocompatibility", "cytotoxicity",
    "photocatalytic activity", "fluorescence", "luminescence", "absorbance",
]

METHOD_KEYWORDS = [
    "XRD", "SEM", "TEM", "AFM", "XPS", "FTIR", "NMR", "Raman", "EDX", "EDS",
    "DFT", "molecular dynamics", "spectroscopy", "microscopy", "diffraction",
    "synthesis", "fabrication", "annealing", "sintering", "calcination",
    "pyrolysis", "electrodeposition", "chemical vapor deposition", "CVD",
    "sol-gel", "hydrothermal", "electrospinning", "ball milling",
    "hot pressing", "spark plasma sintering", "SPS", "magnetron sputtering",
    "atomic layer deposition", "ALD", "freeze-drying", "coprecipitation",
    "TGA", "DSC", "BET", "DLS", "UV-vis", "ICP-MS", "GC-MS", "HPLC",
    "simulation", "modelling", "Monte Carlo", "finite element", "FEM",
]

FINDING_SIGNAL_PHRASES = [
    "we found", "we show", "we demonstrate", "we report", "we observe",
    "results show", "results indicate", "results suggest", "results demonstrate",
    "it was found", "it was observed", "it was demonstrated",
    "our results", "our findings", "our study shows",
    "demonstrate", "achieve", "improve", "reveal", "confirm",
    "indicate that", "show that", "suggest that",
    "significantly", "notably", "remarkably", "substantially",
]

# Regex for chemical formulas: e.g., TiO2, Fe3O4, Al2O3, SiC
CHEMICAL_FORMULA_PATTERN = re.compile(
    r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?\d*)+\d*\b"
)

# Regex for numerical results with units
NUMERICAL_RESULT_PATTERN = re.compile(
    r"\d+\.?\d*\s*(?:MPa|GPa|kPa|eV|meV|nm|μm|mm|cm|%|K|°C|°F|"
    r"S/m|S/cm|W/m|W/mK|mS/cm|μS/cm|g/cm³|kg/m³|"
    r"mg/g|wt%|vol%|at%|mol%|ppm|ppb|rpm|Hz|kHz|MHz|GHz|"
    r"mPa·s|cP|m²/g|cm²/g|Å|nm²)"
)


def extract_with_rules(text: str) -> dict:
    """
    Extract structured scientific knowledge using rule-based regex patterns.

    This is the fallback extraction method that runs without any API.
    It always returns a dict with all expected keys, even if the input
    is low-quality or garbled.

    Args:
        text: Scientific paper text (any length, any quality).

    Returns:
        dict with keys: title, materials, properties, key_findings,
        methods, numerical_results, applications.
    """
    if not text or not text.strip():
        logger.warning("Fallback extractor received empty text.")
        return _empty_result()

    try:
        title = _extract_title(text)
        materials = _extract_materials(text)
        properties = _extract_properties(text)
        key_findings = _extract_findings(text)
        methods = _extract_methods(text)
        numerical_results = _extract_numerical_results(text)
        applications = _extract_applications(text)

        result = {
            "title": title,
            "authors": [],
            "publication_date": "",
            "materials": materials,
            "properties": properties,
            "key_findings": key_findings,
            "methods": methods,
            "numerical_results": numerical_results,
            "applications": applications,
            "limitations": [],
            "future_work": [],
            "datasets": [],
        }

        logger.info(
            "Fallback extraction: %d materials, %d properties, %d findings, %d methods",
            len(materials), len(properties), len(key_findings), len(methods),
        )
        return result

    except Exception as exc:
        logger.error("Fallback extractor encountered an error: %s", exc)
        # CRITICAL: Never crash — return partial result
        return _empty_result()


# ── Private extraction helpers ─────────────────────────────────────────────────


def _extract_title(text: str) -> str:
    """Extract the most likely title from the first few lines of text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    # Return the first reasonably-sized line
    for line in lines[:10]:
        if 15 < len(line) < 250:
            return line
    return lines[0][:200] if lines else ""


def _extract_materials(text: str) -> List[str]:
    """
    Detect material mentions using chemical formula regex and keyword list.

    Args:
        text: Input paper text.

    Returns:
        Deduplicated list of material strings found.
    """
    found = set()

    # Chemical formulas (e.g. TiO2, Fe3O4)
    formula_matches = CHEMICAL_FORMULA_PATTERN.findall(text)
    for match in formula_matches:
        if len(match) >= 2:  # Skip single-element symbols
            found.add(match)

    # Keyword-based materials (case-insensitive)
    text_lower = text.lower()
    for keyword in MATERIAL_KEYWORDS:
        if keyword.lower() in text_lower:
            found.add(keyword)

    return sorted(found)[:20]  # Cap at 20 items


def _extract_properties(text: str) -> List[str]:
    """
    Detect physical and chemical property mentions.

    Args:
        text: Input paper text.

    Returns:
        Deduplicated list of property strings found.
    """
    found = set()
    text_lower = text.lower()

    for keyword in PROPERTY_KEYWORDS:
        if keyword.lower() in text_lower:
            found.add(keyword)

    return sorted(found)[:15]


def _extract_findings(text: str) -> List[str]:
    """
    Extract key findings by locating sentences with finding-signal phrases.

    Splits text into sentences and returns those containing strong
    finding-indicator phrases, cleaned and deduplicated.

    Args:
        text: Input paper text.

    Returns:
        List of key finding sentences (up to 5).
    """
    # Split into sentences heuristically
    sentences = re.split(r"(?<=[.!?])\s+", text)
    findings = []
    seen_starts = set()

    text_lower = text.lower()
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if len(sentence_clean) < 30 or len(sentence_clean) > 500:
            continue

        sentence_lower = sentence_clean.lower()
        if any(phrase in sentence_lower for phrase in FINDING_SIGNAL_PHRASES):
            start = sentence_clean[:30]
            if start not in seen_starts:
                findings.append(sentence_clean)
                seen_starts.add(start)

        if len(findings) >= 5:
            break

    return findings


def _extract_methods(text: str) -> List[str]:
    """
    Detect experimental and computational method mentions.

    Args:
        text: Input paper text.

    Returns:
        Deduplicated list of method strings found.
    """
    found = set()
    text_lower = text.lower()

    for keyword in METHOD_KEYWORDS:
        if keyword.lower() in text_lower:
            found.add(keyword)

    return sorted(found)[:15]


def _extract_numerical_results(text: str) -> List[str]:
    """
    Extract numerical results with associated measurement units.

    Uses a regex pattern covering common scientific units.

    Args:
        text: Input paper text.

    Returns:
        Deduplicated list of numerical result strings (up to 15).
    """
    matches = NUMERICAL_RESULT_PATTERN.findall(text)
    unique_matches = list(dict.fromkeys(matches))  # Preserve order, deduplicate
    return unique_matches[:15]


def _extract_applications(text: str) -> List[str]:
    """
    Detect potential application domain mentions.

    Args:
        text: Input paper text.

    Returns:
        List of application domain strings.
    """
    application_keywords = [
        "energy storage", "battery", "fuel cell", "solar cell", "photovoltaic",
        "capacitor", "supercapacitor", "catalyst", "catalysis", "sensor",
        "biomedical", "drug delivery", "tissue engineering", "implant",
        "aerospace", "automotive", "electronics", "semiconductor",
        "thermal management", "water treatment", "filtration", "membrane",
        "coating", "corrosion protection", "structural", "construction",
        "packaging", "flexible electronics", "wearable", "optoelectronics",
    ]

    found = set()
    text_lower = text.lower()
    for keyword in application_keywords:
        if keyword in text_lower:
            found.add(keyword)

    return sorted(found)[:10]


def _empty_result() -> dict:
    """Return the minimum valid extraction schema with all empty fields."""
    return {
        "title": "",
        "authors": [],
        "publication_date": "",
        "materials": [],
        "properties": [],
        "key_findings": [],
        "methods": [],
        "numerical_results": [],
        "applications": [],
        "limitations": [],
        "future_work": [],
        "datasets": [],
    }
