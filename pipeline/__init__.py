"""
Pipeline package for the Scientific Paper Knowledge Extractor.

This package contains all the core processing modules:
- parser: PDF, ArXiv, and raw text input handling
- extractor: LLM-based knowledge extraction via Groq
- fallback_extractor: Rule-based regex extraction as fallback
- embedder: Sentence-transformer embedding generation
- classifier: Paper type classification
- similar_papers: Semantic Scholar API integration
"""
