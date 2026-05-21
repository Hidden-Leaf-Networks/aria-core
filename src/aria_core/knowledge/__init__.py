"""Knowledge system — tenant-scoped RAG with document ingestion and retrieval.

Provides:
- KnowledgeBase: tenant-scoped document store with vector search
- Document ingestion: text, markdown, JSON, HTML
- Chunking strategies: fixed-size, paragraph, semantic
- Embedding: pluggable providers (OpenAI, local)
- Retrieval: similarity search with metadata filtering

Usage:
    from aria_core.knowledge import KnowledgeBase, Document

    kb = KnowledgeBase(tenant_id=tid)
    await kb.ingest(Document(content="...", metadata={"source": "docs"}))
    results = await kb.search("How does the FSM work?", top_k=5)
"""

from aria_core.knowledge.models import Document, Chunk, SearchResult
from aria_core.knowledge.base import KnowledgeBase

__all__ = ["Document", "Chunk", "KnowledgeBase", "SearchResult"]
