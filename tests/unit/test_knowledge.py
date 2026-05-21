"""Tests for knowledge base / RAG system."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.knowledge.models import Document, Chunk, ChunkStrategy
from aria_core.knowledge.base import (
    KnowledgeBase,
    SimpleEmbedding,
    _chunk_text,
    _cosine_similarity,
)


class TestChunking:
    def test_paragraph_chunking(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = _chunk_text(text, strategy=ChunkStrategy.PARAGRAPH)
        assert len(chunks) >= 1

    def test_fixed_chunking(self) -> None:
        text = "A" * 1000
        chunks = _chunk_text(text, strategy=ChunkStrategy.FIXED, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        assert len(chunks[0]) == 200

    def test_sentence_chunking(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = _chunk_text(text, strategy=ChunkStrategy.SENTENCE, chunk_size=40)
        assert len(chunks) >= 2

    def test_empty_text(self) -> None:
        chunks = _chunk_text("", strategy=ChunkStrategy.PARAGRAPH)
        assert len(chunks) == 1


class TestSimpleEmbedding:
    async def test_embed_returns_vectors(self) -> None:
        emb = SimpleEmbedding(dimension=64)
        vectors = await emb.embed(["hello world", "foo bar"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 64

    async def test_vectors_are_normalized(self) -> None:
        import math
        emb = SimpleEmbedding(dimension=64)
        vectors = await emb.embed(["test"])
        magnitude = math.sqrt(sum(v * v for v in vectors[0]))
        assert abs(magnitude - 1.0) < 0.01

    async def test_similar_texts_have_higher_similarity(self) -> None:
        emb = SimpleEmbedding(dimension=128)
        vecs = await emb.embed([
            "python machine learning AI",
            "python deep learning neural networks",
            "cooking recipes pasta sauce",
        ])
        # ML texts should be more similar to each other than to cooking
        sim_ml = _cosine_similarity(vecs[0], vecs[1])
        sim_unrelated = _cosine_similarity(vecs[0], vecs[2])
        assert sim_ml > sim_unrelated


class TestCosineSimiliarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 0.001

    def test_different_length(self) -> None:
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestKnowledgeBase:
    async def test_ingest_document(self) -> None:
        kb = KnowledgeBase(tenant_id=uuid4())
        doc = Document(
            content="Aria Core is a deterministic AI agent framework.\n\nIt uses an FSM for execution.",
            title="README",
            source="docs",
        )
        chunks = await kb.ingest(doc)
        assert len(chunks) >= 1
        assert kb.document_count == 1
        assert kb.chunk_count >= 1

    async def test_search_returns_results(self) -> None:
        kb = KnowledgeBase(tenant_id=uuid4())
        await kb.ingest(Document(
            content="The FSM runtime provides deterministic agent execution with 8 states.",
            title="Runtime Docs",
        ))
        await kb.ingest(Document(
            content="Deep Bridge enables multi-model consensus voting across LLM providers.",
            title="Orchestration Docs",
        ))

        results = await kb.search("deterministic execution", top_k=2)
        assert len(results) >= 1
        assert results[0].score > 0

    async def test_search_empty_kb(self) -> None:
        kb = KnowledgeBase(tenant_id=uuid4())
        results = await kb.search("anything")
        assert len(results) == 0

    async def test_metadata_filter(self) -> None:
        kb = KnowledgeBase(tenant_id=uuid4())
        await kb.ingest(Document(content="API docs content", metadata={"type": "api"}))
        await kb.ingest(Document(content="User guide content", metadata={"type": "guide"}))

        results = await kb.search("content", metadata_filter={"type": "api"})
        for r in results:
            assert r.chunk.metadata.get("type") == "api"

    async def test_delete_document(self) -> None:
        kb = KnowledgeBase(tenant_id=uuid4())
        doc = Document(content="Temporary content")
        await kb.ingest(doc)
        assert kb.document_count == 1

        deleted = await kb.delete_document(doc.id)
        assert deleted is True
        assert kb.document_count == 0
        assert kb.chunk_count == 0

    async def test_delete_nonexistent(self) -> None:
        kb = KnowledgeBase(tenant_id=uuid4())
        assert await kb.delete_document(uuid4()) is False

    async def test_list_documents(self) -> None:
        kb = KnowledgeBase(tenant_id=uuid4())
        await kb.ingest(Document(content="Doc 1", title="First"))
        await kb.ingest(Document(content="Doc 2", title="Second"))
        docs = kb.list_documents()
        assert len(docs) == 2

    async def test_tenant_scoped(self) -> None:
        """Each KB is scoped to a tenant."""
        tid = uuid4()
        kb = KnowledgeBase(tenant_id=tid)
        doc = Document(content="Test")
        chunks = await kb.ingest(doc)
        assert chunks[0].tenant_id == tid

    async def test_min_score_filter(self) -> None:
        kb = KnowledgeBase(tenant_id=uuid4())
        await kb.ingest(Document(content="Very specific technical content about FSM states"))
        results = await kb.search("completely unrelated topic", min_score=0.99)
        assert len(results) == 0
