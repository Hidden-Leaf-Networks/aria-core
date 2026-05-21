"""KnowledgeBase — tenant-scoped document store with vector search.

In-memory implementation with cosine similarity. For production,
swap the vector store with pgvector, Qdrant, Pinecone, etc.

Embedding is pluggable via EmbeddingProvider protocol.
"""

from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID

from aria_core.knowledge.models import (
    Chunk,
    ChunkStrategy,
    Document,
    SearchResult,
)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


class SimpleEmbedding:
    """Simple bag-of-words embedding for testing/dev.

    NOT for production — use OpenAI, Cohere, or local models.
    """

    def __init__(self, dimension: int = 128) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Create simple hash-based embeddings."""
        results = []
        for text in texts:
            words = text.lower().split()
            vec = [0.0] * self._dimension
            for word in words:
                idx = hash(word) % self._dimension
                vec[idx] += 1.0
            # Normalize
            magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / magnitude for v in vec]
            results.append(vec)
        return results


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _chunk_text(
    text: str,
    strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into chunks."""
    if strategy == ChunkStrategy.PARAGRAPH:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        # Merge small paragraphs
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) > chunk_size and current:
                chunks.append(current)
                current = p
            else:
                current = f"{current}\n\n{p}" if current else p
        if current:
            chunks.append(current)
        return chunks or [text]

    elif strategy == ChunkStrategy.SENTENCE:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) > chunk_size and current:
                chunks.append(current)
                current = s
            else:
                current = f"{current} {s}" if current else s
        if current:
            chunks.append(current)
        return chunks or [text]

    else:  # FIXED
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks or [text]


class KnowledgeBase:
    """Tenant-scoped knowledge base with vector search.

    In-memory vector store. For production, swap with pgvector/Qdrant.
    """

    def __init__(
        self,
        tenant_id: UUID,
        embedding_provider: EmbeddingProvider | None = None,
        chunk_strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        chunk_size: int = 500,
    ) -> None:
        self.tenant_id = tenant_id
        self._embedder = embedding_provider or SimpleEmbedding()
        self._chunk_strategy = chunk_strategy
        self._chunk_size = chunk_size
        self._documents: dict[UUID, Document] = {}
        self._chunks: list[Chunk] = []

    async def ingest(self, document: Document) -> list[Chunk]:
        """Ingest a document: chunk, embed, and store."""
        document = document.model_copy(update={"tenant_id": self.tenant_id})
        self._documents[document.id] = document

        # Chunk the content
        text_chunks = _chunk_text(
            document.content,
            strategy=self._chunk_strategy,
            chunk_size=self._chunk_size,
        )

        # Embed all chunks
        embeddings = await self._embedder.embed(text_chunks)

        # Create chunk objects
        chunks = []
        for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk = Chunk(
                document_id=document.id,
                tenant_id=self.tenant_id,
                content=text,
                index=i,
                embedding=embedding,
                metadata={
                    "document_title": document.title,
                    "document_source": document.source,
                    **document.metadata,
                },
            )
            chunks.append(chunk)

        self._chunks.extend(chunks)
        return chunks

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the knowledge base using vector similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            metadata_filter: Filter chunks by metadata key-value pairs
        """
        if not self._chunks:
            return []

        # Embed the query
        query_embedding = (await self._embedder.embed([query]))[0]

        # Score all chunks
        scored: list[tuple[float, Chunk]] = []
        for chunk in self._chunks:
            if not chunk.embedding:
                continue

            # Metadata filter
            if metadata_filter:
                match = all(
                    chunk.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                )
                if not match:
                    continue

            score = _cosine_similarity(query_embedding, chunk.embedding)
            if score >= min_score:
                scored.append((score, chunk))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build results
        results = []
        for score, chunk in scored[:top_k]:
            doc = self._documents.get(chunk.document_id)
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                document_title=doc.title if doc else "",
                document_source=doc.source if doc else "",
            ))

        return results

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and its chunks."""
        if document_id not in self._documents:
            return False
        del self._documents[document_id]
        self._chunks = [c for c in self._chunks if c.document_id != document_id]
        return True

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def list_documents(self) -> list[Document]:
        return sorted(self._documents.values(), key=lambda d: d.created_at, reverse=True)
