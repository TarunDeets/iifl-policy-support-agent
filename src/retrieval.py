"""
retrieval.py

Embedding-based semantic retrieval over the FAQ chunk pool.

Design choice: we embed only the FAQ *question* text (not question+answer).
Customers phrase things closer to how another question is phrased than to
how a long answer is phrased, so question-to-question similarity gives a
cleaner match. The matched chunk's answer is what gets handed to the LLM
as grounding context.

All chunk embeddings are computed once at startup and kept in memory
(no vector database needed for ~20-100 short FAQ items).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from document_parser import FAQChunk

if TYPE_CHECKING:
    # Only needed for type hints; avoids a hard import-time dependency on the
    # openai package for anything that doesn't actually need to construct one.
    from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class ScoredChunk:
    chunk: FAQChunk
    similarity: float


class FAQRetriever:
    """Holds precomputed embeddings for every FAQ chunk and answers similarity queries."""

    def __init__(self, client: OpenAI, chunks: list[FAQChunk]):
        self.client = client
        self.chunks = chunks
        self._embeddings = self._embed_all(chunks)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Call the OpenAI embeddings API and return an (n, d) normalized matrix."""
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        # Normalize rows so dot product == cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guard against a degenerate zero vector
        return vectors / norms

    def _embed_all(self, chunks: list[FAQChunk]) -> np.ndarray:
        questions = [c.question for c in chunks]
        return self._embed_texts(questions)

    def top_matches(self, query: str, k: int = 3) -> list[ScoredChunk]:
        """Return the k most similar FAQ chunks to the customer's question, best first."""
        query_vec = self._embed_texts([query])[0]
        similarities = self._embeddings @ query_vec  # cosine similarity, since rows are normalized
        top_indices = np.argsort(-similarities)[:k]
        return [ScoredChunk(chunk=self.chunks[i], similarity=float(similarities[i])) for i in top_indices]
