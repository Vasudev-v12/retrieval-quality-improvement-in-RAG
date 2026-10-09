"""A small semantic cache for knowledge retrieved by the RAG system.

This module intentionally does not store chat history.  Each entry represents
reusable knowledge as a title-content pair and lives in its own ChromaDB
collection, separate from the static ``documents`` knowledge base.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Optional

import chromadb


class MemoryKnowledgeReservoir:
    """Persist and semantically retrieve cached RAG knowledge.

    ChromaDB returns a *distance*, not a universal similarity score.  The
    project's existing collection does not set ``hnsw:space``, so ChromaDB's
    default L2 distance is used (smaller values are more relevant).  The
    threshold is consequently an upper bound on distance and is kept here so
    it can be tuned without changing the RAG pipeline.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "memory",
        distance_threshold: Optional[float] = 1.0,
        embedding_function: Optional[Any] = None,
    ) -> None:
        if distance_threshold is not None and distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative or None")

        self.client = chromadb.PersistentClient(path=db_path)
        # A separate dynamic collection prevents cached knowledge from
        # changing the existing static `documents` collection.
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )
        self.distance_threshold = distance_threshold
        self.total_memory_queries = 0
        self.memory_hits = 0
        self.memory_misses = 0

    @staticmethod
    def _stable_id(title: str) -> str:
        """Build a deterministic ID so the same logical title is upserted."""
        normalized_title = " ".join(title.casefold().split())
        digest = hashlib.sha256(normalized_title.encode("utf-8")).hexdigest()
        return f"memory_{digest}"

    @property
    def distance_metric(self) -> str:
        """Return the configured Chroma HNSW metric (L2 when unspecified)."""
        metadata = self.collection.metadata or {}
        return str(metadata.get("hnsw:space", "l2")).lower()

    @property
    def hit_rate(self) -> float:
        if self.total_memory_queries == 0:
            return 0.0
        return self.memory_hits / self.total_memory_queries

    def statistics(self) -> dict[str, Any]:
        """Return lightweight per-instance query statistics."""
        return {
            "total_memory_queries": self.total_memory_queries,
            "memory_hits": self.memory_hits,
            "memory_misses": self.memory_misses,
            "hit_rate": self.hit_rate,
        }

    def add(
        self,
        title: str,
        content: str,
        source: Optional[str] = None,
        query: Optional[str] = None,
    ) -> str:
        """Add or replace a title-content memory and return its stable ID."""
        if not title or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not content or not content.strip():
            raise ValueError("content must be a non-empty string")

        memory_id = self._stable_id(title)
        metadata: dict[str, str] = {
            "title": title.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if source is not None:
            metadata["source"] = source
        if query is not None:
            metadata["query"] = query

        # Upsert replaces an earlier entry with the same stable title ID,
        # avoiding duplicate versions of one logical memory.
        self.collection.upsert(
            ids=[memory_id], documents=[content.strip()], metadatas=[metadata]
        )
        return memory_id

    def is_relevant(self, distance: Optional[float]) -> bool:
        """Evaluate relevance using this reservoir's lower-is-better metric."""
        if distance is None:
            return False
        return self.distance_threshold is None or distance <= self.distance_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantically retrieve memory entries and update hit/miss counters."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.total_memory_queries += 1
        count = self.collection.count()
        if count == 0:
            self.memory_misses += 1
            return []

        results = self.collection.query(
            query_texts=[query.strip()], n_results=min(top_k, count)
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        memories: list[dict[str, Any]] = []
        for index, memory_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = distances[index] if index < len(distances) else None
            memories.append(
                {
                    "id": memory_id,
                    "title": metadata.get("title", ""),
                    "content": documents[index],
                    "distance": distance,
                    "metadata": metadata,
                    "is_relevant": self.is_relevant(distance),
                }
            )

        if any(memory["is_relevant"] for memory in memories):
            self.memory_hits += 1
        else:
            self.memory_misses += 1
        return memories
