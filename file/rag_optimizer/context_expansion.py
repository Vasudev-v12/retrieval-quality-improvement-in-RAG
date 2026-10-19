"""Second-pass context expansion on top of first-pass retrieval results.

Two strategies, both working on the ChromaDB collection behind a
ChromaHybridRag retriever:

1. Centroid search - average the dense embeddings of the top first-pass
   chunks into one "context vector" and run a second similarity search with
   it. This pulls in chunks on the same broader topic, even when they share
   no words with the original question.

2. Neighbor window - read document_id / chunk_index from the metadata of the
   top first-pass chunks and fetch chunk_index - w .. chunk_index + w from the
   same document, so text cut off at a chunk boundary is restored.

Every returned doc keeps the retriever's dict shape (id, content, distance,
retrieved_by) plus an "expansion" field: "seed" for first-pass chunks,
"centroid" or "neighbor" for chunks added here.
"""

from __future__ import annotations
from typing import Any
import numpy as np

EXPANSION_MODES = ("none", "centroid", "neighbor", "both")


class ContextExpander:
    def __init__(self, retriever, centroid_seeds=3, centroid_k=3, neighbor_seeds=5, window=1,
                 max_chunks=None):
        self.collection = retriever.collection
        self.centroid_seeds = centroid_seeds  # first-pass chunks averaged into the centroid
        self.centroid_k = centroid_k          # new chunks the centroid search may add
        self.neighbor_seeds = neighbor_seeds  # chunks whose neighbors get fetched
        self.window = window                  # neighbors on each side of a seed
        self.max_chunks = max_chunks          # cap on the returned context (None = no cap)

    @staticmethod
    def rank_seeds(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Order first-pass docs by how many queries found them, then by dense distance."""
        def key(doc):
            dist = doc.get("distance")
            return (-len(doc.get("retrieved_by", [])), dist if dist is not None else float("inf"))
        return sorted(docs, key=key)

    def centroid_search(self, seeds: list[dict[str, Any]], exclude: set[str]) -> list[dict[str, Any]]:
        """Average the embeddings of `seeds` and query the collection with that vector."""
        if not seeds or self.centroid_k <= 0:
            return []
        data = self.collection.get(ids=[d["id"] for d in seeds], include=["embeddings"])
        embeddings = data.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return []

        centroid = np.mean(np.asarray(embeddings, dtype=np.float32), axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:  # chunk embeddings are unit length; keep the centroid comparable
            centroid = centroid / norm

        # Over-fetch so there are still centroid_k new chunks after dropping known ones.
        n_results = min(self.centroid_k + len(exclude), self.collection.count())
        results = self.collection.query(
            query_embeddings=[centroid.tolist()],
            n_results=n_results,
            include=["documents", "distances"],
        )
        added = []
        for doc_id, content, dist in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
            if doc_id in exclude:
                continue
            added.append({
                "id": doc_id,
                "content": content,
                "distance": dist,
                "retrieved_by": [],
                "expansion": "centroid",
            })
            if len(added) == self.centroid_k:
                break
        return added

    def neighbor_window(self, seeds: list[dict[str, Any]], exclude: set[str]) -> dict[str, list[dict[str, Any]]]:
        """Fetch adjacent chunks of each seed. Returns {seed_id: [neighbors in document order]}."""
        if not seeds or self.window <= 0:
            return {}
        meta = self.collection.get(ids=[d["id"] for d in seeds], include=["metadatas"])

        # Collect the wanted chunk indices per source document.
        wanted: dict[str, dict[int, str]] = {}  # document_id -> {chunk_index: seed_id}
        for seed_id, m in zip(meta["ids"], meta["metadatas"]):
            if not m or "document_id" not in m or "chunk_index" not in m:
                continue
            idx = int(m["chunk_index"])
            for offset in range(-self.window, self.window + 1):
                if offset != 0 and idx + offset >= 0:
                    wanted.setdefault(m["document_id"], {}).setdefault(idx + offset, seed_id)

        neighbors: dict[str, list[dict[str, Any]]] = {}
        seen = set(exclude)
        for document_id, index_to_seed in wanted.items():
            found = self.collection.get(
                where={"$and": [
                    {"document_id": {"$eq": document_id}},
                    {"chunk_index": {"$in": sorted(index_to_seed)}},
                ]},
                include=["documents", "metadatas"],
            )
            rows = sorted(zip(found["ids"], found["documents"], found["metadatas"]),
                          key=lambda r: r[2]["chunk_index"])
            for doc_id, content, m in rows:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                neighbors.setdefault(index_to_seed[int(m["chunk_index"])], []).append({
                    "id": doc_id,
                    "content": content,
                    "distance": None,
                    "retrieved_by": [],
                    "expansion": "neighbor",
                    "chunk_index": int(m["chunk_index"]),
                })
        return neighbors

    def expand(self, docs: list[dict[str, Any]], mode: str = "both") -> list[dict[str, Any]]:
        """Run the second pass over first-pass `docs` and return the combined context.

        Neighbors are placed directly around the chunk they belong to (in
        document order) so the LLM reads contiguous text; centroid chunks
        follow the first-pass results.

        With max_chunks set, only the top max_chunks are kept, by priority:
        seeds in rank order, each together with its neighbors, and the
        centroid chunks right after the seeds they were averaged from.
        The kept chunks stay in the reading order described above.
        """
        if mode not in EXPANSION_MODES:
            raise ValueError(f"mode must be one of {EXPANSION_MODES}, got {mode!r}")
        seeds = [{**d, "expansion": "seed"} for d in self.rank_seeds(docs)]
        if mode == "none" or not seeds:
            return seeds[: self.max_chunks]

        known = {d["id"] for d in seeds}
        centroid_docs = []
        if mode in ("centroid", "both"):
            centroid_docs = self.centroid_search(seeds[: self.centroid_seeds], known)
            known |= {d["id"] for d in centroid_docs}

        neighbors = {}
        if mode in ("neighbor", "both"):
            neighbors = self.neighbor_window(seeds[: self.neighbor_seeds], known)

        seed_index = {}
        if neighbors:
            meta = self.collection.get(ids=list(neighbors), include=["metadatas"])
            seed_index = {i: int(m["chunk_index"]) for i, m in zip(meta["ids"], meta["metadatas"])}

        groups = []  # one group per seed: its neighbors around it, in document order
        for seed in seeds:
            around = neighbors.get(seed["id"], [])
            idx = seed_index.get(seed["id"], 0)
            groups.append([n for n in around if n["chunk_index"] < idx] + [seed]
                          + [n for n in around if n["chunk_index"] > idx])

        if self.max_chunks is not None:
            priority = [d for g in groups[: self.centroid_seeds] for d in g] + centroid_docs \
                + [d for g in groups[self.centroid_seeds:] for d in g]
            keep = {d["id"] for d in priority[: self.max_chunks]}
            groups = [[d for d in g if d["id"] in keep] for g in groups]
            centroid_docs = [d for d in centroid_docs if d["id"] in keep]

        return [d for g in groups for d in g] + centroid_docs
