"""Manual Phase 1 demonstration for the Memory Knowledge Reservoir.

Run with: python test.py
This uses a temporary database so it does not alter the application's
``chroma_db/documents`` collection or leave demo entries in ``memory``.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from math import sqrt
import hashlib

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from memory_reservoir import MemoryKnowledgeReservoir


class DemoEmbeddingFunction(EmbeddingFunction[Documents]):
    """Tiny deterministic embedding for an offline storage/retrieval demo only."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors: Embeddings = []
        for text in input:
            vector = [0.0] * 64
            for word in text.lower().split():
                token = word.strip(".,?!").encode("utf-8")
                index = int.from_bytes(hashlib.sha256(token).digest()[:4]) % len(vector)
                vector[index] += 1.0
            magnitude = sqrt(sum(value * value for value in vector))
            if magnitude:
                vector = [value / magnitude for value in vector]
            vectors.append(vector)
        return vectors


def show_results(label: str, memories: list[dict]) -> None:
    print(f"\n{label}")
    for memory in memories:
        print(
            f"- {memory['title']} | distance={memory['distance']:.4f} | "
            f"relevant={memory['is_relevant']}"
        )
        print(f"  {memory['content']}")


def run_demo() -> None:
    with TemporaryDirectory() as temp_dir:
        reservoir = MemoryKnowledgeReservoir(
            db_path=str(Path(temp_dir) / "chroma_db"),
            distance_threshold=0.75,
            embedding_function=DemoEmbeddingFunction(),
        )
        print(f"Distance metric: {reservoir.distance_metric} (lower is better)")
        print(f"Initial empty search: {reservoir.retrieve('Ferrari Testarossa')}")

        reservoir.add(
            "Ferrari Testarossa",
            "The Ferrari Testarossa is a mid-engine sports car introduced in 1984.",
            source="retrieved document",
            query="Tell me about the Ferrari Testarossa",
        )
        reservoir.add(
            "Coronary Artery Disease",
            "Coronary artery disease narrows coronary arteries through plaque buildup.",
            source="retrieved document",
        )
        reservoir.add(
            "Electrocardiogram",
            "An ECG records the electrical activity of the heart.",
            source="retrieved document",
        )

        show_results("Similar-query search:", reservoir.retrieve("1984 Ferrari Testarossa sports car"))

        # A strict upper distance bound demonstrates a retrieval miss while
        # retaining the returned candidates for inspection.
        reservoir.distance_threshold = 0.01
        show_results("No sufficiently relevant memory:", reservoir.retrieve("How do volcanoes erupt?"))

        reservoir.distance_threshold = 0.75
        original_id = reservoir.add(
            "Ferrari Testarossa", "Outdated information", source="old source"
        )
        updated_id = reservoir.add(
            "Ferrari Testarossa",
            "Updated: the Testarossa was produced from 1984 to 1991.",
            source="new source",
        )
        updated = reservoir.collection.get(ids=[updated_id], include=["documents", "metadatas"])
        print(f"\nUpsert stable ID reused: {original_id == updated_id}")
        print(f"Updated content: {updated['documents'][0]}")
        print(f"Collection entries: {reservoir.collection.count()}")
        print(f"Statistics: {reservoir.statistics()}")
        # ChromaDB keeps Windows file handles open until its system stops.
        # Releasing them lets TemporaryDirectory safely remove demo data.
        reservoir.client._system.stop()


if __name__ == "__main__":
    run_demo()
