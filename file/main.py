import os

from dotenv import load_dotenv

from optimizer import optimize_prompt, parse_queries
from retriever import ChromaRetriever
from llm import GeminiRAG
import query_gen
from memory_reservoir import MemoryKnowledgeReservoir


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")


retriever = ChromaRetriever(
    db_path="./chroma_db",
    collection_name="documents"
)

rag = GeminiRAG(API_KEY)

# Separate semantic cache of reusable knowledge (title/content pairs), kept
# in its own "memory" collection so it never mutates the static "documents"
# knowledge base.
memory = MemoryKnowledgeReservoir(db_path="./chroma_db")

question = "What role do electrical signals play in the cardiac cycle, and how does a doctor visually measure this electrical activity?"#input("\nEnter your question:\n> ")

print("\nChecking memory cache...\n")

cached_memories = memory.retrieve(question, top_k=3)
# `retrieve` returns hits sorted by ascending distance, so the first entry
# is the closest match; if any hit clears the threshold, so does this one.
best_memory = cached_memories[0] if cached_memories else None

if best_memory and best_memory["is_relevant"]:
    print(f"Cache hit: \"{best_memory['title']}\" (distance={best_memory['distance']:.4f})\n")
    print("=" * 80)
    print("FINAL ANSWER (from memory cache)")
    print("=" * 80)
    print(best_memory["content"])

else:
    print("Cache miss, running full retrieval pipeline...\n")
    print("Generating queries...\n")

    f_t, max_l = query_gen.filter_tokens(question)
    queries = query_gen.generate_queries(question=question,max_l=max_l)
    
    for i, q in enumerate(queries, start=1):
        print(f"Q{i}: {q}")
    print("\nRetrieving documents...\n")

    documents = retriever.retrieve_multiple(
        queries,
        top_k=5
    )

    print(f"Retrieved {len(documents)} unique documents\n")
    for idx, doc in enumerate(documents[:10], start=1):
        print("=" * 80)
        print(f"Document {idx}")
        print(
            f"Retrieved by: {len(doc['retrieved_by'])} queries"
        )
        print(doc["content"][:300])
        print()

    print("\nGenerating answer...\n")

    answer = rag.answer(
        question,
        documents
    )

    memory.add(
        title=question,
        content=answer,
        source="generated",
        query=question,
    )

    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(answer)

print(f"\nMemory cache stats: {memory.statistics()}")