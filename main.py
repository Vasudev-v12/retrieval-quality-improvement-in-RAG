import os

from dotenv import load_dotenv

from query_gen import generate_queries
from retriever import ChromaRetriever
from llm import GeminiRAG


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")


retriever = ChromaRetriever(
    db_path="./chroma_db",
    collection_name="documents"
)

rag = GeminiRAG(API_KEY)

question = "What role do electrical signals play in the cardiac cycle, and how does a doctor visually measure this electrical activity?"#input("\nEnter your question:\n> ")

print("\nGenerating queries...\n")

queries = generate_queries(question)

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

print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)
print(answer)