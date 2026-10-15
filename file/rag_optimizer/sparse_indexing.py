from chromadb import Client, Schema, SparseVectorIndexConfig, K, Search, Knn, Rrf
from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction

# 1. Initialize the Chroma Client
client = Client()

# 2. Define a custom Schema with a Sparse Index configuration
# Chroma natively handles the default dense vector index automatically.
schema = Schema()
sparse_ef = ChromaBm25EmbeddingFunction()

schema.create_index(
    config=SparseVectorIndexConfig(
        source_key=K.DOCUMENT,        # Generate sparse vectors directly from document text
        embedding_function=sparse_ef  # Use BM25 algorithm for keyword tracking
    ),
    key="sparse_embedding"            # The metadata key where sparse indices live
)

# 3. Create a collection bound to the defined schema
collection = client.create_collection(
    name="hybrid_search_collection",
    schema=schema
)

# 4. Ingest documents 
# Chroma creates both the dense embedding and the sparse index under the hood
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "ChromaDB natively supports sparse vector indexing for hybrid keyword lookup.",
        "Deep learning systems rely heavily on dense embedding vectors for semantic contextualization.",
        "Hybrid search merges lexical BM25 scores with dense similarity models using RRF fusion."
    ]
)

# 5. Build the Hybrid Retrieval Query
query_text = "sparse vector hybrid search"

# Set up the dense vector ranker (Matches contextual meaning)
dense_rank = Knn(
    query=query_text, 
    return_rank=True, 
    limit=50
)

# Set up the sparse vector ranker (Matches exact keywords)
sparse_rank = Knn(
    query=query_text, 
    key="sparse_embedding",  # Tells Chroma to use the custom sparse index key
    return_rank=True, 
    limit=50
)

# Combine both strategies using Reciprocal Rank Fusion (RRF)
# Adjust weights depending on whether meaning or exact-keywords matter more to you
hybrid_rank = Rrf(
    ranks=[dense_rank, sparse_rank],
    weights=[0.6, 0.4],  # 60% importance to Dense, 40% to Sparse
    k=60                 # Standard RRF constant determining ranking smoothing
)

# Build the complete Search Query object
search_query = (
    Search()
    .rank(hybrid_rank)
    .limit(3)
    .select(K.DOCUMENT, K.SCORE)
)

# 6. Execute Retrieval
results = collection.search(search_query)

# 7. Print the retrieved chunks
print("--- Hybrid Search Results ---")
print(results)
