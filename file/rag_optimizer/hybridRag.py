import hashlib
import os
import uuid
import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


def compute_corpus_version(collection) -> str:
    """Hash of a collection's id list; changes on any insert/upsert/delete.

    The memory reservoir stores this with every cached answer, so a cached
    answer stops being served once the `documents` collection changes.
    """
    all_ids = sorted(collection.get(include=[])["ids"])
    return hashlib.sha256("".join(all_ids).encode("utf-8")).hexdigest()


class ChromaInserter:
    def __init__(self, db_path="./chroma_db", collection_name="documents"):
        # Connect to the persistent database path
        self.client = chromadb.PersistentClient(path=db_path)
        # Get or create the collection (uses default embeddings automatically)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """Splits a document text into smaller overlapping segments."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def _process_and_insert(self, full_text, document_id, source_name, chunk_size, overlap):
        """Internal helper to chunk text, generate metadata, and upsert to ChromaDB."""
        # Chunk the text
        text_chunks = self._chunk_text(full_text, chunk_size, overlap)

        # Combine the document UUID and chunk number for a unique primary key
        ids = [f"{document_id}_chunk_{i}" for i in range(len(text_chunks))]
        metadatas = [
            {
                "document_id": document_id, 
                "source_name": source_name,
                "chunk_index": i
            } 
            for i in range(len(text_chunks))
        ]

        # Insert/Upsert into your ChromaDB collection
        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=text_chunks
        )
        print(f"Successfully processed '{source_name}' -> Assigned UUID: {document_id} ({len(text_chunks)} chunks).")
        return document_id

    def insert_pdf(self, file_path, document_id=None, chunk_size=1000, overlap=200):
        """Parses a single PDF, chunks its text, and inserts it using a unique UUID."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file path '{file_path}' does not exist.")

        if not document_id:
            document_id = str(uuid.uuid4())

        # Extract text from the PDF
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        file_name = os.path.basename(file_path)
        return self._process_and_insert(full_text, document_id, file_name, chunk_size, overlap)

    def insert_string(self, text_data, source_name="raw_string_input", document_id=None, chunk_size=400, overlap=100):
        """Accepts a raw string variable, chunks it, and inserts it using a unique UUID."""
        if not text_data or not text_data.strip():
            raise ValueError("The text_data string cannot be empty.")

        if not document_id:
            document_id = str(uuid.uuid4())

        return self._process_and_insert(text_data, document_id, source_name, chunk_size, overlap)

class ChromaHybridRag:
    def __init__(self, db_path="./chroma_db", collection_name="documents"):
        # Connect to local persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # In-memory storage for BM25 sparse index
        self.corpus_chunks = []
        self.chunk_ids = []
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        """Automatically reloads existing documents from ChromaDB to populate the BM25 index on startup."""
        try:
            data = self.collection.get(include=["documents"])
            if data and data["documents"]:
                self.corpus_chunks = data["documents"]
                self.chunk_ids = data["ids"]
            else:
                self.corpus_chunks = []
                self.chunk_ids = []
        except Exception:
            self.corpus_chunks = []
            self.chunk_ids = []

    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """Splits document text into smaller overlapping segments."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def _process_and_insert(self, full_text, document_id, source_name, chunk_size, overlap):
        """Internal helper to chunk text, upsert to ChromaDB (dense), and update BM25 (sparse)."""
        text_chunks = self._chunk_text(full_text, chunk_size, overlap)

        ids = [f"{document_id}_chunk_{i}" for i in range(len(text_chunks))]
        metadatas = [
            {
                "document_id": document_id, 
                "source_name": source_name,
                "chunk_index": i
            } 
            for i in range(len(text_chunks))
        ]

        # 1. Upsert into ChromaDB (Dense Vector Search)
        self.collection.upsert(
            ids=ids,
            metadatas=metadatas,
            documents=text_chunks
        )

        # 2. Update local BM25 corpus tracker
        self.corpus_chunks.extend(text_chunks)
        self.chunk_ids.extend(ids)

        print(f"Successfully processed '{source_name}' -> Assigned UUID: {document_id} ({len(text_chunks)} chunks).")
        return document_id

    def insert_pdf(self, file_path, document_id=None, chunk_size=1000, overlap=200):
        """Parses a single PDF, chunks its text, and inserts it."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file path '{file_path}' does not exist.")

        if not document_id:
            document_id = str(uuid.uuid4())

        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        file_name = os.path.basename(file_path)
        return self._process_and_insert(full_text, document_id, file_name, chunk_size, overlap)

    def insert_string(self, text_data, source_name="raw_string_input", document_id=None, chunk_size=400, overlap=100):
        """Accepts a raw string variable, chunks it, and inserts it."""
        if not text_data or not text_data.strip():
            raise ValueError("The text_data string cannot be empty.")

        if not document_id:
            document_id = str(uuid.uuid4())

        return self._process_and_insert(text_data, document_id, source_name, chunk_size, overlap)

    def _get_bm25_retriever(self):
        """Builds BM25 index on the current corpus chunks."""
        tokenized_corpus = [doc.lower().split() for doc in self.corpus_chunks]
        return BM25Okapi(tokenized_corpus)

    def retrieve(self, query, top_k=3, dense_weight=0.6, sparse_weight=0.4, rrf_k=60):
        """Performs hybrid retrieval (Dense Chroma + Sparse BM25) for a single query using RRF."""
        if not self.corpus_chunks:
            return []

        # --- 1. Sparse Search (BM25 Lexical Matching) ---
        bm25 = self._get_bm25_retriever()
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        ranked_sparse_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        sparse_ranking = [self.chunk_ids[i] for i in ranked_sparse_indices[:top_k * 2]]

        # --- 2. Dense Search (ChromaDB Semantic Matching) ---
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2
        )
        dense_ranking = dense_results["ids"][0] if dense_results["ids"] and dense_results["ids"][0] else []
        dense_distances = dense_results.get("distances", [[]])[0]
        
        distance_map = {}
        if dense_ranking and dense_distances:
            for doc_id, dist in zip(dense_ranking, dense_distances):
                distance_map[doc_id] = dist

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        fusion_scores = {}
        def add_ranks(ranking, weight):
            for rank, doc_id in enumerate(ranking):
                if doc_id not in fusion_scores:
                    fusion_scores[doc_id] = 0.0
                fusion_scores[doc_id] += weight * (1.0 / (rrf_k + (rank + 1)))

        add_ranks(dense_ranking, dense_weight)
        add_ranks(sparse_ranking, sparse_weight)

        # Sort combined results by RRF score descending
        sorted_fused = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunk_ids = [doc_id for doc_id, score in sorted_fused[:top_k]]

        if not top_chunk_ids:
            return []

        # Fetch matching contents from ChromaDB
        final_results = self.collection.get(ids=top_chunk_ids, include=["documents"])
        
        id_to_doc = {}
        if final_results and final_results["ids"]:
            for i in range(len(final_results["ids"])):
                id_to_doc[final_results["ids"][i]] = final_results["documents"][i]

        docs = []
        for doc_id in top_chunk_ids:
            if doc_id in id_to_doc:
                docs.append({
                    "id": doc_id,
                    "content": id_to_doc[doc_id],
                    "distance": distance_map.get(doc_id, None)
                })
        return docs

    def retrieve_multiple(self, queries, top_k=3):
        """Retrieves documents for multiple queries, merging them and tracking source queries."""
        merged = {}
        for query in queries:
            results = self.retrieve(query, top_k)
            for doc in results:
                doc_id = doc["id"]
                if doc_id not in merged:
                    merged[doc_id] = {
                        "id": doc_id,
                        "content": doc["content"],
                        "distance": doc["distance"],
                        "retrieved_by": [query]
                    }
                else:
                    if query not in merged[doc_id]["retrieved_by"]:
                        merged[doc_id]["retrieved_by"].append(query)
        return list(merged.values())
