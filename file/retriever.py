import chromadb

class ChromaRetriever:
    def __init__(self, db_path="./chroma_db", collection_name="documents"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, query, top_k=3):
        results = self.collection.query(query_texts=query, n_results=top_k)
        docs = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results.get("distances", [[]])[0]
        for i in range(len(ids)):
            docs.append({
                "id": ids[i],
                "content": documents[i],
                "distance": distances[i] if distances else None
            })
        return docs

    def retrieve_multiple(self, queries, top_k=3):
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
                    merged[doc_id]["retrieved_by"].append(query)
        return list(merged.values())