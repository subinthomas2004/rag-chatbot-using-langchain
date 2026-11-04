"""
Vector store management using ChromaDB (in-memory or persistent directory storage).
Handles indexing, embedding association, metadata filtering, and semantic search retrieval.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorStoreManager:
    def __init__(
        self,
        collection_name: str = "rag_knowledge_base",
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG Knowledge Base Documents"}
        )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add text chunks with their pre-calculated embeddings and metadata into ChromaDB.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("chunk_id", i))
            ids.append(f"{self.collection_name}_chunk_{chunk_id}")
            documents.append(chunk["chunk"])
            metadatas.append({
                "page": int(chunk.get("page", 1)),
                "source": str(chunk.get("source", "document"))
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(ids)

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity retrieval returning top_k matched document chunks with metadata and scores.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        matched_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                matched_docs.append({
                    "text": doc_text,
                    "page": meta.get("page", 1),
                    "source": meta.get("source", "unknown"),
                    "distance": float(dist)
                })

        return matched_docs

    def clear(self):
        """Reset the vector store collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG Knowledge Base Documents"}
        )
