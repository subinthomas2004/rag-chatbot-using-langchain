"""
Vector store management using Pinecone Serverless.
Handles upserting chunks with metadata and semantic similarity querying.
"""

import os
import time
from typing import List, Dict, Any, Optional


class VectorStoreManager:
    def __init__(
        self,
        index_name: str = "rag-chatbot-index",
        dimension: int = 384,
        api_key: Optional[str] = None,
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.pinecone_client = None
        self.index = None
        self.mode = "pinecone"
        self._init_pinecone()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_pinecone(self):
        """Connect to Pinecone and create the serverless index if needed."""
        if not self.api_key:
            raise ValueError(
                "Pinecone API key is required. "
                "Set PINECONE_API_KEY in your .env file or pass api_key= explicitly."
            )

        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError(
                "pinecone package is not installed. Run: pip install pinecone>=5.0.0"
            ) from exc

        self.pinecone_client = Pinecone(api_key=self.api_key)

        existing_names = [idx["name"] for idx in self.pinecone_client.list_indexes()]
        if self.index_name not in existing_names:
            print(f"Creating Pinecone index '{self.index_name}' ({self.dimension}d, cosine) ...")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
            # Wait until the index is ready
            self._wait_for_index()

        self.index = self.pinecone_client.Index(self.index_name)
        print(f"✅ Connected to Pinecone index '{self.index_name}'.")

    def _wait_for_index(self, timeout: int = 60):
        """Poll until the index state is 'Ready'."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            statuses = self.pinecone_client.list_indexes()
            for idx in statuses:
                if idx["name"] == self.index_name:
                    state = idx.get("status", {}).get("state", "")
                    if state == "Ready":
                        return
            time.sleep(2)
        raise TimeoutError(
            f"Pinecone index '{self.index_name}' did not become ready within {timeout}s."
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """Upsert chunks with their embeddings into the Pinecone index."""
        if not chunks:
            return 0

        vectors = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{chunk.get('chunk_id', i)}"
            metadata = {
                # Pinecone metadata values must be str / int / float / bool
                "text":   chunk["chunk"][:1000],   # safety cap for large metadata
                "page":   int(chunk.get("page", 1)),
                "source": str(chunk.get("source", "document"))
            }
            vectors.append((chunk_id, embeddings[i], metadata))

        # Pinecone recommends batches of ≤100 vectors
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size])

        return len(vectors)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Return the top-k most similar chunks from Pinecone."""
        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as exc:
            raise RuntimeError(f"Pinecone query failed: {exc}") from exc

        results = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append({
                "text":     meta.get("text", ""),
                "page":     meta.get("page", 1),
                "source":   meta.get("source", "unknown"),
                "score":    float(match.get("score", 0.0))   # cosine similarity (higher = better)
            })
        return results

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self):
        """Delete all vectors from the index (keeps the index itself)."""
        try:
            self.index.delete(delete_all=True)
            print(f"🗑️  All vectors deleted from Pinecone index '{self.index_name}'.")
        except Exception as exc:
            print(f"Warning: could not clear index — {exc}")

    def stats(self) -> Dict[str, Any]:
        """Return index statistics (vector count, dimension, etc.)."""
        return self.index.describe_index_stats()
