"""
Embeddings generator supporting multiple backends:
1. SentenceTransformers (e.g. BAAI/bge-small-en-v1.5, all-MiniLM-L6-v2)
2. OpenAI embeddings (text-embedding-3-small, etc.)
3. Google Gemini embeddings (models/embedding-001 or text-embedding-004)
4. Fast Deterministic Hash/TF-IDF baseline (for offline zero-network tests)
"""

import math
import re
from typing import List, Optional


class EmbeddingsManager:
    def __init__(
        self,
        provider: str = "sentence-transformers",
        model_name: str = "BAAI/bge-small-en-v1.5",
        api_key: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self._model = None
        self._init_backend()

    def _init_backend(self):
        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                # Fallback to local baseline if torch/sentence_transformers has environment restrictions
                self._model = None
                self.fallback_error = str(e)
        elif self.provider == "openai":
            from openai import OpenAI
            self._model = OpenAI(api_key=self.api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self._model = genai

    def _pseudo_embedding(self, text: str, dim: int = 384) -> List[float]:
        """High-entropy deterministic bag-of-words / character n-gram pseudo-embedding."""
        vec = [0.0] * dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = hash(tok)
            idx = abs(h) % dim
            val = (h % 100) / 100.0
            vec[idx] += 1.0 + val
        # Normalize vector
        norm = math.sqrt(sum(x * x for x in vec)) or 1e-9
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.provider == "sentence-transformers" and self._model is not None:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return [e.tolist() for e in embeddings]

        if self.provider == "openai" and self._model is not None:
            resp = self._model.embeddings.create(
                input=texts,
                model=self.model_name or "text-embedding-3-small"
            )
            return [item.embedding for item in resp.data]

        if self.provider == "gemini" and self._model is not None:
            results = []
            for t in texts:
                res = self._model.embed_content(
                    model=self.model_name or "models/embedding-001",
                    content=t
                )
                results.append(res['embedding'])
            return results

        # Fallback pseudo embedding for zero-dependency or sandboxed environments
        return [self._pseudo_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        results = self.embed_documents([text])
        return results[0] if results else [0.0] * 384
