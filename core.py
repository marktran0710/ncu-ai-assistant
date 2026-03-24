from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text:     str
    score:    float
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, text={self.text[:60]!r})"


class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        logger.info(f"Loading embedder: {model_name} on {device}")
        self.model_name = model_name
        self.model      = SentenceTransformer(model_name, device=device)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([text.strip()], convert_to_numpy=True)[0]

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot embed empty list.")
        logger.info(f"Embedding {len(texts)} documents")
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )


class VectorIndex:
    INDEX_VERSION = 2

    def __init__(self, embedder: LocalEmbedder):
        self.embedder  = embedder
        self.vectors:   np.ndarray = np.empty((0,), dtype=np.float32)
        self.documents: list[dict] = []

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "version":    self.INDEX_VERSION,
                "model_name": self.embedder.model_name,
                "vectors":    self.vectors,
                "documents":  self.documents,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Index saved → {path} ({len(self.documents)} docs)")

    @classmethod
    def load(cls, path: str | Path, embedder: LocalEmbedder) -> "VectorIndex":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        if "vectors" not in data or "documents" not in data:
            raise ValueError(f"Corrupt index: {path}")
        stored = data.get("model_name")
        if stored and stored != embedder.model_name:
            logger.warning(f"Index built with '{stored}' but embedder is '{embedder.model_name}'.")
        inst           = cls(embedder)
        inst.vectors   = np.array(data["vectors"], dtype=np.float32)
        inst.documents = data["documents"]
        logger.info(f"Index loaded ← {path} ({len(inst.documents)} docs)")
        return inst

    def add_documents(self, documents: list[dict], batch_size: int = 64) -> None:
        if not documents:
            return
        for i, doc in enumerate(documents):
            if "text" not in doc:
                raise ValueError(f"Document {i} missing 'text' key")
        texts    = [d["text"] for d in documents]
        new_vecs = self.embed_documents(texts, batch_size).astype(np.float32)
        self.vectors   = new_vecs if self.vectors.size == 0 \
                         else np.vstack([self.vectors, new_vecs])
        self.documents.extend(documents)

    def build(self, documents: list[dict], batch_size: int = 64) -> None:
        self.vectors   = np.empty((0,), dtype=np.float32)
        self.documents = []
        self.add_documents(documents, batch_size)

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.embedder.embed_documents(texts, batch_size)

    def search(
        self,
        query:           str,
        top_k:           int            = 3,
        threshold:       float          = 0.35,
        metadata_filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        if not self.documents:
            return []
        if metadata_filter:
            idxs = [
                i for i, d in enumerate(self.documents)
                if all(d.get(k) == v for k, v in metadata_filter.items())
            ]
            if not idxs:
                return []
            cand_vecs = self.vectors[idxs]
        else:
            idxs      = list(range(len(self.documents)))
            cand_vecs = self.vectors

        q = self.embedder.embed_query(query).astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return []

        norms  = np.linalg.norm(cand_vecs, axis=1)
        scores = np.dot(cand_vecs, q) / (norms * q_norm + 1e-9)
        order  = np.argsort(scores)[::-1]

        results = []
        for pos in order:
            if len(results) >= top_k:
                break
            sc = float(scores[pos])
            if sc < threshold:
                break
            orig = idxs[pos]
            results.append(SearchResult(
                text     = self.documents[orig]["text"],
                score    = sc,
                metadata = {k: v for k, v in self.documents[orig].items() if k != "text"},
            ))
        return results

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        dim = self.vectors.shape[1] if self.vectors.ndim == 2 else 0
        return f"VectorIndex(docs={len(self.documents)}, dim={dim}, model='{self.embedder.model_name}')"