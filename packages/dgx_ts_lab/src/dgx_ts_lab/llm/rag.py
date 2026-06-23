"""RAG retriever for mission procedures + reference docs.

Phase 11 ships with a numpy-only cosine retriever (no FAISS dep) — keeps
the air-gap dep surface small and indexes built on this side can be
checked into git as plain .npz files.

Two modes:
    * lexical fallback   — TF-IDF cosine over scikit-learn
    * embedded            — caller passes an `embed_fn(text) -> np.ndarray`
                            (e.g., a sentence-transformer they happen to
                            have on disk); we just normalize + cosine.

Documents are simple `RAGDocument` records — path + title + chunked text.
Pass a list of chunked docs at construction; query returns top-k hits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RAGDocument:
    """A single retrievable chunk."""

    doc_id: str                  # e.g., "procedures/eps_safe_mode.md#chunk_2"
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RAGHit:
    """A retrieval hit with similarity score."""

    document: RAGDocument
    score: float


class CosineRAGIndex:
    """Numpy cosine retriever. Use ``add_lexical`` or ``add_embeddings``.

    Lexical path uses scikit-learn TfidfVectorizer (already a dep).
    """

    def __init__(self) -> None:
        self._docs: list[RAGDocument] = []
        self._vectors: np.ndarray | None = None       # (N, D) row-normalized
        self._vectorizer = None                       # sklearn TfidfVectorizer if lexical

    @property
    def n_docs(self) -> int:
        return len(self._docs)

    # ── Index building ───────────────────────────────────────────────────

    def add_lexical(self, documents: Sequence[RAGDocument]) -> None:
        """Build a TF-IDF index from documents (sklearn). Replaces any prior index."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._docs = list(documents)
        self._vectorizer = TfidfVectorizer(
            stop_words="english", lowercase=True, ngram_range=(1, 2),
        )
        mat = self._vectorizer.fit_transform([d.text for d in self._docs])
        # L2-normalize rows so dot = cosine
        norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        # Convert to dense float32 — RAG corpora here are < 10k chunks
        dense = mat.toarray().astype(np.float32) / norms[:, None]
        self._vectors = dense

    def add_embeddings(
        self,
        documents: Sequence[RAGDocument],
        embed_fn: Callable[[Sequence[str]], np.ndarray],
    ) -> None:
        """Build an index by calling user-supplied ``embed_fn`` on every doc text."""
        self._docs = list(documents)
        vecs = np.asarray(embed_fn([d.text for d in self._docs]), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = vecs / norms
        self._vectorizer = None

    # ── Query ────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    ) -> list[RAGHit]:
        if self._vectors is None or not self._docs:
            return []
        if self._vectorizer is not None:
            q = self._vectorizer.transform([query_text]).toarray().astype(np.float32)[0]
            q_norm = np.linalg.norm(q)
            q = q / (q_norm if q_norm > 0 else 1.0)
        else:
            if embed_fn is None:
                raise ValueError(
                    "embedded index requires embed_fn at query time"
                )
            q = np.asarray(embed_fn([query_text]), dtype=np.float32)[0]
            q_norm = np.linalg.norm(q)
            q = q / (q_norm if q_norm > 0 else 1.0)
        scores = (self._vectors @ q).astype(np.float32)
        top_idx = np.argsort(-scores)[:top_k]
        return [RAGHit(document=self._docs[i], score=float(scores[i])) for i in top_idx]

    # ── Save / load ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._vectors is None:
            raise RuntimeError("index empty — nothing to save")
        meta = {
            "doc_ids": np.asarray([d.doc_id for d in self._docs], dtype=object),
            "titles": np.asarray([d.title for d in self._docs], dtype=object),
            "texts": np.asarray([d.text for d in self._docs], dtype=object),
            "vectors": self._vectors,
        }
        np.savez(path, **meta)

    @classmethod
    def load(cls, path: str | Path) -> CosineRAGIndex:
        data = np.load(Path(path), allow_pickle=True)
        idx = cls()
        idx._docs = [
            RAGDocument(doc_id=str(did), title=str(t), text=str(tx))
            for did, t, tx in zip(data["doc_ids"], data["titles"], data["texts"], strict=False)
        ]
        idx._vectors = data["vectors"]
        return idx


# ── Convenience: load a directory of .md files as a procedure corpus ──


def load_procedures_directory(
    root: str | Path,
    chunk_chars: int = 1500,
    overlap_chars: int = 200,
) -> list[RAGDocument]:
    """Walk a directory, read every ``*.md`` file, chunk it into RAGDocuments.

    Conservative chunking: split at chunk_chars with overlap_chars sliding.
    """
    root = Path(root)
    docs: list[RAGDocument] = []
    for md_path in sorted(root.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8", errors="replace")
        title = md_path.stem
        if len(text) <= chunk_chars:
            docs.append(
                RAGDocument(
                    doc_id=f"{md_path.relative_to(root).as_posix()}#0",
                    title=title,
                    text=text,
                    metadata={"path": md_path.relative_to(root).as_posix()},
                )
            )
            continue
        i = 0
        chunk_idx = 0
        while i < len(text):
            chunk = text[i : i + chunk_chars]
            docs.append(
                RAGDocument(
                    doc_id=f"{md_path.relative_to(root).as_posix()}#{chunk_idx}",
                    title=f"{title} (part {chunk_idx + 1})",
                    text=chunk,
                    metadata={"path": md_path.relative_to(root).as_posix()},
                )
            )
            chunk_idx += 1
            if i + chunk_chars >= len(text):
                break
            i += chunk_chars - overlap_chars
    return docs
