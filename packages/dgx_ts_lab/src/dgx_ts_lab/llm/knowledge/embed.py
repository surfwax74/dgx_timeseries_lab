"""Embedders — pluggable text -> vector, air-gap first.

Three implementations, in descending order of quality:

``TransformersEmbedder``
    A real sentence-embedding model loaded from a **local directory** via
    ``transformers``. Deliberately not ``sentence-transformers``: that
    package is one more dependency to justify at security review, and it
    buys us only mean-pooling, which is nine lines. You sneakernet the
    HF model directory either way.

``TfidfSvdEmbedder``
    LSA — TF-IDF followed by truncated SVD. No model file needed, so it
    works on any box out of the box. It is a genuine fallback rather than
    a stub: SVD recovers some synonymy that raw TF-IDF misses. It is
    still weaker than a trained encoder and should not be the production
    choice when a model is available.

``HashingEmbedder``
    Deterministic, dependency-free, semantically meaningless. Exists so
    tests and CI can exercise the full pipeline without a model on disk.
    Never use it for real retrieval.

All three normalize output rows to unit length, so downstream cosine is
a plain dot product.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Text -> (N, D) float32 matrix with unit-norm rows."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class TransformersEmbedder:
    """Local HF encoder with mean pooling.

    Parameters
    ----------
    model_path
        A **local directory** containing an HF model. Passing a hub id
        works on a connected box but will fail on the air-gapped DGX, so
        prefer a path under ``data/llm_weights/``.
    batch_size
        Texts per forward pass. 32 is safe on CPU; raise substantially
        on a GPU.
    max_length
        Token truncation. Most sentence encoders are trained at 512 and
        degrade past it.
    query_prefix / passage_prefix
        Several strong encoders (E5, BGE) are trained with asymmetric
        instruction prefixes and lose real accuracy without them —
        e.g. ``query: `` and ``passage: `` for E5. Check your model card;
        leaving these empty is correct for MiniLM-style models.
    """

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_path = str(model_path)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        local_only = Path(model_path).exists()
        self._tok = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=local_only
        )
        self._model = AutoModel.from_pretrained(
            self.model_path, local_files_only=local_only
        )
        self._model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(device)

        self.name = f"transformers:{Path(self.model_path).name}"
        self.dim = int(self._model.config.hidden_size)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        prepared = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(prepared), self.batch_size):
                batch = prepared[i : i + self.batch_size]
                enc = self._tok(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**enc).last_hidden_state       # (B, T, D)
                # Mean-pool over real tokens only; padding must not dilute.
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                out.append((summed / counts).cpu().numpy())
        return _l2_normalize(np.vstack(out))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode corpus passages."""
        return self._encode(texts, self.passage_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode queries, applying the query-side prefix if configured."""
        return self._encode(texts, self.query_prefix)


class TfidfSvdEmbedder:
    """LSA fallback: TF-IDF then truncated SVD. Requires fitting.

    Weaker than a trained encoder, but real: SVD collapses correlated
    terms, so it recovers some synonymy that raw TF-IDF cannot see. Use
    when no encoder has been staged onto the box yet.
    """

    def __init__(self, dim: int = 256, random_state: int = 0) -> None:
        self.dim = int(dim)
        self.random_state = int(random_state)
        self.name = f"tfidf-svd:{dim}"
        self._vec = None
        self._svd = None

    @property
    def is_fitted(self) -> bool:
        return self._svd is not None

    def fit(self, texts: Sequence[str]) -> TfidfSvdEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not texts:
            raise ValueError("cannot fit TfidfSvdEmbedder on an empty corpus")
        self._vec = TfidfVectorizer(
            stop_words="english", lowercase=True, ngram_range=(1, 2),
        )
        mat = self._vec.fit_transform(texts)
        # SVD components cannot exceed min(n_samples, n_features) - 1.
        n_comp = max(1, min(self.dim, min(mat.shape) - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self._svd.fit(mat)
        self.dim = n_comp
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._vec is None or self._svd is None:
            raise RuntimeError("TfidfSvdEmbedder.fit() must be called before encode()")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(self._svd.transform(self._vec.transform(list(texts))))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)


class HashingEmbedder:
    """Deterministic token-hash embedder. **Tests only.**

    Produces stable vectors with no semantic content — a document and its
    paraphrase land nowhere near each other. It exists so the pipeline
    can be exercised end to end in CI without staging a model.
    """

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self.name = f"hashing:{dim}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                h = hashlib.blake2b(
                    tok.encode("utf-8"), digest_size=8,
                    key=str(self.seed).encode("utf-8"),
                ).digest()
                out[i, int.from_bytes(h, "little") % self.dim] += 1.0
        return _l2_normalize(out)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)


def build_embedder(
    kind: str = "auto",
    model_path: str | Path | None = None,
    dim: int = 256,
    **kwargs,
) -> Embedder:
    """Construct an embedder by name.

    ``auto`` prefers a real encoder when ``model_path`` points at
    something on disk, and degrades to LSA otherwise — with a printed
    warning, because silently retrieving at lower quality is exactly the
    kind of thing that produces a confusing eval result three weeks later.
    """
    kind = kind.lower()
    if kind == "auto":
        if model_path and Path(model_path).exists():
            return TransformersEmbedder(model_path, **kwargs)
        print(
            "[knowledge] no local encoder at "
            f"{model_path!r}; falling back to TF-IDF+SVD. Retrieval quality "
            "will be lower — stage an encoder and pass --embedder transformers.",
        )
        return TfidfSvdEmbedder(dim=dim)
    if kind in ("transformers", "hf"):
        if not model_path:
            raise ValueError("embedder 'transformers' requires model_path")
        return TransformersEmbedder(model_path, **kwargs)
    if kind in ("tfidf_svd", "lsa", "tfidf"):
        return TfidfSvdEmbedder(dim=dim)
    if kind == "hashing":
        return HashingEmbedder(dim=dim)
    raise ValueError(
        f"unknown embedder {kind!r}; expected auto | transformers | tfidf_svd | hashing"
    )
