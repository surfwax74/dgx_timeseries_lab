"""Graph-augmented retrieval: seed -> expand -> rerank -> pack.

Three retrieval mechanisms, each doing a job the others cannot:

* **Dense** answers *"what is semantically similar to the question?"*
  It handles paraphrase and vocabulary mismatch, and is the reason we
  are not shipping TF-IDF alone.
* **Lexical** answers *"where does this exact string appear?"* Document
  identifiers like ``EPS-201`` are arbitrary tokens that embedding models
  represent badly; a query naming one should hit it deterministically.
  Dense retrieval is *worse* than TF-IDF at this, which is why the fix
  is fusion rather than replacement.
* **Graph** answers *"what else does answering this require?"* This is
  the one that is impossible without stored relationships. In the
  procedures corpus, SYS-000 is neither lexically nor semantically
  similar to "which shed order applies during eclipse" — it is related
  by *logic*. No amount of embedding quality surfaces it. A
  ``resolved_by`` edge surfaces it on the first hop.

That third point is the argument for the whole module. Dense retrieval
scales recall of *similar* text; the graph scales recall of *necessary*
text.

Every hit carries a ``why`` string. Graph expansion that cannot explain
itself is unusable — when it pulls in something irrelevant you need to
see which edge did it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embed import Embedder
from .extract import DOC_ID_RE
from .store import (
    DEFAULT_EDGE_WEIGHTS,
    KnowledgeStore,
    Node,
)


@dataclass
class Hit:
    """One retrieved chunk with its provenance."""

    node: Node
    score: float
    why: str
    dense: float = 0.0
    lexical: float = 0.0
    graph: float = 0.0
    hop: int = 0                       # 0 = seeded directly, 1+ = graph-expanded
    via: list[str] = field(default_factory=list)   # edge path that reached it

    @property
    def doc_id(self) -> str:
        return self.node.doc_id or "?"


@dataclass
class RetrievalConfig:
    """Knobs, with the reasoning behind the defaults.

    ``w_dense`` / ``w_lexical`` / ``w_graph``
        Fusion weights. Dense leads; lexical is a meaningful minority so
        an exact id match cannot be drowned out; graph is scored
        separately so an expanded node can outrank a weakly-similar seed.
    ``seed_k``
        Chunks pulled per mechanism before expansion. Kept modest —
        expansion multiplies this, and a large seed set produces a
        context pack that is mostly noise.
    ``max_hops``
        1 by default. Two hops on a densely-referenced corpus reaches
        most of the graph and stops discriminating. Raise only with
        edge-type filtering.
    ``hop_decay``
        Multiplier applied per hop, so distant nodes rank below near ones.
    ``expand_rels``
        Which edge types to traverse. Defaults exclude ``references``:
        on a real corpus it is the highest-volume, lowest-signal edge and
        is what turns expansion into a hairball. Include it explicitly
        when the corpus is small.
    ``doc_id_boost``
        Additive bonus when the query names a document id outright. Large
        on purpose: if an operator says "EPS-201", they mean it.
    """

    w_dense: float = 1.00
    w_lexical: float = 0.45
    w_graph: float = 0.85
    seed_k: int = 6
    max_hops: int = 1
    hop_decay: float = 0.6
    expand_rels: tuple[str, ...] = (
        "resolved_by", "contradicts", "constrains", "supersedes", "depends_on",
    )
    doc_id_boost: float = 0.75
    max_chunks: int = 12
    max_chars: int = 12000


class GraphAugmentedRetriever:
    """Hybrid retriever over a :class:`KnowledgeStore`."""

    def __init__(
        self,
        store: KnowledgeStore,
        embedder: Embedder | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()

        # Lexical side is built over chunk text at construction. Small
        # corpora only — for very large stores this should move into the
        # store build and be persisted alongside the vectors.
        rows = store.conn.execute(
            "SELECT id, text FROM nodes WHERE kind = 'chunk' ORDER BY vec_row"
        ).fetchall()
        self._chunk_ids = [r["id"] for r in rows]
        texts = [r["text"] for r in rows]
        self._tfidf = None
        self._tfidf_matrix = None
        if texts:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(
                stop_words="english", lowercase=True, ngram_range=(1, 2),
            )
            m = self._tfidf.fit_transform(texts)
            norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
            norms[norms == 0] = 1.0
            self._tfidf_matrix = m.toarray().astype(np.float32) / norms[:, None]

    # ── Seeding ──────────────────────────────────────────────────────

    def _dense_scores(self, query: str) -> dict[str, float]:
        if self.embedder is None or not self.store.has_vectors or self.store.index is None:
            return {}
        encode_q = getattr(self.embedder, "encode_queries", self.embedder.encode)
        qv = np.asarray(encode_q([query]), dtype=np.float32)
        if qv.size == 0:
            return {}
        rows, scores = self.store.index.search(qv[0], self.config.seed_k)
        by_row = self.store.nodes_by_vec_rows(rows.tolist())
        out: dict[str, float] = {}
        for r, s in zip(rows.tolist(), scores.tolist(), strict=True):
            n = by_row.get(int(r))
            if n is not None:
                out[n.id] = float(s)
        return out

    def _lexical_scores(self, query: str) -> dict[str, float]:
        if self._tfidf is None or self._tfidf_matrix is None:
            return {}
        q = self._tfidf.transform([query]).toarray().astype(np.float32)[0]
        qn = np.linalg.norm(q)
        if qn == 0:
            return {}
        sims = self._tfidf_matrix @ (q / qn)
        k = min(self.config.seed_k, sims.shape[0])
        idx = np.argpartition(-sims, k - 1)[:k] if k > 0 else np.empty(0, int)
        return {
            self._chunk_ids[int(i)]: float(sims[int(i)])
            for i in idx
            if sims[int(i)] > 0
        }

    def _named_doc_ids(self, query: str) -> set[str]:
        """Document ids the query mentions outright, restricted to real ones."""
        return {d for d in DOC_ID_RE.findall(query)} & self.store.all_doc_ids()

    # ── Expansion ────────────────────────────────────────────────────

    def _expand(
        self, seeds: dict[str, float]
    ) -> dict[str, tuple[float, int, list[str]]]:
        """Walk typed edges out from seed chunks.

        The walk is chunk -> document -> (typed edge) -> document ->
        chunks, because relationships live at document granularity. It
        returns {chunk_id: (graph_score, hop, edge_path)}.
        """
        cfg = self.config
        found: dict[str, tuple[float, int, list[str]]] = {}
        if not seeds or cfg.max_hops <= 0:
            return found

        # Seed documents, carrying the best score of any of their chunks.
        frontier: dict[str, tuple[float, list[str]]] = {}
        for chunk_id, s in seeds.items():
            n = self.store.get_node(chunk_id)
            if n is None or not n.doc_id:
                continue
            did = f"doc:{n.doc_id}"
            if did not in frontier or s > frontier[did][0]:
                frontier[did] = (s, [])

        visited_docs = set(frontier)
        for hop in range(1, cfg.max_hops + 1):
            next_frontier: dict[str, tuple[float, list[str]]] = {}
            for doc_node_id, (base, path) in frontier.items():
                for edge, other in self.store.neighbors(
                    doc_node_id, rels=cfg.expand_rels, incoming=True
                ):
                    if other.kind != "document" or other.id in visited_docs:
                        continue
                    rel_w = edge.weight or DEFAULT_EDGE_WEIGHTS.get(edge.rel, 0.5)
                    score = base * rel_w * (cfg.hop_decay ** (hop - 1))
                    step = f"{doc_node_id.removeprefix('doc:')} --{edge.rel}--> {other.doc_id}"
                    new_path = [*path, step]
                    prev = next_frontier.get(other.id)
                    if prev is None or score > prev[0]:
                        next_frontier[other.id] = (score, new_path)

            for doc_node_id, (score, path) in next_frontier.items():
                visited_docs.add(doc_node_id)
                doc_node = self.store.get_node(doc_node_id)
                if doc_node is None or not doc_node.doc_id:
                    continue
                for chunk in self.store.nodes_for_doc(doc_node.doc_id):
                    if chunk.kind != "chunk":
                        continue
                    prev = found.get(chunk.id)
                    if prev is None or score > prev[0]:
                        found[chunk.id] = (score, hop, path)
            frontier = next_frontier
            if not frontier:
                break
        return found

    # ── Public API ───────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        """Run the full pipeline and return ranked, explained hits."""
        cfg = self.config
        top_k = top_k or cfg.max_chunks

        dense = self._dense_scores(query)
        lexical = self._lexical_scores(query)
        named = self._named_doc_ids(query)

        seeds: dict[str, float] = {}
        for cid, s in dense.items():
            seeds[cid] = max(seeds.get(cid, 0.0), s)
        for cid, s in lexical.items():
            seeds[cid] = max(seeds.get(cid, 0.0), s)

        expanded = self._expand(seeds)

        combined: dict[str, Hit] = {}
        all_ids = set(seeds) | set(expanded)
        for cid in all_ids:
            node = self.store.get_node(cid)
            if node is None:
                continue
            d = dense.get(cid, 0.0)
            lx = lexical.get(cid, 0.0)
            g, hop, path = expanded.get(cid, (0.0, 0, []))

            score = cfg.w_dense * d + cfg.w_lexical * lx + cfg.w_graph * g
            reasons: list[str] = []
            if d > 0:
                reasons.append(f"dense {d:.2f}")
            if lx > 0:
                reasons.append(f"lexical {lx:.2f}")
            if g > 0:
                reasons.append(f"graph {g:.2f} via {' | '.join(path)}" if path else f"graph {g:.2f}")
            if node.doc_id and node.doc_id in named:
                score += cfg.doc_id_boost
                reasons.append(f"query names {node.doc_id}")

            combined[cid] = Hit(
                node=node, score=score, why="; ".join(reasons) or "no signal",
                dense=d, lexical=lx, graph=g, hop=hop, via=path,
            )

        ranked = sorted(combined.values(), key=lambda h: -h.score)
        return ranked[:top_k]

    def context_pack(
        self, query: str, top_k: int | None = None, max_chars: int | None = None
    ) -> tuple[str, list[Hit]]:
        """Assemble a prompt-ready context block plus the hits behind it.

        Chunks are emitted with their document id as a header so the
        model can cite accurately — a model that cannot see which
        document a passage came from will either omit citations or
        invent them.
        """
        cfg = self.config
        max_chars = max_chars or cfg.max_chars
        hits = self.retrieve(query, top_k=top_k)

        parts: list[str] = []
        used: list[Hit] = []
        total = 0
        for h in hits:
            block = f"[{h.doc_id}] {h.node.title}\n{h.node.text}"
            if total + len(block) > max_chars and used:
                break
            parts.append(block)
            used.append(h)
            total += len(block)
        return "\n\n---\n\n".join(parts), used

    def explain(self, query: str, top_k: int | None = None) -> str:
        """Human-readable trace. For debugging a retrieval that went wrong."""
        hits = self.retrieve(query, top_k=top_k)
        lines = [f"query: {query!r}", f"{len(hits)} hit(s)", ""]
        for i, h in enumerate(hits, 1):
            tag = "seed" if h.hop == 0 else f"hop {h.hop}"
            lines.append(f"{i:2d}. [{h.doc_id}] {h.node.title}")
            lines.append(f"    score {h.score:.3f}  ({tag})  {h.why}")
        return "\n".join(lines)
