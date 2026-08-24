"""KnowledgeStore — one store, two indexes, one identity.

The common GraphRAG mistake is running a vector database and a graph
database side by side. They drift: a chunk's id in the vector store
stops matching its node id in the graph after a partial rebuild, and
retrieval starts silently returning the wrong text for the right node.

So this is a single store:

    * SQLite (``graph.db``) holds nodes and typed edges. Durable,
      inspectable with any SQL client, no server, one file to sneakernet.
    * A numpy matrix (``vectors.npy``) holds embeddings, with **row i
      pinned to node.vec_row = i**. There is exactly one identity — the
      node id — and the vector is an attribute of it, not a parallel
      record.

Rebuilds are atomic: write to a temp directory, then swap. A half-built
index can never be loaded.

Scale notes
-----------
Exact cosine over the full matrix is used deliberately. At 100k chunks x
768 dims that is a ~300 MB float32 matmul, roughly 50 ms — far below the
latency of the LLM call it feeds. ANN indexing (FAISS/HNSW) is a real
need above ~1M chunks; ``VectorIndex`` exists as a seam for that, but
taking the dependency before measuring would be premature.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- A node is a retrievable unit: a whole document, or one chunk of one.
-- `vec_row` is the row index into vectors.npy, or NULL if unembedded.
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,           -- 'document' | 'chunk'
    doc_id      TEXT,                    -- canonical ABC-123, NULL if unparseable
    title       TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER,
    vec_row     INTEGER UNIQUE,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_nodes_doc  ON nodes(doc_id);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);

-- A typed, weighted, directed edge. `rel` carries the logical meaning;
-- `weight` lets the retriever prefer strong relationships during
-- expansion. `provenance` records how the edge was created so a human
-- can audit an auto-extracted edge differently from a declared one.
CREATE TABLE IF NOT EXISTS edges (
    src        TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst        TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    rel        TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    provenance TEXT NOT NULL DEFAULT 'declared',   -- declared | extracted | inferred
    note       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (src, dst, rel)
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(rel);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ── Relationship vocabulary ───────────────────────────────────────────
#
# Edge semantics matter more than edge count. Each type answers a
# different retrieval question, and the retriever weights them
# differently during expansion.

REL_PART_OF = "part_of"          # chunk -> its document
REL_REFERENCES = "references"    # A mentions B (mechanically extracted)
REL_SUPERSEDES = "supersedes"    # A replaces B (revision chain)
REL_CONTRADICTS = "contradicts"  # A and B give conflicting guidance
REL_RESOLVED_BY = "resolved_by"  # a conflict is settled by authority doc B
REL_CONSTRAINS = "constrains"    # B limits an action described in A
REL_DEPENDS_ON = "depends_on"    # A's procedure requires B's data

#: Default expansion weights. `resolved_by` and `contradicts` are the
#: highest because they surface exactly what a query cannot ask for by
#: name — the authority you did not know you needed, and the conflict
#: you did not know existed. `references` is deliberately low: on a
#: large corpus it is the edge type most prone to hairballing.
DEFAULT_EDGE_WEIGHTS: dict[str, float] = {
    REL_RESOLVED_BY: 1.00,
    REL_CONTRADICTS: 0.95,
    REL_CONSTRAINS: 0.80,
    REL_SUPERSEDES: 0.75,
    REL_DEPENDS_ON: 0.60,
    REL_PART_OF: 0.50,
    REL_REFERENCES: 0.35,
}


@dataclass
class Node:
    """One retrievable unit."""

    id: str
    kind: str                       # 'document' | 'chunk'
    text: str = ""
    title: str = ""
    doc_id: str | None = None       # canonical ABC-123
    source_path: str = ""
    chunk_index: int | None = None
    vec_row: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A typed, weighted, directed relationship."""

    src: str
    dst: str
    rel: str
    weight: float = 1.0
    provenance: str = "declared"    # declared | extracted | inferred
    note: str = ""


class VectorIndex(Protocol):
    """Seam for swapping exact search for ANN once corpus size demands it."""

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (row_indices, scores) for the top_k nearest rows."""
        ...


class ExactCosineIndex:
    """Exact cosine over a row-normalized matrix.

    Deliberately simple. Correct by construction, no index to rebuild or
    tune, and fast enough that the LLM call downstream dominates latency
    for any corpus we realistically hold on one box.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"expected 2-D matrix, got shape {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._v = (vectors / norms).astype(np.float32)

    @property
    def n_vectors(self) -> int:
        return int(self._v.shape[0])

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        qn = np.linalg.norm(q)
        q = q / (qn if qn > 0 else 1.0)
        scores = self._v @ q
        k = min(int(top_k), scores.shape[0])
        if k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        # argpartition then sort the small slice — avoids a full sort.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx.astype(np.int64), scores[idx].astype(np.float32)


class KnowledgeStore:
    """SQLite graph + numpy vectors, sharing one node identity.

    Open an existing store with :meth:`open`, or create one with
    :meth:`build`. The build path writes to a temp directory and swaps,
    so a crashed rebuild cannot leave a store that half-loads.
    """

    VECTORS_FILE = "vectors.npy"
    GRAPH_FILE = "graph.db"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._conn: sqlite3.Connection | None = None
        self._vectors: np.ndarray | None = None
        self._index: VectorIndex | None = None

    # ── Lifecycle ────────────────────────────────────────────────────

    @classmethod
    def open(cls, root: str | Path) -> KnowledgeStore:
        store = cls(root)
        db = store.root / cls.GRAPH_FILE
        if not db.exists():
            raise FileNotFoundError(
                f"no knowledge store at {store.root} (expected {cls.GRAPH_FILE}). "
                f"Build one with `dgx-ts kb build` or KnowledgeStore.build(...)."
            )
        store._conn = sqlite3.connect(db)
        store._conn.row_factory = sqlite3.Row
        vec_path = store.root / cls.VECTORS_FILE
        if vec_path.exists():
            store._vectors = np.load(vec_path)
            store._index = ExactCosineIndex(store._vectors)
        return store

    @classmethod
    def build(
        cls,
        root: str | Path,
        nodes: Iterable[Node],
        edges: Iterable[Edge],
        vectors: np.ndarray | None = None,
        meta: dict[str, str] | None = None,
    ) -> KnowledgeStore:
        """Create a store atomically.

        `vectors` rows must correspond to ``node.vec_row``. Nodes with
        ``vec_row is None`` are stored but unsearchable by similarity —
        legitimate for document-level nodes that exist only as graph
        anchors.
        """
        root = Path(root)
        staging = root.parent / f".{root.name}.building"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(staging / cls.GRAPH_FILE)
        try:
            conn.executescript(_SCHEMA)
            node_list = list(nodes)
            conn.executemany(
                "INSERT OR REPLACE INTO nodes "
                "(id, kind, doc_id, title, text, source_path, chunk_index, vec_row, metadata) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        n.id, n.kind, n.doc_id, n.title, n.text, n.source_path,
                        n.chunk_index, n.vec_row, json.dumps(n.metadata),
                    )
                    for n in node_list
                ],
            )

            known = {n.id for n in node_list}
            edge_list = [e for e in edges if e.src in known and e.dst in known]
            conn.executemany(
                "INSERT OR REPLACE INTO edges (src, dst, rel, weight, provenance, note) "
                "VALUES (?,?,?,?,?,?)",
                [(e.src, e.dst, e.rel, e.weight, e.provenance, e.note) for e in edge_list],
            )

            payload = dict(meta or {})
            payload.setdefault("n_nodes", str(len(node_list)))
            payload.setdefault("n_edges", str(len(edge_list)))
            conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                list(payload.items()),
            )
            conn.commit()
        finally:
            conn.close()

        if vectors is not None:
            np.save(staging / cls.VECTORS_FILE, np.asarray(vectors, dtype=np.float32))

        # Atomic-ish swap. Windows cannot rename onto an existing dir.
        if root.exists():
            shutil.rmtree(root)
        staging.rename(root)
        return cls.open(root)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("store is not open")
        return self._conn

    @property
    def index(self) -> VectorIndex | None:
        return self._index

    @property
    def has_vectors(self) -> bool:
        return self._vectors is not None

    def meta(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def count(self) -> tuple[int, int]:
        """(n_nodes, n_edges)."""
        n = self.conn.execute("SELECT COUNT(*) AS c FROM nodes").fetchone()["c"]
        e = self.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]
        return int(n), int(e)

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Node:
        return Node(
            id=row["id"], kind=row["kind"], text=row["text"], title=row["title"],
            doc_id=row["doc_id"], source_path=row["source_path"],
            chunk_index=row["chunk_index"], vec_row=row["vec_row"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get_node(self, node_id: str) -> Node | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return self._row_to_node(row) if row else None

    def nodes_by_vec_rows(self, rows: Iterable[int]) -> dict[int, Node]:
        rows = list(rows)
        if not rows:
            return {}
        marks = ",".join("?" * len(rows))
        found = self.conn.execute(
            f"SELECT * FROM nodes WHERE vec_row IN ({marks})", [int(r) for r in rows]
        ).fetchall()
        return {int(r["vec_row"]): self._row_to_node(r) for r in found}

    def nodes_for_doc(self, doc_id: str) -> list[Node]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def all_doc_ids(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT doc_id FROM nodes WHERE doc_id IS NOT NULL"
        ).fetchall()
        return {r["doc_id"] for r in rows}

    # ── Graph traversal ──────────────────────────────────────────────

    def neighbors(
        self,
        node_id: str,
        rels: Iterable[str] | None = None,
        incoming: bool = True,
    ) -> list[tuple[Edge, Node]]:
        """Adjacent (edge, node) pairs.

        Traverses outgoing edges always, and incoming edges by default.
        Incoming matters: if EPS-201 ``contradicts`` SYS-001, arriving at
        SYS-001 should still surface EPS-201. A directed-only walk would
        miss half the conflicts.
        """
        rel_filter, params = "", [node_id]
        if rels:
            rels = list(rels)
            rel_filter = f" AND rel IN ({','.join('?' * len(rels))})"
            params += list(rels)

        sql = (
            "SELECT src, dst, rel, weight, provenance, note, 'out' AS dir "
            f"FROM edges WHERE src = ?{rel_filter}"
        )
        if incoming:
            sql += (
                " UNION ALL SELECT src, dst, rel, weight, provenance, note, 'in' AS dir "
                f"FROM edges WHERE dst = ?{rel_filter}"
            )
            params = params + [node_id] + (list(rels) if rels else [])

        out: list[tuple[Edge, Node]] = []
        for r in self.conn.execute(sql, params).fetchall():
            other_id = r["dst"] if r["dir"] == "out" else r["src"]
            other = self.get_node(other_id)
            if other is None:
                continue
            out.append((
                Edge(src=r["src"], dst=r["dst"], rel=r["rel"], weight=float(r["weight"]),
                     provenance=r["provenance"], note=r["note"]),
                other,
            ))
        return out

    def to_networkx(self):
        """Materialize as a networkx MultiDiGraph for analysis.

        Not used on the retrieval hot path — SQLite traversal is cheaper
        for the shallow, budgeted walks the retriever does. This is for
        offline work: centrality, community detection, finding orphaned
        documents nothing references.
        """
        import networkx as nx

        g = nx.MultiDiGraph()
        for r in self.conn.execute("SELECT * FROM nodes").fetchall():
            n = self._row_to_node(r)
            g.add_node(n.id, kind=n.kind, doc_id=n.doc_id, title=n.title,
                       chunk_index=n.chunk_index)
        for r in self.conn.execute("SELECT * FROM edges").fetchall():
            g.add_edge(r["src"], r["dst"], key=r["rel"], rel=r["rel"],
                       weight=float(r["weight"]), provenance=r["provenance"])
        return g
