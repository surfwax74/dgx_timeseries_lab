"""Build a KnowledgeStore from a corpus of procedure markdown.

Two granularities, deliberately:

* **Document nodes** are graph anchors. "EPS-201 contradicts SYS-001" is
  a fact about documents, not about paragraph 3 of one of them.
* **Chunk nodes** carry the vectors, because that is the granularity you
  want to put in a context window.

Traversal bridges them: a retrieved chunk walks ``part_of`` up to its
document, follows a typed edge across to a related document, then walks
back down to that document's chunks. Modelling edges at chunk level
instead would multiply every document relationship by the chunk count
and turn the graph into a hairball.

Two sources of edges:

* **Extracted** — mechanically derivable and safe to trust:
  ``references`` from ``ABC-123`` mentions, ``supersedes`` from a
  "Supersedes:" line, ``part_of`` from chunking.
* **Declared** — requires human judgment, so it lives in a reviewed YAML
  sidecar: ``contradicts``, ``resolved_by``, ``constrains``,
  ``depends_on``.

Provenance is recorded on every edge so an auditor can tell which is
which, and so a bad auto-extraction rule can be found and undone later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .embed import Embedder, TfidfSvdEmbedder
from .store import (
    DEFAULT_EDGE_WEIGHTS,
    REL_PART_OF,
    REL_REFERENCES,
    REL_SUPERSEDES,
    Edge,
    KnowledgeStore,
    Node,
)

#: Canonical document identifier, e.g. EPS-201. Doubles as the filename
#: prefix convention so citations can be validated against real files.
DOC_ID_RE = re.compile(r"\b([A-Z]{2,4}-\d{2,4})\b")
_FILENAME_ID_RE = re.compile(r"^([A-Z]{2,4}-\d{2,4})")
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_SUPERSEDES_RE = re.compile(r"^\s*\*{0,2}Supersedes:?\*{0,2}\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


@dataclass
class ParsedDoc:
    """One source document before chunking."""

    doc_id: str
    title: str
    text: str
    path: str
    references: set[str]
    supersedes: set[str]


def parse_document(path: Path, root: Path) -> ParsedDoc | None:
    """Parse one markdown file. Returns None if it has no ``ABC-123`` id.

    The id requirement is intentional: without a stable identifier a
    document cannot be cited, and a citation to it cannot be validated
    against the corpus. Files that do not carry one are skipped loudly
    by the caller rather than silently indexed.
    """
    m = _FILENAME_ID_RE.match(path.name)
    if not m:
        return None
    doc_id = m.group(1)
    text = path.read_text(encoding="utf-8", errors="replace")

    title_m = _TITLE_RE.search(text)
    title = title_m.group(1).strip() if title_m else path.stem

    refs = {d for d in DOC_ID_RE.findall(text) if d != doc_id}

    supersedes: set[str] = set()
    for sm in _SUPERSEDES_RE.finditer(text):
        supersedes.update(d for d in DOC_ID_RE.findall(sm.group(1)) if d != doc_id)
    # A "Supersedes: X" line also reads as a reference; keep the stronger
    # relation only, so the retriever does not double-count the edge.
    refs -= supersedes

    return ParsedDoc(
        doc_id=doc_id, title=title, text=text,
        path=path.relative_to(root).as_posix(),
        references=refs, supersedes=supersedes,
    )


def chunk_text(text: str, chunk_chars: int = 1400, overlap_chars: int = 200) -> list[str]:
    """Split on paragraph boundaries, packing up to ``chunk_chars``.

    Paragraph-aware rather than a fixed character window: procedure
    documents are dense with tables and numbered steps, and slicing
    mid-table produces chunks that retrieve well and read as nonsense.
    Overlap is applied as trailing context carried into the next chunk.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for p in paras:
        # A single oversized paragraph (usually a wide table) becomes its
        # own chunk rather than being split apart.
        if len(p) > chunk_chars:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, size = [], 0
            chunks.append(p)
            continue
        if size + len(p) > chunk_chars and cur:
            chunks.append("\n\n".join(cur))
            tail = chunks[-1][-overlap_chars:] if overlap_chars else ""
            cur = [tail] if tail else []
            size = len(tail)
        cur.append(p)
        size += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def load_declared_edges(path: str | Path | None) -> list[Edge]:
    """Load hand-declared semantic edges from YAML.

    Expected shape::

        edges:
          - src: EPS-201
            dst: SYS-001
            rel: contradicts
            note: "Different shed order; see SYS-000 rule 3."
          - src: EPS-201
            dst: SYS-000
            rel: resolved_by

    ``src``/``dst`` are document ids. Weight defaults to the relation's
    entry in DEFAULT_EDGE_WEIGHTS.
    """
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: list[Edge] = []
    for raw in doc.get("edges", []):
        rel = str(raw["rel"])
        out.append(
            Edge(
                src=f"doc:{raw['src']}",
                dst=f"doc:{raw['dst']}",
                rel=rel,
                weight=float(raw.get("weight", DEFAULT_EDGE_WEIGHTS.get(rel, 0.5))),
                provenance="declared",
                note=str(raw.get("note", "")),
            )
        )
    return out


def build_knowledge_store(
    procedures_dir: str | Path,
    output_dir: str | Path,
    embedder: Embedder | None = None,
    declared_edges_path: str | Path | None = None,
    chunk_chars: int = 1400,
    overlap_chars: int = 200,
    verbose: bool = True,
) -> KnowledgeStore:
    """Parse a corpus, extract edges, embed chunks, and write the store."""
    root = Path(procedures_dir)
    if not root.exists():
        raise FileNotFoundError(f"procedures directory not found: {root}")

    parsed: list[ParsedDoc] = []
    skipped: list[str] = []
    for md in sorted(root.rglob("*.md")):
        if md.name.upper().startswith("README"):
            continue
        doc = parse_document(md, root)
        if doc is None:
            skipped.append(md.name)
            continue
        parsed.append(doc)

    if skipped and verbose:
        print(
            f"[knowledge] skipped {len(skipped)} file(s) with no ABC-123 id prefix: "
            f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}"
        )
    if not parsed:
        raise ValueError(
            f"no documents with an ABC-123 filename prefix under {root}. "
            "Rename e.g. 'EPS-201_undervoltage_response.md'."
        )

    nodes: list[Node] = []
    edges: list[Edge] = []
    chunk_texts: list[str] = []

    # Document anchors.
    for d in parsed:
        nodes.append(
            Node(
                id=f"doc:{d.doc_id}", kind="document", doc_id=d.doc_id,
                title=d.title, text="", source_path=d.path,
                metadata={"n_chars": str(len(d.text))},
            )
        )

    known_docs = {d.doc_id for d in parsed}

    # Extracted document-level edges.
    for d in parsed:
        for ref in sorted(d.references):
            if ref in known_docs:
                edges.append(Edge(
                    src=f"doc:{d.doc_id}", dst=f"doc:{ref}", rel=REL_REFERENCES,
                    weight=DEFAULT_EDGE_WEIGHTS[REL_REFERENCES], provenance="extracted",
                ))
        for sup in sorted(d.supersedes):
            if sup in known_docs:
                edges.append(Edge(
                    src=f"doc:{d.doc_id}", dst=f"doc:{sup}", rel=REL_SUPERSEDES,
                    weight=DEFAULT_EDGE_WEIGHTS[REL_SUPERSEDES], provenance="extracted",
                ))

    # Chunks carry the vectors.
    vec_row = 0
    for d in parsed:
        for i, ct in enumerate(chunk_text(d.text, chunk_chars, overlap_chars)):
            cid = f"chunk:{d.doc_id}#{i}"
            nodes.append(Node(
                id=cid, kind="chunk", doc_id=d.doc_id,
                title=f"{d.title} (part {i + 1})", text=ct,
                source_path=d.path, chunk_index=i, vec_row=vec_row,
            ))
            edges.append(Edge(
                src=cid, dst=f"doc:{d.doc_id}", rel=REL_PART_OF,
                weight=DEFAULT_EDGE_WEIGHTS[REL_PART_OF], provenance="extracted",
            ))
            chunk_texts.append(ct)
            vec_row += 1

    # Declared semantic edges, filtered to documents we actually indexed.
    declared = load_declared_edges(declared_edges_path)
    node_ids = {n.id for n in nodes}
    dropped = [e for e in declared if e.src not in node_ids or e.dst not in node_ids]
    if dropped and verbose:
        print(
            f"[knowledge] {len(dropped)} declared edge(s) reference unknown documents "
            f"and were dropped: {[(e.src, e.rel, e.dst) for e in dropped[:4]]}"
        )
    edges.extend(e for e in declared if e.src in node_ids and e.dst in node_ids)

    # Embed.
    if embedder is None:
        embedder = TfidfSvdEmbedder(dim=256)
    if isinstance(embedder, TfidfSvdEmbedder) and not embedder.is_fitted:
        embedder.fit(chunk_texts)
    vectors: np.ndarray = embedder.encode(chunk_texts) if chunk_texts else np.zeros((0, 1), np.float32)

    store = KnowledgeStore.build(
        root=output_dir, nodes=nodes, edges=edges, vectors=vectors,
        meta={
            "embedder": getattr(embedder, "name", "unknown"),
            "embed_dim": str(int(vectors.shape[1]) if vectors.size else 0),
            "n_documents": str(len(parsed)),
            "n_chunks": str(len(chunk_texts)),
            "source": str(root),
            "chunk_chars": str(chunk_chars),
        },
    )

    if verbose:
        n_nodes, n_edges = store.count()
        by_rel: dict[str, int] = {}
        for e in edges:
            by_rel[e.rel] = by_rel.get(e.rel, 0) + 1
        print(
            f"[knowledge] built {output_dir}: {len(parsed)} docs, "
            f"{len(chunk_texts)} chunks, {n_nodes} nodes, {n_edges} edges"
        )
        for rel in sorted(by_rel, key=lambda r: -by_rel[r]):
            print(f"             {by_rel[rel]:4d}  {rel}")
    return store


def audit_graph(store: KnowledgeStore) -> dict[str, list[str]]:
    """Find structural problems a human should look at.

    Cheap checks that catch the failure modes that actually happen:
    documents nothing points at (likely a missing declared edge), and
    contradictions with no resolving authority (the dangerous one — a
    known conflict that retrieval can surface but nothing settles).
    """
    from .store import REL_CONTRADICTS, REL_RESOLVED_BY

    findings: dict[str, list[str]] = {"orphans": [], "unresolved_contradictions": []}

    doc_rows = store.conn.execute(
        "SELECT id FROM nodes WHERE kind = 'document'"
    ).fetchall()
    for r in doc_rows:
        did = r["id"]
        inbound = store.conn.execute(
            "SELECT COUNT(*) AS c FROM edges WHERE dst = ? AND rel != ?",
            (did, REL_PART_OF),
        ).fetchone()["c"]
        if inbound == 0:
            findings["orphans"].append(did)

    contradicts = store.conn.execute(
        "SELECT src, dst FROM edges WHERE rel = ?", (REL_CONTRADICTS,)
    ).fetchall()
    for r in contradicts:
        resolved = store.conn.execute(
            "SELECT COUNT(*) AS c FROM edges WHERE rel = ? AND src IN (?, ?)",
            (REL_RESOLVED_BY, r["src"], r["dst"]),
        ).fetchone()["c"]
        if resolved == 0:
            findings["unresolved_contradictions"].append(f"{r['src']} <-> {r['dst']}")

    return findings
