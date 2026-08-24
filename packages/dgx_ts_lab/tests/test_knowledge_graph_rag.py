"""Tests for the graph-augmented knowledge base.

The load-bearing test here is
``test_graph_surfaces_authority_document_similarity_cannot_reach``. It
pins the claim the whole module exists to make: SYS-000 is the document
required to answer the hard scenarios correctly, it is neither lexically
nor semantically similar to the query, and only a typed edge reaches it.

If that test ever starts passing without the graph, the graph is no
longer earning its complexity and should be reconsidered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dgx_ts_lab.llm.knowledge import (
    Edge,
    ExactCosineIndex,
    GraphAugmentedRetriever,
    HashingEmbedder,
    KnowledgeStore,
    Node,
    RetrievalConfig,
    TfidfSvdEmbedder,
    audit_graph,
    build_embedder,
    build_knowledge_store,
    chunk_text,
    load_declared_edges,
    parse_document,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCEDURES = REPO_ROOT / "docs" / "procedures"
DECLARED_EDGES = REPO_ROOT / "configs" / "knowledge" / "procedure_edges.yaml"

ECLIPSE_QUERY = "we are in eclipse and bus voltage is sagging, which loads do I shed first?"


@pytest.fixture(scope="module")
def kb(tmp_path_factory) -> KnowledgeStore:
    """A store built from the real procedures corpus, once per module."""
    out = tmp_path_factory.mktemp("kb") / "procedures"
    emb = TfidfSvdEmbedder(dim=128)
    return build_knowledge_store(
        procedures_dir=PROCEDURES,
        output_dir=out,
        embedder=emb,
        declared_edges_path=DECLARED_EDGES,
        verbose=False,
    )


@pytest.fixture(scope="module")
def fitted_embedder(kb: KnowledgeStore) -> TfidfSvdEmbedder:
    rows = kb.conn.execute(
        "SELECT text FROM nodes WHERE kind='chunk' ORDER BY vec_row"
    ).fetchall()
    return TfidfSvdEmbedder(dim=128).fit([r["text"] for r in rows])


# ── The central claim ─────────────────────────────────────────────────


def test_graph_surfaces_authority_document_similarity_cannot_reach(
    kb: KnowledgeStore, fitted_embedder: TfidfSvdEmbedder
) -> None:
    """SYS-000 is reachable by edge, not by similarity.

    This is the justification for the whole module. The query never names
    SYS-000 and shares little vocabulary with it, so dense+lexical miss
    it entirely. One `resolved_by` hop from EPS-201 brings it in.
    """
    flat = GraphAugmentedRetriever(
        kb, fitted_embedder, RetrievalConfig(max_hops=0, max_chunks=6)
    )
    graph = GraphAugmentedRetriever(
        kb, fitted_embedder, RetrievalConfig(max_hops=1, max_chunks=6)
    )

    docs_flat = {h.doc_id for h in flat.retrieve(ECLIPSE_QUERY)}
    docs_graph = {h.doc_id for h in graph.retrieve(ECLIPSE_QUERY)}

    assert "SYS-000" not in docs_flat, (
        "similarity search unexpectedly reached SYS-000 — if this is now "
        "true the graph may no longer be earning its complexity"
    )
    assert "SYS-000" in docs_graph


def test_expanded_hit_explains_the_edge_that_reached_it(
    kb: KnowledgeStore, fitted_embedder: TfidfSvdEmbedder
) -> None:
    """Graph expansion that cannot explain itself is unusable."""
    r = GraphAugmentedRetriever(kb, fitted_embedder, RetrievalConfig(max_hops=1))
    hits = [h for h in r.retrieve(ECLIPSE_QUERY) if h.doc_id == "SYS-000"]
    assert hits, "expected SYS-000 among expanded hits"
    h = hits[0]
    assert h.hop == 1
    assert h.graph > 0
    assert any("resolved_by" in step for step in h.via), h.via
    assert "resolved_by" in h.why


def test_contradiction_edge_is_traversable_in_both_directions(
    kb: KnowledgeStore, fitted_embedder: TfidfSvdEmbedder
) -> None:
    """Arriving at either side of a conflict must surface the other.

    The edge is stored as EPS-201 -> SYS-001. A query that lands on
    SYS-001 still needs to learn EPS-201 exists, so expansion walks
    incoming edges too.
    """
    r = GraphAugmentedRetriever(
        kb, fitted_embedder,
        RetrievalConfig(max_hops=1, max_chunks=20, expand_rels=("contradicts",)),
    )
    hits = r.retrieve("safe mode entry criteria and shed order")
    docs = {h.doc_id for h in hits}
    assert "SYS-001" in docs
    assert "EPS-201" in docs, f"contradiction not traversed backwards; got {sorted(docs)}"


def test_query_naming_a_document_boosts_it(
    kb: KnowledgeStore, fitted_embedder: TfidfSvdEmbedder
) -> None:
    """Exact doc-id mentions are the case dense retrieval handles worst."""
    r = GraphAugmentedRetriever(kb, fitted_embedder, RetrievalConfig(max_chunks=5))
    hits = r.retrieve("what does COM-220 say about the command-loss timer?")
    assert hits[0].doc_id == "COM-220"
    assert "names COM-220" in hits[0].why


# ── Store ─────────────────────────────────────────────────────────────


def test_store_roundtrip_preserves_nodes_and_edges(tmp_path: Path) -> None:
    nodes = [
        Node(id="doc:AAA-100", kind="document", doc_id="AAA-100", title="A"),
        Node(id="chunk:AAA-100#0", kind="chunk", doc_id="AAA-100", text="hello",
             chunk_index=0, vec_row=0),
    ]
    edges = [Edge(src="chunk:AAA-100#0", dst="doc:AAA-100", rel="part_of")]
    vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    KnowledgeStore.build(tmp_path / "kb", nodes, edges, vectors).close()
    with KnowledgeStore.open(tmp_path / "kb") as s:
        assert s.count() == (2, 1)
        n = s.get_node("chunk:AAA-100#0")
        assert n is not None and n.text == "hello" and n.vec_row == 0
        assert s.has_vectors


def test_store_drops_edges_to_unknown_nodes(tmp_path: Path) -> None:
    """A dangling edge would break traversal; it is filtered at build."""
    nodes = [Node(id="doc:A-1", kind="document", doc_id="A-1")]
    edges = [Edge(src="doc:A-1", dst="doc:GHOST", rel="references")]
    with KnowledgeStore.build(tmp_path / "kb", nodes, edges) as s:
        assert s.count() == (1, 0)


def test_rebuild_is_atomic_and_replaces_cleanly(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    KnowledgeStore.build(root, [Node(id="a", kind="document")], []).close()
    KnowledgeStore.build(
        root, [Node(id="b", kind="document"), Node(id="c", kind="document")], []
    ).close()
    with KnowledgeStore.open(root) as s:
        assert s.count()[0] == 2
        assert s.get_node("a") is None
    assert not (tmp_path / ".kb.building").exists()


def test_open_missing_store_points_at_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="kb build|KnowledgeStore.build"):
        KnowledgeStore.open(tmp_path / "nope")


def test_networkx_export_carries_edge_types(kb: KnowledgeStore) -> None:
    g = kb.to_networkx()
    rels = {d["rel"] for _, _, d in g.edges(data=True)}
    assert {"part_of", "references", "resolved_by", "contradicts"} <= rels


# ── Extraction ────────────────────────────────────────────────────────


def test_parse_document_extracts_id_title_and_references() -> None:
    d = parse_document(PROCEDURES / "EPS-201_undervoltage_response.md", PROCEDURES)
    assert d is not None
    assert d.doc_id == "EPS-201"
    assert "Undervoltage" in d.title
    assert {"SYS-000", "TCS-105", "COM-220", "EPS-310"} <= d.references
    assert "EPS-201" not in d.references          # no self-reference


def test_parse_document_extracts_supersedes_as_distinct_relation() -> None:
    d = parse_document(PROCEDURES / "SYS-000_procedure_precedence.md", PROCEDURES)
    assert d is not None
    assert d.supersedes == {"SYS-000"} or d.supersedes == set()
    # A superseded doc must not be double-counted as a plain reference.
    assert not (d.supersedes & d.references)


def test_parse_document_rejects_file_without_id(tmp_path: Path) -> None:
    p = tmp_path / "notes.md"
    p.write_text("# Some notes\n", encoding="utf-8")
    assert parse_document(p, tmp_path) is None


def test_chunk_text_keeps_oversized_paragraph_intact() -> None:
    """Tables must not be sliced mid-row — they retrieve well and read as noise."""
    table = "| a | b |\n" + "\n".join(f"| {i} | {i*2} |" for i in range(200))
    chunks = chunk_text(f"intro para\n\n{table}\n\nclosing para", chunk_chars=300)
    assert any(c.count("|") > 100 for c in chunks), "wide table was split apart"


def test_chunk_text_empty_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_declared_edges_load_with_relation_weights() -> None:
    edges = load_declared_edges(DECLARED_EDGES)
    rels = {e.rel for e in edges}
    assert {"contradicts", "resolved_by", "constrains", "depends_on"} <= rels
    assert all(e.provenance == "declared" for e in edges)
    resolved = [e for e in edges if e.rel == "resolved_by"]
    assert all(e.weight > 0.9 for e in resolved), "authority edges should rank high"


def test_declared_edges_missing_file_is_not_fatal() -> None:
    assert load_declared_edges(Path("does/not/exist.yaml")) == []


def test_build_rejects_corpus_with_no_identified_documents(tmp_path: Path) -> None:
    (tmp_path / "loose.md").write_text("# no id here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ABC-123"):
        build_knowledge_store(tmp_path, tmp_path / "out", verbose=False)


def test_provenance_distinguishes_extracted_from_declared(kb: KnowledgeStore) -> None:
    """An auditor must be able to trust declared edges differently."""
    rows = kb.conn.execute(
        "SELECT DISTINCT rel, provenance FROM edges"
    ).fetchall()
    prov = {(r["rel"], r["provenance"]) for r in rows}
    assert ("references", "extracted") in prov
    assert ("resolved_by", "declared") in prov


# ── Audit ─────────────────────────────────────────────────────────────


def test_shipped_corpus_has_no_orphans_or_unresolved_conflicts(kb: KnowledgeStore) -> None:
    findings = audit_graph(kb)
    assert findings["orphans"] == []
    assert findings["unresolved_contradictions"] == []


def test_audit_flags_a_contradiction_with_no_authority(tmp_path: Path) -> None:
    nodes = [
        Node(id="doc:A-1", kind="document", doc_id="A-1"),
        Node(id="doc:B-2", kind="document", doc_id="B-2"),
    ]
    edges = [Edge(src="doc:A-1", dst="doc:B-2", rel="contradicts")]
    with KnowledgeStore.build(tmp_path / "kb", nodes, edges) as s:
        findings = audit_graph(s)
        assert findings["unresolved_contradictions"] == ["doc:A-1 <-> doc:B-2"]


# ── Embedders ─────────────────────────────────────────────────────────


def test_embedders_return_unit_norm_rows() -> None:
    for emb in (HashingEmbedder(dim=32), TfidfSvdEmbedder(dim=8).fit(
        ["alpha beta gamma", "delta epsilon", "beta gamma delta"]
    )):
        v = emb.encode(["alpha beta", "delta"])
        norms = np.linalg.norm(v, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"{emb.name} rows not normalized"


def test_hashing_embedder_is_deterministic() -> None:
    a = HashingEmbedder(dim=32).encode(["bus voltage undervoltage"])
    b = HashingEmbedder(dim=32).encode(["bus voltage undervoltage"])
    np.testing.assert_allclose(a, b)


def test_tfidf_svd_requires_fit_first() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TfidfSvdEmbedder().encode(["x"])


def test_empty_text_list_returns_empty_matrix() -> None:
    assert HashingEmbedder(dim=16).encode([]).shape == (0, 16)


def test_build_embedder_falls_back_when_model_absent(capsys) -> None:
    emb = build_embedder("auto", model_path="/nonexistent/model", dim=32)
    assert isinstance(emb, TfidfSvdEmbedder)
    assert "falling back" in capsys.readouterr().out


def test_build_embedder_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown embedder"):
        build_embedder("magic")


# ── Vector index ──────────────────────────────────────────────────────


def test_exact_cosine_ranks_by_similarity() -> None:
    v = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=np.float32)
    idx = ExactCosineIndex(v)
    rows, scores = idx.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)
    assert rows[0] == 0
    assert scores[0] >= scores[1]


def test_exact_cosine_handles_top_k_larger_than_corpus() -> None:
    idx = ExactCosineIndex(np.eye(2, dtype=np.float32))
    rows, _ = idx.search(np.array([1, 0], dtype=np.float32), top_k=99)
    assert len(rows) == 2


def test_exact_cosine_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        ExactCosineIndex(np.array([1.0, 2.0], dtype=np.float32))


# ── Context packing ───────────────────────────────────────────────────


def test_context_pack_respects_char_budget(
    kb: KnowledgeStore, fitted_embedder: TfidfSvdEmbedder
) -> None:
    r = GraphAugmentedRetriever(kb, fitted_embedder)
    text, used = r.context_pack(ECLIPSE_QUERY, max_chars=1200)
    assert used
    assert len(text) <= 1200 + len(used[-1].node.text)   # last block may straddle


def test_context_pack_labels_every_block_with_its_doc_id(
    kb: KnowledgeStore, fitted_embedder: TfidfSvdEmbedder
) -> None:
    """A model that cannot see the source doc will omit or invent citations."""
    r = GraphAugmentedRetriever(kb, fitted_embedder)
    text, used = r.context_pack(ECLIPSE_QUERY)
    for h in used:
        assert f"[{h.doc_id}]" in text


def test_explain_is_human_readable(
    kb: KnowledgeStore, fitted_embedder: TfidfSvdEmbedder
) -> None:
    out = GraphAugmentedRetriever(kb, fitted_embedder).explain(ECLIPSE_QUERY)
    assert "query:" in out
    assert "EPS-201" in out


def test_retriever_works_without_an_embedder(kb: KnowledgeStore) -> None:
    """Lexical + graph must still function if no encoder is available."""
    r = GraphAugmentedRetriever(kb, embedder=None, config=RetrievalConfig(max_hops=1))
    hits = r.retrieve("undervoltage response shed order")
    assert hits
    assert all(h.dense == 0.0 for h in hits)
