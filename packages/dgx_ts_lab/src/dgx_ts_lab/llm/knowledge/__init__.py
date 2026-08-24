"""Graph-augmented knowledge base for procedure retrieval.

Three retrieval mechanisms fused, each covering a failure the others
have:

* **dense** — paraphrase and vocabulary mismatch (what TF-IDF alone gets
  wrong)
* **lexical** — exact document identifiers like ``EPS-201``, which
  embedding models represent poorly (what dense alone gets wrong)
* **graph** — documents the query had no way to ask for, reached by
  typed edges such as ``resolved_by`` and ``contradicts`` (what neither
  similarity method can reach at all)

Storage is one SQLite file plus one numpy array sharing a single node
identity — not two databases that drift apart on rebuild. See
``store.KnowledgeStore``.

Build one::

    from dgx_ts_lab.llm.knowledge import build_knowledge_store, build_embedder

    store = build_knowledge_store(
        procedures_dir="docs/procedures",
        output_dir="data/knowledge/procedures",
        embedder=build_embedder("auto", model_path="data/llm_weights/bge-small-en-v1.5"),
        declared_edges_path="configs/knowledge/procedure_edges.yaml",
    )

Query it::

    from dgx_ts_lab.llm.knowledge import GraphAugmentedRetriever
    r = GraphAugmentedRetriever(store, embedder)
    print(r.explain("which shed order applies during eclipse?"))
"""

from .embed import (
    Embedder,
    HashingEmbedder,
    TfidfSvdEmbedder,
    TransformersEmbedder,
    build_embedder,
)
from .extract import (
    DOC_ID_RE,
    audit_graph,
    build_knowledge_store,
    chunk_text,
    load_declared_edges,
    parse_document,
)
from .retrieve import GraphAugmentedRetriever, Hit, RetrievalConfig
from .store import (
    DEFAULT_EDGE_WEIGHTS,
    REL_CONSTRAINS,
    REL_CONTRADICTS,
    REL_DEPENDS_ON,
    REL_PART_OF,
    REL_REFERENCES,
    REL_RESOLVED_BY,
    REL_SUPERSEDES,
    Edge,
    ExactCosineIndex,
    KnowledgeStore,
    Node,
    VectorIndex,
)

__all__ = [
    "DEFAULT_EDGE_WEIGHTS",
    "DOC_ID_RE",
    "Edge",
    "Embedder",
    "ExactCosineIndex",
    "GraphAugmentedRetriever",
    "HashingEmbedder",
    "Hit",
    "KnowledgeStore",
    "Node",
    "REL_CONSTRAINS",
    "REL_CONTRADICTS",
    "REL_DEPENDS_ON",
    "REL_PART_OF",
    "REL_REFERENCES",
    "REL_RESOLVED_BY",
    "REL_SUPERSEDES",
    "RetrievalConfig",
    "TfidfSvdEmbedder",
    "TransformersEmbedder",
    "VectorIndex",
    "audit_graph",
    "build_embedder",
    "build_knowledge_store",
    "chunk_text",
    "load_declared_edges",
    "parse_document",
]
