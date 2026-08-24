"""Build and inspect the graph-augmented knowledge base.

Build it::

    # Offline fallback (no encoder staged) — works on any box
    python scripts/build_knowledge_base.py build

    # With a real encoder sneakernet'd onto the box
    python scripts/build_knowledge_base.py build \\
        --embedder transformers \\
        --model-path data/llm_weights/bge-small-en-v1.5

Inspect what it did::

    python scripts/build_knowledge_base.py audit
    python scripts/build_knowledge_base.py query "which shed order applies in eclipse?"

The ``query`` subcommand prints the retrieval trace including which edge
reached each expanded hit — use it when retrieval returns something
surprising, before blaming the model.

``--no-graph`` runs the same query with expansion disabled, which is the
A/B that shows what the graph is contributing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCEDURES = REPO_ROOT / "docs" / "procedures"
DEFAULT_EDGES = REPO_ROOT / "configs" / "knowledge" / "procedure_edges.yaml"
DEFAULT_STORE = REPO_ROOT / "data" / "knowledge" / "procedures"


def _fitted_embedder(store, kind: str, model_path: str | None, dim: int):
    """Rebuild the query-side embedder to match what the store was built with.

    The LSA fallback is stateful — it must be re-fitted on the same chunk
    text, or query vectors land in a different space than the index and
    dense retrieval silently returns noise.
    """
    from dgx_ts_lab.llm.knowledge import TfidfSvdEmbedder, build_embedder

    built_with = store.meta().get("embedder", "")
    if built_with.startswith("tfidf-svd") or kind in ("tfidf_svd", "lsa", "tfidf"):
        rows = store.conn.execute(
            "SELECT text FROM nodes WHERE kind='chunk' ORDER BY vec_row"
        ).fetchall()
        return TfidfSvdEmbedder(dim=dim).fit([r["text"] for r in rows])
    return build_embedder(kind, model_path=model_path, dim=dim)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("command", choices=["build", "audit", "query", "stats"])
    p.add_argument("text", nargs="?", default="", help="query text (for `query`)")
    p.add_argument("--procedures", type=Path, default=DEFAULT_PROCEDURES)
    p.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--embedder", default="auto",
                   choices=["auto", "transformers", "tfidf_svd", "hashing"])
    p.add_argument("--model-path", default=None,
                   help="local HF encoder directory (air-gap: stage it first)")
    p.add_argument("--dim", type=int, default=256, help="dim for the LSA fallback")
    p.add_argument("--chunk-chars", type=int, default=1400)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--no-graph", action="store_true",
                   help="disable expansion — the A/B baseline")
    args = p.parse_args()

    from dgx_ts_lab.llm.knowledge import (
        GraphAugmentedRetriever,
        KnowledgeStore,
        RetrievalConfig,
        audit_graph,
        build_embedder,
        build_knowledge_store,
    )

    if args.command == "build":
        emb = build_embedder(args.embedder, model_path=args.model_path, dim=args.dim)
        store = build_knowledge_store(
            procedures_dir=args.procedures,
            output_dir=args.store,
            embedder=emb,
            declared_edges_path=args.edges,
            chunk_chars=args.chunk_chars,
        )
        findings = audit_graph(store)
        if any(findings.values()):
            print("\n[audit] issues worth a look:")
            for k, v in findings.items():
                if v:
                    print(f"  {k}: {v}")
        print(f"\nQuery it:  python {Path(__file__).name} query \"...\"")
        return 0

    if not (args.store / "graph.db").exists():
        print(f"No store at {args.store}. Run `build` first.", file=sys.stderr)
        return 2

    with KnowledgeStore.open(args.store) as store:
        if args.command == "stats":
            n_nodes, n_edges = store.count()
            print(f"store:  {args.store}")
            print(f"nodes:  {n_nodes}\nedges:  {n_edges}")
            for k, v in sorted(store.meta().items()):
                print(f"  {k:14} {v}")
            rows = store.conn.execute(
                "SELECT rel, provenance, COUNT(*) c FROM edges "
                "GROUP BY rel, provenance ORDER BY c DESC"
            ).fetchall()
            print("\nedges by type:")
            for r in rows:
                print(f"  {r['c']:5d}  {r['rel']:<14} ({r['provenance']})")
            return 0

        if args.command == "audit":
            findings = audit_graph(store)
            clean = True
            for k, v in findings.items():
                if v:
                    clean = False
                    print(f"{k}:")
                    for item in v:
                        print(f"  - {item}")
            if clean:
                print("No orphaned documents, no unresolved contradictions.")
            return 0

        # query
        if not args.text:
            print("`query` needs text. Example:\n"
                  '  python scripts/build_knowledge_base.py query "eclipse shed order"',
                  file=sys.stderr)
            return 2
        emb = _fitted_embedder(store, args.embedder, args.model_path, args.dim)
        cfg = RetrievalConfig(
            max_hops=0 if args.no_graph else args.hops,
            max_chunks=args.top_k,
        )
        r = GraphAugmentedRetriever(store, emb, cfg)
        print(r.explain(args.text))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
