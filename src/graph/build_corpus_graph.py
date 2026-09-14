"""
build_corpus_graph.py — Build the structural + semantic corpus graph.

Graph structure
---------------
Nodes  : one per chunk (node_id = chunk_id hash string)
Edges  :
  NEXT_CHUNK  chunk_i → chunk_{i+1}  within the same document (by chunk_index)
  PREV_CHUNK  chunk_{i+1} → chunk_i  reverse, for bidirectional traversal

Node attributes stored on every chunk node:
  source_type, repo, url, chunk_index, total_chunks, token_count,
  topic_id, corpus_pos

Lookup structures saved alongside the graph (plain dicts, O(1) access):
  doc_to_chunks     : {url: [chunk_id, ...]}  ordered by chunk_index
  cluster_to_chunks : {topic_id: [chunk_id, ...]}
  chunk_to_pos      : {chunk_id: corpus_position}
  pos_to_chunk_id   : {corpus_position: chunk_id}

Output
------
  data/models/corpus_graph.pkl   (pickle, HIGHEST_PROTOCOL)

Usage
-----
  python src/graph/build_corpus_graph.py
  python src/graph/build_corpus_graph.py --verify
"""

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).parents[2]
CORPUS_PATH = ROOT / "data" / "processed" / "corpus_with_topics.jsonl"
OUT_PATH    = ROOT / "data" / "models" / "corpus_graph.pkl"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build() -> dict:
    """Load corpus and build graph + lookup structures."""

    # ------------------------------------------------------------------
    # 1. Load corpus
    # ------------------------------------------------------------------
    log.info("Loading corpus from %s ...", CORPUS_PATH)
    corpus: list[dict] = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    log.info("  %d chunks loaded", len(corpus))

    # ------------------------------------------------------------------
    # 2. Add chunk nodes + accumulate lookup data
    # ------------------------------------------------------------------
    log.info("Adding chunk nodes ...")
    G = nx.DiGraph()

    # {url: [(chunk_index, chunk_id)]}  — will be sorted before edge pass
    doc_buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)

    cluster_to_chunks: dict[int, list[str]] = defaultdict(list)
    chunk_to_pos:      dict[str, int]       = {}

    for pos, chunk in enumerate(corpus):
        cid      = chunk["chunk_id"]
        url      = chunk["url"]
        topic_id = int(chunk.get("topic_id", -1))

        G.add_node(
            cid,
            node_type   = "chunk",
            source_type = chunk["source_type"],
            repo        = chunk["repo"],
            url         = url,
            chunk_index = chunk["chunk_index"],
            total_chunks= chunk["total_chunks"],
            token_count = chunk["token_count"],
            topic_id    = topic_id,
            corpus_pos  = pos,
        )

        doc_buckets[url].append((chunk["chunk_index"], cid))
        cluster_to_chunks[topic_id].append(cid)
        chunk_to_pos[cid] = pos

    log.info("  %d chunk nodes added", G.number_of_nodes())
    log.info("  %d unique documents", len(doc_buckets))
    log.info("  %d clusters (including noise -1)", len(cluster_to_chunks))

    # ------------------------------------------------------------------
    # 3. Add NEXT_CHUNK / PREV_CHUNK edges + build doc_to_chunks
    # ------------------------------------------------------------------
    log.info("Adding sequential edges ...")
    doc_to_chunks: dict[str, list[str]] = {}
    edge_count = 0

    for url, items in doc_buckets.items():
        ordered = [cid for _, cid in sorted(items, key=lambda x: x[0])]
        doc_to_chunks[url] = ordered

        for i in range(len(ordered) - 1):
            G.add_edge(ordered[i],     ordered[i + 1], relation="NEXT_CHUNK", weight=1.0)
            G.add_edge(ordered[i + 1], ordered[i],     relation="PREV_CHUNK", weight=1.0)
            edge_count += 2

    log.info("  %d NEXT/PREV_CHUNK edges", edge_count)

    # ------------------------------------------------------------------
    # 4. Finalise lookup structures
    # ------------------------------------------------------------------
    pos_to_chunk_id: dict[int, str] = {pos: cid for cid, pos in chunk_to_pos.items()}

    return {
        "graph":             G,
        "doc_to_chunks":     doc_to_chunks,
        "cluster_to_chunks": dict(cluster_to_chunks),
        "chunk_to_pos":      chunk_to_pos,
        "pos_to_chunk_id":   pos_to_chunk_id,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(artifact: dict) -> None:
    G                 = artifact["graph"]
    doc_to_chunks     = artifact["doc_to_chunks"]
    cluster_to_chunks = artifact["cluster_to_chunks"]

    doc_sizes = [len(v) for v in doc_to_chunks.values()]
    signal    = {k: v for k, v in cluster_to_chunks.items() if k != -1}
    cl_sizes  = [len(v) for v in signal.values()]

    source_counts: dict[str, int] = defaultdict(int)
    repo_counts:   dict[str, int] = defaultdict(int)
    for _, data in G.nodes(data=True):
        source_counts[data["source_type"]] += 1
        repo_counts[data["repo"]]          += 1

    log.info("=" * 52)
    log.info("  Corpus graph stats")
    log.info("=" * 52)
    log.info("  Nodes (chunks)       : %d", G.number_of_nodes())
    log.info("  Edges (NEXT+PREV)    : %d", G.number_of_edges())
    log.info("  Documents            : %d", len(doc_to_chunks))
    log.info("  Avg chunks / doc     : %.1f", sum(doc_sizes) / len(doc_sizes))
    log.info("  Max chunks / doc     : %d", max(doc_sizes))
    log.info("  Single-chunk docs    : %d", sum(1 for s in doc_sizes if s == 1))
    log.info("  Signal clusters      : %d", len(signal))
    log.info("  Avg chunks / cluster : %.1f", sum(cl_sizes) / max(len(cl_sizes), 1))
    log.info("  Max chunks / cluster : %d", max(cl_sizes) if cl_sizes else 0)
    log.info("  Noise chunks (-1)    : %d", len(cluster_to_chunks.get(-1, [])))
    log.info("  Source type breakdown:")
    for st, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        log.info("    %-22s %d", st, cnt)
    log.info("  Repo breakdown:")
    for repo, cnt in sorted(repo_counts.items(), key=lambda x: -x[1]):
        log.info("    %-30s %d", repo, cnt)
    log.info("=" * 52)


# ---------------------------------------------------------------------------
# Verify (optional)
# ---------------------------------------------------------------------------

def verify(artifact: dict) -> None:
    """Spot-check graph integrity after loading from disk."""
    G                 = artifact["graph"]
    doc_to_chunks     = artifact["doc_to_chunks"]
    cluster_to_chunks = artifact["cluster_to_chunks"]
    chunk_to_pos      = artifact["chunk_to_pos"]
    pos_to_chunk_id   = artifact["pos_to_chunk_id"]

    errors = 0

    # 1. Every node has required attributes
    required = {"source_type", "repo", "url", "chunk_index",
                "total_chunks", "token_count", "topic_id", "corpus_pos"}
    for cid, data in G.nodes(data=True):
        missing = required - set(data.keys())
        if missing:
            log.error("Node %s missing attributes: %s", cid, missing)
            errors += 1

    # 2. NEXT_CHUNK edges respect chunk_index order
    sample_docs = list(doc_to_chunks.items())[:20]
    for url, cids in sample_docs:
        for i in range(len(cids) - 1):
            if not G.has_edge(cids[i], cids[i + 1]):
                log.error("Missing NEXT_CHUNK edge: %s → %s", cids[i], cids[i + 1])
                errors += 1
            if not G.has_edge(cids[i + 1], cids[i]):
                log.error("Missing PREV_CHUNK edge: %s → %s", cids[i + 1], cids[i])
                errors += 1

    # 3. chunk_to_pos and pos_to_chunk_id are inverses
    sample_chunks = list(chunk_to_pos.items())[:100]
    for cid, pos in sample_chunks:
        if pos_to_chunk_id.get(pos) != cid:
            log.error("pos_to_chunk_id mismatch at pos=%d", pos)
            errors += 1

    # 4. cluster_to_chunks covers all nodes
    all_mapped = sum(len(v) for v in cluster_to_chunks.values())
    if all_mapped != G.number_of_nodes():
        log.error(
            "cluster_to_chunks covers %d chunks, expected %d",
            all_mapped, G.number_of_nodes()
        )
        errors += 1

    # 5. doc_to_chunks covers all nodes
    all_doc_mapped = sum(len(v) for v in doc_to_chunks.values())
    if all_doc_mapped != G.number_of_nodes():
        log.error(
            "doc_to_chunks covers %d chunks, expected %d",
            all_doc_mapped, G.number_of_nodes()
        )
        errors += 1

    if errors == 0:
        log.info("Verification passed — no errors found.")
    else:
        log.error("Verification finished with %d error(s).", errors)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build corpus graph.")
    parser.add_argument(
        "--verify", action="store_true",
        help="Re-load saved file and run integrity checks after building."
    )
    args = parser.parse_args()

    artifact = build()
    print_stats(artifact)

    log.info("Saving to %s ...", OUT_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved (%.1f MB)", OUT_PATH.stat().st_size / 1e6)

    if args.verify:
        log.info("Running verification ...")
        with open(OUT_PATH, "rb") as f:
            loaded = pickle.load(f)
        verify(loaded)


if __name__ == "__main__":
    main()
