"""
forget_gate.py — Graph-centrality eviction (the "forget gate" analogy to LSTMs).

Mechanism
---------
Every K steps, or when the context window exceeds a token threshold, the forget
gate replaces naive time-based eviction (evict_older_than) with a score derived
from two signals:

  1. PageRank centrality — chunks retrieved by many queries and connected to
     other highly-retrieved chunks score higher.

  2. Recency decay — chunks not retrieved recently are down-weighted:
       recency_decay = exp(-λ · steps_since_last_retrieval)

  Combined score:
       chunk_score = PageRank(chunk) × recency_decay(chunk)

  Chunks whose score falls below the τ-th percentile are evicted from the
  ContextRegistry.  Evicted chunk_ids are returned to the caller (typically
  SessionRetriever) so they can be recorded as EVICTED edges in the episodic store.

PageRank subgraph
-----------------
Built fresh each time the gate fires, containing only the current session:

  Nodes : query nodes + chunk nodes in the running context
  Edges :
    RETRIEVED   query_i → chunk_x   weight = retrieval_score (from episodic store)
    NEXT_CHUNK  chunk_x → chunk_x+1 weight = 1.0             (from corpus graph)

LSTM analogy
------------
  Cell state   ←→  Session graph (persistent episodic memory)
  Hidden state ←→  ContextRegistry (working memory / running context)
  Forget gate  ←→  This module — PageRank + recency eviction every K queries

Usage
-----
  from src.graph.forget_gate import ForgetGate
  import pickle

  with open("data/models/corpus_graph.pkl", "rb") as f:
      corpus_artifact = pickle.load(f)

  gate = ForgetGate(corpus_artifact=corpus_artifact)

  # Pass as forget_gate_fn to SessionRetriever:
  sr = SessionRetriever(..., forget_gate_fn=gate)

  # Or call directly:
  evicted_ids = gate(store, registry, session_id)
"""

import logging
import math
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

# Default hyperparameters (from design doc)
_LAMBDA_DECAY        = 0.1    # recency decay rate
_EVICTION_PERCENTILE = 20.0   # evict bottom N% by combined score
_MIN_CHUNKS_TO_EVICT = 5      # don't fire if registry has fewer chunks than this


class ForgetGate:
    """
    Callable forget gate — pass as ``forget_gate_fn`` to SessionRetriever.

    Parameters
    ----------
    corpus_artifact : dict
        Loaded corpus_graph.pkl — provides the corpus graph for NEXT_CHUNK edges.
    lambda_decay : float
        Recency decay rate λ.  Higher → more aggressively forgets old context.
    eviction_percentile : float
        Evict chunks whose combined score is below this percentile (0–100).
        Default 20 → bottom 20% evicted each time the gate fires.
    min_chunks : int
        Skip eviction if the registry has fewer than this many chunks.
    damping : float
        PageRank damping factor (standard is 0.85).
    """

    def __init__(
        self,
        corpus_artifact:      dict,
        lambda_decay:         float = _LAMBDA_DECAY,
        eviction_percentile:  float = _EVICTION_PERCENTILE,
        min_chunks:           int   = _MIN_CHUNKS_TO_EVICT,
        damping:              float = 0.85,
    ):
        self._corpus_graph: nx.DiGraph = corpus_artifact["graph"]
        self.lambda_decay        = lambda_decay
        self.eviction_percentile = eviction_percentile
        self.min_chunks          = min_chunks
        self.damping             = damping

    def __call__(self, store, registry, session_id: str) -> list[str]:
        """
        Run the forget gate.

        Parameters
        ----------
        store      : EpisodicStore
        registry   : ContextRegistry
        session_id : str

        Returns
        -------
        list[str]  — chunk_ids evicted from the registry.
        """
        return run_forget_gate(
            store               = store,
            registry            = registry,
            session_id          = session_id,
            corpus_graph        = self._corpus_graph,
            lambda_decay        = self.lambda_decay,
            eviction_percentile = self.eviction_percentile,
            min_chunks          = self.min_chunks,
            damping             = self.damping,
        )


# ---------------------------------------------------------------------------
# Core logic (also callable standalone)
# ---------------------------------------------------------------------------

def run_forget_gate(
    store,
    registry,
    session_id:           str,
    corpus_graph:         nx.DiGraph,
    lambda_decay:         float = _LAMBDA_DECAY,
    eviction_percentile:  float = _EVICTION_PERCENTILE,
    min_chunks:           int   = _MIN_CHUNKS_TO_EVICT,
    damping:              float = 0.85,
) -> list[str]:
    """
    Standalone forget gate — computes PageRank + recency scores on the active
    context, evicts the bottom percentile, returns evicted chunk_ids.

    Parameters
    ----------
    store               : EpisodicStore
    registry            : ContextRegistry
    session_id          : str
    corpus_graph        : nx.DiGraph  (from corpus_graph.pkl)
    lambda_decay        : float
    eviction_percentile : float  (0–100)
    min_chunks          : int
    damping             : float  PageRank damping

    Returns
    -------
    list[str]  evicted chunk_ids (already removed from registry)
    """
    chunk_ids = list(registry.loaded_ids)

    if len(chunk_ids) < min_chunks:
        log.debug(
            "Forget gate skipped — registry has %d chunks (min=%d).",
            len(chunk_ids), min_chunks,
        )
        return []

    current_step = registry.current_step

    # ── 1. Build PageRank subgraph ─────────────────────────────────────────
    G_pr = _build_pagerank_subgraph(
        store, session_id, chunk_ids, corpus_graph
    )

    # ── 2. Run PageRank ────────────────────────────────────────────────────
    try:
        pr = nx.pagerank(G_pr, alpha=damping, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        log.warning("PageRank did not converge — using degree centrality fallback.")
        pr = nx.degree_centrality(G_pr)

    # ── 3. Recency decay per chunk ─────────────────────────────────────────
    last_step = _last_retrieval_step(store, session_id, chunk_ids)

    chunk_scores: dict[str, float] = {}
    for cid in chunk_ids:
        pagerank_score  = pr.get(cid, 0.0)
        steps_since     = current_step - last_step.get(cid, 0)
        decay           = math.exp(-lambda_decay * max(steps_since, 0))
        chunk_scores[cid] = pagerank_score * decay

    # ── 4. Threshold: bottom eviction_percentile ───────────────────────────
    scores_arr = np.array(list(chunk_scores.values()), dtype=np.float64)
    threshold  = float(np.percentile(scores_arr, eviction_percentile))

    to_evict = [
        cid for cid, score in chunk_scores.items()
        if score <= threshold
    ]

    log.info(
        "Forget gate: %d chunks scored | threshold=%.6f (p%.0f) | "
        "evicting %d chunks",
        len(chunk_ids), threshold, eviction_percentile, len(to_evict),
    )
    if log.isEnabledFor(logging.DEBUG):
        sorted_scores = sorted(chunk_scores.items(), key=lambda x: x[1])
        for cid, sc in sorted_scores[:5]:
            log.debug("  evict candidate: %s  score=%.6f  last_step=%d",
                      cid, sc, last_step.get(cid, 0))

    # ── 5. Evict from registry ─────────────────────────────────────────────
    evicted_ids = [cid for cid in to_evict if cid in registry.loaded_ids]
    if evicted_ids:
        registry.evict_unreferenced(
            referenced_ids=registry.loaded_ids - set(evicted_ids)
        )

    log.info(
        "Forget gate complete: evicted %d / %d chunks  (registry now %d, tokens=%d)",
        len(evicted_ids), len(chunk_ids),
        registry.size,
        registry.token_count(),
    )
    return evicted_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pagerank_subgraph(
    store,
    session_id: str,
    chunk_ids:  list[str],
    corpus_graph: nx.DiGraph,
) -> nx.DiGraph:
    """
    Build the PageRank graph for the active session context.

    Nodes : query nodes (this session) + chunk nodes (in running context)
    Edges :
      RETRIEVED  query → chunk   weight = retrieval_score
      NEXT_CHUNK chunk → chunk   weight = 1.0  (from corpus graph, within context)
    """
    G = nx.DiGraph()
    chunk_set = set(chunk_ids)

    # Add all chunk nodes
    G.add_nodes_from(chunk_ids)

    # Add RETRIEVED edges from episodic store (query → chunk)
    session_graph = store._G
    for nid, data in session_graph.nodes(data=True):
        if data.get("node_type") != "query":
            continue
        if data.get("session_id") != session_id:
            continue

        G.add_node(nid)   # query node

        for _, target, edata in session_graph.out_edges(nid, data=True):
            if edata.get("relation") != "RETRIEVED":
                continue
            if target not in chunk_set:
                continue
            weight = float(edata.get("weight", 1.0))
            # Accumulate weight if edge already exists (chunk retrieved multiple times)
            if G.has_edge(nid, target):
                G[nid][target]["weight"] += weight
            else:
                G.add_edge(nid, target, weight=weight)

    # Add NEXT_CHUNK edges within the active context chunk set
    for cid in chunk_ids:
        if not corpus_graph.has_node(cid):
            continue
        for _, neighbor, edata in corpus_graph.out_edges(cid, data=True):
            if edata.get("relation") == "NEXT_CHUNK" and neighbor in chunk_set:
                G.add_edge(cid, neighbor, weight=1.0)

    return G


def _last_retrieval_step(
    store,
    session_id: str,
    chunk_ids:  list[str],
) -> dict[str, int]:
    """
    For each chunk in chunk_ids, find the most recent step at which it was
    retrieved in this session.  Returns {chunk_id: step}.
    """
    last_step: dict[str, int] = {}
    session_graph = store._G
    chunk_set     = set(chunk_ids)

    for nid, data in session_graph.nodes(data=True):
        if data.get("node_type") != "query":
            continue
        if data.get("session_id") != session_id:
            continue

        for _, target, edata in session_graph.out_edges(nid, data=True):
            if edata.get("relation") != "RETRIEVED":
                continue
            if target not in chunk_set:
                continue
            step = int(edata.get("step", 0))
            if target not in last_step or step > last_step[target]:
                last_step[target] = step

    return last_step


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def explain_scores(
    store,
    registry,
    session_id:   str,
    corpus_graph: nx.DiGraph,
    lambda_decay: float = _LAMBDA_DECAY,
    damping:      float = 0.85,
    top_n:        int   = 20,
) -> list[dict]:
    """
    Return a ranked list of chunk scores without evicting anything.
    Useful for paper analysis and ablation.

    Returns list of dicts sorted by combined_score desc:
      { chunk_id, pagerank, recency_decay, combined_score, last_step,
        steps_since, source_type, repo }
    """
    chunk_ids    = list(registry.loaded_ids)
    current_step = registry.current_step

    G_pr = _build_pagerank_subgraph(store, session_id, chunk_ids, corpus_graph)

    try:
        pr = nx.pagerank(G_pr, alpha=damping, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        pr = nx.degree_centrality(G_pr)

    last_step = _last_retrieval_step(store, session_id, chunk_ids)

    rows = []
    for cid in chunk_ids:
        pg          = pr.get(cid, 0.0)
        steps_since = current_step - last_step.get(cid, 0)
        decay       = math.exp(-lambda_decay * max(steps_since, 0))
        combined    = pg * decay

        node_attrs  = corpus_graph.nodes.get(cid, {})
        rows.append({
            "chunk_id":      cid,
            "pagerank":      round(pg, 8),
            "recency_decay": round(decay, 4),
            "combined_score":round(combined, 8),
            "last_step":     last_step.get(cid, 0),
            "steps_since":   steps_since,
            "source_type":   node_attrs.get("source_type", ""),
            "repo":          node_attrs.get("repo", ""),
        })

    rows.sort(key=lambda x: -x["combined_score"])
    return rows[:top_n]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    import pickle, tempfile, os, sys
    # Ensure project root is on path when running as __main__
    _root = str(Path(__file__).parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    from src.graph.episodic_store import EpisodicStore
    from src.agent.context_manager import ContextRegistry

    ROOT = Path(__file__).parents[2]
    corpus_path = ROOT / "data" / "models" / "corpus_graph.pkl"
    with open(corpus_path, "rb") as f:
        artifact = pickle.load(f)

    corpus_graph: nx.DiGraph = artifact["graph"]

    # --- Build a minimal fake session in an ephemeral store ---
    tmp = Path(tempfile.mktemp(suffix=".pkl"))
    store    = EpisodicStore(path=tmp, embed_queries=False)
    registry = ContextRegistry(token_budget=8192)

    sid = store.start_session(project_id="smoke_test")

    # Pull 10 real chunk_ids from the corpus
    real_chunk_ids = list(artifact["doc_to_chunks"].values())[0][:10]

    # Simulate 3 steps
    for step in range(1, 4):
        registry.advance_step()
        qid = store.record_query(
            sid, text=f"query step {step}", step_number=step,
            embedding=np.random.randn(384).astype("float32"),
        )
        for i, cid in enumerate(real_chunk_ids):
            score = 0.9 - i * 0.05
            # Fake chunk dict
            registry.load({"chunk_id": cid, "text": f"text for {cid}", "score": score})
            store.record_retrieved(qid, cid, score=score, step=step)

    log.info("Registry size before gate: %d", registry.size)
    log.info("Token count before gate  : %d", registry.token_count())

    gate = ForgetGate(corpus_artifact=artifact, eviction_percentile=30.0)
    evicted = gate(store, registry, sid)

    log.info("Evicted chunk_ids: %s", evicted)
    log.info("Registry size after gate: %d", registry.size)
    assert len(evicted) > 0, "Expected at least one eviction"
    assert all(cid not in registry.loaded_ids for cid in evicted), \
        "Evicted chunks still in registry"

    # explain_scores should work post-eviction
    rows = explain_scores(store, registry, sid, corpus_graph)
    assert len(rows) == registry.size
    log.info("Top scored chunk: %s (combined=%.6f)", rows[0]["chunk_id"], rows[0]["combined_score"])

    store.save()
    os.unlink(tmp)
    log.info("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
