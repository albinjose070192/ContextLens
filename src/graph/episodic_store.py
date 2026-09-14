"""
episodic_store.py — Persistent episodic knowledge graph for session-aware retrieval.

Graph structure
---------------
Nodes:
  session_id  (node_type="session")
  query_id    (node_type="query")
  chunk_id    (node_type="chunk", shared namespace with corpus graph)

Edges:
  CONTAINS    session  → query                  (containment)
  NEXT_STEP   query_N  → query_{N+1}            (sequential steps within session)
  RETRIEVED   query    → chunk                  (weight=score, step=N)
  EVICTED     query    → chunk                  (step=N, strategy=str)
  SIMILAR_TO  query_i  → query_j               (cosine_sim > threshold, cross-session)

All three layers (corpus graph, this store, session graph) share the same chunk_id
namespace — chunk nodes added here are lightweight stubs; full attributes live in
corpus_graph.pkl.

Persistence
-----------
  data/models/session_graph.pkl   (appended across sessions)

Usage
-----
  from src.graph.episodic_store import EpisodicStore

  store = EpisodicStore()
  sid = store.start_session(project_id="msds453")
  qid = store.record_query(sid, text="How do I use FastAPI?", step_number=1)
  store.record_retrieved(qid, chunk_id="abc123", score=0.91, step=1)
  store.end_session(sid, summary="Covered FastAPI routing basics.")
  store.save()
"""

import logging
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

ROOT       = Path(__file__).parents[2]
STORE_PATH = ROOT / "data" / "models" / "session_graph.pkl"

SIMILAR_TO_THRESHOLD = 0.88   # default cosine similarity threshold for push-back


class EpisodicStore:
    """
    Persistent episodic memory graph.

    Maintains a DiGraph of sessions, queries, and chunk interactions.
    The chunk_id namespace is shared with corpus_graph — no data duplication.

    Parameters
    ----------
    path : Path
        Where to load/save the session graph pickle.
    embed_queries : bool
        If True, load MiniLM on init and auto-embed query text inside
        record_query(). Set False when embeddings are supplied by the caller.
    """

    def __init__(
        self,
        path:          Path = STORE_PATH,
        embed_queries: bool = True,
    ):
        self.path      = path
        self._embedder = None

        if path.exists():
            log.info("Loading existing session graph from %s ...", path)
            with open(path, "rb") as f:
                self._G: nx.DiGraph = pickle.load(f)
            log.info(
                "  %d nodes, %d edges loaded",
                self._G.number_of_nodes(),
                self._G.number_of_edges(),
            )
        else:
            log.info("No existing session graph — starting fresh.")
            self._G = nx.DiGraph()

        if embed_queries:
            self._load_embedder()

    # -----------------------------------------------------------------------
    # Embedder (lazy-loaded)
    # -----------------------------------------------------------------------

    def _load_embedder(self) -> None:
        try:
            from src.evaluation.embedding_ablation import MiniLMEmbedder
            self._embedder = MiniLMEmbedder()
        except Exception as exc:
            log.warning("Could not load MiniLM embedder (%s) — embeddings disabled.", exc)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        """Return (384,) float32 L2-normalised embedding, or None if unavailable."""
        if self._embedder is None:
            return None
        return self._embedder.embed(text)

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    def start_session(self, project_id: str = "default") -> str:
        """
        Open a new session node. Returns session_id.
        """
        sid = f"session_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        self._G.add_node(
            sid,
            node_type  = "session",
            project_id = project_id,
            start_time = now,
            end_time   = None,
            n_steps    = 0,
            summary    = None,
        )
        log.info("Session started: %s  (project=%s)", sid, project_id)
        return sid

    def end_session(
        self,
        session_id: str,
        summary:    Optional[str] = None,
    ) -> None:
        """
        Close a session: record end_time, step count, optional summary.
        """
        if not self._G.has_node(session_id):
            raise KeyError(f"Unknown session_id: {session_id!r}")

        query_ids = self._session_query_ids(session_id)
        self._G.nodes[session_id]["end_time"] = datetime.now(timezone.utc).isoformat()
        self._G.nodes[session_id]["n_steps"]  = len(query_ids)
        self._G.nodes[session_id]["summary"]  = summary
        log.info(
            "Session ended: %s  (%d steps, summary=%s)",
            session_id, len(query_ids), "yes" if summary else "no",
        )

    # -----------------------------------------------------------------------
    # Query recording
    # -----------------------------------------------------------------------

    def record_query(
        self,
        session_id:       str,
        text:             str,
        step_number:      int,
        embedding:        Optional[np.ndarray] = None,
        response_summary: Optional[str] = None,
    ) -> str:
        """
        Add a query node and wire it into the session.

        Adds:
          - Query node with embedding (auto-computed if not provided)
          - CONTAINS edge (session → query)
          - NEXT_STEP edge from the previous query in this session

        Returns query_id.
        """
        if not self._G.has_node(session_id):
            raise KeyError(f"Unknown session_id: {session_id!r}")

        qid = f"query_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()

        if embedding is None:
            embedding = self._embed(text)

        emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        self._G.add_node(
            qid,
            node_type        = "query",
            text             = text,
            embedding        = emb_list,
            timestamp        = now,
            step_number      = step_number,
            session_id       = session_id,
            project_id       = self._G.nodes[session_id]["project_id"],
            response_summary = response_summary,
        )

        # session → query containment
        self._G.add_edge(session_id, qid, relation="CONTAINS")

        # chain NEXT_STEP from previous query
        prior = [q for q in self._session_query_ids(session_id) if q != qid]
        if prior:
            last_query = max(prior, key=lambda q: self._G.nodes[q].get("step_number", 0))
            self._G.add_edge(last_query, qid, relation="NEXT_STEP")

        return qid

    def update_response_summary(self, query_id: str, summary: str) -> None:
        """Set or update response_summary on an existing query node (call after LLM responds)."""
        if not self._G.has_node(query_id):
            raise KeyError(f"Unknown query_id: {query_id!r}")
        self._G.nodes[query_id]["response_summary"] = summary

    # -----------------------------------------------------------------------
    # Chunk interaction recording
    # -----------------------------------------------------------------------

    def record_retrieved(
        self,
        query_id: str,
        chunk_id: str,
        score:    float,
        step:     int,
    ) -> None:
        """Add RETRIEVED edge: query → chunk (weight=score)."""
        self._ensure_chunk_node(chunk_id)
        self._G.add_edge(
            query_id, chunk_id,
            relation = "RETRIEVED",
            weight   = float(score),
            step     = step,
        )

    def record_evicted(
        self,
        query_id: str,
        chunk_id: str,
        step:     int,
        strategy: str = "pagerank",
    ) -> None:
        """Add EVICTED edge: query → chunk."""
        self._ensure_chunk_node(chunk_id)
        self._G.add_edge(
            query_id, chunk_id,
            relation = "EVICTED",
            step     = step,
            strategy = strategy,
        )

    # -----------------------------------------------------------------------
    # Context state queries
    # -----------------------------------------------------------------------

    def get_retrieved_chunks(self, session_id: str) -> set[str]:
        """
        Chunk_ids retrieved but not yet evicted in this session.
        Approximates the current running context from the graph perspective.
        """
        retrieved: set[str] = set()
        evicted:   set[str] = set()
        for qid in self._session_query_ids(session_id):
            for _, target, data in self._G.out_edges(qid, data=True):
                rel = data.get("relation")
                if rel == "RETRIEVED":
                    retrieved.add(target)
                elif rel == "EVICTED":
                    evicted.add(target)
        return retrieved - evicted

    def get_evicted_chunks(self, session_id: str) -> set[str]:
        """All chunk_ids evicted at any point in this session."""
        evicted: set[str] = set()
        for qid in self._session_query_ids(session_id):
            for _, target, data in self._G.out_edges(qid, data=True):
                if data.get("relation") == "EVICTED":
                    evicted.add(target)
        return evicted

    def get_session_chunk_scores(self, session_id: str) -> dict[str, float]:
        """
        Maps chunk_id → max retrieval score across all steps in this session.
        Used by forget_gate.py as seed weights for PageRank.
        """
        scores: dict[str, float] = {}
        for qid in self._session_query_ids(session_id):
            for _, target, data in self._G.out_edges(qid, data=True):
                if data.get("relation") == "RETRIEVED":
                    scores[target] = max(
                        scores.get(target, 0.0),
                        float(data.get("weight", 0.0)),
                    )
        return scores

    def get_visited_clusters(self, session_id: str, corpus_graph: nx.DiGraph) -> set[int]:
        """
        Topic_ids of clusters visited in this session.
        Requires the corpus_graph so we can look up topic_id on chunk nodes.
        Used by session_retriever.py step 2.
        """
        clusters: set[int] = set()
        for chunk_id in self.get_retrieved_chunks(session_id):
            if corpus_graph.has_node(chunk_id):
                tid = corpus_graph.nodes[chunk_id].get("topic_id")
                if tid is not None:
                    clusters.add(int(tid))
        return clusters

    # -----------------------------------------------------------------------
    # Similar query search (push-back feature)
    # -----------------------------------------------------------------------

    def find_similar_queries(
        self,
        embedding:  np.ndarray,
        threshold:  float = SIMILAR_TO_THRESHOLD,
        top_n:      int   = 5,
        project_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Search prior query nodes by cosine similarity to `embedding`.

        Returns up to top_n results (sorted desc by similarity), each:
          { query_id, similarity, text, response_summary, session_id, timestamp }

        MiniLM embeddings are already L2-normalised, so dot product = cosine sim.
        """
        if embedding is None:
            return []

        # Normalise defensively (in case caller passes un-normalised vector)
        norm = np.linalg.norm(embedding)
        q_unit = embedding / (norm + 1e-9)

        results = []
        for node_id, data in self._G.nodes(data=True):
            if data.get("node_type") != "query":
                continue
            if project_id and data.get("project_id") != project_id:
                continue
            stored = data.get("embedding")
            if stored is None:
                continue

            v = np.asarray(stored, dtype=np.float32)
            v_unit = v / (np.linalg.norm(v) + 1e-9)
            sim = float(np.dot(q_unit, v_unit))

            if sim >= threshold:
                results.append({
                    "query_id":        node_id,
                    "similarity":      round(sim, 4),
                    "text":            data.get("text", ""),
                    "response_summary": data.get("response_summary"),
                    "session_id":      data.get("session_id"),
                    "timestamp":       data.get("timestamp"),
                })

        results.sort(key=lambda x: -x["similarity"])
        return results[:top_n]

    def add_similar_to_edge(
        self,
        query_id_a: str,
        query_id_b: str,
        similarity: float,
    ) -> None:
        """Explicitly record a SIMILAR_TO edge (called from similar_queries.py)."""
        self._G.add_edge(
            query_id_a, query_id_b,
            relation   = "SIMILAR_TO",
            similarity = float(similarity),
        )

    # -----------------------------------------------------------------------
    # Graph-level queries
    # -----------------------------------------------------------------------

    def session_subgraph(self, session_id: str) -> nx.DiGraph:
        """
        Return the subgraph induced by this session's nodes + their chunk neighbours.
        Used by forget_gate.py to run PageRank on just the active session.
        """
        nodes = {session_id}
        for qid in self._session_query_ids(session_id):
            nodes.add(qid)
            for _, target in self._G.out_edges(qid):
                nodes.add(target)
        return self._G.subgraph(nodes).copy()

    def list_sessions(self, project_id: Optional[str] = None) -> list[dict]:
        """Summary list of all sessions, optionally filtered by project_id."""
        sessions = []
        for nid, data in self._G.nodes(data=True):
            if data.get("node_type") != "session":
                continue
            if project_id and data.get("project_id") != project_id:
                continue
            sessions.append({
                "session_id": nid,
                "project_id": data.get("project_id"),
                "start_time": data.get("start_time"),
                "end_time":   data.get("end_time"),
                "n_steps":    data.get("n_steps", 0),
                "summary":    data.get("summary"),
            })
        sessions.sort(key=lambda x: x["start_time"] or "")
        return sessions

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self) -> None:
        """Persist session graph to disk (overwrites previous file)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self._G, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = self.path.stat().st_size / 1e6
        log.info(
            "Session graph saved → %s  (%.2f MB, %d nodes, %d edges)",
            self.path, size_mb,
            self._G.number_of_nodes(),
            self._G.number_of_edges(),
        )

    # -----------------------------------------------------------------------
    # Stats / diagnostics
    # -----------------------------------------------------------------------

    def print_stats(self) -> None:
        G = self._G
        type_counts: dict[str, int] = {}
        for _, data in G.nodes(data=True):
            t = data.get("node_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        edge_counts: dict[str, int] = {}
        for _, _, data in G.edges(data=True):
            rel = data.get("relation", "unknown")
            edge_counts[rel] = edge_counts.get(rel, 0) + 1

        log.info("=" * 52)
        log.info("  Episodic store stats")
        log.info("=" * 52)
        log.info("  Node types:")
        for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            log.info("    %-20s %d", t, cnt)
        log.info("  Edge types:")
        for rel, cnt in sorted(edge_counts.items(), key=lambda x: -x[1]):
            log.info("    %-20s %d", rel, cnt)
        log.info("=" * 52)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _session_query_ids(self, session_id: str) -> list[str]:
        """All query node IDs for this session, sorted by step_number."""
        qids = [
            target
            for _, target, data in self._G.out_edges(session_id, data=True)
            if data.get("relation") == "CONTAINS"
        ]
        return sorted(qids, key=lambda q: self._G.nodes[q].get("step_number", 0))

    def _ensure_chunk_node(self, chunk_id: str) -> None:
        """Add a lightweight chunk stub if not already present (shared namespace)."""
        if not self._G.has_node(chunk_id):
            self._G.add_node(chunk_id, node_type="chunk")


# ---------------------------------------------------------------------------
# Smoke test (run directly)
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import tempfile, os
    tmp = Path(tempfile.mktemp(suffix=".pkl"))

    store = EpisodicStore(path=tmp, embed_queries=False)

    sid = store.start_session(project_id="test")

    q1 = store.record_query(
        sid, text="How do I define a FastAPI route?", step_number=1,
        embedding=np.random.randn(384).astype("float32"),
        response_summary="Use @app.get('/path') decorator.",
    )
    store.record_retrieved(q1, chunk_id="chunk_001", score=0.91, step=1)
    store.record_retrieved(q1, chunk_id="chunk_002", score=0.84, step=1)

    q2 = store.record_query(
        sid, text="What is dependency injection in FastAPI?", step_number=2,
        embedding=np.random.randn(384).astype("float32"),
    )
    store.record_retrieved(q2, chunk_id="chunk_003", score=0.88, step=2)
    store.record_evicted(q2, chunk_id="chunk_001", step=2, strategy="pagerank")

    store.end_session(sid, summary="Covered FastAPI routing and DI basics.")
    store.print_stats()

    # Test context state
    active  = store.get_retrieved_chunks(sid)
    evicted = store.get_evicted_chunks(sid)
    assert "chunk_001" not in active,  "chunk_001 should be evicted"
    assert "chunk_002" in active,      "chunk_002 should be active"
    assert "chunk_003" in active,      "chunk_003 should be active"
    assert "chunk_001" in evicted,     "chunk_001 should appear in evicted set"

    # Test subgraph
    sg = store.session_subgraph(sid)
    assert sid in sg.nodes
    assert q1  in sg.nodes
    assert q2  in sg.nodes

    # Test session listing
    sessions = store.list_sessions(project_id="test")
    assert len(sessions) == 1
    assert sessions[0]["n_steps"] == 2

    # Test similar queries (random embeddings — no real similarity expected)
    q_emb = np.random.randn(384).astype("float32")
    hits = store.find_similar_queries(q_emb, threshold=-1.0, top_n=10)
    assert len(hits) == 2   # q1 and q2 (threshold=-1 guarantees all match)

    # Test persistence round-trip
    store.save()
    store2 = EpisodicStore(path=tmp, embed_queries=False)
    assert store2._G.number_of_nodes() == store._G.number_of_nodes()
    assert store2._G.number_of_edges() == store._G.number_of_edges()

    os.unlink(tmp)
    log.info("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
