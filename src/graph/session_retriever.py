"""
session_retriever.py — Session-aware retrieval pipeline.

Replaces the single-pass MiniLMRoutingRetriever for stateful multi-step queries.

Pipeline (per query step)
--------------------------
1. Route      MiniLM centroids → top-K clusters
2. Session    check evicted chunks + push-back warning for similar prior queries
3. Select     rank candidate documents per cluster with weighted score:
              score = α·max_faiss_score + β·context_frac + γ·history_score
              (α=0.6, β=0.3, γ=0.1 — from design doc)
              Pick top-M documents per cluster.
4. Traverse   anchor chunk ± traversal_depth steps via corpus graph NEXT/PREV edges.
              Synthetic neighbor score = anchor_score / (1 + distance).
              Skip evicted chunks and chunks below anchor_score * score_gate.
5. Load       dedup into ContextRegistry; record to EpisodicStore.
6. Forget     optionally trigger forget gate (by token count or step interval).

Text for traversed neighbors is resolved from the base Retriever's in-memory corpus
list via corpus_pos stored on each corpus graph node.

Usage
-----
  from src.graph.session_retriever import SessionRetriever
  from src.graph.episodic_store import EpisodicStore
  from src.retrieval.retriever import Retriever
  from src.evaluation.embedding_ablation import MiniLMEmbedder
  from src.agent.context_manager import ContextRegistry
  import pickle

  with open("data/models/corpus_graph.pkl", "rb") as f:
      corpus_artifact = pickle.load(f)

  base     = Retriever(top_clusters=2, top_k=10)
  minilm   = MiniLMEmbedder()
  store    = EpisodicStore()
  registry = ContextRegistry(token_budget=8192)
  sid      = store.start_session(project_id="msds453")

  sr = SessionRetriever(
      base_retriever  = base,
      minilm          = minilm,
      store           = store,
      corpus_artifact = corpus_artifact,
      registry        = registry,
      session_id      = sid,
  )

  chunks, qid = sr.retrieve("How do I define a FastAPI route?", step_number=1)
  # ... call LLM, then:
  store.update_response_summary(qid, "Use @app.get('/path') decorator.")
  store.save()
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]

# Subgraph document scoring weights
_ALPHA = 0.6   # max FAISS score for the document
_BETA  = 0.3   # fraction of doc's chunks already in context
_GAMMA = 0.1   # normalised prior retrieval count for the document

_TRAVERSAL_DEPTH   = 2      # ±N chunks from anchor
_ANCHOR_SCORE_GATE = 0.5    # stop traversal below this fraction of anchor score
_TOP_M_DOCS        = 3      # documents to traverse per cluster
_TOKEN_THRESHOLD   = 4_000  # trigger forget gate above this token count
_FORGET_INTERVAL   = 10     # also trigger every K steps
_PUSH_BACK_SIM     = 0.88   # cosine similarity threshold for push-back warning


class SessionRetriever:
    """
    Session-aware retrieval pipeline for stateful multi-step agent queries.

    Parameters
    ----------
    base_retriever : Retriever
        CodeBERT FAISS retriever — provides _embed_query, _search_cluster,
        and the in-memory _corpus list used for text lookup.
    minilm : MiniLMEmbedder
        MiniLM embedder for cluster routing and episodic query embeddings.
    store : EpisodicStore
        Persistent episodic graph (session state).
    corpus_artifact : dict
        Loaded corpus_graph.pkl — keys: graph, doc_to_chunks, cluster_to_chunks,
        chunk_to_pos, pos_to_chunk_id.
    registry : ContextRegistry
        Active context window manager (dedup + token budget).
    session_id : str
        Active session ID (from store.start_session()).
    top_clusters : int
        Clusters to route to per query.
    top_m_docs : int
        Documents to traverse per cluster after scoring.
    traversal_depth : int
        Walk ± this many steps from the anchor chunk.
    anchor_score_gate : float
        Skip neighbors whose synthetic score < anchor_score * gate.
    alpha, beta, gamma : float
        Document scoring weights.
    forget_gate_fn : callable or None
        Signature: (store, registry, session_id) → list[str] (evicted chunk_ids).
    token_threshold : int
        Token count above which forget gate fires.
    forget_gate_interval : int
        Also fire forget gate every K steps.
    push_back_threshold : float
        Cosine similarity threshold for similar-query warning.
    """

    def __init__(
        self,
        base_retriever,
        minilm,
        store,
        corpus_artifact:      dict,
        registry,
        session_id:           str,
        top_clusters:         int   = 2,
        top_m_docs:           int   = _TOP_M_DOCS,
        traversal_depth:      int   = _TRAVERSAL_DEPTH,
        anchor_score_gate:    float = _ANCHOR_SCORE_GATE,
        alpha:                float = _ALPHA,
        beta:                 float = _BETA,
        gamma:                float = _GAMMA,
        forget_gate_fn:       Optional[Callable] = None,
        token_threshold:      int   = _TOKEN_THRESHOLD,
        forget_gate_interval: int   = _FORGET_INTERVAL,
        push_back_threshold:  float = _PUSH_BACK_SIM,
    ):
        self._base              = base_retriever
        self._minilm            = minilm
        self._store             = store
        self._corpus_graph: nx.DiGraph = corpus_artifact["graph"]
        self._doc_to_chunks     = corpus_artifact["doc_to_chunks"]
        self._cluster_to_chunks = corpus_artifact["cluster_to_chunks"]
        self._registry          = registry
        self._session_id        = session_id

        self.top_clusters         = top_clusters
        self.top_m_docs           = top_m_docs
        self.traversal_depth      = traversal_depth
        self.anchor_score_gate    = anchor_score_gate
        self.alpha                = alpha
        self.beta                 = beta
        self.gamma                = gamma
        self._forget_gate_fn      = forget_gate_fn
        self.token_threshold      = token_threshold
        self.forget_gate_interval = forget_gate_interval
        self.push_back_threshold  = push_back_threshold

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query:       str,
        step_number: int,
    ) -> tuple[list[dict], str]:
        """
        Run the full session-aware retrieval pipeline for one query step.

        Returns
        -------
        (chunks, query_id)
          chunks    — chunk dicts newly loaded into the ContextRegistry
          query_id  — call store.update_response_summary(qid, ...) after LLM responds
        """
        # ── Step 1: embed + route ──────────────────────────────────────────
        minilm_emb  = self._minilm.embed(query)
        cluster_ids = self._minilm.route(query, self.top_clusters)
        log.info("Step %d | routed → clusters %s", step_number, cluster_ids)

        # ── Step 2: session state ──────────────────────────────────────────
        evicted = self._store.get_evicted_chunks(self._session_id)
        self._push_back_check(minilm_emb)

        # ── Steps 3 + 4: FAISS → document scoring → traversal ─────────────
        candidates = self._search_and_traverse(cluster_ids, query, evicted)

        # ── Step 5: record query + load into registry ──────────────────────
        qid = self._store.record_query(
            self._session_id,
            text        = query,
            step_number = step_number,
            embedding   = minilm_emb,
        )

        loaded: list[dict] = []
        for chunk in candidates:
            if self._registry.load(chunk):
                loaded.append(chunk)
                self._store.record_retrieved(
                    qid,
                    chunk_id = chunk["chunk_id"],
                    score    = float(chunk.get("score", 0.0)),
                    step     = step_number,
                )

        log.info(
            "Step %d | %d candidates → %d newly loaded (registry size=%d, tokens=%d)",
            step_number,
            len(candidates),
            len(loaded),
            self._registry.size,
            self._registry.token_count(),
        )

        # ── Step 6: optional forget gate ──────────────────────────────────
        self._maybe_forget(qid, step_number)

        return loaded, qid

    # -----------------------------------------------------------------------
    # Step 2: push-back
    # -----------------------------------------------------------------------

    def _push_back_check(self, embedding: np.ndarray) -> None:
        """Warn if a very similar query was asked in a prior step."""
        session_data = self._store._G.nodes.get(self._session_id, {})
        project_id   = session_data.get("project_id")

        hits = self._store.find_similar_queries(
            embedding,
            threshold  = self.push_back_threshold,
            project_id = project_id,
        )
        if hits:
            best = hits[0]
            log.warning(
                "Push-back: similar prior query (sim=%.3f) — \"%s\"\n"
                "           Response summary: %s",
                best["similarity"],
                best["text"][:120],
                best.get("response_summary") or "(none recorded)",
            )

    # -----------------------------------------------------------------------
    # Steps 3 + 4: FAISS search → document scoring → traversal
    # -----------------------------------------------------------------------

    def _search_and_traverse(
        self,
        cluster_ids: list[int],
        query:       str,
        evicted:     set[str],
    ) -> list[dict]:
        """
        For each routed cluster:
          1. FAISS search → hits grouped by document (url)
          2. Score each document → select top-M
          3. Traverse ±depth from anchor in corpus graph

        Returns deduplicated flat list of chunk dicts with score fields.
        """
        cb_emb       = self._base._embed_query(query)
        registry_ids = self._registry.loaded_ids
        session_scores = self._store.get_session_chunk_scores(self._session_id)

        all_chunks: list[dict] = []

        for cluster_id in cluster_ids:
            # FAISS search — fetch more than top_m_docs to have doc diversity
            hits = self._base._search_cluster(
                cluster_id, cb_emb, k=self.top_m_docs * 6
            )
            if not hits:
                continue

            # Group hits by document URL
            doc_hits: dict[str, list[dict]] = defaultdict(list)
            for h in hits:
                doc_hits[h["url"]].append(h)

            # Score each document
            doc_scores: list[tuple[str, float]] = []
            for url, doc_chunks in doc_hits.items():
                score = self._score_document(
                    url, doc_chunks, registry_ids, session_scores
                )
                doc_scores.append((url, score))

            doc_scores.sort(key=lambda x: -x[1])
            top_docs = [url for url, _ in doc_scores[: self.top_m_docs]]

            for url in top_docs:
                traversed = self._traverse_document(url, doc_hits[url], evicted)
                all_chunks.extend(traversed)

        # Deduplicate by chunk_id, keep highest score
        seen: dict[str, dict] = {}
        for chunk in all_chunks:
            cid = chunk["chunk_id"]
            if cid not in seen or chunk.get("score", 0.0) > seen[cid].get("score", 0.0):
                seen[cid] = chunk

        return list(seen.values())

    def _score_document(
        self,
        url:            str,
        doc_chunks:     list[dict],
        registry_ids:   set[str],
        session_scores: dict[str, float],
    ) -> float:
        """
        Weighted document relevance score:
          α · max_faiss_score
          β · context_frac        (fraction of doc's chunks already in registry)
          γ · history_score       (normalised prior retrieval count for this doc)
        """
        max_faiss = max(c["score"] for c in doc_chunks)

        all_doc_cids = set(self._doc_to_chunks.get(url, []))
        n_doc        = max(len(all_doc_cids), 1)

        in_context   = len(all_doc_cids & registry_ids)
        context_frac = in_context / n_doc

        prior_count  = sum(1 for cid in all_doc_cids if cid in session_scores)
        history_score = min(prior_count / n_doc, 1.0)

        return (
            self.alpha * max_faiss
            + self.beta  * context_frac
            + self.gamma * history_score
        )

    def _traverse_document(
        self,
        url:        str,
        faiss_hits: list[dict],
        evicted:    set[str],
    ) -> list[dict]:
        """
        Anchor on the highest-scoring FAISS hit in the document, then walk
        ± traversal_depth steps via the sequential chunk ordering.

        Neighbor score = anchor_score / (1 + distance).
        Chunks below anchor_score * score_gate are skipped.
        Text for non-FAISS-hit neighbors is resolved from base._corpus via corpus_pos.

        Returns list of chunk dicts with score fields.
        """
        anchor       = max(faiss_hits, key=lambda c: c["score"])
        anchor_id    = anchor["chunk_id"]
        anchor_score = anchor["score"]
        score_cutoff = anchor_score * self.anchor_score_gate

        faiss_by_id  = {c["chunk_id"]: c for c in faiss_hits}
        ordered_cids = self._doc_to_chunks.get(url, [])

        if not ordered_cids:
            return [anchor] if anchor_id not in evicted else []

        try:
            anchor_pos = ordered_cids.index(anchor_id)
        except ValueError:
            return [anchor] if anchor_id not in evicted else []

        result: list[dict] = []
        lo = max(0, anchor_pos - self.traversal_depth)
        hi = min(len(ordered_cids), anchor_pos + self.traversal_depth + 1)

        for idx in range(lo, hi):
            cid = ordered_cids[idx]

            if cid in evicted:
                continue

            distance       = abs(idx - anchor_pos)
            neighbor_score = anchor_score / (1.0 + distance)

            if neighbor_score < score_cutoff:
                continue

            # Real FAISS score takes priority over synthetic score
            if cid in faiss_by_id:
                chunk = dict(faiss_by_id[cid])
            else:
                chunk = self._lookup_chunk(cid, neighbor_score)

            if chunk is not None:
                chunk["score"] = max(chunk.get("score", 0.0), neighbor_score)
                result.append(chunk)

        return result

    def _lookup_chunk(self, chunk_id: str, score: float) -> Optional[dict]:
        """
        Resolve a chunk by ID using corpus_pos from the corpus graph node,
        then index into the base retriever's in-memory corpus list.
        Returns None if the node or position is not found.
        """
        if not self._corpus_graph.has_node(chunk_id):
            return None

        node_attrs = self._corpus_graph.nodes[chunk_id]
        corpus_pos = node_attrs.get("corpus_pos")
        if corpus_pos is None:
            return None

        try:
            chunk = dict(self._base._corpus[corpus_pos])
        except IndexError:
            return None

        chunk["score"]    = score
        chunk["topic_id"] = node_attrs.get("topic_id", chunk.get("topic_id", -1))
        return chunk

    # -----------------------------------------------------------------------
    # Step 6: forget gate
    # -----------------------------------------------------------------------

    def _maybe_forget(self, query_id: str, step_number: int) -> None:
        """Trigger the forget gate if token budget or step interval is hit."""
        if self._forget_gate_fn is None:
            return

        tokens      = self._registry.token_count()
        over_budget = tokens > self.token_threshold
        at_interval = step_number > 0 and (step_number % self.forget_gate_interval == 0)

        if not (over_budget or at_interval):
            return

        log.info(
            "Forget gate triggered at step %d "
            "(tokens=%d, over_budget=%s, at_interval=%s)",
            step_number, tokens, over_budget, at_interval,
        )
        evicted_ids = self._forget_gate_fn(
            self._store, self._registry, self._session_id
        )
        for cid in evicted_ids:
            self._store.record_evicted(
                query_id, cid, step=step_number, strategy="pagerank"
            )
        log.info("Forget gate evicted %d chunks.", len(evicted_ids))
