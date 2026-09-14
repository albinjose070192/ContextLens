"""
similar_queries.py — Cross-session similar query detection (push-back feature).

Wraps EpisodicStore.find_similar_queries() with:
  - Text-based interface (embed then search, no raw embedding required)
  - SIMILAR_TO edge recording between matched query nodes
  - Formatted push-back string ready for injection into an agent prompt

Motivation
----------
Practitioners (e.g. Sidharth Ganesh) currently work around the lack of
cross-session memory by manually writing session notes to Obsidian via a cron
job.  This module automates that: when a new query closely matches a prior one,
the system surfaces what was covered and which chunks were retrieved, offering
to reload them rather than re-retrieving cold.

Usage
-----
  from src.graph.similar_queries import SimilarQueryFinder

  finder = SimilarQueryFinder(store, minilm)
  result = finder.find("How do I use FastAPI dependency injection?")

  if result.hits:
      print(result.prompt_warning)   # inject into agent prompt
      # Optionally reload prior chunks:
      prior_chunks = finder.get_prior_chunks(result.hits[0], corpus_artifact)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.88
_DEFAULT_TOP_N     = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimilarQueryResult:
    """
    Returned by SimilarQueryFinder.find().

    Attributes
    ----------
    query_text : str
        The incoming query that was searched.
    hits : list[dict]
        Matched prior queries, sorted by similarity desc.
        Each dict: { query_id, similarity, text, response_summary,
                     session_id, timestamp }
    prompt_warning : str
        Pre-formatted string for injection into an agent prompt.
        Empty string if no hits.
    """
    query_text:     str
    hits:           list[dict] = field(default_factory=list)
    prompt_warning: str        = ""

    @property
    def has_hits(self) -> bool:
        return len(self.hits) > 0

    @property
    def best(self) -> Optional[dict]:
        return self.hits[0] if self.hits else None


# ---------------------------------------------------------------------------
# SimilarQueryFinder
# ---------------------------------------------------------------------------

class SimilarQueryFinder:
    """
    Cross-session similar query detector.

    Parameters
    ----------
    store : EpisodicStore
        Persistent episodic graph.
    minilm : MiniLMEmbedder
        For embedding incoming query text.
    threshold : float
        Cosine similarity threshold for a match (default 0.88).
    top_n : int
        Maximum number of similar queries to return.
    record_edges : bool
        If True, add SIMILAR_TO edges in the episodic graph for each match found.
        Useful for graph analysis; disable in high-frequency settings.
    """

    def __init__(
        self,
        store,
        minilm,
        threshold:    float = _DEFAULT_THRESHOLD,
        top_n:        int   = _DEFAULT_TOP_N,
        record_edges: bool  = True,
    ):
        self._store        = store
        self._minilm       = minilm
        self.threshold     = threshold
        self.top_n         = top_n
        self.record_edges  = record_edges

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def find(
        self,
        query_text:   str,
        project_id:   Optional[str] = None,
        current_qid:  Optional[str] = None,
    ) -> SimilarQueryResult:
        """
        Embed query_text and search the episodic store for similar prior queries.

        Parameters
        ----------
        query_text  : str   The incoming query.
        project_id  : str   Restrict search to this project (optional).
        current_qid : str   The current query's node ID — excluded from results
                            and used as source for SIMILAR_TO edges.

        Returns
        -------
        SimilarQueryResult
        """
        embedding = self._minilm.embed(query_text)

        hits = self._store.find_similar_queries(
            embedding,
            threshold  = self.threshold,
            top_n      = self.top_n,
            project_id = project_id,
        )

        # Exclude the current query itself (if already recorded)
        if current_qid:
            hits = [h for h in hits if h["query_id"] != current_qid]

        # Record SIMILAR_TO edges
        if self.record_edges and current_qid and hits:
            for hit in hits:
                self._store.add_similar_to_edge(
                    current_qid,
                    hit["query_id"],
                    similarity=hit["similarity"],
                )

        warning = _format_warning(query_text, hits) if hits else ""

        if hits:
            log.info(
                "Similar query found (sim=%.3f): \"%s\"",
                hits[0]["similarity"],
                hits[0]["text"][:100],
            )

        return SimilarQueryResult(
            query_text     = query_text,
            hits           = hits,
            prompt_warning = warning,
        )

    def get_prior_chunks(
        self,
        hit:             dict,
        corpus_artifact: dict,
    ) -> list[dict]:
        """
        Retrieve the corpus chunks that were loaded during a prior similar query.

        Looks up RETRIEVED edges from the matched query node and resolves full
        chunk dicts via corpus_pos in the corpus graph.

        Parameters
        ----------
        hit             : dict   One entry from SimilarQueryResult.hits.
        corpus_artifact : dict   Loaded corpus_graph.pkl.

        Returns
        -------
        list[dict]  chunk dicts (with score from original retrieval).
        """
        qid          = hit["query_id"]
        corpus_graph = corpus_artifact["graph"]
        base_corpus  = None   # resolved lazily if needed

        chunks = []
        G      = self._store._G

        if not G.has_node(qid):
            return []

        for _, chunk_id, edata in G.out_edges(qid, data=True):
            if edata.get("relation") != "RETRIEVED":
                continue
            if not corpus_graph.has_node(chunk_id):
                continue

            node_attrs = corpus_graph.nodes[chunk_id]
            corpus_pos = node_attrs.get("corpus_pos")

            if corpus_pos is not None:
                # Resolve text from corpus list if available
                if "corpus" in corpus_artifact:
                    chunk = dict(corpus_artifact["corpus"][corpus_pos])
                else:
                    # Fallback: build from graph node attributes (no text)
                    chunk = {
                        "chunk_id":    chunk_id,
                        "source_type": node_attrs.get("source_type", ""),
                        "repo":        node_attrs.get("repo", ""),
                        "url":         node_attrs.get("url", ""),
                        "chunk_index": node_attrs.get("chunk_index", 0),
                        "total_chunks":node_attrs.get("total_chunks", 0),
                        "token_count": node_attrs.get("token_count", 0),
                        "topic_id":    node_attrs.get("topic_id", -1),
                    }
            else:
                chunk = {"chunk_id": chunk_id}

            chunk["score"] = float(edata.get("weight", 0.0))
            chunks.append(chunk)

        chunks.sort(key=lambda c: -c.get("score", 0.0))
        return chunks


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_warning(query_text: str, hits: list[dict]) -> str:
    """
    Build a prompt-injectable push-back string.

    Example output:
      [SIMILAR PRIOR QUERY DETECTED]
      Similarity 0.923 — Session session_abc123 | 2026-05-17T14:32:11+00:00
      Prior query : "How do I handle FastAPI dependency injection?"
      Response    : "Use Depends() with a callable — injected at request time."

      You may be revisiting covered ground. Load prior chunks instead of
      re-retrieving, or continue to fetch fresh context.
    """
    lines = ["[SIMILAR PRIOR QUERY DETECTED]"]
    for hit in hits:
        ts      = (hit.get("timestamp") or "")[:19].replace("T", " ")
        summary = hit.get("response_summary") or "(no summary recorded)"
        lines += [
            f"  Similarity {hit['similarity']:.3f} | "
            f"Session {hit['session_id']} | {ts}",
            f"  Prior query : \"{hit['text'][:120]}\"",
            f"  Response    : \"{summary[:200]}\"",
            "",
        ]
    lines.append(
        "You may be revisiting covered ground. Consider loading prior chunks "
        "or continue to fetch fresh context."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    import sys, pickle, tempfile, os
    _root = str(Path(__file__).parents[2])
    if _root not in sys.path:
        sys.path.insert(0, _root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.graph.episodic_store import EpisodicStore
    from src.evaluation.embedding_ablation import MiniLMEmbedder

    tmp   = Path(tempfile.mktemp(suffix=".pkl"))
    store = EpisodicStore(path=tmp, embed_queries=False)
    minilm = MiniLMEmbedder()

    # Session A — prior session
    sid_a = store.start_session(project_id="test")
    q1 = store.record_query(
        sid_a, text="How do I define a route in FastAPI?", step_number=1,
        embedding=minilm.embed("How do I define a route in FastAPI?"),
        response_summary="Use @app.get('/path') on an async function.",
    )
    store.end_session(sid_a, summary="FastAPI routing basics.")

    # Session B — new session, similar query
    sid_b = store.start_session(project_id="test")
    q2 = store.record_query(
        sid_b, text="FastAPI route definition example", step_number=1,
        embedding=minilm.embed("FastAPI route definition example"),
    )

    finder = SimilarQueryFinder(store, minilm, threshold=0.5, record_edges=True)
    result = finder.find(
        "How do I create a FastAPI endpoint?",
        project_id  = "test",
        current_qid = q2,
    )

    log.info("Hits found: %d", len(result.hits))
    log.info("Has hits: %s", result.has_hits)
    if result.has_hits:
        log.info("Best match sim=%.3f: \"%s\"", result.best["similarity"], result.best["text"])
        log.info("Prompt warning:\n%s", result.prompt_warning)

    assert result.has_hits, "Expected at least one similar query hit"
    assert result.prompt_warning.startswith("[SIMILAR PRIOR QUERY DETECTED]")

    # Verify SIMILAR_TO edge was recorded
    G = store._G
    edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("relation") == "SIMILAR_TO"
    ]
    assert len(edges) > 0, "Expected SIMILAR_TO edges to be recorded"
    log.info("SIMILAR_TO edges recorded: %d", len(edges))

    store.save()
    os.unlink(tmp)
    log.info("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
