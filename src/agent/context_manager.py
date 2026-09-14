"""
Layer 3 — Selective Context Injection.

ContextRegistry manages what lives in the LLM's context window across agent
steps.  Rather than accumulating all retrieved chunks indefinitely, it:

  1. Loads only chunks retrieved for the current step.
  2. Deduplicates by content hash — same text under a different chunk_id is
     not reloaded.
  3. Evicts chunks that are no longer referenced after each step, keeping the
     context window bounded.

Typical per-step lifecycle
--------------------------
  registry.advance_step()
  registry.load_chunks(retriever.retrieve(query))
  prompt = registry.build_prompt(query)
  response = llm(prompt)
  cited_ids = registry.parse_citations(response)   # or supply manually
  registry.evict_unreferenced(cited_ids)
  tokens_used = registry.token_count()
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

log = logging.getLogger(__name__)

# Tiktoken encoding — cl100k_base matches GPT-4 / text-embedding-3 family and
# is a reasonable proxy for any modern transformer.
_ENCODING = "cl100k_base"


# ---------------------------------------------------------------------------
# Data class for a single slot in the registry
# ---------------------------------------------------------------------------

@dataclass
class _Entry:
    chunk: dict          # full chunk dict from the retriever
    content_hash: str    # SHA-256 of stripped text
    step_loaded: int     # agent step at which this chunk was first loaded


# ---------------------------------------------------------------------------
# ContextRegistry
# ---------------------------------------------------------------------------

class ContextRegistry:
    """
    Tracks the active context window for one agent session.

    Parameters
    ----------
    token_budget : int
        Soft cap on context window tokens.  load_chunks() will warn (but not
        hard-fail) if loading a batch would exceed this limit.
    encoding : str
        tiktoken encoding name used for token counting.
    """

    def __init__(self, token_budget: int = 8192, encoding: str = _ENCODING):
        self._budget: int = token_budget
        self._enc = tiktoken.get_encoding(encoding)

        # chunk_id → _Entry
        self._entries: dict[str, _Entry] = {}
        # content_hash → chunk_id  (dedup index)
        self._hash_index: dict[str, str] = {}

        self._step: int = 0

        # Per-step telemetry — list of dicts, one per step
        self.history: list[dict] = []

    # -----------------------------------------------------------------------
    # Step lifecycle
    # -----------------------------------------------------------------------

    def advance_step(self) -> None:
        """Increment the step counter.  Call once at the start of each agent turn."""
        self._step += 1
        log.debug(f"ContextRegistry: step → {self._step}")

    @property
    def current_step(self) -> int:
        return self._step

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def load(self, chunk: dict) -> bool:
        """
        Load one chunk dict (as returned by Retriever.retrieve).

        Returns True if the chunk was newly added, False if it was a duplicate
        (same chunk_id or same content hash).
        """
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            log.warning("Chunk missing chunk_id — skipping.")
            return False

        # Already present by id
        if chunk_id in self._entries:
            return False

        text = chunk.get("text", "")
        content_hash = _sha256(text)

        # Duplicate content under a different id
        if content_hash in self._hash_index:
            log.debug(
                f"Duplicate content for {chunk_id} "
                f"(matches {self._hash_index[content_hash]}) — skipping."
            )
            return False

        self._entries[chunk_id] = _Entry(
            chunk=chunk,
            content_hash=content_hash,
            step_loaded=self._step,
        )
        self._hash_index[content_hash] = chunk_id
        return True

    def load_chunks(self, chunks: list[dict]) -> dict:
        """
        Load a batch of chunk dicts from the retriever.

        Returns
        -------
        dict with keys:
          loaded            — chunks newly added
          skipped_duplicate — chunks rejected (duplicate id or content)
          token_count       — total tokens in registry after loading
          over_budget       — True if token_count exceeds the configured budget
        """
        loaded = 0
        skipped = 0
        for chunk in chunks:
            if self.load(chunk):
                loaded += 1
            else:
                skipped += 1

        tc = self.token_count()
        over = tc > self._budget
        if over:
            log.warning(
                f"Context window exceeds budget: {tc} > {self._budget} tokens "
                f"({len(self._entries)} chunks loaded)."
            )

        result = {
            "loaded": loaded,
            "skipped_duplicate": skipped,
            "token_count": tc,
            "over_budget": over,
        }
        log.debug(f"load_chunks: {result}")
        return result

    # -----------------------------------------------------------------------
    # Eviction
    # -----------------------------------------------------------------------

    def evict_unreferenced(self, referenced_ids: set[str]) -> int:
        """
        Remove every chunk whose chunk_id is NOT in referenced_ids.

        Call this after the LLM responds, passing the set of chunk IDs the
        model cited (or the IDs you want to keep for the next step).

        Returns the number of chunks evicted.
        """
        to_evict = [cid for cid in self._entries if cid not in referenced_ids]
        for cid in to_evict:
            h = self._entries[cid].content_hash
            self._hash_index.pop(h, None)
            del self._entries[cid]

        if to_evict:
            log.debug(f"Evicted {len(to_evict)} unreferenced chunks.")
        return len(to_evict)

    def evict_older_than(self, n_steps: int) -> int:
        """
        Evict chunks loaded more than n_steps ago.

        Useful when the LLM does not emit explicit citations: chunks from
        step t are dropped at step t + n_steps.

        Returns the number of chunks evicted.
        """
        cutoff = self._step - n_steps
        to_evict = [
            cid for cid, e in self._entries.items() if e.step_loaded < cutoff
        ]
        for cid in to_evict:
            h = self._entries[cid].content_hash
            self._hash_index.pop(h, None)
            del self._entries[cid]

        if to_evict:
            log.debug(f"Evicted {len(to_evict)} chunks older than {n_steps} steps.")
        return len(to_evict)

    def evict_all(self) -> None:
        """Clear the registry entirely (hard reset between sessions)."""
        self._entries.clear()
        self._hash_index.clear()

    # -----------------------------------------------------------------------
    # Context window output
    # -----------------------------------------------------------------------

    def get_context_window(self) -> str:
        """
        Return the current context as a plain string suitable for injection
        into an LLM prompt.

        Chunks are ordered by (step_loaded ASC, score DESC) so that the most
        recently retrieved, highest-scoring chunks appear last (recency bias).
        """
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (e.step_loaded, e.chunk.get("score", 0.0)),
        )

        parts = []
        for e in sorted_entries:
            c = e.chunk
            header = (
                f"[chunk_id={c.get('chunk_id', '?')} | "
                f"topic={c.get('topic_id', '?')} | "
                f"type={c.get('source_type', '?')} | "
                f"repo={c.get('repo', '?')} | "
                f"score={c.get('score', 0.0):.4f}]"
            )
            parts.append(f"{header}\n{c.get('text', '')}")

        return "\n\n".join(parts)

    def build_prompt(self, query: str, system: Optional[str] = None) -> str:
        """
        Convenience method: wrap the context window + query into a prompt.

        The prompt instructs the LLM to cite chunk_ids so that
        parse_citations() can determine what to keep.
        """
        ctx = self.get_context_window()
        sys_block = (system or _DEFAULT_SYSTEM).strip()
        return (
            f"{sys_block}\n\n"
            f"## Retrieved Context\n\n{ctx}\n\n"
            f"## Query\n\n{query}\n\n"
            f"When answering, cite the chunk_id(s) you relied on using the "
            f"format: [used: chunk_id_1, chunk_id_2]"
        )

    # -----------------------------------------------------------------------
    # Citation parsing
    # -----------------------------------------------------------------------

    @staticmethod
    def parse_citations(response: str) -> set[str]:
        """
        Extract chunk IDs from an LLM response that contains a citation block
        of the form:  [used: id1, id2, id3]

        Returns an empty set if no citation block is found.
        """
        match = re.search(r"\[used:\s*([^\]]+)\]", response, re.IGNORECASE)
        if not match:
            return set()
        raw = match.group(1)
        return {cid.strip() for cid in raw.split(",") if cid.strip()}

    # -----------------------------------------------------------------------
    # Telemetry
    # -----------------------------------------------------------------------

    def token_count(self) -> int:
        """Token count of the full context window (using tiktoken)."""
        return len(self._enc.encode(self.get_context_window()))

    def record_step(self, query: str, n_retrieved: int, n_evicted: int) -> dict:
        """
        Append a telemetry record for the current step.  Call after eviction.

        Returns the record dict (also appended to self.history).
        """
        record = {
            "step": self._step,
            "query": query,
            "n_chunks_in_window": len(self._entries),
            "n_retrieved": n_retrieved,
            "n_evicted": n_evicted,
            "token_count": self.token_count(),
        }
        self.history.append(record)
        return record

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    @property
    def loaded_ids(self) -> set[str]:
        return set(self._entries.keys())

    @property
    def size(self) -> int:
        return len(self._entries)

    def summary(self) -> dict:
        return {
            "step": self._step,
            "chunks_in_window": len(self._entries),
            "token_count": self.token_count(),
            "token_budget": self._budget,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


_DEFAULT_SYSTEM = (
    "You are a helpful software engineering assistant. "
    "Answer the query using only the retrieved context provided below. "
    "Be concise and precise."
)
