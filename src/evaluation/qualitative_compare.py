"""
Qualitative response comparison — Flat RAG vs Selective RAG.
Re-runs specific queries and prints full responses side by side.
"""

import json
import os
import textwrap
from pathlib import Path

import openai

from src.retrieval.retriever import Retriever
from src.agent.context_manager import ContextRegistry
from src.evaluation.embedding_ablation import MiniLMEmbedder, MiniLMRoutingRetriever
from src.evaluation.harness import _build_flat_prompt

ROOT    = Path(__file__).parents[2]
SO_PATH = ROOT / "data" / "raw" / "stackoverflow" / "stackoverflow_all_raw.json"

_OAI_MODEL = "gpt-4o-mini"

QUERY_IDS = [
    70710874,   # selective wins (flat=0, sel=1.0)  — base64 image + FastAPI
    60238433,   # flat wins    (flat=1.0, sel=0)    — pydantic models/schemas in FastAPI
    67569529,   # both high    (flat=1.0, sel=1.0)  — Optional field with description
]


def load_record(qid: int) -> dict:
    with open(SO_PATH) as f:
        data = json.load(f)
    for r in data:
        if r.get("question_id") == qid:
            return r
    raise ValueError(f"question_id {qid} not found")


def call_llm(client, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=_OAI_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def wrap(text: str, width: int = 90, indent: str = "  ") -> str:
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para, width=width, initial_indent=indent,
                                       subsequent_indent=indent))
        else:
            lines.append("")
    return "\n".join(lines)


def run(query_ids: list[int]):
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print("Loading retriever + MiniLM routing...")
    base = Retriever(top_clusters=2, top_k=5)
    minilm = MiniLMEmbedder()
    retriever = MiniLMRoutingRetriever(base, minilm, top_clusters=2)
    registry = ContextRegistry(token_budget=8192)

    for qid in query_ids:
        record = load_record(qid)
        query = record["title"]
        gt    = record.get("answer_body", "")[:600]
        tag   = record.get("tag", "")

        print("\n" + "=" * 100)
        print(f"  QUERY [{tag}]  (id={qid})")
        print(f"  {query}")
        print("=" * 100)
        print(f"\n  GROUND TRUTH (excerpt):\n{wrap(gt)}\n")

        # --- Flat RAG ---
        flat_chunks = retriever.retrieve_flat(query, top_k=5)
        flat_prompt = _build_flat_prompt(query, flat_chunks)
        flat_resp   = call_llm(client, flat_prompt)

        # --- Selective RAG ---
        registry.evict_all()
        registry.advance_step()
        sel_chunks = retriever.retrieve(query, top_k=5, top_clusters=2)
        registry.load_chunks(sel_chunks)
        sel_prompt = registry.build_prompt(query)
        sel_resp   = call_llm(client, sel_prompt)
        registry.evict_all()

        print("  ── FLAT RAG RESPONSE ──────────────────────────────────────────────────────────")
        print(wrap(flat_resp))
        print()
        print("  ── SELECTIVE RAG RESPONSE ─────────────────────────────────────────────────────")
        print(wrap(sel_resp))
        print()

        # Chunk sources summary
        flat_topics  = sorted(set(c.get("topic_id") for c in flat_chunks))
        sel_topics   = sorted(set(c.get("topic_id") for c in sel_chunks))
        print(f"  Flat chunks from topics:     {flat_topics}")
        print(f"  Selective chunks from topics: {sel_topics}")
        print(f"  Overlap: {set(flat_topics) & set(sel_topics)}")


if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(level=logging.WARNING)
    run(QUERY_IDS)
