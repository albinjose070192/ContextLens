#!/usr/bin/env python3
"""
Preprocessing pipeline: Clean → Chunk → Dedup → Tag → Save JSONL.

Input:  data/raw/github/*.json  +  data/raw/stackoverflow/*.json
Output: data/processed/corpus_all.jsonl
        data/processed/corpus_stats.json

Usage:
    python pipeline.py
"""

import json
import hashlib
import logging
from collections import Counter, defaultdict
from pathlib import Path

from clean import extract_text
from chunk import load_tokenizer, chunk_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).parents[2]
RAW_GITHUB   = ROOT / "data" / "raw" / "github"
RAW_SO       = ROOT / "data" / "raw" / "stackoverflow" / "stackoverflow_all_raw.json"
OUT_DIR      = ROOT / "data" / "processed"
OUT_CORPUS   = OUT_DIR / "corpus_all.jsonl"
OUT_STATS    = OUT_DIR / "corpus_stats.json"

MIN_CLEAN_CHARS = 50   # drop artifact if cleaned text is shorter than this


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """SHA-256 of lowercased, whitespace-normalised text."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def process_artifacts(
    artifacts: list[dict],
    tokenizer,
    seen_hashes: set,
    source_label: str,
) -> list[dict]:
    """
    Clean, chunk, dedup, and tag a list of raw artifacts.
    Modifies `seen_hashes` in place for cross-file dedup.
    """
    chunks_out = []
    skipped_short  = 0
    skipped_dedup  = 0

    for artifact in artifacts:
        text = extract_text(artifact)

        if not text or len(text) < MIN_CLEAN_CHARS:
            skipped_short += 1
            continue

        windows = chunk_text(text, tokenizer)
        total   = len(windows)

        for idx, (chunk_str, token_count) in enumerate(windows):
            h = content_hash(chunk_str)
            if h in seen_hashes:
                skipped_dedup += 1
                continue
            seen_hashes.add(h)

            chunks_out.append({
                "chunk_id":     h[:16],
                "text":         chunk_str,
                "source_type":  artifact.get("source_type", "unknown"),
                "repo":         artifact.get("repo", source_label),
                "url":          artifact.get("url", ""),
                "created_at":   artifact.get("created_at", ""),
                "token_count":  token_count,
                "chunk_index":  idx,
                "total_chunks": total,
            })

    log.info(
        f"  {source_label}: {len(artifacts)} artifacts → "
        f"{len(chunks_out)} chunks "
        f"(skipped short={skipped_short}, dedup={skipped_dedup})"
    )
    return chunks_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer   = load_tokenizer()
    seen_hashes: set[str] = set()
    all_chunks:  list[dict] = []

    # ── GitHub repos ────────────────────────────────────────────────────────
    github_files = sorted(RAW_GITHUB.glob("*_raw.json"))
    log.info(f"Found {len(github_files)} GitHub raw files.")

    for path in github_files:
        log.info(f"Processing {path.name} ...")
        with open(path, encoding="utf-8") as f:
            artifacts = json.load(f)
        chunks = process_artifacts(artifacts, tokenizer, seen_hashes, path.stem)
        all_chunks.extend(chunks)

    # ── Stack Overflow ───────────────────────────────────────────────────────
    if RAW_SO.exists():
        log.info(f"Processing {RAW_SO.name} ...")
        with open(RAW_SO, encoding="utf-8") as f:
            so_artifacts = json.load(f)
        chunks = process_artifacts(so_artifacts, tokenizer, seen_hashes, "stackoverflow")
        all_chunks.extend(chunks)
    else:
        log.warning(f"Stack Overflow file not found: {RAW_SO}")

    # ── Save JSONL ───────────────────────────────────────────────────────────
    log.info(f"Writing {len(all_chunks)} chunks to {OUT_CORPUS} ...")
    with open(OUT_CORPUS, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── Stats ────────────────────────────────────────────────────────────────
    by_source  = Counter(c["source_type"] for c in all_chunks)
    by_repo    = Counter(c["repo"] for c in all_chunks)
    token_counts = [c["token_count"] for c in all_chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    # Per-repo × source_type breakdown
    repo_source: dict[str, Counter] = defaultdict(Counter)
    for c in all_chunks:
        repo_source[c["repo"]][c["source_type"]] += 1

    stats = {
        "total_chunks":        len(all_chunks),
        "avg_token_count":     round(avg_tokens, 1),
        "min_token_count":     min(token_counts) if token_counts else 0,
        "max_token_count":     max(token_counts) if token_counts else 0,
        "by_source_type":      dict(by_source.most_common()),
        "by_repo":             dict(by_repo.most_common()),
        "by_repo_source_type": {
            repo: dict(counter.most_common())
            for repo, counter in sorted(repo_source.items())
        },
    }

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n=== Preprocessing complete ===")
    log.info(f"  Total chunks : {stats['total_chunks']:,}")
    log.info(f"  Avg tokens   : {stats['avg_token_count']}")
    log.info(f"  Output       : {OUT_CORPUS}")
    log.info(f"  Stats        : {OUT_STATS}")
    log.info("\n  By source type:")
    for src, count in by_source.most_common():
        log.info(f"    {src:<20} {count:>6,}")
    log.info("\n  By repo:")
    for repo, count in by_repo.most_common():
        log.info(f"    {repo:<40} {count:>6,}")


if __name__ == "__main__":
    main()
