"""
Chunking logic using CodeBERT's tokenizer.

Strategy: sliding window — 512 tokens, 64-token stride.
Uses offset_mapping to extract exact substrings from the original text,
so decoded chunks are byte-faithful to the source (no BPE decode artifacts).
"""

import logging
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

CHUNK_SIZE  = 512   # CodeBERT max sequence length
STRIDE      = 64    # overlap = CHUNK_SIZE - STRIDE = 448 tokens
MIN_TOKENS  = 50    # discard chunks shorter than this

TOKENIZER_NAME = "microsoft/codebert-base"


def load_tokenizer() -> AutoTokenizer:
    log.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    log.info("Tokenizer ready.")
    return tokenizer


def chunk_text(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_size: int = CHUNK_SIZE,
    stride: int = STRIDE,
    min_tokens: int = MIN_TOKENS,
) -> list[tuple[str, int]]:
    """
    Split text into overlapping chunks of at most `chunk_size` tokens.

    Returns a list of (chunk_text, token_count) tuples.
    Uses offset_mapping to extract substrings from the original text directly,
    avoiding BPE decode artifacts at chunk boundaries.
    """
    if not text or not text.strip():
        return []

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    ids     = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n       = len(ids)

    if n < min_tokens:
        return []

    # Short text fits in one chunk
    if n <= chunk_size:
        return [(text.strip(), n)]

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        window_len = end - start

        if window_len < min_tokens:
            break

        char_start = offsets[start][0]
        char_end   = offsets[end - 1][1]
        chunk_str  = text[char_start:char_end].strip()

        if chunk_str:
            chunks.append((chunk_str, window_len))

        if end >= n:
            break

        start += chunk_size - stride

    return chunks
