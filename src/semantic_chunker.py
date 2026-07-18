"""Semantic chunking using Gemini embeddings.

Splits text into sentences, embeds them, and groups semantically similar
sentences into chunks. Splits where cosine similarity drops below threshold.
"""
import re
import logging
import math
from dataclasses import dataclass

logger = logging.getLogger("rag.chunker")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class TextChunk:
    text: str
    sentences: list[str]
    start_idx: int
    end_idx: int


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    text = re.sub(r"\s+", " ", text.strip())
    raw = _SENTENCE_RE.split(text)
    return [s.strip() for s in raw if len(s.strip()) > 5]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _find_breakpoints(similarities: list[float], threshold: float) -> list[int]:
    """Return indices where similarity drops below threshold (topic boundaries)."""
    return [i + 1 for i, sim in enumerate(similarities) if sim < threshold]


def _group_sentences(sentences: list[str], breakpoints: list[int], max_chars: int) -> list[TextChunk]:
    """Group sentences into chunks based on breakpoints."""
    chunks = []
    start = 0

    all_breaks = sorted(set([0] + breakpoints + [len(sentences)]))

    for i in range(len(all_breaks) - 1):
        chunk_start = all_breaks[i]
        chunk_end = all_breaks[i + 1]
        chunk_sentences = sentences[chunk_start:chunk_end]
        text = " ".join(chunk_sentences)

        if len(text.strip()) < 10:
            continue

        # If chunk is too large, split further by max_chars
        if len(text) > max_chars:
            sub_chunks = _split_large_chunk(chunk_sentences, max_chars)
            for sc in sub_chunks:
                chunks.append(sc)
        else:
            chunks.append(TextChunk(
                text=text,
                sentences=chunk_sentences,
                start_idx=chunk_start,
                end_idx=chunk_end,
            ))

    return chunks


def _split_large_chunk(sentences: list[str], max_chars: int) -> list[TextChunk]:
    """Split a large group of sentences into smaller chunks by character limit."""
    chunks = []
    current = []
    current_len = 0
    start_idx = 0

    for i, sent in enumerate(sentences):
        if current_len + len(sent) > max_chars and current:
            text = " ".join(current)
            chunks.append(TextChunk(
                text=text,
                sentences=list(current),
                start_idx=start_idx,
                end_idx=start_idx + len(current),
            ))
            start_idx = start_idx + len(current)
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(sent)

    if current:
        text = " ".join(current)
        chunks.append(TextChunk(
            text=text,
            sentences=current,
            start_idx=start_idx,
            end_idx=start_idx + len(current),
        ))

    return chunks


def semantic_chunk(
    text: str,
    embed_fn,
    threshold: float = 0.45,
    max_chunk_chars: int = 1500,
    min_chunk_chars: int = 100,
) -> list[TextChunk]:
    """Split text into semantically coherent chunks.

    Args:
        text: Raw document text.
        embed_fn: Callable that takes a list of strings and returns list of embeddings.
        threshold: Cosine similarity threshold below which to split.
        max_chunk_chars: Maximum characters per chunk.
        min_chunk_chars: Minimum characters per chunk (merge if smaller).
    """
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return [TextChunk(text=text, sentences=sentences, start_idx=0, end_idx=len(sentences))]

    # Embed all sentences
    logger.info("Embedding %d sentences for semantic chunking...", len(sentences))
    embeddings = embed_fn(sentences)

    # Compute pairwise cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Find breakpoints where similarity drops
    breakpoints = _find_breakpoints(similarities, threshold)
    logger.info("Found %d breakpoints (threshold=%.2f)", len(breakpoints), threshold)

    # Group sentences into chunks
    chunks = _group_sentences(sentences, breakpoints, max_chunk_chars)

    # Merge tiny chunks
    if len(chunks) > 1:
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(merged[-1].text) < min_chunk_chars:
                merged[-1].text += " " + chunk.text
                merged[-1].sentences.extend(chunk.sentences)
                merged[-1].end_idx = chunk.end_idx
            else:
                merged.append(chunk)
        chunks = merged

    # Final filter
    chunks = [c for c in chunks if len(c.text.strip()) >= 20]

    logger.info("Produced %d semantic chunks from %d sentences", len(chunks), len(sentences))
    return chunks


def semantic_chunk_with_parents(
    text: str,
    embed_fn,
    threshold: float = 0.45,
    parent_max_chars: int = 2000,
    child_max_chars: int = 500,
    min_chunk_chars: int = 100,
) -> list[tuple[str, str]]:
    """Produce (child_text, parent_text) pairs for parent-child indexing.

    1. Semantic chunking at parent level (larger, coherent topics).
    2. Sub-chunk parents into children by character limit.
    """
    parent_chunks = semantic_chunk(
        text,
        embed_fn,
        threshold=threshold,
        max_chunk_chars=parent_max_chars,
        min_chunk_chars=min_chunk_chars,
    )

    results = []
    for parent in parent_chunks:
        parent_text = parent.text
        # Sub-chunk parent into children
        child_chunks = _split_large_chunk(parent.sentences, child_max_chars)
        for child in child_chunks:
            results.append((child.text, parent_text))

    return results
