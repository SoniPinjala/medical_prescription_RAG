"""BM25 keyword-search baseline - the classic SPARSE retrieval method.

BM25 scores a document by how many query terms it contains, weighted by:
  - IDF: rare terms count for more than common ones ("aphasia" > "the")
  - term frequency saturation: the 5th mention of a word adds less than the 2nd
  - length normalisation: long documents don't win just by being long

No embeddings, no neural network, no training. This is the baseline that dense vector
search has to BEAT to justify its extra cost and complexity.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric words. BM25 matches these literally."""
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _load():
    """Build the BM25 index in memory (~0.5s for 43k chunks), then cache it."""
    chunks = [json.loads(l) for l in CHUNKS_FILE.open(encoding="utf-8")]
    corpus = [tokenize(c["chunk"]) for c in chunks]
    return chunks, BM25Okapi(corpus)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return the k highest-BM25-scoring chunks. Same signature as rag.retrieve()."""
    chunks, bm25 = _load()
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(scores)[::-1][:k]          # highest scores first
    return [dict(chunks[i], score=float(scores[i])) for i in top]


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What are the symptoms of a panic attack?"
    print(f"Query: {query}\n")
    for rank, c in enumerate(retrieve(query), start=1):
        print(f"[{rank}] score={c['score']:.2f}  source={c['source']}")
        print(f"    Q: {c['question']}")
        print(f"    A: {c['chunk'][:150]}...\n")