"""Embed every chunk and store the vectors in a FAISS HNSW index for fast search."""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
INDEX_FILE = Path("data/processed/faiss.index")

# 1. Load the chunks. Their order here == their position in the index later.
chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
texts = [c["chunk"] for c in chunks]
print(f"Embedding {len(texts)} chunks ...")

# 2. Turn text into vectors. normalize=True makes them unit-length so that
#    inner product == cosine similarity (the standard measure for text search).
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
).astype("float32")   # FAISS requires float32

dim = embeddings.shape[1]   # 384 for this model
print(f"Got {embeddings.shape[0]} vectors of dimension {dim}")

# 3. Build an HNSW (ANN) index. METRIC_INNER_PRODUCT + normalized vectors = cosine.
#    M=32 is the graph connectivity; higher = more accurate but bigger/slower to build.
index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
index.add(embeddings)
print(f"Index now holds {index.ntotal} vectors")

# 4. Save the index to disk. The chunks.jsonl is our metadata: row i <-> vector i.
faiss.write_index(index, str(INDEX_FILE))
print(f"Saved index to {INDEX_FILE}")