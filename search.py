"""Ask a question, get the most similar chunks back from the FAISS index."""
import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

chunks = [json.loads(line) for line in Path("data/processed/chunks.jsonl").open(encoding="utf-8")]
index = faiss.read_index("data/processed/faiss.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = sys.argv[1] if len(sys.argv) > 1 else "How do I lower my blood pressure?"
print(f"Query: {query}\n")

# Embed the query the SAME way we embedded the chunks (normalized), then search.
q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
scores, positions = index.search(q_vec, k=5)   # top 5 nearest chunks

for rank, (pos, score) in enumerate(zip(positions[0], scores[0]), start=1):
    c = chunks[pos]                              # position -> metadata via file order
    print(f"[{rank}] score={score:.3f}  source={c['source']}")
    print(f"    Q: {c['question']}")
    print(f"    A: {c['chunk'][:200]}...")
    print(f"    refs: {c['references']}\n")