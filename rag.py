"""Full RAG: retrieve relevant chunks, then have a local LLM answer using only them."""
import json
import sys
from pathlib import Path

import faiss
import ollama
from sentence_transformers import SentenceTransformer

MODEL = "llama3.2:3b"
TOP_K = 5

# --- Load everything we built earlier ---
chunks = [json.loads(line) for line in Path("data/processed/chunks.jsonl").open(encoding="utf-8")]
index = faiss.read_index("data/processed/faiss.index")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed the query and return the k most similar chunks (default TOP_K).

    Each returned chunk carries a "score": the cosine similarity to the query, in
    [0, 1]. Routing logic uses it to decide whether the corpus actually knows the
    answer. dict(...) copies the chunk so we never mutate the cached list.
    """
    q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, positions = index.search(q_vec, k)
    return [
        dict(chunks[pos], score=float(score))
        for pos, score in zip(positions[0], scores[0])
    ]


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block the model can cite."""
    blocks = []
    for i, c in enumerate(hits, start=1):
        ref = c["references"][0] if c["references"] else "no source URL"
        blocks.append(f"[{i}] (source: {ref})\n{c['chunk']}")
    return "\n\n".join(blocks)


def answer(query: str) -> tuple[str, list[dict]]:
    """Return the generated answer and the chunks it was grounded in."""
    hits = retrieve(query)
    context = build_context(hits)

    system_prompt = (
        "You are a health information assistant. Answer the user's question using ONLY the "
        "numbered context passages provided. If the context does not contain the answer, say "
        "you don't have that information rather than guessing. Cite the sources you used by their "
        "[number]. Always end with: 'This is general information, not medical advice.'"
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"], hits


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the symptoms of a panic attack?"
    print(f"Q: {query}\n")

    reply, hits = answer(query)
    print(reply)

    print("\n--- Sources used ---")
    for i, c in enumerate(hits, start=1):
        ref = c["references"][0] if c["references"] else "(no citation)"
        print(f"[{i}] {c['source']} — {ref}")