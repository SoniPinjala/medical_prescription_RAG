"""Split long answers into token-sized chunks; short answers pass through unchanged."""
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
MAX_TOKENS = model.max_seq_length
BUDGET = 200  # stay safely under MAX_TOKENS, leaving room so nothing gets truncated

def token_count(text: str) -> int:
    return len(model.tokenizer(text, truncation=False)["input_ids"])

def split_into_chunks(answer: str, overlap_sentences: int = 1) -> list[str]:
    if token_count(answer) <= BUDGET:
        return [answer]

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    chunks, current = [], []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if token_count(candidate) > BUDGET and current:
            chunks.append(" ".join(current))
            # start next chunk with the last N sentences of this one (the overlap)
            current = current[-overlap_sentences:] + [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks

records = [json.loads(line) for line in Path("data/processed/dataset.jsonl").open()]
out_path = Path("data/processed/chunks.jsonl")

n_chunks = 0
with out_path.open("w", encoding="utf-8") as f:
    for r in records:
        for i, chunk in enumerate(split_into_chunks(r["answer"])):
            # every chunk KEEPS the parent's metadata so we can still cite the source
            record = {
                "id": f"{r['id']}-chunk{i}",
                "parent_id": r["id"],
                "source": r["source"],
                "question": r["question"],
                "chunk": chunk,
                "references": r["references"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_chunks += 1

print(f"{len(records)} answers -> {n_chunks} chunks, written to {out_path}")