"""Measure how many answers are too long for the embedding model to handle whole."""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")
MAX_TOKENS = model.max_seq_length  # the hard limit; anything longer gets silently cut off
print(f"Model truncates at {MAX_TOKENS} tokens\n")

records = [json.loads(line) for line in Path("data/processed/dataset.jsonl").open()]

def token_count(text: str) -> int:
    # tokenize WITHOUT auto-truncation so we see the true length, even past the limit
    return len(model.tokenizer(text, truncation=False)["input_ids"])

lengths = [token_count(r["answer"]) for r in records]

over = [n for n in lengths if n > MAX_TOKENS]
print(f"total answers:        {len(lengths)}")
print(f"longest answer:       {max(lengths)} tokens")
print(f"answers over {MAX_TOKENS}:  {len(over)}  ({100*len(over)/len(lengths):.1f}%)")

# Show the longest few so we know what we're dealing with
longest = sorted(records, key=lambda r: token_count(r["answer"]), reverse=True)[:3]
for r in longest:
    print("\n---", r["source"], "|", token_count(r["answer"]), "tokens ---")
    print("Q:", r["question"])
    print("A:", r["answer"][:300], "...")