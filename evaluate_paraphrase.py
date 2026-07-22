"""Measure how much the retrieval scores are inflated by dataset phrasing.

The standard evaluation queries the index with the dataset's OWN questions, which share
~59% of their vocabulary with their own answers (they were generated as pairs). Real
users phrase things differently, so those scores overstate real-world performance.

This script paraphrases each question into colloquial user language, then evaluates the
SAME sample twice - original vs paraphrased - for both retrieval methods. The gap
quantifies the bias.

Paraphrases are cached to results/retrieval/paraphrases.json so re-runs are free.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

import ollama

import bm25_search
import rag

PARAPHRASE_MODEL = "llama3.2:3b"
K_VALUES = [1, 3, 5]
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
OUT_DIR = Path("results/retrieval")
CACHE_FILE = OUT_DIR / "paraphrases.json"

PARAPHRASE_PROMPT = """You rewrite medical questions the way an ordinary worried person
would type them into a search box.

Rules:
- Keep the EXACT same meaning and topic.
- Use everyday words instead of clinical or formal ones.
- Write it as a real person would: informal, sometimes first-person.
- Do NOT answer the question. Only rewrite it.

Reply with JSON only: {"paraphrase": "..."}

QUESTION: What are the risks of high blood pressure?
{"paraphrase": "my blood pressure is high, what bad stuff can happen?"}

QUESTION: What is the recommended treatment for bronchiolitis?
{"paraphrase": "how do you treat a baby's chest infection?"}

QUESTION: What are the symptoms of panic disorder?
{"paraphrase": "how do i know if im having panic attacks?"}
"""


def paraphrase(question: str) -> str | None:
    """Rewrite one question colloquially. Returns None if the model fails."""
    try:
        resp = ollama.chat(
            model=PARAPHRASE_MODEL,
            messages=[
                {"role": "system", "content": PARAPHRASE_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}"},
            ],
            format="json",
        )
        text = (json.loads(resp["message"]["content"]).get("paraphrase") or "").strip()
        return text or None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def score_run(questions: list[tuple[str, str]], retrieve_fn) -> dict:
    """questions is [(gold_parent_id, query_text), ...]. Returns metric dict."""
    max_k = max(K_VALUES)
    hits_at_k = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    latencies = []

    for gold_parent, query in questions:
        t0 = time.perf_counter()
        results = retrieve_fn(query, k=max_k)
        latencies.append(time.perf_counter() - t0)

        rank = next((j for j, c in enumerate(results) if c["parent_id"] == gold_parent), None)
        if rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1 / (rank + 1))
            for k in K_VALUES:
                if rank < k:
                    hits_at_k[k] += 1

    n = len(questions)
    metrics = {f"recall@{k}": round(hits_at_k[k] / n, 4) for k in K_VALUES}
    metrics["mrr"] = round(sum(reciprocal_ranks) / n, 4)
    metrics["median_latency_ms"] = round(statistics.median(latencies) * 1000, 2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--sample-size", type=int, default=100)
    parser.add_argument("--regenerate", action="store_true", help="ignore cached paraphrases")
    args = parser.parse_args()

    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(l) for l in CHUNKS_FILE.open(encoding="utf-8")]
    by_parent = {}
    for c in chunks:
        by_parent.setdefault(c["parent_id"], c["question"])
    sample = random.sample(list(by_parent.items()), min(args.sample_size, len(by_parent)))

    # --- paraphrase (cached: this is the expensive part) ---
    cache = {}
    if CACHE_FILE.exists() and not args.regenerate:
        cache = json.loads(CACHE_FILE.read_text())

    pairs = []  # (parent_id, original, paraphrase)
    failures = 0
    for i, (pid, question) in enumerate(sample, start=1):
        if pid in cache:
            para = cache[pid]
        else:
            para = paraphrase(question)
            if para:
                cache[pid] = para
            if i % 20 == 0:
                print(f"  paraphrased {i}/{len(sample)}")
        if para:
            pairs.append((pid, question, para))
        else:
            failures += 1

    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    print(f"\n{len(pairs)} paraphrased ({failures} failures)\n")

    originals = [(pid, q) for pid, q, _ in pairs]
    paraphrased = [(pid, p) for pid, _, p in pairs]

    results = {}
    for label, fn in [("bm25", bm25_search.retrieve), ("dense", rag.retrieve)]:
        results[label] = {
            "original": score_run(originals, fn),
            "paraphrased": score_run(paraphrased, fn),
        }
        o, p = results[label]["original"], results[label]["paraphrased"]
        print(f"{label:6} original    Recall@5={o['recall@5']:.3f}  MRR={o['mrr']:.3f}")
        print(f"{label:6} paraphrased Recall@5={p['recall@5']:.3f}  MRR={p['mrr']:.3f}")
        drop = (o["recall@5"] - p["recall@5"]) / o["recall@5"] * 100 if o["recall@5"] else 0
        print(f"{label:6} -> Recall@5 drops {drop:.1f}%\n")

    payload = {
        "sample_size": len(pairs),
        "paraphrase_model": PARAPHRASE_MODEL,
        "corpus_chunks": len(chunks),
        "results": results,
        "examples": [{"original": q, "paraphrase": p} for _, q, p in pairs[:10]],
    }
    out_file = OUT_DIR / "paraphrase.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
