"""Evaluate retrieval quality using held-out questions as ground truth.

Supports two retrieval backends so we can compare a sparse BASELINE (BM25) against
dense vector search. Both are scored by identical code on an identical sample.

Results are written to results/retrieval/<method>.json
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=["bm25", "dense"], required=True)
parser.add_argument("-n", "--sample-size", type=int, default=500)
args = parser.parse_args()

# Import lazily: loading the dense stack pulls in FAISS + the embedding model, which
# we don't want to pay for when evaluating BM25.
if args.method == "bm25":
    from bm25_search import retrieve
    method_label = "BM25 (sparse keyword)"
else:
    from rag import retrieve
    method_label = "Dense vectors + FAISS HNSW"

random.seed(42)  # same sample for every method = fair comparison
K_VALUES = [1, 3, 5]

chunks = [json.loads(l) for l in Path("data/processed/chunks.jsonl").open(encoding="utf-8")]

# Eval set: one (question, gold parent_id) per UNIQUE answer. Sampling by parent rather
# than by chunk avoids over-weighting long answers that split into several chunks.
by_parent = {}
for c in chunks:
    by_parent.setdefault(c["parent_id"], c["question"])

sample = random.sample(list(by_parent.items()), min(args.sample_size, len(by_parent)))
max_k = max(K_VALUES)

hits_at_k = {k: 0 for k in K_VALUES}
reciprocal_ranks = []
latencies = []

for i, (gold_parent, question) in enumerate(sample, start=1):
    t0 = time.perf_counter()
    results = retrieve(question, k=max_k)
    latencies.append(time.perf_counter() - t0)

    # Rank (0-based) of the first chunk that came from the correct answer.
    rank = next((j for j, c in enumerate(results) if c["parent_id"] == gold_parent), None)

    if rank is None:
        reciprocal_ranks.append(0.0)
    else:
        reciprocal_ranks.append(1 / (rank + 1))
        for k in K_VALUES:
            if rank < k:
                hits_at_k[k] += 1

    if i % 100 == 0:
        print(f"  {i}/{len(sample)}")

n = len(sample)
metrics = {f"recall@{k}": round(hits_at_k[k] / n, 4) for k in K_VALUES}
metrics["mrr"] = round(sum(reciprocal_ranks) / n, 4)
metrics["median_latency_ms"] = round(statistics.median(latencies) * 1000, 2)

result = {
    "method": args.method,
    "method_label": method_label,
    "sample_size": n,
    "corpus_chunks": len(chunks),
    "metrics": metrics,
}

out_dir = Path("results/retrieval")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f"{args.method}.json"
out_file.write_text(json.dumps(result, indent=2))

print(f"\n=== {method_label} (n={n}) ===")
for k in K_VALUES:
    print(f"Recall@{k}: {metrics[f'recall@{k}']:.3f}")
print(f"MRR:       {metrics['mrr']:.3f}")
print(f"Latency:   {metrics['median_latency_ms']:.1f} ms/query (median)")
print(f"\nSaved to {out_file}")