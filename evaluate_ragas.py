"""Evaluate BOTH retrieval and generation with RAGAS, judged by a local Ollama model.

This is the "industry-standard library" counterpart to our hand-rolled
evaluate_retrieval.py / evaluate_generation.py. Running both lets us cross-check:
if RAGAS's Faithfulness roughly agrees with our own judge, that validates both.

Needs: pip install -r requirements-eval.txt   (and `ollama serve` running)
"""
import argparse
import json
import random
import re
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from rag import answer  # the exact pipeline the app uses

JUDGE_MODEL = "llama3.2:3b"
EMBED_MODEL = "all-MiniLM-L6-v2"

parser = argparse.ArgumentParser(description=__doc__)
# Default is deliberately tiny: every sample triggers MANY local LLM calls, and
# Faithfulness alone decomposes each answer into claims and verifies them one by one.
parser.add_argument("-n", "--sample-size", type=int, default=5)
# RAGAS asks the judge for structured JSON. Small models often answer with prose or even
# Python code instead, which RAGAS cannot parse -> the metric comes back NaN. A larger
# judge (llama3.1:8b) follows the JSON instruction far more reliably, at ~2x the time.
parser.add_argument("--judge", default=JUDGE_MODEL, help="Ollama model used as judge")
args = parser.parse_args()
JUDGE_MODEL = args.judge

random.seed(42)

# --- Build the eval set from our own data -------------------------------------------
# The dataset's original answer IS the ground truth, so no manual labelling is needed.
records = [json.loads(l) for l in Path("data/processed/dataset.jsonl").open(encoding="utf-8")]
sample = random.sample(records, args.sample_size)

DISCLAIMER = "This is general information, not medical advice."


def clean_response(text: str) -> str:
    """Strip citation markers and the fixed disclaimer before judging.

    Our generator cites passages as [1], [2]... but RAGAS passes contexts to the judge
    as a plain list with no numbering, so those markers look like references to sources
    the judge cannot see - and it penalises them. RAGAS's Faithfulness also decomposes
    the response into atomic claims, and the boilerplate disclaimer becomes a "claim"
    that appears in no context. Removing both leaves only the medical claims we mean
    to score.
    """
    text = re.sub(r"\[\d+\]", "", text)      # drop [1], [2], ...
    text = text.replace(DISCLAIMER, "")
    return re.sub(r"\s+", " ", text).strip()  # tidy the whitespace left behind


samples = []
for i, r in enumerate(sample, start=1):
    generated, hits = answer(r["question"])  # runs the full RAG pipeline
    samples.append(
        SingleTurnSample(
            user_input=r["question"],                       # the question asked
            response=clean_response(generated),             # our answer, markers stripped
            retrieved_contexts=[h["chunk"] for h in hits],  # what retrieval fed it
            reference=r["answer"],                          # true answer from the dataset
        )
    )
    print(f"generated {i}/{args.sample_size}")

dataset = EvaluationDataset(samples=samples)

# --- Point RAGAS at LOCAL models instead of its OpenAI default -----------------------
judge = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL))
embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))

# RAGAS defaults to max_workers=16 and timeout=180, which assumes a hosted API. Firing
# 16 concurrent requests at one local model just thrashes it and every job times out.
# Serialise the calls and give each one room to finish.
run_config = RunConfig(timeout=900, max_workers=1, max_retries=3)

result = evaluate(
    dataset=dataset,
    metrics=[
        LLMContextPrecisionWithReference(),  # RETRIEVAL: were the fetched chunks useful?
        LLMContextRecall(),                  # RETRIEVAL: did we fetch what was needed?
        Faithfulness(),                      # GENERATION: grounded, or hallucinated?
        ResponseRelevancy(),                 # GENERATION: does it answer the question?
    ],
    llm=judge,
    embeddings=embeddings,
    run_config=run_config,
)

print("\n=== RAGAS results ===")
print(result)
print(
    "\nNote: NaN scores mean the local judge failed to emit parseable output for that "
    "sample. A few are normal with a 3B judge; if most are NaN, try a bigger model "
    "(ollama pull llama3.1:8b) rather than fighting the parser."
)
