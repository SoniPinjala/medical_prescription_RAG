"""LLM-judged generation metrics: faithfulness + answer relevance. No extra deps."""
import argparse
import json
import random
from pathlib import Path

import ollama

from rag import answer, build_context  # reuse the exact pipeline the app uses

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-n", "--sample-size", type=int, default=20)
args = parser.parse_args()

random.seed(42)
SAMPLE_SIZE = args.sample_size
JUDGE_MODEL = "llama3.2:3b"

FAITHFULNESS_PROMPT = """Evaluate whether an ANSWER is supported by the CONTEXT.

The CONTEXT is a list of passages numbered [1], [2], ... The ANSWER may cite them by
number. Judge ONLY the medical/factual claims, and ignore:
  - citation markers like [1] - these are expected, not unsupported claims
  - the fixed disclaimer "This is general information, not medical advice."
  - anything the ANSWER declines to answer

An ANSWER is faithful if its factual claims are stated in, or directly follow from, the
CONTEXT. Do not mark it unfaithful for omitting information or for lacking caveats.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only: {{"faithful": true or false, "reason": "one short sentence"}}"""

RELEVANCE_PROMPT = """Evaluate whether an ANSWER addresses the QUESTION.

QUESTION: {question}

ANSWER:
{answer}

Does the ANSWER directly address the QUESTION?
Reply with JSON only: {{"relevant": true or false, "reason": "one short sentence"}}"""


def judge(prompt: str) -> dict | None:
    """Ask the local model to score something. format='json' forces parseable output."""
    try:
        resp = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",  # Ollama constrains the model to emit valid JSON
        )
        return json.loads(resp["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return None  # judge failed to produce usable output


records = [json.loads(l) for l in Path("data/processed/dataset.jsonl").open(encoding="utf-8")]
sample = random.sample(records, SAMPLE_SIZE)

faithful, relevant, failures = 0, 0, 0

for i, r in enumerate(sample, start=1):
    generated, hits = answer(r["question"])
    # Must be build_context(), NOT the raw chunks: the generator sees passages numbered
    # [1], [2]... and cites them. A judge shown unnumbered text can't resolve those
    # citations and marks every citing answer unfaithful.
    context = build_context(hits)

    f = judge(FAITHFULNESS_PROMPT.format(context=context, answer=generated))
    v = judge(RELEVANCE_PROMPT.format(question=r["question"], answer=generated))

    if f is None or v is None:
        failures += 1
    else:
        faithful += bool(f.get("faithful"))
        relevant += bool(v.get("relevant"))

    print(f"{i}/{SAMPLE_SIZE}  faithful={f and f.get('faithful')}  relevant={v and v.get('relevant')} question={r['question'][:50]}... answer={generated[:50]}...")

scored = SAMPLE_SIZE - failures
print(f"\n=== Generation metrics (n={scored}, judge failures={failures}) ===")
if scored:
    print(f"Faithfulness:    {faithful / scored:.2f}   (answer grounded in retrieved context)")
    print(f"Answer relevance:{relevant / scored:.2f}   (answer addresses the question)")

# --- Refusal test -------------------------------------------------------------------
# The corpus has no drug dosages and nothing outside general health, so the assistant
# should DECLINE these rather than invent an answer. Read these by hand: no metric
# substitutes for seeing whether it makes up an ibuprofen dose.
OUT_OF_CORPUS = [
    "Who is the best Cricketer?",
    "What is the capital of France?",
    "Write me a Python function to sort a list.",
]

print("\n=== Refusal test (should decline / say it doesn't know) ===")
for q in OUT_OF_CORPUS:
    reply, _ = answer(q)
    print(f"\nQ: {q}\nA: {reply[:250]}...")