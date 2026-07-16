"""
Step 1 of the RAG pipeline: download the FunPang/medical_dataset from Hugging Face
and turn it into a clean list of question/answer records we can later embed and index.

Run it with:
    python load_data.py
"""

import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd

# --- Where the data comes from and where it goes -----------------------------------

DATASET_REPO = "FunPang/medical_dataset"

# Every CSV in the dataset. The NHS files include a source URL per answer, the other
# two don't - we keep that distinction in the output so answers can be cited later.
DATASET_FILES = [
    "prepared_generated_data_for_nhs_uk_qa_1.csv",
    "prepared_generated_data_for_nhs_uk_conversations_1.csv",
    "prepared_generated_data_for_medical_tasks_1.csv",
    "mental_health_conversation.csv",
]

RAW_DIR = Path("data/raw")
PROCESSED_FILE = Path("data/processed/dataset.jsonl")

# --- Parsing raw <HUMAN>/<ASSISTANT> transcripts into Q&A pairs --------------------

# Each row's "text" column looks like:
#   "<HUMAN>: question <ASSISTANT>: answer <|eos|> <HUMAN>: next question ... <|eod|>"
# One row can contain several back-and-forth turns, so we split it into one record
# per question/answer pair.
TURN_PATTERN = re.compile(
    r"<HUMAN>:\s*(?P<question>.*?)\s*<ASSISTANT>:\s*(?P<answer>.*?)(?=<HUMAN>:|<\|eod\|>|$)",
    re.DOTALL,
)
REFERENCES_PATTERN = re.compile(r"\n?References:\s*(?P<refs>(?:-.*(?:\n|$))+)", re.MULTILINE)


def parse_conversation(raw_text: str) -> list[dict]:
    """Turn one raw transcript into a list of {question, answer, references} dicts."""
    text = str(raw_text).strip()
    turns = []

    for match in TURN_PATTERN.finditer(text):
        question = match.group("question").strip()
        answer = match.group("answer").replace("<|eos|>", "").replace("<|eod|>", "").strip()

        # NHS answers end with a "References:\n- url" block. Pull it out into its own
        # field so downstream code can cite the source instead of leaving it in the text.
        references = []
        ref_match = REFERENCES_PATTERN.search(answer)
        if ref_match:
            references = [
                line.lstrip("- ").strip()
                for line in ref_match.group("refs").strip().splitlines()
                if line.strip()
            ]
            answer = REFERENCES_PATTERN.sub("", answer).strip()

        if question and answer:
            turns.append({"question": question, "answer": answer, "references": references})

    return turns


# --- Download + build -----------------------------------------------------------------


def download_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename in DATASET_FILES:
        print(f"Downloading {filename} ...")
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=RAW_DIR,
        )


def build_processed_dataset() -> None:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with PROCESSED_FILE.open("w", encoding="utf-8") as out_file:
        for filename in DATASET_FILES:
            source = filename.removesuffix(".csv")
            df = pd.read_csv(RAW_DIR / filename)
            count = 0

            # `raw_data_id` in these CSVs is a constant 0 for every row (not a real row
            # id), so we use the row's position in the file instead to keep ids unique.
            for row_idx, row in enumerate(df.itertuples(index=False)):
                for turn_idx, turn in enumerate(parse_conversation(row.text)):
                    record = {
                        "id": f"{source}-{row_idx}-{turn_idx}",
                        "source": source,
                        "question": turn["question"],
                        "answer": turn["answer"],
                        "references": turn["references"],
                    }
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

            counts[source] = count

    print(f"\nWrote {sum(counts.values())} question/answer records to {PROCESSED_FILE}")
    for source, count in counts.items():
        print(f"  {source:<55} {count} records")


if __name__ == "__main__":
    download_dataset()
    build_processed_dataset()
