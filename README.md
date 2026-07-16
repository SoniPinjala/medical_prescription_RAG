# Medical RAG — NHS-Grounded Health Q&A Assistant

**Course project idea:** a Retrieval-Augmented Generation (RAG) system that answers general health
questions the way NHS.uk would — retrieving a relevant passage and citing the source URL, instead of
just generating an answer from nowhere.

**Scope:** patient education ("what is X", "what are the symptoms of Y", "when should I see a
doctor"). This is **not** a diagnostic or prescribing tool, and it does not have drug-dosage data.

## The dataset

Source: [`FunPang/medical_dataset`](https://huggingface.co/datasets/FunPang/medical_dataset) on
Hugging Face. It's four CSVs of medical Q&A conversations, written as raw text like:

```
<HUMAN>: What is high blood pressure? <ASSISTANT>: High blood pressure is ... <|eos|> <|eod|>
```

| File | What it is | Records | Has a source URL? |
|---|---|---|---|
| `prepared_generated_data_for_nhs_uk_qa_1.csv` | Single-turn NHS UK Q&A | 24,665 | Yes |
| `prepared_generated_data_for_nhs_uk_conversations_1.csv` | Multi-turn NHS UK conversations | 12,234 | Yes |
| `prepared_generated_data_for_medical_tasks_1.csv` | Clinical reasoning / procedures | 4,689 | No |
| `mental_health_conversation.csv` | Mental health Q&A | 172 | No |

The two NHS files (88% of the data) are the heart of this project — they're the only rows with a real
citation attached, which is what will let the RAG system show "here's where this answer came from"
instead of asking the user to trust it blindly.

## What `load_data.py` does

One script, four steps:
1. **Download** the 4 CSVs from Hugging Face into `data/raw/`.
2. **Parse** each row's raw `<HUMAN>/<ASSISTANT>` text into separate question/answer pairs (a single
   row can contain several back-and-forth turns).
3. **Extract citations** — for NHS rows, pull the `References:` URL out of the answer text into its
   own field instead of leaving it mixed into the answer.
4. **Save** everything as one JSONL file, `data/processed/dataset.jsonl` — one question/answer record
   per line, ready to be chunked and embedded in the next stage of the pipeline.

## Running it

```bash
python3 -m venv .venv              # if you don't already have one
source .venv/bin/activate
pip install -r requirements.txt
python load_data.py
```

Each line of `data/processed/dataset.jsonl` looks like:

```json
{"id": "prepared_generated_data_for_nhs_uk_qa_1-1-0", "source": "prepared_generated_data_for_nhs_uk_qa_1", "question": "What are the risks of high blood pressure?", "answer": "...", "references": ["https://www.nhs.uk/conditions/..."]}
```

## What's next (not built yet)

- Chunk long answers and embed them into a vector store (e.g. Chroma/FAISS)
- Retriever that prefers cited NHS records over the uncited clinical/mental-health ones
- A generation step (LLM) that answers using only retrieved passages and always cites its source
- A simple way to ask it questions (CLI or small web UI)
