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
