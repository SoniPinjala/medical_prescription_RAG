"""Streamlit UI for the medical RAG assistant."""
import json
from pathlib import Path

import faiss
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL = "llama3.2:3b"
TOP_K = 5

st.set_page_config(page_title="NHS Health Q&A", page_icon="🩺")


# --- Load heavy resources ONCE (cached across reruns) ---
@st.cache_resource
def load_resources():
    chunks = [json.loads(l) for l in Path("data/processed/chunks.jsonl").open(encoding="utf-8")]
    index = faiss.read_index("data/processed/faiss.index")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return chunks, index, embedder


chunks, index, embedder = load_resources()


def retrieve(query: str) -> list[dict]:
    q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    _scores, positions = index.search(q_vec, TOP_K)
    return [chunks[pos] for pos in positions[0]]


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(hits, start=1):
        ref = c["references"][0] if c["references"] else "no source URL"
        blocks.append(f"[{i}] (source: {ref})\n{c['chunk']}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = (
    "You are a health information assistant. Answer the user's question using ONLY the "
    "numbered context passages provided. If the context does not contain the answer, say "
    "you don't have that information rather than guessing. Cite the sources you used by their "
    "[number]. Always end with: 'This is general information, not medical advice.'"
)


# --- UI ---
st.title("🩺 NHS-Grounded Health Q&A")
st.caption("Answers are generated from retrieved NHS/medical passages. Not medical advice.")

# 1. Initialise chat history once (survives reruns via session_state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Re-draw the whole conversation on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for i, c in enumerate(msg["sources"], start=1):
                    ref = c["references"][0] if c["references"] else "(no citation)"
                    st.markdown(f"**[{i}] {c['source']}**  \nQ: *{c['question']}*  \n{ref}")

# 3. Persistent input box pinned to the bottom
if query := st.chat_input("Ask a health question..."):
    # -- show and store the user's message --
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # -- retrieve for THIS question --
    hits = retrieve(query)
    context_prompt = f"Context:\n{build_context(hits)}\n\nQuestion: {query}"

    # -- build the message list: system + past turns + current (with context) --
    model_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[:-1]:          # prior turns, plain text
        model_messages.append({"role": m["role"], "content": m["content"]})
    model_messages.append({"role": "user", "content": context_prompt})  # current + context

    # -- generate and stream into an assistant bubble --
    with st.chat_message("assistant"):
        stream = ollama.chat(model=MODEL, messages=model_messages, stream=True)
        answer = st.write_stream(chunk["message"]["content"] for chunk in stream)
        with st.expander("Sources used"):
            for i, c in enumerate(hits, start=1):
                ref = c["references"][0] if c["references"] else "(no citation)"
                st.markdown(f"**[{i}] {c['source']}**  \nQ: *{c['question']}*  \n{ref}")

    # -- store the assistant's reply so it persists --
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits})