"""Streamlit UI for the medical RAG assistant.

Thin front-end over router.answer(): the router does retrieval, guardrails
(refuse / corpus / web-search) and generation. This file only renders.
"""
import streamlit as st

from router import answer as routed_answer

st.set_page_config(page_title="NHS Health Q&A", page_icon="🩺")

BADGES = {
    "corpus": "From NHS corpus",
    "web": "From web search",
    "web_no_results": "Web search — nothing found",
    "refuse": "Out of scope",
}


def render_sources(sources: list[dict]) -> None:
    """Show retrieved sources. Handles both corpus chunks and web results."""
    if not sources:
        return
    with st.expander("Sources used"):
        for i, s in enumerate(sources, start=1):
            if "url" in s:  # web result: {title, url, snippet}
                st.markdown(f"**[{i}] {s['title']}**  \n{s['url']}")
            else:  # corpus chunk
                ref = s["references"][0] if s["references"] else "(no citation)"
                st.markdown(f"**[{i}] {s['source']}**  \nQ: *{s['question']}*  \n{ref}")


# --- UI ---
st.title("🩺 NHS-Grounded Health Q&A")
st.caption("Answers come from a curated NHS corpus, with a trusted-web fallback. Not medical advice.")

# Chat history survives Streamlit reruns via session_state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-draw the whole conversation on every rerun.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("caption"):
            st.caption(msg["caption"])
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

# Persistent input box pinned to the bottom.
if query := st.chat_input("Ask a health question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # [:-1] excludes the message we just appended - it's passed as `query`.
            out = routed_answer(query, history=st.session_state.messages[:-1])

        # caption = f"{BADGES.get(out['decision'], out['decision'])} · top score {out['top_score']:.2f}"
        caption = 'Bot_Assistant'
        st.caption(caption)
        st.markdown(out["answer"])
        render_sources(out["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"],
         "sources": out["sources"], "caption": caption}
    )
