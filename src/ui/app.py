import streamlit as st
import requests
import os

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
PIPELINE_API_URL = os.getenv("API_URL", "http://localhost:8888/search")
API_KEY = os.getenv("API_KEY", "")  # Optional: matches server key

st.set_page_config(
    page_title="Dark Data RAG Agent",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar Settings
# ------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/dusk/128/document--v1.png", width=80)
    st.title("⚙️ Settings")
    top_k = st.slider("Results to retrieve", 1, 10, 3)
    show_sources = st.toggle("Show source chunks", value=True)
    st.divider()
    st.caption("Dark Data RAG Pipeline v2.0")
    st.caption("Powered by Qdrant · HuggingFace · FastAPI")

# ------------------------------------------------------------------
# HuggingFace Answer Synthesis (local, no API cost)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading answer synthesizer...")
def load_synthesizer():
    try:
        from transformers import pipeline  # type: ignore
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-base",       # ~300MB, free, fast, runs on CPU
            max_new_tokens=200,
        )
    except Exception as e:
        st.warning(f"Synthesizer unavailable: {e}. Showing raw chunks only.")
        return None


def synthesize_answer(query: str, chunks: list, synthesizer) -> str:
    """Build a grounded answer from retrieved chunks using flan-t5."""
    if not synthesizer or not chunks:
        return ""

    context = "\n\n".join(
        [f"[{i+1}] {c['text'][:400]}" for i, c in enumerate(chunks)]
    )
    prompt = (
        f"Answer the following question using ONLY the provided context.\n"
        f"If the answer is not in the context, say 'Information not found in documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    try:
        result = synthesizer(prompt, truncation=True)
        return result[0]["generated_text"].strip()
    except Exception as e:
        return f"Synthesis error: {e}"


# ------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------
st.title("🕵️ Dark Data RAG Agent")
st.markdown(
    "Ask questions about ingested unstructured documents. "
    "The pipeline retrieves relevant chunks from **Qdrant** and synthesizes a grounded answer."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and "sources" in msg:
            with st.expander(f"📄 View {len(msg['sources'])} source chunks", expanded=False):
                for i, chunk in enumerate(msg["sources"]):
                    src = chunk["metadata"].get("source", "Unknown")
                    score = chunk["score"]
                    st.markdown(f"**[{i+1}]** `{src}` · Score: `{score:.3f}`")
                    st.info(chunk["text"][:500])

# Chat Input
prompt = st.chat_input("Ask anything about your documents…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Step 1: Retrieve from Qdrant via API
        with st.spinner("🔍 Searching Qdrant Vector DB..."):
            headers = {"X-API-Key": API_KEY} if API_KEY else {}
            try:
                resp = requests.post(
                    PIPELINE_API_URL,
                    json={"query": prompt, "top_k": top_k},
                    headers=headers,
                    timeout=15,
                )
                resp.raise_for_status()
                chunks = resp.json()
            except Exception as e:
                st.error(
                    f"❌ Could not reach the API at `{PIPELINE_API_URL}`. "
                    f"Is `make api` running?\n\nError: {e}"
                )
                st.stop()

        if not chunks:
            message_placeholder.warning("No relevant information found in the ingested documents.")
            st.stop()

        # Step 2: Synthesize answer
        with st.spinner("🤖 Synthesizing answer from retrieved context..."):
            synthesizer = load_synthesizer()
            answer = synthesize_answer(prompt, chunks, synthesizer)

        # Step 3: Display
        if answer:
            message_placeholder.markdown(f"### 💬 Answer\n\n{answer}")
            st.divider()
        else:
            message_placeholder.markdown(
                "Here is the most relevant information from your documents:"
            )

        if show_sources:
            with st.expander(f"📄 View {len(chunks)} source chunks used", expanded=True):
                for i, chunk in enumerate(chunks):
                    src = chunk["metadata"].get("source", "Unknown")
                    score = chunk["score"]
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"**[{i+1}]** `{src}`")
                    col2.metric("Score", f"{score:.3f}")
                    st.info(chunk["text"][:500])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer or "Retrieved relevant document chunks.",
            "sources": chunks,
        })
