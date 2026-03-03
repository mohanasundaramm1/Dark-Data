import streamlit as st
import requests
import json
import logging

try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None
    st.error("langchain-community not installed. Please install it.")

st.set_page_config(page_title="Dark Data Retrieval API", page_icon="🕵️‍♂️", layout="centered")

st.title("Dark Data RAG Agent 🕵️‍♂️")
st.markdown("Ask questions about the unstructured documents ingested via our Data Engineering pipeline.")

PIPELINE_API_URL = "http://localhost:8888/search"
MODEL_NAME = "llama3"

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Source Chunks from Qdrant"):
                for i, chunk in enumerate(msg["sources"]):
                    st.markdown(f"**Document**: `{chunk['metadata'].get('source', 'Unknown')}` (Sim Score: {chunk['score']:.3f})")
                    st.info(chunk["text"])

# Input from user
prompt = st.chat_input("E.g., What are the essential business skills?")

if prompt:
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Step A: Query FastAPI -> Qdrant
        with st.spinner("Querying Qdrant Vector DB..."):
            try:
                # We request Top 3 chunks
                api_response = requests.post(PIPELINE_API_URL, json={"query": prompt, "top_k": 3})
                api_response.raise_for_status()
                chunks = api_response.json()
            except Exception as e:
                st.error(f"❌ Failed to reach the Semantic Search API at {PIPELINE_API_URL}. Is `make api` running?")
                st.stop()
                
        if not chunks:
            message_placeholder.markdown("No relevant information found in the ingested documents.")
            st.stop()

        # Step B: Present Results
        with st.spinner("Extracting insights..."):
            message_placeholder.markdown("Here is the relevant information I found in our ingested Data Lake:")
            
            # Render sources
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Result {i+1}**: `{chunk['metadata'].get('source', 'Unknown')}` (Sim Score: {chunk['score']:.3f})")
                st.info(chunk["text"])
                    
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Here is the relevant information I found in our ingested Data Lake:", 
                "sources": chunks
            })
