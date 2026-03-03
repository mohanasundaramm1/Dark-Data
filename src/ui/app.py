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

        # Step B: Synthesize Answer via Local LLM (Ollama)
        with st.spinner(f"Synthesizing answer using {MODEL_NAME}..."):
            context = "\n\n---\n\n".join([f"Source: {c['metadata'].get('source')}\nContent: {c['text']}" for c in chunks])
            
            system_prompt = f"""
            You are an expert Data Analyst and AI Assistant. Answer the user's question based strictly on the provided context excerpts from our internal dataset.
            If the context doesn't contain the answer, say "I cannot answer this based on the provided documents."
            Do not make up external information.
            
            CONTEXT EXTRACTED FROM QDRANT DATABASE:
            {context}
            
            USER QUESTION: {prompt}
            """
            
            try:
                if Ollama is None:
                    raise ImportError("Ollama integration missing.")
                
                llm = Ollama(model=MODEL_NAME)
                response = llm.invoke(system_prompt)
                message_placeholder.markdown(response)
                
                # Render sources
                with st.expander("View Source Chunks from Qdrant"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Document**: `{chunk['metadata'].get('source', 'Unknown')}` (Sim Score: {chunk['score']:.3f})")
                        st.info(chunk["text"])
                        
                st.session_state.messages.append({"role": "assistant", "content": response, "sources": chunks})

            except Exception as e:
                message_placeholder.error(f"❌ Failed to communicate with local Ollama model. Is Ollama running? Run `ollama run {MODEL_NAME}` in your terminal. Error: {e}")
