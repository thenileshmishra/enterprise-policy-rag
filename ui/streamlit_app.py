"""Simple Streamlit UI for the minimal RAG backend."""

import streamlit as st
import requests

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="Simple RAG", page_icon="📄", layout="centered")
st.title("📄 Simple RAG")
st.caption("Upload one PDF, then ask questions from that document.")


uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is not None:
    if st.button("Index PDF", type="primary"):
        with st.spinner("Indexing PDF..."):
            response = requests.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                timeout=120,
            )

        if response.status_code == 200:
            data = response.json()
            st.success(f"Indexed {data['chunks_indexed']} chunks from {data['document_name']}")
        else:
            st.error(response.text)

query = st.text_input("Ask a question")
if st.button("Ask") and query.strip():
    with st.spinner("Generating answer..."):
        response = requests.post(
            f"{API_BASE}/query",
            json={"query": query, "top_k": 5},
            timeout=120,
        )

    if response.status_code == 200:
        data = response.json()
        st.subheader("Answer")
        st.write(data["answer"])
        st.caption(f"Confidence: {data['confidence']}")
        st.caption(f"Sources: {', '.join(data['sources']) if data['sources'] else 'None'}")
    else:
        st.error(response.text)
