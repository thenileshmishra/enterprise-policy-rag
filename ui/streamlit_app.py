"""Streamlit UI for Advanced RAG with multi-document sessions."""

import streamlit as st
import requests

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="Advanced RAG", page_icon="📄", layout="centered")
st.title("📄 Advanced RAG")
st.caption("Upload multiple PDFs, choose retrieval mode, and ask questions.")

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "documents" not in st.session_state:
    st.session_state.documents = []

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Index All", type="primary"):
        for uf in uploaded_files:
            if uf.name in st.session_state.documents:
                continue
            with st.spinner(f"Indexing {uf.name}..."):
                data_payload = {"session_id": st.session_state.session_id} if st.session_state.session_id else {}
                response = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (uf.name, uf.getvalue(), "application/pdf")},
                    data=data_payload,
                    timeout=120,
                )
            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.documents.append(uf.name)
                st.success(f"{uf.name}: {data['chunks_indexed']} chunks")
            else:
                st.error(f"{uf.name}: {response.text}")

    if st.session_state.documents:
        st.divider()
        st.write(f"**Session:** `{st.session_state.session_id}`")
        st.write(f"**Documents:** {len(st.session_state.documents)}")
        for doc in st.session_state.documents:
            st.write(f"- {doc}")

    if st.button("New Session"):
        st.session_state.session_id = None
        st.session_state.documents = []
        st.rerun()

# --- Query Settings ---
st.subheader("Query Settings")
col1, col2, col3 = st.columns(3)
with col1:
    mode = st.selectbox("Retrieval Mode", ["hybrid", "dense", "sparse"])
with col2:
    use_reranking = st.checkbox("Rerank", value=False)
with col3:
    top_k = st.slider("Top K", 1, 10, 5)

# --- Query ---
query = st.text_input("Ask a question")
if st.button("Ask", type="primary") and query.strip():
    if not st.session_state.session_id:
        st.warning("Upload and index a PDF first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            response = requests.post(
                f"{API_BASE}/query",
                json={
                    "query": query,
                    "session_id": st.session_state.session_id,
                    "top_k": top_k,
                    "mode": mode,
                    "use_reranking": use_reranking,
                },
                timeout=120,
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            # Metrics row
            c1, c2, c3 = st.columns(3)
            c1.metric("Faithfulness", f"{data['faithfulness_score']:.2f}")
            c2.metric("Grounded", "Yes" if data["is_grounded"] else "No")
            c3.metric("Method", data["retrieval_method"])

            if data["sources"]:
                st.caption(f"Sources: {', '.join(data['sources'])}")
        else:
            st.error(response.text)
