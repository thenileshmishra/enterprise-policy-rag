import streamlit as st
import requests

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="Enterprise Policy RAG", layout="centered")

st.title("📄 Enterprise Policy & Compliance Assistant")

st.write("Upload policy PDFs and ask compliance questions.")


# ---------- Upload Section ----------
st.subheader("Upload Policy Documents")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Uploading & processing..."):
        res = requests.post(
            f"{API_BASE}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        )

    if res.status_code == 200:
        st.success("Indexed successfully!")
    else:
        st.error("Upload failed")


# ---------- Chat Section ----------
st.subheader("Ask a Question")
query = st.text_input("Enter your question")

if st.button("Ask"):
    if not query:
        st.warning("Enter a query")
    else:
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_BASE}/query", json={"query": query})

        if res.status_code == 200:
            data = res.json()
            st.markdown(f"### Answer\n{data['answer']}")
            st.markdown(f"**Confidence:** {data['confidence']}")

            st.markdown("### Sources")
            for s in data["sources"]:
                st.write(s)
        else:
            st.error("Query failed")
