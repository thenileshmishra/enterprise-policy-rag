"""
NotebookLLM-style Streamlit UI for RAG system.
Features: Multi-PDF upload, session management, citations, and chat history.
"""

import streamlit as st
import requests
import uuid
from typing import List, Dict, Optional

# API Configuration
API_BASE = "http://localhost:8000/api"

# Page Configuration
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .citation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .source-tag {
        background-color: #e1e5eb;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        margin-right: 5px;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_citations" not in st.session_state:
        st.session_state.current_citations = []


def upload_document(file) -> Dict:
    """Upload a document to the API."""
    try:
        res = requests.post(
            f"{API_BASE}/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            params={"session_id": st.session_state.session_id}
        )
        if res.status_code == 200:
            return {"success": True, "data": res.json()}
        else:
            return {"success": False, "error": res.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def query_documents(query: str) -> Dict:
    """Send a query to the RAG API."""
    try:
        res = requests.post(
            f"{API_BASE}/query",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "include_citations": True
            }
        )
        if res.status_code == 200:
            return {"success": True, "data": res.json()}
        else:
            return {"success": False, "error": res.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_citation(citation: Dict, index: int):
    """Render a single citation with expandable details."""
    source = citation.get("source", "Unknown")
    page = citation.get("page", "?")
    section = citation.get("section", "")
    snippet = citation.get("snippet", "")

    with st.expander(f"[{index}] {source} - Page {page}", expanded=False):
        if section:
            st.markdown(f"**Section:** {section}")
        if snippet:
            st.markdown(f"**Excerpt:** _{snippet[:300]}..._" if len(snippet) > 300 else f"**Excerpt:** _{snippet}_")


def render_confidence_badge(score: float):
    """Render a confidence score badge."""
    if score >= 0.75:
        return f'<span class="confidence-high">● High Confidence ({score:.0%})</span>'
    elif score >= 0.55:
        return f'<span class="confidence-medium">● Medium Confidence ({score:.0%})</span>'
    else:
        return f'<span class="confidence-low">● Low Confidence ({score:.0%})</span>'


def render_chat_message(role: str, content: str, citations: Optional[List] = None):
    """Render a chat message with optional citations."""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"

    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.title()}</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)

    if citations and role == "assistant":
        with st.expander("📚 View Sources", expanded=False):
            for i, cit in enumerate(citations, 1):
                render_citation(cit, i)


def clear_session():
    """Clear the current session."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.uploaded_documents = []
    st.session_state.chat_history = []
    st.session_state.current_citations = []


# Initialize session
init_session_state()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("📚 Research Assistant")
    st.markdown("---")

    # Session Info
    st.markdown("### Session")
    st.caption(f"ID: {st.session_state.session_id[:8]}...")

    if st.button("🔄 New Session", use_container_width=True):
        clear_session()
        st.rerun()

    st.markdown("---")

    # Document Upload Section
    st.markdown("### 📄 Upload Documents")
    st.caption("Upload research papers to start chatting")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_documents:
                with st.spinner(f"Processing {file.name}..."):
                    result = upload_document(file)

                if result["success"]:
                    st.session_state.uploaded_documents.append(file.name)
                    st.success(f"✓ {file.name}")
                else:
                    st.error(f"✗ {file.name}: {result.get('error', 'Upload failed')}")

    # Uploaded Documents List
    if st.session_state.uploaded_documents:
        st.markdown("---")
        st.markdown("### 📁 Loaded Documents")
        for doc in st.session_state.uploaded_documents:
            st.markdown(f"• {doc}")

    st.markdown("---")

    # Settings
    with st.expander("⚙️ Settings"):
        st.slider("Number of sources to retrieve", 3, 15, 5, key="retrieval_k")
        st.checkbox("Show confidence scores", value=True, key="show_confidence")
        st.checkbox("Use hybrid search", value=True, key="use_hybrid")


# ==================== MAIN CONTENT ====================
st.title("💬 Chat with Your Documents")

# Check if documents are uploaded
if not st.session_state.uploaded_documents:
    st.info("👈 Upload research papers in the sidebar to get started!")

    # Demo section
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Upload PDFs** - Add your research papers using the sidebar
        2. **Ask Questions** - Type your questions about the documents
        3. **Get Cited Answers** - Receive answers with source citations
        4. **Explore Sources** - Click on citations to see the original text

        **Features:**
        - Multi-document support
        - Hybrid search (semantic + keyword)
        - Source citations with page numbers
        - Confidence scoring
        - Session-based memory
        """)
else:
    # Chat History
    for msg in st.session_state.chat_history:
        render_chat_message(
            msg["role"],
            msg["content"],
            msg.get("citations")
        )

    # Query Input
    st.markdown("---")

    col1, col2 = st.columns([6, 1])

    with col1:
        query = st.text_input(
            "Ask a question about your documents",
            placeholder="What are the main findings of the research?",
            key="query_input",
            label_visibility="collapsed"
        )

    with col2:
        submit = st.button("Send", type="primary", use_container_width=True)

    # Handle Query
    if submit and query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        with st.spinner("Searching documents and generating answer..."):
            result = query_documents(query)

        if result["success"]:
            data = result["data"]
            answer = data.get("answer", "No answer generated")
            confidence = data.get("confidence", data.get("faithfulness_score", 0.5))
            citations = data.get("citations", [])
            sources = data.get("sources", [])

            # Build response with confidence
            if st.session_state.get("show_confidence", True):
                confidence_badge = render_confidence_badge(confidence)
                response_content = f"{answer}\n\n{confidence_badge}"
            else:
                response_content = answer

            # Format sources as citations if not already provided
            if not citations and sources:
                citations = [{"source": s, "page": "N/A"} for s in sources]

            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "confidence": confidence
            })

            st.session_state.current_citations = citations

        else:
            error_msg = f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg
            })

        st.rerun()

    # Citation Panel (for current response)
    if st.session_state.current_citations:
        st.markdown("---")
        st.markdown("### 📚 Sources for Last Response")

        cols = st.columns(min(len(st.session_state.current_citations), 3))
        for i, citation in enumerate(st.session_state.current_citations[:6]):
            with cols[i % 3]:
                source = citation.get("source", "Unknown")
                page = citation.get("page", "?")
                section = citation.get("section", "")

                st.markdown(f"""
                <div class="citation-box">
                    <strong>[{i+1}]</strong> {source}<br>
                    <span class="source-tag">Page {page}</span>
                    {f'<span class="source-tag">{section}</span>' if section else ''}
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("📚 Research Paper Assistant | Powered by RAG + Llama")
