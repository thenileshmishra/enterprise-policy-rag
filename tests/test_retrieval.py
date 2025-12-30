from app.retrieval.retriever import Retriever


def test_retrieval_pipeline():
    retriever = Retriever()

    dummy_chunks = [
        {"text" : "Employees are entitled to 24 days leave.", "source" :"policy.pdf", "page":2},
        {"text": "Company follows strict data privacy rules.", "source": "policy.pdf", "page": 4},
    ]

    retriever.index_documents(dummy_chunks)
    res = retriever.retrive("How many leave days?", top_k=1)

    assert len(res) > 0