from fastapi import APIRouter
from app.retrieval.retriever import retrieve_topk
from app.retrieval.rag_pipeline import rag_answer

router = APIRouter()


@router.post("/query")
async def query_rag(request: dict):
    question = request.get("query")

    chunks = retrieve_topk(question, top_k=5)

    result = rag_answer(question, chunks)

    return result
