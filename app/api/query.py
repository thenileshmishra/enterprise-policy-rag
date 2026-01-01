from fastapi import APIRouter, HTTPException
from app.core.exception import CustomException
from app.core.logger import logger
from app.retrieval.retriever import retrieve_topk
from app.retrieval.rag_pipeline import rag_answer
from app.core.monitor import track_latency

router = APIRouter()


@router.post("/query")
@track_latency
async def ask_question(payload: dict):
    try:
        question = payload.get("query")

        if not question:
            raise HTTPException(status_code=400, detail="Query missing")
        
        if len(question) > 500:
            raise CustomException("Query too long")

        logger.info(f"User query: {question}")

        chunks = retrieve_topk(question, top_k=5)

        result = rag_answer(question, chunks)

        return result

    except HTTPException:
        # re-raise known HTTP errors
        raise
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
