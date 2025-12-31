from fastapi import FastAPI
from app.api import upload, query, health

app = FastAPI(title="Enterprise Policy RAG")

app.include_router(upload.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(health.router, prefix="/api")
