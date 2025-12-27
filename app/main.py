from fastapi import FastAPI
from app.api.health import router as health_router
from app.core.logger import logger
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME)

@app.on_event("startup")
async def startup_even():
    logger.info("Application startup")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")

app.include_router(health_router, prefix="/api")
