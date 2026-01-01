import time
from app.core.logger import logger


def track_latency(func):
    """
    Decorator to track latency of critical API operations
    """

    async def wrapper(*args, **kwargs):
        start = time.time()
        resp = await func(*args, **kwargs)
        end = time.time()

        logger.info(f"{func.__name__} latency: {(end-start)*1000:.2f} ms")
        return resp

    return wrapper
