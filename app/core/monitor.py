import time
import inspect
from functools import wraps
from app.core.logger import logger


def track_latency(func):
    """
    Decorator to track latency of critical API operations
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        resp = await func(*args, **kwargs)
        end = time.time()

        logger.info(f"{func.__name__} latency: {(end-start)*1000:.2f} ms")
        return resp

    # Preserve original function signature so FastAPI can read parameters correctly
    wrapper.__signature__ = inspect.signature(func)

    return wrapper
