from loguru import logger
import sys
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# File logging (added)
logger.add(
    f"{LOG_DIR}/app.log",
    level="INFO",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    format="{time} | {level} | {message}",
)

__all__ = ["logger"]