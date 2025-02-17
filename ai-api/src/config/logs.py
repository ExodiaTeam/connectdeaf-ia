import logging

logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)

logger = logging.getLogger("uvicorn")
