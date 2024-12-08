import logging

# Configura os níveis de logging para várias bibliotecas para reduzir a verbosidade
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)

# Cria um logger para o servidor 'uvicorn'
logger = logging.getLogger("uvicorn")
