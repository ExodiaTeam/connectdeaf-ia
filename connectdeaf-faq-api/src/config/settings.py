import os

class Settings:
    """Classe para carregar as configurações do ambiente."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_API_VERSION: str = os.getenv("OPENAI_API_VERSION")
    OPENAI_AZURE_ENDPOINT: str = os.getenv("OPENAI_AZURE_ENDPOINT")
    OPENAI_GPT_MODEL: str = os.getenv("OPENAI_GPT_MODEL")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL")
    AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY")
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT")

settings = Settings()
