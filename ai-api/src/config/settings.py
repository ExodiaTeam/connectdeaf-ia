from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Classe para carregar as configurações do ambiente."""

    SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENAI_API_KEY: str
    OPENAI_API_VERSION: str
    OPENAI_AZURE_ENDPOINT: str
    OPENAI_GPT_MODEL: str
    OPENAI_EMBEDDING_MODEL: str
    AZURE_SEARCH_API_KEY: str
    AZURE_SEARCH_ENDPOINT: str
    AZURE_OCR_ENDPOINT: str
    AZURE_OCR_KEY: str
    AZURE_STORAGE_CONNECTION_STRING: str
    DOCUMENTS_CONTAINER_NAME: str


settings = Settings()
