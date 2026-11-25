from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    QDRANT_MODE: Literal["local", "server"] = "local"
    QDRANT_URL: str = "http://localhost:6333"

    @property
    def qdrant_storage_path(self) -> str | None:
        return "./qdrant_storage" if self.QDRANT_MODE == "local" else None

    @property
    def qdrant_server_url(self) -> str | None:
        return self.QDRANT_URL if self.QDRANT_MODE == "server" else None


settings = Settings()
