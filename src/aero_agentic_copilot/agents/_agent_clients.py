from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Protocol

from google.genai import types
from qdrant_client import QdrantClient

from aero_agentic_copilot.rag.ingestion import create_qdrant_client


class EmbeddingValue(Protocol):
    values: Sequence[float] | None


class EmbeddingResponse(Protocol):
    embeddings: Sequence[EmbeddingValue] | None


class TextResponse(Protocol):
    text: str | None


class GeminiModels(Protocol):
    def embed_content(
        self,
        *,
        model: str,
        contents: str,
    ) -> EmbeddingResponse: ...

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
    ) -> TextResponse: ...


class GeminiClient(Protocol):
    models: GeminiModels

    def close(self) -> None: ...


@contextmanager
def qdrant_client_scope(
    shared_client: QdrantClient | None,
) -> Generator[QdrantClient, None, None]:
    if shared_client is not None:
        yield shared_client
        return

    client = create_qdrant_client()
    try:
        yield client
    finally:
        client.close()
