import json
from dataclasses import dataclass

import pytest
from google.genai import types
from qdrant_client import QdrantClient, models

import aero_agentic_copilot.agents._agent_clients as agent_clients
from aero_agentic_copilot.agents.contracts_rag_agent import ContractsRagAgent
from aero_agentic_copilot.agents.survey_agent import SurveyIntelligenceAgent
from aero_agentic_copilot.config import settings


@dataclass
class FakeEmbeddingValue:
    values: list[float]


@dataclass
class FakeEmbeddingResponse:
    embeddings: list[FakeEmbeddingValue]


@dataclass
class FakeTextResponse:
    text: str


class FakeGeminiModels:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.embedding_models: list[str] = []
        self.generated_models: list[str] = []

    def embed_content(
        self,
        *,
        model: str,
        contents: str,
    ) -> FakeEmbeddingResponse:
        assert contents
        self.embedding_models.append(model)
        return FakeEmbeddingResponse([FakeEmbeddingValue([1.0, 0.0, 0.0])])

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
    ) -> FakeTextResponse:
        assert contents
        assert config.response_mime_type == "application/json"
        self.generated_models.append(model)
        return FakeTextResponse(json.dumps(self.response))


class FakeGeminiClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.models = FakeGeminiModels(response)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def create_collection(client: QdrantClient, name: str) -> None:
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )


def test_survey_agent_filters_and_returns_grounded_stats_and_sources() -> None:
    qdrant_client = QdrantClient(location=":memory:")
    create_collection(qdrant_client, "surveys")
    qdrant_client.upsert(
        "surveys",
        [
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0, 0.0],
                payload={
                    "survey_id": "SRV-012",
                    "airline_name": "Horizon Air",
                    "domain": "Cabin Customisation",
                    "rating": 2,
                    "date": "2026-03-01",
                    "verbatim_text": "Cabin customisation took longer than planned.",
                },
            ),
            models.PointStruct(
                id=2,
                vector=[1.0, 0.0, 0.0],
                payload={
                    "survey_id": "SRV-013",
                    "airline_name": "Horizon Air",
                    "domain": "Support",
                    "rating": 4,
                    "date": "2026-03-02",
                    "verbatim_text": "The support team resolved our question.",
                },
            ),
        ],
    )
    client = FakeGeminiClient(
        {
            "summary": "Cabin customisation lead time was a concern.",
            "pain_points": [
                {
                    "finding": "The configuration process took longer than planned.",
                    "survey_ids": ["SRV-012"],
                }
            ],
            "positive_signals": [],
            "verbatim_quotes": [
                {
                    "survey_id": "SRV-012",
                    "quote": "Cabin customisation took longer than planned.",
                }
            ],
        }
    )

    try:
        result = SurveyIntelligenceAgent(
            qdrant_client=qdrant_client,
            client=client,
        ).run(
            "What are the cabin configuration concerns?",
            airline_name="Horizon Air",
            domain="Cabin Customisation",
            max_rating=2,
        )

        assert result.sources == ["SRV-012"]
        assert result.data["count"] == 1
        assert result.data["average_rating"] == 2
        assert result.trace.agent_name == "SurveyIntelligenceAgent"
        assert client.models.embedding_models == [settings.GEMINI_EMBEDDING_MODEL]
        assert client.models.generated_models == [settings.GEMINI_MODEL]
        assert qdrant_client.collection_exists("surveys")
    finally:
        qdrant_client.close()


def test_contracts_agent_returns_cited_document_sections() -> None:
    qdrant_client = QdrantClient(location=":memory:")
    create_collection(qdrant_client, "commercial_docs")
    content = "Standard cases are acknowledged within two business days."
    source_file = "commercial_sla_and_penalties.md"
    section_title = "Case Acknowledgement Targets"
    qdrant_client.upsert(
        "commercial_docs",
        [
            models.PointStruct(
                id=1,
                vector=[1.0, 0.0, 0.0],
                payload={
                    "source_file": source_file,
                    "section_title": section_title,
                    "content": content,
                },
            )
        ],
    )
    client = FakeGeminiClient(
        {
            "summary": "Standard requests have a two-business-day acknowledgement target.",
            "relevant_clauses": [
                {
                    "finding": "Standard cases have a two-business-day target.",
                    "source_file": source_file,
                    "section_title": section_title,
                    "supporting_quote": content,
                }
            ],
        }
    )

    try:
        result = ContractsRagAgent(
            qdrant_client=qdrant_client,
            client=client,
        ).run("What is the standard acknowledgement target?")

        expected_source = f"{source_file} :: {section_title}"
        assert result.sources == [expected_source]
        assert result.trace.sources == [expected_source]
        assert result.data["retrieved_section_count"] == 1
        assert client.models.embedding_models == [settings.GEMINI_EMBEDDING_MODEL]
        assert client.models.generated_models == [settings.GEMINI_MODEL]
        assert qdrant_client.collection_exists("commercial_docs")
    finally:
        qdrant_client.close()


def test_qdrant_client_scope_closes_only_clients_it_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackedClient:
        closed = False

        def close(self) -> None:
            self.closed = True

    owned_client = TrackedClient()
    monkeypatch.setattr(agent_clients, "create_qdrant_client", lambda: owned_client)

    with agent_clients.qdrant_client_scope(None) as qdrant_client:
        assert qdrant_client is owned_client
    assert owned_client.closed

    shared_client = TrackedClient()
    with agent_clients.qdrant_client_scope(shared_client) as qdrant_client:
        assert qdrant_client is shared_client
    assert not shared_client.closed
