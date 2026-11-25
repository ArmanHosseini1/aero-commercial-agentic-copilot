import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from google.genai import types
from qdrant_client import QdrantClient, models

import aero_agentic_copilot.agents.orchestrator as orchestrator_module
from aero_agentic_copilot.agents.customer360_agent import (
    DEFAULT_DATABASE_PATH,
    Customer360Agent,
    validate_read_only_sql,
)
from aero_agentic_copilot.agents.orchestrator import AeroCommercialOrchestrator
from aero_agentic_copilot.agents.survey_agent import SurveyIntelligenceAgent
from aero_agentic_copilot.config import settings


@dataclass
class FakeEmbedding:
    values: list[float]


@dataclass
class FakeResponse:
    text: str | None = None
    embeddings: list[FakeEmbedding] | None = None


class FakeGeminiModels:
    def __init__(self, response_texts: list[str]) -> None:
        self.response_texts = iter(response_texts)
        self.embedding_models: list[str] = []
        self.generation_calls: list[tuple[str, str, types.GenerateContentConfig | None]] = []

    def embed_content(self, *, model: str, contents: str) -> FakeResponse:
        self.embedding_models.append(model)
        return FakeResponse(embeddings=[FakeEmbedding([1.0, 0.0, 0.0])])

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
    ) -> FakeResponse:
        self.generation_calls.append((model, contents, config))
        return FakeResponse(text=next(self.response_texts))


class FakeGeminiClient:
    def __init__(self, response_texts: list[str]) -> None:
        self.models = FakeGeminiModels(response_texts)
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM airlines",
        "SELECT name FROM airlines;",
        "WITH tier_one AS (SELECT name FROM airlines) SELECT name FROM tier_one",
    ],
)
def test_validate_read_only_sql_accepts_select(sql: str) -> None:
    assert validate_read_only_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE airlines",
        "DELETE FROM airlines",
        "UPDATE airlines SET nps_score=99",
        "ATTACH DATABASE 'other.db' AS other",
        "SELECT * FROM airlines; DROP TABLE airlines;",
    ],
)
def test_validate_read_only_sql_rejects_mutations_and_multiple_statements(
    sql: str,
) -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql(sql)


def test_customer360_executes_mocked_query_against_crm_database() -> None:
    client = FakeGeminiClient(
        ["SELECT name, nps_score FROM airlines WHERE account_tier = 'Tier-1' ORDER BY id"]
    )
    agent = Customer360Agent(database_path=DEFAULT_DATABASE_PATH, client=client)

    result = agent.run("List Tier-1 airline names and their NPS scores.")

    assert result.data["sql"] == (
        "SELECT name, nps_score FROM airlines "
        "WHERE account_tier = 'Tier-1' ORDER BY id"
    )
    assert result.data["rows"] == [
        {"name": "Horizon Air", "nps_score": 19},
        {"name": "NordWind Aviation", "nps_score": 22},
    ]
    assert result.trace.agent_name == "Customer360Agent"
    assert result.trace.latency_ms >= 0


def test_survey_agent_filters_and_aggregates_mocked_results() -> None:
    survey_rows = [
        {
            "survey_id": "SRV-012",
            "airline_name": "Horizon Air",
            "domain": "Delivery",
            "rating": 2,
            "date": "2026-03-01",
            "verbatim_text": "The delivery tracker was not updated.",
        },
        {
            "survey_id": "SRV-013",
            "airline_name": "Horizon Air",
            "domain": "Delivery",
            "rating": 4,
            "date": "2026-03-02",
            "verbatim_text": "The weekly delivery update was useful.",
        },
    ]

    class FakeQdrantClient:
        def __init__(self) -> None:
            self.query_arguments: dict[str, object] = {}

        def query_points(self, **kwargs: object) -> SimpleNamespace:
            self.query_arguments = kwargs
            query_filter = kwargs["query_filter"]
            assert isinstance(query_filter, models.Filter)
            conditions = query_filter.must
            assert isinstance(conditions, list)
            assert {condition.key for condition in conditions} == {
                "airline_name",
                "domain",
                "rating",
            }
            rating_filter = next(
                condition for condition in conditions if condition.key == "rating"
            )
            assert rating_filter.range is not None
            assert rating_filter.range.lte == 4
            return SimpleNamespace(
                points=[
                    SimpleNamespace(id=index, payload=payload)
                    for index, payload in enumerate(survey_rows, start=1)
                ]
            )

    qdrant_client = FakeQdrantClient()
    client = FakeGeminiClient(
        [
            json.dumps(
                {
                    "summary": "Delivery feedback includes both an update issue and a positive update.",
                    "pain_points": [
                        {
                            "finding": "The tracker was not updated.",
                            "survey_ids": ["SRV-012"],
                        }
                    ],
                    "positive_signals": [
                        {
                            "finding": "The weekly update was useful.",
                            "survey_ids": ["SRV-013"],
                        }
                    ],
                    "verbatim_quotes": [
                        {
                            "survey_id": "SRV-012",
                            "quote": "The delivery tracker was not updated.",
                        }
                    ],
                }
            )
        ]
    )
    agent = SurveyIntelligenceAgent(
        qdrant_client=qdrant_client,
        client=client,
    )

    result = agent.run(
        "Summarize Horizon Air delivery feedback.",
        airline_name="Horizon Air",
        domain="Delivery",
        max_rating=4,
    )

    assert result.data["count"] == 2
    assert result.data["average_rating"] == 3
    assert result.sources == ["SRV-012", "SRV-013"]
    assert result.trace.agent_name == "SurveyIntelligenceAgent"
    assert client.models.embedding_models == [settings.GEMINI_EMBEDDING_MODEL]
    assert qdrant_client.query_arguments["collection_name"] == "surveys"


def test_orchestrator_routes_and_synthesizes_with_shared_clients() -> None:
    route = {
        "use_customer360": True,
        "use_surveys": False,
        "use_contracts": False,
        "customer360_query": "List Tier-1 airline names and NPS scores.",
        "survey_query": None,
        "airline_name": None,
        "domain": None,
        "max_rating": None,
        "contracts_query": None,
    }
    synthesis = {
        "executive_summary": "Two Tier-1 airlines have low NPS scores.",
        "key_findings": ["Horizon Air has an NPS score of 19."],
        "recommended_actions": ["Review Horizon Air's account plan."],
        "citations": ["data/crm_sales.db"],
    }
    client = FakeGeminiClient(
        [
            json.dumps(route),
            "SELECT name, nps_score FROM airlines WHERE account_tier = 'Tier-1' ORDER BY id",
            json.dumps(synthesis),
        ]
    )
    qdrant_client = QdrantClient(location=":memory:")
    orchestrator = AeroCommercialOrchestrator(
        qdrant_client=qdrant_client,
        genai_client=client,
    )

    try:
        result = orchestrator.run("Which Tier-1 accounts need attention?")

        assert result.executive_summary == synthesis["executive_summary"]
        assert result.citations == ["data/crm_sales.db"]
        assert [trace.agent_name for trace in result.traces] == [
            "AeroCommercialOrchestrator",
            "Customer360Agent",
            "AeroCommercialOrchestrator",
        ]
        assert orchestrator.survey_agent._qdrant_client is qdrant_client
        assert orchestrator.contracts_agent._qdrant_client is qdrant_client
        assert orchestrator.customer360_agent._client is client
    finally:
        orchestrator.close()
        assert not client.closed
        qdrant_client.close()


def test_orchestrator_closes_clients_it_creates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackedQdrantClient:
        closed = False

        def close(self) -> None:
            self.closed = True

    qdrant_client = TrackedQdrantClient()
    genai_client = FakeGeminiClient([])
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        orchestrator_module.genai,
        "Client",
        lambda api_key: genai_client,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "create_qdrant_client",
        lambda: qdrant_client,
    )

    with AeroCommercialOrchestrator() as orchestrator:
        assert orchestrator.qdrant_client is qdrant_client
        assert orchestrator.genai_client is genai_client

    assert qdrant_client.closed
    assert genai_client.closed
