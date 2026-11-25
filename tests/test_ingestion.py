from pathlib import Path

import pytest
from qdrant_client import QdrantClient

import aero_agentic_copilot.rag.ingestion as ingestion
from aero_agentic_copilot.rag.ingestion import (
    EmbeddingRateLimiter,
    create_and_populate_collection,
    load_surveys,
    recreate_collections,
    split_markdown_sections,
)


def test_load_surveys_includes_all_rows_and_payload_fields() -> None:
    records = load_surveys()

    assert len(records) == 80
    assert set(records[0][2]) == {
        "survey_id",
        "airline_name",
        "domain",
        "rating",
        "date",
        "verbatim_text",
    }


def test_split_markdown_sections_covers_all_commercial_documents() -> None:
    records = split_markdown_sections()

    assert records
    assert {
        payload["source_file"]
        for _point_id, _text, payload in records
    } == {
        "cabin_customisation_catalog.md",
        "commercial_sla_and_penalties.md",
        "aircraft_family_positioning.md",
        "pricing_and_concessions_guidelines.md",
    }
    assert all(payload["section_title"] and payload["content"] for _, _, payload in records)


def test_embedding_rate_limiter_waits_for_quota_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    delays: list[float] = []

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        delays.append(seconds)
        now += seconds

    monkeypatch.setattr(ingestion, "monotonic", lambda: now)
    monkeypatch.setattr(ingestion, "sleep", fake_sleep)
    limiter = EmbeddingRateLimiter(items_per_minute=3, window_seconds=60.0)

    limiter.wait_for_capacity(3)
    limiter.wait_for_capacity(1)

    assert delays == [60.0]


def test_recreate_collection_uses_embedding_dimension_and_replaces_collection(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "qdrant"
    client = QdrantClient(path=str(storage_path))
    records = [
        (1, "First survey", {"survey_id": "SUR-0001"}),
        (2, "Second survey", {"survey_id": "SUR-0002"}),
    ]

    try:
        create_and_populate_collection(
            client,
            "surveys",
            records,
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )
        assert client.get_collection("surveys").config.params.vectors.size == 3
        assert client.count("surveys", exact=True).count == 2

        recreate_collections(client, ("surveys",))
        create_and_populate_collection(
            client,
            "surveys",
            records[:1],
            [[0.7, 0.8]],
        )
        assert client.get_collection("surveys").config.params.vectors.size == 2
        assert client.count("surveys", exact=True).count == 1
    finally:
        client.close()
