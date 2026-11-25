import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
from google.genai import types

from aero_agentic_copilot.agents.customer360_agent import (
    Customer360Agent,
    validate_read_only_sql,
)


@dataclass
class FakeResponse:
    text: str | None


class FakeModels:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompt = ""

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
    ) -> FakeResponse:
        assert model
        self.prompt = contents
        return FakeResponse(text=self.response_text)


class FakeClient:
    def __init__(self, response_text: str) -> None:
        self.models = FakeModels(response_text)

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("raw_sql", "expected_sql"),
    [
        ("SELECT name FROM airlines", "SELECT name FROM airlines"),
        ("```sql\nSELECT name FROM airlines;\n```", "SELECT name FROM airlines;"),
        (
            "WITH tier_one AS (SELECT name FROM airlines) SELECT name FROM tier_one",
            "WITH tier_one AS (SELECT name FROM airlines) SELECT name FROM tier_one",
        ),
    ],
)
def test_validate_read_only_sql_accepts_select_queries(
    raw_sql: str,
    expected_sql: str,
) -> None:
    assert validate_read_only_sql(raw_sql) == expected_sql


@pytest.mark.parametrize(
    "raw_sql",
    [
        "",
        "DROP TABLE airlines",
        "SELECT * FROM airlines; DELETE FROM airlines",
        "WITH tier_one AS (SELECT name FROM airlines)",
        "PRAGMA table_info(airlines)",
        "UPDATE airlines SET name = 'Changed'",
        "Here is your query: SELECT * FROM airlines",
    ],
)
def test_validate_read_only_sql_rejects_invalid_queries(raw_sql: str) -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql(raw_sql)


def create_test_database(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE airlines (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE aircraft_orders (order_id INTEGER PRIMARY KEY);
            CREATE TABLE contract_milestones (milestone_id INTEGER PRIMARY KEY);
            INSERT INTO airlines (id, name) VALUES (1, 'Horizon Air');
            """
        )


def test_customer360_agent_returns_rows_sql_and_trace(tmp_path: Path) -> None:
    database_path = tmp_path / "crm_sales.db"
    create_test_database(database_path)
    client = FakeClient("```sql\nSELECT name FROM airlines WHERE id = 1;\n```")
    agent = Customer360Agent(database_path=database_path, client=client)

    result = agent.run("Which airline has ID 1?")

    assert "aircraft_orders" in client.models.prompt
    assert "Which airline has ID 1?" in client.models.prompt
    assert result.data == {
        "sql": "SELECT name FROM airlines WHERE id = 1;",
        "rows": [{"name": "Horizon Air"}],
    }
    assert result.trace.agent_name == "Customer360Agent"
    assert result.trace.sources == ["data/crm_sales.db"]
    assert result.trace.latency_ms >= 0


def test_customer360_agent_does_not_execute_rejected_sql(tmp_path: Path) -> None:
    database_path = tmp_path / "crm_sales.db"
    create_test_database(database_path)
    client = FakeClient("DELETE FROM airlines;")
    agent = Customer360Agent(database_path=database_path, client=client)

    with pytest.raises(ValueError, match="forbidden"):
        agent.run("Remove airline records")

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM airlines").fetchone()[0] == 1
