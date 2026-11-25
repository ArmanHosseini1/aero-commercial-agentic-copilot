import re
import sqlite3
from contextlib import closing
from pathlib import Path
from time import perf_counter

from google import genai
from google.genai import types

from aero_agentic_copilot.agents._agent_clients import GeminiClient
from aero_agentic_copilot.agents.schemas import AgentOutput, AgentStepTrace
from aero_agentic_copilot.config import settings
from aero_agentic_copilot.llm_utils import call_with_retry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "crm_sales.db"
DATABASE_TABLES = ("airlines", "aircraft_orders", "contract_milestones")
FENCED_SQL_PATTERN = re.compile(
    r"\A```(?:sql|sqlite)?[ \t]*\r?\n(.*?)\r?\n```\Z",
    re.IGNORECASE | re.DOTALL,
)
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(?:DROP|DELETE|UPDATE|INSERT|ALTER|ATTACH|DETACH|PRAGMA|REPLACE|CREATE)\b",
    re.IGNORECASE,
)
SQL_WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")


def _has_top_level_select(sql: str, start: int) -> bool:
    depth = 0
    index = start
    while index < len(sql):
        if sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            if newline == -1:
                return False
            index = newline + 1
            continue
        if sql.startswith("/*", index):
            comment_end = sql.find("*/", index + 2)
            if comment_end == -1:
                return False
            index = comment_end + 2
            continue

        character = sql[index]
        if character in ("'", '"', "`", "["):
            closing_quote = "]" if character == "[" else character
            index += 1
            while index < len(sql):
                if sql[index] == closing_quote:
                    if index + 1 < len(sql) and sql[index + 1] == closing_quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue
        if character == "(":
            depth += 1
            index += 1
            continue
        if character == ")":
            depth -= 1
            index += 1
            continue
        if depth == 0:
            word_match = SQL_WORD_PATTERN.match(sql, index)
            if word_match is not None:
                if word_match.group().upper() == "SELECT":
                    return True
                index = word_match.end()
                continue
        index += 1
    return False


def validate_read_only_sql(sql: str) -> str:
    if not isinstance(sql, str):
        raise ValueError("SQL query must be a string.")

    query = sql.strip()
    if query.startswith("```"):
        fenced_match = FENCED_SQL_PATTERN.fullmatch(query)
        if fenced_match is None:
            raise ValueError("SQL code fence is malformed.")
        query = fenced_match.group(1).strip()

    if not query:
        raise ValueError("SQL query cannot be empty.")
    if ";" in query[:-1]:
        raise ValueError("Only a single SQL statement is allowed.")
    if FORBIDDEN_SQL_PATTERN.search(query):
        raise ValueError("SQL query contains a forbidden keyword.")

    start_match = re.match(r"(SELECT|WITH)\b", query, re.IGNORECASE)
    if start_match is None:
        raise ValueError("SQL query must start with SELECT or WITH.")
    if (
        start_match.group(1).upper() == "WITH"
        and not _has_top_level_select(query, start_match.end())
    ):
        raise ValueError("A WITH query must contain a SELECT statement.")

    return query


class Customer360Agent:
    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        client: GeminiClient | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        if client is None:
            if not settings.GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY is required to use Customer360Agent.")
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self._owns_client = True
        else:
            self._owns_client = False
        self._client = client

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
            self._owns_client = False

    def inspect_schema(self) -> str:
        database_uri = f"{self.database_path.resolve().as_uri()}?mode=ro"
        placeholders = ", ".join("?" for _ in DATABASE_TABLES)
        with closing(sqlite3.connect(database_uri, uri=True)) as connection:
            table_rows = connection.execute(
                f"""
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'table' AND name IN ({placeholders})
                """,
                DATABASE_TABLES,
            ).fetchall()

        schema_by_name = {name: create_sql for name, create_sql in table_rows}
        missing_tables = [name for name in DATABASE_TABLES if name not in schema_by_name]
        if missing_tables:
            raise ValueError(
                "CRM database is missing required tables: "
                + ", ".join(missing_tables)
            )
        return "\n\n".join(
            f"{name}:\n{schema_by_name[name]}" for name in DATABASE_TABLES
        )

    def _generate_sql(self, business_question: str, schema: str) -> str:
        prompt = f"""Generate one read-only SQLite query to answer the business question.
Use only the schema below. Return only a single SELECT statement (a WITH ... SELECT
query is allowed), with no markdown fence, explanation, or other text. Do not write
to the database.

SQLite schema:
{schema}

Business question:
{business_question}
"""
        response = call_with_retry(
            lambda: self._client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(),
            )
        )
        if not response.text or not response.text.strip():
            raise ValueError("Gemini returned an empty SQL query.")
        return validate_read_only_sql(response.text)

    def _execute_read_only_sql(self, sql: str) -> list[dict[str, object]]:
        database_uri = f"{self.database_path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(database_uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            rows = connection.execute(sql).fetchall()
        return [dict(row) for row in rows]

    def run(self, business_question: str) -> AgentOutput:
        if not business_question.strip():
            raise ValueError("Business question cannot be empty.")

        started_at = perf_counter()
        schema = self.inspect_schema()
        sql = self._generate_sql(business_question, schema)
        rows = self._execute_read_only_sql(sql)
        latency_ms = (perf_counter() - started_at) * 1000
        source = "data/crm_sales.db"
        result_summary = (
            f"Question: {business_question} "
            f"Read-only query returned {len(rows)} row(s)."
        )

        trace = AgentStepTrace(
            agent_name="Customer360Agent",
            tool_used="Gemini SQL generation and read-only SQLite",
            query_or_input=sql,
            result_summary=result_summary,
            sources=[source],
            latency_ms=latency_ms,
        )
        return AgentOutput(
            summary=result_summary,
            data={"sql": sql, "rows": rows},
            sources=[source],
            trace=trace,
        )
