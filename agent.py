"""Natural-language -> SQL agent (read-only) plus chart-spec proposal, powered by Claude.

Safety model:
- The model may only ever produce a single SELECT (or WITH ... SELECT) statement.
- Any other statement type is rejected before it ever touches DuckDB.
- If a query errors, the error is fed back to the model for a bounded number of
  self-correction retries.
- Result sets are capped in size, and any column that looks like a direct identifier
  (name/email/phone/etc.) is redacted before being shown or handed back to the model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import duckdb
import pandas as pd
from anthropic import Anthropic

from ingest import PII_COLUMN_PATTERN

MAX_ROWS_RETURNED = 5000
MAX_SQL_ATTEMPTS = 3

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|COPY|PRAGMA|EXPORT|IMPORT|CALL|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a data analyst assistant. You answer questions about a table called {table_name} \
in a DuckDB database by writing a single read-only SQL SELECT query.

Schema and data dictionary:
{schema}

Rules:
- Only ever write a single SELECT statement (or WITH ... SELECT). Never write INSERT/UPDATE/DELETE/DROP/ALTER \
or any other statement that modifies data or schema.
- Use DuckDB SQL syntax.
- Quote column names with double quotes if they need it.
- Prefer aggregating (COUNT, AVG, GROUP BY, etc.) over returning raw rows, unless the question specifically \
asks for individual records.
- Avoid selecting columns marked "direct identifier" in the schema unless the question specifically requires \
looking up a single individual - prefer counts/aggregates over those columns instead.
- If the question implies a chart (a trend, a breakdown by category, a comparison, a distribution), propose \
one; otherwise set "chart" to null and let a table answer the question.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{
  "sql": "<the SELECT statement>",
  "explanation": "<one sentence, plain language, describing what the query returns>",
  "chart": {{"type": "bar|line|pie|scatter|histogram", "x": "<column or alias>", "y": "<column or alias, or null for histogram>", "group_by": "<column or alias, or null>", "title": "<short chart title>"}} or null
}}
"""


@dataclass
class QueryResult:
    sql: str
    explanation: str
    chart_spec: dict | None
    dataframe: pd.DataFrame
    attempts: list[str] = field(default_factory=list)
    truncated: bool = False
    redacted_columns: list[str] = field(default_factory=list)


class SQLAgentError(Exception):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise SQLAgentError(f"Could not find a JSON object in the model's response:\n{text}")
    return json.loads(match.group(0))


def _validate_sql(sql: str) -> None:
    stripped = sql.strip().rstrip(";")
    if not re.match(r"(?is)^\s*(WITH|SELECT)\b", stripped):
        raise SQLAgentError("Only SELECT (or WITH ... SELECT) statements are allowed.")
    if ";" in stripped:
        raise SQLAgentError("Multiple statements are not allowed.")
    if FORBIDDEN_KEYWORDS.search(stripped):
        raise SQLAgentError("Query contains a disallowed keyword.")


def _redact_pii_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    redacted = [c for c in df.columns if PII_COLUMN_PATTERN.search(str(c))]
    if redacted:
        df = df.copy()
        for c in redacted:
            df[c] = "*** redacted ***"
    return df, redacted


def ask_question(
    client: Anthropic,
    model: str,
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    schema_context: str,
    question: str,
    chat_history: list[dict] | None = None,
) -> QueryResult:
    system = SYSTEM_PROMPT.format(table_name=table_name, schema=schema_context)
    messages = list(chat_history or [])
    messages.append({"role": "user", "content": question})

    attempts_log: list[str] = []
    last_error = None

    for attempt in range(MAX_SQL_ATTEMPTS):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        raw_text = "".join(block.text for block in response.content if block.type == "text")

        try:
            payload = _extract_json(raw_text)
            sql = payload["sql"]
            _validate_sql(sql)
            df = con.execute(sql).fetchdf()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure triggers a retry
            last_error = str(exc)
            attempts_log.append(f"Attempt {attempt + 1} failed: {last_error}")
            messages.append({"role": "assistant", "content": raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That query failed with this error: {last_error}. Please fix it and respond "
                        "again with the same JSON format."
                    ),
                }
            )
            continue

        truncated = len(df) > MAX_ROWS_RETURNED
        if truncated:
            df = df.head(MAX_ROWS_RETURNED)

        df, redacted_columns = _redact_pii_columns(df)

        return QueryResult(
            sql=sql,
            explanation=payload.get("explanation", ""),
            chart_spec=payload.get("chart"),
            dataframe=df,
            attempts=attempts_log,
            truncated=truncated,
            redacted_columns=redacted_columns,
        )

    raise SQLAgentError(
        f"Could not produce a working query after {MAX_SQL_ATTEMPTS} attempts. Last error: {last_error}"
    )
