"""Data ingestion: load a CSV/Excel upload into DuckDB and build a data dictionary.

The data dictionary is the context we hand to the SQL agent so it knows what each
column means without ever seeing raw PII rows.
"""
from __future__ import annotations

import re

import duckdb
import pandas as pd


def load_file_to_dataframe(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame with clean column names."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    df.columns = [_clean_column_name(c) for c in df.columns]
    return df


def _clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "column"


def build_duckdb_table(df: pd.DataFrame, table_name: str = "complaints") -> duckdb.DuckDBPyConnection:
    """Load the dataframe into an in-memory DuckDB table, best-effort casting date-like columns."""
    con = duckdb.connect(database=":memory:")
    con.register("_raw_df", df)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _raw_df")

    for col in df.columns:
        if _looks_like_datetime_column(col, df[col]):
            for expr in (
                f'TRY_STRPTIME("{col}", \'%m/%d/%Y %H:%M\')',
                f'TRY_CAST("{col}" AS TIMESTAMP)',
            ):
                try:
                    con.execute(
                        f'ALTER TABLE {table_name} ALTER COLUMN "{col}" TYPE TIMESTAMP USING {expr}'
                    )
                    break
                except Exception:
                    continue
    return con


def _looks_like_datetime_column(name: str, series: pd.Series) -> bool:
    if not any(k in name for k in ("date", "datetime", "_at", "_time")):
        return False
    sample = series.dropna().astype(str).head(20)
    if len(sample) == 0:
        return False
    hits = sample.str.match(r"^\d{1,2}/\d{1,2}/\d{4}").sum()
    return hits / len(sample) > 0.6


# Columns matching these patterns hold direct identifiers. We still describe their
# type/cardinality to the model (so it can filter/join/count on them) but we never put
# actual values from these columns into the prompt.
PII_COLUMN_PATTERN = re.compile(
    r"(first_name|last_name|full_name|^name$|email|phone|ssn|ssn_|address|street|"
    r"dob|birth|drivers_license|passport|account_number)",
    re.IGNORECASE,
)


def build_data_dictionary(
    con: duckdb.DuckDBPyConnection, table_name: str = "complaints", max_categories: int = 12
) -> str:
    """Produce a compact text description of every column for the SQL agent's system prompt.

    Sample/category values are withheld for columns that look like direct identifiers
    (names, emails, phone numbers, addresses, etc.) so PII never gets embedded in the
    prompt sent to the model - only the column's existence, type, and distinct count do.
    """
    schema = con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
    total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    lines = [f"Table: {table_name} ({total_rows} rows)", ""]
    for _, row in schema.iterrows():
        col, dtype = row["name"], row["type"]
        try:
            n_distinct = con.execute(f'SELECT COUNT(DISTINCT "{col}") FROM {table_name}').fetchone()[0]
        except Exception:
            n_distinct = None

        if PII_COLUMN_PATTERN.search(col):
            distinct_note = f", {n_distinct} distinct" if n_distinct is not None else ""
            lines.append(f"- {col} ({dtype}{distinct_note}): direct identifier, values withheld from context")
            continue

        detail = ""
        try:
            if dtype in ("BIGINT", "DOUBLE", "INTEGER", "DECIMAL", "HUGEINT", "FLOAT"):
                mn, mx, avg = con.execute(
                    f'SELECT MIN("{col}"), MAX("{col}"), AVG("{col}") FROM {table_name}'
                ).fetchone()
                detail = f"range {mn} to {mx}, avg {round(avg, 2) if avg is not None else 'n/a'}"
            elif dtype in ("TIMESTAMP", "DATE"):
                mn, mx = con.execute(f'SELECT MIN("{col}"), MAX("{col}") FROM {table_name}').fetchone()
                detail = f"range {mn} to {mx}"
            elif n_distinct is not None and n_distinct <= max_categories:
                vals = con.execute(
                    f'SELECT DISTINCT "{col}" FROM {table_name} WHERE "{col}" IS NOT NULL LIMIT {max_categories}'
                ).fetchdf()[col].tolist()
                detail = f"values: {vals}"
            else:
                sample = con.execute(
                    f'SELECT "{col}" FROM {table_name} WHERE "{col}" IS NOT NULL LIMIT 3'
                ).fetchdf()[col].tolist()
                detail = f"e.g. {sample}"
        except Exception:
            pass

        distinct_note = f", {n_distinct} distinct" if n_distinct is not None else ""
        lines.append(f"- {col} ({dtype}{distinct_note}): {detail}")

    return "\n".join(lines)
