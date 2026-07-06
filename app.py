"""Streamlit demo: upload complaint data, ask questions in plain English, get
SQL-backed answers and charts.

Run with:
    streamlit run app.py

Pipeline: upload (CSV/Excel) -> DuckDB table + data dictionary -> Claude SQL agent
(read-only, retries on error) -> result table -> chart-spec -> Plotly chart.
"""
from __future__ import annotations

import streamlit as st
from anthropic import Anthropic

from agent import SQLAgentError, ask_question
from chart import render_chart
from ingest import build_data_dictionary, build_duckdb_table, load_file_to_dataframe

MODELS = ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001", "claude-fable-5"]
TABLE_NAME = "complaints"

st.set_page_config(page_title="Complaint Data Analyst", layout="wide")
st.title("Complaint Data Analyst (demo)")
st.caption("Upload a complaint export (CSV or Excel), then ask questions in plain English.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Anthropic API key", type="password", help="Used only for this session, never written to disk."
    )
    model = st.selectbox("Model", MODELS, index=0)
    st.divider()
    uploaded_file = st.file_uploader("Upload data file", type=["csv", "xlsx", "xls"])

for key, default in {
    "con": None,
    "schema_context": None,
    "chat_history": [],
    "messages": [],
    "loaded_filename": None,
}.items():
    st.session_state.setdefault(key, default)

if uploaded_file is not None and st.session_state.loaded_filename != uploaded_file.name:
    with st.spinner("Reading file and profiling columns..."):
        df = load_file_to_dataframe(uploaded_file)
        con = build_duckdb_table(df, TABLE_NAME)
        schema_context = build_data_dictionary(con, TABLE_NAME)
    st.session_state.con = con
    st.session_state.schema_context = schema_context
    st.session_state.loaded_filename = uploaded_file.name
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns into table `{TABLE_NAME}`.")

if st.session_state.con is not None:
    with st.expander("Data dictionary (this is the context sent to the model - no PII values in it)"):
        st.text(st.session_state.schema_context)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.write(msg["explanation"])
                with st.expander("SQL used"):
                    st.code(msg["sql"], language="sql")
                st.dataframe(msg["dataframe"])
                if msg.get("figure") is not None:
                    st.plotly_chart(msg["figure"], use_container_width=True)
                if msg.get("truncated"):
                    st.info("Result truncated for display.")
                if msg.get("redacted_columns"):
                    st.caption(f"Redacted before display: {', '.join(msg['redacted_columns'])}")
            else:
                st.write(msg["content"])

    question = st.chat_input("Ask a question about the complaint data...")
    if question:
        if not api_key:
            st.error("Enter your Anthropic API key in the sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = Anthropic(api_key=api_key)
                        result = ask_question(
                            client=client,
                            model=model,
                            con=st.session_state.con,
                            table_name=TABLE_NAME,
                            schema_context=st.session_state.schema_context,
                            question=question,
                            chat_history=st.session_state.chat_history,
                        )
                        figure = render_chart(result.dataframe, result.chart_spec)

                        st.write(result.explanation)
                        with st.expander("SQL used"):
                            st.code(result.sql, language="sql")
                        st.dataframe(result.dataframe)
                        if figure is not None:
                            st.plotly_chart(figure, use_container_width=True)
                        if result.truncated:
                            st.info("Result truncated for display.")
                        if result.redacted_columns:
                            st.caption(f"Redacted before display: {', '.join(result.redacted_columns)}")

                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": f"SQL: {result.sql}\n{result.explanation}"}
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "explanation": result.explanation,
                                "sql": result.sql,
                                "dataframe": result.dataframe,
                                "figure": figure,
                                "truncated": result.truncated,
                                "redacted_columns": result.redacted_columns,
                            }
                        )
                    except SQLAgentError as exc:
                        st.error(str(exc))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Something went wrong: {exc}")
else:
    st.info("Upload a CSV or Excel file in the sidebar to get started.")
