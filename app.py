# app.py  -- Gemini-powered, intent→SQL templates, safe execution
import os
import ast
import re
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# ----------------------
# Config / API key
# ----------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    # If running on Streamlit Cloud, set in Secrets as GEMINI_API_KEY
    st.error("GEMINI_API_KEY not found in environment. Set it before running the app.")
    st.stop()

# Configure client
genai.configure(api_key=GEMINI_KEY)

# Create model handle (Gemini Flash)
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="GenAI Data Copilot (Gemini)", layout="wide")
st.title("GenAI Data Copilot — Gemini (Safe NL→SQL + NL→Chart)")

# ----------------------
# Helpers
# ----------------------
def df_to_sqlite(conn, df, table_name="data"):
    df.to_sql(table_name, conn, if_exists="replace", index=False)

def parse_model_dict(text):
    """
    Try to extract a dict-like substring and parse to Python dict.
    """
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    piece = text[start:end+1]
    try:
        return ast.literal_eval(piece)
    except Exception:
        try:
            import json
            return json.loads(piece.replace("'", '"'))
        except Exception:
            return None

# SQL template builder — safe, only uses existing columns
def build_sql_from_intent(intent, table_name, df_cols):
    """
    intent: dict expected keys: task ('aggregation'/'list'/'filter'), metric (sum/avg/count), value_column, groupby, filters
    """
    task = intent.get("task")
    if task == "aggregation":
        value_col = intent.get("value_column")
        if value_col not in df_cols:
            return None, f"Column '{value_col}' not found."
        groupby = intent.get("groupby")
        if groupby and groupby not in df_cols:
            return None, f"Group-by column '{groupby}' not found."
        metric = intent.get("metric", "sum").lower()
        metric_fn = "SUM" if metric == "sum" else ("AVG" if metric == "avg" else None)
        if metric_fn is None:
            return None, f"Unsupported metric '{metric}'."
        where_clause = ""
        filters = intent.get("filters", {})
        # simple year_range filter example
        if filters:
            parts = []
            for fcol, cond in filters.items():
                if fcol not in df_cols:
                    return None, f"Filter column '{fcol}' not found."
                # support range filter like {'year_from':2010,'year_to':2020} mapped by model to actual column
                if isinstance(cond, dict) and 'from' in cond and 'to' in cond:
                    parts.append(f"{fcol} BETWEEN {int(cond['from'])} AND {int(cond['to'])}")
            if parts:
                where_clause = " WHERE " + " AND ".join(parts)
        sql = f"SELECT {groupby} AS group_col, {metric_fn}({value_col}) AS metric_val FROM {table_name} {where_clause} GROUP BY {groupby} ORDER BY group_col"
        return sql, None

    elif task == "timeseries":
        # expects value_column and time_column
        value_col = intent.get("value_column")
        time_col = intent.get("time_column")
        if value_col not in df_cols:
            return None, f"Value column '{value_col}' not found."
        if time_col not in df_cols:
            return None, f"Time column '{time_col}' not found."
        # Attempt to group by year if time is unix timestamp or date
        sql = f"SELECT STRFTIME('%Y', datetime({time_col}, 'unixepoch')) AS year, SUM({value_col}) AS total FROM {table_name} GROUP BY year ORDER BY year"
        return sql, None

    elif task == "list":
        # simple SELECT columns LIMIT n
        cols = intent.get("columns", [])
        cols_valid = [c for c in cols if c in df_cols]
        if not cols_valid:
            return None, "No requested columns found in dataset."
        limit = int(intent.get("limit", 20))
        sql = f"SELECT {', '.join(cols_valid)} FROM {table_name} LIMIT {limit}"
        return sql, None

    else:
        return None, f"Unsupported task '{task}' in intent."

# ----------------------
# Prompts (Gemini)
# ----------------------
def intent_prompt_for_sql(nl_query, df_columns, sample_rows=None):
    cols_text = ", ".join(df_columns)
    sample_text = ("\nSample rows:\n" + sample_rows) if sample_rows else ""
    prompt = (
        "Extract a minimal structured intent from this user's request for SQL. Output a Python dict only.\n"
        "Fields: task (one of 'aggregation','timeseries','list'), metric (sum|avg|count), value_column (column name), "
        "groupby (column name), time_column (column name), filters (dict for simple filters), limit (int).\n"
        f"Available columns: {cols_text}.{sample_text}\n\nUser request: {nl_query}\n\nIntent dict:"
    )
    return prompt

def chart_prompt(nl_request, df_columns, sample_rows=None):
    cols_text = ", ".join(df_columns)
    sample_text = ("\nSample rows:\n" + sample_rows) if sample_rows else ""
    prompt = (
        "Create a tiny JSON-like Python dict ONLY describing a chart plan. Example: {'chart_type':'scatter','x':'age','y':'salary','color':'gender'}.\n"
        "Allowed chart_type: scatter, line, bar, hist, pie, box, heatmap. Use only available columns.\n"
        f"Available columns: {cols_text}.{sample_text}\n\nUser request: {nl_request}\n\nChart plan dict:"
    )
    return prompt

# ----------------------
# UI
# ----------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to start. (Note: your previously uploaded file path was: /mnt/data/app (1).py )")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
st.dataframe(df.head())

# prepare sqlite
conn = sqlite3.connect(":memory:")
df_to_sqlite(conn, df, table_name="data")
cols = df.columns.tolist()
sample_rows = df.head(5).to_string()

# Natural language -> SQL (intent extraction)
st.header("NL → SQL (Gemini intent → templated SQL)")

nl_sql = st.text_input("Describe the query (e.g., 'Show yearly sales trend of revenue from 2010 to 2020')", key="nl_sql")
if st.button("Generate and run safe SQL"):
    if not nl_sql.strip():
        st.warning("Please provide an instruction.")
    else:
        # Ask Gemini to produce intent dict
        prompt = intent_prompt_for_sql(nl_sql, cols, sample_rows)
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        st.markdown("**Gemini intent (raw)**")
        st.code(raw)
        intent = parse_model_dict(raw)
        if intent is None:
            st.error("Could not parse intent. Try rephrasing (e.g., 'Sum revenue by year between 2010 and 2020').")
        else:
            # Build safe SQL from template using only valid columns
            sql, err = build_sql_from_intent(intent, "data", cols)
            if err:
                st.error("Could not build SQL: " + err)
            else:
                st.markdown("**Generated SQL (safe template)**")
                st.code(sql)
                try:
                    res = pd.read_sql_query(sql, conn)
                    st.success("Query executed.")
                    st.dataframe(res)
                except Exception as e:
                    st.error(f"SQL execution error: {e}")

# Natural language -> Chart (Gemini chart plan)
st.header("NL → Chart (Gemini chart plan)")

nl_chart = st.text_input("Describe the chart you want (e.g., 'scatter plot of movieId vs rating')", key="nl_chart")
if st.button("Plan and draw chart"):
    if not nl_chart.strip():
        st.warning("Please enter a chart description.")
    else:
        prompt = chart_prompt(nl_chart, cols, sample_rows)
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        st.markdown("**Gemini chart-plan (raw)**")
        st.code(raw)
        plan = parse_model_dict(raw)
        if plan is None:
            st.error("Could not parse chart plan. Try simple phrasing.")
        else:
            # validate plan keys
            chart_type = plan.get("chart_type")
            x = plan.get("x"); y = plan.get("y"); color = plan.get("color")
            missing = [c for c in [x,y,color] if c and c not in cols]
            if missing:
                st.error(f"Columns not found: {missing}")
            else:
                fig, ax = plt.subplots(figsize=(8,4))
                try:
                    if chart_type == "scatter":
                        ax.scatter(df[x], df[y], c=df[color] if color else None, alpha=0.7)
                        ax.set_xlabel(x); ax.set_ylabel(y)
                    elif chart_type == "hist":
                        ax.hist(df[x].dropna(), bins=25)
                    elif chart_type == "bar":
                        res = df.groupby(x)[y].mean().reset_index() if y else df[x].value_counts().reset_index()
                        if y:
                            ax.bar(res[x].astype(str), res[y])
                        else:
                            ax.bar(res['index'].astype(str), res[x])
                            plt.xticks(rotation=45)
                    elif chart_type == "heatmap":
                        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        sns.heatmap(df[num_cols].corr(), annot=True, ax=ax)
                    else:
                        st.error("Unsupported chart type: " + str(chart_type))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error rendering chart: {e}")

st.info("Notes: Gemini is used to extract intent and plan; SQL and chart rendering are created by the app using only validated columns. This prevents hallucination and SQL injection. If something fails, try shorter, more explicit phrasing (e.g., 'sum revenue by year between 2010 and 2020').")
