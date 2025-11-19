# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import re
import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# Page / Model Setup
# -----------------------------
st.set_page_config(page_title="GenAI Data Copilot (NL→SQL & NL→Chart)", layout="wide")
st.title("🤖📈 GenAI Data Copilot — NL→SQL & NL→Chart")
st.caption("Use natural language to run queries and build charts. Safety checks included.")

@st.cache_resource
def load_llm():
    try:
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return gen
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None

llm = load_llm()

# -----------------------------
# Helper: In-memory SQLite from DataFrame
# -----------------------------
def df_to_sqlite(conn, df, table_name="data"):
    df.to_sql(table_name, conn, if_exists="replace", index=False)

# -----------------------------
# Helper: Sanitize SQL
# -----------------------------
ALLOWED_SQL_START = ("select", "with", "pragma")  # we'll disallow everything except safe selects; pragma only allowed read-only
DISALLOWED_PATTERNS = [
    r";",  # disallow multiple statements separated by semicolons
    r"drop\s", r"delete\s", r"insert\s", r"update\s", r"alter\s",
    r"attach\s", r"detach\s", r"pragma\s+", r"vacuum\s", r"create\s", r"replace\s"
]
def is_sql_safe(sql_text: str) -> bool:
    sql = sql_text.strip().lower()
    # single statement check (no semicolons)
    if ";" in sql[:-1]:
        return False
    # must start with select or with
    if not sql.startswith("select") and not sql.startswith("with"):
        return False
    # disallow dangerous keywords
    for pat in DISALLOWED_PATTERNS:
        if re.search(pat, sql):
            return False
    return True

# -----------------------------
# Helper: Parse LLM JSON-like chart plan
# Expectation: LLM returns either a Python dict literal or simple JSON like:
# {"chart_type":"scatter","x":"age","y":"salary"}
# -----------------------------
def parse_chart_plan(text: str):
    # try to extract a JSON/dict substring
    # common LLM outputs include backticks or surrounding text — find first brace
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    sub = text[start:end+1]
    try:
        plan = ast.literal_eval(sub)
        if isinstance(plan, dict):
            return plan
    except Exception:
        # try json-like by replacing single quotes with double quotes
        try:
            import json
            sub2 = sub.replace("'", '"')
            plan = json.loads(sub2)
            return plan
        except Exception:
            return None
    return None

# -----------------------------
# LLM Prompts
# -----------------------------
def make_sql_prompt(nl_query: str, table_name="data", columns=None, sample_rows=None):
    # Provide a constrained instruction: output only the SQL SELECT statement; no explanation.
    cols_text = ""
    if columns:
        cols_text = f"\nAvailable columns: {', '.join(columns)}."
    sample_text = ""
    if sample_rows is not None:
        sample_text = f"\nHere are a few sample rows:\n{sample_rows}"
    prompt = (
        "You are a SQL generator. Produce a single valid SQLite SELECT statement that answers the user's request. "
        "Do NOT use semicolons, do NOT output any commentary — output ONLY the SQL statement. "
        f"Table name: {table_name}.{cols_text}{sample_text}\n\nUser request: {nl_query}\n\nSQL:"
    )
    return prompt

def make_chart_prompt(nl_request: str, columns=None, sample_rows=None):
    # Request a small dict describing chart: chart_type, x, y (optional), agg(optional), color(optional)
    cols_text = ""
    if columns:
        cols_text = f"\nAvailable columns: {', '.join(columns)}."
    sample_text = ""
    if sample_rows is not None:
        sample_text = f"\nSample rows:\n{sample_rows}"
    prompt = (
        "You are a chart-planning assistant. From the user's natural language, produce a small Python-dict describing the chart. "
        "Output ONLY a Python dict literal (e.g. {'chart_type':'scatter','x':'age','y':'salary'}) with keys: chart_type (one of 'scatter','line','bar','hist','box','pie','heatmap'), "
        "and the relevant columns (x,y,aggregate,color) as strings. Do NOT include any explanation, only the dict.\n"
        f"{cols_text}{sample_text}\n\nUser request: {nl_request}\n\nChart plan:"
    )
    return prompt

# -----------------------------
# UI: Upload + show DataFrame
# -----------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to start — then ask in natural language to query or chart your data.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
st.dataframe(df.head())

# prepare sqlite
conn = sqlite3.connect(":memory:")
df_to_sqlite(conn, df, table_name="data")

columns = df.columns.tolist()
sample_rows = df.head(5).to_string()

# -----------------------------
# Panel: NL -> SQL
# -----------------------------
st.header("1) Natural language → SQL (safe execution)")
nl_sql = st.text_input("Ask a data question (example: 'Top 10 movies by average rating')", key="nl_sql")
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Generate SQL"):
        if not llm:
            st.error("LLM not loaded.")
        else:
            prompt = make_sql_prompt(nl_sql, table_name="data", columns=columns, sample_rows=sample_rows)
            gen = llm(prompt, max_length=200, do_sample=False)
            raw_sql = gen[0]["generated_text"].strip()
            # extract first line that looks like SELECT
            # sometimes model may include newlines; find first SELECT
            m = re.search(r"select[\s\S]*", raw_sql, flags=re.IGNORECASE)
            sql_candidate = m.group(0) if m else raw_sql
            # clean up trailing periods or accidental full stops
            sql_candidate = sql_candidate.strip().rstrip(".")
            st.code(sql_candidate, language="sql")

            # validate
            if is_sql_safe(sql_candidate):
                try:
                    res = pd.read_sql_query(sql_candidate, conn)
                    st.success("Query executed successfully — showing results:")
                    st.dataframe(res.head(200))
                    # small summary
                    st.markdown(f"**Rows returned:** {len(res)}")
                except Exception as e:
                    st.error(f"SQL execution error: {e}")
            else:
                st.error("Generated SQL failed safety checks. The app will not execute it.")
                st.write("Try rephrasing or use simpler wording. As a fallback, the app can run a rule-based aggregate if you want.")

with col2:
    st.markdown("**Or ask for a quick rule-based aggregate (safe fallback)**")
    agg_option = st.selectbox("Quick aggregate:", ["None","Count rows","Average of a column","Group by a column (count)"])
    if agg_option == "Average of a column":
        avg_col = st.selectbox("Choose column for average:", [c for c in columns if np.issubdtype(df[c].dtype, np.number)])
        if st.button("Run average"):
            val = df[avg_col].mean()
            st.write(f"Average of `{avg_col}` = {val:.4f}")
    elif agg_option == "Count rows":
        if st.button("Count rows"):
            st.write(f"Row count = {len(df)}")
    elif agg_option == "Group by a column (count)":
        gid = st.selectbox("Group by column:", columns)
        if st.button("Run group count"):
            res = df.groupby(gid).size().reset_index(name="count").sort_values("count", ascending=False)
            st.dataframe(res.head(200))

# -----------------------------
# Panel: NL -> Chart
# -----------------------------
st.header("2) Natural language → Chart (LLM suggests a chart plan)")

nl_chart = st.text_input("Describe the chart you want (example: 'Scatter plot of age vs salary, color by gender')", key="nl_chart")
if st.button("Generate Chart Plan"):
    if not llm:
        st.error("LLM not loaded.")
    else:
        prompt = make_chart_prompt(nl_chart, columns=columns, sample_rows=sample_rows)
        gen = llm(prompt, max_length=200, do_sample=False)
        raw = gen[0]["generated_text"].strip()
        st.write("LLM output (raw):")
        st.code(raw)

        plan = parse_chart_plan(raw)
        if plan is None:
            st.error("Could not parse chart plan. Try simpler phrasing (e.g., 'scatter age salary').")
        else:
            # Validate keys
            chart_type = plan.get("chart_type")
            x = plan.get("x")
            y = plan.get("y")
            agg = plan.get("agg", None)
            color = plan.get("color", None)

            # Basic validation
            if chart_type not in ["scatter","line","bar","hist","box","pie","heatmap"]:
                st.error(f"Unsupported chart_type: {chart_type}")
            else:
                # Ensure referenced columns exist
                missing_cols = [c for c in [x,y,color] if c and c not in columns]
                if missing_cols:
                    st.error(f"Columns not found in dataset: {missing_cols}")
                else:
                    st.success("Chart plan parsed and validated. Rendering chart...")
                    fig, ax = plt.subplots(figsize=(7,4))
                    try:
                        if chart_type == "scatter":
                            ax.scatter(df[x], df[y], alpha=0.7, c=df[color] if color else None)
                            ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
                            st.pyplot(fig)
                        elif chart_type == "line":
                            ax.plot(df[x], df[y])
                            ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{y} over {x}")
                            st.pyplot(fig)
                        elif chart_type == "bar":
                            if agg == "count" or agg is None:
                                res = df.groupby(x).size().reset_index(name="count").sort_values("count", ascending=False)
                                ax.bar(res[x].astype(str).head(20), res["count"].head(20))
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                            elif agg == "mean":
                                res = df.groupby(x)[y].mean().reset_index()
                                ax.bar(res[x].astype(str).head(20), res[y].head(20))
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                        elif chart_type == "hist":
                            ax.hist(df[x].dropna(), bins=25)
                            ax.set_title(f"Histogram of {x}")
                            st.pyplot(fig)
                        elif chart_type == "box":
                            sns.boxplot(x=x, y=y, data=df, ax=ax)
                            st.pyplot(fig)
                        elif chart_type == "pie":
                            counts = df[x].value_counts().head(20)
                            ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
                            ax.set_title(f"Pie of {x}")
                            st.pyplot(fig)
                        elif chart_type == "heatmap":
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            if len(numeric_cols) < 2:
                                st.error("Not enough numeric columns for heatmap.")
                            else:
                                corr = df[numeric_cols].corr()
                                sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
                                st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error rendering chart: {e}")

# -----------------------------
# Panel: NL → Natural-Language Insights (Composite)
# -----------------------------
st.header("3) Natural language insights (RAG style: retrieve → generate)")
nl_insight = st.text_input("Ask an insight question (e.g., 'Summarize rating distribution and top movies')", key="nl_insight")
if st.button("Get Insight"):
    # We'll build a short rule-based summary, then ask LLM to rewrite it into a natural, pretty report.
    # This avoids hallucinations: LLM only paraphrases a factual summary we produce.
    # Build rule-based facts:
    facts = []
    facts.append(f"Rows: {len(df)}; Columns: {len(df.columns)}.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        for c in numeric_cols:
            s = df[c].dropna()
            facts.append(f"Column `{c}`: mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}.")
    # a simple top-n for categorical columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in cat_cols[:3]:
        top = df[c].value_counts().head(3)
        top_str = "; ".join([f"{idx}({cnt})" for idx,cnt in top.items()])
        facts.append(f"Column `{c}` top values: {top_str}.")
    # top correlated pair
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_idx = np.unravel_index(corr.values.argmax(), corr.shape)
        a = corr.index[max_idx[0]]
        b = corr.columns[max_idx[1]]
        facts.append(f"Strongest numeric correlation: `{a}` vs `{b}` = {df[a].corr(df[b]):.2f}.")

    rule_summary = "\n".join(facts)
    st.markdown("### Rule-based summary (facts):")
    st.code(rule_summary)

    # Now ask LLM to rewrite prettily
    if llm:
        prompt = (
            "Rewrite the following facts into a short, clear, professional data-insights paragraph. "
            "Do not invent new facts; just rewrite clearly.\n\n"
            f"{rule_summary}"
        )
        res = llm(prompt, max_length=200)
        pretty = res[0]["generated_text"]
        st.markdown("### 🤖 GenAI Insight (rewritten)")
        st.write(pretty)
    else:
        st.info("LLM not available — showing rule-based facts.")

# -----------------------------
# Footer: tips & safety
# -----------------------------
st.markdown("---")
st.markdown("**How to use effectively**")
st.markdown(
    "- For NL→SQL: ask focused questions (e.g., 'Top 5 movies by average rating' or 'Average rating per movieId').\n"
    "- For NL→Chart: start simple (e.g., 'scatter age salary' or 'histogram of rating').\n"
    "- If the LLM output fails safety/validation, rephrase the question or use the rule-based fallbacks.\n"
)
st.markdown("**Safety**: All generated SQL is checked to allow only read-only SELECT-style queries. Charts are validated for existing columns.")

