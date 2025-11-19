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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="GenAI Data Copilot — Full", layout="wide")
st.title("🤖📊 GenAI Data Copilot — Full Suite")
st.caption("NL→SQL, NL→Chart, AutoEDA, Clean-up Assistant, RAG insights, AutoML. Powered by google/flan-t5-small (lightweight).")

# -------------------------
# Load LLM (FLAN-T5-small)
# -------------------------
@st.cache_resource
def load_llm():
    try:
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return gen
    except Exception as e:
        st.error(f"LLM load failed: {e}")
        return None

llm = load_llm()

# -------------------------
# Utilities: SQL safety
# -------------------------
ALLOWED_START = ("select", "with")
DISALLOWED_PATTERNS = [
    r";", r"drop\s", r"delete\s", r"insert\s", r"update\s", r"alter\s",
    r"attach\s", r"detach\s", r"vacuum\s", r"create\s", r"replace\s", r"pragma\s"
]
def is_sql_safe(sql_text: str) -> bool:
    sql = sql_text.strip().lower()
    if ";" in sql[:-1]:
        return False
    if not (sql.startswith("select") or sql.startswith("with")):
        return False
    for pat in DISALLOWED_PATTERNS:
        if re.search(pat, sql):
            return False
    return True

# -------------------------
# Utilities: parse LLM chart plan
# -------------------------
def parse_chart_plan(text: str):
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    sub = text[start:end+1]
    try:
        plan = ast.literal_eval(sub)
        if isinstance(plan, dict):
            return plan
    except Exception:
        try:
            import json
            plan = json.loads(sub.replace("'", '"'))
            return plan
        except Exception:
            return None
    return None

# -------------------------
# Prompt makers
# -------------------------
def make_sql_prompt(nl_query: str, table_name="data", columns=None, sample=None):
    cols = f"\nAvailable columns: {', '.join(columns)}." if columns else ""
    sample_text = f"\nSample rows:\n{sample}" if sample is not None else ""
    prompt = (
        "You are a SQL generator for SQLite. Produce a single SELECT statement (no semicolons, no commentary) "
        "that answers the user's question. Use the table name given and available columns. "
        f"Table: {table_name}.{cols}{sample_text}\n\nUser question: {nl_query}\n\nSQL:"
    )
    return prompt

def make_chart_prompt(nl_request: str, columns=None, sample=None):
    cols = f"\nAvailable columns: {', '.join(columns)}." if columns else ""
    sample_text = f"\nSample rows:\n{sample}" if sample is not None else ""
    prompt = (
        "You are a chart-planning assistant. Output ONLY a Python-dict literal describing a chart plan. "
        "Keys: chart_type in ['scatter','line','bar','hist','box','pie','heatmap'], x (str), y (str, optional), agg (optional: 'mean'|'count'), color (optional). "
        f"{cols}{sample_text}\n\nUser request: {nl_request}\n\nChart plan:"
    )
    return prompt

def paraphrase_prompt(text: str):
    return (
        "Rewrite the following text into a short, clear, professional paragraph. Do NOT invent or add new facts. "
        "Make it easy to read.\n\n" + text
    )

# -------------------------
# Rule-based helpers (facts & EDA)
# -------------------------
def basic_facts(df: pd.DataFrame, max_cat_top=3):
    facts = []
    facts.append(f"Rows: {len(df)}; Columns: {len(df.columns)}.")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        s = df[c].dropna()
        facts.append(f"Column `{c}`: mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}.")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in cat_cols[:5]:
        vc = df[c].value_counts().head(max_cat_top)
        top_str = "; ".join([f"{idx}({cnt})" for idx, cnt in vc.items()])
        facts.append(f"Categorical `{c}` top: {top_str}.")
    return "\n".join(facts)

def auto_eda(df: pd.DataFrame):
    out = []
    out.append(f"Rows: {len(df)}; Columns: {len(df.columns)}.")
    # missing
    miss = df.isnull().sum()
    total_missing = int(miss.sum())
    out.append(f"Total missing values: {total_missing}.")
    if total_missing > 0:
        for col, m in miss[miss>0].items():
            out.append(f"- `{col}` has {int(m)} missing values.")
    # numeric summaries
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().T[['mean','50%','std']]
        for col in desc.index:
            out.append(f"- `{col}`: mean={desc.loc[col,'mean']:.2f}, median={desc.loc[col,'50%']:.2f}, std={desc.loc[col,'std']:.2f}.")
        # correlations top pair
        if len(num_cols) > 1:
            corr = df[num_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            i,j = np.unravel_index(corr.values.argmax(), corr.shape)
            a = corr.index[i]; b = corr.columns[j]
            out.append(f"- Strongest numeric correlation: `{a}` vs `{b}` = {df[a].corr(df[b]):.2f}.")
    else:
        out.append("- No numeric columns.")
    # categorical highlights
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in cat_cols[:5]:
        top = df[c].value_counts(normalize=True).head(3)
        top_text = ", ".join([f"{idx} ({v*100:.1f}%)" for idx, v in top.items()])
        out.append(f"- `{c}` top: {top_text}.")
    # outlier check (IQR)
    outliers = []
    for c in num_cols:
        s = df[c].dropna()
        if len(s) < 5: continue
        q1 = s.quantile(0.25); q3 = s.quantile(0.75); iqr = q3 - q1
        if iqr == 0: continue
        upper = q3 + 1.5*iqr; lower = q1 - 1.5*iqr
        n_out = ((s>upper) | (s<lower)).sum()
        if n_out>0:
            outliers.append(f"- `{c}` has {int(n_out)} suspected outliers ({n_out/len(s)*100:.2f}%).")
    if outliers:
        out.extend(outliers)
    else:
        out.append("- No large outliers detected by IQR rule.")
    return "\n".join(out)

def cleaning_suggestions(df: pd.DataFrame):
    sugg = []
    # missing
    miss = df.isnull().sum()
    for col,m in miss.items():
        if m>0:
            if np.issubdtype(df[col].dtype, np.number):
                sugg.append(f"- Impute missing numeric `{col}` with median or mean.")
            else:
                sugg.append(f"- Impute missing categorical `{col}` with mode or 'Unknown'.")
    # timestamp
    if 'timestamp' in df.columns:
        sugg.append("- Convert `timestamp` (unix) to datetime for time-based analysis.")
    # scaling
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols)>0:
        sugg.append("- Consider scaling numeric features if you use distance-based models.")
    return "\n".join(sugg) if sugg else "No cleaning suggestions — dataset looks clean."

# -------------------------
# Helper: DataFrame -> SQLite
# -------------------------
def df_to_sqlite(conn, df, name="data"):
    df.to_sql(name, conn, if_exists="replace", index=False)

# -------------------------
# Upload & base setup
# -------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to start. This app provides NL→SQL, NL→Chart, AutoEDA, Cleaning suggestions, and AutoML.")
    st.stop()

df = pd.read_csv(uploaded)
st.sidebar.markdown(f"**Dataset:** {len(df)} rows × {len(df.columns)} cols")
st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10))

# prepare sqlite
conn = sqlite3.connect(":memory:")
df_to_sqlite(conn, df, "data")
columns = df.columns.tolist()
sample_rows = df.head(5).to_string()

# -------------------------
# Tabs layout (Option A)
# -------------------------
tabs = st.tabs(["📊 Dashboard", "💬 Chat with Data (RAG)", "🔎 Auto EDA", "🧹 Cleaning Assistant", "🧠 AutoML", "🧾 NL→SQL", "📈 NL→Chart"])

# -------------------------
# Tab: Dashboard
# -------------------------
with tabs[0]:
    st.header("📊 Dashboard (Quick Insights)")
    st.markdown("Quick rule-based facts + a one-button GenAI polish.")
    facts = basic_facts(df)
    st.markdown("### Rule-based facts")
    st.code(facts)

    if st.button("Polish facts with GenAI"):
        if llm:
            prompt = paraphrase_prompt(facts)
            res = llm(prompt, max_length=200)
            st.markdown("### 🤖 GenAI Polished Summary")
            st.write(res[0]["generated_text"])
        else:
            st.info("LLM not available; showing raw facts.")

    st.markdown("---")
    st.markdown("### Quick Visuals")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        if num_cols:
            col_n = st.selectbox("Histogram column (quick):", num_cols, key="dash_hist")
            fig, ax = plt.subplots()
            ax.hist(df[col_n].dropna(), bins=25)
            ax.set_title(f"Histogram: {col_n}")
            st.pyplot(fig)
    with col2:
        if cat_cols:
            col_c = st.selectbox("Pie column (quick):", cat_cols, key="dash_pie")
            counts = df[col_c].value_counts().head(8)
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
            ax.set_title(f"Pie: {col_c}")
            st.pyplot(fig)

# -------------------------
# Tab: Chat with Data (RAG)
# -------------------------
with tabs[1]:
    st.header("💬 Chat with Data (RAG-style)")
    st.markdown("Ask questions in natural language; the system builds rule-based facts and the LLM rewrites them into polished insights.")
    q = st.text_input("Ask a question about the dataset (e.g., 'Summarize rating distribution and top movies')", key="chat_q")
    if st.button("Get Insight (RAG)"):
        facts = []
        facts.append(f"Rows: {len(df)}; Columns: {len(df.columns)}.")
        # add quick stats for numeric cols
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            s = df[c].dropna()
            facts.append(f"`{c}` mean={s.mean():.2f}, median={s.median():.2f}, std={s.std():.2f}.")
        # cat top
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for c in cat_cols[:3]:
            top = df[c].value_counts().head(3)
            facts.append(f"`{c}` top: " + ", ".join([f"{idx}({cnt})" for idx,cnt in top.items()]))
        # simple pair
        if len(num_cols)>1:
            corr = df[num_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            i,j = np.unravel_index(corr.values.argmax(), corr.shape)
            a = corr.index[i]; b = corr.columns[j]
            facts.append(f"Strongest correlation: `{a}` vs `{b}` = {df[a].corr(df[b]):.2f}.")
        rule_summary = "\n".join(facts)
        st.markdown("### Rule-based facts")
        st.code(rule_summary)

        if llm:
            prompt = paraphrase_prompt(f"User question: {q}\n\nFacts:\n{rule_summary}\n\nAnswer the user's question concisely using only the facts above.")
            res = llm(prompt, max_length=250)
            st.markdown("### 🤖 GenAI Answer")
            st.write(res[0]["generated_text"])
        else:
            st.info("LLM not available; showing facts.")

# -------------------------
# Tab: Auto EDA
# -------------------------
with tabs[2]:
    st.header("🔎 Auto EDA (one-click)")
    st.markdown("Run a fast exploratory analysis and get a polished GenAI report.")
    if st.button("Run Auto EDA"):
        eda = auto_eda(df)
        st.markdown("### Rule-based EDA results")
        st.code(eda)

        if llm:
            prompt = paraphrase_prompt(eda)
            res = llm(prompt, max_length=400)
            st.markdown("### 🤖 GenAI EDA Report")
            st.write(res[0]["generated_text"])
        else:
            st.info("LLM not available; showing rule-based EDA.")

    st.markdown("You can also download a short CSV of the top correlations if desired.")
    if st.button("Show top correlations (if numeric columns)"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
            top = corr.head(20).reset_index()
            top.columns = ["feature_a","feature_b","abs_corr"]
            st.dataframe(top)
        else:
            st.info("Not enough numeric columns for correlations.")

# -------------------------
# Tab: Cleaning Assistant
# -------------------------
with tabs[3]:
    st.header("🧹 Data Cleaning Assistant")
    st.markdown("Get intelligent cleaning suggestions and optionally apply common fixes.")
    suggestions = cleaning_suggestions(df)
    st.markdown("### Suggestions (rule-based)")
    st.code(suggestions)

    if llm:
        if st.button("Polish cleaning suggestions with GenAI"):
            prompt = paraphrase_prompt(suggestions)
            res = llm(prompt, max_length=200)
            st.markdown("### 🤖 GenAI Suggestions")
            st.write(res[0]["generated_text"])

    st.markdown("### Apply common fixes (preview first)")
    cols_with_missing = [c for c in df.columns if df[c].isnull().sum()>0]
    if cols_with_missing:
        selected_cols = st.multiselect("Select columns to impute (numeric -> median / categorical -> mode):", cols_with_missing)
        if st.button("Apply imputation"):
            df_copy = df.copy()
            for c in selected_cols:
                if np.issubdtype(df_copy[c].dtype, np.number):
                    med = df_copy[c].median()
                    df_copy[c].fillna(med, inplace=True)
                else:
                    mode = df_copy[c].mode().iloc[0] if not df_copy[c].mode().empty else "Unknown"
                    df_copy[c].fillna(mode, inplace=True)
            st.success("Imputation applied in preview. (Not saved to original unless you download.)")
            st.dataframe(df_copy.head())
            if st.button("Save imputed as dataset (overwrite in memory)"):
                df = df_copy
                df_to_sqlite(conn, df, "data")
                st.success("Dataset overwritten in session.")
    else:
        st.info("No missing values detected.")

# -------------------------
# Tab: AutoML
# -------------------------
with tabs[4]:
    st.header("🧠 AutoML Model Builder (safe & simple)")
    st.markdown("Choose a target column to predict. AutoML will build a simple pipeline (impute, scale, RandomForest) and show metrics.")
    target = st.selectbox("Select target column:", options=[None] + list(df.columns))
    if target:
        if st.button("Suggest features automatically"):
            candidate_feats = [c for c in df.columns if c != target and np.issubdtype(df[c].dtype, np.number)]
            if not candidate_feats:
                st.info("No numeric feature candidates found. You can manually pick categorical encodings, but AutoML expects numeric features.")
            else:
                st.info(f"Suggested numeric features: {candidate_feats[:8]}")
        st.markdown("Select features to use (numeric columns recommended):")
        feat_choices = st.multiselect("Features:", options=[c for c in df.columns if c != target], default=[c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c!=target][:5])

        task_type = st.radio("Task type:", ["Regression", "Classification"])
        test_size = st.slider("Test set proportion", 0.1, 0.4, 0.2)
        if st.button("Train AutoML model"):
            if not feat_choices:
                st.error("Pick at least one feature.")
            else:
                X = df[feat_choices].copy()
                y = df[target].copy()
                # Basic preprocessing: impute numeric; drop rows with non-numeric in selected features
                # convert non-numeric features to numeric where possible
                X_num = X.select_dtypes(include=[np.number])
                non_numeric = [c for c in feat_choices if c not in X_num.columns]
                if non_numeric:
                    st.warning(f"Dropping non-numeric features for AutoML: {non_numeric}")
                    X = X_num
                else:
                    X = X_num

                # drop rows with missing target
                mask = y.notnull()
                X = X[mask]; y = y[mask]

                # impute
                imputer = SimpleImputer(strategy="median")
                X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_imp.columns)

                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

                if task_type == "Regression":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mse = mean_squared_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    st.success(f"Regression results — MSE: {mse:.4f}, R2: {r2:.4f}")
                    # feature importances
                    fi = pd.Series(model.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
                    st.markdown("### Feature importances")
                    st.dataframe(fi)
                else:
                    # classification: drop rows where target not categorical -> convert
                    # try to map target to integers if not already
                    y_enc = y.astype('category').cat.codes
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train if y_train.dtype.kind in 'iu' else y_enc.loc[y_train.index])
                    preds = model.predict(X_test)
                    # try computing accuracy
                    try:
                        acc = accuracy_score(y_test, preds)
                        st.success(f"Classification accuracy: {acc:.4f}")
                    except Exception:
                        st.success("Model trained. Could not compute accuracy due to label types.")
                    fi = pd.Series(model.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
                    st.markdown("### Feature importances")
                    st.dataframe(fi)

# -------------------------
# Tab: NL -> SQL
# -------------------------
with tabs[5]:
    st.header("🧾 Natural Language → SQL (safe)")
    nl_sql = st.text_input("Ask a data question (e.g., 'Top 10 movies by average rating')", key="nl_sql_tab")
    if st.button("Generate + Run SQL"):
        if not nl_sql.strip():
            st.warning("Type a question first.")
        else:
            if llm:
                prompt = make_sql_prompt(nl_sql, table_name="data", columns=columns, sample=sample_rows)
                gen = llm(prompt, max_length=200, do_sample=False)
                raw_sql = gen[0]["generated_text"].strip()
                # extract first select...
                m = re.search(r"(select[\s\S]*)", raw_sql, flags=re.IGNORECASE)
                sql_candidate = m.group(1).strip().rstrip(".") if m else raw_sql
                st.markdown("### Generated SQL")
                st.code(sql_candidate, language="sql")
                if is_sql_safe(sql_candidate):
                    try:
                        res = pd.read_sql_query(sql_candidate, conn)
                        st.success("Executed.")
                        st.dataframe(res.head(200))
                    except Exception as e:
                        st.error(f"SQL Execution error: {e}")
                else:
                    st.error("SQL failed safety checks — not executed.")
            else:
                st.error("LLM not available.")

    st.markdown("**Safe rule-based fallbacks**")
    if st.button("Count rows"):
        st.write(f"Row count: {len(df)}")
    if st.button("Show columns"):
        st.write(columns)

# -------------------------
# Tab: NL -> Chart
# -------------------------
with tabs[6]:
    st.header("📈 Natural Language → Chart")
    nl_chart = st.text_input("Describe the chart you want (e.g., 'scatter age salary color by gender')", key="nl_chart_tab")
    if st.button("Generate Chart Plan"):
        if not nl_chart.strip():
            st.warning("Please write a chart instruction.")
        else:
            if llm:
                prompt = make_chart_prompt(nl_chart, columns=columns, sample=sample_rows)
                gen = llm(prompt, max_length=200, do_sample=False)
                raw = gen[0]["generated_text"].strip()
                st.markdown("### LLM raw output")
                st.code(raw)
                plan = parse_chart_plan(raw)
                if plan is None:
                    st.error("Could not parse plan. Try simpler phrasing.")
                else:
                    # Validate plan
                    chart_type = plan.get("chart_type")
                    x = plan.get("x"); y = plan.get("y"); agg = plan.get("agg"); color = plan.get("color")
                    if chart_type not in ['scatter','line','bar','hist','box','pie','heatmap']:
                        st.error("Unsupported chart_type.")
                    else:
                        missing_cols = [c for c in [x,y,color] if c and c not in columns]
                        if missing_cols:
                            st.error(f"Columns not found: {missing_cols}")
                        else:
                            st.success("Plan validated. Rendering chart...")
                            fig, ax = plt.subplots(figsize=(8,4))
                            try:
                                if chart_type == "scatter":
                                    ax.scatter(df[x], df[y], c=df[color] if color else None, alpha=0.7)
                                    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
                                elif chart_type == "line":
                                    ax.plot(df[x], df[y])
                                    ax.set_xlabel(x); ax.set_ylabel(y)
                                elif chart_type == "bar":
                                    if agg in [None,"count"]:
                                        res = df.groupby(x).size().reset_index(name="count")
                                        ax.bar(res[x].astype(str).head(20), res["count"].head(20))
                                        plt.xticks(rotation=45)
                                    elif agg=="mean":
                                        res = df.groupby(x)[y].mean().reset_index()
                                        ax.bar(res[x].astype(str).head(20), res[y].head(20)); plt.xticks(rotation=45)
                                elif chart_type == "hist":
                                    ax.hist(df[x].dropna(), bins=25)
                                elif chart_type == "box":
                                    sns.boxplot(x=x, y=y, data=df, ax=ax)
                                elif chart_type == "pie":
                                    counts = df[x].value_counts().head(20)
                                    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
                                elif chart_type == "heatmap":
                                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                    if len(numeric_cols) < 2:
                                        st.error("Not enough numeric columns for heatmap.")
                                    else:
                                        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", ax=ax)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Chart render error: {e}")
            else:
                st.error("LLM not available — cannot generate plan.")

    st.markdown("**Tips:** use short phrasing like 'scatter age salary', 'histogram of rating', 'bar count by movieId'.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("**Notes & Next steps**")
st.markdown(
    "- The app uses a lightweight LLM (google/flan-t5-small). If you want higher-quality NL→SQL / NL→Chart generation, consider flan-t5-base (more RAM) or an API-based LLM (OpenAI/GPT) for production.\n"
    "- All LLM outputs are sanitised and validated before execution. The LLM is used primarily to generate SQL or chart plans and to rewrite facts into polished text."
)
