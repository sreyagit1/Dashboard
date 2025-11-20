# AI Dashboard Assistant (Groq Cloud Version) with Plot Explanations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# -----------------------------------------------------
# Streamlit UI Setup
# -----------------------------------------------------
st.set_page_config(page_title="AI Dashboard Assistant", layout="wide")
st.title("📊 AI Dashboard Assistant (Groq Cloud Version)")
st.info("💡 Upload a dataset and get simple, clear, AI-generated insights instantly.")

# -----------------------------------------------------
# Groq Client Setup
# -----------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY", None)

if not api_key:
    st.error("❌ Please add your Groq API key in Streamlit Secrets as `GROQ_API_KEY`.")
    st.stop()

client = Groq(api_key=api_key)

# -----------------------------------------------------
# Helper Functions for Plot Explanation
# -----------------------------------------------------
def _count_outliers_iqr(series):
    series = series.dropna()
    if len(series) < 4:
        return 0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low_cut = q1 - 1.5 * iqr
    high_cut = q3 + 1.5 * iqr
    return int(((series < low_cut) | (series > high_cut)).sum())


def generate_plot_explanation(client, model, plot_type, df, x_col=None, y_col=None, cat_col=None, numeric_col=None):
    n_points = len(df)
    context = f"Plot type: {plot_type}. Total data points: {n_points}.\n"

    if plot_type == "Histogram":
        series = df[numeric_col].dropna()
        outliers = _count_outliers_iqr(series)
        skewness = series.skew()
        skew_desc = "fairly symmetric"
        if skewness > 0.7:
            skew_desc = "skewed right (long tail to the right)"
        elif skewness < -0.7:
            skew_desc = "skewed left (long tail to the left)"
        elif abs(skewness) > 0.3:
            skew_desc = "slightly skewed"
        context += f"Column: {numeric_col}. Distribution appears {skew_desc}. Estimated outliers: {outliers}.\n"

    elif plot_type == "Pie":
        counts = df[cat_col].value_counts(dropna=True)
        top_ratio = counts.iloc[0] / counts.sum() if len(counts) > 0 else 0
        if top_ratio > 0.7:
            dominance = "one category dominates"
        elif top_ratio > 0.4:
            dominance = "a few categories dominate"
        else:
            dominance = "fairly balanced categories"
        context += f"Column: {cat_col}. {dominance}.\n"

    elif plot_type == "Scatter":
        pair = df[[x_col, y_col]].dropna()
        corr_desc = "no clear trend"
        if len(pair) >= 3:
            corr = pair[x_col].corr(pair[y_col])
            if not pd.isna(corr):
                if abs(corr) > 0.6:
                    corr_desc = "strong trend"
                elif abs(corr) > 0.3:
                    corr_desc = "moderate trend"
                else:
                    corr_desc = "weak or no trend"
        context += f"X: {x_col}, Y: {y_col}. Trend: {corr_desc}. Possible few outliers.\n"

    elif plot_type == "Box":
        groups = df.groupby(cat_col)[numeric_col]
        group_outliers = 0
        for _, g in groups:
            group_outliers += _count_outliers_iqr(g)
        context += f"Numeric: {numeric_col} by category {cat_col}. Estimated outliers: {group_outliers}.\n"

    elif plot_type == "Heatmap":
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        context += f"Correlations between numeric columns: {', '.join(nums[:6])}.\n"

    prompt = f"""
You are a friendly assistant who explains charts in very simple English.
Task: Based on the context below, produce 3 to 6 short bullet points.
Rules:
- Very short sentences.
- No technical words.
- No exact numbers.
- Mention things like outliers, trends, imbalance, unusual shapes.

Context:
{context}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You explain charts in simple English."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=180,
        temperature=0.35
    )

    explanation = response.choices[0].message.content
    return explanation.splitlines()[:6]

# -----------------------------------------------------
# File Upload
# -----------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file is None:
    st.warning("⬆️ Please upload a CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("✅ File uploaded successfully!")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------------------------
# Visualizations
# -----------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

st.subheader("📊 Visual Analysis")
plot_type = st.selectbox(
    "Choose a plot:",
    [
        "Histogram (Numeric)",
        "Pie Chart (Categorical)",
        "Scatter Plot",
        "Box Plot",
        "Correlation Heatmap"
    ]
)

# -----------------------------------------------------
# Plot Logic + Explanation
# -----------------------------------------------------
if plot_type == "Histogram (Numeric)":
    if numeric_cols:
        col = st.selectbox("Numeric column:", numeric_cols)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

        with st.expander("Explain this plot"):
            explanation = generate_plot_explanation(client, "llama-3.1-8b-instant", "Histogram", df, numeric_col=col)
            for line in explanation:
                if line.strip():
                    st.markdown(f"- {line.strip()}")
    else:
        st.warning("No numeric columns found.")

elif plot_type == "Pie Chart (Categorical)":
    if categorical_cols:
        col = st.selectbox("Categorical column:", categorical_cols)
        fig, ax = plt.subplots(figsize=(6, 5))
        df[col].value_counts().head(6).plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

        with st.expander("Explain this plot"):
            explanation = generate_plot_explanation(client, "llama-3.1-8b-instant", "Pie", df, cat_col=col)
            for line in explanation:
                st.markdown(f"- {line.strip()}")
    else:
        st.warning("No categorical columns found.")

elif plot_type == "Scatter Plot":
    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis:", numeric_cols)
        y = st.selectbox("Y-axis:", numeric_cols)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df[x], df[y])
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

        with st.expander("Explain this plot"):
            explanation = generate_plot_explanation(client, "llama-3.1-8b-instant", "Scatter", df, x_col=x, y_col=y)
            for line in explanation:
                st.markdown(f"- {line.strip()}")
    else:
        st.warning("Need at least two numeric columns.")

elif plot_type == "Box Plot":
    if numeric_cols and categorical_cols:
        num = st.selectbox("Numeric column:", numeric_cols)
        cat = st.selectbox("Category column:", categorical_cols)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=cat, y=num, data=df, ax=ax)
        st.pyplot(fig)

        with st.expander("Explain this plot"):
            explanation = generate_plot_explanation(client, "llama-3.1-8b-instant", "Box", df, numeric_col=num, cat_col=cat)
            for line in explanation:
                st.markdown(f"- {line.strip()}")
    else:
        st.warning("Need both numeric and categorical columns.")

elif plot_type == "Correlation Heatmap":
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        with st.expander("Explain this plot"):
            explanation = generate_plot_explanation(
                client,
                "llama-3.1-8b-instant",
                "Heatmap",
                df
            )
            for line in explanation:
                st.markdown(f"- {line.strip()}")
    else:
        st.warning("Not enough numeric columns to generate a heatmap.")


def summarize_dataframe_for_ai(df):
    summary_lines = []

    # Basic shape
    summary_lines.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Column overview
    summary_lines.append("\nColumn Overview:")
    for col in df.columns:
        summary_lines.append(f"- {col} ({df[col].dtype})")

    # Missing values
    missing = df.isnull().sum()
    summary_lines.append("\nMissing Values:")
    for col in df.columns:
        summary_lines.append(f"- {col}: {missing[col]} missing")

    # Numeric summaries
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        summary_lines.append("\nNumeric Column Summary:")
        desc = df[num_cols].describe().to_dict()
        for col in num_cols:
            info = desc[col]
            summary_lines.append(
                f"- {col}: min={info['min']}, max={info['max']}, mean={info['mean']:.2f}, std={info['std']:.2f}"
            )

        # Correlation matrix (rounded)
        corr = df[num_cols].corr().round(3)
        summary_lines.append("\nCorrelations:")
        for c1 in num_cols:
            for c2 in num_cols:
                if c1 != c2:
                    summary_lines.append(f"- {c1} vs {c2}: {corr.loc[c1, c2]}")

    # Categorical summaries
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if cat_cols:
        summary_lines.append("\nCategorical Columns Summary:")
        for col in cat_cols:
            counts = df[col].value_counts().head(5)
            summary_lines.append(f"- {col}: top values → {dict(counts)}")

    # Sample rows
    summary_lines.append("\nSample rows:")
    summary_lines.append(df.head(5).to_string())

    return "\n".join(summary_lines)

# -----------------------------------------------------
# Chat With This Dataset (AI Chatbot)
# -----------------------------------------------------
st.subheader("💬 Chat With This Dataset")

user_question = st.text_input("Ask a question about your dataset:")

if user_question:
    with st.spinner("Thinking..."):
        
        dataset_summary = summarize_dataframe_for_ai(df)

        prompt = f"""
You are a helpful data analyst. 
Answer questions about this dataset using simple English, but with deeper reasoning.

Rules:
- Use the dataset summary below.
- Think step-by-step.
- Use trends, correlations, category patterns, and outliers.
- If user asks for analysis, give it.
- If user asks for suggestions (e.g., what plot to use), give them.
- If something is unclear, politely say so.
- Keep answers short but meaningful, not just basic facts.

Here is the dataset summary:
{dataset_summary}

User question:
{user_question}

Now give a clear, helpful answer.
"""


        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You explain data in simple English."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )

        ai_answer = response.choices[0].message.content

        st.markdown("### 🤖 Answer:")
        st.write(ai_answer)

