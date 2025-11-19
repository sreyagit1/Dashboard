import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="GenAI Explainability Dashboard", layout="wide")
st.title("🤖📊 GenAI Explainability Dashboard")
st.caption("Powered by google/flan-t5-small — Real Generative AI + Smart Rule-Based Insights")

# -------------------------------
# Load FLAN-T5 Small (Generative AI)
# -------------------------------
@st.cache_resource
def load_llm():
    try:
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return gen
    except Exception as e:
        st.error(f"Failed to load GenAI model: {e}")
        return None

llm = load_llm()

# -------------------------------
# Rule-Based Dataset Insights
# -------------------------------
def generate_basic_statistics(df):
    lines = []
    lines.append(f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    missing = df.isnull().sum().sum()
    if missing > 0:
        lines.append(f"There are **{missing} missing values**.")
    else:
        lines.append("There are **no missing values**.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe()
        for col in numeric_cols:
            mean = desc.loc["mean", col]
            std = desc.loc["std", col]
            lines.append(f"- `{col}` → mean: {mean:.2f}, std: {std:.2f}")
    else:
        lines.append("No numeric columns found.")

    return "\n".join(lines)


# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # -------------------------------
    # Basic Rule-Based Insights
    # -------------------------------
    st.subheader("⚙️ Rule-Based Insights")
    with st.expander("Show Insights"):
        stats_text = generate_basic_statistics(df)
        st.markdown(stats_text)

        if st.button("Explain Dataset Using GenAI"):
            if llm:
                prompt = (
                    f"Explain this dataset in simple terms:\n\n"
                    f"{stats_text}\n\n"
                    f"Sample rows:\n{df.head(3).to_string()}"
                )
                result = llm(prompt, max_length=200)
                st.markdown("### 🤖 GenAI Summary")
                st.write(result[0]["generated_text"])

    # -------------------------------
    # Visualizations + GenAI Explainability
    # -------------------------------
    st.subheader("📊 Visual Analysis + GenAI Explainability")

    plot_type = st.selectbox(
        "Choose a plot type:",
        ["Histogram", "Pie Chart", "Scatter Plot", "Correlation Heatmap"]
    )

    # ---------------- HISTOGRAM ----------------
    if plot_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Select numeric column:", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=20)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

            if st.button("Explain Histogram (GenAI)"):
                summary = (
                    f"The histogram shows the distribution of column {col}. "
                    f"Mean = {df[col].mean():.2f}, Median = {df[col].median():.2f}, "
                    f"Std = {df[col].std():.2f}."
                )
                prompt = (
                    f"Explain this histogram to a beginner:\n{summary}\n"
                    f"Describe shape, skewness, and what it means."
                )
                result = llm(prompt, max_length=180)
                st.markdown("### 🤖 GenAI Explanation")
                st.write(result[0]["generated_text"])
        else:
            st.warning("No numeric columns found.")

    # ---------------- PIE CHART ----------------
    elif plot_type == "Pie Chart":
        if categorical_cols:
            col = st.selectbox("Select categorical column:", categorical_cols)
            fig, ax = plt.subplots()
            df[col].value_counts().plot.pie(autopct="%1.1f%%")
            ax.set_title(f"Pie Chart of {col}")
            ax.set_ylabel("")
            st.pyplot(fig)

            if st.button("Explain Pie Chart (GenAI)"):
                value_counts = df[col].value_counts()
                summary = f"Top categories: {value_counts.head(3).to_dict()}"
                prompt = f"Explain this pie chart in simple English:\n{summary}"
                result = llm(prompt, max_length=150)
                st.markdown("### 🤖 GenAI Explanation")
                st.write(result[0]["generated_text"])
        else:
            st.warning("No categorical columns found.")

    # ---------------- SCATTER PLOT ----------------
    elif plot_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x = st.selectbox("X-axis:", numeric_cols)
            y = st.selectbox("Y-axis:", numeric_cols)
            fig, ax = plt.subplots()
            ax.scatter(df[x], df[y])
            ax.set_title(f"{x} vs {y}")
            st.pyplot(fig)

            if st.button("Explain Scatter Plot (GenAI)"):
                corr = df[x].corr(df[y])
                summary = f"Scatter plot of {x} vs {y}. Correlation = {corr:.2f}."
                prompt = (
                    f"Explain this scatter plot in simple terms: {summary}. "
                    f"What does the relationship mean?"
                )
                result = llm(prompt, max_length=180)
                st.markdown("### 🤖 GenAI Explanation")
                st.write(result[0]["generated_text"])
        else:
            st.warning("Need at least 2 numeric columns.")

    # ---------------- HEATMAP ----------------
    elif plot_type == "Correlation Heatmap":
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots()
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            if st.button("Explain Heatmap (GenAI)"):
                top_pair = corr.abs().unstack().sort_values(ascending=False)[1]
                summary = f"Most correlated pair has correlation {top_pair:.2f}."
                prompt = (
                    f"Explain what this correlation heatmap means. "
                    f"Explain strong, weak, and negative correlations simply. "
                    f"Also explain this: {summary}"
                )
                result = llm(prompt, max_length=200)
                st.markdown("### 🤖 GenAI Explanation")
                st.write(result[0]["generated_text"])
        else:
            st.warning("Not enough numeric columns for heatmap.")

    # -------------------------------
    # GenAI Report Generator
    # -------------------------------
    st.subheader("📘 GenAI Report Generator")
    if st.button("Generate Full Report (GenAI)"):
        stats_text = df.describe(include='all').to_string()
        sample = df.head(5).to_string()

        prompt = (
            "Create a clear, professional report from this dataset.\n"
            "Explain:\n- key patterns\n- correlations\n- trends\n- risks\n- recommendations\n\n"
            f"Statistics:\n{stats_text}\n\nSample:\n{sample}"
        )

        result = llm(prompt, max_length=350)
        st.markdown("### 📘 AI-Generated Report")
        st.write(result[0]["generated_text"])

else:
    st.warning("Upload a CSV file to begin!")
