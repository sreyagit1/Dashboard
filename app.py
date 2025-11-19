import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="GenAI Explainable Dashboard", layout="wide")
st.title("🤖📊 GenAI Explainable Dashboard")
st.caption("Hybrid System: Rule-based Analysis + FLAN-T5 Paraphrasing (Safe & Smart GenAI)")

# --------------------------------------------------
# Load FLAN-T5 Small (Paraphrasing Mode)
# --------------------------------------------------
@st.cache_resource
def load_llm():
    try:
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return generator
    except Exception as e:
        st.error(f"Error loading FLAN: {e}")
        return None

llm = load_llm()

# --------------------------------------------------
# RULE-BASED LOGIC (SAFE, ACCURATE)
# --------------------------------------------------

def explain_histogram(col, s):
    s = s.dropna()
    mean = s.mean()
    median = s.median()
    skew = s.skew()
    std = s.std()

    text = (
        f"The histogram for column '{col}' shows the distribution of values. "
        f"The mean is {mean:.2f}, while the median is {median:.2f}. "
        f"The skewness is {skew:.2f}, which indicates whether the distribution "
        f"is symmetric or tilted to one side. The standard deviation is {std:.2f}, "
        f"showing how spread out the values are."
    )
    return text


def explain_pie(col, s):
    vc = s.value_counts()
    top = vc.index[0]
    pct = (vc.iloc[0] / len(s)) * 100

    text = (
        f"The pie chart for '{col}' shows category distribution. "
        f"The most common value is '{top}' appearing in {pct:.1f}% of rows. "
        f"Other categories have lower proportions."
    )
    return text


def explain_scatter(x, y, df):
    corr = df[x].corr(df[y])

    text = (
        f"The scatter plot between '{x}' and '{y}' shows how the two variables relate. "
        f"The correlation value is {corr:.2f}. "
        f"A positive correlation means both values increase together, "
        f"while a negative value means one increases as the other decreases."
    )
    return text


def explain_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    strongest = corr.unstack().sort_values(ascending=False).drop_duplicates().iloc[1]
    text = (
        "The heatmap shows correlations between numeric features. "
        f"The strongest relationship has a correlation value of {strongest:.2f}. "
        f"High positive values mean two variables move together, while negative values "
        f"mean they move in opposite directions."
    )
    return text


def rewrite_with_genai(text):
    """ FLAN only rewrites existing explanation — not analyzing numbers. """
    if llm is None:
        return text

    prompt = (
        "Rewrite this explanation in simple, clear English without adding new details. "
        "Do NOT generate numbers or complex interpretations.\n\n"
        f"{text}"
    )

    result = llm(prompt, max_length=150)
    return result[0]["generated_text"]


# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --------------------------------------------------
    # PLOTS + EXPLANATION
    # --------------------------------------------------
    st.subheader("📊 Visual Analysis + GenAI Explanation")

    plot = st.selectbox(
        "Choose a plot:",
        ["Histogram", "Pie Chart", "Scatter Plot", "Correlation Heatmap"]
    )

    # ---------------- HISTOGRAM ----------------
    if plot == "Histogram" and numeric_cols:

        col = st.selectbox("Select numeric column:", numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

        rule_text = explain_histogram(col, df[col])
        st.markdown("### Rule-Based Explanation")
        st.write(rule_text)

        if st.button("Rewrite with GenAI"):
            gen_text = rewrite_with_genai(rule_text)
            st.markdown("### 🤖 GenAI Explanation")
            st.write(gen_text)

    # ---------------- PIE CHART ----------------
    elif plot == "Pie Chart" and categorical_cols:

        col = st.selectbox("Select categorical column:", categorical_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot.pie(autopct="%1.1f%%")
        ax.set_title(f"Pie Chart of {col}")
        ax.set_ylabel("")
        st.pyplot(fig)

        rule_text = explain_pie(col, df[col])
        st.markdown("### Rule-Based Explanation")
        st.write(rule_text)

        if st.button("Rewrite with GenAI"):
            st.markdown("### 🤖 GenAI Explanation")
            st.write(rewrite_with_genai(rule_text))

    # ---------------- SCATTER PLOT ----------------
    elif plot == "Scatter Plot" and len(numeric_cols) >= 2:

        x = st.selectbox("X-axis:", numeric_cols)
        y = st.selectbox("Y-axis:", numeric_cols)
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y])
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

        rule_text = explain_scatter(x, y, df)
        st.markdown("### Rule-Based Explanation")
        st.write(rule_text)

        if st.button("Rewrite with GenAI"):
            st.markdown("### 🤖 GenAI Explanation")
            st.write(rewrite_with_genai(rule_text))

    # ---------------- HEATMAP ----------------
    elif plot == "Correlation Heatmap" and len(numeric_cols) > 1:

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        rule_text = explain_heatmap(df, numeric_cols)
        st.markdown("### Rule-Based Explanation")
        st.write(rule_text)

        if st.button("Rewrite with GenAI"):
            st.markdown("### 🤖 GenAI Explanation")
            st.write(rewrite_with_genai(rule_text))

else:
    st.info("Upload a CSV file to begin.")
