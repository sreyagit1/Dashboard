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
# File Upload
# -----------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file is None:
    st.warning("⬆️ Please upload a CSV file to continue.")
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

# Histogram
if plot_type == "Histogram (Numeric)":
    if numeric_cols:
        col = st.selectbox("Numeric column:", numeric_cols)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[col].dropna(), bins=20)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found.")

# Pie Chart
elif plot_type == "Pie Chart (Categorical)":
    if categorical_cols:
        col = st.selectbox("Categorical column:", categorical_cols)
        fig, ax = plt.subplots(figsize=(6, 5))
        df[col].value_counts().head(6).plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.warning("No categorical columns found.")

# Scatter Plot
elif plot_type == "Scatter Plot":
    if len(numeric_cols) >= 2:
        x = st.selectbox("X-axis:", numeric_cols)
        y = st.selectbox("Y-axis:", numeric_cols)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df[x], df[y])
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)
    else:
        st.warning("Need at least two numeric columns.")

# Box Plot
elif plot_type == "Box Plot":
    if numeric_cols and categorical_cols:
        num = st.selectbox("Numeric column:", numeric_cols)
        cat = st.selectbox("Category column:", categorical_cols)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=cat, y=num, data=df, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Need both numeric and categorical columns.")

# Heatmap
elif plot_type == "Correlation Heatmap":
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns.")

# -----------------------------------------------------
# Generative Insights (Groq Cloud - Simple English)
# -----------------------------------------------------
st.subheader("🧠 AI-Generated Insights (Simple English)")

with st.spinner("Generating friendly insights..."):

    sample = df.head(3).to_string()

    prompt = f"""
You are a helpful data assistant. Explain this dataset in very simple English so anyone can understand.

Rules:
- Do NOT copy exact numbers.
- Do NOT repeat the sample rows.
- No technical jargon.
- Short, clear sentences.
- Focus on general patterns, not stats.
- If unsure, skip it.

Format:
1. One-sentence simple summary.
2. 3–6 bullet points with easy insights.
3. One short caution about the data.

Here is a small sample of the dataset:
{sample}
"""

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You explain things in simple English."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=250,
    temperature=0.4


    )

    output = response.choices[0].message["content"]

st.markdown("### 🤖 Insights")
st.write(output)
