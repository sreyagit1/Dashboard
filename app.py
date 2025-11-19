import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch

# ------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------
st.set_page_config(page_title="GenAI Data Dashboard", layout="wide")
st.title("🤖📊 GenAI-Powered Data Dashboard")
st.info("Upload a dataset → Explore charts → Chat with your data → Generate AI reports")

# ------------------------------------------------
# Load Local LLM (DistilGPT-2)
# ------------------------------------------------
@st.cache_resource
def load_local_llm():
    model_name = "distilgpt2"
    try:
        st.info(f"Loading model **{model_name}**...")
        generator = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto"
        )
        st.success("LLM loaded successfully!")
        return generator
    except Exception as e:
        st.error(f"LLM could not load: {e}")
        return None

local_generator = load_local_llm()

# ------------------------------------------------
# Utility: Rule-Based Insights
# ------------------------------------------------
def generate_rule_based_insights(df, max_insights=10):
    insights = []

    if df.empty:
        return ["No data available for analysis."]

    insights.append(f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"There are **{missing} missing values** in the dataset.")
    else:
        insights.append("There are no missing values.")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        mean_vals = df[num_cols].mean().round(2)
        for col in mean_vals.head(max_insights).index:
            insights.append(f"The average value of **{col}** is **{mean_vals[col]}**.")
    else:
        insights.append("No numeric columns found.")

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        for col in cat_cols[:max_insights]:
            top_value = df[col].mode()[0]
            insights.append(f"Most common value in **{col}** is **{top_value}**.")
    else:
        insights.append("No categorical columns found.")

    return insights[:max_insights]

# ------------------------------------------------
# File Upload
# ------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ------------------------------------------------
    # Visual Analysis
    # ------------------------------------------------
    st.subheader("📊 Visual Analysis")
    plot_type = st.selectbox(
        "Select plot type:",
        ["Histogram", "Pie Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"]
    )

    # Histogram
    if plot_type == "Histogram":
        if numeric_cols:
            col = st.selectbox("Select numeric column:", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=20)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Pie Chart
    elif plot_type == "Pie Chart":
        if categorical_cols:
            col = st.selectbox("Select categorical column:", categorical_cols)
            fig, ax = plt.subplots()
            df[col].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

    # Scatter Plot
    elif plot_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x = st.selectbox("X-axis:", numeric_cols)
            y = st.selectbox("Y-axis:", numeric_cols)
            fig, ax = plt.subplots()
            ax.scatter(df[x], df[y])
            ax.set_title(f"{x} vs {y}")
            st.pyplot(fig)

    # Box Plot
    elif plot_type == "Box Plot":
        if numeric_cols and categorical_cols:
            num_col = st.selectbox("Numeric column:", numeric_cols)
            cat_col = st.selectbox("Categorical column:", categorical_cols)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
            st.pyplot(fig)

    # Correlation Heatmap
    elif plot_type == "Correlation Heatmap":
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots()
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    # ------------------------------------------------
    # AI INSIGHTS SECTION
    # ------------------------------------------------
    st.subheader("🧠 AI Insights")

    mode = st.radio("Choose AI Mode:", ["Rule-Based", "Generative"])

    if mode == "Rule-Based":
        insights = generate_rule_based_insights(df)
        st.markdown("### ⚙️ Rule-Based Insights")
        for i, ins in enumerate(insights, 1):
            st.markdown(f"**{i}.** {ins}")

    else:
        if local_generator is None:
            st.error("Generative model unavailable.")
        else:
            base_insights = generate_rule_based_insights(df)
            prompt = (
                "Analyze this dataset:\n"
                + "\n".join(base_insights)
                + "\n\nSample rows:\n"
                + df.head().to_string()
            )[:800]

            result = local_generator(prompt, max_new_tokens=150)
            st.markdown("### 🤖 Generated Insights")
            st.write(result[0]["generated_text"])

    # ------------------------------------------------
    # 🔥 CHAT WITH YOUR DATASET
    # ------------------------------------------------
    st.subheader("💬 Chat With Your Dataset (GenAI)")

    user_query = st.text_input("Ask anything about your data:")

    if user_query:
        if local_generator is None:
            st.error("LLM not available.")
        else:
            sample = df.head(5).to_string()
            prompt = f"""
            You are an AI data analyst.

            Dataset Preview:
            {sample}

            Columns:
            {', '.join(df.columns)}

            User Question:
            {user_query}

            Provide a clear and correct answer.
            """

            prompt = prompt[:900]
            result = local_generator(prompt, max_new_tokens=150)
            st.markdown("### 🤖 AI Response")
            st.write(result[0]["generated_text"])

    # ------------------------------------------------
    # 🔥 AI REPORT GENERATOR
    # ------------------------------------------------
    st.subheader("📄 Auto-Generate AI Report")

    if st.button("Generate Report"):
        if local_generator is None:
            st.error("LLM unavailable.")
        else:
            stats = df.describe(include='all').to_string()
            prompt = f"""
            Generate a detailed professional report including:
            - Overview
            - Important insights
            - Patterns & correlations
            - Business recommendations
            - Risks/anomalies

            Dataset Stats:
            {stats}
            """

            prompt = prompt[:1000]
            result = local_generator(prompt, max_new_tokens=300)
            st.markdown("### 📘 AI-Generated Report")
            st.write(result[0]["generated_text"])

    # ------------------------------------------------
    # 🔥 AI CHART CREATOR (Natural Language → Chart)
    # ------------------------------------------------
    st.subheader("📈 AI Chart Creator")

    query = st.text_input("Describe a chart (e.g., 'plot age vs salary'):")

    if query:
        cols = df.columns.tolist()
        found_cols = [c for c in cols if c.lower() in query.lower()]

        if len(found_cols) >= 2 and all(col in numeric_cols for col in found_cols[:2]):
            x, y = found_cols[:2]
            fig, ax = plt.subplots()
            ax.scatter(df[x], df[y])
            ax.set_title(f"AI Plot: {x} vs {y}")
            st.pyplot(fig)
        else:
            st.warning("AI couldn’t detect required numeric columns. Try using exact names.")

else:
    st.warning("⬆️ Upload a CSV file to continue.")
