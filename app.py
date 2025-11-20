import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------
# Load Data & Model
# -------------------------------
DATA_PATH = "/mnt/data/student_scores.csv"
MODEL_PATH = "/mnt/data/Student_model (5).pkl"

df = pd.read_csv(DATA_PATH)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Student Score Prediction", layout="wide")

st.title("🎓 Student Performance Dashboard & Prediction App")

menu = ["📊 Dashboard", "🎯 Predict Score"]
choice = st.sidebar.selectbox("Navigation", menu)

# -----------------------------------------
# 📌 Dashboard Section
# -----------------------------------------
if choice == "📊 Dashboard":
    st.header("📈 Dataset Overview")
    st.dataframe(df)

    # Select numeric column
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    st.subheader("📌 Select Column for Visualization")
    col_choice = st.selectbox("Select a numeric column", numeric_cols)

    fig = px.histogram(df, x=col_choice, title=f"Distribution of {col_choice}", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📌 Correlation Heatmap")
    fig2 = px.imshow(df.corr(), text_auto=True, aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------
# 📌 Prediction Section
# -----------------------------------------
elif choice == "🎯 Predict Score":
    st.header("📘 Predict Student Score")

    # Assuming model requires single input: study_hours
    study_hours = st.number_input(
        "Enter Study Hours:",
        min_value=0.0,
        max_value=15.0,
        step=0.5
    )

    if st.button("Predict Score"):
        prediction = model.predict([[study_hours]])
        st.success(f"📌 Predicted Score: **{prediction[0]:.2f}**")
