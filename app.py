import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------
# Load Data & Model
# -------------------------------
DATA_PATH = "/mnt/data/student_scores (1).csv"
MODEL_PATH = "/mnt/data/Student_model (2).pkl"

df = pd.read_csv(DATA_PATH)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Student Score Dashboard", layout="wide")

st.title("📊 Student Score Dynamic Dashboard & Prediction App")

menu = ["Dashboard", "Predict Score"]
choice = st.sidebar.selectbox("Navigation", menu)

# -----------------------------------------
# 📌 Dashboard
# -----------------------------------------
if choice == "Dashboard":
    st.header("📈 Data Overview")

    st.dataframe(df)

    st.subheader("🔍 Select Column to Visualize")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    col = st.selectbox("Choose a numeric column", numeric_cols)

    fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Correlation Heatmap")
    fig2 = px.imshow(df.corr(), text_auto=True, aspect="auto",
                     title="Correlation Matrix")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------
# 📌 Predict Score
# -----------------------------------------
elif choice == "Predict Score":
    st.header("🎯 Predict Student Score")

    # assume the model takes study_hours as input
    study_hours = st.number_input("Enter Study Hours:", min_value=0.0, max_value=12.0, step=0.5)

    if st.button("Predict"):
        prediction = model.predict([[study_hours]])
        st.success(f"Predicted Score: **{prediction[0]:.2f}**")

