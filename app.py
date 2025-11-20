import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------
# Load Data & Model
# -------------------------------
DATA_PATH = "student_scores.csv"
MODEL_PATH = "Student_model.pkl"

df = pd.read_csv(DATA_PATH)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="Student Score Prediction App", layout="wide")

st.title("🎓 Student Score Dashboard & Prediction App")

menu = ["Dashboard", "Predict Score"]
choice = st.sidebar.selectbox("Navigation", menu)

# -----------------------------------------
# Dashboard
# -----------------------------------------
if choice == "Dashboard":
    st.header("📊 Dataset Overview")
    st.dataframe(df)

    st.subheader("📌 Select Column to Visualize")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if numeric_cols.any():
        col = st.selectbox("Select Column", numeric_cols)
        fig = px.histogram(df, x=col, nbins=20)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        fig2 = px.imshow(df.corr(), text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------
# Prediction
# -----------------------------------------
elif choice == "Predict Score":
    st.header("🎯 Predict Student Score")

    # Assuming model needs only Study Hours
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=15.0, step=0.5)

    if st.button("Predict"):
        prediction = model.predict([[study_hours]])
        st.success(f"Predicted Score: {prediction[0]:.2f}")
