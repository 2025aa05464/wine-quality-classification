import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

# Title
st.title("Wine Quality Classification App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Model selection dropdown
model_choice = st.selectbox(
    "Choose a model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Run model when button is clicked
if uploaded_file is not None and st.button("Run Model"):
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    # Split into features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Load chosen model
    model_path = f"model/{model_choice.lower().replace(' ', '_')}.pkl"
    model = joblib.load(model_path)

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    st.write("### Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("Precision:", precision_score(y, y_pred, average="weighted"))
    st.write("Recall:", recall_score(y, y_pred, average="weighted"))
    st.write("F1:", f1_score(y, y_pred, average="weighted"))
    st.write("MCC:", matthews_corrcoef(y, y_pred))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))