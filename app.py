from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_DIR = Path(__file__).resolve().parent / "model"
TARGET_COLUMN = "default payment next month"

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree Classifier": "decision_tree_classifier.joblib",
    "KNN Classifier": "knn_classifier.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Random Forest Classifier": "random_forest_classifier.joblib",
}


def load_model(model_name: str):
    model_path = MODEL_DIR / MODEL_OPTIONS[model_name]
    return joblib.load(model_path)


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.set_page_config(page_title="Credit Default Prediction", page_icon="💳", layout="wide")

st.title("Credit Default Prediction")
st.caption("Evaluate a saved credit default model on the uploaded test split.")

with st.sidebar:
    st.header("Model Selection")
    model_name = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
    st.markdown("---")
    st.caption("Target variable: Default payment next month")

uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if TARGET_COLUMN not in df.columns:
        st.error(f"The uploaded CSV must contain the target column '{TARGET_COLUMN}'.")
        st.stop()

    X = df.drop(columns=[TARGET_COLUMN])
    y_true = df[TARGET_COLUMN]

    model = load_model(model_name)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.subheader(f"Evaluation metrics for {model_name}")
        metric_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
        metric_df["Value"] = metric_df["Value"].map(lambda value: round(float(value), 4))
        st.dataframe(metric_df.set_index("Metric"), use_container_width=True)

    with col2:
        st.subheader("Dataset summary")
        st.metric("Rows", len(df))
        st.metric("Default cases", int(y_true.sum()))
        st.metric("No default cases", int((1 - y_true).sum()))

    st.markdown("---")

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Actual vs Predicted Default Status")
        st.pyplot(fig)

    with right_col:
        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, target_names=["No Default", "Default"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(4)
        st.dataframe(report_df, use_container_width=True)
else:
    st.info("Please upload the generated test_data.csv file to evaluate the selected model.")
