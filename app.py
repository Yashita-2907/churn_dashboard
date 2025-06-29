import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# --- Load Model ---
model = joblib.load("model.pkl")

# --- Load Built-in Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/churn.csv")
    return df

df = load_data()

# --- Page Setup ---
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction Dashboard")
st.markdown("Predict the likelihood of customer churn using a pre-trained machine learning model.")

# --- Sidebar: Filter for Preview ---
st.sidebar.header("🔍 Data Filter")
contract_type = st.sidebar.multiselect("Contract Type", options=df["Contract"].unique(), default=df["Contract"].unique())
internet_service = st.sidebar.multiselect("Internet Service", options=df["InternetService"].unique(), default=df["InternetService"].unique())

filtered_df = df[
    (df["Contract"].isin(contract_type)) &
    (df["InternetService"].isin(internet_service))
]

st.subheader("🔢 Sample of Input Data")
st.dataframe(filtered_df.head(20), use_container_width=True)

# --- Preprocessing (match model expectations) ---
def preprocess(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(0, inplace=True)

    # Drop non-model columns if present
    drop_cols = ["customerID", "Churn"] if "Churn" in df.columns else ["customerID"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df)
    return df

# --- Predict ---
X = preprocess(filtered_df)
# Align columns to match model input
model_features = joblib.load("model_features.pkl")  # Contains list of columns model was trained on
for col in model_features:
    if col not in X.columns:
        X[col] = 0
X = X[model_features]

preds = model.predict_proba(X)[:, 1]
filtered_df["Churn Probability"] = preds
filtered_df["Risk Level"] = pd.cut(preds, bins=[0, 0.3, 0.7, 1], labels=["Low", "Medium", "High"])

# --- Display Predictions ---
st.subheader("📈 Churn Risk Predictions")
st.dataframe(filtered_df[["customerID", "Churn Probability", "Risk Level"]].head(20), use_container_width=True)

# --- Visualization ---
fig = px.histogram(filtered_df, x="Risk Level", color="Risk Level", title="Distribution of Churn Risk Levels")
st.plotly_chart(fig, use_container_width=True)

# --- Download Button ---
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Predictions as CSV", csv, "churn_predictions.csv", "text/csv")

# --- Footer ---
st.markdown("---")
st.markdown("Made using Streamlit & XGBoost | Demo limited to Telco Churn dataset.")
