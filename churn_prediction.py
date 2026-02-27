# -*- coding: utf-8 -*-
"""
Telco Churn AI - Premium Single Page Dashboard
Author: Govind
"""

import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Churn AI",
    layout="wide",
    page_icon="📡"
)

# --------------------------------------------------
# SESSION STATE DEFAULTS
# --------------------------------------------------
defaults = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 1000.0,
    "Contract": "Month-to-month",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# MODERN CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f172a, #020617, #1e293b, #000000);
    background-size: 400% 400%;
    animation: gradient 18s ease infinite;
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.section {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 40px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}
.stButton>button {
    border-radius: 50px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    background: linear-gradient(90deg,#06b6d4,#6366f1,#8b5cf6);
    color: white;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL LOADING (RENDER SAFE)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

@st.cache_resource
def load_model():
    with open(MODEL_DIR / "xgboost_churn_model.sav", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------------------------------
# FEATURE ALIGNMENT
# --------------------------------------------------
def align_features(df, model):
    df = pd.get_dummies(df)
    model_features = model.get_booster().feature_names

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df = df[model_features]
    return df

# --------------------------------------------------
# DEMO + CLEAR FUNCTIONS
# --------------------------------------------------
def load_demo():
    st.session_state.tenure = 2
    st.session_state.MonthlyCharges = 95.0
    st.session_state.TotalCharges = 200.0
    st.session_state.Contract = "Month-to-month"
    st.session_state.InternetService = "Fiber optic"
    st.session_state.OnlineSecurity = "No"
    st.session_state.TechSupport = "No"
    st.session_state.PaperlessBilling = "Yes"
    st.session_state.PaymentMethod = "Electronic check"

def clear_values():
    for key in defaults:
        st.session_state[key] = defaults[key]

# ==================================================
# HERO
# ==================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.title("📡 Telco Customer Churn Intelligence Platform")
st.markdown("""
AI-powered system to predict customer churn risk  
and generate actionable retention strategies.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# INPUT SECTION
# ==================================================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🧾 Customer Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.slider("Tenure (Months)", 0, 72,
              key="tenure")
    st.number_input("Monthly Charges",
                    min_value=0.0,
                    key="MonthlyCharges")
    st.number_input("Total Charges",
                    min_value=0.0,
                    key="TotalCharges")

with col2:
    st.selectbox("Contract Type",
                 ["Month-to-month","One year","Two year"],
                 key="Contract")
    st.selectbox("Internet Service",
                 ["DSL","Fiber optic","No"],
                 key="InternetService")
    st.selectbox("Online Security",
                 ["Yes","No"],
                 key="OnlineSecurity")

with col3:
    st.selectbox("Tech Support",
                 ["Yes","No"],
                 key="TechSupport")
    st.selectbox("Paperless Billing",
                 ["Yes","No"],
                 key="PaperlessBilling")
    st.selectbox("Payment Method",
                 ["Electronic check",
                  "Mailed check",
                  "Bank transfer (automatic)",
                  "Credit card (automatic)"],
                 key="PaymentMethod")

colA, colB = st.columns(2)
colA.button("✨ Load Demo Values", on_click=load_demo)
colB.button("🧹 Clear Values", on_click=clear_values)

st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# PREDICTION
# ==================================================
if st.button("🚀 Run Prediction"):

    input_data = pd.DataFrame([{
        "tenure": st.session_state.tenure,
        "MonthlyCharges": st.session_state.MonthlyCharges,
        "TotalCharges": st.session_state.TotalCharges,
        "Contract": st.session_state.Contract,
        "InternetService": st.session_state.InternetService,
        "OnlineSecurity": st.session_state.OnlineSecurity,
        "TechSupport": st.session_state.TechSupport,
        "PaperlessBilling": st.session_state.PaperlessBilling,
        "PaymentMethod": st.session_state.PaymentMethod
    }])

    df = align_features(input_data, model)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        gauge={'axis': {'range':[0,100]}}
    ))
    col1.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        col2.error("🚨 High Risk of Churn")
    else:
        col2.success("✅ Likely to Stay")

    col2.metric("Probability", f"{probability*100:.2f}%")

    st.markdown("### 🎯 Retention Strategy")
    if probability > 0.6:
        st.write("- Offer loyalty discount")
        st.write("- Provide long-term contract")
        st.write("- Assign priority support")
    elif probability > 0.3:
        st.write("- Send promotional bundle offer")
    else:
        st.write("- Continue engagement")

    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# BATCH
# ==================================================
st.subheader("📁 Bulk Prediction")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    batch_df = pd.read_csv(uploaded)
    st.dataframe(batch_df.head())

    if st.button("Run Batch Prediction"):
        batch_encoded = align_features(batch_df, model)
        preds = model.predict(batch_encoded)
        batch_df["Churn_Prediction"] = preds
        st.success("Batch Prediction Complete")
        st.dataframe(batch_df.head())

        st.download_button(
            "Download Results",
            batch_df.to_csv(index=False),
            "churn_results.csv"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
---
Built by Govind | Production-ready • Render compatible • Feature-aligned
""")