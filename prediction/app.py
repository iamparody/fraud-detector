import streamlit as st
import pandas as pd
import random
import joblib

# ---- Load model ----
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()  # ✅ define it globally once

st.title("💳 Fraud Detection Demo")
st.write("Enter transaction details or use the sample buttons below:")

# ---- Sample generators ----
def sample_legit():
    return {
        "Time": random.uniform(10000, 50000),
        "Amount": random.uniform(1, 200),
        **{f"V{i}": random.uniform(-1, 1) for i in range(1, 29)}
    }

def sample_fraud():
    return {
        "Time": random.uniform(50000, 90000),
        "Amount": random.uniform(500, 3000),
        **{f"V{i}": random.uniform(-5, 5) for i in range(1, 29)}
    }

# ---- Buttons to populate sample data ----
col1, col2 = st.columns(2)
if "example" not in st.session_state:
    st.session_state.example = sample_legit()

with col1:
    if st.button("🎯 Load Legitimate Example"):
        st.session_state.example = sample_legit()
with col2:
    if st.button("⚠️ Load Fraud Example"):
        st.session_state.example = sample_fraud()

example = st.session_state.example

# ---- Form ----
with st.form("fraud_form"):
    st.subheader("Transaction Details")

    time = st.number_input("Time", value=example["Time"])
    amount = st.number_input("Amount", value=example["Amount"])

    v_features = {}
    for i in range(1, 5):  # only first 4 to simplify UI
        v_features[f"V{i}"] = st.number_input(f"V{i}", value=example[f"V{i}"])

    submitted = st.form_submit_button("🔍 Predict")

# ---- Prediction ----
if submitted:
    input_data = pd.DataFrame([{
        "Time": time,
        "Amount": amount,
        **v_features,
        **{f"V{i}": 0 for i in range(5, 29)}  # fill missing
    }])

    prob = model.predict_proba(input_data)[0][1]
    label = "⚠️ FRAUD" if prob > 0.5 else "✅ LEGITIMATE"

    st.markdown("---")
    st.metric("Prediction", label, delta=f"{prob*100:.2f}% fraud probability")
