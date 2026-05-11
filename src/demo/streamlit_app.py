# src/demo/streamlit_app.py
import streamlit as st
import requests
import json
import base64
import uuid

st.title("Fraud Detection Demo")
st.caption("End-to-end ML pipeline on GCP — PaySim dataset")

with st.form("transaction"):
    col1, col2 = st.columns(2)
    with col1:
        tx_type  = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT"])
        amount   = st.number_input("Amount", value=50000.0)
        old_bal  = st.number_input("Origin old balance", value=50000.0)
        new_bal  = st.number_input("Origin new balance", value=0.0)
    with col2:
        step     = st.number_input("Step (hour)", value=1, min_value=1, max_value=743)
        old_dest = st.number_input("Dest old balance", value=0.0)
        new_dest = st.number_input("Dest new balance", value=50000.0)

    submitted = st.form_submit_button("Check for fraud")

if submitted:
    payload = {
        "event_id": uuid.uuid4().hex,
        "step": int(step), "type": tx_type, "amount": amount,
        "name_orig": "C123456789", "old_balance_org": old_bal,
        "new_balance_org": new_bal, "name_dest": "C987654321",
        "old_balance_dest": old_dest, "new_balance_dest": new_dest,
        "is_fraud": 0
    }
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    envelope = {
        "message": {"data": encoded, "messageId": "demo", "publishTime": "2026-01-01T00:00:00Z"},
        "subscription": "demo"
    }
    # Call serve endpoint locally (or Cloud Run)
    try:
        with st.spinner("Scoring transaction..."):
            response = requests.post("http://localhost:8081/pubsub", json=envelope)
        result = response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Inference service not running. Start uvicorn on port 8081 first.")
        st.stop()
    

    col1, col2 = st.columns(2)
    col1.metric("Fraud score", f"{result['fraud_score']:.6f}")
    label = "🚨 FRAUD" if result["predicted_label"] else "✅ Legitimate"
    col2.metric("Decision", label)