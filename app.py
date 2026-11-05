import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Predictor")
st.caption("Enter customer details to estimate the chance they will leave.")

YN = ["Yes", "No"]

with st.form("customer_form"):
    st.subheader("Customer profile")
    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    senior = c2.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = c1.selectbox("Partner", YN)
    dependents = c2.selectbox("Dependents", YN)

    st.subheader("Services")
    c1, c2 = st.columns(2)
    phone = c1.selectbox("Phone service", YN)
    multi = c2.selectbox("Multiple lines", YN)
    internet = c1.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = c2.selectbox("Online security", YN, index=1)
    backup = c1.selectbox("Online backup", YN, index=1)
    protection = c2.selectbox("Device protection", YN, index=1)
    support = c1.selectbox("Tech support", YN, index=1)
    tv = c2.selectbox("Streaming TV", YN, index=1)
    movies = c1.selectbox("Streaming movies", YN, index=1)

    st.subheader("Account")
    c1, c2 = st.columns(2)
    contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = c2.selectbox("Paperless billing", YN)
    payment = c1.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    tenure = c2.number_input("Tenure (months)", 0, 100, 12)
    monthly = c1.number_input("Monthly charges", 0.0, 500.0, 70.0, step=0.5)
    total = c2.number_input("Total charges", 0.0, 100000.0, 840.0, step=1.0)

    submitted = st.form_submit_button("Predict churn", type="primary")

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone,
        "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": float(monthly), "TotalCharges": float(total),
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if resp.status_code == 200:
            out = resp.json()
            prob = out["churn_probability"]
            st.divider()
            st.metric("Churn probability", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            msg = f"{out['label']}  |  Risk level: {out['risk_level']}"
            if out["churn_prediction"]:
                st.error(msg)
            else:
                st.success(msg)
        else:
            st.error(f"API error {resp.status_code}: {resp.json().get('detail', resp.text)}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}. Is it running?\n\n{exc}")