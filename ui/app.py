from __future__ import annotations

import json
import urllib.error
import urllib.request

import streamlit as st

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Churn Demo", page_icon="📊", layout="centered")

st.title("Customer Churn Prediction Demo")
st.write("Enter customer details below and click the button to predict whether the customer is likely to leave the service.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], index=0)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0,
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=1.0,
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "tenure": int(tenure),
        "Contract": contract,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PhoneService": phone_service,
        "PaperlessBilling": paperless_billing,
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        st.error(f"API returned HTTP {exc.code}: {detail}")
    except urllib.error.URLError:
        st.error("Could not connect to the API at http://localhost:8000. Start the FastAPI service first.")
    else:
        prob = result["churn_probability"] * 100
        threshold = result["threshold"] * 100

        st.subheader("Prediction Result")
        st.progress(min(int(prob), 100))

        st.write(f"**Chance of customer leaving:** {prob:.1f}%")
        st.write(
            f"**Prediction:** {'Customer likely to leave' if result['prediction'] == 'Yes' else 'Customer likely to stay'}"
        )
        st.write(f"**Decision threshold used:** {threshold:.0f}%")

        if result["prediction"] == "Yes":
            st.error("This customer is at higher risk of churning.")
        else:
            st.success("This customer is likely to stay.")