# Copyright (c) 2025 [B. C. PUSHYA]
# Licensed under the MIT License (see LICENSE for details)

import streamlit as st
import requests

st.title("Customer Segment Predictor")

# Input fields matching ALL required features
customer_id = st.number_input("Customer ID", min_value=1, value=1000)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=10000, value=50000)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
work_experience = st.number_input("Work Experience (years)", min_value=0, value=5)
family_size = st.number_input("Family Size", min_value=1, value=2)
purchases_last_month = st.number_input("Purchases Last Month", min_value=0, value=3)
feedback = st.text_area("Feedback (optional)")

if st.button("Predict Customer Segment"):
    try:
        data = {
            "numerical_features": [
                customer_id,
                age,
                annual_income,
                spending_score,
                work_experience,
                family_size,
                purchases_last_month
            ],
            "text": feedback if feedback else ""
        }
        
        response = requests.post(
            "http://localhost:5000/predict",
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Segment: {result['predicted_segment']}")
        else:
            st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
