import streamlit as st
import numpy as np
import pickle


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(page_title="Credit Card Usage Segmentation", layout="centered")
st.title("💳 Credit Card Usage Segmentation")
st.markdown("Enter customer data to predict their segment/group based on usage patterns.")

# -----------------------------
# Input fields (update based on your features)
# -----------------------------
balance = st.number_input("Balance", min_value=0.0, format="%.2f")
purchases = st.number_input("Purchases", min_value=0.0, format="%.2f")
cash_advance = st.number_input("Cash Advance", min_value=0.0, format="%.2f")
credit_limit = st.number_input("Credit Limit", min_value=0.0, format="%.2f")
payments = st.number_input("Payments", min_value=0.0, format="%.2f")
tenure = st.slider("Tenure (in months)", 6, 24, 12)

# Combine input into a single array (match model training order)
features = np.array([[balance, purchases, cash_advance, credit_limit, payments, tenure]])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Segment Customer"):
    if scaler:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features

    cluster = model.predict(features_scaled)[0]

    st.success(f"🧾 This customer belongs to *Segment {cluster}*.")
    st.info("This segment represents customers with similar credit card usage patterns.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit | Clustering Model Deployment © 2025")