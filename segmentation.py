import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

cluster_labels = {
    0: "High income, high spending → Premium Customers",
    1: "Medium income, high spending → Value Seekers",
    2: "High web purchases, low store purchases → Digital Buyers",
    3: "Low income, medium spending → Thrifty Customers",
    4: "Young, medium income → Potential Loyalists",
    5: "Low recency, inactive → Dormant Customers",
    6: "Low income, low spending → Budget Customers"
}

st.title("Customer Segmentation App")  
st.write("Enter customer details to predict the segment.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=0, max_value=20000, value=5000)
spending_score = st.number_input("Total Spending (Sum of purchases)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
recency = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=365, value=30)


input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Total_spending': [spending_score],
    'NumWebPurchases': [num_web_purchases],
    'NumStorePurchases': [num_store_purchases],
    'NumWebVisitsMonth': [num_web_visits],
    'Recency': [recency]
})


input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    label = cluster_labels.get(cluster, "Unknown Segment")
    st.success(f"The predicted customer segment is Cluster {cluster}: {label}")
