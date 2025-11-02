import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load('purchase_likelihood_model.pkl')

# Load the demo customer data
data = pd.read_csv('demo_customer_data.csv')

# Preprocess the data (same as done during training)
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

le_category = LabelEncoder()
data['CategoryViewed'] = le_category.fit_transform(data['CategoryViewed'])

# Features for prediction
features = ['Age', 'Gender', 'TotalSpend', 'TotalPurchases', 'DaysSinceLastPurchase', 'CategoryViewed', 'TimeSpentOnCategory', 'CartAbandonmentRate']
X = data[features]

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predictions
predictions = model.predict(X_scaled)
data['PurchaseLikelihood'] = predictions

# Map the numeric predictions back to labels
data['PurchaseLikelihood'] = data['PurchaseLikelihood'].map({2: 'High', 1: 'Medium', 0: 'Low'})

# Streamlit UI setup
st.title("Customer Segmentation for Retail Sales Strategy")

# Total number of customers
total_customers = len(data)
st.subheader(f"Total Number of Customers: {total_customers}")

# Purchase likelihood distribution
purchase_counts = data['PurchaseLikelihood'].value_counts()
st.subheader("Purchase Likelihood Distribution")
st.bar_chart(purchase_counts)

# High Priority Data - Focus on High Chance Customers
high_chance_customers = data[data['PurchaseLikelihood'] == 'High']

# Number of high-priority customers
high_priority_count = len(high_chance_customers)
st.subheader(f"High Priority Customers (Likelihood = High): {high_priority_count}")

# Top high-priority customers table
st.subheader("Top High-Priority Customers")
high_priority_columns = ['CustomerID', 'Age', 'Gender', 'TotalSpend', 'TotalPurchases', 'DaysSinceLastPurchase', 'CategoryViewed', 'TimeSpentOnCategory', 'CartAbandonmentRate']
st.dataframe(high_chance_customers[high_priority_columns].head())

# Show summary statistics for High-Priority Customers
high_priority_summary = high_chance_customers[['TotalSpend', 'TotalPurchases', 'DaysSinceLastPurchase', 'TimeSpentOnCategory']].describe()
st.subheader("Summary Statistics for High Priority Customers")
st.dataframe(high_priority_summary)

# Sales Strategy Recommendations
st.subheader("Sales Strategy Recommendations")
st.write("For **High Priority Customers**, focus on immediate sales campaigns, personalized offers, or loyalty programs.")
st.write(f"High Priority Customers: {high_chance_customers['CustomerID'].tolist()}")

# Medium Priority Data - Focus on Medium Chance Customers
medium_chance_customers = data[data['PurchaseLikelihood'] == 'Medium']

# Number of medium-priority customers
medium_priority_count = len(medium_chance_customers)
st.subheader(f"Medium Priority Customers (Likelihood = Medium): {medium_priority_count}")

# Top medium-priority customers table
st.subheader("Top Medium-Priority Customers")
medium_priority_columns = ['CustomerID', 'Age', 'Gender', 'TotalSpend', 'TotalPurchases', 'DaysSinceLastPurchase', 'CategoryViewed', 'TimeSpentOnCategory', 'CartAbandonmentRate']
st.dataframe(medium_chance_customers[medium_priority_columns].head())

# Show summary statistics for Medium-Priority Customers
medium_priority_summary = medium_chance_customers[['TotalSpend', 'TotalPurchases', 'DaysSinceLastPurchase', 'TimeSpentOnCategory']].describe()
st.subheader("Summary Statistics for Medium Priority Customers")
st.dataframe(medium_priority_summary)

st.write("For **Medium Priority Customers**, consider offering discounts or targeted promotions to increase the likelihood of purchase.")

# Actionable Insights Based on Segmentation
st.subheader("Actionable Insights")
st.write("For **Low Priority Customers**, low engagement is detected. It's important to re-engage them with marketing efforts such as retargeting ads, emails, or special offers.")
