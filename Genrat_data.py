import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate a demo dataset with 1000 customers
n_customers = 1000

# Basic customer info
ages = np.random.randint(18, 65, size=n_customers)  # Age between 18 and 65
genders = np.random.choice(['Male', 'Female'], size=n_customers)  # Gender
total_spend = np.random.uniform(100, 5000, size=n_customers)  # Total spend between 100 and 5000

# Customer activity
total_purchases = np.random.randint(0, 50, size=n_customers)  # Number of purchases
days_since_last_purchase = np.random.randint(1, 365, size=n_customers)  # Days since last purchase
category_viewed = np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Furniture', 'Toys'], size=n_customers)
time_spent_on_category = np.random.uniform(5, 180, size=n_customers)  # Time spent on category in minutes
cart_abandonment_rate = np.random.uniform(0, 1, size=n_customers)  # Cart abandonment rate (0 to 1)

# Target: Purchase likelihood (High, Medium, Low)
purchase_likelihood = np.random.choice(['High', 'Medium', 'Low'], size=n_customers, p=[0.3, 0.5, 0.2])

# Combine into a DataFrame
data = pd.DataFrame({
    'CustomerID': np.arange(1, n_customers + 1),
    'Age': ages,
    'Gender': genders,
    'TotalSpend': total_spend,
    'TotalPurchases': total_purchases,
    'DaysSinceLastPurchase': days_since_last_purchase,
    'CategoryViewed': category_viewed,
    'TimeSpentOnCategory': time_spent_on_category,
    'CartAbandonmentRate': cart_abandonment_rate,
    'PurchaseLikelihood': purchase_likelihood
})

# Save to CSV
data.to_csv('demo_customer_data.csv', index=False)

# Show a snippet of the data
print(data.head())
