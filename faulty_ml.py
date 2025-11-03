# ML program with syntax errors



import pandas as pd

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Load dataset (syntax error below)

data = pd.read_csv('sales_data.csv'

# Define features and target

X = data[['marketing_spend', 'store_visits']]

y = data['total_sales'

# Split data (missing parenthesis)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 random_state=42)

# Train model (syntax mistake in function call)

model = LinearRegression[]

model.fit(X_train, y_train)

# Predict

predictions = model.predict(X_test)

print("Predictions:", predictions)


