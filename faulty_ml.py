import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

csv_path = 'sales_data.csv'

# If CSV doesn't exist, create a small demo dataset
if not os.path.exists(csv_path):
    df = pd.DataFrame({
        'marketing_spend': [1000, 1500, 2000, 2500, 3000, 1200, 1800, 2200, 2700, 3200],
        'store_visits': [200, 230, 260, 300, 330, 210, 250, 280, 310, 350],
        'total_sales': [5000, 6200, 7100, 8300, 9200, 5500, 6800, 7600, 8900, 9800]
    })
    df.to_csv(csv_path, index=False)

# Load dataset
data = pd.read_csv(csv_path)

# Define features and target
X = data[['marketing_spend', 'store_visits']]
y = data['total_sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("Predictions:", predictions)


