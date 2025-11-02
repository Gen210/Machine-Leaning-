import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import joblib

# Load the demo dataset
data = pd.read_csv('demo_customer_data.csv')

# Preprocess the data
# Encode categorical features (Gender, CategoryViewed)
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

le_category = LabelEncoder()
data['CategoryViewed'] = le_category.fit_transform(data['CategoryViewed'])

# Features and target
features = ['Age', 'Gender', 'TotalSpend', 'TotalPurchases', 'DaysSinceLastPurchase', 'CategoryViewed', 'TimeSpentOnCategory', 'CartAbandonmentRate']
X = data[features]
y = data['PurchaseLikelihood']

# Convert target to numeric for classification (High = 2, Medium = 1, Low = 0)
y = y.map({'High': 2, 'Medium': 1, 'Low': 0})

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model (XGBoost)
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)



# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Evaluate the model
print(classification_report(y_test, y_pred))

# Save the trained model for later use
joblib.dump(model, 'purchase_likelihood_model.pkl')
