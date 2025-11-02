from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the demo dataset
data = pd.read_csv('demo_customer_data.csv')

# Preprocess the data
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

le_category = LabelEncoder()
data['CategoryViewed'] = le_category.fit_transform(data['CategoryViewed'])

# Features for prediction
features = ['Age', 'Gender', 'TotalSpend', 'TotalPurchases', 'DaysSinceLastPurchase', 'CategoryViewed', 'TimeSpentOnCategory', 'CartAbandonmentRate']
X = data[features]

# Target variable
y = data['PurchaseLikelihood'].map({'High': 2, 'Medium': 1, 'Low': 0})

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Predict on the test data
y_pred_knn = knn.predict(X_test)

# Evaluate the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of KNN: {accuracy_knn:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
