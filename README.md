# Machine Learning - Customer Purchase Likelihood Prediction

A machine learning project that predicts customer purchase likelihood using XGBoost and KNN algorithms. The project includes a Streamlit web application for interactive predictions and data visualization.

## 📋 Overview

This project predicts customer purchase likelihood (High, Medium, Low) based on various customer behavioral features such as age, gender, spending patterns, browsing behavior, and purchase history.

## ✨ Features

- **XGBoost Model**: Gradient boosting model for purchase likelihood prediction
- **KNN Model**: K-Nearest Neighbors classifier as an alternative model
- **Streamlit Web App**: Interactive web interface for making predictions and visualizing results
- **Data Generation**: Script to generate synthetic customer data for training
- **Pre-trained Model**: Includes a trained model ready for predictions

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost streamlit joblib
```

Or install from requirements.txt (if available):

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Machine-Leaning-/
├── main.py                      # Streamlit web application
├── machine_leaning_model.py     # XGBoost model training script
├── knn-model.py                 # KNN model training script
├── Genrat_data.py               # Data generation script
├── demo_customer_data.csv        # Demo customer dataset
├── purchase_likelihood_model.pkl # Pre-trained XGBoost model
└── README.md                    # Project documentation
```

## 🎯 Usage

### 1. Generate Demo Data (Optional)

If you need to generate new customer data:

```bash
python Genrat_data.py
```

This will create or update `demo_customer_data.csv` with synthetic customer data.

### 2. Train the Model

Train a new XGBoost model:

```bash
python machine_leaning_model.py
```

This will:
- Load and preprocess the customer data
- Train an XGBoost classifier
- Save the trained model as `purchase_likelihood_model.pkl`
- Display model accuracy and classification report

### 3. Train KNN Model (Alternative)

Train a K-Nearest Neighbors model:

```bash
python knn-model.py
```

### 4. Run the Streamlit Application

Launch the interactive web application:

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501` where you can:
- View customer data with predictions
- See purchase likelihood scores
- Analyze customer behavior patterns

## 🔧 Model Details

### Features Used for Prediction

- **Age**: Customer age (18-65)
- **Gender**: Male/Female
- **TotalSpend**: Total amount spent by customer
- **TotalPurchases**: Number of purchases made
- **DaysSinceLastPurchase**: Days since last purchase (1-365)
- **CategoryViewed**: Product category viewed (Electronics, Clothing, Groceries, Furniture, Toys)
- **TimeSpentOnCategory**: Time spent browsing category (in minutes)
- **CartAbandonmentRate**: Rate of cart abandonment (0-1)

### Target Variable

- **PurchaseLikelihood**: Categorical variable with three levels
  - High (2)
  - Medium (1)
  - Low (0)

## 📊 Model Performance

The models are evaluated using:
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)

## 🛠️ Technologies Used

- **Python**: Programming language
- **XGBoost**: Gradient boosting framework
- **scikit-learn**: Machine learning library
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **joblib**: Model serialization

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**Gen210**

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a demo project using synthetic data. For production use, ensure you have appropriate real-world data and model validation procedures in place.
