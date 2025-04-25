import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Load the dataset (public Kaggle dataset)
df = pd.read_csv("creditcard.csv")

# Use a sample for faster prototyping (optional)
df = df.sample(frac=0.1, random_state=1)

# Separate features and labels
X = df.drop(['Class'], axis=1)
y = df['Class']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Isolation Forest for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.001, random_state=42)
model.fit(X_train)

# Predict anomalies (-1 = fraud, 1 = normal)
y_pred = model.predict(X_test)
y_pred = [1 if i == -1 else 0 for i in y_pred]  # Convert to binary: 1 = fraud

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# Simulate guarding a new transaction
def guard_transaction(transaction_data):
    pred = model.predict([transaction_data])
    if pred[0] == -1:
        return "🚨 ALERT: Transaction flagged as fraudulent!"
    else:
        return "✅ Transaction is approved."

# Example usage with a new transaction (simulate a new row from X_test)
example_transaction = X_test.iloc[0].values
result = guard_transaction(example_transaction)
print("\nReal-time Guard Result:")
print(result)
