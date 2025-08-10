# using logistic regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

df = pd.read_csv('Delinquency_prediction_dataset.csv')

features = ["Income", "Credit_Utilization", "Missed_Payments"]
target = "Delinquent_Account"   

X = df[features]
y = df[target]

X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

new_customer = pd.DataFrame({"Income": [55000], "Credit_Utilization": [0.45], "Missed_Payments": [2]})

new_customer_scaled = scaler.transform(new_customer)

risk_prob = model.predict_proba(new_customer_scaled)[:, 1][0]
print(f"Predicted Delinquency Risk: {risk_prob:.2%}") 