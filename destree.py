# using Decision Tree classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

df = pd.read_csv('Delinquency_prediction_dataset.csv')

features = ["Income", "Credit_Utilization", "Missed_Payments"]
target = "Delinquent_Account"  

X = df[features].copy()
y = df[target].copy()

X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),          
    ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
])

param_grid = {
    "clf__max_depth": [3, 5, 8, 12, None],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf": [1, 2, 5, 10],
    "clf__criterion": ["gini", "entropy"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

tree = best_model.named_steps["clf"]
importances = tree.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)
print("\nFeature importances:\n", feat_importance)

try:
    txt = export_text(tree, feature_names=features)
    print("\nDecision tree rules:\n", txt)
except Exception:
    pass

plt.figure(figsize=(10, 6))
plot_tree(tree, feature_names=features, class_names=["No", "Yes"], filled=True, rounded=True, max_depth=3)
plt.title("Decision Tree (top levels)")
plt.show()

new_customer = pd.DataFrame({
    "Income": [55000],
    "Credit_Utilization": [0.45],
    "Missed_Payments": [2]
})

risk_prob = best_model.predict_proba(new_customer)[:, 1][0]
risk_label = best_model.predict(new_customer)[0]
print(f"\nNew customer predicted delinquency probability: {risk_prob:.2%}, label: {risk_label}") 