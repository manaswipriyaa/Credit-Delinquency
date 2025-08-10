# Credit Delinquency Prediction

## 📌 Overview
This project predicts **which customers are most likely to become credit delinquent** using multiple machine learning models.  
By analyzing financial and behavioral data such as income, credit utilization, and payment history, the models estimate a **risk score** for each customer.  
We compare **five different approaches** — **Logistic Regression**, **Decision Tree**, **Random Forest**, **XGBoost**, and **LightGBM** — to identify the most accurate and fair predictor.

---

## 📊 Features Used
- **Income**
- **Credit Utilization Ratio**
- **Number of Missed Payments**
- **Credit History Length**
- **Debt-to-Income Ratio**
- Other relevant financial/behavioral variables

---

## ⚙️ Models Implemented

### 1️⃣ Logistic Regression
- **Type:** Linear model
- **Strengths:** Highly interpretable, fast to train
- **Weaknesses:** Cannot capture complex patterns without feature engineering
- **Best Use Case:** Simple, explainable credit scoring

### 2️⃣ Decision Tree
- **Type:** Single decision tree classifier
- **Strengths:** Easy to visualize and interpret, captures non-linear relationships
- **Weaknesses:** Can overfit on small datasets
- **Best Use Case:** Clear decision-making rules

### 3️⃣ Random Forest
- **Type:** Ensemble of decision trees (bagging)
- **Strengths:** Captures non-linearities, robust to noise
- **Weaknesses:** Less interpretable
- **Best Use Case:** General-purpose prediction

### 4️⃣ XGBoost
- **Type:** Gradient boosting
- **Strengths:** High accuracy, handles complex patterns
- **Weaknesses:** Requires tuning, slower on very large data
- **Best Use Case:** Competitive modeling where accuracy is top priority

### 5️⃣ LightGBM
- **Type:** Gradient boosting (leaf-wise growth)
- **Strengths:** Very fast on large datasets, native categorical support
- **Weaknesses:** Can overfit small datasets
- **Best Use Case:** Very large datasets where speed matters

---

## 📈 Evaluation Metrics
- **ROC-AUC** – Ability to rank high vs. low risk customers  
- **Precision** – Percent of flagged customers truly high-risk  
- **Recall** – Percent of actual high-risk customers caught  
- **F1-Score** – Balance between precision and recall  
- **Calibration** – Probability predictions match real-world outcomes  

**Interpretation:**  
- Higher values generally = better performance  
- Precision reduces false alarms, recall finds more risky customers  
- Calibration ensures risk scores are realistic  

---

## 🛡 Bias Detection & Mitigation
- Compare performance across demographic groups  
- Use fairness metrics to check for disparities  
- Retrain, re-weight, or adjust features if bias is detected  

---

## ⚖ Ethical Considerations
- Avoid using features that could cause discrimination  
- Explain predictions and allow human review  
- Protect customer privacy and data security
 
---

## 👤 Author
**Maddu Manaswi Priya**  
**August 2025**
