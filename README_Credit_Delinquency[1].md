# Credit Delinquency Prediction

A machine learning project to predict credit delinquency risk - identifying borrowers likely to default or miss payments based on financial and behavioural indicators. Built as part of the Tata GenAI Powered Data Analytics Job Simulation on Forage.

---

## Problem Statement

Credit delinquency is a major risk for financial institutions. Identifying high-risk borrowers before delinquency occurs allows banks to intervene early - offering restructured repayment plans or flagging accounts for review. This project builds a predictive model to classify borrowers as high-risk or low-risk based on their financial profile.

---

## Dataset

- **Type:** Financial borrower dataset with credit history and repayment behaviour
- **Features:** Credit score, loan amount, repayment history, employment status, income, debt-to-income ratio, number of missed payments
- **Target:** Binary — delinquent (1) / not delinquent (0)

---

## Approach

1. **EDA** - analysed distributions of financial features, delinquency rates by segment, missing value patterns
2. **Data Preprocessing** - handled nulls, encoded categorical variables, scaled numerical features
3. **Feature Engineering** - created derived features such as debt-to-income ratio buckets and payment history scores
4. **Model Building** - trained and evaluated classification models
5. **Risk Assessment Framework** - proposed a tiered risk scoring system based on model output probabilities

---

## Models Used

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline interpretable model |
| Random Forest | Improved accuracy with feature importance |
| XGBoost | Best overall performance |

---

## Key Outputs

- Risk score for each borrower (probability of delinquency)
- Top 5 features driving delinquency risk
- Tiered risk segmentation: Low / Medium / High risk bands
- Recommendations for credit risk management strategy

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Power BI |
| Gen-AI Tools | Used for insight generation and report structuring |
| Notebook | Jupyter Notebook |

---

## Project Structure

```
Credit-Delinquency/
│
├── data/
│   └── credit_data.csv
├── notebooks/
│   └── credit_delinquency.ipynb
├── outputs/
│   ├── risk_distribution.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
└── README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/manaswipriyaa/Credit-Delinquency.git
cd Credit-Delinquency

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

# Launch the notebook
jupyter notebook notebooks/credit_delinquency.ipynb
```

---

## Context

This project was completed as part of the **Tata GenAI Powered Data Analytics Virtual Job Simulation** on Forage (August 2025), simulating a real-world credit risk analytics engagement.

---

## Author

**Manaswi Priya Maddu**
B.Tech - AI & Machine Learning | Acharya Nagarjuna University
[LinkedIn](https://linkedin.com/in/manaswi-priya-2126481b8) | [GitHub](https://github.com/manaswipriyaa)
