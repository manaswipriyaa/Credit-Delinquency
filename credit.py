import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('Delinquency_prediction_dataset.csv')

# print(df.head())

# print("Shape:", df.shape)
# print("\nColumn Names:", df.columns.tolist())

missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing %': missing_percent}).sort_values(by='Missing %', ascending=False)
# print("\n--- Missing Value Summary ---")
# print(missing_summary[missing_summary['Missing Count'] > 0])

duplicate_rows = df[df.duplicated()]
# print(f"\nNumber of Duplicate Rows: {len(duplicate_rows)}")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
z_scores = np.abs(zscore(df[numeric_cols].dropna()))
outlier_counts = (z_scores > 3).sum(axis=0)
# print("\nOutlier Count (Z > 3) per Numeric Column : ")
# print(pd.Series(outlier_counts, index=numeric_cols))

if 'Income' in df.columns and 'Employment_Status' in df.columns and 'Location' in df.columns:
    df['Income'] = df.groupby(['Employment_Status', 'Location'])['Income'].transform(lambda x: x.fillna(x.median()))

if 'Credit_Score' in df.columns:
    df.fillna(df['Credit_Score'].mean(), inplace=True)

if 'Loan_Balance' in df.columns:
    df.fillna(df['Loan_Balance'].median(), inplace=True)

if 'Credit_Utilization' in df.columns:
    df.fillna(df['Credit_Utilization'].median(), inplace=True)

if 'Debt_to_Income_Ratio' in df.columns:
    df.fillna(df['Debt_to_Income_Ratio'].median(), inplace=True)

if 'Missed_Payments' in df.columns:
    df.fillna({'Missed_Payments': 0}, inplace=True)

if 'Account_Tenure' in df.columns:
    df.fillna(df['Account_Tenure'].median(), inplace=True)

# print("\nRemaining Missing Values After Imputation : ")
# print(df.isnull().sum()[df.isnull().sum() > 0])

# df.to_csv("Cleaned_Delinquency_Dataset.csv", index=False)
# print("\nCleaned dataset saved as 'Cleaned_Delinquency_Dataset.csv'") 

# print(df) 

# print("Missing values before handling:")
# print(df['Credit_Utilization'].isnull().sum())

df['Credit_Utilization_missing'] = df['Credit_Utilization'].isnull().astype(int)

median_util = df['Credit_Utilization'].median()
df['Credit_Utilization_median_imputed'] = df['Credit_Utilization'].fillna(median_util)

if 'Employment_Status' in df.columns:
    df['Credit_Utilization_group_imputed'] = df['Credit_Utilization']
    df['Credit_Utilization_group_imputed'] = (df.groupby('Employment_Status')['Credit_Utilization'].transform(lambda x: x.fillna(x.median())))

features = ['Income', 'Loan_Balance', 'Account_Tenure', 'Credit_Score']
features = [f for f in features if f in df.columns]

model_data = df[features + ['Credit_Utilization']].dropna()
missing_data = df[df['Credit_Utilization'].isnull()]

model_data = model_data.dropna()
if not model_data.empty and not missing_data.empty:
    X = model_data[features]
    y = model_data['Credit_Utilization']

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

#     df.loc[df['Credit_Utilization'].isnull(), 'Credit_Utilization_model_imputed'] = rf_model.predict(df.loc[df['Credit_Utilization'].isnull(), features])

if 'Credit_Utilization_model_imputed' in df.columns:
    df['Credit_Utilization_final'] = df['Credit_Utilization']
    df.loc[df['Credit_Utilization'].isnull(), 'Credit_Utilization_final'] = df['Credit_Utilization_model_imputed']
elif 'Credit_Utilization_group_imputed' in df.columns:
    df['Credit_Utilization_final'] = df['Credit_Utilization_group_imputed']
else:
    df['Credit_Utilization_final'] = df['Credit_Utilization_median_imputed']

# print("\nMissing values after imputation (final column):")
# print(df['Credit_Utilization_final'].isnull().sum())

df.drop(columns=['Credit_Utilization_median_imputed', 'Credit_Utilization_group_imputed', 'Credit_Utilization_model_imputed'], inplace=True, errors='ignore')

# # df.to_csv("Cleaned_Credit_Utilization.csv", index=False)
# # print("\nCleaned dataset saved as 'Cleaned_Credit_Utilization.csv'") 

income_non_missing = df['Income'].dropna()
income_mean = income_non_missing.mean()
income_std = income_non_missing.std()

# print(f"Estimated Income Mean: {income_mean:.2f}")
# print(f"Estimated Income Std Dev: {income_std:.2f}")

num_missing = df['Income'].isnull().sum()
# print(f"Number of missing income entries: {num_missing}")

synthetic_income = np.random.normal(loc=income_mean, scale=income_std, size=num_missing)

synthetic_income = np.clip(synthetic_income, a_min=0, a_max=None)

# df.loc[df['Income'].isnull(), 'Income'] = synthetic_income

# print(f"Remaining missing in Income: {df['Income'].isnull().sum()}") 

# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()

df['Income_Quartile'] = pd.qcut(df['Income'], 4, labels=["Low", "Mid-Low", "Mid-High", "High"])
delinq_by_income = df.groupby('Income_Quartile')['Delinquent_Account'].mean()
# print(delinq_by_income)

df['Util_Band'] = pd.cut(df['Credit_Utilization'], bins=[0, 0.5, 0.8, 1.0], labels=["Low", "Moderate", "High"])
delinq_by_util = df.groupby('Util_Band')['Delinquent_Account'].mean()
# print(delinq_by_util) 

numeric_cols = df.select_dtypes(include='number')
corr = numeric_cols.corr()
# plt.figure(figsize=(10, 6))
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title("Correlation Matrix")
# plt.show()

features_to_plot = ['Credit_Score', 'Income', 'Credit_Utilization', 'Missed_Payments', 'Debt_to_Income_Ratio']

# for feature in features_to_plot:
#     sns.boxplot(x='Delinquent_Account', y=feature, data=df)
#     plt.title(f'{feature} vs Delinquency')
#     plt.show()

categorical_features = ['Employment_Status', 'Location']
for cat in categorical_features:
    cross_tab = pd.crosstab(df[cat], df['Delinquent_Account'], normalize='index')
    cross_tab.plot(kind='bar', stacked=True, colormap='coolwarm')
    # plt.title(f'{cat} vs Delinquency Rate')
    # plt.ylabel('Proportion')
    # plt.show()  