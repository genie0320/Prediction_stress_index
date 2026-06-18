import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

# Prepare all features
df = train.copy()
df['medical_history'] = df['medical_history'].fillna('none')
df['family_medical_history'] = df['family_medical_history'].fillna('none')
df['edu_level'] = df['edu_level'].fillna('Unknown')
df['mean_working'] = df['mean_working'].fillna(-1)

# One-hot encode categorical features
cat_cols = ['gender', 'activity', 'smoke_status', 'medical_history', 'family_medical_history', 'sleep_pattern', 'edu_level']
df_encoded = pd.get_dummies(df.drop(columns=['ID', 'stress_score']), columns=cat_cols, drop_first=False)

# Target
y = df['stress_score'].values
X = df_encoded.values

# Compute Pearson correlation
corrs = []
for col in df_encoded.columns:
    corr = df_encoded[col].corr(df['stress_score'])
    corrs.append((col, corr))

corrs_df = pd.DataFrame(corrs, columns=['feature', 'pearson_r']).sort_values('pearson_r', key=abs, ascending=False)
print("=== Top Pearson Correlations with stress_score ===")
print(corrs_df.head(20))

# Compute Mutual Information
mi = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': df_encoded.columns,
    'mutual_info': mi
}).sort_values('mutual_info', ascending=False)
print("\n=== Top Mutual Information with stress_score ===")
print(mi_df.head(20))

# Linear Regression on all encoded features
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
print(f"\nLinear Regression Train R²: {r2_score(y, y_pred):.6f}")
