import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

# Fill na simply for checking
train['medical_history'] = train['medical_history'].fillna('none')
train['family_medical_history'] = train['family_medical_history'].fillna('none')
train['edu_level'] = train['edu_level'].fillna('Unknown')
train['mean_working'] = train['mean_working'].fillna(-1)

# Categorical columns
cat_cols = ['gender', 'activity', 'smoke_status', 'medical_history', 'family_medical_history', 'sleep_pattern', 'edu_level', 'mean_working']

print("=== Mean stress_score by Categorical Features ===")
for col in cat_cols:
    grouped = train.groupby(col)['stress_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(f"\n--- {col} ---")
    print(grouped)

# Label encoding for ML
df_encoded = train.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

features_cat = cat_cols
features_all = [c for c in train.columns if c not in ['ID', 'stress_score']]

# Let's encode remaining categories in all features
for col in features_all:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Check R2 and MAE on training set using Decision Tree
print("\n=== Decision Tree on All Features (Train Set Fit) ===")
X_all = df_encoded[features_all]
y = df_encoded['stress_score']

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_all, y)
y_pred_dt = dt.predict(X_all)
print(f"Decision Tree Train R² : {r2_score(y, y_pred_dt):.6f}")
print(f"Decision Tree Train MAE: {mean_absolute_error(y, y_pred_dt):.6f}")

print("\n=== Random Forest (5-Fold CV / OOF) ===")
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_oof = np.zeros(len(train))
for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    X_tr, y_tr = X_all.iloc[train_idx], y.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_oof[val_idx] = rf.predict(X_val)

print(f"RF OOF R² : {r2_score(y, rf_oof):.6f}")
print(f"RF OOF MAE: {mean_absolute_error(y, rf_oof):.6f}")

# Feature importances of Random Forest trained on all data
rf_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_full.fit(X_all, y)
importances = pd.DataFrame({
    'feature': features_all,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)
print("\n=== Random Forest Feature Importances ===")
print(importances)
