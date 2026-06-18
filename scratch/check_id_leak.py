import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
test_path = "/Users/admin/Documents/dev_src/stress_index/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Extract numeric ID
train['id_num'] = train['ID'].apply(lambda x: int(x.split('_')[1]))
test['id_num'] = test['ID'].apply(lambda x: int(x.split('_')[1]))

y = train['stress_score'].values

# Check correlation of target with id_num
corr, p = pearsonr(train['id_num'], y)
print(f"Correlation of ID number with stress_score: {corr:.6f} (p-value: {p:.8f})")

# Check if features are correlated with ID
base_features = [
    "age", "height", "weight", "cholesterol", 
    "systolic_blood_pressure", "diastolic_blood_pressure", 
    "glucose", "bone_density"
]

print("\n=== Correlation of ID number with features ===")
for col in base_features:
    corr_tr, p_tr = pearsonr(train['id_num'], train[col])
    corr_te, p_te = pearsonr(test['id_num'], test[col])
    print(f"{col:25} | Train Corr: {corr_tr: .6f} (p: {p_tr:.4f}) | Test Corr: {corr_te: .6f} (p: {p_te:.4f})")

# Let's check rolling mean of stress_score over ID number
rolling_mean = train.sort_values('id_num')['stress_score'].rolling(window=100, center=True).mean()
print("\nRolling mean of target over sorted ID (first 10 non-nan):")
print(rolling_mean.dropna().head(10))
print("Rolling mean std:", rolling_mean.std())
