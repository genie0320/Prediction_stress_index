import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

base_features = [
    "age", "height", "weight", "cholesterol", 
    "systolic_blood_pressure", "diastolic_blood_pressure", 
    "glucose", "bone_density"
]
target = "stress_score"

X_raw = train[base_features].copy()
y = train[target].values

# Generate features
X_trans = pd.DataFrame()

# 1. Unary transformations
for col in base_features:
    x = X_raw[col]
    X_trans[f"{col}"] = x
    X_trans[f"{col}_sq"] = x ** 2
    X_trans[f"{col}_cub"] = x ** 3
    X_trans[f"{col}_sqrt"] = np.sqrt(np.abs(x) + 1e-5)
    X_trans[f"{col}_log"] = np.log(np.abs(x) + 1e-5)
    X_trans[f"{col}_inv"] = 1.0 / (x + 1e-5)
    X_trans[f"{col}_sin"] = np.sin(x)
    X_trans[f"{col}_cos"] = np.cos(x)

# 2. Binary transformations
for i in range(len(base_features)):
    for j in range(i, len(base_features)):
        col1 = base_features[i]
        col2 = base_features[j]
        x1 = X_raw[col1]
        x2 = X_raw[col2]
        
        # Product
        X_trans[f"{col1}_x_{col2}"] = x1 * x2
        
        # Ratios
        if i != j:
            X_trans[f"{col1}_div_{col2}"] = x1 / (x2 + 1e-5)
            X_trans[f"{col2}_div_{col1}"] = x2 / (x1 + 1e-5)
            X_trans[f"abs_diff_{col1}_{col2}"] = np.abs(x1 - x2)
            X_trans[f"sum_sq_{col1}_{col2}"] = (x1 + x2) ** 2
            X_trans[f"diff_sq_{col1}_{col2}"] = (x1 - x2) ** 2

print(f"Generated {X_trans.shape[1]} non-linear features.")

# Handle NaNs/Infs if any
X_trans = X_trans.replace([np.inf, -np.inf], np.nan)
X_trans = X_trans.fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_trans)

# Fit LassoCV
print("\nFitting LassoCV...")
lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=10000)
lasso.fit(X_scaled, y)
y_pred_lasso = lasso.predict(X_scaled)
print(f"Lasso Train R²: {r2_score(y, y_pred_lasso):.6f}")
print(f"Lasso Train MAE: {mean_absolute_error(y, y_pred_lasso):.6f}")
print(f"Lasso Selected Alpha: {lasso.alpha_:.6f}")

# Fit RidgeCV
print("\nFitting RidgeCV...")
ridge = RidgeCV(cv=5)
ridge.fit(X_scaled, y)
y_pred_ridge = ridge.predict(X_scaled)
print(f"Ridge Train R²: {r2_score(y, y_pred_ridge):.6f}")
print(f"Ridge Train MAE: {mean_absolute_error(y, y_pred_ridge):.6f}")

# Look at Lasso coefficients
coefs = pd.DataFrame({
    'feature': X_trans.columns,
    'coef': lasso.coef_,
    'abs_coef': np.abs(lasso.coef_)
}).sort_values('abs_coef', ascending=False)

print("\n=== Top 20 Lasso Coefficients ===")
print(coefs[coefs['coef'] != 0].head(20))
