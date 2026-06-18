import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

base_features = [
    "age", "height", "weight", "cholesterol", 
    "systolic_blood_pressure", "diastolic_blood_pressure", 
    "glucose", "bone_density"
]
target = "stress_score"

X = train[base_features].values
y = train[target].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = [
    ("DecisionTree (no limit)", DecisionTreeRegressor(random_state=42)),
    ("DecisionTree (depth=5)", DecisionTreeRegressor(max_depth=5, random_state=42)),
    ("DecisionTree (depth=10)", DecisionTreeRegressor(max_depth=10, random_state=42)),
    ("ExtraTrees", ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ("LightGBM (extreme)", LGBMRegressor(n_estimators=2000, num_leaves=2047, max_depth=15, min_child_samples=1, random_state=42, n_jobs=-1, verbose=-1)),
    ("XGBoost (extreme)", XGBRegressor(n_estimators=2000, max_depth=15, min_child_weight=0, random_state=42, n_jobs=-1, verbosity=0)),
    ("MLP (default)", MLPRegressor(max_iter=1000, random_state=42)),
    ("MLP (large)", MLPRegressor(hidden_layer_sizes=(512, 256, 128), max_iter=2000, random_state=42, activation='relu')),
    ("SVR (RBF, C=10)", SVR(C=10.0, epsilon=0.01)),
    ("SVR (RBF, C=100)", SVR(C=100.0, epsilon=0.001))
]

print("=== Training Set Fit (MAE and R² on the SAME training data) ===")
for name, model in models:
    if "SVR" in name or "MLP" in name:
        X_fit = X_scaled
    else:
        X_fit = X
        
    model.fit(X_fit, y)
    y_pred = model.predict(X_fit)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{name:25} | Train MAE: {mae:.6f} | Train R²: {r2:.6f}")
