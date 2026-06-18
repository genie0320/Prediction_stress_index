import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

# 8 base numerical features
base_features = [
    "age", "height", "weight", "cholesterol", 
    "systolic_blood_pressure", "diastolic_blood_pressure", 
    "glucose", "bone_density"
]

# All features (with simple imputing and encoding)
train_encoded = train.copy()
train_encoded['medical_history'] = train_encoded['medical_history'].fillna('none')
train_encoded['family_medical_history'] = train_encoded['family_medical_history'].fillna('none')
train_encoded['edu_level'] = train_encoded['edu_level'].fillna('Unknown')
train_encoded['mean_working'] = train_encoded['mean_working'].fillna(-1)

all_features = [c for c in train.columns if c not in ['ID', 'stress_score']]

for col in all_features:
    if train_encoded[col].dtype == 'object':
        le = LabelEncoder()
        train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))

y = train['stress_score'].values

# KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model_class, model_params, X, y, name):
    oof = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        
        # Scaling for SVM/MLP/Ridge
        if name in ['SVR', 'MLP', 'Ridge']:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            
        model = model_class(**model_params)
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
    
    mae = mean_absolute_error(y, oof)
    return mae

models = [
    ("ExtraTrees", ExtraTreesRegressor, {"n_estimators": 300, "random_state": 42, "n_jobs": -1}),
    ("RandomForest", RandomForestRegressor, {"n_estimators": 300, "random_state": 42, "n_jobs": -1}),
    ("LightGBM", LGBMRegressor, {"n_estimators": 300, "random_state": 42, "n_jobs": -1, "verbose": -1}),
    ("XGBoost", XGBRegressor, {"n_estimators": 300, "random_state": 42, "n_jobs": -1, "verbosity": 0}),
    ("CatBoost", CatBoostRegressor, {"iterations": 500, "random_state": 42, "verbose": 0}),
    ("SVR", SVR, {"C": 1.0, "epsilon": 0.1}),
    ("MLP", MLPRegressor, {"hidden_layer_sizes": (128, 64), "max_iter": 500, "random_state": 42}),
    ("Ridge", Ridge, {"alpha": 1.0})
]

print("=== Evaluating on 8 Base Features ===")
X_base = train[base_features].values
results_base = []
for name, model_class, params in models:
    mae = evaluate_model(model_class, params, X_base, y, name)
    print(f"{name:15} OOF MAE: {mae:.6f}")
    results_base.append((name, mae))

print("\n=== Evaluating on All Features (Label Encoded) ===")
X_all = train_encoded[all_features].values
results_all = []
for name, model_class, params in models:
    mae = evaluate_model(model_class, params, X_all, y, name)
    print(f"{name:15} OOF MAE: {mae:.6f}")
    results_all.append((name, mae))
