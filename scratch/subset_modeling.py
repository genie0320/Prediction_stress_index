import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
import warnings

warnings.filterwarnings("ignore")

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

base_features = [
    "age", "height", "weight", "cholesterol", 
    "systolic_blood_pressure", "diastolic_blood_pressure", 
    "glucose", "bone_density"
]
target = "stress_score"

# 1. Base Global Model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_global = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    X_tr = train.iloc[train_idx][base_features].values
    y_tr = train.iloc[train_idx][target].values
    X_val = train.iloc[val_idx][base_features].values
    
    model = ExtraTreesRegressor(n_estimators=300, criterion='absolute_error', max_features=1.0, bootstrap=False, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    oof_global[val_idx] = model.predict(X_val)

mae_global = mean_absolute_error(train[target], oof_global)
print(f"Global OOF MAE: {mae_global:.6f}")

def evaluate_split(split_col):
    oof_split = np.zeros(len(train))
    unique_vals = train[split_col].unique()
    
    # We will do KFold cross-validation, but within each fold, we fit separate models for each category
    for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]
        
        for val in unique_vals:
            # Subsets
            tr_sub = train_fold[train_fold[split_col] == val]
            val_sub = val_fold[val_fold[split_col] == val]
            
            if len(val_sub) == 0:
                continue
            if len(tr_sub) == 0:
                # Fallback to global model if category not in training fold
                model = ExtraTreesRegressor(n_estimators=300, criterion='absolute_error', max_features=1.0, bootstrap=False, min_samples_leaf=1, random_state=42, n_jobs=-1)
                model.fit(train_fold[base_features].values, train_fold[target].values)
                pred = model.predict(val_sub[base_features].values)
            else:
                model = ExtraTreesRegressor(n_estimators=300, criterion='absolute_error', max_features=1.0, bootstrap=False, min_samples_leaf=1, random_state=42, n_jobs=-1)
                model.fit(tr_sub[base_features].values, tr_sub[target].values)
                pred = model.predict(val_sub[base_features].values)
                
            oof_split[val_sub.index] = pred
            
    mae = mean_absolute_error(train[target], oof_split)
    print(f"OOF MAE Split by '{split_col}': {mae:.6f} (Change: {mae - mae_global:+.6f})")

# Test splits
evaluate_split("gender")
evaluate_split("sleep_pattern")
evaluate_split("activity")
evaluate_split("smoke_status")

# Test combined split (gender + sleep_pattern)
train['gender_sleep'] = train['gender'].astype(str) + "_" + train['sleep_pattern'].astype(str)
evaluate_split("gender_sleep")
