import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
import itertools
import warnings
import time

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

X = train[base_features].values
y = train[target].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define parameter grid
param_grid = {
    "criterion": ["squared_error", "absolute_error"],
    "max_features": [1.0, 0.8, 0.6, "sqrt"],
    "bootstrap": [True, False],
    "min_samples_leaf": [1, 2]
}

keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Total experiments to run: {len(experiments)}")

results = []

for idx, params in enumerate(experiments):
    start_time = time.time()
    oof = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        
        # Use n_estimators=300 for faster grid search
        model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1, **params)
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        
    mae = mean_absolute_error(y, oof)
    elapsed = time.time() - start_time
    print(f"[{idx+1}/{len(experiments)}] Params: {params} | OOF MAE: {mae:.6f} | Time: {elapsed:.1f}s")
    
    results.append({
        "params": params,
        "OOF MAE": mae,
        "time": elapsed
    })

results_df = pd.DataFrame(results).sort_values("OOF MAE")
print("\n=== Top 10 Configurations ===")
for i in range(min(10, len(results_df))):
    row = results_df.iloc[i]
    print(f"Rank {i+1}: MAE = {row['OOF MAE']:.6f} | Params: {row['params']}")
