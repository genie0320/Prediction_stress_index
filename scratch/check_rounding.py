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

X = train[base_features].values
y = train[target].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(y))
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val = X[val_idx]
    
    # Best ExtraTrees params found
    model = ExtraTreesRegressor(n_estimators=500, criterion='absolute_error', max_features=1.0, bootstrap=False, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    oof[val_idx] = model.predict(X_val)

mae_raw = mean_absolute_error(y, oof)
mae_clip = mean_absolute_error(y, np.clip(oof, 0.0, 1.0))
mae_round2 = mean_absolute_error(y, np.round(np.clip(oof, 0.0, 1.0), 2))
mae_round1 = mean_absolute_error(y, np.round(np.clip(oof, 0.0, 1.0), 1))

print(f"OOF MAE (Raw)              : {mae_raw:.6f}")
print(f"OOF MAE (Clipped)          : {mae_clip:.6f}")
print(f"OOF MAE (Clipped & Round 2): {mae_round2:.6f}")
print(f"OOF MAE (Clipped & Round 1): {mae_round1:.6f}")
