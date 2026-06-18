import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
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

scalers = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "None": None
}

for scaler_name, scaler in scalers.items():
    print(f"\n=== Scaler: {scaler_name} ===")
    for k in [1, 2, 3, 5, 10, 20]:
        oof = np.zeros(len(y))
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            
            if scaler is not None:
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
            else:
                X_tr_s = X_tr
                X_val_s = X_val
                
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_tr_s, y_tr)
            oof[val_idx] = model.predict(X_val_s)
            
        mae = mean_absolute_error(y, oof)
        print(f"KNN k={k} | OOF MAE: {mae:.6f}")
