import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
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

models = {
    "SVR (C=1, g=scale)": SVR(C=1.0, epsilon=0.01),
    "SVR (C=10, g=scale)": SVR(C=10.0, epsilon=0.01),
    "SVR (C=100, g=scale)": SVR(C=100.0, epsilon=0.01),
    "SVR (C=10, g=0.1)": SVR(C=10.0, gamma=0.1, epsilon=0.01),
    "SVR (C=10, g=0.01)": SVR(C=10.0, gamma=0.01, epsilon=0.01),
    
    "MLP (256,128,64 | L2=0.0001)": MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=0.0001, max_iter=1000, random_state=42),
    "MLP (256,128,64 | L2=0.01)": MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=0.01, max_iter=1000, random_state=42),
    "MLP (256,128,64 | L2=1.0)": MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=1.0, max_iter=1000, random_state=42),
    "MLP (512,256 | L2=0.01)": MLPRegressor(hidden_layer_sizes=(512, 256), alpha=0.01, max_iter=1000, random_state=42),
    "MLP (128,64 | L2=0.1)": MLPRegressor(hidden_layer_sizes=(128, 64), alpha=0.1, max_iter=1000, random_state=42)
}

print("=== Evaluating MLP and SVR on 8 Base Features (StandardScaler) ===")
for name, model in models.items():
    start_time = time.time()
    oof = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        model.fit(X_tr_s, y_tr)
        oof[val_idx] = model.predict(X_val_s)
        
    mae = mean_absolute_error(y, oof)
    print(f"{name:30} | OOF MAE: {mae:.6f} | Time: {time.time() - start_time:.1f}s")
