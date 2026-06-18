import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def count_items(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    items = [item.strip() for item in str(x).split(',') if item.strip()]
    return len(items)

def preprocess_data(df):
    data = df.copy()
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    data['mean_working_filled'] = data['mean_working'].fillna(0)
    data['age_bmi'] = data['age'] * data['bmi']
    data['gluc_chol'] = data['glucose'] / data['cholesterol']
    
    activity_map = {'light': 1, 'moderate': 2, 'intense': 3}
    data['activity_enc'] = data['activity'].map(activity_map).fillna(0).astype(int)
    edu_map = {'high school diploma': 1, 'bachelors degree': 2, 'graduate degree': 3}
    data['edu_enc'] = data['edu_level'].map(edu_map).fillna(0).astype(int)
    sleep_map = {'sleep difficulty': 1, 'normal': 2, 'oversleeping': 3}
    data['sleep_enc'] = data['sleep_pattern'].map(sleep_map).fillna(0).astype(int)
    gender_map = {'F': 0, 'M': 1}
    data['gender_enc'] = data['gender'].map(gender_map).fillna(0).astype(int)
    smoke_map = {'non-smoker': 0, 'ex-smoker': 1, 'current-smoker': 2}
    data['smoke_enc'] = data['smoke_status'].map(smoke_map).fillna(0).astype(int)
    
    data['n_medical'] = data['medical_history'].apply(count_items)
    data['n_family'] = data['family_medical_history'].apply(count_items)
    return data

def evaluate_params(C, gamma, nq, X, y):
    seed_everything(42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        qt = QuantileTransformer(n_quantiles=int(nq), output_distribution='normal', random_state=42)
        y_train_trans = qt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=0.0)
        model.fit(X_train_scaled, y_train_trans)
        
        val_pred_trans = model.predict(X_val_scaled)
        val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
        oof_preds[val_idx] = val_pred
        
    return mean_absolute_error(y, oof_preds)

def main():
    train = pd.read_csv('train.csv')
    train_proc = preprocess_data(train)
    
    features = [
        'age', 'height', 'weight', 'cholesterol',
        'systolic_blood_pressure', 'diastolic_blood_pressure',
        'glucose', 'bone_density',
        'bmi', 'pp', 'map_val', 'mean_working_filled',
        'activity_enc', 'edu_enc', 'sleep_enc', 'gender_enc', 'smoke_enc',
        'n_medical', 'n_family'
    ]
    
    X = train_proc[features]
    y = train_proc['stress_score']
    
    # 1. v16 (C=3.56, gamma=0.83, nq=983)
    mae_v16 = evaluate_params(C=3.56, gamma=0.83, nq=983, X=X, y=y)
    print(f"v16 파라미터 → 10-Fold OOF MAE: {mae_v16:.6f}")
    
    # 2. 신규 파라미터 (C=28.60, gamma=1.05, nq=1365)
    mae_new = evaluate_params(C=28.60, gamma=1.05, nq=1365, X=X, y=y)
    print(f"신규 파라미터 → 10-Fold OOF MAE: {mae_new:.6f}")
    
if __name__ == '__main__':
    main()
