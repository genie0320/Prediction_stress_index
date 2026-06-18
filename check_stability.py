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

def main():
    train = pd.read_csv('train.csv')
    train_proc = preprocess_data(train)
    
    features = [
        'age','height','weight','cholesterol',
        'systolic_blood_pressure','diastolic_blood_pressure',
        'glucose','bone_density',
        'bmi','pp','map_val',
        'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
        'n_medical','n_family'
    ]
    
    X = train_proc[features]
    y = train_proc['stress_score']
    
    seeds = [42, 777, 2026]
    maes = []
    
    for s in seeds:
        seed_everything(s)
        kf = KFold(n_splits=10, shuffle=True, random_state=s)
        oof_preds = np.zeros(len(train))
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            qt = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=s)
            y_train_trans = qt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            
            model = SVR(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
            model.fit(X_train_scaled, y_train_trans)
            
            val_pred_trans = model.predict(X_val_scaled)
            val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
            oof_preds[val_idx] = val_pred
            
        oof_mae_raw = mean_absolute_error(y, oof_preds)
        maes.append(oof_mae_raw)
        print(f"seed={s:<4d} -> OOF MAE: {oof_mae_raw:.6f}")
        
    print(f"평균 MAE:   {np.mean(maes):.6f}")
    print(f"Std:        {np.std(maes):.6f}")

if __name__ == '__main__':
    main()
