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

def run_cv(seed, train_df, y, test_df, features):
    seed_everything(seed)
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df)) if test_df is not None else None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y)):
        X_train, y_train = train_df.iloc[train_idx][features], y.iloc[train_idx]
        X_val, y_val = train_df.iloc[val_idx][features], y.iloc[val_idx]
        
        # RobustScaler was used in 17-2
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        if test_df is not None:
            X_test_scaled = scaler.transform(test_df[features])
            
        qt = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
        y_train_trans = qt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # User specified epsilon=0.0
        model = SVR(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
        model.fit(X_train_scaled, y_train_trans)
        
        val_pred_trans = model.predict(X_val_scaled)
        val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
        val_pred = np.clip(val_pred, 0, 1)
        oof_preds[val_idx] = val_pred
        
        if test_df is not None:
            test_pred_trans = model.predict(X_test_scaled)
            test_pred = qt.inverse_transform(test_pred_trans.reshape(-1, 1)).flatten()
            test_pred = np.clip(test_pred, 0, 1)
            test_preds += test_pred / 10.0
            
        if seed == 42 and test_df is not None:
            fold_mae = mean_absolute_error(y_val, val_pred)
            print(f"Fold {fold+1:02d} MAE: {fold_mae:.5f}")
            
    return oof_preds, test_preds

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    
    features = [
        'age','height','weight','cholesterol',
        'systolic_blood_pressure','diastolic_blood_pressure',
        'glucose','bone_density',
        'bmi','pp','map_val',
        'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
        'n_medical','n_family'
    ]
    
    y = train_proc['stress_score']
    
    print("[10-Fold CV seed=42]")
    oof_42, test_42 = run_cv(42, train_proc, y, test_proc, features)
    mae_42 = mean_absolute_error(y, oof_42)
    print(f"전체 OOF MAE (seed=42): {mae_42:.6f}")
    
    print("\n[3-Seed 평가]")
    seed_maes = [mae_42]
    
    for s in [123, 456]:
        oof_s, _ = run_cv(s, train_proc, y, None, features)
        mae_s = mean_absolute_error(y, oof_s)
        print(f"seed={s} MAE: {mae_s:.6f}")
        seed_maes.append(mae_s)
        
    print(f"3-Seed 평균 MAE: {np.mean(seed_maes):.6f}")
    print(f"3-Seed Std: {np.std(seed_maes):.6f}")
    
    print("\n[Test 예측값 통계 (seed=42)]")
    print(f"Mean       : {test_42.mean():.6f}")
    print(f"Std Dev    : {test_42.std():.6f}")
    print(f"Min        : {test_42.min():.6f}")
    print(f"Max        : {test_42.max():.6f}")
    print(f"N_unique   : {len(np.unique(np.round(test_42, 2)))}")
    
    os.makedirs('result/v17', exist_ok=True)
    pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_42
    }).to_csv('result/v17/submit_v17-12_epsilon.csv', index=False)

if __name__ == '__main__':
    main()
