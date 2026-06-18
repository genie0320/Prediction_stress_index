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

def preprocess_vac(df):
    data = df.copy()
    
    # 1. Base derived features
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    
    # 2. Basic Encodings
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
    
    # 3. mean_working_filled
    data['mean_working_filled'] = data['mean_working'].fillna(0)
    
    # 4. medical_history One-hot
    medical_cats = ['heart disease', 'diabetes', 'high blood pressure', 'None']
    for cat in medical_cats:
        data[f'med_{cat.replace(" ", "_")}'] = data['medical_history'].apply(
            lambda x: 1 if isinstance(x, str) and cat in x else 0
        )
        
    family_cats = ['heart disease', 'diabetes', 'high blood pressure', 'None']
    for cat in family_cats:
        data[f'fam_{cat.replace(" ", "_")}'] = data['family_medical_history'].apply(
            lambda x: 1 if isinstance(x, str) and cat in x else 0
        )
        
    # 5. smoke_status One-hot
    smoke_dummies = pd.get_dummies(data['smoke_status'], prefix='smoke').astype(int)
    data = pd.concat([data, smoke_dummies], axis=1)
    
    return data

def run_cv(seed, train_df, y, test_df, features):
    seed_everything(seed)
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    oof_pred = np.zeros(len(train_df))
    test_pred_sum = np.zeros(len(test_df)) if test_df is not None else None
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
        X_tr, y_tr = train_df.iloc[tr_idx][features], y.iloc[tr_idx]
        X_va, y_va = train_df.iloc[va_idx][features], y.iloc[va_idx]
        
        # 1. Scaler: RobustScaler
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        if test_df is not None:
            X_te_s = scaler.transform(test_df[features])
            
        # 2. Target Transform
        nq = min(1000, len(y_tr))
        qt = QuantileTransformer(n_quantiles=nq, output_distribution='normal', random_state=42)
        y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        # 3. SVR Params
        svr = SVR(
            kernel='rbf',
            C=3.96,
            gamma=1.06,
            epsilon=0.0,
            shrinking=True,
            cache_size=500,
            max_iter=-1
        )
        svr.fit(X_tr_s, y_tr_trans)
        
        val_pred = qt.inverse_transform(svr.predict(X_va_s).reshape(-1, 1)).flatten()
        val_pred = np.clip(val_pred, 0, 1)
        oof_pred[va_idx] = val_pred
        
        if test_df is not None:
            test_pred = qt.inverse_transform(svr.predict(X_te_s).reshape(-1, 1)).flatten()
            test_pred = np.clip(test_pred, 0, 1)
            test_pred_sum += test_pred / 10.0
            
        if seed == 42 and test_df is not None:
            fold_mae = mean_absolute_error(y_va, val_pred)
            print(f"Fold {fold+1:02d} MAE: {fold_mae:.5f}")
            
    return oof_pred, test_pred_sum

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_vac(train)
    test_proc = preprocess_vac(test)
    y = train_proc['stress_score']
    
    # Feature list
    base_feats = [
        'age', 'height', 'weight', 'cholesterol',
        'systolic_blood_pressure', 'diastolic_blood_pressure',
        'glucose', 'bone_density',
        'bmi', 'pp', 'map_val',
        'activity_enc', 'edu_enc', 'sleep_enc', 'gender_enc', 'smoke_enc',
        'mean_working_filled'
    ]
    
    med_feats = [f for f in train_proc.columns if f.startswith('med_')]
    fam_feats = [f for f in train_proc.columns if f.startswith('fam_')]
    smoke_feats = [f for f in train_proc.columns if f.startswith('smoke_') and f not in ['smoke_enc', 'smoke_status']]
    
    features = base_feats + med_feats + fam_feats + smoke_feats
    
    # Make sure test has same columns
    for col in features:
        if col not in test_proc.columns:
            test_proc[col] = 0
            
    print(f"최종 피처 수: {len(features)}개")
    print(f"피처 목록: {features}")
    
    print("\n[10-Fold CV seed=42]")
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
    }).to_csv('result/v17/submit_v17-11_vac.csv', index=False)

if __name__ == '__main__':
    main()
