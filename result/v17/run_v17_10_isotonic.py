import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.isotonic import IsotonicRegression
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

def run_cv_isotonic(seed, train_df, y, test_df, features):
    seed_everything(seed)
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    oof_raw = np.zeros(len(train_df))
    oof_cal = np.zeros(len(train_df))
    test_cal_sum = np.zeros(len(test_df)) if test_df is not None else None
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
        X_tr, y_tr = train_df.iloc[tr_idx][features], y.iloc[tr_idx]
        X_va, y_va = train_df.iloc[va_idx][features], y.iloc[va_idx]
        
        # Scale
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        if test_df is not None:
            X_te_s = scaler.transform(test_df[features])
            
        # Target transform
        qt = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
        y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        # SVR model
        svr = SVR(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
        svr.fit(X_tr_s, y_tr_trans)
        
        # Val prediction raw
        svr_val_pred = qt.inverse_transform(svr.predict(X_va_s).reshape(-1, 1)).flatten()
        svr_val_pred = np.clip(svr_val_pred, 0, 1)
        oof_raw[va_idx] = svr_val_pred
        
        # Train prediction for Isotonic Regression
        svr_train_pred = qt.inverse_transform(svr.predict(X_tr_s).reshape(-1, 1)).flatten()
        svr_train_pred = np.clip(svr_train_pred, 0, 1)
        
        # Fit Isotonic Regression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(svr_train_pred, y_tr)
        
        # Val prediction calibrated
        svr_val_cal = iso.predict(svr_val_pred)
        svr_val_cal = np.clip(svr_val_cal, 0, 1)
        oof_cal[va_idx] = svr_val_cal
        
        # Test prediction
        if test_df is not None:
            svr_test_pred = qt.inverse_transform(svr.predict(X_te_s).reshape(-1, 1)).flatten()
            svr_test_pred = np.clip(svr_test_pred, 0, 1)
            svr_test_cal = iso.predict(svr_test_pred)
            svr_test_cal = np.clip(svr_test_cal, 0, 1)
            test_cal_sum += svr_test_cal / 10.0
            
        if seed == 42 and test_df is not None:
            fold_raw_mae = mean_absolute_error(y_va, svr_val_pred)
            fold_cal_mae = mean_absolute_error(y_va, svr_val_cal)
            print(f"Fold {fold+1:02d} | 보정전: {fold_raw_mae:.5f} | 보정후: {fold_cal_mae:.5f}")
            
    return oof_raw, oof_cal, test_cal_sum

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    y = train_proc['stress_score']
    
    features = [
        'age','height','weight','cholesterol',
        'systolic_blood_pressure','diastolic_blood_pressure',
        'glucose','bone_density',
        'bmi','pp','map_val',
        'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
        'n_medical','n_family'
    ]
    
    print("[10-Fold CV (SVR + Isotonic)]")
    oof_raw, oof_cal, test_cal = run_cv_isotonic(42, train_proc, y, test_proc, features)
    
    mae_raw = mean_absolute_error(y, oof_raw)
    mae_cal = mean_absolute_error(y, oof_cal)
    
    baseline_17_2 = 0.13578
    improvement = baseline_17_2 - mae_cal
    imp_str = "개선" if improvement > 0 else "악화"
    
    print("\n[OOF MAE 비교]")
    print(f"보정 전 OOF MAE : {mae_raw:.6f}")
    print(f"보정 후 OOF MAE : {mae_cal:.6f}")
    print(f"17-2 기준값      : {baseline_17_2}")
    print(f"개선폭           : {improvement:+.6f} ({imp_str})")
    
    print("\n[예측값 분포 변화]")
    print(f"보정 전 | Mean: {oof_raw.mean():.3f}  Std: {oof_raw.std():.3f}  N_unique: {len(np.unique(np.round(oof_raw, 5)))}")
    print(f"보정 후 | Mean: {oof_cal.mean():.3f}  Std: {oof_cal.std():.3f}  N_unique: {len(np.unique(np.round(oof_cal, 5)))}")
    print(f"실제 타깃| Mean: {y.mean():.3f}  Std: {y.std():.3f}  N_unique: {y.nunique()}")
    
    print("\n[3-Seed 안정성]")
    seeds = [42, 777, 2026]
    seed_maes = []
    
    # 42 is already run, but we just use the cal MAE
    seed_maes.append(mae_cal)
    print(f"seed=42   → MAE: {mae_cal:.6f}")
    
    for s in [777, 2026]:
        _, oof_cal_s, _ = run_cv_isotonic(s, train_proc, y, None, features)
        mae_s = mean_absolute_error(y, oof_cal_s)
        print(f"seed={s:<4} → MAE: {mae_s:.6f}")
        seed_maes.append(mae_s)
        
    mean_mae = np.mean(seed_maes)
    std_mae = np.std(seed_maes)
    print(f"평균 MAE:   {mean_mae:.6f}")
    print(f"Std:        {std_mae:.6f}")
    
    if std_mae > 0.003:
        print("[Warning] 시드 간 편차가 큽니다. 불안정한 보정일 수 있습니다.")
    else:
        print("[Safe] 시드 간 성능 편차가 안정적입니다.")
        
    print("\n[Test 예측값 통계]")
    # Post processing rule usually limits to 0~1 and round(2) for this competition?
    # Actually the instruction says: Mean / Std / Min / Max / N_unique
    # The isotonic test prediction is test_cal
    test_cal_post = np.round(np.clip(test_cal, 0.0, 1.0), 2)
    print(f"Mean       : {test_cal_post.mean():.6f}")
    print(f"Std Dev    : {test_cal_post.std():.6f}")
    print(f"Min        : {test_cal_post.min():.6f}")
    print(f"Max        : {test_cal_post.max():.6f}")
    print(f"N_unique   : {len(np.unique(test_cal_post))}")
    
    os.makedirs('result/v17', exist_ok=True)
    submit_path = 'result/v17/submit_v17-10_isotonic.csv'
    pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_cal_post
    }).to_csv(submit_path, index=False)

if __name__ == "__main__":
    main()
