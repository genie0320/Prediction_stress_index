# result/v17/run_v17_15_histogram_matching.py
# 목적: 17-12 SVR 예측값에 대해 True Histogram Matching 적용.
# 예측값의 순서(Rank)를 기반으로 Train의 실제 정답값(y) 분포와 1:1 매칭.

import os, random, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
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
    return len([i.strip() for i in str(x).split(',') if i.strip()])

def preprocess_data(df):
    data = df.copy()
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    data['activity_enc'] = data['activity'].map({'light':1,'moderate':2,'intense':3}).fillna(0).astype(int)
    data['edu_enc'] = data['edu_level'].map({'high school diploma':1,'bachelors degree':2,'graduate degree':3}).fillna(0).astype(int)
    data['sleep_enc'] = data['sleep_pattern'].map({'sleep difficulty':1,'normal':2,'oversleeping':3}).fillna(0).astype(int)
    data['gender_enc'] = data['gender'].map({'F':0,'M':1}).fillna(0).astype(int)
    data['smoke_enc'] = data['smoke_status'].map({'non-smoker':0,'ex-smoker':1,'current-smoker':2}).fillna(0).astype(int)
    data['n_medical'] = data['medical_history'].apply(count_items)
    data['n_family'] = data['family_medical_history'].apply(count_items)
    return data

FEATURES = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density',
    'bmi','pp','map_val',
    'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
    'n_medical','n_family'
]

SVR_PARAMS = {
    'kernel': 'rbf',
    'C': 28.6022,
    'gamma': 1.0566,
    'epsilon': 0.0
}
SEEDS = [42, 123, 456]
N_SPLITS = 10
NQ = 1365

def run_svr(X, y, X_test, seeds):
    final_oof  = np.zeros(len(X))
    final_test = np.zeros(len(X_test))
    
    for seed in seeds:
        seed_everything(seed)
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof  = np.zeros(len(X))
        seed_test = np.zeros(len(X_test))
        
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va       = X.iloc[va_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_test)
            
            qt = QuantileTransformer(n_quantiles=NQ, output_distribution='normal', random_state=42)
            y_tr_t = qt.fit_transform(y_tr.values.reshape(-1,1)).flatten()
            
            model = SVR(**SVR_PARAMS)
            model.fit(X_tr_s, y_tr_t)
            
            va_pred = qt.inverse_transform(model.predict(X_va_s).reshape(-1,1)).flatten()
            te_pred = qt.inverse_transform(model.predict(X_te_s).reshape(-1,1)).flatten()
            
            seed_oof[va_idx] = np.clip(va_pred, 0, 1)
            seed_test += np.clip(te_pred, 0, 1) / N_SPLITS
            
        final_oof  += seed_oof  / len(seeds)
        final_test += seed_test / len(seeds)
        
    return final_oof, final_test

def main():
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    train_p = preprocess_data(train)
    test_p  = preprocess_data(test)

    X      = train_p[FEATURES]
    y      = train_p['stress_score']
    X_test = test_p[FEATURES]

    print("="*65)
    print(">>> [v17-15] SVR 3-Seed 예측 후 True Histogram Matching 적용")
    print("="*65)

    print("1. SVR 기본 훈련 및 추론 중... (약 1~2분 소요)")
    oof_pred, test_pred = run_svr(X, y, X_test, SEEDS)
    
    base_mae = mean_absolute_error(y, oof_pred)
    print(f"\n[Before] 변환 전 Blended OOF MAE: {base_mae:.6f}")
    print(f"         예측값 std: {oof_pred.std():.5f} (타깃 std: {y.std():.5f})")
    
    # ── True Histogram Matching ──
    print("\n2. True Histogram Matching 적용 중...")
    y_sorted = np.sort(y.values)
    
    # 1) OOF 매칭 (1:1 매칭)
    oof_sort_idx = np.argsort(oof_pred)
    oof_matched = np.zeros_like(oof_pred)
    oof_matched[oof_sort_idx] = y_sorted
    
    # 2) Test 매칭 (랭크 비례 보간 매칭)
    test_ranks = np.argsort(np.argsort(test_pred))
    mapped_indices = test_ranks / (len(test_pred) - 1) * (len(y_sorted) - 1)
    test_matched = np.interp(mapped_indices, np.arange(len(y_sorted)), y_sorted)
    
    matched_mae = mean_absolute_error(y, oof_matched)
    print(f"\n[After] 변환 후 Histogram Matched OOF MAE: {matched_mae:.6f}")
    print(f"        예측값 std: {oof_matched.std():.5f} (타깃 std와 동일: {y_sorted.std():.5f})")
    
    diff = matched_mae - base_mae
    if diff > 0:
        print(f"    => 결과: {diff:+.6f} (악화) ❌")
    else:
        print(f"    => 결과: {diff:+.6f} (개선) ✅")
    
    # ── 저장 ──
    os.makedirs('result/v17', exist_ok=True)
    
    submit_pred = np.round(np.clip(test_matched, 0, 1), 2)
    submit_df = pd.DataFrame({'ID': test['ID'], 'stress_score': submit_pred})
    submit_df.to_csv('result/v17/submit_v17-15_hist_match.csv', index=False)
    
    print("\n[Test Stats]")
    print(f"mean={submit_pred.mean():.5f}, std={submit_pred.std():.5f}, unique={len(np.unique(submit_pred))}")
    print("저장 완료: result/v17/submit_v17-15_hist_match.csv")

if __name__ == '__main__':
    main()
