# result/v17/run_v17_17_pseudo_labeling.py
# 목적: SVR의 강력한 예측값(17-12)을 Test셋의 가짜 정답(Pseudo Labels)으로 활용하여 
# Train 데이터 크기를 3000개 -> 6000개로 늘린 뒤 재학습 (Semi-Supervised Learning)

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

def run_pseudo_svr(X, y, X_test, pseudo_y, seeds):
    final_oof  = np.zeros(len(X))
    final_test = np.zeros(len(X_test))
    
    for seed in seeds:
        seed_everything(seed)
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof  = np.zeros(len(X))
        seed_test = np.zeros(len(X_test))
        
        for tr_idx, va_idx in kf.split(X):
            X_tr_orig, y_tr_orig = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va       = X.iloc[va_idx]
            
            # ── 핵심: Pseudo-Labeling (Test 데이터를 Train에 포함) ──
            X_tr_combined = pd.concat([X_tr_orig, X_test], axis=0).reset_index(drop=True)
            y_tr_combined = pd.concat([y_tr_orig, pseudo_y], axis=0).reset_index(drop=True)
            
            # 피처 스케일링은 풍부해진 combined 데이터로 학습
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_combined)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_test)
            
            # 타깃 변환기(Quantile)는 오직 진짜 정답(y_tr_orig) 분포로만 fit
            qt = QuantileTransformer(n_quantiles=NQ, output_distribution='normal', random_state=42)
            qt.fit(y_tr_orig.values.reshape(-1,1))
            y_tr_t = qt.transform(y_tr_combined.values.reshape(-1,1)).flatten()
            
            # 모델 학습 (6000개 데이터)
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

    # 가짜 정답(Pseudo Labels) 불러오기 (가장 성능이 좋았던 17-12 SVR 예측값)
    print("가짜 정답 로드 중: result/v17/submit_v17-12_epsilon.csv")
    pseudo_df = pd.read_csv('result/v17/submit_v17-12_epsilon.csv')
    pseudo_y = pseudo_df['stress_score']

    print("="*65)
    print(">>> [v17-17] Pseudo-Labeling + SVR (Train N=6000)")
    print("="*65)

    print("1. 6000개 데이터 기반 SVR 3-Seed 훈련 중... (약 2~4분 소요)")
    oof_pred, test_pred = run_pseudo_svr(X, y, X_test, pseudo_y, SEEDS)
    
    mae = mean_absolute_error(y, oof_pred)
    print(f"\n[Result] Blended OOF MAE (True Valid Set만 계산): {mae:.6f}")
    
    v172_baseline = 0.134652
    diff = mae - v172_baseline
    if diff > 0:
        print(f"    => 베이스라인 대비: {diff:+.6f} (악화) ❌")
    else:
        print(f"    => 베이스라인 대비: {diff:+.6f} (개선) ✅")
    
    # ── 저장 ──
    os.makedirs('result/v17', exist_ok=True)
    
    submit_pred = np.round(np.clip(test_pred, 0, 1), 2)
    submit_df = pd.DataFrame({'ID': test['ID'], 'stress_score': submit_pred})
    submit_df.to_csv('result/v17/submit_v17-17_pseudo.csv', index=False)
    
    print("\n[Test Stats]")
    print(f"mean={submit_pred.mean():.5f}, std={submit_pred.std():.5f}, unique={len(np.unique(submit_pred))}")
    print("저장 완료: result/v17/submit_v17-17_pseudo.csv")

if __name__ == '__main__':
    main()
