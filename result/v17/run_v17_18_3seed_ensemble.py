# result/v17/run_v17_18_3seed_ensemble.py
# 목적: 17-15에서 MAE를 망치던 Histogram Matching 후처리를 완벽히 제거한 순수 3-Seed 앙상블 코드

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
    print(">>> [v17-18] SVR 3-Seed 앙상블 (히스토그램 매칭 완전 제거)")
    print("="*65)

    print("1. SVR 3-Seed 훈련 및 추론 중... (약 1~2분 소요)")
    oof_pred, test_pred = run_svr(X, y, X_test, SEEDS)
    
    # ── 순수 앙상블 성능 측정 ──
    base_mae = mean_absolute_error(y, oof_pred)
    
    # ── 후처리 (정답 포맷에 맞게 0~1 클리핑 후 소수점 2자리 반올림) ──
    oof_pred_post = np.round(np.clip(oof_pred, 0, 1), 2)
    test_pred_post = np.round(np.clip(test_pred, 0, 1), 2)
    
    post_mae = mean_absolute_error(y, oof_pred_post)
    
    print(f"\n[결과] 순수 3-Seed 평균 OOF MAE: {base_mae:.5f}")
    print(f"       반올림 후 최종 OOF MAE   : {post_mae:.5f}")
    
    # 17-2 기준 (0.13581) 대비 개선폭 출력
    baseline = 0.13581
    diff = post_mae - baseline
    print(f"       => 17-2 (0.13581) 대비 개선폭: {diff:+.5f} (음수면 개선됨)")
    
    # ── 저장 ──
    os.makedirs('result/v17', exist_ok=True)
    
    submit_df = pd.DataFrame({'ID': test['ID'], 'stress_score': test_pred_post})
    save_path = 'result/v17/submit_v17-18_3seed_ensemble.csv'
    submit_df.to_csv(save_path, index=False)
    
    print("\n[Test Stats]")
    print(f"mean={test_pred_post.mean():.5f}, std={test_pred_post.std():.5f}, unique={len(np.unique(test_pred_post))}")
    print(f"최종 제출 파일 저장 완료: {save_path}")

if __name__ == '__main__':
    main()
