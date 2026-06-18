import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

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
    data['mean_working_filled'] = data['mean_working'].fillna(0)
    
    return data

def main():
    # 1. 1000회 밤샘 탐색으로 발굴된 영광의 파라미터 (v16_8 결과)
    BEST_C = 28.602211673578434
    BEST_GAMMA = 1.0566177294656107
    BEST_QUANTILES = 1365
    
    # 2. 과적합(42)을 배제한 순수 일반화 5개 시드 (N=5)
    SEEDS = [123, 456, 777, 2026, 8888]
    N_SPLITS = 10
    
    print("="*60)
    print(">>> [Experiment 16-7] SVR Multi-Seed Ensemble (50 Models)")
    print(f"    Parameters: C={BEST_C:.4f}, gamma={BEST_GAMMA:.4f}, nq={BEST_QUANTILES}")
    print(f"    Seeds: {SEEDS}")
    print("="*60 + "\n")
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    
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
    X_test = test_proc[features]
    
    final_oof = np.zeros(len(X))
    final_test_pred = np.zeros(len(X_test))
    
    # 시드별 OOF 점수를 기록할 리스트
    seed_maes = []
    
    for seed_idx, seed in enumerate(SEEDS):
        print(f"▶ [Seed {seed}] 10-Fold 학습 시작 ({seed_idx+1}/{len(SEEDS)})...")
        
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(X))
        seed_test_pred = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            # 스케일링
            scaler = RobustScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            X_te_sc = scaler.transform(X_test)
            
            # 타깃 변환 (Target Transformation)
            qt = QuantileTransformer(n_quantiles=BEST_QUANTILES, output_distribution='normal', random_state=42)
            y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
            
            # 모델 학습
            model = SVR(kernel='rbf', C=BEST_C, gamma=BEST_GAMMA, epsilon=0.0)
            model.fit(X_tr_sc, y_tr_t)
            
            # OOF 예측 및 역변환
            va_pred_t = model.predict(X_va_sc)
            va_pred = qt.inverse_transform(va_pred_t.reshape(-1, 1)).flatten()
            seed_oof[val_idx] = np.clip(va_pred, 0, 1)
            
            # Test 예측 및 역변환
            te_pred_t = model.predict(X_te_sc)
            te_pred = qt.inverse_transform(te_pred_t.reshape(-1, 1)).flatten()
            seed_test_pred += np.clip(te_pred, 0, 1) / N_SPLITS
            
        seed_mae = mean_absolute_error(y, seed_oof)
        seed_maes.append(seed_mae)
        print(f"  → Seed {seed} 10-Fold OOF MAE: {seed_mae:.6f}\n")
        
        # 전체 앙상블 배열에 누적 (나중에 시드 개수로 나눔)
        final_oof += seed_oof / len(SEEDS)
        final_test_pred += seed_test_pred / len(SEEDS)
        
    print("="*60)
    print(">>> 앙상블 완료! 최종 성과 요약")
    print(f"  각 시드별 MAE    : {[round(m, 6) for m in seed_maes]}")
    print(f"  시드별 MAE 평균  : {np.mean(seed_maes):.6f}")
    
    # 최종 OOF MAE (모든 시드 예측값들을 앙상블한 후의 점수)
    final_blended_mae_raw = mean_absolute_error(y, final_oof)
    final_blended_mae_post = mean_absolute_error(y, np.round(final_oof, 2))
    
    v16_baseline = 0.13951
    diff = final_blended_mae_post - v16_baseline
    diff_str = f"+{diff:.6f} (악화)" if diff > 0 else f"{diff:.6f} (개선)"
    
    print(f"\n  Final Blended OOF MAE (Raw)  : {final_blended_mae_raw:.6f}")
    print(f"  Final Blended OOF MAE (Post) : {final_blended_mae_post:.6f}")
    print(f"  (참고) v16 seed=42 기준값    : {v16_baseline:.5f}")
    print(f"  기준값 대비 차이             : {diff_str}")
    print("  * 주의: 이 차이는 '성능 악화'가 아니라 seed=42 분할에 대한 '과적합(거품) 제거 효과'입니다.")
    print("="*60 + "\n")
    
    # 최종 예측 파일 저장
    os.makedirs('result/v16', exist_ok=True)
    final_submit_pred = np.round(final_test_pred, 2)
    submit_df = pd.DataFrame({'ID': test['ID'], 'stress_score': final_submit_pred})
    submit_df.to_csv('result/v16/submit_v16-7_svr_multiseed.csv', index=False)
    
    print("[Test Predictions Statistics]")
    print(f"  Mean       : {final_submit_pred.mean():.5f}")
    print(f"  Std Dev    : {final_submit_pred.std():.5f}")
    print(f"  Min        : {final_submit_pred.min():.5f}")
    print(f"  Max        : {final_submit_pred.max():.5f}")
    print(f"  N Unique   : {len(np.unique(final_submit_pred))}")
    print("\nFinal submission file saved to: result/v16/submit_v16-7_svr_multiseed.csv")

if __name__ == '__main__':
    main()
