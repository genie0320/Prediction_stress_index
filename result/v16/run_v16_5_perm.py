import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

def count_items(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    items = [item.strip() for item in str(x).split(',') if item.strip()]
    return len(items)

def preprocess_data(df):
    data = df.copy()
    
    # 파생 피처
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    
    # 범주형 인코딩
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
    
    # 텍스트 카운트
    data['n_medical'] = data['medical_history'].apply(count_items)
    data['n_family'] = data['family_medical_history'].apply(count_items)
    
    # 기본 결측치 임퓨테이션
    data['mean_working_filled'] = data['mean_working'].fillna(0)
    
    return data

def build_pipeline():
    # v16 베이스라인 파라미터 적용
    model = SVR(kernel='rbf', C=3.5579, gamma=0.8347, epsilon=0.0)
    qt = QuantileTransformer(n_quantiles=983, output_distribution='normal', random_state=42)
    ttr = TransformedTargetRegressor(regressor=model, transformer=qt)
    
    pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', ttr)
    ])
    return pipe

def evaluate_5fold(features, X_full, y_full):
    X = X_full[features]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X, y_full):
        X_tr, y_tr = X.iloc[train_idx], y_full.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y_full.iloc[val_idx]
        
        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        oof_preds[val_idx] = pipe.predict(X_va)
        
    return mean_absolute_error(y_full, oof_preds)

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    
    features_all = [
        'age', 'height', 'weight', 'cholesterol',
        'systolic_blood_pressure', 'diastolic_blood_pressure',
        'glucose', 'bone_density',
        'bmi', 'pp', 'map_val', 'mean_working_filled',
        'activity_enc', 'edu_enc', 'sleep_enc', 'gender_enc', 'smoke_enc',
        'n_medical', 'n_family'
    ]
    
    X = train_proc[features_all]
    y = train_proc['stress_score']
    X_test = test_proc[features_all]
    
    print(">>> 19개 피처 셋팅 및 v16 SVR Pipeline 구축 완료")
    
    # ----------------------------------------------------------------
    # Step 1: Permutation Importance
    # ----------------------------------------------------------------
    print("\n[Step 1: Permutation Importance 계산 중 (전체 Train 학습)...]")
    pipe = build_pipeline()
    pipe.fit(X, y)
    
    result = permutation_importance(
        pipe, X, y, 
        n_repeats=10, 
        random_state=42, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    # 양수일수록 중요, 음수일수록 노이즈
    feature_imp = pd.DataFrame({
        'feature': features_all,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n[Step 1: Permutation Importance 결과]")
    print(f"{'피처명':<30} {'importance_mean':>15}   {'importance_std':>15}   {'판정'}")
    
    for _, row in feature_imp.iterrows():
        f = row['feature']
        m = row['importance_mean']
        s = row['importance_std']
        status = "유효" if m > 0 else "제거후보"
        print(f"{f:<30} {m:15.6f}   {s:15.5f}   {status}")
        
    # ----------------------------------------------------------------
    # Step 2: 피처셋별 OOF MAE (5-Fold)
    # ----------------------------------------------------------------
    feat_A = features_all
    feat_B = ['age', 'height', 'weight', 'cholesterol', 'systolic_blood_pressure', 
              'diastolic_blood_pressure', 'glucose', 'bone_density', 
              'mean_working_filled', 'n_medical', 'n_family']
    feat_C = feature_imp[feature_imp['importance_mean'] > 0.0005]['feature'].tolist()
    feat_D = feature_imp[feature_imp['importance_mean'] > 0]['feature'].tolist()
    
    combinations = [
        ('A', feat_A),
        ('B', feat_B),
        ('C', feat_C),
        ('D', feat_D)
    ]
    
    print("\n[Step 2: 피처셋별 OOF MAE (5-Fold 빠른 비교)]")
    print(f"{'조합':<4} {'피처수':<5} {'OOF MAE':<10} {'v16 대비(0.13951)'}")
    
    best_combo_name = ''
    best_features = []
    best_mae = float('inf')
    
    for name, feats in combinations:
        if len(feats) == 0:
            print(f"[{name}]    0     Skip (피처 없음)")
            continue
            
        mae = evaluate_5fold(feats, X, y)
        diff = mae - 0.13951
        diff_str = f"+{diff:.5f}" if diff > 0 else f"{diff:.5f}"
        print(f"[{name}]    {len(feats):<5} {mae:.5f}    {diff_str}")
        
        if mae < best_mae:
            best_mae = mae
            best_combo_name = name
            best_features = feats
            
    print(f"\n>>> 최선 조합: [{best_combo_name}] (피처 {len(best_features)}개) 로 10-Fold 진행")

    # ----------------------------------------------------------------
    # Step 3: 최선 조합 10-Fold 재검증
    # ----------------------------------------------------------------
    print("\n[Step 3: 최선 조합 10-Fold 재검증]")
    kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
    X_best = X[best_features]
    X_test_best = X_test[best_features]
    
    oof_preds_10 = np.zeros(len(X_best))
    test_preds_10 = np.zeros(len(X_test_best))
    
    for fold, (train_idx, val_idx) in enumerate(kf10.split(X_best, y)):
        X_tr, y_tr = X_best.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_best.iloc[val_idx], y.iloc[val_idx]
        
        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        
        va_pred = pipe.predict(X_va)
        oof_preds_10[val_idx] = va_pred
        
        te_pred = pipe.predict(X_test_best)
        test_preds_10 += te_pred / 10.0
        
        print(f"  Fold {fold+1:02d} MAE: {mean_absolute_error(y_va, va_pred):.5f}")

    oof_mae_raw_10 = mean_absolute_error(y, oof_preds_10)
    oof_preds_post_10 = np.round(np.clip(oof_preds_10, 0.0, 1.0), 2)
    oof_mae_post_10 = mean_absolute_error(y, oof_preds_post_10)
    
    v16_baseline = 0.13951
    diff10 = oof_mae_post_10 - v16_baseline
    diff10_str = f"+{diff10:.6f} (악화)" if diff10 > 0 else f"{diff10:.6f} (개선)"
    
    print(f"  10-Fold OOF MAE (Raw)            : {oof_mae_raw_10:.6f}")
    print(f"  10-Fold OOF MAE (Post-processed) : {oof_mae_post_10:.6f}")
    print(f"  v16 기준값                        : {v16_baseline:.5f}")
    print(f"  개선폭                            : {diff10_str}")

    test_preds_post = np.round(np.clip(test_preds_10, 0.0, 1.0), 2)
    
    print("\n[Test Predictions Statistics]")
    print(f"  Mean       : {test_preds_post.mean():.5f}")
    print(f"  Std Dev    : {test_preds_post.std():.5f}")
    print(f"  Min        : {test_preds_post.min():.5f}")
    print(f"  Max        : {test_preds_post.max():.5f}")
    print(f"  N Unique   : {len(np.unique(test_preds_post))}")

    os.makedirs('result/v16', exist_ok=True)
    submit_path = 'result/v16/submit_v16-5_feat_reduced.csv'
    pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_preds_post
    }).to_csv(submit_path, index=False)
    
    print(f"\nFinal submission file saved to: {submit_path}")

if __name__ == '__main__':
    main()
