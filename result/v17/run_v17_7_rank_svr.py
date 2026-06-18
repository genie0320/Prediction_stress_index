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

def get_rank_feats(X_tr, X_val, X_te, num_feats):
    tr_ranks = X_tr[num_feats].rank(pct=True)
    val_ranks = pd.DataFrame(index=X_val.index)
    te_ranks = pd.DataFrame(index=X_te.index) if X_te is not None else None
    
    for feat in num_feats:
        sorted_tr = np.sort(X_tr[feat].values)
        # Using searchsorted(side='right') / N is perfectly equivalent to (X_tr <= v).mean()
        val_ranks[f'rank_{feat}'] = np.searchsorted(sorted_tr, X_val[feat].values, side='right') / len(sorted_tr)
        if te_ranks is not None:
            te_ranks[f'rank_{feat}'] = np.searchsorted(sorted_tr, X_te[feat].values, side='right') / len(sorted_tr)
            
    tr_ranks.columns = [f'rank_{feat}' for feat in num_feats]
    return tr_ranks, val_ranks, te_ranks

def evaluate_fold(X_tr, y_tr, X_va, y_va, X_te=None):
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te) if X_te is not None else None
    
    qt = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
    y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
    
    model = SVR(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
    model.fit(X_tr_sc, y_tr_trans)
    
    va_pred = qt.inverse_transform(model.predict(X_va_sc).reshape(-1, 1)).flatten()
    te_pred = qt.inverse_transform(model.predict(X_te_sc).reshape(-1, 1)).flatten() if X_te is not None else None
    
    return va_pred, te_pred

def run_cv(X, y, X_test, feature_set, num_feats, rank_feats, n_splits=5, seed=42):
    seed_everything(seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        tr_ranks, va_ranks, te_ranks = get_rank_feats(X_tr, X_va, X_test, num_feats)
        
        X_tr_full = pd.concat([X_tr, tr_ranks], axis=1)
        X_va_full = pd.concat([X_va, va_ranks], axis=1)
        X_te_full = pd.concat([X_test, te_ranks], axis=1) if X_test is not None else None
        
        va_pred, te_pred = evaluate_fold(
            X_tr_full[feature_set], y_tr,
            X_va_full[feature_set], y_va,
            X_te_full[feature_set] if X_te_full is not None else None
        )
        oof_preds[val_idx] = va_pred
        if X_test is not None:
            test_preds += te_pred / n_splits
            
    return oof_preds, test_preds

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    y = train_proc['stress_score']
    
    features_base = [
        'age','height','weight','cholesterol',
        'systolic_blood_pressure','diastolic_blood_pressure',
        'glucose','bone_density',
        'bmi','pp','map_val',
        'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
        'n_medical','n_family'
    ]
    
    num_feats = ['age','height','weight','cholesterol',
                 'systolic_blood_pressure','diastolic_blood_pressure',
                 'glucose','bone_density']
                 
    rank_feats = [f'rank_{c}' for c in num_feats]
    
    features_A = features_base.copy()
    features_B = features_base + rank_feats
    features_C = rank_feats.copy()
    features_D = rank_feats + ['n_medical', 'n_family', 'activity_enc', 'edu_enc', 'sleep_enc', 'gender_enc', 'smoke_enc']
    
    combos = {
        '[A]': features_A,
        '[B]': features_B,
        '[C]': features_C,
        '[D]': features_D
    }
    
    baseline_mae = 0.13578
    
    print("[5-Fold 빠른 비교]")
    print(f"{'조합':<4} {'피처수':<5} {'OOF MAE':<10} 17-2 대비(0.13578)")
    
    best_mae = 999
    best_combo = None
    best_feats = None
    
    for combo_name, feats in combos.items():
        oof_preds, _ = run_cv(train_proc, y, None, feats, num_feats, rank_feats, n_splits=5, seed=42)
        mae = mean_absolute_error(y, oof_preds)
        diff = baseline_mae - mae
        diff_str = "기준" if combo_name == '[A]' else f"{diff:+.6f}"
        print(f"{combo_name:<4} {len(feats):<5} {mae:.6f}   {diff_str}")
        
        if mae < best_mae:
            best_mae = mae
            best_combo = combo_name
            best_feats = feats
            
    print(f"\n최선 조합: {best_combo} (피처 수: {len(best_feats)})")
    
    print("\n[최선 조합 10-Fold 재검증]")
    seed_everything(42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds_10 = np.zeros(len(train_proc))
    test_preds_10 = np.zeros(len(test_proc))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_proc, y)):
        X_tr, y_tr = train_proc.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = train_proc.iloc[val_idx], y.iloc[val_idx]
        
        tr_ranks, va_ranks, te_ranks = get_rank_feats(X_tr, X_va, test_proc, num_feats)
        
        X_tr_full = pd.concat([X_tr, tr_ranks], axis=1)
        X_va_full = pd.concat([X_va, va_ranks], axis=1)
        X_te_full = pd.concat([test_proc, te_ranks], axis=1)
        
        va_pred, te_pred = evaluate_fold(
            X_tr_full[best_feats], y_tr,
            X_va_full[best_feats], y_va,
            X_te_full[best_feats]
        )
        oof_preds_10[val_idx] = va_pred
        test_preds_10 += te_pred / 10.0
        fold_mae = mean_absolute_error(y_va, va_pred)
        print(f"Fold {fold+1:02d} MAE: {fold_mae:.5f}")
        
    oof_mae_raw_10 = mean_absolute_error(y, oof_preds_10)
    oof_preds_post_10 = np.round(np.clip(oof_preds_10, 0.0, 1.0), 2)
    oof_mae_post_10 = mean_absolute_error(y, oof_preds_post_10)
    
    imp_raw = baseline_mae - oof_mae_raw_10
    imp_str = "개선" if imp_raw > 0 else "악화"
    
    print(f"10-Fold OOF MAE (Raw)            : {oof_mae_raw_10:.6f}")
    print(f"10-Fold OOF MAE (Post-processed) : {oof_mae_post_10:.6f}")
    print(f"17-2 기준값                      : {baseline_mae}")
    print(f"개선폭                           : {imp_raw:+.6f} ({imp_str})")
    
    print("\n[3-Seed 안정성 검증]")
    seeds = [42, 777, 2026]
    seed_maes = []
    for s in seeds:
        oof_preds_s, _ = run_cv(train_proc, y, None, best_feats, num_feats, rank_feats, n_splits=5, seed=s)
        mae_s = mean_absolute_error(y, oof_preds_s)
        print(f"seed={s:<4} → MAE: {mae_s:.6f}")
        seed_maes.append(mae_s)
        
    print(f"평균 MAE:   {np.mean(seed_maes):.6f}")
    print(f"Std:        {np.std(seed_maes):.6f}")
    
    test_preds_post = np.round(np.clip(test_preds_10, 0.0, 1.0), 2)
    print("\n[Test 예측값 통계]")
    print(f"Mean       : {test_preds_post.mean():.6f}")
    print(f"Std Dev    : {test_preds_post.std():.6f}")
    print(f"Min        : {test_preds_post.min():.6f}")
    print(f"Max        : {test_preds_post.max():.6f}")
    print(f"N_unique   : {len(np.unique(test_preds_post))}")
    
    os.makedirs('result/v17', exist_ok=True)
    submit_path = 'result/v17/submit_v17-7_rank_svr.csv'
    pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_preds_post
    }).to_csv(submit_path, index=False)
    print(f"\n최선 조합 예측 결과 저장 완료: {submit_path}")

if __name__ == "__main__":
    main()
