import os
import time
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    
    data['age_bmi'] = data['age'] * data['bmi']
    data['gluc_chol'] = data['glucose'] / data['cholesterol']
    
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
    
    # 16-3 Features
    data['mean_working_isna'] = data['mean_working'].isna().astype(int)
    data['is_retired'] = ((data['age'] <= 18) | (data['age'] >= 66)).astype(int)
    data['mean_working_filled'] = data['mean_working'].fillna(0)
    
    return data

def run_cv(params, kf, X, y):
    oof_preds = np.zeros(len(X))
    c_val = params['C']
    gamma_val = params['gamma']
    nq = params['n_quantiles']
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        qt = QuantileTransformer(n_quantiles=nq, output_distribution='normal', random_state=42)
        y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        model = SVR(kernel='rbf', C=c_val, gamma=gamma_val, epsilon=0.0)
        model.fit(X_tr_sc, y_tr_trans)
        
        val_pred_trans = model.predict(X_va_sc)
        val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
        oof_preds[val_idx] = val_pred
        
    return mean_absolute_error(y, oof_preds)

def main():
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
        'n_medical', 'n_family',
        'mean_working_isna', 'is_retired'
    ]
    
    X = train_proc[features]
    y = train_proc['stress_score']
    X_test = test_proc[features]
    
    print("\n[Optuna 탐색]")
    
    results = []
    
    def objective(trial):
        start_time = time.time()
        
        c_val = trial.suggest_float("C", 1.0, 15.0, log=True)
        gamma_val = trial.suggest_float("gamma", 0.3, 5.0, log=True)
        nq = trial.suggest_int("n_quantiles", 500, 2000)
        
        # Guard 1: Trial 마다 다른 KFold seed 배정
        kf_seed = 42 + trial.number
        kf = KFold(n_splits=5, shuffle=True, random_state=kf_seed)
        
        mae = run_cv({'C': c_val, 'gamma': gamma_val, 'n_quantiles': nq}, kf, X, y)
        
        elapsed = time.time() - start_time
        
        print(f"  [{trial.number:03d}] C={c_val:7.4f}  gamma={gamma_val:6.4f}  nq={nq:4d}  → OOF MAE={mae:.6f}  ({elapsed:.1f}s)")
        
        results.append({
            'trial': trial.number,
            'C': c_val,
            'gamma': gamma_val,
            'n_q': nq,
            'OOF MAE': mae
        })
        
        return mae

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=100)
    
    # Top 10 결과 출력
    print("\n[Top 10 파라미터]")
    df_results = pd.DataFrame(results).sort_values(by='OOF MAE', ascending=True).reset_index(drop=True)
    df_results.index = df_results.index + 1
    df_results.index.name = '순위'
    # Drop trial number for cleaner table matching user's requested columns
    print(df_results[['C', 'gamma', 'n_q', 'OOF MAE']].head(10).to_string())
    
    best_params = study.best_params
    best_c = best_params['C']
    best_gamma = best_params['gamma']
    best_nq = best_params['n_quantiles']
    
    print("\n[안정성 검증]")
    seed_maes = []
    for val_seed in [42, 2026, 777]:
        kf = KFold(n_splits=5, shuffle=True, random_state=val_seed)
        mae = run_cv(best_params, kf, X, y)
        seed_maes.append(mae)
        print(f"  seed={val_seed:<4d}  → MAE: {mae:.6f}")
        
    std_mae = np.std(seed_maes)
    print(f"  3-Seed MAE std: {std_mae:.6f}")
    if std_mae > 0.003:
        print("  [Warning] 안정성 기준(0.003) 초과 - 과적합 우려")
    else:
        print("  [Safe] 파라미터 안정성 확인됨")
        
    print("\n[10-Fold 재검증]")
    kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds_10 = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf10.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)
        
        qt = QuantileTransformer(n_quantiles=best_nq, output_distribution='normal', random_state=42)
        y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        model = SVR(kernel='rbf', C=best_c, gamma=best_gamma, epsilon=0.0)
        model.fit(X_tr_sc, y_tr_trans)
        
        va_pred = qt.inverse_transform(model.predict(X_va_sc).reshape(-1, 1)).flatten()
        oof_preds_10[val_idx] = va_pred
        
        te_pred = qt.inverse_transform(model.predict(X_te_sc).reshape(-1, 1)).flatten()
        test_preds += te_pred / 10.0
        
        print(f"  Fold {fold+1:02d} MAE: {mean_absolute_error(y_va, va_pred):.5f}")

    oof_mae_raw_10 = mean_absolute_error(y, oof_preds_10)
    oof_preds_10_post = np.round(np.clip(oof_preds_10, 0.0, 1.0), 2)
    oof_mae_post_10 = mean_absolute_error(y, oof_preds_10_post)
    
    v16_baseline = 0.13951
    diff = oof_mae_post_10 - v16_baseline
    diff_str = f"+{diff:.6f} (악화)" if diff > 0 else f"{diff:.6f} (개선)"
    
    print(f"  10-Fold OOF MAE (Raw)            : {oof_mae_raw_10:.6f}")
    print(f"  10-Fold OOF MAE (Post-processed) : {oof_mae_post_10:.6f}")
    print(f"  v16 기준값                        : {v16_baseline:.5f}")
    print(f"  개선폭                            : {diff_str}")

    test_preds_post = np.round(np.clip(test_preds, 0.0, 1.0), 2)
    
    print("\n[Test Predictions Statistics]")
    print(f"  Mean       : {test_preds_post.mean():.5f}")
    print(f"  Std Dev    : {test_preds_post.std():.5f}")
    print(f"  Min        : {test_preds_post.min():.5f}")
    print(f"  Max        : {test_preds_post.max():.5f}")
    print(f"  N Unique   : {len(np.unique(test_preds_post))}")

    os.makedirs('result/v16', exist_ok=True)
    submit_path = 'result/v16/submit_v16-4_retired_tuned.csv'
    pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_preds_post
    }).to_csv(submit_path, index=False)
    
    print(f"\nFinal submission file saved to: {submit_path}")

if __name__ == '__main__':
    main()
