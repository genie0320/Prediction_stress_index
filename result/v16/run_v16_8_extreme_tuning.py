import os
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
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

def objective(trial, X, y):
    # Search Space
    C = trial.suggest_float('C', 1.0, 100.0, log=True)
    gamma = trial.suggest_float('gamma', 0.01, 10.0, log=True)
    n_quantiles = trial.suggest_int('n_quantiles', 500, 2000)
    
    # Guard 1: Dynamic Seed (Preventing split overfitting)
    current_seed = 42 + trial.number
    kf = KFold(n_splits=5, shuffle=True, random_state=current_seed)
    
    fold_maes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal', random_state=42)
        y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=0.0)
        model.fit(X_tr_sc, y_tr_t)
        
        pred_t = model.predict(X_va_sc)
        pred = qt.inverse_transform(pred_t.reshape(-1, 1)).flatten()
        pred = np.clip(pred, 0, 1)
        
        mae = mean_absolute_error(y_va, pred)
        fold_maes.append(mae)
        
        # Guard 2: Median Pruning
        trial.report(mae, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return np.mean(fold_maes)

def validate_best_params(best_params, X, y):
    print("\n" + "="*50)
    print(">>> Guard 3: Multi-seed Stability Check for Best Params...")
    seeds = [42, 2026, 777]
    seed_maes = []
    
    for seed in seeds:
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        
        for train_idx, val_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            scaler = RobustScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            
            qt = QuantileTransformer(n_quantiles=best_params['n_quantiles'], output_distribution='normal', random_state=42)
            y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
            
            model = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], epsilon=0.0)
            model.fit(X_tr_sc, y_tr_t)
            
            pred = qt.inverse_transform(model.predict(X_va_sc).reshape(-1, 1)).flatten()
            oof[val_idx] = np.clip(pred, 0, 1)
            
        mae = mean_absolute_error(y, oof)
        seed_maes.append(mae)
        print(f"  seed={seed} → MAE: {mae:.6f}")
        
    mean_mae = np.mean(seed_maes)
    std_mae = np.std(seed_maes)
    print(f"\n→ Mean MAE: {mean_mae:.6f}")
    print(f"→ Std MAE : {std_mae:.6f}")
    
    if std_mae > 0.003:
        print("  [WARNING] Parameter set is UNSTABLE! (High Variance)")
    else:
        print("  [Safe] Parameter set is STABLE.")
    print("="*50 + "\n")
    return mean_mae, std_mae

def main():
    train = pd.read_csv('train.csv')
    train_proc = preprocess_data(train)
    
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
    
    print(">>> Starting Overnight Extreme Tuning (1000 Trials)")
    print("    Search Space: C [1.0~100.0], gamma [0.01~10.0], n_quantiles [500~2000]")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1)
    )
    
    study.optimize(lambda trial: objective(trial, X, y), n_trials=1000, show_progress_bar=True)
    
    print("\n" + "="*50)
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best MAE  : {study.best_value:.6f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    # Validation
    mean_mae, std_mae = validate_best_params(study.best_params, X, y)
    
    # Save to file
    os.makedirs('result/v16', exist_ok=True)
    with open('result/v16/v16_8_best_params.txt', 'w') as f:
        f.write("=== SVR Extreme Tuning Results (1000 Trials) ===\n")
        f.write(f"Best Trial: {study.best_trial.number}\n")
        f.write(f"Best Optuna 5-Fold MAE: {study.best_value:.6f}\n")
        f.write("\nParameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n10-Fold 3-Seed Validation:\n")
        f.write(f"Mean MAE: {mean_mae:.6f}\n")
        f.write(f"Std MAE : {std_mae:.6f}\n")

if __name__ == '__main__':
    main()
