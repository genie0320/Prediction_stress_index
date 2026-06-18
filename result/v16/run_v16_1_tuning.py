import os
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
from optuna.pruners import MedianPruner

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
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    data['mean_working_filled'] = data['mean_working'].fillna(0)
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
    return data

def run_cv(params, kf, X, y):
    oof_preds = np.zeros(len(X))
    c_val = params['C']
    gamma_val = params['gamma']
    nq = params['n_quantiles']
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
        X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_f)
        X_val_scaled = scaler.transform(X_val_f)
        
        qt = QuantileTransformer(n_quantiles=nq, output_distribution='normal', random_state=42)
        y_train_trans = qt.fit_transform(y_train_f.values.reshape(-1, 1)).flatten()
        
        model = SVR(kernel='rbf', C=c_val, gamma=gamma_val, epsilon=0.0)
        model.fit(X_train_scaled, y_train_trans)
        
        val_pred_trans = model.predict(X_val_scaled)
        val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
        oof_preds[val_idx] = val_pred
        
    return mean_absolute_error(y, oof_preds)

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
    
    print(f">>> Optuna Parameter Search Started... (Trials=150)")
    
    def objective(trial):
        c_val = trial.suggest_float("C", 1.0, 50.0, log=True)
        gamma_val = trial.suggest_float("gamma", 0.01, 10.0, log=True)
        nq = trial.suggest_int("n_quantiles", 500, 3000)
        
        # Guard 1: Use trial.number to change KFold seed to prevent overfitting to a specific split
        kf_seed = 42 + trial.number
        kf = KFold(n_splits=5, shuffle=True, random_state=kf_seed)
        
        fold_maes = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
            X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_f)
            X_val_scaled = scaler.transform(X_val_f)
            
            qt = QuantileTransformer(n_quantiles=nq, output_distribution='normal', random_state=42)
            y_train_trans = qt.fit_transform(y_train_f.values.reshape(-1, 1)).flatten()
            
            model = SVR(kernel='rbf', C=c_val, gamma=gamma_val, epsilon=0.0)
            model.fit(X_train_scaled, y_train_trans)
            
            val_pred_trans = model.predict(X_val_scaled)
            val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
            
            fold_mae = mean_absolute_error(y_val_f, val_pred)
            fold_maes.append(fold_mae)
            
            # Guard 2: Report intermediate value to Pruner
            trial.report(np.mean(fold_maes), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(fold_maes)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=150)
    
    print("\n" + "="*50)
    print("Best Trial:", study.best_trial.number)
    print("Best MAE  :", study.best_value)
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*50)
    
    best_params = study.best_params
    
    # Guard 3: Multi-seed final validation
    print("\n>>> Guard 3: Multi-seed Stability Check for Best Params...")
    seed_maes = []
    for val_seed in [42, 2026, 777]:
        kf = KFold(n_splits=5, shuffle=True, random_state=val_seed)
        mae = run_cv(best_params, kf, X, y)
        seed_maes.append(mae)
        print(f"  seed={val_seed} → MAE: {mae:.6f}")
        
    std_mae = np.std(seed_maes)
    print(f"\n→ Mean MAE: {np.mean(seed_maes):.6f}")
    print(f"→ Std MAE : {std_mae:.6f}")
    
    if std_mae >= 0.003:
        print("  [Warning] Parameter set is UNSTABLE (std >= 0.003)")
    else:
        print("  [Safe] Parameter set is STABLE.")
    print("="*50)

if __name__ == '__main__':
    main()
