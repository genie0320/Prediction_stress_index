import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import optuna

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
        'n_medical', 'n_family'
    ]
    
    X = train_proc[features]
    y = train_proc['stress_score']
    X_test = test_proc[features]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Generate SVR OOF & Test
    print(">>> Generating SVR Predictions (using best known proxy params C=3.56, gamma=0.83)...")
    svr_oof = np.zeros(len(train))
    svr_test = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)
        
        qt = QuantileTransformer(n_quantiles=3000, output_distribution='normal', random_state=42)
        y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        # User should replace these with Optuna best params
        svr = SVR(kernel='rbf', C=3.56, gamma=0.83, epsilon=0.0)
        svr.fit(X_tr_sc, y_tr_trans)
        
        va_pred = qt.inverse_transform(svr.predict(X_va_sc).reshape(-1, 1)).flatten()
        svr_oof[val_idx] = va_pred
        
        te_pred = qt.inverse_transform(svr.predict(X_te_sc).reshape(-1, 1)).flatten()
        svr_test += te_pred / 5.0
        
    print(f"SVR OOF MAE: {mean_absolute_error(y, svr_oof):.6f}")
    
    # 2. Generate ExtraTrees OOF & Test
    print("\n>>> Generating ExtraTrees Predictions...")
    et_oof = np.zeros(len(train))
    et_test = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        et = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        et.fit(X_tr, y_tr)
        
        et_oof[val_idx] = et.predict(X_va)
        et_test += et.predict(X_test) / 5.0
        
    print(f"ExtraTrees OOF MAE: {mean_absolute_error(y, et_oof):.6f}")
    
    # 3. Blending Weight Optimization using Optuna
    print("\n>>> Optimizing Ensemble Weights using Optuna...")
    
    def objective(trial):
        w = trial.suggest_float('w', 0.0, 1.0)
        blend = w * svr_oof + (1 - w) * et_oof
        return mean_absolute_error(y, blend)
        
    # Turn off excessive logging for this fast search
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100)
    
    best_w = study.best_params['w']
    best_mae = study.best_value
    
    print("\n" + "="*50)
    print(f"Best Weight (w for SVR): {best_w:.5f}")
    print(f"Best Weight (1-w for ET) : {1 - best_w:.5f}")
    print(f"Blended OOF MAE          : {best_mae:.6f}")
    print("="*50)
    
    # 4. Final Inference and Save
    final_test_preds = best_w * svr_test + (1 - best_w) * et_test
    final_test_preds_post = np.round(np.clip(final_test_preds, 0.0, 1.0), 2)
    
    os.makedirs('result/v16', exist_ok=True)
    submit_path = 'result/v16/submit_v16_2_blend.csv'
    pd.DataFrame({
        'ID': test['ID'],
        'stress_score': final_test_preds_post
    }).to_csv(submit_path, index=False)
    
    print(f"\nSaved Blended submission to: {submit_path}")

if __name__ == '__main__':
    main()
