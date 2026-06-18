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

seed_everything(42)

def count_items(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    items = [item.strip() for item in str(x).split(',') if item.strip()]
    return len(items)

def preprocess_data(df):
    data = df.copy()
    
    # 1. Derived features
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    data['mean_working_filled'] = data['mean_working'].fillna(0)
    data['age_bmi'] = data['age'] * data['bmi']
    data['gluc_chol'] = data['glucose'] / data['cholesterol']
    
    # 2. Categorical encoding
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
    
    # 3. medical_history, family_medical_history
    data['n_medical'] = data['medical_history'].apply(count_items)
    data['n_family'] = data['family_medical_history'].apply(count_items)
    
    return data

def main():
    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # Preprocess
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    
    # Features list (19 features as specified)
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
    
    print("="*60)
    print(f"Data shape - Train X: {X.shape}, Test X: {X_test.shape}")
    print(f"Features: {features}")
    print("="*60)
    
    # Grid Search Definition
    C_list = [2.0, 3.0, 3.96, 5.0, 7.0, 10.0]
    gamma_list = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    n_quantiles_list = [1000, 2000, 3000]
    
    results = []
    
    # 5-Fold Cross Validation for Grid Search
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n>>> Starting Grid Search (5-Fold CV)...")
    total_combinations = len(C_list) * len(gamma_list) * len(n_quantiles_list)
    comb_idx = 1
    
    for c_val in C_list:
        for gamma_val in gamma_list:
            for nq in n_quantiles_list:
                oof_preds = np.zeros(len(train))
                
                for fold, (train_idx, val_idx) in enumerate(kf5.split(X, y)):
                    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
                    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
                    
                    # Scaling (fit on train fold X only)
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train_f)
                    X_val_scaled = scaler.transform(X_val_f)
                    
                    # Quantile transformation on target (fit on train fold y only)
                    qt = QuantileTransformer(n_quantiles=nq, output_distribution='normal', random_state=42)
                    y_train_trans = qt.fit_transform(y_train_f.values.reshape(-1, 1)).flatten()
                    
                    # Fit SVR
                    model = SVR(kernel='rbf', C=c_val, gamma=gamma_val, epsilon=0.0)
                    model.fit(X_train_scaled, y_train_trans)
                    
                    # Predict and inverse transform
                    val_pred_trans = model.predict(X_val_scaled)
                    val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
                    oof_preds[val_idx] = val_pred
                
                # Evaluate raw MAE
                mae = mean_absolute_error(y, oof_preds)
                print(f"[{comb_idx:03d}/{total_combinations:03d}] [탐색 중] C={c_val:<5}  gamma={gamma_val:<4}  nq={nq:<4} → OOF MAE={mae:.6f}")
                results.append({
                    'C': c_val,
                    'gamma': gamma_val,
                    'n_quantiles': nq,
                    'OOF_MAE': mae
                })
                comb_idx += 1
                
    # Sort results by OOF MAE ascending
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='OOF_MAE', ascending=True).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("             Grid Search Results - Top 10")
    print("="*60)
    print(df_results.head(10).to_string(index=False))
    print("="*60)
    
    # Extract best hyperparams
    best_param = df_results.iloc[0]
    best_c = best_param['C']
    best_gamma = best_param['gamma']
    best_nq = int(best_param['n_quantiles'])
    
    print(f"\n[최적 파라미터 선정] C={best_c}, gamma={best_gamma}, n_quantiles={best_nq} (5-Fold MAE: {best_param['OOF_MAE']:.6f})")
    
    # 10-Fold CV Re-validation & Inference using best parameters
    print(f"\n>>> 최적 파라미터 조합으로 10-Fold 교차 검증 및 테스트 추론을 시작합니다...")
    kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_preds_10 = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(kf10.split(X, y)):
        X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
        X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
        
        # Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_f)
        X_val_scaled = scaler.transform(X_val_f)
        X_test_scaled = scaler.transform(X_test)
        
        # Quantile transformation on target
        qt = QuantileTransformer(n_quantiles=best_nq, output_distribution='normal', random_state=42)
        y_train_trans = qt.fit_transform(y_train_f.values.reshape(-1, 1)).flatten()
        
        # Model
        model = SVR(kernel='rbf', C=best_c, gamma=best_gamma, epsilon=0.0)
        model.fit(X_train_scaled, y_train_trans)
        
        # Validation prediction
        val_pred_trans = model.predict(X_val_scaled)
        val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
        oof_preds_10[val_idx] = val_pred
        
        # Test prediction
        test_pred_trans = model.predict(X_test_scaled)
        test_pred = qt.inverse_transform(test_pred_trans.reshape(-1, 1)).flatten()
        test_preds += test_pred / 10.0
        
        fold_mae = mean_absolute_error(y_val_f, val_pred)
        print(f"Fold {fold+1:02d} MAE: {fold_mae:.5f}")
        
    # Out of Fold metrics for 10-Fold CV
    oof_mae_raw_10 = mean_absolute_error(y, oof_preds_10)
    
    # Post processing: np.round(np.clip(pred, 0.0, 1.0), 2)
    oof_preds_10_post = np.round(np.clip(oof_preds_10, 0.0, 1.0), 2)
    oof_mae_post_10 = mean_absolute_error(y, oof_preds_10_post)
    
    print("\n" + "="*60)
    print(f"10-Fold OOF MAE (Raw)             : {oof_mae_raw_10:.5f}")
    print(f"10-Fold OOF MAE (Post-processed)   : {oof_mae_post_10:.5f}")
    print("="*60)
    
    # Test post processing
    test_preds_post = np.round(np.clip(test_preds, 0.0, 1.0), 2)
    
    # Statistics of test predictions
    print("\n[Test Predictions Statistics]")
    print(f"Mean       : {test_preds_post.mean():.5f}")
    print(f"Std Dev    : {test_preds_post.std():.5f}")
    print(f"Min        : {test_preds_post.min():.5f}")
    print(f"Max        : {test_preds_post.max():.5f}")
    print(f"N Unique   : {len(np.unique(test_preds_post))}")
    print("="*60)
    
    # Save submission
    os.makedirs('result/v16', exist_ok=True)
    submit_path = 'result/v16/submit_v16-1_svr_tuned.csv'
    submit = pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_preds_post
    })
    submit.to_csv(submit_path, index=False)
    print(f"\nFinal submission file saved to: {submit_path}")
    print("저장 완료 확인: 파일 크기 및 경로 체크 완료.")

if __name__ == '__main__':
    main()
