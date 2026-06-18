import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from datetime import datetime

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

def evaluate_features(features, X_all, y, seeds, n_splits=5):
    X = X_all[features]
    C = 28.6022
    gamma = 1.0566
    nq = 1365
    
    seed_maes = []
    
    for seed in seeds:
        seed_everything(seed)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X))
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            qt = QuantileTransformer(n_quantiles=nq, output_distribution='normal', random_state=seed)
            y_train_trans = qt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=0.0)
            model.fit(X_train_scaled, y_train_trans)
            
            val_pred_trans = model.predict(X_val_scaled)
            val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
            oof_preds[val_idx] = val_pred
            
        seed_mae = mean_absolute_error(y, oof_preds)
        seed_maes.append(seed_mae)
        
    return np.mean(seed_maes)

def main():
    print(f"[{datetime.now()}] Started run_v17_1_backward.py")
    train = pd.read_csv('train.csv')
    train_proc = preprocess_data(train)
    
    base_features = [
        'age', 'height', 'weight', 'cholesterol',
        'systolic_blood_pressure', 'diastolic_blood_pressure',
        'glucose', 'bone_density',
        'bmi', 'pp', 'map_val', 'mean_working_filled',
        'activity_enc', 'edu_enc', 'sleep_enc', 'gender_enc', 'smoke_enc',
        'n_medical', 'n_family'
    ]
    
    y = train_proc['stress_score']
    X_all = train_proc[base_features]
    seeds = [42, 777, 2026]
    n_splits = 5
    
    current_features = list(base_features)
    print(f"Initial Features: {len(current_features)}")
    
    best_mae = evaluate_features(current_features, X_all, y, seeds, n_splits)
    print(f"Base MAE (19 features): {best_mae:.6f}\n")
    
    history = [(current_features.copy(), best_mae)]
    
    os.makedirs('result', exist_ok=True)
    log_file = 'result/run_v17_1_backward_log.txt'
    
    with open(log_file, 'w') as f:
        f.write(f"[{datetime.now()}] Started Backward Elimination\n")
        f.write(f"Parameters: C=28.6022, gamma=1.0566, nq=1365\n")
        f.write(f"Base MAE (19 features): {best_mae:.6f}\n\n")
    
    while len(current_features) > 8:
        print(f"=== Testing removal of 1 feature from {len(current_features)} features ===")
        removal_results = []
        
        for feature_to_remove in current_features:
            features_to_test = [f for f in current_features if f != feature_to_remove]
            mae = evaluate_features(features_to_test, X_all, y, seeds, n_splits)
            removal_results.append((feature_to_remove, mae))
            print(f"  - Remove {feature_to_remove:25s} -> MAE: {mae:.6f}")
            
        removal_results.sort(key=lambda x: x[1])
        best_removal_feature, best_removal_mae = removal_results[0]
        
        print(f"\n[Result] Best removal: {best_removal_feature} (MAE: {best_removal_mae:.6f})")
        
        with open(log_file, 'a') as f:
            f.write(f"Step {19 - len(current_features) + 1}:\n")
            for f_rem, m in removal_results:
                f.write(f"  - Remove {f_rem:25s} -> MAE: {m:.6f}\n")
            f.write(f"  => Best removal: {best_removal_feature} (MAE: {best_removal_mae:.6f})\n")
        
        if best_removal_mae < best_mae:
            print("  -> MAE Improved! Continuing elimination.")
            best_mae = best_removal_mae
            current_features.remove(best_removal_feature)
            history.append((current_features.copy(), best_mae))
            with open(log_file, 'a') as f:
                f.write("  -> MAE Improved! Removed.\n\n")
        else:
            print("  -> MAE did not improve. Stopping backward elimination.")
            with open(log_file, 'a') as f:
                f.write("  -> MAE did not improve. Stopping.\n\n")
            break
            
        print("-" * 50)
        
    print("\n" + "="*50)
    print("Backward Elimination Finished.")
    print(f"Final Selected Features ({len(current_features)}): {current_features}")
    print(f"Final Best MAE: {best_mae:.6f}")
    
    with open(log_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("Backward Elimination Finished.\n")
        f.write(f"Final Selected Features ({len(current_features)}): {current_features}\n")
        f.write(f"Final Best MAE: {best_mae:.6f}\n")

if __name__ == '__main__':
    main()
