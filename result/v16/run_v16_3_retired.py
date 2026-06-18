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
    
    # Existing derived features
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    
    data['age_bmi'] = data['age'] * data['bmi']
    data['gluc_chol'] = data['glucose'] / data['cholesterol']
    
    # Categorical encodings
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
    
    # Phase 3 Enhancements: explicitly separating NaN and retirement signals
    data['mean_working_isna'] = data['mean_working'].isna().astype(int)
    data['is_retired'] = ((data['age'] <= 18) | (data['age'] >= 66)).astype(int)
    data['mean_working_filled'] = data['mean_working'].fillna(0)
    
    return data

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    
    # 21 features with the explicit NA/retired flags
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
    
    print(f"Data shape - Train X: {X.shape}, Test X: {X_test.shape}")
    print(f"Features ({len(features)}): {features}\n")
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    print(">>> 10-Fold CV Training (Using base SVR proxy params C=3.56, gamma=0.83)...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        qt = QuantileTransformer(n_quantiles=3000, output_distribution='normal', random_state=42)
        y_train_trans = qt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        model = SVR(kernel='rbf', C=3.56, gamma=0.83, epsilon=0.0)
        model.fit(X_train_scaled, y_train_trans)
        
        val_pred_trans = model.predict(X_val_scaled)
        val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
        oof_preds[val_idx] = val_pred
        
        test_pred_trans = model.predict(X_test_scaled)
        test_pred = qt.inverse_transform(test_pred_trans.reshape(-1, 1)).flatten()
        test_preds += test_pred / 10.0
        
        print(f"  Fold {fold+1:02d} MAE: {mean_absolute_error(y_val, val_pred):.5f}")
        
    oof_mae_raw = mean_absolute_error(y, oof_preds)
    oof_preds_post = np.round(np.clip(oof_preds, 0.0, 1.0), 2)
    oof_mae_post = mean_absolute_error(y, oof_preds_post)
    
    print("\n" + "="*50)
    print(f"10-Fold OOF MAE (Raw)             : {oof_mae_raw:.5f}")
    print(f"10-Fold OOF MAE (Post-processed)   : {oof_mae_post:.5f}")
    print("="*50)
    
    test_preds_post = np.round(np.clip(test_preds, 0.0, 1.0), 2)
    
    os.makedirs('result/v16', exist_ok=True)
    submit_path = 'result/v16/submit_v16_3_retired.csv'
    pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_preds_post
    }).to_csv(submit_path, index=False)
    
    print(f"\nFinal submission file saved to: {submit_path}")

if __name__ == '__main__':
    main()
