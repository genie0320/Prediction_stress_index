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

# Seed setting for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

def count_items(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    # Split by comma and filter out empty items after stripping
    items = [item.strip() for item in str(x).split(',') if item.strip()]
    return len(items)

def preprocess_data(df):
    data = df.copy()
    
    # 1. Derived features (excluding mean_working_filled)
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    
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
    
    # Features list (18 features)
    features = [
        'age','height','weight','cholesterol',
        'systolic_blood_pressure','diastolic_blood_pressure',
        'glucose','bone_density',
        'bmi','pp','map_val',
        'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
        'n_medical','n_family'
    ]
    
    X = train_proc[features]
    y = train_proc['stress_score']
    X_test = test_proc[features]
    
    print(f"Data shape - Train X: {X.shape}, Test X: {X_test.shape}")
    print(f"Features: {features}\n")
    
    # 10-Fold Cross Validation Setup
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Scaling inputs
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Transforming targets
        qt = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
        y_train_trans = qt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        
        # SVR Model
        model = SVR(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
        model.fit(X_train_scaled, y_train_trans)
        
        # Predict on validation fold & Inverse Transform
        val_pred_trans = model.predict(X_val_scaled)
        val_pred = qt.inverse_transform(val_pred_trans.reshape(-1, 1)).flatten()
        oof_preds[val_idx] = val_pred
        
        # Predict on test data & Inverse Transform
        test_pred_trans = model.predict(X_test_scaled)
        test_pred = qt.inverse_transform(test_pred_trans.reshape(-1, 1)).flatten()
        test_preds += test_pred / 10.0
        
        fold_mae = mean_absolute_error(y_val, val_pred)
        print(f"Fold {fold+1:02d} - MAE: {fold_mae:.5f}")
        
    # Out of Fold metrics
    oof_mae_raw = mean_absolute_error(y, oof_preds)
    
    # Post processing: np.round(np.clip(pred, 0.0, 1.0), 2)
    oof_preds_post = np.round(np.clip(oof_preds, 0.0, 1.0), 2)
    oof_mae_post = mean_absolute_error(y, oof_preds_post)
    
    v16_baseline = 0.13951
    improvement = v16_baseline - oof_mae_raw
    
    print("\n" + "="*50)
    print(f"10-Fold OOF MAE (Raw)             : {oof_mae_raw:.5f}")
    print(f"10-Fold OOF MAE (Post-processed)  : {oof_mae_post:.5f}")
    print(f"v16 기준값(0.13951) 대비 개선폭     : {improvement:+.5f}")
    print("="*50)
    
    # Test prediction post processing
    test_preds_post = np.round(np.clip(test_preds, 0.0, 1.0), 2)
    
    # Statistics of test predictions
    print("\n[Test Predictions Statistics]")
    print(f"Mean       : {test_preds_post.mean():.5f}")
    print(f"Std Dev    : {test_preds_post.std():.5f}")
    print(f"Min        : {test_preds_post.min():.5f}")
    print(f"Max        : {test_preds_post.max():.5f}")
    print(f"N Unique   : {len(np.unique(test_preds_post))}")
    print("="*50)
    
    # Save submission
    os.makedirs('result/v16', exist_ok=True)
    submit_path = 'result/v16/submit_v17-2_final.csv'
    submit = pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_preds_post
    })
    submit.to_csv(submit_path, index=False)
    print(f"\nFinal submission file saved to: {submit_path}")

if __name__ == '__main__':
    main()
