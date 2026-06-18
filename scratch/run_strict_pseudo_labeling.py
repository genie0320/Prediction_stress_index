import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
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
    return len([i.strip() for i in str(x).split(',') if i.strip()])

def preprocess_data(df):
    data = df.copy()
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    data['activity_enc'] = data['activity'].map({'light':1,'moderate':2,'intense':3}).fillna(0).astype(int)
    data['edu_enc'] = data['edu_level'].map({'high school diploma':1,'bachelors degree':2,'graduate degree':3}).fillna(0).astype(int)
    data['sleep_enc'] = data['sleep_pattern'].map({'sleep difficulty':1,'normal':2,'oversleeping':3}).fillna(0).astype(int)
    data['gender_enc'] = data['gender'].map({'F':0,'M':1}).fillna(0).astype(int)
    data['smoke_enc'] = data['smoke_status'].map({'non-smoker':0,'ex-smoker':1,'current-smoker':2}).fillna(0).astype(int)
    data['n_medical'] = data['medical_history'].apply(count_items)
    data['n_family'] = data['family_medical_history'].apply(count_items)
    return data

FEATURES = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density',
    'bmi','pp','map_val',
    'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
    'n_medical','n_family'
]

SVR_PARAMS = {'kernel': 'rbf', 'C': 28.6022, 'gamma': 1.0566, 'epsilon': 0.0}
SEEDS = [42, 123, 456]
N_SPLITS = 10
NQ = 1365

def run_strict_pseudo_labeling(X, y, X_test, seeds):
    final_oof = np.zeros(len(X))
    final_test = np.zeros(len(X_test))
    
    for seed in seeds:
        seed_everything(seed)
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(X))
        seed_test = np.zeros(len(X_test))
        
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va       = X.iloc[va_idx]
            
            # --- 1. Train base model on fold ---
            scaler1 = StandardScaler()
            X_tr_s1 = scaler1.fit_transform(X_tr)
            X_te_s1 = scaler1.transform(X_test)
            
            qt1 = QuantileTransformer(n_quantiles=NQ, output_distribution='normal', random_state=42)
            y_tr_t1 = qt1.fit_transform(y_tr.values.reshape(-1,1)).flatten()
            
            model1 = SVR(**SVR_PARAMS)
            model1.fit(X_tr_s1, y_tr_t1)
            
            # --- 2. Generate STRICT pseudo labels for Test ---
            # This prediction ONLY saw y_tr, so it has 0 knowledge of y_va
            fold_pseudo_y = qt1.inverse_transform(model1.predict(X_te_s1).reshape(-1,1)).flatten()
            fold_pseudo_y = np.clip(fold_pseudo_y, 0, 1)
            
            # --- 3. Combine Data ---
            X_tr_combined = pd.concat([X_tr, X_test], axis=0, ignore_index=True)
            y_tr_combined = np.concatenate([y_tr.values, fold_pseudo_y])
            
            # --- 4. Retrain model on combined data ---
            scaler2 = StandardScaler()
            X_tr_s2 = scaler2.fit_transform(X_tr_combined)
            X_va_s2 = scaler2.transform(X_va)
            X_te_s2 = scaler2.transform(X_test)
            
            qt2 = QuantileTransformer(n_quantiles=NQ, output_distribution='normal', random_state=42)
            y_tr_t2 = qt2.fit_transform(y_tr_combined.reshape(-1,1)).flatten()
            
            model2 = SVR(**SVR_PARAMS)
            model2.fit(X_tr_s2, y_tr_t2)
            
            # --- 5. Predict on Validation and Test ---
            va_pred = qt2.inverse_transform(model2.predict(X_va_s2).reshape(-1,1)).flatten()
            te_pred = qt2.inverse_transform(model2.predict(X_te_s2).reshape(-1,1)).flatten()
            
            seed_oof[va_idx] = np.clip(va_pred, 0, 1)
            seed_test += np.clip(te_pred, 0, 1) / N_SPLITS
            
        final_oof += seed_oof / len(seeds)
        final_test += seed_test / len(seeds)
        
    return final_oof, final_test

def main():
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    train_p = preprocess_data(train)
    test_p  = preprocess_data(test)

    X = train_p[FEATURES]
    y = train_p['stress_score']
    X_test = test_p[FEATURES]

    print("="*50)
    print("Strict Pseudo-Labeling 학습 (Fold 내에서만 Pseudo 생성) 진행 중...")
    oof_pseudo, final_test_preds = run_strict_pseudo_labeling(X, y, X_test, SEEDS)
    
    pseudo_mae = mean_absolute_error(y, oof_pseudo)
    print(f"[Strict] Pseudo-Labeled OOF MAE: {pseudo_mae:.6f}")
    
    # 저장
    submit_df = pd.DataFrame({'ID': test['ID'], 'stress_score': np.round(final_test_preds, 2)})
    submit_df.to_csv('scratch/submit_strict_pseudo.csv', index=False)

if __name__ == '__main__':
    main()
