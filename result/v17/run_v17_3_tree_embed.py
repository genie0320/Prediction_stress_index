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
from datetime import datetime

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

def run_method_A(X, y, X_num, X_test, X_test_num, depth, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        X_tr_num = X_num.iloc[train_idx]
        X_va_num = X_num.iloc[val_idx]
        
        et = ExtraTreesRegressor(n_estimators=200, max_depth=depth, min_samples_leaf=5, random_state=42)
        et.fit(X_tr_num, y_tr)
        
        leaf_tr = et.apply(X_tr_num)
        leaf_va = et.apply(X_va_num)
        
        X_tr_embed = np.hstack([X_tr.values, leaf_tr])
        X_va_embed = np.hstack([X_va.values, leaf_va])
        
        if X_test is not None:
            leaf_te = et.apply(X_test_num)
            X_te_embed = np.hstack([X_test.values, leaf_te])
        else:
            X_te_embed = None
            
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_embed)
        X_va_scaled = scaler.transform(X_va_embed)
        if X_te_embed is not None:
            X_te_scaled = scaler.transform(X_te_embed)
            
        qt = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
        y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        model = SVR(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
        model.fit(X_tr_scaled, y_tr_trans)
        
        va_pred_trans = model.predict(X_va_scaled)
        va_pred = qt.inverse_transform(va_pred_trans.reshape(-1, 1)).flatten()
        oof_preds[val_idx] = va_pred
        
        if X_test is not None:
            te_pred_trans = model.predict(X_te_scaled)
            te_pred = qt.inverse_transform(te_pred_trans.reshape(-1, 1)).flatten()
            test_preds += te_pred / n_splits
            
    oof_mae = mean_absolute_error(y, oof_preds)
    return oof_mae, oof_preds, test_preds

def run_method_B(X, y, X_num, X_test, X_test_num, depth, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        X_tr_num = X_num.iloc[train_idx]
        X_va_num = X_num.iloc[val_idx]
        
        et = ExtraTreesRegressor(n_estimators=200, max_depth=depth, min_samples_leaf=5, random_state=42)
        et.fit(X_tr_num, y_tr)
        
        # tree.predict expects 2D, but let's be safe
        pred_tr = np.column_stack([tree.predict(X_tr_num.values.astype(np.float32)) for tree in et.estimators_])
        pred_va = np.column_stack([tree.predict(X_va_num.values.astype(np.float32)) for tree in et.estimators_])
        
        X_tr_embed = np.hstack([X_tr.values, pred_tr])
        X_va_embed = np.hstack([X_va.values, pred_va])
        
        if X_test is not None:
            pred_te = np.column_stack([tree.predict(X_test_num.values.astype(np.float32)) for tree in et.estimators_])
            X_te_embed = np.hstack([X_test.values, pred_te])
        else:
            X_te_embed = None
            
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_embed)
        X_va_scaled = scaler.transform(X_va_embed)
        if X_te_embed is not None:
            X_te_scaled = scaler.transform(X_te_embed)
            
        qt = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
        y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        model = SVR(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
        model.fit(X_tr_scaled, y_tr_trans)
        
        va_pred_trans = model.predict(X_va_scaled)
        va_pred = qt.inverse_transform(va_pred_trans.reshape(-1, 1)).flatten()
        oof_preds[val_idx] = va_pred
        
        if X_test is not None:
            te_pred_trans = model.predict(X_te_scaled)
            te_pred = qt.inverse_transform(te_pred_trans.reshape(-1, 1)).flatten()
            test_preds += te_pred / n_splits
            
    oof_mae = mean_absolute_error(y, oof_preds)
    return oof_mae, oof_preds, test_preds

def main():
    print(f"[{datetime.now()}] Started run_v17_3_tree_embed.py")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    
    num_features = [
        'age','height','weight','cholesterol',
        'systolic_blood_pressure','diastolic_blood_pressure',
        'glucose','bone_density'
    ]
    
    base_features = [
        'age','height','weight','cholesterol',
        'systolic_blood_pressure','diastolic_blood_pressure',
        'glucose','bone_density',
        'bmi','pp','map_val',
        'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
        'n_medical','n_family'
    ]
    
    y = train_proc['stress_score']
    X = train_proc[base_features]
    X_num = train_proc[num_features]
    
    X_test = test_proc[base_features]
    X_test_num = test_proc[num_features]
    
    print("[ET Depth 민감도 (방식 A, 5-Fold)]")
    depths = [5, 10, 15, 20, 30]
    best_depth = None
    best_depth_mae = float('inf')
    
    for d in depths:
        mae, _, _ = run_method_A(X, y, X_num, None, None, depth=d, n_splits=5)
        print(f"  depth={d:<2d}  -> OOF MAE: {mae:.6f}")
        if mae < best_depth_mae:
            best_depth_mae = mae
            best_depth = d
            
    print(f"  -> 최적 depth: {best_depth}\n")
    
    print("[방식 A: Leaf Node 번호 피처 (10-Fold)]")
    mae_A, oof_A, test_A = run_method_A(X, y, X_num, X_test, X_test_num, depth=best_depth, n_splits=10)
    oof_A_post = np.round(np.clip(oof_A, 0.0, 1.0), 2)
    mae_A_post = mean_absolute_error(y, oof_A_post)
    print(f"  OOF MAE (Raw)            : {mae_A:.6f}")
    print(f"  OOF MAE (Post-processed) : {mae_A_post:.6f}")
    print(f"  v16 기준값(0.13951) 대비  : {mae_A - 0.13951:+.6f}")
    print(f"  17-2 기준(0.13578) 대비   : {mae_A_post - 0.13578:+.6f}\n")
    
    print("[방식 B: 트리별 예측값 피처 (10-Fold)]")
    mae_B, oof_B, test_B = run_method_B(X, y, X_num, X_test, X_test_num, depth=best_depth, n_splits=10)
    oof_B_post = np.round(np.clip(oof_B, 0.0, 1.0), 2)
    mae_B_post = mean_absolute_error(y, oof_B_post)
    print(f"  OOF MAE (Raw)            : {mae_B:.6f}")
    print(f"  OOF MAE (Post-processed) : {mae_B_post:.6f}")
    print(f"  17-2 기준(0.13578) 대비   : {mae_B_post - 0.13578:+.6f}\n")
    
    better_method = "A" if mae_A_post < mae_B_post else "B"
    best_test_preds = test_A if better_method == "A" else test_B
    best_test_preds_post = np.round(np.clip(best_test_preds, 0.0, 1.0), 2)
    
    print(f"[Test 예측값 통계 (더 나은 방식 기준: {better_method})]")
    print(f"  Mean       : {best_test_preds_post.mean():.6f}")
    print(f"  Std Dev    : {best_test_preds_post.std():.6f}")
    print(f"  Min        : {best_test_preds_post.min():.6f}")
    print(f"  Max        : {best_test_preds_post.max():.6f}")
    print(f"  N Unique   : {len(np.unique(best_test_preds_post))}")
    print("="*50)
    
    os.makedirs('result/v16', exist_ok=True)
    submit_path = 'result/v16/submit_v17-3_tree_embed.csv'
    submit = pd.DataFrame({'ID': test['ID'], 'stress_score': best_test_preds_post})
    submit.to_csv(submit_path, index=False)
    print(f"Final submission saved to {submit_path}")

if __name__ == '__main__':
    main()
