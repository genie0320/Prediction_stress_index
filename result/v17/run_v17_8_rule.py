import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error, r2_score

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

def get_rank_feats(X_tr, X_val, X_te, num_feats):
    tr_ranks = X_tr[num_feats].rank(pct=True)
    val_ranks = pd.DataFrame(index=X_val.index)
    te_ranks = pd.DataFrame(index=X_te.index) if X_te is not None else None
    
    for feat in num_feats:
        sorted_tr = np.sort(X_tr[feat].values)
        val_ranks[f'rank_{feat}'] = np.searchsorted(sorted_tr, X_val[feat].values, side='right') / len(sorted_tr)
        if te_ranks is not None:
            te_ranks[f'rank_{feat}'] = np.searchsorted(sorted_tr, X_te[feat].values, side='right') / len(sorted_tr)
            
    tr_ranks.columns = [f'rank_{feat}' for feat in num_feats]
    return tr_ranks, val_ranks, te_ranks

def evaluate_fold(X_tr, y_tr, X_va, y_va, X_te=None):
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te) if X_te is not None else None
    
    qt = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
    y_tr_trans = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
    
    model = SVR(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
    model.fit(X_tr_sc, y_tr_trans)
    
    va_pred = qt.inverse_transform(model.predict(X_va_sc).reshape(-1, 1)).flatten()
    te_pred = qt.inverse_transform(model.predict(X_te_sc).reshape(-1, 1)).flatten() if X_te is not None else None
    
    return va_pred, te_pred

def run_method_cv(method_num, best_depth, X, y, X_test, feature_set, num_feats, rank_feats, n_splits=5, seed=42):
    seed_everything(seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        tr_ranks, va_ranks, te_ranks = get_rank_feats(X_tr, X_va, X_test, num_feats)
        
        dt = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
        dt.fit(tr_ranks, y_tr)
        
        if method_num == 1:
            va_pred = dt.predict(va_ranks)
            te_pred = dt.predict(te_ranks) if X_test is not None else None
            oof_preds[val_idx] = va_pred
            if X_test is not None: test_preds += te_pred / n_splits
            
        elif method_num == 2:
            tr_dt_pred = dt.predict(tr_ranks)
            va_dt_pred = dt.predict(va_ranks)
            
            X_tr_full = pd.concat([X_tr, tr_ranks], axis=1)
            X_va_full = pd.concat([X_va, va_ranks], axis=1)
            
            X_tr_full['dt_pred'] = tr_dt_pred
            X_va_full['dt_pred'] = va_dt_pred
            
            curr_feats = feature_set + ['dt_pred']
            
            if X_test is not None:
                te_dt_pred = dt.predict(te_ranks)
                X_te_full = pd.concat([X_test, te_ranks], axis=1)
                X_te_full['dt_pred'] = te_dt_pred
            else:
                X_te_full = None
                
            va_pred, te_pred = evaluate_fold(X_tr_full[curr_feats], y_tr, X_va_full[curr_feats], y_va, X_te_full[curr_feats] if X_te_full is not None else None)
            oof_preds[val_idx] = va_pred
            if X_test is not None: test_preds += te_pred / n_splits
            
        elif method_num == 3:
            tr_leaf = dt.apply(tr_ranks).reshape(-1, 1)
            va_leaf = dt.apply(va_ranks).reshape(-1, 1)
            
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_res = ohe.fit_transform(tr_leaf)
            tr_leaf_ohe = pd.DataFrame(ohe_res, index=X_tr.index, columns=[f'leaf_{i}' for i in range(ohe_res.shape[1])])
            va_leaf_ohe = pd.DataFrame(ohe.transform(va_leaf), index=X_va.index, columns=tr_leaf_ohe.columns)
            
            X_tr_full = pd.concat([X_tr, tr_ranks, tr_leaf_ohe], axis=1)
            X_va_full = pd.concat([X_va, va_ranks, va_leaf_ohe], axis=1)
            
            curr_feats = feature_set + list(tr_leaf_ohe.columns)
            
            if X_test is not None:
                te_leaf = dt.apply(te_ranks).reshape(-1, 1)
                te_leaf_ohe = pd.DataFrame(ohe.transform(te_leaf), index=X_test.index, columns=tr_leaf_ohe.columns)
                X_te_full = pd.concat([X_test, te_ranks, te_leaf_ohe], axis=1)
            else:
                X_te_full = None
                
            va_pred, te_pred = evaluate_fold(X_tr_full[curr_feats], y_tr, X_va_full[curr_feats], y_va, X_te_full[curr_feats] if X_te_full is not None else None)
            oof_preds[val_idx] = va_pred
            if X_test is not None: test_preds += te_pred / n_splits
            
    return oof_preds, test_preds

def extract_leaf_rules(dt, feature_names):
    tree_ = dt.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    leaves = []
    
    def recurse(node, path):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left child (<=)
            path_left = path + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], path_left)
            
            # Right child (>)
            path_right = path + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], path_right)
        else:
            leaves.append({
                'path': " & ".join(path),
                'n_samples': tree_.n_node_samples[node],
                'value': tree_.value[node][0][0]
            })

    recurse(0, [])
    
    leaves = sorted(leaves, key=lambda x: x['value'])
    print("\n[Leaf 요약]")
    print(f"{'조건':<70} {'샘플수':<8} {'평균stress'}")
    for leaf in leaves:
        print(f"{leaf['path']:<70} {leaf['n_samples']:<8} {leaf['value']:.4f}")

def main():
    seed_everything(42)
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    y = train_proc['stress_score']
    
    num_feats = ['age','height','weight','cholesterol',
                 'systolic_blood_pressure','diastolic_blood_pressure',
                 'glucose','bone_density']
    rank_feats = [f'rank_{c}' for c in num_feats]
    
    # Base 18 + 8 rank = 26 features for Method 2 & 3
    features_base = [
        'age','height','weight','cholesterol',
        'systolic_blood_pressure','diastolic_blood_pressure',
        'glucose','bone_density',
        'bmi','pp','map_val',
        'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
        'n_medical','n_family'
    ]
    feature_set = features_base + rank_feats
    
    print("[Step 2: depth별 성능]")
    print(f"{'depth':<6} {'train_R²':<10} {'train_MAE':<10} {'OOF_MAE':<10}")
    
    best_depth = 3
    best_oof_mae_dt = 999
    
    # Precompute full rank_train for quick DT eval
    rank_train = train_proc[num_feats].rank(pct=True)
    rank_train.columns = rank_feats
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for depth in [3, 4, 5, 6, 7, 8, 10]:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(rank_train, y)
        r2 = dt.score(rank_train, y)
        mae_train = mean_absolute_error(y, dt.predict(rank_train))
        
        # OOF MAE (Properly with split inside CV though for just DT, we can just use cross_val_predict)
        # Because rank_train is global, doing cross_val_predict directly on rank_train leaks a tiny bit 
        # but instruction says "oof = cross_val_predict(dt, rank_train, y_train, cv=kf)"
        oof = cross_val_predict(dt, rank_train, y, cv=kf)
        mae_oof = mean_absolute_error(y, np.clip(oof, 0, 1))
        
        print(f"{depth:<6} {r2:<10.4f} {mae_train:<10.4f} {mae_oof:<10.4f}")
        if mae_oof < best_oof_mae_dt:
            best_oof_mae_dt = mae_oof
            best_depth = depth
            
    print(f"→ 최적 depth: {best_depth} (OOF MAE 기준)")
    
    print("\n[Step 3: 트리 규칙]")
    dt_best = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    dt_best.fit(rank_train, y)
    rules = export_text(dt_best, feature_names=rank_feats)
    print(rules)
    extract_leaf_rules(dt_best, rank_feats)
    
    print("\n[Step 4: 방법별 5-Fold OOF MAE]")
    oof1, _ = run_method_cv(1, best_depth, train_proc, y, None, feature_set, num_feats, rank_feats)
    mae1 = mean_absolute_error(y, oof1)
    
    oof2, _ = run_method_cv(2, best_depth, train_proc, y, None, feature_set, num_feats, rank_feats)
    mae2 = mean_absolute_error(y, oof2)
    
    oof3, _ = run_method_cv(3, best_depth, train_proc, y, None, feature_set, num_feats, rank_feats)
    mae3 = mean_absolute_error(y, oof3)
    
    print(f"방법1 (트리 직접)    : {mae1:.6f}")
    print(f"방법2 (트리+SVR)    : {mae2:.6f}")
    print(f"방법3 (leaf+SVR)    : {mae3:.6f}")
    
    maes = {1: mae1, 2: mae2, 3: mae3}
    best_method = min(maes, key=maes.get)
    print(f"→ 최선: 방법{best_method}")
    
    print("\n[Step 5: 10-Fold 재검증]")
    oof_10, test_10 = run_method_cv(best_method, best_depth, train_proc, y, test_proc, feature_set, num_feats, rank_feats, n_splits=10, seed=42)
    
    baseline_mae = 0.13578
    oof_mae_raw_10 = mean_absolute_error(y, oof_10)
    imp_raw = baseline_mae - oof_mae_raw_10
    
    # We can print fold MAEs during run_method_cv, but since we didn't, let's just calculate them manually
    # or actually we didn't print them during 10-fold. Let's just print the overall result to save time, or modify run_method_cv to print.
    # Instruction says "Fold 01 MAE: ..." We will re-implement fold printing in the loop manually for 10-fold to exactly match the request format.
    
    kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_idx = 1
    for _, val_idx in kf10.split(train_proc, y):
        f_mae = mean_absolute_error(y.iloc[val_idx], oof_10[val_idx])
        print(f"Fold {fold_idx:02d} MAE: {f_mae:.5f}")
        fold_idx += 1
        
    print(f"10-Fold OOF MAE (Raw)            : {oof_mae_raw_10:.6f}")
    print(f"17-2 기준값                      : {baseline_mae}")
    print(f"개선폭                           : {imp_raw:+.6f}")
    
    print("\n[3-Seed 안정성]")
    seeds = [42, 777, 2026]
    seed_maes = []
    for s in seeds:
        oof_s, _ = run_method_cv(best_method, best_depth, train_proc, y, None, feature_set, num_feats, rank_feats, n_splits=5, seed=s)
        mae_s = mean_absolute_error(y, oof_s)
        print(f"seed={s:<4} → MAE: {mae_s:.6f}")
        seed_maes.append(mae_s)
        
    print(f"평균 MAE:   {np.mean(seed_maes):.6f}")
    print(f"Std:        {np.std(seed_maes):.6f}")
    
    test_preds_post = np.round(np.clip(test_10, 0.0, 1.0), 2)
    print("\n[Test 예측값 통계]")
    print(f"Mean       : {test_preds_post.mean():.6f}")
    print(f"Std Dev    : {test_preds_post.std():.6f}")
    print(f"Min        : {test_preds_post.min():.6f}")
    print(f"Max        : {test_preds_post.max():.6f}")
    print(f"N_unique   : {len(np.unique(test_preds_post))}")
    
    os.makedirs('result/v17', exist_ok=True)
    submit_path = 'result/v17/submit_v17-8_rule.csv'
    pd.DataFrame({
        'ID': test['ID'],
        'stress_score': test_preds_post
    }).to_csv(submit_path, index=False)

if __name__ == "__main__":
    main()
