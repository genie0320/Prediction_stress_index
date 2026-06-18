import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error, r2_score
from collections import Counter

def main():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    
    feature_names = ['age', 'height', 'weight', 'cholesterol',
                     'systolic_blood_pressure', 'diastolic_blood_pressure',
                     'glucose', 'bone_density']
    
    X_num = train[feature_names]
    y = train['stress_score']
    
    print("\n" + "="*50)
    print("[Step 1: 단일 트리 분기 규칙]")
    print("="*50)
    for depth in [2, 3, 4, 5]:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_num, y)
        pred = dt.predict(X_num)
        r2 = r2_score(y, pred)
        mae = mean_absolute_error(y, pred)
        print(f"\n[depth={depth}] R²={r2:.6f}  MAE={mae:.6f}")
        print(export_text(dt, feature_names=feature_names))

    print("\n" + "="*50)
    print("[Step 2: 피처별 분기 임계값]")
    print("="*50)
    dt_10 = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt_10.fit(X_num, y)
    
    tree_ = dt_10.tree_
    feature_thresholds = {feat: [] for feat in feature_names}
    
    # Traverse tree
    for i in range(tree_.node_count):
        if tree_.children_left[i] != tree_.children_right[i]: # Not a leaf
            feat_idx = tree_.feature[i]
            if feat_idx != -2: # TREE_UNDEFINED is -2
                feat_name = feature_names[feat_idx]
                threshold = tree_.threshold[i]
                feature_thresholds[feat_name].append(threshold)
                
    print(f"{'피처명':<25} {'사용횟수':<10} {'주요 임계값 (빈도/값)'}")
    for feat in feature_names:
        thresholds = feature_thresholds[feat]
        count = len(thresholds)
        if count > 0:
            # 빈도수 순으로 정렬 후 상위 5개 추출
            thresh_counts = Counter([round(float(t), 2) for t in thresholds])
            top_5 = [float(v) for v, c in thresh_counts.most_common(5)]
            print(f"{feat:<25} {count:<10} {top_5}")
        else:
            print(f"{feat:<25} {count:<10} []")

    print("\n" + "="*50)
    print("[Step 3: 중복 행 분석]")
    print("="*50)
    dup = train.groupby(feature_names)['stress_score'].agg(['nunique', 'mean', 'std']).reset_index()
    # 중복 피처 조합 수: unique combination count where the combination appears more than once? 
    # Or just unique combinations? The prompt says "동일 피처값인데 다른 타깃인 행"
    # dup contains one row per unique feature combination.
    # We want to find feature combinations that have multiple rows in the original dataframe.
    
    # Calculate sizes
    combination_sizes = train.groupby(feature_names).size().reset_index(name='count')
    dup = pd.merge(dup, combination_sizes, on=feature_names)
    
    # Only consider combinations that actually have duplicates (count > 1)
    # Wait, the prompt says: dup = train.groupby(feature_names)['stress_score'].agg(['nunique','mean','std'])
    # "중복 피처 조합 수: XX" could mean the number of distinct feature combinations that appear >1 times.
    duplicated_combinations = dup[dup['count'] > 1]
    
    total_duplicate_combinations = len(duplicated_combinations)
    different_target = len(duplicated_combinations[duplicated_combinations['nunique'] > 1])
    same_target = len(duplicated_combinations[duplicated_combinations['nunique'] == 1])
    
    print(f"중복 피처 조합 수: {total_duplicate_combinations}")
    print(f"타깃이 다른 조합: {different_target}")
    print(f"타깃이 항상 동일: {same_target}")
    
    print("\n" + "="*50)
    print("[Step 4: 타깃 구간별 피처 평균]")
    print("="*50)
    # 0~1.0을 10개 구간으로
    bins = np.linspace(0, 1.0, 11)
    labels = [f"{bins[i]:.1f}~{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    train['target_bin'] = pd.cut(train['stress_score'], bins=bins, labels=labels, include_lowest=True)
    
    bin_stats = train.groupby('target_bin')[feature_names].mean().round(2)
    print(bin_stats.to_string())

if __name__ == '__main__':
    main()
