import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import warnings

def main():
    # 경고 무시
    warnings.filterwarnings("ignore")
    
    # 스크립트 위치 기준으로 프로젝트 루트 디렉토리 결정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # 데이터 경로 설정
    train_path = os.path.join(project_root, "train.csv")
    test_path = os.path.join(project_root, "test.csv")
    
    print("1. 데이터 로딩 중...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    base_features = [
        "age", "height", "weight", "cholesterol", 
        "systolic_blood_pressure", "diastolic_blood_pressure", 
        "glucose", "bone_density"
    ]
    target = "stress_score"
    
    train_clean = train.dropna(subset=[target]).copy()
    
    X_train = train_clean[base_features].values
    y_train = train_clean[target].values
    X_test = test[base_features].values
    
    # KFold 설정
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 실험 설정 정의 (n_jobs=1로 고정하여 OpenMP 스레드 스케줄링 병목 해결)
    configs = {
        "[A] 극한 깊이": {
            "num_leaves": 2047,
            "max_depth": 15,
            "min_child_samples": 1,         # min_data_in_leaf
            "min_child_weight": 0.0,        # min_sum_hessian_in_leaf
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1                     # 싱글 스레드로 속도 극대화
        },
        "[B] 극한 깊이 + 낮은 learning rate": {
            "num_leaves": 2047,
            "max_depth": 15,
            "min_child_samples": 1,
            "learning_rate": 0.01,
            "n_estimators": 5000,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1
        },
        "[C] ExtraTrees 스타일 (랜덤 분할)": {
            "num_leaves": 2047,
            "max_depth": 15,
            "min_child_samples": 1,
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "extra_trees": True,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1
        },
        "[D] 중간 깊이": {
            "num_leaves": 511,
            "max_depth": 12,
            "min_child_samples": 1,
            "learning_rate": 0.05,
            "n_estimators": 2000,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1
        }
    }
    
    results = []
    
    print("\n2. LightGBM 극한 설정 실험 시작 (스레드 최적화 완료)...")
    global_start_time = time.time()
    
    for name, params in configs.items():
        print(f"\n==========================================")
        print(f"실험 시작: {name}")
        print(f"==========================================")
        
        oof_preds = np.zeros(len(train_clean))
        config_start_time = time.time()
        
        # 5-Fold OOF 측정
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            fold_start_time = time.time()
            print(f"  → Fold {fold+1}/5 학습 중...", end="", flush=True)
            
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val = X_train[val_idx]
            
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            
            oof_preds[val_idx] = model.predict(X_val)
            
            fold_elapsed = time.time() - fold_start_time
            fold_mae = mean_absolute_error(y_train[val_idx], oof_preds[val_idx])
            print(f" 완료! (소요시간: {fold_elapsed:.2f}초 | Fold MAE: {fold_mae:.6f})")
            
        config_elapsed = time.time() - config_start_time
        oof_mae = mean_absolute_error(y_train, oof_preds)
        print(f"  [결과] 5-Fold OOF MAE: {oof_mae:.6f} (총 소요시간: {config_elapsed:.2f}초)")
        
        # Train 전체 fit → Test 예측 std 측정
        print(f"  → 전체 데이터 학습 및 Test 예측 중...", end="", flush=True)
        fit_start_time = time.time()
        
        model_full = LGBMRegressor(**params)
        model_full.fit(X_train, y_train)
        test_preds = model_full.predict(X_test)
        test_std = np.std(test_preds)
        
        fit_elapsed = time.time() - fit_start_time
        print(f" 완료! (소요시간: {fit_elapsed:.2f}초 | Test 예측 Std: {test_std:.6f})")
        
        results.append({
            "설정": name,
            "OOF MAE": oof_mae,
            "Test 예측 Std": test_std,
            "소요시간(초)": round(config_elapsed + fit_elapsed, 1)
        })
        
    # 결과 비교표 출력
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("실험 결과 비교표")
    print("="*80)
    print(results_df.to_string(index=False))
    
    total_elapsed = time.time() - global_start_time
    print(f"\n전체 실험 완료! (총 누적 시간: {total_elapsed/60:.2f}분)")

if __name__ == "__main__":
    main()
