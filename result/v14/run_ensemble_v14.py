import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
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
    
    # 5-Fold 설정
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    et_params = {
        "n_estimators": 2000,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "random_state": 42,
        "n_jobs": -1
    }
    
    lgb_params = {
        "num_leaves": 2047,
        "max_depth": 15,
        "min_child_samples": 1,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "extra_trees": True,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 1  # 스레드 락 방지를 위해 싱글 스레드 유지
    }
    
    et_oof = np.zeros(len(train_clean))
    lgb_oof = np.zeros(len(train_clean))
    
    print("\n2. 각 모델 5-Fold OOF 예측값 생성 시작...")
    global_start = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        fold_start = time.time()
        print(f"\n--- Fold {fold+1}/5 ---")
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val = X_train[val_idx]
        
        # ExtraTrees
        print("  → ExtraTrees 학습 중...", end="", flush=True)
        et_start = time.time()
        et_model = ExtraTreesRegressor(**et_params)
        et_model.fit(X_tr, y_tr)
        et_oof[val_idx] = et_model.predict(X_val)
        print(f" 완료! (소요시간: {time.time() - et_start:.2f}초)")
        
        # LightGBM[C]
        print("  → LightGBM [C] 학습 중...", end="", flush=True)
        lgb_start = time.time()
        lgb_model = LGBMRegressor(**lgb_params)
        lgb_model.fit(X_tr, y_tr)
        lgb_oof[val_idx] = lgb_model.predict(X_val)
        print(f" 완료! (소요시간: {time.time() - lgb_start:.2f}초)")
        
        print(f"  Fold {fold+1} 완료! (총 소요시간: {time.time() - fold_start:.2f}초)")
        
    print(f"\n5-Fold OOF 생성 완료! (총 누적 시간: {(time.time() - global_start)/60:.2f}분)")
    
    # 각 모델 단독 MAE
    et_mae = mean_absolute_error(y_train, et_oof)
    lgb_mae = mean_absolute_error(y_train, lgb_oof)
    
    print("\n=== 개별 모델 OOF MAE ===")
    print(f"ExtraTrees OOF MAE  : {et_mae:.6f}")
    print(f"LightGBM[C] OOF MAE : {lgb_mae:.6f}")
    
    # Pearson r
    r_val, _ = pearsonr(et_oof, lgb_oof)
    print(f"\nPearson r (ET_oof vs LGBM_oof): {r_val:.6f}")
    
    # 앙상블 블렌딩 실험
    print("\n=== 앙상블 블렌딩 비율별 OOF MAE ===")
    ratios = [
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
        (0.8, 0.2)
    ]
    
    best_mae = 999.0
    best_ratio = None
    
    for w_et, w_lgb in ratios:
        blend_oof = et_oof * w_et + lgb_oof * w_lgb
        mae = mean_absolute_error(y_train, blend_oof)
        print(f"ET {w_et:.1f} + LGBM {w_lgb:.1f} | OOF MAE: {mae:.6f}")
        if mae < best_mae:
            best_mae = mae
            best_ratio = (w_et, w_lgb)
            
    print(f"\n최적 블렌드 비율: ET {best_ratio[0]:.1f} + LGBM {best_ratio[1]:.1f} (MAE: {best_mae:.6f})")
    
    # 전체 데이터 적합 및 test 예측 생성
    print("\n3. 전체 데이터로 최종 학습 및 Test 예측 시작...")
    
    # ExtraTrees 전체 학습
    print("  → ExtraTrees 전체 학습 중...", end="", flush=True)
    et_full = ExtraTreesRegressor(**et_params)
    et_full.fit(X_train, y_train)
    test_et = et_full.predict(X_test)
    print(" 완료!")
    
    # LightGBM 전체 학습
    print("  → LightGBM [C] 전체 학습 중...", end="", flush=True)
    lgb_full = LGBMRegressor(**lgb_params)
    lgb_full.fit(X_train, y_train)
    test_lgb = lgb_full.predict(X_test)
    print(" 완료!")
    
    # 블렌딩 적용
    w_et, w_lgb = best_ratio
    test_blend = test_et * w_et + test_lgb * w_lgb
    
    # 결과 저장
    out_dir = os.path.join(project_root, "result", "v14")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "submit_v14_blend_ET_LGBM.csv")
    
    sub = pd.DataFrame({
        "ID": test["ID"],
        "stress_score": test_blend
    })
    sub.to_csv(out_path, index=False)
    print(f"\n4. 저장 완료: {out_path}")
    
    # test 예측값 통계 출력
    print("\n=== 블렌딩 Test 예측값 통계 ===")
    print(f"Mean : {np.mean(test_blend):.6f}")
    print(f"Std  : {np.std(test_blend):.6f}")
    print(f"Min  : {np.min(test_blend):.6f}")
    print(f"Max  : {np.max(test_blend):.6f}")

if __name__ == "__main__":
    main()
