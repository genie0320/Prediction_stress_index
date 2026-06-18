import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
import warnings

def main():
    warnings.filterwarnings("ignore")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    train_path = os.path.join(project_root, "train.csv")
    test_path = os.path.join(project_root, "test.csv")
    
    print("1. 데이터 로드 및 전처리 시작...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 타깃 보존 및 제거
    y_train = train['stress_score'].values
    train_features = train.drop(columns=['stress_score'])
    
    # train과 test 결합하여 일관되게 원-핫 인코딩 수행
    combined = pd.concat([train_features, test], axis=0).reset_index(drop=True)
    
    # 성별 0/1 매핑
    d_gender = {"F" : 0, "M" : 1}
    combined['mean_working'] = combined['mean_working'].fillna(0)
    combined = combined.fillna('Unknown')
    combined['gender'] = combined['gender'].map(d_gender)
    
    # 모든 범주형/순서형 피처를 원-핫 인코딩하여 SVR이 직교적인 피처 가중치를 학습할 수 있도록 함
    mh_dummies    = pd.get_dummies(combined['medical_history'], prefix="mh", dtype='int')
    fmh_dummies   = pd.get_dummies(combined['family_medical_history'], prefix="fmh", dtype='int')
    smo_dummies   = pd.get_dummies(combined['smoke_status'], prefix="smo", dtype='int')
    sleep_dummies = pd.get_dummies(combined['sleep_pattern'], prefix="sleep", dtype='int')
    act_dummies   = pd.get_dummies(combined['activity'], prefix="act", dtype='int')
    edu_dummies   = pd.get_dummies(combined['edu_level'], prefix="edu", dtype='int')
    
    combined = pd.concat([combined, mh_dummies, fmh_dummies, smo_dummies, sleep_dummies, act_dummies, edu_dummies], axis=1)
    combined = combined.drop(["ID", "medical_history", "family_medical_history", "smoke_status", "sleep_pattern", "activity", "edu_level"], axis=1)
    
    # BMI 피처 생성
    combined['bmi'] = (combined['weight'] / ((combined['height'] / 100.0) ** 2)).round(2)
    # 수축기-이완기 혈압을 활용한 유용한 연속형 피처 엔지니어링 (맥압 및 평균 동맥압)
    combined['pulse_pressure'] = combined['systolic_blood_pressure'] - combined['diastolic_blood_pressure']
    combined['map'] = (combined['systolic_blood_pressure'] + 2 * combined['diastolic_blood_pressure']) / 3.0
    
    # 다시 train과 test 분리
    X_train = combined.iloc[:len(train)].copy()
    X_test = combined.iloc[len(train):].copy()
    
    print("전처리 완료 후 Train 피처 형태:", X_train.shape)
    print("피처 목록:", X_train.columns.tolist())
    
    # SVR 10-Fold CV 및 Multi-seed 앙상블 설정
    print("\n2. SVR 10-Fold 교차검증 및 모델 학습 시작...")
    
    # Optuna로 OHE 피처 세트에 대해 찾은 최고 파라미터 적용
    best_params = {
        'C': 3.5579149674791153,
        'gamma': 0.8347094999174096,
        'n_quantiles': 983
    }
    
    seeds = [42, 43, 44, 45, 46]
    oof_preds_list = []
    test_preds_list = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} 10-Fold 교차검증 시작 ---")
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        
        seed_oof = np.zeros(len(y_train))
        seed_test_preds = np.zeros(len(X_test))
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            # SVR 파이프라인 생성 (RobustScaler + SVR + QuantileTransformer)
            pipe = make_pipeline(
                RobustScaler(),
                TransformedTargetRegressor(
                    regressor=SVR(
                        C=best_params['C'],
                        gamma=best_params['gamma'],
                        kernel="rbf",
                        epsilon=0.0
                    ),
                    transformer=QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=min(best_params['n_quantiles'], len(y_tr)),
                        random_state=42
                    )
                )
            )
            
            pipe.fit(X_tr, y_tr)
            
            # 검증 폴드 예측
            seed_oof[val_idx] = pipe.predict(X_val)
            
            # 테스트 데이터 예측 누적
            seed_test_preds += pipe.predict(X_test) / 10.0
            
        seed_mae = mean_absolute_error(y_train, seed_oof)
        seed_mae_post = mean_absolute_error(y_train, np.round(np.clip(seed_oof, 0.0, 1.0), 2))
        print(f"  → Seed {seed} CV MAE (raw): {seed_mae:.6f} | (rounded): {seed_mae_post:.6f}")
        
        oof_preds_list.append(seed_oof)
        test_preds_list.append(seed_test_preds)
        
    # Multi-seed 평균 OOF 성능 측정
    avg_oof = np.mean(oof_preds_list, axis=0)
    avg_oof_rounded = np.round(np.clip(avg_oof, 0.0, 1.0), 2)
    final_cv_mae = mean_absolute_error(y_train, avg_oof_rounded)
    print(f"\n[전체] Multi-seed 앙상블 적용 후 OOF MAE: {final_cv_mae:.6f}")
    
    # 테스트 예측 평균 앙상블
    test_pred_mean = np.mean(test_preds_list, axis=0)
    test_pred_rounded = np.round(np.clip(test_pred_mean, 0.0, 1.0), 2)
    
    # 결과 저장 디렉토리 생성
    out_dir = os.path.join(project_root, "result", "v15")
    os.makedirs(out_dir, exist_ok=True)
    
    # 반올림 후처리 버전 저장
    out_path_rounded = os.path.join(out_dir, "submit_v15_svr_optuna.csv")
    sub_rounded = pd.DataFrame({
        "ID": test["ID"],
        "stress_score": test_pred_rounded
    })
    sub_rounded.to_csv(out_path_rounded, index=False)
    print(f"\n3. 제출 파일 저장 완료 (반올림 후처리): {out_path_rounded}")
    
    # Raw 버전 저장 (비교용)
    out_path_raw = os.path.join(out_dir, "submit_v15_svr_optuna_raw.csv")
    sub_raw = pd.DataFrame({
        "ID": test["ID"],
        "stress_score": np.clip(test_pred_mean, 0.0, 1.0)
    })
    sub_raw.to_csv(out_path_raw, index=False)
    print(f"   제출 파일 저장 완료 (Raw clipped): {out_path_raw}")
    
    # 예측 통계 출력
    print("\n=== Test 예측 데이터 요약 통계 (반올림) ===")
    print(sub_rounded["stress_score"].describe())
    print("Unique 값 개수:", sub_rounded["stress_score"].nunique())
    print("\n=== Test 예측 데이터 요약 통계 (Raw) ===")
    print(sub_raw["stress_score"].describe())
    print("Unique 값 개수:", sub_raw["stress_score"].nunique())

if __name__ == "__main__":
    main()
