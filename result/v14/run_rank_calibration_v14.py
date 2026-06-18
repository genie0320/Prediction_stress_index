import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

def main():
    # 스크립트 위치 기준으로 프로젝트 루트 디렉토리 결정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # 데이터 경로 설정
    train_path = os.path.join(project_root, "train.csv")
    test_path = os.path.join(project_root, "test.csv")
    out_dir = os.path.join(project_root, "result", "v14")
    
    # 디렉토리 생성
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. 데이터 로딩 중...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 결측치 처리 (v2 기반)
    train['medical_history'] = train['medical_history'].fillna('none')
    test['medical_history'] = test['medical_history'].fillna('none')

    train['family_medical_history'] = train['family_medical_history'].fillna('none')
    test['family_medical_history'] = test['family_medical_history'].fillna('none')

    train['edu_level'] = train['edu_level'].fillna('Unknown')
    test['edu_level'] = test['edu_level'].fillna('Unknown')

    # mean_working 결측치 처리
    train['is_working_missing'] = train['mean_working'].isnull().astype(int)
    test['is_working_missing'] = test['mean_working'].isnull().astype(int)

    train['age_group'] = (train['age'] // 10) * 10
    test['age_group'] = (test['age'] // 10) * 10

    age_working_medians = train.groupby('age_group')['mean_working'].median()
    overall_median = train['mean_working'].median()

    train['mean_working'] = train.apply(
        lambda row: age_working_medians.get(row['age_group'], overall_median) if pd.isnull(row['mean_working']) else row['mean_working'],
        axis=1
    )
    test['mean_working'] = test.apply(
        lambda row: age_working_medians.get(row['age_group'], overall_median) if pd.isnull(row['mean_working']) else row['mean_working'],
        axis=1
    )

    train = train.drop('age_group', axis=1)
    test = test.drop('age_group', axis=1)

    # 피처 엔지니어링 (v2 기반)
    for df in [train, test]:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['pulse_pressure'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
        df['map'] = (df['systolic_blood_pressure'] + 2 * df['diastolic_blood_pressure']) / 3
        df['is_extreme_overwork'] = (df['mean_working'] >= 12).astype(int)
        df['is_low_bone_density'] = (df['bone_density'] <= -1.0).astype(int)
        df['is_high_pulse_pressure'] = (df['pulse_pressure'] > 80).astype(int)

    # Ordinal Encoding
    edu_map = {'Unknown': 0, 'high school diploma': 1, 'bachelors degree': 2, 'graduate degree': 3}
    activity_map = {'light': 1, 'moderate': 2, 'intense': 3}

    for df in [train, test]:
        df['edu_level_encoded'] = df['edu_level'].map(edu_map)
        df['activity_encoded'] = df['activity'].map(activity_map)

    train = train.drop(columns=['edu_level', 'activity'])
    test = test.drop(columns=['edu_level', 'activity'])

    # Label Encoding for remaining categorical columns
    categorical_cols = ['gender', 'smoke_status', 'medical_history', 'family_medical_history', 'sleep_pattern']
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # 피처 목록
    features = [col for col in train.columns if col not in ['ID', 'stress_score']]
    X_train = train[features].values
    y_train = train['stress_score'].values
    X_test = test[features].values

    # Step 1: OOF 기반 Rank Calibration
    print("\n2. 5-Fold KFold 교차검증 진행 중 (ExtraTrees)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val = X_train[val_idx]
        
        model = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        
        oof_preds[val_idx] = model.predict(X_val)
        fold_mae = mean_absolute_error(y_train[val_idx], oof_preds[val_idx])
        print(f"  Fold {fold+1} MAE: {fold_mae:.6f}")

    # Calibration 적용
    oof_sorted_indices = np.argsort(oof_preds)
    calibrated_oof = np.zeros_like(oof_preds)
    calibrated_oof[oof_sorted_indices] = np.sort(y_train)

    mae_before = mean_absolute_error(y_train, oof_preds)
    mae_after = mean_absolute_error(y_train, calibrated_oof)

    print("\n=== OOF Calibration 결과 ===")
    print(f"ET 원본 OOF MAE      : {mae_before:.6f}")
    print(f"Calibrated OOF MAE   : {mae_after:.6f}")

    # Step 2: Test 예측에 적용
    print("\n3. 전체 Train 데이터로 학습 후 Test 예측 중...")
    model_full = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model_full.fit(X_train, y_train)
    test_preds = model_full.predict(X_test)

    # Test Calibration 적용
    test_sorted_indices = np.argsort(test_preds)
    test_calibrated_vals = np.round(np.linspace(0, 1, len(test)), 2)
    
    calibrated_test_preds = np.zeros_like(test_preds)
    calibrated_test_preds[test_sorted_indices] = test_calibrated_vals

    # 저장
    out_path = os.path.join(out_dir, "submit_v14_ET_calibrated.csv")
    sub = pd.DataFrame({
        "ID": test["ID"],
        "stress_score": calibrated_test_preds
    })
    sub.to_csv(out_path, index=False)
    print(f"\n4. 저장 완료: {out_path}")

    print("\n=== Test 예측값 통계 비교 ===")
    print("구분           | 원본 예측값  | Calibrated 예측값")
    print("-" * 50)
    print(f"Mean           | {np.mean(test_preds):.6f}     | {np.mean(calibrated_test_preds):.6f}")
    print(f"Std            | {np.std(test_preds):.6f}     | {np.std(calibrated_test_preds):.6f}")
    print(f"Min            | {np.min(test_preds):.6f}     | {np.min(calibrated_test_preds):.6f}")
    print(f"Max            | {np.max(test_preds):.6f}     | {np.max(calibrated_test_preds):.6f}")
    print(f"Unique values  | {len(np.unique(test_preds)):d}         | {len(np.unique(calibrated_test_preds)):d}")

if __name__ == "__main__":
    main()
