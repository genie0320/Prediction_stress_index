import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr

def main():
    # 스크립트 위치 기준으로 프로젝트 루트 디렉토리 결정 (결과적으로 /Users/admin/Documents/dev_src/stress_index)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # 데이터 경로 설정
    train_path = os.path.join(project_root, "train.csv")
    test_path = os.path.join(project_root, "test.csv")
    et_path = os.path.join(project_root, "result", "v12", "submit_v12_ET.csv")
    out_dir = os.path.join(project_root, "result", "v13")
    
    # 디렉토리 생성
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. 데이터 로딩 중...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    base_features = [
        "age", "height", "weight", "cholesterol", 
        "systolic_blood_pressure", "diastolic_blood_pressure", 
        "glucose", "bone_density"
    ]
    target = "stress_score"
    
    # 결측치 처리 및 데이터 추출
    train_clean = train.dropna(subset=[target]).copy()
    
    X_train = train_clean[base_features].values
    y_train = train_clean[target].values
    X_test = test[base_features].values
    
    # 2. 피처 표준화 (train 기준으로 fit)
    print("2. 피처 표준화(StandardScaler) 적용 중...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. KNN 모델 예측 (k = 1, 3, 5)
    print("\n3. KNN 모델 적합 및 예측 시작...")
    k_list = [1, 3, 5]
    predictions = {}
    
    print("\n" + "="*50)
    print("KNN k별 Test 예측 결과 통계")
    print("="*50)
    
    for k in k_list:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        pred = knn.predict(X_test_scaled)
        predictions[k] = pred
        
        mean_val = np.mean(pred)
        std_val = np.std(pred)
        min_val = np.min(pred)
        max_val = np.max(pred)
        n_unique = len(np.unique(pred))
        
        print(f"k={k:d} | Mean: {mean_val:.6f} | Std: {std_val:.6f} | Min: {min_val:.6f} | Max: {max_val:.6f} | 고유값 수(n_unique): {n_unique:d}")
    
    # 4. ExtraTrees(ET) 모델 예측값과의 상관계수 비교
    print("\n4. ExtraTrees(ET) 모델 예측값과의 상관계수 비교 중...")
    if os.path.exists(et_path):
        et_df = pd.read_csv(et_path)
        knn1_pred = predictions[1]
        et_pred = et_df["stress_score"].values
        
        r_val, _ = pearsonr(knn1_pred, et_pred)
        print(f"→ KNN k=1 vs ET Pearson r: {r_val:.6f}")
    else:
        print(f"[경고] ExtraTrees 결과 파일 ({et_path})을 찾을 수 없어 상관계수 비교를 건너뜁니다.")
            
    # 5. k=1 예측 결과 제출 파일로 저장
    out_path = os.path.join(out_dir, "submit_v13_knn1.csv")
    sub = pd.DataFrame({
        "ID": test["ID"],
        "stress_score": predictions[1]
    })
    sub.to_csv(out_path, index=False)
    print(f"\n5. 저장 완료: KNN k=1 예측 결과를 {out_path} 에 저장했습니다.")

if __name__ == "__main__":
    main()
