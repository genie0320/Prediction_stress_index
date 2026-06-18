import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    train = pd.read_csv('train.csv')
    features = ['age', 'height', 'weight', 'cholesterol', 'systolic_blood_pressure', 'diastolic_blood_pressure', 'glucose', 'bone_density']
    target = 'stress_score'

    # Step 1: 각 피처의 percentile rank 계산
    rank_df = pd.DataFrame()
    for feat in features:
        rank_df[feat] = train[feat].rank(pct=True)

    # Step 2: 순위값과 타깃의 상관관계 확인
    print("--- Step 2: 순위값과 타깃의 상관관계 확인 ---")
    print(f"{'피처명':<20} {'Pearson_r':<15} {'Spearman_r':<15}")
    for feat in features:
        p_r, _ = pearsonr(rank_df[feat], train[target])
        s_r, _ = spearmanr(rank_df[feat], train[target])
        print(f"{feat:<20} {p_r:<15.4f} {s_r:<15.4f}")

    # Step 3: 단순 가중합 탐색
    print("\n--- Step 3: 단순 가중합 탐색 ---")
    def objective(weights):
        pred = np.dot(rank_df.values, weights)
        return mean_absolute_error(train[target], pred)

    initial_weights = np.ones(len(features)) / len(features)
    result = minimize(objective, initial_weights, method='Powell')
    optimal_weights = result.x
    best_mae = result.fun

    print("최적 가중치:", np.round(optimal_weights, 5).tolist())
    print(f"train MAE: {best_mae:.5f}")

    # Step 4: 순위값으로 depth 트리 학습
    print("\n--- Step 4: 순위값으로 단일 트리 학습 ---")
    for depth in [10, 30]:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree.fit(rank_df, train[target])
        pred = tree.predict(rank_df)
        mae = mean_absolute_error(train[target], pred)
        r2 = r2_score(train[target], pred)
        print(f"depth={depth} 단일 트리 R²: {r2:.4f}, MAE: {mae:.5f}")

    # Step 5: stress_score 자체가 순위인지 확인
    print("\n--- Step 5: stress_score 자체가 순위인지 확인 ---")
    stress_rank = train[target].rank(pct=True)
    corr, _ = pearsonr(train[target], stress_rank)
    print(f"stress_score와 순위(rank)의 상관계수: {corr:.5f}")

    val_counts = train[target].value_counts().sort_index()
    unique_count = len(val_counts)
    mean_count = val_counts.mean()
    min_count = val_counts.min()
    max_count = val_counts.max()

    print(f"stress_score 고유값 수: {unique_count}")
    print(f"각 값의 평균 등장 횟수: {mean_count:.1f}")
    print(f"최소 등장 횟수: {min_count}")
    print(f"최대 등장 횟수: {max_count}")

    is_uniform = "균등" if val_counts.std() < (mean_count * 0.2) else "불균등"
    print(f"균등 여부: {is_uniform}")

if __name__ == "__main__":
    main()
