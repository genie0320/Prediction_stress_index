"""
스트레스 공식 역추적 분석
Steps 1-5: 파생 피처 생성 → R² 전수조사 → 다중선형 탐색 → 비선형 변환 → 잔차 분석
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────
df = pd.read_csv("/Users/admin/Documents/dev_src/stress_index/train.csv")

BASE_FEATS = ["age", "height", "weight", "cholesterol",
              "systolic_blood_pressure", "diastolic_blood_pressure",
              "glucose", "bone_density"]
TARGET = "stress_score"

# 결측 제거 (수치형 피처 + 타깃)
use_cols = BASE_FEATS + [TARGET]
df = df[use_cols].dropna()

y = df[TARGET].values
print(f"샘플 수: {len(df)}")
print(f"stress_score 기술통계:\n{df[TARGET].describe()}\n")

# ──────────────────────────────────────────
# Step 1: 파생 피처 생성
# ──────────────────────────────────────────
sbp = df["systolic_blood_pressure"]
dbp = df["diastolic_blood_pressure"]

df["BMI"]           = df["weight"] / (df["height"] / 100) ** 2
df["MAP"]           = (sbp + 2 * dbp) / 3
df["PP"]            = sbp - dbp
df["bmi_age"]       = df["BMI"] * df["age"]
df["glucose_chol"]  = df["glucose"] / df["cholesterol"]
df["glucose_chol2"] = df["glucose"] * df["cholesterol"]
df["map_age"]       = df["MAP"] * df["age"]
df["map_bmi"]       = df["MAP"] * df["BMI"]
df["bone_bmi"]      = df["bone_density"] * df["BMI"]
df["bone_age"]      = df["bone_density"] * df["age"]
df["sbp_glucose"]   = sbp * df["glucose"]
df["dbp_glucose"]   = dbp * df["glucose"]
df["age_bone"]      = df["age"] / df["bone_density"]
df["height_weight"] = df["height"] / df["weight"]

DERIVED_FEATS = ["BMI", "MAP", "PP", "bmi_age", "glucose_chol", "glucose_chol2",
                 "map_age", "map_bmi", "bone_bmi", "bone_age",
                 "sbp_glucose", "dbp_glucose", "age_bone", "height_weight"]

ALL_FEATS = BASE_FEATS + DERIVED_FEATS
print(f"전체 피처 수: {len(ALL_FEATS)} (원본 {len(BASE_FEATS)} + 파생 {len(DERIVED_FEATS)})\n")

# ──────────────────────────────────────────
# Step 2: 단일 피처 R² / 상관계수 전수조사
# ──────────────────────────────────────────
print("=" * 70)
print("STEP 2 — 단일 피처 R² / 상관계수 순위표")
print("=" * 70)

records = []
for feat in ALL_FEATS:
    x = df[feat].values
    # Pearson r
    r, p = pearsonr(x, y)
    # 단일 피처 LinearRegression R²
    X = x.reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    r2 = r2_score(y, lr.predict(X))
    records.append({"feature": feat, "pearson_r": round(r, 6), "R2": round(r2, 6),
                    "p_value": round(p, 8), "coef": round(lr.coef_[0], 8), "intercept": round(lr.intercept_, 6)})

step2 = pd.DataFrame(records).sort_values("R2", ascending=False).reset_index(drop=True)
step2.index += 1
print(step2.to_string())
print()

# ──────────────────────────────────────────
# Step 3: 다중 선형 조합 탐색
# ──────────────────────────────────────────
print("=" * 70)
print("STEP 3 — 다중 선형 조합 탐색")
print("=" * 70)

def fit_report(label, features, df_data, y_val):
    X = df_data[features].values
    lr = LinearRegression().fit(X, y_val)
    r2 = r2_score(y_val, lr.predict(X))
    coef_str = ", ".join([f"{f}={c:.6f}" for f, c in zip(features, lr.coef_)])
    print(f"\n[{label}]")
    print(f"  R²       = {r2:.6f}")
    print(f"  intercept= {lr.intercept_:.6f}")
    print(f"  coef     = {coef_str}")
    return r2, lr

# 3-A: 원본 8개
r2_base, lr_base = fit_report("원본 8개 피처", BASE_FEATS, df, y)

# 3-B: 파생 14개
r2_deriv, lr_deriv = fit_report("파생 14개 피처", DERIVED_FEATS, df, y)

# 3-C: 전체 22개
r2_all, lr_all = fit_report("전체 22개 피처", ALL_FEATS, df, y)

# 3-D: 상위 R² Top5 단일 피처 조합
top5 = step2.head(5)["feature"].tolist()
print(f"\n  [Top5 단일 R² 피처]: {top5}")
r2_top5, lr_top5 = fit_report("Top5 피처 조합", top5, df, y)

# best combo 선택
best_r2 = max(r2_base, r2_deriv, r2_all, r2_top5)
best_label = ["원본 8개", "파생 14개", "전체 22개", "Top5"][
    [r2_base, r2_deriv, r2_all, r2_top5].index(best_r2)]
best_feats = [BASE_FEATS, DERIVED_FEATS, ALL_FEATS, top5][
    [r2_base, r2_deriv, r2_all, r2_top5].index(best_r2)]
print(f"\n→ 최고 R²={best_r2:.6f} (조합: {best_label})")

# ──────────────────────────────────────────
# Step 4: 비선형 단조 변환 탐색
# ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — 비선형 단조 변환 탐색 (Top3 파생 피처)")
print("=" * 70)

# 파생 피처 중 R² 상위 3
derived_r2 = step2[step2["feature"].isin(DERIVED_FEATS)].head(3)["feature"].tolist()
print(f"대상 피처: {derived_r2}\n")

transform_records = []
for feat in derived_r2:
    x = df[feat].values
    for name, tx in [
        ("raw",       x),
        ("log(x)",    np.where(x > 0, np.log(x + 1e-9), np.nan)),
        ("x²",        x ** 2),
        ("sqrt(|x|)", np.sqrt(np.abs(x))),
        ("1/x",       np.where(np.abs(x) > 1e-9, 1.0 / x, np.nan)),
    ]:
        valid = ~np.isnan(tx)
        if valid.sum() < 10:
            continue
        X = tx[valid].reshape(-1, 1)
        y_v = y[valid]
        lr = LinearRegression().fit(X, y_v)
        r2 = r2_score(y_v, lr.predict(X))
        transform_records.append({"feature": feat, "transform": name, "R²": round(r2, 6)})

step4 = pd.DataFrame(transform_records).sort_values(["feature", "R²"], ascending=[True, False])
print(step4.to_string(index=False))

# ──────────────────────────────────────────
# Step 5: 잔차 분석
# ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — 잔차 분석 (최고 R² 조합 기준)")
print("=" * 70)

best_lr_map = {
    "원본 8개": lr_base, "파생 14개": lr_deriv,
    "전체 22개": lr_all, "Top5": lr_top5
}
best_lr = best_lr_map[best_label]

y_pred = best_lr.predict(df[best_feats].values)
residual = y - y_pred

print(f"\n조합: {best_label}")
print(f"잔차 분포:")
print(f"  mean = {residual.mean():.6f}")
print(f"  std  = {residual.std():.6f}")
print(f"  min  = {residual.min():.6f}")
print(f"  max  = {residual.max():.6f}")

print("\n잔차 vs 원본 8개 피처 Pearson r:")
resid_corr = []
for feat in BASE_FEATS:
    r, p = pearsonr(df[feat].values, residual)
    resid_corr.append({"feature": feat, "corr_with_residual": round(r, 6), "p_value": round(p, 8)})
resid_df = pd.DataFrame(resid_corr).sort_values("corr_with_residual", key=abs, ascending=False)
print(resid_df.to_string(index=False))

# ──────────────────────────────────────────
# 요약 판정
# ──────────────────────────────────────────
print("\n" + "=" * 70)
print("요약 판정")
print("=" * 70)
if best_r2 >= 0.5:
    print(f"✅ R²={best_r2:.4f} ≥ 0.5 → '{best_label}' 조합이 공식 뼈대 후보!")
    print(f"   피처: {best_feats}")
else:
    print(f"⚠️  최고 R²={best_r2:.4f} < 0.5 → 잔차 패턴 확인 필요")

print("\n분석 완료")
