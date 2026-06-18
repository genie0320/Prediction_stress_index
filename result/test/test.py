import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["stress_score"]

num_cols = [
    "age",
    "height",
    "weight",
    "cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "glucose",
    "bone_density",
]

# ── M-1: NaN 행과 비NaN 행을 완전히 분리해서 분석 ──
nan_mask = train["mean_working"].isnull()
has_mask = ~nan_mask

print("[M-1] NaN 행 vs 비NaN 행 — Full Tree 각각 학습")

# 비NaN 행만
tr_has = train[has_mask].copy()
feats_has = num_cols + ["mean_working"]
X_has = tr_has[feats_has].values
y_has = tr_has["stress_score"].values

dt_has = DecisionTreeRegressor(max_depth=None, random_state=42)
dt_has.fit(X_has, y_has)
print(
    f"  비NaN 행 (n={len(tr_has)}): train MAE={mean_absolute_error(y_has, dt_has.predict(X_has)):.6f}"
)
print(f"  비NaN 행 R²={dt_has.score(X_has, y_has):.6f}")
print(f"  비NaN 행 리프 수: {dt_has.get_n_leaves()}")

# NaN 행만
tr_nan = train[nan_mask].copy()
X_nan = tr_nan[num_cols].values
y_nan = tr_nan["stress_score"].values

dt_nan = DecisionTreeRegressor(max_depth=None, random_state=42)
dt_nan.fit(X_nan, y_nan)
print(
    f"\n  NaN 행 (n={len(tr_nan)}): train MAE={mean_absolute_error(y_nan, dt_nan.predict(X_nan)):.6f}"
)
print(f"  NaN 행 R²={dt_nan.score(X_nan, y_nan):.6f}")
print(f"  NaN 행 리프 수: {dt_nan.get_n_leaves()}")

# ── M-2: 비NaN 행에서 mean_working을 제거하면 R² 얼마나 떨어지나? ──
print("\n[M-2] 비NaN 행에서 mean_working 제거 시 R²")
X_no_wk = tr_has[num_cols].values
dt_no = DecisionTreeRegressor(max_depth=None, random_state=42)
dt_no.fit(X_no_wk, y_has)
print(
    f"  mean_working 제거: R²={dt_no.score(X_no_wk, y_has):.6f}  "
    f"MAE={mean_absolute_error(y_has, dt_no.predict(X_no_wk)):.6f}"
)
print(f"  (R²=1.000이면 mean_working이 없어도 완전 결정됨)")
print(f"  (R²<1.000이면 mean_working이 공식의 핵심 인자)")

# ── M-3: 비NaN 행의 depth별 R² ──
print("\n[M-3] 비NaN 행 depth별 R² (mean_working 포함)")
for d in [5, 10, 15, 20, 25, 30, None]:
    dt2 = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt2.fit(X_has, y_has)
    r2 = dt2.score(X_has, y_has)
    mae = mean_absolute_error(y_has, np.clip(dt2.predict(X_has), 0, 1))
    label = str(d) if d else "None"
    print(f"  depth={label:5s}: R²={r2:.6f}  MAE={mae:.6f}")

# ── M-4: NaN 행의 depth별 R² ──
print("\n[M-4] NaN 행 depth별 R² (mean_working 없이 수치 8개만)")
for d in [5, 10, 15, 20, 25, 30, None]:
    dt3 = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt3.fit(X_nan, y_nan)
    r2 = dt3.score(X_nan, y_nan)
    mae = mean_absolute_error(y_nan, np.clip(dt3.predict(X_nan), 0, 1))
    label = str(d) if d else "None"
    print(f"  depth={label:5s}: R²={r2:.6f}  MAE={mae:.6f}")

# ── M-5: 비NaN 행에서 mean_working이 정수인가? 연속인가? ──
print("\n[M-5] mean_working 값 분포")
wk = train["mean_working"].dropna()
print(f"  unique 값 수: {wk.nunique()}")
print(f"  값 목록: {sorted(wk.unique())}")
print(f"  정수 여부: {(wk == wk.round()).all()}")
