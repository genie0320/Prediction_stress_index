# result/v17/run_v17_13_nystroem_svr.py
# 목적: Nystroem 커널 근사 + LinearSVR으로 더 넓은 C 탐색
# Data Leakage 없음: fit은 train fold만, test는 transform만

import os, random, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def count_items(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    return len([i.strip() for i in str(x).split(',') if i.strip()])

def preprocess_data(df):
    data = df.copy()
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    data['activity_enc'] = data['activity'].map({'light':1,'moderate':2,'intense':3}).fillna(0).astype(int)
    data['edu_enc'] = data['edu_level'].map({'high school diploma':1,'bachelors degree':2,'graduate degree':3}).fillna(0).astype(int)
    data['sleep_enc'] = data['sleep_pattern'].map({'sleep difficulty':1,'normal':2,'oversleeping':3}).fillna(0).astype(int)
    data['gender_enc'] = data['gender'].map({'F':0,'M':1}).fillna(0).astype(int)
    data['smoke_enc'] = data['smoke_status'].map({'non-smoker':0,'ex-smoker':1,'current-smoker':2}).fillna(0).astype(int)
    data['n_medical'] = data['medical_history'].apply(count_items)
    data['n_family'] = data['family_medical_history'].apply(count_items)
    return data

FEATURES = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density',
    'bmi','pp','map_val',
    'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
    'n_medical','n_family'
]

# ── 탐색 파라미터 ──────────────────────────────────────────
# Nystroem: gamma=1.0566 (v17-2와 동일), n_components 탐색
# LinearSVR: C 탐색 (더 넓은 범위 가능)
CONFIGS = [
    # (n_components, C, label)
    (300,  28.60,  'baseline_nystroem'),
    (500,  28.60,  'n500_c28'),
    (1000, 28.60,  'n1000_c28'),
    (500,  50.0,   'n500_c50'),
    (500,  100.0,  'n500_c100'),
    (1000, 50.0,   'n1000_c50'),
    (1000, 100.0,  'n1000_c100'),
    (1000, 200.0,  'n1000_c200'),
]
SEEDS   = [42, 123, 456]
N_SPLITS = 10
GAMMA   = 1.0566   # v17-2와 동일
NQ      = 1365     # v17-2와 동일

def run_config(X, y, X_test, n_comp, C, seeds):
    """한 config에 대해 multi-seed 10-Fold 실행, OOF MAE와 test pred 반환"""
    seed_maes = []
    final_oof  = np.zeros(len(X))
    final_test = np.zeros(len(X_test))

    for seed in seeds:
        seed_everything(seed)
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof  = np.zeros(len(X))
        seed_test = np.zeros(len(X_test))

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va       = X.iloc[va_idx]

            # ── 스케일링 (train fold만 fit) ──
            scaler = RobustScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_test)

            # ── 타깃 변환 (train fold만 fit) ──
            qt = QuantileTransformer(n_quantiles=NQ,
                                     output_distribution='normal',
                                     random_state=42)
            y_tr_t = qt.fit_transform(y_tr.values.reshape(-1,1)).flatten()

            # ── Nystroem + LinearSVR ──
            nys = Nystroem(kernel='rbf', gamma=GAMMA,
                           n_components=n_comp, random_state=42)
            X_tr_k = nys.fit_transform(X_tr_s)   # train fold만 fit
            X_va_k = nys.transform(X_va_s)
            X_te_k = nys.transform(X_te_s)

            model = LinearSVR(C=C, max_iter=5000, random_state=42)
            model.fit(X_tr_k, y_tr_t)

            va_pred = qt.inverse_transform(
                model.predict(X_va_k).reshape(-1,1)).flatten()
            te_pred = qt.inverse_transform(
                model.predict(X_te_k).reshape(-1,1)).flatten()

            seed_oof[va_idx] = np.clip(va_pred, 0, 1)
            seed_test += np.clip(te_pred, 0, 1) / N_SPLITS

        s_mae = mean_absolute_error(y, seed_oof)
        seed_maes.append(s_mae)
        final_oof  += seed_oof  / len(seeds)
        final_test += seed_test / len(seeds)

    blended_mae = mean_absolute_error(y, final_oof)
    return blended_mae, np.mean(seed_maes), np.std(seed_maes), final_test

def main():
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    train_p = preprocess_data(train)
    test_p  = preprocess_data(test)

    X      = train_p[FEATURES]
    y      = train_p['stress_score']
    X_test = test_p[FEATURES]

    print("="*65)
    print(">>> [v17-13] Nystroem + LinearSVR 탐색")
    print(f"    Seeds: {SEEDS}, Folds: {N_SPLITS}")
    print(f"    gamma={GAMMA}, nq={NQ}")
    print("="*65)

    results = []
    best_mae  = 9999
    best_pred = None
    best_label = ''

    for n_comp, C, label in CONFIGS:
        print(f"\n▶ [{label}] n_comp={n_comp}, C={C} 실행 중...")
        blend_mae, mean_mae, std_mae, test_pred = run_config(
            X, y, X_test, n_comp, C, SEEDS)
        results.append((label, n_comp, C, blend_mae, mean_mae, std_mae))
        print(f"   Blended OOF MAE : {blend_mae:.6f}")
        print(f"   Seed Mean MAE   : {mean_mae:.6f} ± {std_mae:.6f}")
        if blend_mae < best_mae:
            best_mae   = blend_mae
            best_pred  = test_pred.copy()
            best_label = label

    # ── 결과 요약 ──
    print("\n" + "="*65)
    print(">>> 탐색 완료 — 결과 요약")
    print(f"{'Label':<22} {'n_comp':>7} {'C':>7} {'Blended':>10} {'Mean':>10} {'Std':>8}")
    print("-"*65)
    for label, nc, c, bm, mm, sm in sorted(results, key=lambda x: x[3]):
        marker = " ◀ BEST" if bm == best_mae else ""
        print(f"{label:<22} {nc:>7} {c:>7.1f} {bm:>10.6f} {mm:>10.6f} {sm:>8.6f}{marker}")

    v172_baseline = 0.13578
    diff = best_mae - v172_baseline
    diff_str = f"+{diff:.6f} (악화)" if diff > 0 else f"{diff:.6f} (개선)"
    print(f"\n v17-2 기준(0.13578) 대비: {diff_str}")
    print("="*65)

    # ── 최선 config 저장 ──
    if best_pred is not None:
        os.makedirs('result/v17', exist_ok=True)
        submit_pred = np.round(np.clip(best_pred, 0, 1), 2)
        pd.DataFrame({'ID': test['ID'], 'stress_score': submit_pred}) \
          .to_csv(f'result/v17/submit_v17-13_{best_label}.csv', index=False)
        print(f"\n저장 완료: result/v17/submit_v17-13_{best_label}.csv")
        print(f"[Test Stats] mean={submit_pred.mean():.5f}, "
              f"std={submit_pred.std():.5f}, "
              f"unique={len(np.unique(submit_pred))}")

if __name__ == '__main__':
    main()
