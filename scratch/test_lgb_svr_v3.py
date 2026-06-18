import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

def count_items(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    return len([i.strip() for i in str(x).split(',') if i.strip()])

def build_all(df):
    d = df.copy()
    d['bmi'] = d['weight'] / (d['height']/100)**2
    d['pp'] = d['systolic_blood_pressure'] - d['diastolic_blood_pressure']
    d['map_val'] = d['diastolic_blood_pressure'] + d['pp']/3
    d['mean_working_filled'] = d['mean_working'].fillna(0)
    d['is_retired'] = d['mean_working'].isna().astype(int)
    d['gender_enc'] = d['gender'].map({'M':0,'F':1}).fillna(0)
    d['activity_enc'] = d['activity'].map({'light':0,'moderate':1,'intense':2}).fillna(0)
    d['sleep_enc'] = d['sleep_pattern'].map({'normal':0,'oversleeping':1,'sleep difficulty':2}).fillna(0)
    d['edu_enc'] = d['edu_level'].map({'high school diploma':0,'bachelors degree':1,'graduate degree':2}).fillna(0)
    d['smoke_enc'] = d['smoke_status'].map({'non-smoker':0,'ex-smoker':1,'current-smoker':2}).fillna(0)
    # 병종별 flag
    d['has_hbp']      = d['medical_history'].str.contains('high blood pressure', na=False).astype(int)
    d['has_diabetes'] = d['medical_history'].str.contains('diabetes', na=False).astype(int)
    d['has_heart']    = d['medical_history'].str.contains('heart disease', na=False).astype(int)
    d['fam_hbp']      = d['family_medical_history'].str.contains('high blood pressure', na=False).astype(int)
    d['fam_diabetes'] = d['family_medical_history'].str.contains('diabetes', na=False).astype(int)
    d['fam_heart']    = d['family_medical_history'].str.contains('heart disease', na=False).astype(int)
    d['disease_count']     = d['has_hbp'] + d['has_diabetes'] + d['has_heart']
    d['fam_disease_count'] = d['fam_hbp'] + d['fam_diabetes'] + d['fam_heart']
    # 연속형으로 표현 가능한 파생변수
    d['age_x_disease']  = d['age'] * d['disease_count']          # 고령+질환 정도를 연속형으로
    d['activity_x_disease'] = d['activity_enc'] * d['disease_count']
    d['working_x_sleep'] = d['mean_working_filled'] * d['sleep_enc']
    d['pp_x_age']       = d['pp'] * d['age'] / 100
    d['glucose_x_age']  = d['glucose'] * d['age'] / 1000
    d['bmi_x_age']      = d['bmi'] * d['age'] / 100
    
    d['n_medical'] = d['medical_history'].apply(count_items)
    d['n_family'] = d['family_medical_history'].apply(count_items)
    return d

df_feat = build_all(df)
y_raw = df['stress_score'].values

# y-QT 변환 (기존 v17 방식)
qt_y = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
y_trans = qt_y.fit_transform(y_raw.reshape(-1,1)).ravel()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ── 경로 1: SVR + 연속형 파생변수만 추가 ──
svr_cols_v3 = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density','bmi','pp','map_val',
    'mean_working_filled','is_retired',
    'gender_enc','activity_enc','sleep_enc','edu_enc','smoke_enc',
    # 연속형 파생만 추가 (binary 없음)
    'disease_count','fam_disease_count',
    'age_x_disease','activity_x_disease',
    'working_x_sleep','pp_x_age','glucose_x_age','bmi_x_age',
]

SVR_PARAMS = dict(kernel='rbf', C=28.6022, gamma=1.0566, epsilon=0.0)
pipe_svr = Pipeline([('sc', StandardScaler()), ('svr', SVR(**SVR_PARAMS))])

X_v3 = df_feat[svr_cols_v3].values
maes = []
for tr_idx, val_idx in kf.split(X_v3):
    pipe_svr.fit(X_v3[tr_idx], y_trans[tr_idx])
    pred_trans = pipe_svr.predict(X_v3[val_idx])
    pred_orig = qt_y.inverse_transform(pred_trans.reshape(-1,1)).ravel()
    pred_orig = np.clip(pred_orig, 0, 1)
    maes.append(np.mean(np.abs(pred_orig - y_raw[val_idx])))
print(f"SVR + 연속형파생(v3)  OOF MAE: {np.mean(maes):.5f} ± {np.std(maes):.5f}")

# ── 경로 2: LightGBM + feature_v2 전체 + y-QT ──
lgb_cols = svr_cols_v3 + [
    'has_hbp','has_diabetes','has_heart',
    'fam_hbp','fam_diabetes','fam_heart',
]

X_lgb = df_feat[lgb_cols].values

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.02,
    num_leaves=31, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
    n_jobs=-1
)

maes_lgb = []
maes_lgb_raw = []  # y-QT 없이 비교용
for tr_idx, val_idx in kf.split(X_lgb):
    # with y-QT
    lgb_model.fit(X_lgb[tr_idx], y_trans[tr_idx])
    pred_trans = lgb_model.predict(X_lgb[val_idx])
    pred_orig = qt_y.inverse_transform(pred_trans.reshape(-1,1)).ravel()
    pred_orig = np.clip(pred_orig, 0, 1)
    maes_lgb.append(np.mean(np.abs(pred_orig - y_raw[val_idx])))
    # without y-QT
    lgb_model.fit(X_lgb[tr_idx], y_raw[tr_idx])
    pred_raw = lgb_model.predict(X_lgb[val_idx])
    pred_raw = np.clip(pred_raw, 0, 1)
    maes_lgb_raw.append(np.mean(np.abs(pred_raw - y_raw[val_idx])))

print(f"LightGBM + y-QT       OOF MAE: {np.mean(maes_lgb):.5f} ± {np.std(maes_lgb):.5f}")
print(f"LightGBM (raw y)      OOF MAE: {np.mean(maes_lgb_raw):.5f} ± {np.std(maes_lgb_raw):.5f}")

# ── 기준선 재확인 ──
best_cols = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density',
    'bmi','pp','map_val',
    'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
    'n_medical','n_family'
]
if best_cols:
    maes_best = []
    for tr_idx, val_idx in kf.split(df_feat[best_cols].values):
        X_b = df_feat[best_cols].values
        pipe_svr.fit(X_b[tr_idx], y_trans[tr_idx])
        pred_t = pipe_svr.predict(X_b[val_idx])
        pred_o = qt_y.inverse_transform(pred_t.reshape(-1,1)).ravel()
        pred_o = np.clip(pred_o, 0, 1)
        maes_best.append(np.mean(np.abs(pred_o - y_raw[val_idx])))
    print(f"SVR best(v17)         OOF MAE: {np.mean(maes_best):.5f} ± {np.std(maes_best):.5f}")
