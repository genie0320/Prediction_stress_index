import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

def count_items(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    return len([i.strip() for i in str(x).split(',') if i.strip()])

def build_features(df):
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
    d['has_hbp']      = d['medical_history'].str.contains('high blood pressure', na=False).astype(int)
    d['has_diabetes'] = d['medical_history'].str.contains('diabetes', na=False).astype(int)
    d['has_heart']    = d['medical_history'].str.contains('heart disease', na=False).astype(int)
    d['fam_hbp']      = d['family_medical_history'].str.contains('high blood pressure', na=False).astype(int)
    d['fam_diabetes'] = d['family_medical_history'].str.contains('diabetes', na=False).astype(int)
    d['fam_heart']    = d['family_medical_history'].str.contains('heart disease', na=False).astype(int)
    d['disease_count']     = d['has_hbp'] + d['has_diabetes'] + d['has_heart']
    d['fam_disease_count'] = d['fam_hbp'] + d['fam_diabetes'] + d['fam_heart']
    d['hbp_stage1']      = (d['systolic_blood_pressure'] >= 140).astype(int)
    d['hbp_stage2']      = (d['systolic_blood_pressure'] >= 160).astype(int)
    d['dbp_high']        = (d['diastolic_blood_pressure'] >= 90).astype(int)
    d['glucose_prediab'] = (d['glucose'] >= 100).astype(int)
    d['glucose_diab']    = (d['glucose'] >= 126).astype(int)
    d['overweight']      = (d['bmi'] >= 25).astype(int)
    d['obese']           = (d['bmi'] >= 30).astype(int)
    d['chol_borderline'] = (d['cholesterol'] >= 200).astype(int)
    d['chol_high']       = (d['cholesterol'] >= 240).astype(int)
    d['age_senior']      = (d['age'] >= 65).astype(int)
    d['age_old']         = (d['age'] >= 75).astype(int)
    d['map_high']        = (d['map_val'] >= 100).astype(int)
    d['intense_x_heart']   = (d['activity_enc']==2).astype(int) * d['has_heart']
    d['intense_x_hbp']     = (d['activity_enc']==2).astype(int) * d['has_hbp']
    d['light_x_heart']     = (d['activity_enc']==0).astype(int) * d['has_heart']
    d['sleepdiff_x_smoke'] = (d['sleep_enc']==2).astype(int) * (d['smoke_enc']==2).astype(int)
    d['senior_x_disease']  = d['age_senior'] * d['disease_count']
    d['senior_x_retired']  = d['age_senior'] * d['is_retired']
    d['overwork']          = (d['mean_working_filled'] >= 10).astype(int)
    d['overwork_x_sleep']  = d['overwork'] * (d['sleep_enc']==2).astype(int)
    d['hbp_x_diabetes']    = d['has_hbp'] * d['has_diabetes']
    
    # ─── v17-15에서 사용하던 n_medical, n_family 재현 ───
    d['n_medical'] = d['medical_history'].apply(count_items)
    d['n_family'] = d['family_medical_history'].apply(count_items)
    
    return d

feature_v2_cols = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density','bmi','pp','map_val',
    'mean_working_filled','is_retired',
    'gender_enc','activity_enc','sleep_enc','edu_enc','smoke_enc',
    'has_hbp','has_diabetes','has_heart',
    'fam_hbp','fam_diabetes','fam_heart',
    'disease_count','fam_disease_count',
    'hbp_stage1','hbp_stage2','dbp_high',
    'glucose_prediab','glucose_diab',
    'overweight','obese','chol_borderline','chol_high',
    'age_senior','age_old','map_high',
    'intense_x_heart','intense_x_hbp','light_x_heart',
    'sleepdiff_x_smoke','senior_x_disease','senior_x_retired',
    'overwork','overwork_x_sleep','hbp_x_diabetes',
    'n_medical', 'n_family'
]

df_feat = build_features(df)
y = df['stress_score'].values

# ── KFold seed를 기존 best와 맞춰야 공정 비교 ──
KF_SEED = 42  

kf = KFold(n_splits=10, shuffle=True, random_state=KF_SEED) # v17-15 uses 10 splits

SVR_PARAMS = dict(kernel='rbf', C=28.6022, epsilon=0.0, gamma=1.0566)

print("=== [방식 1] 요청하신 코드 스니펫 그대로 (X만 QT) ===")
pipe = Pipeline([
    ('qt', QuantileTransformer(n_quantiles=500, output_distribution='normal', random_state=42)),
    ('svr', SVR(**SVR_PARAMS))
])

X_v2 = df_feat[feature_v2_cols].values
scores_v2 = cross_val_score(pipe, X_v2, y, cv=kf, scoring='neg_mean_absolute_error')
print(f"feature_v2  : {-scores_v2.mean():.5f} ± {scores_v2.std():.5f}")

BEST_FEATURE_COLS = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density',
    'bmi','pp','map_val',
    'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
    'n_medical','n_family'
]

X_best = df_feat[BEST_FEATURE_COLS].values
scores_best = cross_val_score(pipe, X_best, y, cv=kf, scoring='neg_mean_absolute_error')
print(f"feature_best: {-scores_best.mean():.5f} ± {scores_best.std():.5f}")
print(f"개선폭       : {(-scores_best.mean()) - (-scores_v2.mean()):.5f}")


print("\n=== [방식 2] 기존 v17_15 파이프라인 완벽 재현 (X는 SS, y를 QT 변환) ===")
# v17은 X를 StandardScaler로, y를 QuantileTransformer(nq=1365)로 변환했음
qt_y = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
pipe_x = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(**SVR_PARAMS))
])
model_v17_exact = TransformedTargetRegressor(regressor=pipe_x, transformer=qt_y)

scores_v2_exact = cross_val_score(model_v17_exact, X_v2, y, cv=kf, scoring='neg_mean_absolute_error')
print(f"feature_v2 (v17 pipeline)  : {-scores_v2_exact.mean():.5f} ± {scores_v2_exact.std():.5f}")

scores_best_exact = cross_val_score(model_v17_exact, X_best, y, cv=kf, scoring='neg_mean_absolute_error')
print(f"feature_best (v17 pipeline): {-scores_best_exact.mean():.5f} ± {scores_best_exact.std():.5f}")
print(f"개선폭                     : {(-scores_best_exact.mean()) - (-scores_v2_exact.mean()):.5f}")
