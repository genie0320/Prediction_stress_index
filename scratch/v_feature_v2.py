import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def build_features(df):
    d = df.copy()
    
    # ─── 기존 피처 ───
    d['bmi'] = d['weight'] / (d['height']/100)**2
    d['pp'] = d['systolic_blood_pressure'] - d['diastolic_blood_pressure']
    d['map_val'] = d['diastolic_blood_pressure'] + d['pp']/3
    d['mean_working_filled'] = d['mean_working'].fillna(0)
    d['is_retired'] = d['mean_working'].isna().astype(int)
    
    # ─── 기존 인코딩 ───
    d['gender_enc'] = d['gender'].map({'M':0,'F':1}).fillna(0)
    d['activity_enc'] = d['activity'].map({'light':0,'moderate':1,'intense':2}).fillna(0)
    d['sleep_enc'] = d['sleep_pattern'].map({'normal':0,'oversleeping':1,'sleep difficulty':2}).fillna(0)
    d['edu_enc'] = d['edu_level'].map({'high school diploma':0,'bachelors degree':1,'graduate degree':2}).fillna(0)
    d['smoke_enc'] = d['smoke_status'].map({'non-smoker':0,'ex-smoker':1,'current-smoker':2}).fillna(0)
    
    # ─── [NEW] 병종별 binary flag ───
    d['has_hbp']     = d['medical_history'].str.contains('high blood pressure', na=False).astype(int)
    d['has_diabetes']= d['medical_history'].str.contains('diabetes', na=False).astype(int)
    d['has_heart']   = d['medical_history'].str.contains('heart disease', na=False).astype(int)
    d['fam_hbp']     = d['family_medical_history'].str.contains('high blood pressure', na=False).astype(int)
    d['fam_diabetes']= d['family_medical_history'].str.contains('diabetes', na=False).astype(int)
    d['fam_heart']   = d['family_medical_history'].str.contains('heart disease', na=False).astype(int)
    d['disease_count']= d['has_hbp'] + d['has_diabetes'] + d['has_heart']
    d['fam_disease_count']= d['fam_hbp'] + d['fam_diabetes'] + d['fam_heart']
    
    # ─── [NEW] 의학적 threshold features ───
    # 혈압 기준: JNC8 기준 (Stage 1: sys>=140, Stage 2: sys>=160)
    d['hbp_stage1'] = (d['systolic_blood_pressure'] >= 140).astype(int)
    d['hbp_stage2'] = (d['systolic_blood_pressure'] >= 160).astype(int)
    d['dbp_high']   = (d['diastolic_blood_pressure'] >= 90).astype(int)
    # 혈당: 공복혈당 당뇨전단계 100+, 당뇨 126+
    d['glucose_prediab'] = (d['glucose'] >= 100).astype(int)
    d['glucose_diab']    = (d['glucose'] >= 126).astype(int)
    # BMI: 과체중 25+, 비만 30+
    d['overweight'] = (d['bmi'] >= 25).astype(int)
    d['obese']      = (d['bmi'] >= 30).astype(int)
    # 콜레스테롤: 경계 200+, 고위험 240+
    d['chol_borderline'] = (d['cholesterol'] >= 200).astype(int)
    d['chol_high']       = (d['cholesterol'] >= 240).astype(int)
    # 고령: 65세+, 75세+
    d['age_senior']   = (d['age'] >= 65).astype(int)
    d['age_old']      = (d['age'] >= 75).astype(int)
    # MAP: 고위험 100+
    d['map_high'] = (d['map_val'] >= 100).astype(int)
    
    # ─── [NEW] Interaction features ───
    # 핵심: activity × disease 조합 (방향이 꺾이는 구조)
    d['intense_x_heart']    = (d['activity_enc']==2).astype(int) * d['has_heart']
    d['intense_x_hbp']      = (d['activity_enc']==2).astype(int) * d['has_hbp']
    d['light_x_heart']      = (d['activity_enc']==0).astype(int) * d['has_heart']
    # 수면장애 × 흡연
    d['sleepdiff_x_smoke']  = (d['sleep_enc']==2).astype(int) * (d['smoke_enc']==2).astype(int)
    # 고령 × 질환 수
    d['senior_x_disease']   = d['age_senior'] * d['disease_count']
    # 고령 × 은퇴
    d['senior_x_retired']   = d['age_senior'] * d['is_retired']
    # 장시간 노동 (10h+)
    d['overwork'] = (d['mean_working_filled'] >= 10).astype(int)
    d['overwork_x_sleep']   = d['overwork'] * (d['sleep_enc']==2).astype(int)
    # 혈압 high × 당뇨 comorbidity
    d['hbp_x_diabetes']     = d['has_hbp'] * d['has_diabetes']
    
    return d

feature_cols = [
    # 기본
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density',
    'bmi','pp','map_val',
    'mean_working_filled','is_retired',
    # 인코딩
    'gender_enc','activity_enc','sleep_enc','edu_enc','smoke_enc',
    # 병종별 flag
    'has_hbp','has_diabetes','has_heart',
    'fam_hbp','fam_diabetes','fam_heart',
    'disease_count','fam_disease_count',
    # threshold
    'hbp_stage1','hbp_stage2','dbp_high',
    'glucose_prediab','glucose_diab',
    'overweight','obese',
    'chol_borderline','chol_high',
    'age_senior','age_old','map_high',
    # interaction
    'intense_x_heart','intense_x_hbp','light_x_heart',
    'sleepdiff_x_smoke','senior_x_disease','senior_x_retired',
    'overwork','overwork_x_sleep','hbp_x_diabetes',
]

df_feat = build_features(df)
X = df_feat[feature_cols].values
y = df['stress_score'].values

# SVR(RBF) — 기존과 동일 세팅으로 공정 비교
pipe = Pipeline([
    ('qt', QuantileTransformer(n_quantiles=500, output_distribution='normal', random_state=42)),
    ('svr', SVR(kernel='rbf', C=5, epsilon=0.05, gamma='scale'))
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=kf, scoring='neg_mean_absolute_error')
print(f"SVR(RBF) + feature_v2  OOF MAE: {-scores.mean():.5f} ± {scores.std():.5f}")

# 비교: 기존 피처만으로도 같은 세팅
base_cols = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density','bmi','pp','map_val',
    'mean_working_filled','is_retired',
    'gender_enc','activity_enc','sleep_enc','edu_enc','smoke_enc',
]
df_base = build_features(df)
X_base = df_base[base_cols].values
scores_base = cross_val_score(pipe, X_base, y, cv=kf, scoring='neg_mean_absolute_error')
print(f"SVR(RBF) + feature_base OOF MAE: {-scores_base.mean():.5f} ± {scores_base.std():.5f}")

print(f"\n개선: {(-scores_base.mean() - (-scores.mean())):.5f}")
