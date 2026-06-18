import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

def build_tree_features(df):
    d = df.copy()
    
    # ─── 기초 파생 변수 ───
    d['bmi'] = d['weight'] / (d['height'] / 100)**2
    d['pp'] = d['systolic_blood_pressure'] - d['diastolic_blood_pressure']
    d['map_val'] = d['diastolic_blood_pressure'] + d['pp']/3
    
    # ─── 인코딩 (트리는 연속형처럼 취급해도 분기를 잘 치므로 단순 정수 인코딩) ───
    d['gender_enc'] = d['gender'].map({'M':0,'F':1}).fillna(-1)
    d['activity_enc'] = d['activity'].map({'light':0,'moderate':1,'intense':2}).fillna(-1)
    d['sleep_enc'] = d['sleep_pattern'].map({'normal':0,'oversleeping':1,'sleep difficulty':2}).fillna(-1)
    d['edu_enc'] = d['edu_level'].map({'high school diploma':0,'bachelors degree':1,'graduate degree':2}).fillna(-1)
    d['smoke_enc'] = d['smoke_status'].map({'non-smoker':0,'ex-smoker':1,'current-smoker':2}).fillna(-1)
    
    # ─── 병종별 플래그 ───
    d['has_hbp']      = d['medical_history'].str.contains('high blood pressure', na=False).astype(int)
    d['has_diabetes'] = d['medical_history'].str.contains('diabetes', na=False).astype(int)
    d['has_heart']    = d['medical_history'].str.contains('heart disease', na=False).astype(int)
    d['fam_hbp']      = d['family_medical_history'].str.contains('high blood pressure', na=False).astype(int)
    d['fam_diabetes'] = d['family_medical_history'].str.contains('diabetes', na=False).astype(int)
    d['fam_heart']    = d['family_medical_history'].str.contains('heart disease', na=False).astype(int)
    d['disease_count']     = d['has_hbp'] + d['has_diabetes'] + d['has_heart']
    d['fam_disease_count'] = d['fam_hbp'] + d['fam_diabetes'] + d['fam_heart']
    
    # ─── 분절 구간 (Threshold Features) ───
    d['hbp_stage1'] = (d['systolic_blood_pressure'] >= 140).astype(int)
    d['hbp_stage2'] = (d['systolic_blood_pressure'] >= 160).astype(int)
    d['dbp_high']   = (d['diastolic_blood_pressure'] >= 90).astype(int)
    d['glucose_prediab'] = (d['glucose'] >= 100).astype(int)
    d['glucose_diab']    = (d['glucose'] >= 126).astype(int)
    d['overweight'] = (d['bmi'] >= 25).astype(int)
    d['obese']      = (d['bmi'] >= 30).astype(int)
    d['chol_borderline'] = (d['cholesterol'] >= 200).astype(int)
    d['chol_high']       = (d['cholesterol'] >= 240).astype(int)
    d['age_senior']   = (d['age'] >= 65).astype(int)
    d['age_old']      = (d['age'] >= 75).astype(int)
    d['map_high'] = (d['map_val'] >= 100).astype(int)
    
    # ─── 교차 변수 (Interaction) ───
    d['intense_x_heart']   = (d['activity_enc']==2).astype(int) * d['has_heart']
    d['intense_x_hbp']     = (d['activity_enc']==2).astype(int) * d['has_hbp']
    d['light_x_heart']     = (d['activity_enc']==0).astype(int) * d['has_heart']
    d['sleepdiff_x_smoke'] = (d['sleep_enc']==2).astype(int) * (d['smoke_enc']==2).astype(int)
    d['senior_x_disease']  = d['age_senior'] * d['disease_count']
    d['hbp_x_diabetes']    = d['has_hbp'] * d['has_diabetes']
    
    # mean_working_filled는 interaction 용으로만 쓰고, 원본 mean_working(결측치 포함)은 그대로 보존
    mean_working_f = d['mean_working'].fillna(0)
    d['overwork']          = (mean_working_f >= 10).astype(int)
    d['overwork_x_sleep']  = d['overwork'] * (d['sleep_enc']==2).astype(int)
    d['is_retired'] = d['mean_working'].isna().astype(int)
    
    return d

feature_cols = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density','bmi','pp','map_val',
    'mean_working', # 결측치 그대로 (트리가 알아서 처리)
    'is_retired',
    'gender_enc','activity_enc','sleep_enc','edu_enc','smoke_enc',
    'has_hbp','has_diabetes','has_heart',
    'fam_hbp','fam_diabetes','fam_heart',
    'disease_count','fam_disease_count',
    'hbp_stage1','hbp_stage2','dbp_high',
    'glucose_prediab','glucose_diab',
    'overweight','obese','chol_borderline','chol_high',
    'age_senior','age_old','map_high',
    'intense_x_heart','intense_x_hbp','light_x_heart',
    'sleepdiff_x_smoke','senior_x_disease','hbp_x_diabetes',
    'overwork','overwork_x_sleep'
]

df_feat = build_tree_features(df)
X = df_feat[feature_cols].values
y = df['stress_score'].values # 변환 없이 순정 타겟값 사용

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LightGBM': LGBMRegressor(n_estimators=300, random_state=42, verbose=-1, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=300, random_state=42, n_jobs=-1, eval_metric='mae'),
    'CatBoost': CatBoostRegressor(n_estimators=300, random_state=42, verbose=0, thread_count=-1),
    'HistGBM': HistGradientBoostingRegressor(max_iter=300, random_state=42)
}

print("=== 트리 모델 벤치마크 (Raw Target, Base Parameters) ===")
print(f"사용 피처 개수: {len(feature_cols)}개")
print("-" * 50)

for name, model in models.items():
    maes = []
    for tr_idx, val_idx in kf.split(X):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        
        # 예측값 클리핑 (타겟이 0~1 사이이므로)
        pred = np.clip(pred, 0, 1)
        
        mae = mean_absolute_error(y_va, pred)
        maes.append(mae)
        
    print(f"{name:<12} | OOF MAE: {np.mean(maes):.5f} ± {np.std(maes):.5f}")

