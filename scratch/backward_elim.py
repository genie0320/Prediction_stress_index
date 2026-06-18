import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import warnings; warnings.filterwarnings('ignore')

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

df = pd.read_csv('train.csv')
train_p = preprocess_data(df)

FEATURES = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density','bmi','pp','map_val',
    'activity_enc','edu_enc','sleep_enc','gender_enc','smoke_enc',
    'n_medical','n_family'
]

X = train_p[FEATURES]
y = train_p['stress_score']

# Fast CV settings to quickly check elimination
kf = KFold(n_splits=5, shuffle=True, random_state=42)
qt_y = QuantileTransformer(n_quantiles=1365, output_distribution='normal', random_state=42)
SVR_PARAMS = {'kernel': 'rbf', 'C': 28.6022, 'gamma': 1.0566, 'epsilon': 0.0}

def get_oof_mae(features_list):
    X_sub = train_p[features_list].values
    pipe_x = Pipeline([('scaler', StandardScaler()), ('svr', SVR(**SVR_PARAMS))])
    model = TransformedTargetRegressor(regressor=pipe_x, transformer=qt_y)
    scores = cross_val_score(model, X_sub, y.values, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
    return -np.mean(scores)

print("=== Backward Feature Elimination Test (SVR) ===")
base_mae = get_oof_mae(FEATURES)
print(f"Base (18 features) MAE: {base_mae:.5f}")
print("-" * 50)

elimination_results = {}
for feat in FEATURES:
    sub_feats = [f for f in FEATURES if f != feat]
    mae = get_oof_mae(sub_feats)
    diff = mae - base_mae
    elimination_results[feat] = diff
    status = "개선 (Improve)" if diff < 0 else "악화 (Worse)"
    print(f"Dropped '{feat:<25}': {mae:.5f} ({diff:+.5f}) -> {status}")
    
best_drop = min(elimination_results, key=elimination_results.get)
if elimination_results[best_drop] < 0:
    print(f"\n=> 결론: '{best_drop}' 피처를 제거하면 오차가 {elimination_results[best_drop]:.5f} 만큼 개선됩니다.")
else:
    print(f"\n=> 결론: 어떤 피처를 빼도 성능이 악화됩니다. 유저의 말씀대로 현재 18개 피처가 최적의 조합입니다.")
