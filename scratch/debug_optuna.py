import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.special import logit, expit
import os

data_dir = '.'
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

for df in [train, test]:
    df['medical_history'] = df['medical_history'].fillna('none')
    df['family_medical_history'] = df['family_medical_history'].fillna('none')
    df['edu_level'] = df['edu_level'].fillna('Unknown')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

numeric_cols_for_imputation = ['age', 'height', 'weight', 'cholesterol', 'systolic_blood_pressure', 
                               'diastolic_blood_pressure', 'glucose', 'bone_density', 'mean_working']

imputer = IterativeImputer(random_state=42, max_iter=20)
train[numeric_cols_for_imputation] = imputer.fit_transform(train[numeric_cols_for_imputation])
test[numeric_cols_for_imputation] = imputer.transform(test[numeric_cols_for_imputation])

for df in [train, test]:
    df['medical_disease_count'] = df['medical_history'].apply(lambda x: 0 if x == 'none' else len(str(x).split(',')))
    df['family_disease_count'] = df['family_medical_history'].apply(lambda x: 0 if x == 'none' else len(str(x).split(',')))
    
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
    df['map'] = (df['systolic_blood_pressure'] + 2 * df['diastolic_blood_pressure']) / 3
    df['glucose_cholesterol_ratio'] = df['glucose'] / (df['cholesterol'] + 1)
    df['overwork_and_poor_sleep'] = ((df['mean_working'] >= 12) & (df['sleep_pattern'] == 'sleep difficulty')).astype(int)
    df['vascular_bone_risk'] = ((df['bone_density'] <= -1.0) & (df['pulse_pressure'] > 80)).astype(int)

diseases = set()
for col in ['medical_history', 'family_medical_history']:
    for val in train[col].dropna().unique():
        for d in val.split(','):
            diseases.add(d.strip())
diseases.discard('none')
diseases = sorted(list(diseases))

for col, prefix in [('medical_history', 'med'), ('family_medical_history', 'fam')]:
    for disease in diseases:
        feat_name = f'{prefix}_{disease.lower().replace(" ", "_")}'
        train[feat_name] = train[col].apply(lambda x: 1 if disease in [d.strip() for d in x.split(",")] else 0)
        test[feat_name] = test[col].apply(lambda x: 1 if disease in [d.strip() for d in x.split(",")] else 0)

edu_map = {'Unknown': 0, 'high school diploma': 1, 'bachelors degree': 2, 'graduate degree': 3}
activity_map = {'light': 1, 'moderate': 2, 'intense': 3}
for df in [train, test]:
    df['edu_level_encoded'] = df['edu_level'].map(edu_map)
    df['activity_encoded'] = df['activity'].map(activity_map)

categorical_cols = ['gender', 'smoke_status', 'sleep_pattern', 'activity', 'edu_level', 'medical_history', 'family_medical_history']
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

x_train = train.drop(['ID', 'stress_score'], axis=1)
y_train = train['stress_score']
x_test = test.drop('ID', axis=1)

import shap
import lightgbm as lgb
baseline_model = lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbose=-1, n_jobs=1)
baseline_model.fit(x_train, y_train)
explainer = shap.TreeExplainer(baseline_model)
shap_values = explainer.shap_values(x_train)
if isinstance(shap_values, list):
    shap_avg = np.abs(shap_values[0]).mean(axis=0)
else:
    shap_avg = np.abs(shap_values).mean(axis=0)
shap_imp = pd.Series(shap_avg, index=x_train.columns).sort_values(ascending=False)
N = 3
features_to_drop = list(shap_imp.tail(N).index)
x_train = x_train.drop(columns=[f for f in features_to_drop if f in x_train.columns], errors='ignore')
x_test = x_test.drop(columns=[f for f in features_to_drop if f in x_test.columns], errors='ignore')

# Now run the optuna test directly for 1 trial to check for errors
import optuna
from sklearn.model_selection import KFold
import xgboost as xgb

epsilon = 1e-5
def mae_sigmoid_eval(y_true, y_pred):
    orig_preds = expit(y_pred)
    orig_labels = expit(y_true)
    mae = np.mean(np.abs(orig_preds - orig_labels))
    return mae

y_train_clipped = np.clip(y_train, epsilon, 1.0 - epsilon)
y_train_logit = logit(y_train_clipped)

def objective(trial):
    params = {
        'objective': 'reg:absoluteerror',
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': 1,
        'enable_categorical': True,
        'tree_method': 'hist',
        'n_estimators': 10,
        'eval_metric': mae_sigmoid_eval
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(x_train, y_train_logit):
        X_tr, y_tr_logit = x_train.iloc[train_idx], y_train_logit.iloc[train_idx]
        X_val, y_val_logit = x_train.iloc[val_idx], y_train_logit.iloc[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr_logit, eval_set=[(X_val, y_val_logit)], verbose=False)
    return 0.0

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1)
print("Optuna cell check: Success!")
