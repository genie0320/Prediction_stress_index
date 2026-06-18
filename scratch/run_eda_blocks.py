import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Block 1
os.makedirs('output', exist_ok=True)
df = pd.read_csv('train.csv')

# 파생변수
df['bmi'] = df['weight'] / (df['height']/100)**2
df['pp'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
df['map_val'] = df['diastolic_blood_pressure'] + df['pp']/3

targets = ['age', 'glucose', 'cholesterol', 'map_val', 'bmi']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(targets):
    df[f'{col}_bin'] = pd.qcut(df[col], q=10, duplicates='drop')
    grp = df.groupby(f'{col}_bin', observed=False)['stress_score'].agg(['mean','std']).reset_index()
    ax = axes[i]
    ax.bar(range(len(grp)), grp['mean'], yerr=grp['std'], capsize=4, color='steelblue', alpha=0.7)
    ax.set_title(f'{col} decile vs stress_score')
    ax.set_xlabel('decile bin (low→high)')
    ax.set_ylabel('mean stress_score')
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([str(b) for b in grp[f'{col}_bin']], rotation=45, ha='right', fontsize=7)

axes[5].axis('off')
plt.tight_layout()
plt.savefig('output/eda_nonlinear_decile.png', dpi=150, bbox_inches='tight')
print("저장 완료: output/eda_nonlinear_decile.png\n")

# Block 2
df['smoke_simple'] = df['smoke_status'].map({'non-smoker':'non', 'ex-smoker':'ex', 'current-smoker':'current'})
df['sleep_simple'] = df['sleep_pattern'].map({'normal':'normal', 'oversleeping':'over', 'sleep difficulty':'diff'})

pivot = df.groupby(['activity', 'sleep_simple'], observed=False)['stress_score'].agg(['mean','count']).round(3)
print("=== activity × sleep_pattern ===")
print(pivot.to_string())

pivot2 = df.groupby(['activity', 'smoke_simple'], observed=False)['stress_score'].agg(['mean','count']).round(3)
print("\n=== activity × smoke_status ===")
print(pivot2.to_string())

df['has_medical'] = df['medical_history'].notna().astype(int)
df['has_family'] = df['family_medical_history'].notna().astype(int)

pivot3 = df.groupby(['activity', 'has_medical'], observed=False)['stress_score'].agg(['mean','std','count']).round(3)
print("\n=== activity × has_medical (핵심 interaction) ===")
print(pivot3.to_string())
print("\n")

# Block 3
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

test = pd.read_csv('test.csv')
df_tr = pd.read_csv('train.csv')

common_cols = [c for c in df_tr.columns if c in test.columns and c != 'ID']

combined = pd.concat([
    df_tr[common_cols].assign(is_train=1),
    test[common_cols].assign(is_train=0)
], ignore_index=True)

cat_cols = combined.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    combined[c] = LabelEncoder().fit_transform(combined[c].fillna('__missing__'))

combined = combined.fillna(-999)
X_lgb = combined.drop('is_train', axis=1)
y_lgb = combined['is_train']

clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
scores = cross_val_score(clf, X_lgb, y_lgb, cv=5, scoring='roc_auc')
print(f"Train vs Test 분류 AUC: {scores.mean():.4f} ± {scores.std():.4f}")
print("→ 0.5에 가까울수록 분포 동일, 0.7+ 이면 심각한 shift")

print("\n주요 변수 train/test 평균 비교:")
num_cols = df_tr[common_cols].select_dtypes(include='number').columns.tolist()
for c in num_cols:
    tr_mean = df_tr[c].mean()
    te_mean = test[c].mean() if c in test.columns else np.nan
    diff_pct = abs(tr_mean - te_mean) / (abs(tr_mean) + 1e-9) * 100
    print(f"  {c:30s}: train={tr_mean:.2f}, test={te_mean:.2f}, diff={diff_pct:.1f}%")
print("\n")

# Block 4
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')

df['bmi'] = df['weight'] / (df['height']/100)**2
df['pp'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
df['map_val'] = df['diastolic_blood_pressure'] + df['pp']/3
df['mean_working_filled'] = df['mean_working'].fillna(0)
df['is_retired'] = (df['mean_working'].isna()).astype(int)

encode_map = {
    'gender': {'M':0, 'F':1},
    'activity': {'light':0, 'moderate':1, 'intense':2},
    'sleep_pattern': {'normal':0, 'oversleeping':1, 'sleep difficulty':2},
    'edu_level': {'high school diploma':0, 'bachelors degree':1, 'graduate degree':2},
    'smoke_status': {'non-smoker':0, 'ex-smoker':1, 'current-smoker':2},
    'medical_history': {'high blood pressure':1, 'diabetes':2, 'heart disease':3},
    'family_medical_history': {'high blood pressure':1, 'diabetes':2, 'heart disease':3},
}
for col, mapping in encode_map.items():
    df[col+'_enc'] = df[col].map(mapping).fillna(0)

feature_cols = [
    'age','height','weight','cholesterol',
    'systolic_blood_pressure','diastolic_blood_pressure',
    'glucose','bone_density','bmi','pp','map_val',
    'mean_working_filled','is_retired',
    'gender_enc','activity_enc','sleep_pattern_enc',
    'edu_level_enc','smoke_status_enc',
    'medical_history_enc','family_medical_history_enc',
]

X_models = df[feature_cols].values
y_models = df['stress_score'].values

models = {
    'Ridge(SS)': Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=1.0))]),
    'Ridge(QT)': Pipeline([('sc', QuantileTransformer(n_quantiles=500, output_distribution='normal')), ('m', Ridge(alpha=1.0))]),
    'Lasso(QT)': Pipeline([('sc', QuantileTransformer(n_quantiles=500, output_distribution='normal')), ('m', Lasso(alpha=0.001))]),
    'ExtraTrees(d5)': ExtraTreesRegressor(n_estimators=200, max_depth=5, random_state=42),
    'ExtraTrees(d10)': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42),
    'RandomForest(d5)': RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
}

print("Model                   | OOF MAE")
print("-"*40)
for name, model in models.items():
    scores = cross_val_score(model, X_models, y_models, cv=5, scoring='neg_mean_absolute_error')
    print(f"{name:25s}| {-scores.mean():.5f} ± {scores.std():.5f}")
