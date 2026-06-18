import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")

# Load raw data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

# 1. Preprocessing as done by 1st place SVR
d_gender = {"F" : 0, "M" : 1}
d_activity = {"light" : 0, "moderate" : 1, "intense" : 2}
d_smoke_status = {"non-smoker" : 0, "ex-smoker" : 1, "current-smoker" : 2}
d_edu_level = {'high school diploma' : 1, 'bachelors degree' : 2, 'graduate degree' : 3, 'Unknown' : 0}
d_sleep_pattern = {'sleep difficulty' : 0, 'normal' : 1, 'oversleeping' : 2}

def preprocess_df(df_in, is_train=True):
    df = df_in.copy()
    
    df['mean_working'] = df['mean_working'].fillna(0)
    df = df.fillna('Unknown')
    
    df['gender'] = df['gender'].map(d_gender)
    df['activity'] = df['activity'].map(d_activity)
    df['smoke_status'] = df['smoke_status'].map(d_smoke_status)
    df['edu_level'] = df['edu_level'].map(d_edu_level)
    df['sleep_pattern'] = df['sleep_pattern'].map(d_sleep_pattern)
    
    mh_dummies    = pd.get_dummies(df['medical_history'], prefix="mh", dtype='int')
    fmh_dummies   = pd.get_dummies(df['family_medical_history'], prefix="fmh", dtype='int')
    smo_dummies   = pd.get_dummies(df['smoke_status'], prefix="smo", dtype='int')
    
    df = pd.concat([df, mh_dummies, fmh_dummies, smo_dummies], axis=1)
    
    cols_to_drop = ["ID", 'medical_history', 'family_medical_history', 'smoke_status']
    if is_train:
        cols_to_drop.append("stress_score")
        
    df = df.drop(cols_to_drop, axis=1)
    df['bmi'] = (df['weight'] / ((df['height'] / 100.0) ** 2)).round(2)
    return df

X = preprocess_df(train, is_train=True)
y = train['stress_score'].values

# KFold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

svr_oof = np.zeros(len(y))
et_oof = np.zeros(len(y))

# Best params
best_params_svr = {'C': 3.963530707518144, 'gamma': 1.0631617004546035}
best_params_et = {'criterion': 'absolute_error', 'max_features': 1.0, 'bootstrap': False, 'min_samples_leaf': 1}

print("Running 5-fold CV for SVR and ExtraTrees...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    # Train SVR
    pipe_svr = make_pipeline(
        RobustScaler(),
        TransformedTargetRegressor(
            regressor=SVR(**best_params_svr, kernel="rbf", epsilon=0.0),
            transformer=QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(y_tr)), random_state=42)
        )
    )
    pipe_svr.fit(X_tr, y_tr)
    svr_oof[val_idx] = pipe_svr.predict(X_val)
    
    # Train ExtraTrees (note: ET doesn't need RobustScaler but we train on preprocessed data)
    model_et = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1, **best_params_et)
    model_et.fit(X_tr, y_tr)
    et_oof[val_idx] = model_et.predict(X_val)
    
    print(f"  Fold {fold+1} complete.")

# Calculate individual MAE
svr_mae = mean_absolute_error(y, svr_oof)
et_mae = mean_absolute_error(y, et_oof)

print(f"\nSVR OOF MAE: {svr_mae:.6f}")
print(f"ET OOF MAE : {et_mae:.6f}")

# Check correlation between OOF predictions
from scipy.stats import pearsonr
r, _ = pearsonr(svr_oof, et_oof)
print(f"Correlation (SVR vs ET): {r:.6f}")

# Blending Grid Search
print("\n=== Blending Ratios OOF MAE ===")
best_blend_mae = 999.0
best_weight = None

for w_svr in np.linspace(0.0, 1.0, 11):
    w_et = 1.0 - w_svr
    blend = svr_oof * w_svr + et_oof * w_et
    # Clip and Round
    blend_post = np.round(np.clip(blend, 0.0, 1.0), 2)
    mae = mean_absolute_error(y, blend_post)
    print(f"SVR {w_svr:.1f} + ET {w_et:.1f} | MAE: {mae:.6f}")
    
    if mae < best_blend_mae:
        best_blend_mae = mae
        best_weight = (w_svr, w_et)

print(f"\nBest Blend: SVR {best_weight[0]:.1f} + ET {best_weight[1]:.1f} | MAE: {best_blend_mae:.6f}")
