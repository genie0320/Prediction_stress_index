import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")

# Load raw data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
test_path = "/Users/admin/Documents/dev_src/stress_index/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 1. Preprocessing as done by 1st place
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
    
    # We must match the dummies across train and test to avoid missing columns
    # Let's write the dummy extraction carefully. In their code, they did it separately,
    # but since train and test have the same unique values for these columns, it should match.
    mh_dummies    = pd.get_dummies(df['medical_history'], prefix="mh", dtype='int')
    fmh_dummies   = pd.get_dummies(df['family_medical_history'], prefix="fmh", dtype='int')
    smo_dummies   = pd.get_dummies(df['smoke_status'], prefix="smo", dtype='int')
    
    df = pd.concat([df, mh_dummies, fmh_dummies, smo_dummies], axis=1)
    
    cols_to_drop = ["ID", 'medical_history', 'family_medical_history', 'smoke_status']
    if is_train:
        cols_to_drop.append("stress_score")
        
    df = df.drop(cols_to_drop, axis=1)
    
    # Add BMI
    df['bmi'] = (df['weight'] / ((df['height'] / 100.0) ** 2)).round(2)
    return df

X = preprocess_df(train, is_train=True)
y = train['stress_score']
X_test = preprocess_df(test, is_train=False)

# Re-align test columns to match train columns exactly
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("Preprocessed shape:", X.shape)
print("Features:", X.columns.tolist())

# KFold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))

# The best parameters from the post
best_params = {'C': 3.963530707518144, 'gamma': 1.0631617004546035}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    pipe = make_pipeline(
        RobustScaler(),
        TransformedTargetRegressor(
            regressor=SVR(
                **best_params,
                kernel="rbf",
                epsilon=0.0
            ),
            transformer=QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, len(y_tr)),
                random_state=42
            )
        )
    )
    
    pipe.fit(X_tr, y_tr)
    oof[val_idx] = pipe.predict(X_val)
    
    fold_mae = mean_absolute_error(y_val, oof[val_idx])
    print(f"Fold {fold+1} MAE: {fold_mae:.6f}")

total_mae = mean_absolute_error(y, oof)
total_mae_clip = mean_absolute_error(y, np.clip(oof, 0.0, 1.0))
total_mae_round = mean_absolute_error(y, np.round(np.clip(oof, 0.0, 1.0), 2))

print(f"\nOverall OOF MAE (Raw)              : {total_mae:.6f}")
print(f"Overall OOF MAE (Clipped)          : {total_mae_clip:.6f}")
print(f"Overall OOF MAE (Clipped & Round 2): {total_mae_round:.6f}")
