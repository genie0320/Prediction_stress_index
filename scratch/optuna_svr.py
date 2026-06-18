import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load raw data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

# Preprocessing
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

def objective(trial):
    # Suggest parameters
    scaler_name = trial.suggest_categorical("scaler", ["standard", "robust", "minmax"])
    C = trial.suggest_float("C", 0.1, 50.0, log=True)
    gamma = trial.suggest_float("gamma", 0.01, 10.0, log=True)
    n_quantiles = trial.suggest_int("n_quantiles", 100, 2000)
    
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()
        
    # KFold CV (10 splits)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    maes = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        pipe = make_pipeline(
            scaler,
            TransformedTargetRegressor(
                regressor=SVR(C=C, gamma=gamma, kernel="rbf", epsilon=0.0),
                transformer=QuantileTransformer(output_distribution="normal", n_quantiles=min(n_quantiles, len(y_tr)), random_state=42)
            )
        )
        
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        maes.append(mean_absolute_error(y_val, pred))
        
    return np.mean(maes)

print("Starting Optuna optimization (50 trials)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, n_jobs=1) # n_jobs=1 for stability, SVR can use multiple threads internally if needed

print("\n=== Optuna Optimization Complete ===")
print("Best 10-Fold CV MAE:", study.best_value)
print("Best Params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
