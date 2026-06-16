# 1. 사전 환경 준비
# !pip install shap wandb optuna-integration python-dotenv


import os
import wandb
from dotenv import load_dotenv

# 2. W&B 로그인 (env 파일에 저장된 API 키 로드)
load_dotenv()
wandb_key = os.getenv('WANDB_API_KEY')
if wandb_key and wandb_key != 'your_wandb_api_key_here':
    wandb.login(key=wandb_key)
    print("W&B Logged in successfully!")
else:
    print("W&B API key not found in .env. Please set it!")
    wandb.login() # Interactive prompt fallback


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data_dir = '.' if os.path.exists('train.csv') else '../..'
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
print("Train shape:", train.shape, "Test shape:", test.shape)


# ==========================================
# 3. 결측치 처리 및 Data-Centric 보강
# ==========================================
for df in [train, test]:
    df['medical_history'] = df['medical_history'].fillna('none')
    df['family_medical_history'] = df['family_medical_history'].fillna('none')
    df['edu_level'] = df['edu_level'].fillna('Unknown')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# IterativeImputer를 사용하여 수치형 변수 기반 mean_working 대치 (Train으로만 fit)
numeric_cols_for_imputation = ['age', 'height', 'weight', 'cholesterol', 'systolic_blood_pressure', 
                               'diastolic_blood_pressure', 'glucose', 'bone_density', 'mean_working']

imputer = IterativeImputer(random_state=42, max_iter=20)

# Data Leakage 방지를 위해 Train 데이터로만 Imputer를 학습시킴
train[numeric_cols_for_imputation] = imputer.fit_transform(train[numeric_cols_for_imputation])
test[numeric_cols_for_imputation] = imputer.transform(test[numeric_cols_for_imputation])

print("mean_working imputed with IterativeImputer.")


# ==========================================
# 4. 파생 변수 추가 (원본 변수는 절대 삭제하지 않음)
# ==========================================
for df in [train, test]:
    # disease_count 파생 변수 (희소성 보완)
    df['medical_disease_count'] = df['medical_history'].apply(lambda x: 0 if x == 'none' else len(str(x).split(',')))
    df['family_disease_count'] = df['family_medical_history'].apply(lambda x: 0 if x == 'none' else len(str(x).split(',')))
    
    # 비선형 생체 지표 도메인 파생 변수 추가
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['systolic_blood_pressure'] - df['diastolic_blood_pressure']
    df['map'] = (df['systolic_blood_pressure'] + 2 * df['diastolic_blood_pressure']) / 3
    df['glucose_cholesterol_ratio'] = df['glucose'] / (df['cholesterol'] + 1)
    df['overwork_and_poor_sleep'] = ((df['mean_working'] >= 12) & (df['sleep_pattern'] == 'sleep difficulty')).astype(int)
    df['vascular_bone_risk'] = ((df['bone_density'] <= -1.0) & (df['pulse_pressure'] > 80)).astype(int)

# 복합 질환(medical_history) 동적 이진화 플래그 생성 (train 기준으로 추출)
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

# Ordinal Encoding 매핑
edu_map = {'Unknown': 0, 'high school diploma': 1, 'bachelors degree': 2, 'graduate degree': 3}
activity_map = {'light': 1, 'moderate': 2, 'intense': 3}
for df in [train, test]:
    df['edu_level_encoded'] = df['edu_level'].map(edu_map)
    df['activity_encoded'] = df['activity'].map(activity_map)

# 문자열 데이터는 Label Encoding
categorical_cols = ['gender', 'smoke_status', 'sleep_pattern', 'activity', 'edu_level', 'medical_history', 'family_medical_history']
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

print("Feature Engineering Complete. Train shape:", train.shape)


x_train = train.drop(['ID', 'stress_score'], axis=1)
y_train = train['stress_score']
x_test = test.drop('ID', axis=1)


# ==========================================
# 5. SHAP 기반 노이즈 제어 (최하위 3개 제명)
# ==========================================
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

print("Calculating SHAP values for baseline model feature selection...")

# Baseline LightGBM 학습
baseline_model = lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbose=-1, n_jobs=1)
baseline_model.fit(x_train, y_train)

# SHAP TreeExplainer 기여도 계산
explainer = shap.TreeExplainer(baseline_model)
shap_values = explainer.shap_values(x_train)

if isinstance(shap_values, list):
    shap_avg = np.abs(shap_values[0]).mean(axis=0)
else:
    shap_avg = np.abs(shap_values).mean(axis=0)

shap_imp = pd.Series(shap_avg, index=x_train.columns).sort_values(ascending=False)
print("\n--- SHAP Feature Importances (Absolute Mean) ---")
for feat, val in shap_imp.items():
    print(f"{feat}: {val:.6f}")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, x_train, plot_type='bar', show=False)
plt.tight_layout()
output_dir = 'result/v7' if os.path.exists('result/v7') else '.'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'shap_summary_plot_v7.png'))
plt.close()

# 최하위 3개 피처 자동 제거
N = 3
features_to_drop = list(shap_imp.tail(N).index)
print(f"\nRemoving bottom {N} features from dataset: {features_to_drop}")
x_train = x_train.drop(columns=[f for f in features_to_drop if f in x_train.columns], errors='ignore')
x_test = x_test.drop(columns=[f for f in features_to_drop if f in x_test.columns], errors='ignore')


# ==========================================
# 6. Optuna Tuning with W&B Callbacks (Dual-Track Heterogeneous)
# ==========================================
import optuna
from optuna_integration.wandb import WeightsAndBiasesCallback
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# 동적 컬럼 분류
cat_features = x_train.select_dtypes(include=['category']).columns.tolist()
num_features = x_train.select_dtypes(exclude=['category']).columns.tolist()
print(f"Categorical Track Features ({len(cat_features)}): {cat_features}")
print(f"Numerical Track Features ({len(num_features)}): {num_features}")

def tune_lgbm(x_train, y_train):
    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 1,
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores, mse_scores, r2_scores = [], [], []
        for train_idx, val_idx in kf.split(x_train, y_train):
            X_tr, y_tr = x_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = x_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = model.predict(X_val)
            
            mae_scores.append(mean_absolute_error(y_val, preds))
            mse_scores.append(mean_squared_error(y_val, preds))
            r2_scores.append(r2_score(y_val, preds))
        
        mean_mae = np.mean(mae_scores)
        mean_mse = np.mean(mse_scores)
        mean_rmse = np.sqrt(mean_mse)
        mean_r2 = np.mean(r2_scores)
        
        trial.set_user_attr("mse", mean_mse)
        trial.set_user_attr("rmse", mean_rmse)
        trial.set_user_attr("r2", mean_r2)
        wandb.log({"val_mae": mean_mae, "val_mse": mean_mse, "val_rmse": mean_rmse, "val_r2": mean_r2, "trial_number": trial.number})
        return mean_mae
    
    wandb_kwargs = {"project": "stress_index_v7", "name": "lgbm_tuning", "reinit": True}
    wandbc = WeightsAndBiasesCallback(metric_name="mae", wandb_kwargs=wandb_kwargs)
    
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, callbacks=[wandbc])
    wandb.finish()
    return study.best_params

def tune_xgboost(x_train, y_train):
    def objective(trial):
        params = {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': 1,
            'enable_categorical': True,
            'tree_method': 'hist',
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
            'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores, mse_scores, r2_scores = [], [], []
        for train_idx, val_idx in kf.split(x_train, y_train):
            X_tr, y_tr = x_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = x_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            
            mae_scores.append(mean_absolute_error(y_val, preds))
            mse_scores.append(mean_squared_error(y_val, preds))
            r2_scores.append(r2_score(y_val, preds))
        
        mean_mae = np.mean(mae_scores)
        mean_mse = np.mean(mse_scores)
        mean_rmse = np.sqrt(mean_mse)
        mean_r2 = np.mean(r2_scores)
        
        trial.set_user_attr("mse", mean_mse)
        trial.set_user_attr("rmse", mean_rmse)
        trial.set_user_attr("r2", mean_r2)
        wandb.log({"val_mae": mean_mae, "val_mse": mean_mse, "val_rmse": mean_rmse, "val_r2": mean_r2, "trial_number": trial.number})
        return mean_mae

    wandb_kwargs = {"project": "stress_index_v7", "name": "xgboost_tuning", "reinit": True}
    wandbc = WeightsAndBiasesCallback(metric_name="mae", wandb_kwargs=wandb_kwargs)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, callbacks=[wandbc])
    wandb.finish()
    return study.best_params

def tune_ridge(x_train, y_train):
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 100.0, log=True)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores, mse_scores, r2_scores = [], [], []
        for train_idx, val_idx in kf.split(x_train, y_train):
            X_tr, y_tr = x_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = x_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Dual-Track B 전처리 (오직 Ridge를 위해 K-Fold 내부에서 초기화 후 fit_transform)
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', RobustScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
                ])
            X_tr_ridge = preprocessor.fit_transform(X_tr)
            X_val_ridge = preprocessor.transform(X_val)
            
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_tr_ridge, y_tr)
            preds = model.predict(X_val_ridge)
            
            mae_scores.append(mean_absolute_error(y_val, preds))
            mse_scores.append(mean_squared_error(y_val, preds))
            r2_scores.append(r2_score(y_val, preds))
            
        mean_mae = np.mean(mae_scores)
        mean_mse = np.mean(mse_scores)
        mean_rmse = np.sqrt(mean_mse)
        mean_r2 = np.mean(r2_scores)
        
        trial.set_user_attr("mse", mean_mse)
        trial.set_user_attr("rmse", mean_rmse)
        trial.set_user_attr("r2", mean_r2)
        wandb.log({"val_mae": mean_mae, "val_mse": mean_mse, "val_rmse": mean_rmse, "val_r2": mean_r2, "trial_number": trial.number})
        return mean_mae

    wandb_kwargs = {"project": "stress_index_v7", "name": "ridge_tuning", "reinit": True}
    wandbc = WeightsAndBiasesCallback(metric_name="mae", wandb_kwargs=wandb_kwargs)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, callbacks=[wandbc])
    wandb.finish()
    return study.best_params

print("Tuning LightGBM...")
best_lgb = tune_lgbm(x_train, y_train)
print(f"Best LGBM Params: {best_lgb}")

print("Tuning XGBoost...")
best_xgb = tune_xgboost(x_train, y_train)
print(f"Best XGBoost Params: {best_xgb}")

print("Tuning Ridge Regression...")
best_ridge = tune_ridge(x_train, y_train)
print(f"Best Ridge Params: {best_ridge}")


# ==========================================
# 7. Seed Averaging & SLSQP Blending
# ==========================================
from scipy.optimize import minimize

def train_with_seeds(x_train, y_train, x_test, model_type, best_params, seeds=[42, 2026, 777]):
    oof_preds = np.zeros(len(x_train))
    test_preds = np.zeros(len(x_test))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
        X_tr, y_tr = x_train.iloc[train_idx].copy(), y_train.iloc[train_idx]
        X_val, y_val = x_train.iloc[val_idx].copy(), y_train.iloc[val_idx]
        
        X_test_fold = x_test.copy()
        
        # Dual-Track B: Ridge 모델의 경우 K-Fold 별 전처리 수행
        if model_type == 'ridge':
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', RobustScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
                ])
            X_tr = preprocessor.fit_transform(X_tr)
            X_val = preprocessor.transform(X_val)
            X_test_fold = preprocessor.transform(X_test_fold)
        
        fold_val_preds = np.zeros(len(X_val))
        fold_test_preds = np.zeros(len(x_test))
        
        for seed in seeds:
            if model_type == 'lgb':
                params = best_params.copy()
                params.update({'objective': 'regression_l1', 'metric': 'mae', 'random_state': seed, 'verbose': -1, 'n_jobs': 1})
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
            elif model_type == 'xgb':
                params = best_params.copy()
                params.update({'objective': 'reg:absoluteerror', 'eval_metric': 'mae', 'random_state': seed, 'verbosity': 0, 'enable_categorical': True, 'tree_method': 'hist', 'n_jobs': 1})
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif model_type == 'ridge':
                # Ridge doesn't use seed for its analytic solution, but uses it if solver uses random process.
                model = Ridge(alpha=best_params['alpha'], random_state=seed)
                model.fit(X_tr, y_tr)
                
            fold_val_preds += model.predict(X_val) / len(seeds)
            fold_test_preds += model.predict(X_test_fold) / len(seeds)
            
        oof_preds[val_idx] = fold_val_preds
        test_preds += fold_test_preds / 5
        
    mae = mean_absolute_error(y_train, oof_preds)
    print(f'[{model_type.upper()}] Seed Averaged OOF MAE: {mae:.6f}')
    return oof_preds, test_preds

oof_lgb, test_lgb = train_with_seeds(x_train, y_train, x_test, 'lgb', best_lgb)
oof_xgb, test_xgb = train_with_seeds(x_train, y_train, x_test, 'xgb', best_xgb)
oof_ridge, test_ridge = train_with_seeds(x_train, y_train, x_test, 'ridge', best_ridge)

oof_preds_list = [oof_lgb, oof_xgb, oof_ridge]

def mae_objective(weights):
    w = weights / np.sum(weights)
    blended_pred = (w[0] * oof_preds_list[0]) + (w[1] * oof_preds_list[1]) + (w[2] * oof_preds_list[2])
    return mean_absolute_error(y_train, blended_pred)

constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1), (0, 1), (0, 1)]
initial_weights = [1/3, 1/3, 1/3]

res = minimize(mae_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x / np.sum(res.x)

print(f"\nOptimal Blending Weights (LGB, XGB, RIDGE): {best_weights}")
blended_oof = (best_weights[0] * oof_lgb) + (best_weights[1] * oof_xgb) + (best_weights[2] * oof_ridge)
blended_oof_clipped = np.clip(blended_oof, 0.0, 1.0)
print(f"Final Blended Clipped OOF MAE: {mean_absolute_error(y_train, blended_oof_clipped):.6f}")

test_preds = (best_weights[0] * test_lgb) + (best_weights[1] * test_xgb) + (best_weights[2] * test_ridge)
test_preds = np.clip(test_preds, 0.0, 1.0)


submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
submission['stress_score'] = test_preds
output_dir = 'result/v7' if os.path.exists('result/v7') else '.'
submission.to_csv(os.path.join(output_dir, 'submit_07.csv'), index=False)
print("Submission saved to submit_07.csv")

