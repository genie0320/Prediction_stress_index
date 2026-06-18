# 1. 사전 환경 준비
# !pip install shap wandb optuna-integration python-dotenv


import os
import wandb
from dotenv import load_dotenv

# 2. W&B 로그인 (env 파일에 저장된 API 키 로드)
load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")
if wandb_key and wandb_key != "your_wandb_api_key_here":
    wandb.login(key=wandb_key)
    print("W&B Logged in successfully!")
else:
    print("W&B API key not found in .env. Please set it!")
    wandb.login()  # Interactive prompt fallback


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.special import logit, expit

data_dir = "." if os.path.exists("train.csv") else "../.."
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
print("Train shape:", train.shape, "Test shape:", test.shape)


# ==========================================
# 3. 결측치 처리 및 Data-Centric 보강
# ==========================================
for df in [train, test]:
    df["medical_history"] = df["medical_history"].fillna("none")
    df["family_medical_history"] = df["family_medical_history"].fillna("none")
    df["edu_level"] = df["edu_level"].fillna("Unknown")

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# IterativeImputer를 사용하여 수치형 변수 기반 mean_working 대치 (Train으로만 fit)
numeric_cols_for_imputation = [
    "age",
    "height",
    "weight",
    "cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "glucose",
    "bone_density",
    "mean_working",
]

imputer = IterativeImputer(random_state=42, max_iter=20)

# Data Leakage 방지를 위해 Train 데이터로만 Imputer를 학습시킴
train[numeric_cols_for_imputation] = imputer.fit_transform(
    train[numeric_cols_for_imputation]
)
test[numeric_cols_for_imputation] = imputer.transform(test[numeric_cols_for_imputation])

print("mean_working imputed with IterativeImputer.")


# ==========================================
# 4. 파생 변수 추가 (원본 변수는 절대 삭제하지 않음)
# ==========================================
for df in [train, test]:
    # disease_count 파생 변수 (희소성 보완)
    df["medical_disease_count"] = df["medical_history"].apply(
        lambda x: 0 if x == "none" else len(str(x).split(","))
    )
    df["family_disease_count"] = df["family_medical_history"].apply(
        lambda x: 0 if x == "none" else len(str(x).split(","))
    )

    # 비선형 생체 지표 도메인 파생 변수 추가
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["pulse_pressure"] = (
        df["systolic_blood_pressure"] - df["diastolic_blood_pressure"]
    )
    df["map"] = (df["systolic_blood_pressure"] + 2 * df["diastolic_blood_pressure"]) / 3
    df["glucose_cholesterol_ratio"] = df["glucose"] / (df["cholesterol"] + 1)
    df["overwork_and_poor_sleep"] = (
        (df["mean_working"] >= 12) & (df["sleep_pattern"] == "sleep difficulty")
    ).astype(int)
    df["vascular_bone_risk"] = (
        (df["bone_density"] <= -1.0) & (df["pulse_pressure"] > 80)
    ).astype(int)

# 복합 질환(medical_history) 동적 이진화 플래그 생성 (train 기준으로 추출)
diseases = set()
for col in ["medical_history", "family_medical_history"]:
    for val in train[col].dropna().unique():
        for d in val.split(","):
            diseases.add(d.strip())
diseases.discard("none")
diseases = sorted(list(diseases))

for col, prefix in [("medical_history", "med"), ("family_medical_history", "fam")]:
    for disease in diseases:
        feat_name = f'{prefix}_{disease.lower().replace(" ", "_")}'
        train[feat_name] = train[col].apply(
            lambda x: 1 if disease in [d.strip() for d in x.split(",")] else 0
        )
        test[feat_name] = test[col].apply(
            lambda x: 1 if disease in [d.strip() for d in x.split(",")] else 0
        )

# Ordinal Encoding 매핑
edu_map = {
    "Unknown": 0,
    "high school diploma": 1,
    "bachelors degree": 2,
    "graduate degree": 3,
}
activity_map = {"light": 1, "moderate": 2, "intense": 3}
for df in [train, test]:
    df["edu_level_encoded"] = df["edu_level"].map(edu_map)
    df["activity_encoded"] = df["activity"].map(activity_map)

# 문자열 데이터는 Label Encoding
categorical_cols = [
    "gender",
    "smoke_status",
    "sleep_pattern",
    "activity",
    "edu_level",
    "medical_history",
    "family_medical_history",
]
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

print("Feature Engineering Complete. Train shape:", train.shape)


x_train = train.drop(["ID", "stress_score"], axis=1)
y_train = train["stress_score"]
x_test = test.drop("ID", axis=1)


# ==========================================
# 5. SHAP 기반 노이즈 제어 (최하위 3개 제명)
# ==========================================
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

print("Calculating SHAP values for baseline model feature selection...")

# Baseline LightGBM 학습
baseline_model = lgb.LGBMRegressor(
    objective="regression_l1", random_state=42, verbose=-1, n_jobs=1
)
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
shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
plt.tight_layout()
output_dir = "result/v8" if os.path.exists("result/v8") else "."
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "shap_summary_plot_v8.png"))
plt.close()

# 최하위 3개 피처 자동 제거
N = 3
features_to_drop = list(shap_imp.tail(N).index)
print(f"\nRemoving bottom {N} features from dataset: {features_to_drop}")
x_train = x_train.drop(
    columns=[f for f in features_to_drop if f in x_train.columns], errors="ignore"
)
x_test = x_test.drop(
    columns=[f for f in features_to_drop if f in x_test.columns], errors="ignore"
)


# ==========================================
# 6. Optuna Tuning (XGBoost Single Hyper-Specialization with Logit-Sigmoid Target Transformation)
# ==========================================
import optuna
from optuna_integration.wandb import WeightsAndBiasesCallback
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

epsilon = 1e-5


# Custom Eval Metric for XGBoost inside K-Fold so that validation performance is evaluated in Sigmoid (Original) space.
def mae_sigmoid_eval(y_true, y_pred):
    orig_preds = expit(y_pred)
    orig_labels = expit(y_true)
    mae = np.mean(np.abs(orig_preds - orig_labels))
    return mae


def tune_xgboost(x_train, y_train):
    # Logit transformation of target variable y
    y_train_clipped = np.clip(y_train, epsilon, 1.0 - epsilon)
    y_train_logit = logit(y_train_clipped)

    def objective(trial):
        params = {
            "objective": "reg:absoluteerror",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": 1,
            "enable_categorical": True,
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-4, 10.0, log=True),
            # XGBoost v1.6+ 에서는 생성자(init) 레벨에 eval_metric을 인자로 미리 전달합니다.
            "eval_metric": mae_sigmoid_eval,
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores, mse_scores, r2_scores = [], [], []

        for train_idx, val_idx in kf.split(x_train, y_train_logit):
            X_tr, y_tr_logit = x_train.iloc[train_idx], y_train_logit.iloc[train_idx]
            X_val, y_val_logit = x_train.iloc[val_idx], y_train_logit.iloc[val_idx]
            y_val_orig = y_train.iloc[
                val_idx
            ]  # Original space target for validation score

            model = xgb.XGBRegressor(**params)

            # Train on logit space
            # fit() 내부의 'eval_metric' 대신 생성자 params로 전달하도록 변경하여 IDE 빨간줄 및 Warning 해결
            model.fit(X_tr, y_tr_logit, eval_set=[(X_val, y_val_logit)], verbose=False)

            # Predict in logit space
            preds_logit = model.predict(X_val)
            # Restore back to original space using sigmoid (expit)
            preds_orig = expit(preds_logit)

            # Core calculation must happen in original y space
            mae_scores.append(mean_absolute_error(y_val_orig, preds_orig))
            mse_scores.append(mean_squared_error(y_val_orig, preds_orig))
            r2_scores.append(r2_score(y_val_orig, preds_orig))

        mean_mae = np.mean(mae_scores)
        mean_mse = np.mean(mse_scores)
        mean_rmse = np.sqrt(mean_mse)
        mean_r2 = np.mean(r2_scores)

        trial.set_user_attr("mse", mean_mse)
        trial.set_user_attr("rmse", mean_rmse)
        trial.set_user_attr("r2", mean_r2)

        # Log metrics to W&B in original target scale
        # wandb.log(
        #     {
        #         "val_mae": mean_mae,
        #         "val_mse": mean_mse,
        #         "val_rmse": mean_rmse,
        #         "val_r2": mean_r2,
        #         "trial_number": trial.number,
        #     }
        # )
        return mean_mae

    wandb_kwargs = {
        "project": "stress_index_v8",
        "name": "xgboost_logit_tuning",
        "reinit": True,
    }
    wandbc = WeightsAndBiasesCallback(metric_name="mae", wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction="minimize")
    # 100 trials requested
    study.optimize(objective, n_trials=100, callbacks=[wandbc])
    wandb.finish()
    return study.best_params


print("Tuning XGBoost Hyperparameters...")
best_xgb = tune_xgboost(x_train, y_train)
print(f"Best XGBoost Params: {best_xgb}")


# ==========================================
# 7. Seed Averaging (XGBoost Spec) on Target Logit Space
# ==========================================
def train_with_seeds(x_train, y_train, x_test, best_params, seeds=[42, 2026, 777]):
    # Pre-transform target y_train
    y_train_clipped = np.clip(y_train, epsilon, 1.0 - epsilon)
    y_train_logit = logit(y_train_clipped)

    oof_preds_orig = np.zeros(len(x_train))
    test_preds_orig = np.zeros(len(x_test))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train_logit)):
        X_tr, y_tr_logit = x_train.iloc[train_idx].copy(), y_train_logit.iloc[train_idx]
        X_val, y_val_logit = x_train.iloc[val_idx].copy(), y_train_logit.iloc[val_idx]

        fold_val_preds_orig = np.zeros(len(X_val))
        fold_test_preds_orig = np.zeros(len(x_test))

        for seed in seeds:
            params = best_params.copy()
            params.update(
                {
                    "objective": "reg:absoluteerror",
                    "random_state": seed,
                    "verbosity": 0,
                    "enable_categorical": True,
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "eval_metric": mae_sigmoid_eval,
                }
            )

            model = xgb.XGBRegressor(**params)
            # Train model in Logit space
            model.fit(X_tr, y_tr_logit, eval_set=[(X_val, y_val_logit)], verbose=False)

            # Predict logit and transform back to original scale
            val_pred_orig = expit(model.predict(X_val))
            test_pred_orig = expit(model.predict(x_test))

            fold_val_preds_orig += val_pred_orig / len(seeds)
            fold_test_preds_orig += test_pred_orig / len(seeds)

        oof_preds_orig[val_idx] = fold_val_preds_orig
        test_preds_orig += fold_test_preds_orig / 5

    mae = mean_absolute_error(y_train, oof_preds_orig)
    print(f"[XGBoost Single] Seed Averaged OOF MAE: {mae:.6f}")
    return oof_preds_orig, test_preds_orig


oof_xgb, test_xgb = train_with_seeds(x_train, y_train, x_test, best_xgb)

final_oof_clipped = np.clip(oof_xgb, 0.0, 1.0)
print(f"Final Clipped OOF MAE: {mean_absolute_error(y_train, final_oof_clipped):.6f}")

test_preds = np.clip(test_xgb, 0.0, 1.0)


submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
submission["stress_score"] = test_preds
output_dir = "result/v8" if os.path.exists("result/v8") else "."
submission.to_csv(os.path.join(output_dir, "submit_08.csv"), index=False)
print("Submission saved to submit_08.csv")
