import json
import os

base_path = '/Users/admin/Documents/dev_src/stress_index/result/v8/LightGBM_v8_executed.ipynb'
out_path = '/Users/admin/Documents/dev_src/stress_index/result/v10/LightGBM_v10.ipynb'

with open(base_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 3: Imputation
nb['cells'][3]['source'] = [
    "# ==========================================\n",
    "# 3. 결측치 처리 및 Data-Centric 보강\n",
    "# ==========================================\n",
    "for df in [train, test]:\n",
    "    df['medical_history'] = df['medical_history'].fillna('none')\n",
    "    df['family_medical_history'] = df['family_medical_history'].fillna('none')\n",
    "    df['edu_level'] = df['edu_level'].fillna('Unknown')\n",
    "\n",
    "from sklearn.experimental import enable_iterative_imputer\n",
    "from sklearn.impute import IterativeImputer\n",
    "\n",
    "# IterativeImputer를 사용하여 수치형 변수 대치 (mean_working 제외)\n",
    "numeric_cols_for_imputation = ['age', 'height', 'weight', 'cholesterol', 'systolic_blood_pressure', \n",
    "                               'diastolic_blood_pressure', 'glucose', 'bone_density']\n",
    "\n",
    "imputer = IterativeImputer(random_state=42, max_iter=20)\n",
    "\n",
    "# Data Leakage 방지를 위해 Train 데이터로만 Imputer를 학습시킴\n",
    "train[numeric_cols_for_imputation] = imputer.fit_transform(train[numeric_cols_for_imputation])\n",
    "test[numeric_cols_for_imputation] = imputer.transform(test[numeric_cols_for_imputation])\n",
    "\n",
    "# mean_working 결측치는 -1로 대치하여 트리가 결측 자체를 인식하게 유도\n",
    "train['mean_working'] = train['mean_working'].fillna(-1)\n",
    "test['mean_working'] = test['mean_working'].fillna(-1)\n",
    "\n",
    "print(\"Imputation complete.\")\n"
]

# Cell 4: Feature Engineering
old_c4 = nb['cells'][4]['source']
new_c4 = []
for line in old_c4:
    new_c4.append(line)
    if "df['medical_disease_count']" in line:
        # Insert the missingness indicator right before it
        new_c4.insert(-1, "    df['is_working_missing'] = (df['mean_working'] == -1).astype(int)\n")
nb['cells'][4]['source'] = new_c4

# Cell 7: Optuna tuning
nb['cells'][7]['source'] = [
    "# ==========================================\n",
    "# 6. Optuna Tuning (Original Target Space with robust regularization)\n",
    "# ==========================================\n",
    "import optuna\n",
    "from optuna_integration.wandb import WeightsAndBiasesCallback\n",
    "from sklearn.model_selection import KFold\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "import xgboost as xgb\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "def tune_xgboost(x_train, y_train):\n",
    "    \n",
    "    def objective(trial):\n",
    "        params = {\n",
    "            'objective': 'reg:absoluteerror',\n",
    "            'random_state': 42,\n",
    "            'verbosity': 0,\n",
    "            'n_jobs': 1,\n",
    "            'enable_categorical': True,\n",
    "            'tree_method': 'hist',\n",
    "            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),\n",
    "            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),\n",
    "            'max_depth': trial.suggest_int('max_depth', 5, 12),\n",
    "            'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),\n",
    "            'subsample': trial.suggest_float('subsample', 0.5, 0.9),\n",
    "            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),\n",
    "            'gamma': trial.suggest_float('gamma', 0.0, 5.0),\n",
    "            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),\n",
    "            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),\n",
    "        }\n",
    "        \n",
    "        kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
    "        mae_scores, mse_scores, r2_scores = [], [], []\n",
    "        \n",
    "        for train_idx, val_idx in kf.split(x_train, y_train):\n",
    "            X_tr, y_tr = x_train.iloc[train_idx], y_train.iloc[train_idx]\n",
    "            X_val, y_val = x_train.iloc[val_idx], y_train.iloc[val_idx]\n",
    "            \n",
    "            model = xgb.XGBRegressor(**params)\n",
    "            model.fit(\n",
    "                X_tr, y_tr, \n",
    "                eval_set=[(X_val, y_val)],\n",
    "                verbose=False\n",
    "            )\n",
    "            \n",
    "            preds = model.predict(X_val)\n",
    "            # Clip predictions within [0, 1] range as per task bounds\n",
    "            preds_clipped = np.clip(preds, 0.0, 1.0)\n",
    "            \n",
    "            mae_scores.append(mean_absolute_error(y_val, preds_clipped))\n",
    "            mse_scores.append(mean_squared_error(y_val, preds_clipped))\n",
    "            r2_scores.append(r2_score(y_val, preds_clipped))\n",
    "        \n",
    "        mean_mae = np.mean(mae_scores)\n",
    "        mean_mse = np.mean(mse_scores)\n",
    "        mean_rmse = np.sqrt(mean_mse)\n",
    "        mean_r2 = np.mean(r2_scores)\n",
    "        \n",
    "        trial.set_user_attr(\"mse\", mean_mse)\n",
    "        trial.set_user_attr(\"rmse\", mean_rmse)\n",
    "        trial.set_user_attr(\"r2\", mean_r2)\n",
    "        \n",
    "        return mean_mae\n",
    "\n",
    "    wandb_kwargs = {\"project\": \"stress_index_v10\", \"name\": \"xgboost_v10_tuning\", \"reinit\": True}\n",
    "    wandbc = WeightsAndBiasesCallback(metric_name=\"mae\", wandb_kwargs=wandb_kwargs)\n",
    "    \n",
    "    study = optuna.create_study(direction='minimize')\n",
    "    study.optimize(objective, n_trials=100, callbacks=[wandbc])\n",
    "    wandb.finish()\n",
    "    return study.best_params\n",
    "\n",
    "print(\"Tuning XGBoost Hyperparameters...\")\n",
    "best_xgb = tune_xgboost(x_train, y_train)\n",
    "print(f\"Best XGBoost Params: {best_xgb}\")\n"
]

# Cell 8: Seed Averaging
nb['cells'][8]['source'] = [
    "# ==========================================\n",
    "# 7. Seed Averaging on Target Space\n",
    "# ==========================================\n",
    "def train_with_seeds(x_train, y_train, x_test, best_params, seeds=[42, 2026, 777]):\n",
    "    oof_preds = np.zeros(len(x_train))\n",
    "    test_preds = np.zeros(len(x_test))\n",
    "    kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
    "    \n",
    "    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train)):\n",
    "        X_tr, y_tr = x_train.iloc[train_idx].copy(), y_train.iloc[train_idx]\n",
    "        X_val, y_val = x_train.iloc[val_idx].copy(), y_train.iloc[val_idx]\n",
    "        \n",
    "        fold_val_preds = np.zeros(len(X_val))\n",
    "        fold_test_preds = np.zeros(len(x_test))\n",
    "        \n",
    "        for seed in seeds:\n",
    "            params = best_params.copy()\n",
    "            params.update({\n",
    "                'objective': 'reg:absoluteerror',\n",
    "                'random_state': seed,\n",
    "                'verbosity': 0,\n",
    "                'enable_categorical': True,\n",
    "                'tree_method': 'hist',\n",
    "                'n_jobs': 1,\n",
    "            })\n",
    "            \n",
    "            model = xgb.XGBRegressor(**params)\n",
    "            model.fit(\n",
    "                X_tr, y_tr,\n",
    "                eval_set=[(X_val, y_val)],\n",
    "                verbose=False\n",
    "            )\n",
    "            \n",
    "            fold_val_preds += np.clip(model.predict(X_val), 0.0, 1.0) / len(seeds)\n",
    "            fold_test_preds += np.clip(model.predict(x_test), 0.0, 1.0) / len(seeds)\n",
    "            \n",
    "        oof_preds[val_idx] = fold_val_preds\n",
    "        test_preds += fold_test_preds / 5\n",
    "        \n",
    "    mae = mean_absolute_error(y_train, oof_preds)\n",
    "    print(f'[XGBoost Single] Seed Averaged OOF MAE: {mae:.6f}')\n",
    "    return oof_preds, test_preds\n",
    "\n",
    "oof_xgb, test_xgb = train_with_seeds(x_train, y_train, x_test, best_xgb)\n",
    "\n",
    "final_oof_clipped = np.clip(oof_xgb, 0.0, 1.0)\n",
    "print(f\"Final Clipped OOF MAE: {mean_absolute_error(y_train, final_oof_clipped):.6f}\")\n",
    "\n",
    "test_preds = np.clip(test_xgb, 0.0, 1.0)\n"
]

# Cell 9: Submit
nb['cells'][9]['source'] = [
    "submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))\n",
    "submission['stress_score'] = test_preds\n",
    "output_dir = 'result/v10' if os.path.exists('result/v10') else '.'\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "submission.to_csv(os.path.join(output_dir, 'submit_10.csv'), index=False)\n",
    "print(\"Submission saved to submit_10.csv\")\n"
]

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("v10 Notebook created successfully.")
