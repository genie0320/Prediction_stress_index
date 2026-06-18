import json

v9_path = '/Users/admin/Documents/dev_src/stress_index/result/v9/LightGBM_v9_executed.ipynb'

with open(v9_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Fix cell 7 (index 7, which is the 8th cell)
cell_8 = notebook['cells'][7]
new_source_8 = []
in_params = False
for line in cell_8['source']:
    if "params = {" in line:
        in_params = True
        new_source_8.append('        params = {\n')
        new_source_8.append('            "objective": "reg:absoluteerror",\n')
        new_source_8.append('            "random_state": 42,\n')
        new_source_8.append('            "verbosity": 0,\n')
        new_source_8.append('            "n_jobs": 1,\n')
        new_source_8.append('            "enable_categorical": True,\n')
        new_source_8.append('            "tree_method": "hist",\n')
        new_source_8.append('            "n_estimators": trial.suggest_int("n_estimators", 200, 800),\n')
        new_source_8.append('            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),\n')
        new_source_8.append('            "max_depth": trial.suggest_int("max_depth", 3, 8),\n')
        new_source_8.append('            "min_child_weight": trial.suggest_int("min_child_weight", 10, 50),\n')
        new_source_8.append('            "subsample": trial.suggest_float("subsample", 0.5, 0.9),\n')
        new_source_8.append('            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),\n')
        new_source_8.append('            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),\n')
        new_source_8.append('            "lambda": trial.suggest_float("lambda", 1e-4, 10.0, log=True),\n')
        new_source_8.append('        }\n')
        continue
    
    if in_params:
        if "}" in line:
            in_params = False
        continue
        
    new_source_8.append(line)

notebook['cells'][7]['source'] = new_source_8

# Fix last cell (index 9)
cell_last = notebook['cells'][-1]
new_source_last = [
    'submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))\n',
    'submission["stress_score"] = test_preds\n',
    'output_dir = "result/v9" if os.path.exists("result/v9") else "."\n',
    'os.makedirs(output_dir, exist_ok=True)\n',
    'submission.to_csv(os.path.join(output_dir, "submit_09.csv"), index=False)\n',
    'print("Submission saved to submit_09.csv")'
]
notebook['cells'][-1]['source'] = new_source_last

with open(v9_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook successfully fixed.")
