import json
import os

v8_path = '/Users/admin/Documents/dev_src/stress_index/result/v8/LightGBM_v8_executed.ipynb'
v9_path = '/Users/admin/Documents/dev_src/stress_index/result/v9/LightGBM_v9_executed.ipynb'

with open(v8_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            line = line.replace("trial.suggest_int('max_depth', 3, 15)", "trial.suggest_int('max_depth', 3, 8)")
            line = line.replace("trial.suggest_int('min_child_weight', 1, 30)", "trial.suggest_int('min_child_weight', 10, 50)")
            line = line.replace("trial.suggest_int('n_estimators', 200, 1500)", "trial.suggest_int('n_estimators', 200, 800)")
            line = line.replace("trial.suggest_float('subsample', 0.5, 1.0)", "trial.suggest_float('subsample', 0.5, 0.9)")
            line = line.replace("trial.suggest_float('colsample_bytree', 0.4, 1.0)", "trial.suggest_float('colsample_bytree', 0.5, 0.9)")
            
            line = line.replace('"stress_index_v8"', '"stress_index_v9"')
            line = line.replace("'result/v8'", "'result/v9'")
            line = line.replace("'submit_08.csv'", "'submit_09.csv'")
            line = line.replace("'shap_summary_plot_v8.png'", "'shap_summary_plot_v9.png'")
            
            new_source.append(line)
        cell['source'] = new_source

with open(v9_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("v9 notebook created successfully.")
