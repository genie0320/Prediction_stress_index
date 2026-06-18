import json

v9_path = '/Users/admin/Documents/dev_src/stress_index/result/v9/LightGBM_v9_executed.ipynb'

with open(v9_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # update learning rate
            if "trial.suggest_float('learning_rate', 0.001, 0.1, log=True)" in line:
                line = line.replace("0.001", "0.01")
            new_source.append(line)
        cell['source'] = new_source

with open(v9_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
