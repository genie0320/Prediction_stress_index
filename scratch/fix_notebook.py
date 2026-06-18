import json

notebook_path = '/Users/admin/Documents/dev_src/stress_index/result/v8/LightGBM_v8_executed.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # We want to remove the lines adding eval_metric to params
            if "'eval_metric': mae_sigmoid_eval" in line:
                # Also need to make sure we remove the comma from the previous line if it was the last item
                # But looking at the code, alpha/lambda has a comma. We'll just skip this line.
                pass
            elif "생성자(init) 레벨에 eval_metric을 인자로 미리 전달합니다" in line:
                pass
            elif "'n_jobs': 1," in line and "eval_metric" not in line:
                # In train_with_seeds, the previous line is 'n_jobs': 1,
                # if we remove eval_metric, 'n_jobs': 1, will have a trailing comma, which is fine in python.
                new_source.append(line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
