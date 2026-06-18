import json

nb_path = '/Users/admin/Documents/dev_src/stress_index/result/v10/LightGBM_v10.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix 1: Add is_working_missing
c4_source = nb['cells'][4]['source']
new_c4 = []
for line in c4_source:
    new_c4.append(line)
    if "for df in [train, test]:" in line:
        new_c4.append("    # ag가 누락했던 결측치 시그널 파생 변수 강제 추가\n")
        new_c4.append("    df[\"is_working_missing\"] = (df[\"mean_working\"] == -1.0).astype(int)\n")
nb['cells'][4]['source'] = new_c4

# Fix 2: Drop ratio 0.2 instead of N=3
c6_source = nb['cells'][6]['source']
new_c6 = []
for line in c6_source:
    if line.startswith("# 최하위 3개 피처 자동 제거"):
        new_c6.append("# 최하위 20% 피처 대규모 자동 제거 (소심한 N=3 방기)\n")
    elif line.startswith("N = 3"):
        new_c6.append("drop_ratio = 0.2\n")
        new_c6.append("N = int(len(shap_imp) * drop_ratio)\n")
    else:
        new_c6.append(line)
nb['cells'][6]['source'] = new_c6

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook v10 fixed successfully.")
