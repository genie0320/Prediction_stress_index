import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

def main():
    # Load data
    test = pd.read_csv('test.csv')
    df_16_7 = pd.read_csv('result/v16/submit_v16-7_svr_multiseed.csv')
    df_17_2 = pd.read_csv('result/v16/submit_v17-2_final.csv')
    
    pred_16_7 = df_16_7['stress_score']
    pred_17_2 = df_17_2['stress_score']
    
    # Step 1: Residual basic stats
    diff = pred_17_2 - pred_16_7
    
    print("[Step 1: 잔차 기본 통계]")
    print(f"diff(17-2 - 16-7) 통계:")
    print(f"Mean: {diff.mean():.5f}")
    print(f"Std:  {diff.std():.5f}")
    print(f"Min:  {diff.min():.5f}")
    print(f"Max:  {diff.max():.5f}")
    print(f"|diff|>0.1: {(diff.abs() > 0.1).sum()}개")
    print(f"|diff|>0.2: {(diff.abs() > 0.2).sum()}개")
    
    r, _ = pearsonr(pred_16_7, pred_17_2)
    print(f"Pearson r (두 예측값): {r:.6f}")
    
    # Step 2: Feature means by bin
    test['pred_16_7'] = pred_16_7
    test['pred_17_2'] = pred_17_2
    test['diff'] = diff
    
    def get_bin(d):
        if d < -0.1: return "크게음수(<-0.1)"
        elif d < -0.03: return "음수"
        elif d <= 0.03: return "동의"
        elif d <= 0.1: return "양수"
        else: return "크게양수(>0.1)"
        
    test['bin'] = test['diff'].apply(get_bin)
    
    feats = ['age', 'glucose', 'bone_density', 'systolic_blood_pressure', 
             'diastolic_blood_pressure', 'cholesterol', 'height', 'weight']
    
    print("\n[Step 2: 구간별 피처 평균]")
    print(f"{'구간':<15} {'샘플수':<5} {'16-7':<6} {'17-2':<6} {'age':<5} {'glucose':<7} {'bone':<6} {'sys':<5} {'dia':<5} {'chol':<5} {'height':<6} {'weight':<6}")
    
    order = ["크게음수(<-0.1)", "음수", "동의", "양수", "크게양수(>0.1)"]
    for b in order:
        sub = test[test['bin'] == b]
        if len(sub) == 0:
            continue
        n = len(sub)
        p16 = sub['pred_16_7'].mean()
        p17 = sub['pred_17_2'].mean()
        means = sub[feats].mean()
        
        print(f"{b:<15} {n:<5} {p16:<6.3f} {p17:<6.3f} {means['age']:<5.1f} {means['glucose']:<7.1f} {means['bone_density']:<6.3f} {means['systolic_blood_pressure']:<5.1f} {means['diastolic_blood_pressure']:<5.1f} {means['cholesterol']:<5.1f} {means['height']:<6.1f} {means['weight']:<6.1f}")
        
    # Step 3: Top 50 mismatches
    print("\n[Step 3: 불일치 상위 50개]")
    test['abs_diff'] = test['diff'].abs()
    top50 = test.sort_values('abs_diff', ascending=False).head(50)
    
    disp_cols = ['ID', 'pred_16_7', 'pred_17_2', 'diff', 'age', 'glucose', 'bone_density']
    print(top50[disp_cols].to_string(index=False))
    
    print("\n전체 test 평균 vs 불일치 50개 평균 비교:")
    all_mean = test[feats].mean()
    top50_mean = top50[feats].mean()
    for f in feats:
        print(f"{f:<25} 전체: {all_mean[f]:.3f} | 불일치50: {top50_mean[f]:.3f}")
        
    # Step 4: High confidence distribution
    print("\n[Step 4: 고신뢰 구간 분포]")
    agree = test[test['bin'] == '동의']
    print(f"|diff|<=0.03인 샘플: {len(agree)}개")
    print("stress 구간별 분포:")
    bins = np.arange(0.0, 1.1, 0.1)
    cuts = pd.cut(agree['pred_17_2'], bins, right=False)
    counts = cuts.value_counts().sort_index()
    for interval, count in counts.items():
        print(f"{interval.left:.1f}~{interval.right:.1f}: {count}개  ", end='')
    print()
    
    # Step 5: Blending stats
    print("\n[Step 5: 블렌딩별 통계]")
    print(f"{'alpha':<6} {'Mean':<7} {'Std':<7} {'N_unique'}")
    
    target_std = 0.288
    closest_alpha = None
    min_std_diff = 999
    closest_std = 0
    
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        blend = alpha * pred_17_2 + (1 - alpha) * pred_16_7
        blend_round = np.round(np.clip(blend, 0, 1), 2)
        mean_val = blend_round.mean()
        std_val = blend_round.std()
        n_uq = blend_round.nunique()
        
        print(f"{alpha:<6} {mean_val:<7.4f} {std_val:<7.4f} {n_uq}")
        
        std_diff = abs(std_val - target_std)
        if std_diff < min_std_diff:
            min_std_diff = std_diff
            closest_alpha = alpha
            closest_std = std_val
            
    print(f"실제 타깃 std 목표: {target_std}")
    print(f"가장 가까운 alpha: {closest_alpha} (Std={closest_std:.4f})")

if __name__ == '__main__':
    main()
