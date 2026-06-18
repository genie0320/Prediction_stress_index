import os
import copy
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import wandb

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

seed_everything(42)

def count_items(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    items = [item.strip() for item in str(x).split(',') if item.strip()]
    return len(items)

def preprocess_data(df):
    data = df.copy()
    
    data['bmi'] = data['weight'] / ((data['height'] / 100.0) ** 2)
    data['pp'] = data['systolic_blood_pressure'] - data['diastolic_blood_pressure']
    data['map_val'] = (data['systolic_blood_pressure'] + 2 * data['diastolic_blood_pressure']) / 3
    
    activity_map = {'light': 1, 'moderate': 2, 'intense': 3}
    data['activity_enc'] = data['activity'].map(activity_map).fillna(0).astype(int)
    
    edu_map = {'high school diploma': 1, 'bachelors degree': 2, 'graduate degree': 3}
    data['edu_enc'] = data['edu_level'].map(edu_map).fillna(0).astype(int)
    
    sleep_map = {'sleep difficulty': 1, 'normal': 2, 'oversleeping': 3}
    data['sleep_enc'] = data['sleep_pattern'].map(sleep_map).fillna(0).astype(int)
    
    gender_map = {'F': 0, 'M': 1}
    data['gender_enc'] = data['gender'].map(gender_map).fillna(0).astype(int)
    
    smoke_map = {'non-smoker': 0, 'ex-smoker': 1, 'current-smoker': 2}
    data['smoke_enc'] = data['smoke_status'].map(smoke_map).fillna(0).astype(int)
    
    data['n_medical'] = data['medical_history'].apply(count_items)
    data['n_family'] = data['family_medical_history'].apply(count_items)
    data['mean_working_filled'] = data['mean_working'].fillna(0)
    
    return data

class StressMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(19, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def main():
    wandb.init(project="stress-index-prediction", name="v16_6_mlp_v2_svr_ensemble", config={
        "architecture": "MLP 256-256-128",
        "epochs": 200,
        "batch_size": 256,
        "lr": 3e-4,          # 수정 3 반영
        "patience": 30,      # 수정 2 반영
        "inner_val": 0.2,    # 수정 1 반영
        "weight_decay": 1e-4,
        "features": 19
    })
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    train_proc = preprocess_data(train)
    test_proc = preprocess_data(test)
    
    features = [
        'age', 'height', 'weight', 'cholesterol',
        'systolic_blood_pressure', 'diastolic_blood_pressure',
        'glucose', 'bone_density',
        'bmi', 'pp', 'map_val', 'mean_working_filled',
        'activity_enc', 'edu_enc', 'sleep_enc', 'gender_enc', 'smoke_enc',
        'n_medical', 'n_family'
    ]
    
    X = train_proc[features]
    y = train_proc['stress_score']
    X_test = test_proc[features]
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # ==========================================
    # Phase 1: SVR OOF 생성
    # ==========================================
    print("[Phase 1: SVR 10-Fold OOF]")
    svr_oof = np.zeros(len(X))
    svr_test_pred = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)
        
        qt = QuantileTransformer(n_quantiles=983, output_distribution='normal', random_state=42)
        y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        model_svr = SVR(kernel='rbf', C=3.5579, gamma=0.8347, epsilon=0.0)
        model_svr.fit(X_tr_sc, y_tr_t)
        
        va_pred = qt.inverse_transform(model_svr.predict(X_va_sc).reshape(-1, 1)).flatten()
        svr_oof[val_idx] = np.clip(va_pred, 0, 1)
        
        te_pred = qt.inverse_transform(model_svr.predict(X_te_sc).reshape(-1, 1)).flatten()
        svr_test_pred += np.clip(te_pred, 0, 1) / 10.0
        
        print(f"  Fold {fold+1:02d} MAE: {mean_absolute_error(y_va, svr_oof[val_idx]):.5f}")
        
    svr_mae_raw = mean_absolute_error(y, svr_oof)
    svr_mae_post = mean_absolute_error(y, np.round(svr_oof, 2))
    print(f"  SVR OOF MAE (Raw)            : {svr_mae_raw:.6f}")
    print(f"  SVR OOF MAE (Post-processed) : {svr_mae_post:.6f}\n")
    
    # ==========================================
    # Phase 2: MLP OOF 생성
    # ==========================================
    print("[Phase 2: MLP 10-Fold OOF (v2 Settings)]")
    mlp_oof = np.zeros(len(X))
    mlp_test_pred = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Step A. 스케일링/변환
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        X_te_sc = scaler.transform(X_test)
        
        qt = QuantileTransformer(n_quantiles=983, output_distribution='normal', random_state=42)
        y_tr_t = qt.fit_transform(y_tr.values.reshape(-1, 1)).flatten()
        
        # Step B. inner_val 분리 (수정 1: 80:20 분할)
        X_in_tr, X_in_va, y_in_tr, y_in_va = train_test_split(
            X_tr_sc, y_tr_t, test_size=0.2, random_state=fold, shuffle=True
        )
        
        train_dataset = TensorDataset(torch.FloatTensor(X_in_tr), torch.FloatTensor(y_in_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_in_va), torch.FloatTensor(y_in_va))
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Step C. MLP 학습 (수정 3: lr=3e-4)
        model = StressMLP().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
        
        best_val_loss = float('inf')
        patience = 30 # 수정 2: patience 완화
        patience_counter = 0
        best_model_state = None
        stopped_epoch = 200
        
        for epoch in range(200):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bx.size(0)
            train_loss /= len(train_dataset)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx)
                    loss = criterion(pred, by)
                    val_loss += loss.item() * bx.size(0)
            val_loss /= len(val_dataset)
            
            scheduler.step()
            wandb.log({f"fold_{fold+1}/train_loss": train_loss, f"fold_{fold+1}/val_loss": val_loss, "epoch": epoch})
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                break
                
        # 가중치 복원
        model.load_state_dict(best_model_state)
        
        # Step D. OOF 예측 (Outer validation set)
        model.eval()
        with torch.no_grad():
            t_va = torch.FloatTensor(X_va_sc).to(device)
            raw_va_pred = model(t_va).cpu().numpy()
            inv_va_pred = qt.inverse_transform(raw_va_pred.reshape(-1, 1)).flatten()
            mlp_oof[val_idx] = np.clip(inv_va_pred, 0, 1)
            
            t_te = torch.FloatTensor(X_te_sc).to(device)
            raw_te_pred = model(t_te).cpu().numpy()
            inv_te_pred = qt.inverse_transform(raw_te_pred.reshape(-1, 1)).flatten()
            mlp_test_pred += np.clip(inv_te_pred, 0, 1) / 10.0
            
        # Step E. fold 로그 출력
        print(f"  Fold {fold+1:02d} | stopped_epoch: {stopped_epoch:3d} | MAE: {mean_absolute_error(y_va, mlp_oof[val_idx]):.5f}")
        
    mlp_mae_raw = mean_absolute_error(y, mlp_oof)
    mlp_mae_post = mean_absolute_error(y, np.round(mlp_oof, 2))
    
    v16_baseline = 0.13951
    diff = mlp_mae_post - v16_baseline
    diff_str = f"+{diff:.6f} (악화)" if diff > 0 else f"{diff:.6f} (개선)"
    
    print(f"  MLP OOF MAE (Raw)            : {mlp_mae_raw:.6f}")
    print(f"  MLP OOF MAE (Post-processed) : {mlp_mae_post:.6f}")
    print(f"  v16 기준값                    : {v16_baseline:.5f}")
    print(f"  개선폭                        : {diff_str}\n")
    
    # ==========================================
    # Phase 3: 상관계수 및 블렌딩
    # ==========================================
    print("[Phase 3: SVR vs MLP 상관계수]")
    r, _ = pearsonr(svr_oof, mlp_oof)
    print(f"  Pearson r: {r:.6f}")
    if r < 0.85:
        print("  판정: 앙상블 효과 기대 가능\n")
    else:
        print("  판정: 앙상블 효과 제한적\n")
        
    print("[Phase 3: 블렌딩 비율별 OOF MAE]")
    print(f"  alpha(SVR비율)   OOF MAE     v16 대비")
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_blend_mae = float('inf')
    best_alpha = 0.5
    
    for alpha in alphas:
        blend_oof = alpha * svr_oof + (1 - alpha) * mlp_oof
        blend_mae = mean_absolute_error(y, blend_oof)
        b_diff = blend_mae - v16_baseline
        b_diff_str = f"+{b_diff:.6f}" if b_diff > 0 else f"{b_diff:.6f}"
        print(f"  {alpha:.2f}             {blend_mae:.6f}    {b_diff_str}")
        
        if blend_mae < best_blend_mae:
            best_blend_mae = blend_mae
            best_alpha = alpha
            
    print(f"  최적 alpha: {best_alpha:.1f} → OOF MAE: {best_blend_mae:.6f}\n")
    
    # ==========================================
    # Phase 4: 결과 저장
    # ==========================================
    os.makedirs('result/v16', exist_ok=True)
    
    # MLP 단독
    test_mlp_final = np.round(np.clip(mlp_test_pred, 0, 1), 2)
    pd.DataFrame({'ID': test['ID'], 'stress_score': test_mlp_final}).to_csv('result/v16/submit_v16-6_mlp_v2.csv', index=False)
    
    # 블렌딩
    test_blend = best_alpha * svr_test_pred + (1 - best_alpha) * mlp_test_pred
    test_blend_final = np.round(np.clip(test_blend, 0, 1), 2)
    pd.DataFrame({'ID': test['ID'], 'stress_score': test_blend_final}).to_csv('result/v16/submit_v16-6_blend_svr_mlp_v2.csv', index=False)
    
    print("[Phase 4: Test 예측값 통계]")
    print(f"  MLP 단독     — Mean: {test_mlp_final.mean():.2f}  Std: {test_mlp_final.std():.2f}  N_unique: {len(np.unique(test_mlp_final))}")
    print(f"  SVR+MLP 블렌딩 — Mean: {test_blend_final.mean():.2f}  Std: {test_blend_final.std():.2f}  N_unique: {len(np.unique(test_blend_final))}")

    wandb.finish()

if __name__ == '__main__':
    main()
