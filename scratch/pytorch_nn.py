import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data
train_path = "/Users/admin/Documents/dev_src/stress_index/train.csv"
train = pd.read_csv(train_path)

base_features = [
    "age", "height", "weight", "cholesterol", 
    "systolic_blood_pressure", "diastolic_blood_pressure", 
    "glucose", "bone_density"
]
target = "stress_score"

X = train[base_features].values
y = train[target].values.reshape(-1, 1)

# PyTorch MLP definition
class StressNet(nn.Module):
    def __init__(self, input_dim):
        super(StressNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(X, y, epochs=300, batch_size=64, lr=0.005, weight_decay=1e-4):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1}/5 ---")
        
        # Split and Scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        y_tr = y[train_idx]
        y_val = y[val_idx]
        
        # To Tensors
        X_tr_t = torch.FloatTensor(X_tr).to(device)
        y_tr_t = torch.FloatTensor(y_tr).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        
        # Dataloader
        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Model, Loss, Optimizer
        model = StressNet(len(base_features)).to(device)
        criterion = nn.L1Loss() # MAE Loss
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
        
        # Train Loop
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            
            epoch_loss /= len(X_tr)
            scheduler.step(epoch_loss)
            
            if (epoch + 1) % 50 == 0:
                # Eval on validation to print progress
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_mae = mean_absolute_error(y_val, val_outputs.cpu().numpy())
                model.train()
                print(f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {epoch_loss:.5f} | Val MAE: {val_mae:.5f}")
                
        # Predict OOF
        model.eval()
        with torch.no_grad():
            oof_preds = model(X_val_t)
            oof[val_idx] = oof_preds.cpu().numpy().flatten()
            
    total_mae = mean_absolute_error(y, oof)
    print(f"\nFinal PyTorch MLP OOF MAE: {total_mae:.6f}")
    return total_mae

train_and_evaluate(X, y)
