import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import KFold

def mae_sigmoid_eval(y_true, y_pred):
    orig_preds = expit(y_pred)
    orig_labels = expit(y_true)
    mae = np.mean(np.abs(orig_preds - orig_labels))
    return mae

x_train = pd.DataFrame(np.random.rand(100, 5), columns=[f'col_{i}' for i in range(5)])
x_train['cat_col'] = pd.Series(['A', 'B'] * 50, dtype='category')
y_train = pd.Series(np.random.rand(100))

epsilon = 1e-5
y_train_clipped = np.clip(y_train, epsilon, 1.0 - epsilon)
y_train_logit = logit(y_train_clipped)

params = {
    'objective': 'reg:absoluteerror',
    'random_state': 42,
    'verbosity': 1,
    'n_jobs': 1,
    'enable_categorical': True,
    'tree_method': 'hist',
    'n_estimators': 10,
    'eval_metric': mae_sigmoid_eval
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(x_train, y_train_logit):
    X_tr, y_tr_logit = x_train.iloc[train_idx], y_train_logit.iloc[train_idx]
    X_val, y_val_logit = x_train.iloc[val_idx], y_train_logit.iloc[val_idx]
    
    model = xgb.XGBRegressor(**params)
    
    try:
        model.fit(
            X_tr, y_tr_logit, 
            eval_set=[(X_val, y_val_logit)],
            verbose=True
        )
        print("Fit succeeded")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Error during fit:", type(e), e)
    break
