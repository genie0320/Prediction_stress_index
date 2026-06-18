import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.special import expit

def mae_sigmoid_eval(y_true, y_pred):
    orig_preds = expit(y_pred)
    orig_labels = expit(y_true)
    mae = np.mean(np.abs(orig_preds - orig_labels))
    return mae

X = pd.DataFrame(np.random.rand(10, 2), columns=['A', 'B'])
y = pd.Series(np.random.rand(10))

params = {
    'eval_metric': mae_sigmoid_eval,
    'n_estimators': 2
}
model = xgb.XGBRegressor(**params)
model.fit(X, y, eval_set=[(X, y)])
