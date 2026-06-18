import xgboost as xgb
import numpy as np
from scipy.special import expit

def mae_sigmoid_eval(y_true, y_pred):
    orig_preds = expit(y_pred)
    orig_labels = expit(y_true)
    mae = np.mean(np.abs(orig_preds - orig_labels))
    return mae

X = np.random.rand(10, 2)
y = np.random.rand(10)

params = {
    'eval_metric': mae_sigmoid_eval,
    'n_estimators': 2
}
model = xgb.XGBRegressor(**params)
model.fit(X, y, eval_set=[(X, y)])
