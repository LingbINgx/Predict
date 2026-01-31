
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import cross_val_score


import sys
sys.path.append('..')
import models.model as model

data = pd.read_csv('./data/ai4i2020_processed.csv')
features = data.drop(columns=['Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
labels = data['Machine_failure']
features = features.astype(np.float64)
labels = labels.astype(np.float64)  


X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.8,
    random_state=42,
    stratify=labels
)
# smote = SMOTE(random_state=42, sampling_strategy=0.8)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr', 
)
lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    class_weight='balanced',
    subsample=0.5,
    min_data_in_leaf=5,
    num_leaves=15,
    importance_type='gain',
)
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    max_samples=0.5
)


def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    }
    c1 = XGBClassifier(**xgb_params, eval_metric='aucpr')

    score = cross_val_score(c1, X_train, y_train, n_jobs=-1, cv=3, scoring='f1')
    return score.mean()

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=500) 
# print("最佳参数: ", study.best_params)


best_params =  {
     'n_estimators': 300, 
     'max_depth': 9, 
     'learning_rate': 0.05,
    'objective':'binary:logistic',
    'eval_metric':'aucpr', 
    'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    'seed':42
}

mod = model.BaseMod(XGBClassifier(), task='classification')
mod.set_params(**best_params)
mod.fit(X_train, y_train)
y_prob = mod.predict_proba(X_test)[:, 1]
threshold = 0.6
y_pred = (y_prob > threshold).astype(int)
print("Confusion Matrix:\n", model.confusion_matrix(y_test, y_pred))
print("Classification Report:\n", model.classification_report(y_test, y_pred))

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = f1.argmax()
th = thresholds[best_idx]
y_pred = (y_prob > th).astype(int)
print("Confusion Matrix:\n", model.confusion_matrix(y_test, y_pred))
print("Classification Report:\n", model.classification_report(y_test, y_pred))
print(mod.get_params())
