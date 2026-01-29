
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import xgboost as xgb

import sys
sys.path.append('..')
import models.model as model

data = pd.read_csv('./fresh.csv')
features = data.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
labels = data['Machine failure']
features = features.astype(np.float64)
labels = labels.astype(np.float64)  


X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.5,
    random_state=42,
    stratify=labels
)

models = {
    "logistic_regression":model.BaseMod(model=LogisticRegression(), task='classification'),
    "XGBoost":model.BaseMod(model=XGBClassifier(), task='classification'),}


# for name, mod in models.items():
#     mod.fit(X_train, y_train)
#     print(mod.score(X_test, y_test))
#     print(mod.confusion_matrix(X_test, y_test))
#     print(mod.classification_report(X_test, y_test))

mod = model.LogisticRegressionModel()

mod.fit(X_train, y_train)
print(mod.score(X_test, y_test))
print(mod.confusion_matrix(X_test, y_test))
print(mod.classification_report(X_test, y_test))