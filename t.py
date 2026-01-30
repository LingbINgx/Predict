
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier


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
    test_size=0.7,
    random_state=42,
    stratify=labels
)
smote = SMOTE(random_state=42, sampling_strategy=0.8)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


clf1 = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    scale_pos_weight = (y_train_resampled == 0).sum() / (y_train_resampled == 1).sum(),
    objective='binary:logistic',
    eval_metric='aucpr', 
    subsample=0.5,
)
clf2 = LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    class_weight='balanced',
    subsample=0.5,
)
clf3 = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
)

models = {
    'XGBClassifier': clf1,
    'LGBMClassifier': clf2,
    'RandomForestClassifier': clf3,
    'VotingClassifier': VotingClassifier(
        estimators=[('xgb', clf1), ('lgbm', clf2), ('rf', clf3)],
        voting='soft',
        weights=[0.33, 0.34, 0.33]
    ),
}

f1 = []

for name, mod in models.items():
    mod = model.BaseMod(mod, task='classification')
    #mod.fit(X_train, y_train)
    mod.fit(X_train_resampled, y_train_resampled)
    print(name)
    print(mod.score(X_test, y_test))
    print(mod.confusion_matrix(X_test, y_test))
    print(mod.classification_report(X_test, y_test))
    f1.append(mod.f1_score(X_test, y_test))

print(f1[0]/sum(f1[0:3]))
print(f1[1]/sum(f1[0:3]))
print(f1[2]/sum(f1[0:3]))