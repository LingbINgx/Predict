
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve

import sys
sys.path.append('..')
import utils.activeLearn

data = pd.read_csv('../data/ai4i2020_processed.csv')
features = data.drop(columns=['Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
labels = data['Machine_failure']
features = features.astype(np.float64)
labels = labels.astype(np.float64)  



clf1 = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    # scale_pos_weight = ,
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

mod = VotingClassifier(
    estimators=[('xgb', clf1), ('lgbm', clf2), ('rf', clf3)],
    voting='soft',
    weights=[0.33, 0.34, 0.33]
)

train = utils.activeLearn.BasicActiveLearner(estimator=mod, query_strategy='entropy', random_state=42)
train.load_data(features.values, labels.values, test_size=0.7, imbalance_handle='None',  sampling_strategy=0.8)
train.initialize_data(initial_size=100)
n_queries = 500
query_size = 100
threshold = 0.55
metrics = train.active_learning_cycle(n_queries=n_queries, query_size=query_size, threshold=threshold)

print("Best accuracy:", max(metrics["acc"]) ) 
print("best recall", max([report['1.0']['recall'] for report in metrics['report']]) )
print("best precision", max([report['1.0']['precision'] for report in metrics['report']]) )

train.plot_precision_recall_curve(metrics)

precision, recall, thresholds = precision_recall_curve(train.y_test, train.learner.predict_proba(train.X_test)[:, 1])
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = f1.argmax()
print("\nBest threshold:", thresholds[best_idx])
print("Precision:", precision[best_idx])
print("Recall:", recall[best_idx])
print("F1:", f1[best_idx])

th = thresholds[best_idx]
y_prob = train.learner.predict_proba(train.X_test)[:, 1]
y_pred = (y_prob > th).astype(int)

import datetime
with open(f'../out/voting_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
    f.write(f'Best threshold: {th}\n')
    f.write(f'Precision: {precision[best_idx]}\n')
    f.write(f'Recall: {recall[best_idx]}\n')
    f.write(f'F1: {f1[best_idx]}\n')
    f.write(f'Confusion Matrix:\n{utils.activeLearn.confusion_matrix(train.y_test, y_pred)}\n')
    f.write(f'Classification Report:\n{utils.activeLearn.classification_report(train.y_test, y_pred)}\n')
    
    