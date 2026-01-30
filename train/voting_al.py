
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
import utils.logres as logres

data = pd.read_csv('../data/ai4i2020_processed.csv')
features = data.drop(columns=['Machine_failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
labels = data['Machine_failure']
features = features.astype(np.float64)
labels = labels.astype(np.float64)  



clf1 = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    objective='binary:logistic',
    eval_metric='aucpr', 
    subsample=0.5,
    min_child_weight=5,
)
clf2 = LGBMClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    class_weight='balanced',
    subsample=0.5,
    min_data_in_leaf=5,
    num_leaves=15,
    importance_type='gain',
)
clf3 = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    max_samples=0.5
)

mod = VotingClassifier(
    estimators=[('xgb', clf1), ('lgbm', clf2), ('rf', clf3)],
    voting='soft',
    weights=[0.33, 0.34, 0.33]
)

metrics, train = utils.activeLearn.instant_launch(features, labels,
                                                  estimator=mod,
                                                  query_strategy='entropy',
                                                  random_state=42,
                                                  initial_size=100,
                                                  n_queries=500,
                                                  query_size=100,
                                                  threshold=0.55,
                                                  test_size=0.7,
                                                  imbalance_handle='None',
                                                  )

logres.write_results_to_file(mod, train.X_test, train.y_test)

train.plot_precision_recall_curve(metrics)

