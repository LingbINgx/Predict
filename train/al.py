
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

import sys
sys.path.append('..')
import utils.modal

data = pd.read_csv('../fresh.csv')
features = data.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
labels = data['Machine failure']
features = features.astype(np.float64)
labels = labels.astype(np.float64)  

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
)

lr = LogisticRegression(
        class_weight={0:1, 1:4},
        max_iter=1000
    )

svm = SVC(
        kernel='rbf',
        class_weight={0:1, 1:4},
        probability=True
    )

rf = RandomForestClassifier(
        n_estimators=300,
        class_weight={0:1, 1:3},
        random_state=42
    )



voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svm', svm),
        ('rf', rf)
    ],
    voting='soft',
    weights=[0.45, 0.35, 0.2]
)

train = utils.modal.BasicActiveLearner(estimator=voting, query_strategy='uncertainty', random_state=42)
train.load_data(features.values, labels.values, test_size=0.5)
train.initialize_data(initial_size=100)
train.create_learner()
n_queries = 500
query_size = 100
metrics = train.active_learning_cycle(n_queries=n_queries, query_size=query_size, th=0.3)


print("Best accuracy:", max(metrics["acc"]) ) 
print("best recall", max([report['1.0']['recall'] for report in metrics['report']]) )
print("best precision", max([report['1.0']['precision'] for report in metrics['report']]) )

train.plot_precision_recall_curve(metrics)