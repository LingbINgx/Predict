from sklearn.linear_model import (
    LogisticRegression, 
    LinearRegression, 
    Ridge, 
    Lasso, 
    ElasticNet,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import confusion_matrix, classification_report


class BaseMod():
    def __init__(self, model, task='auto', **kwargs):
        self.model = model
        self.task = task
        if self.task == 'auto':
            if 'Classifier' in model.__class__.__name__ or 'DiscriminantAnalysis' in model.__class__.__name__ or 'NB' in model.__class__.__name__:
                self.task = 'classification'
            else:
                self.task = 'regression'
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        if self.task == "classification":
            return accuracy_score(y, y_pred)
        else:
            return r2_score(y, y_pred)
        
    def set_params(self, **params):
        self.model.set_params(**params)
        return self
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)
    
    def confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def classification_report(self, X, y):
        y_pred = self.predict(X)
        return classification_report(y, y_pred)

    def get_task(self):
        return self.task

    def f1_score(self, X, y):
        y_pred = self.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        if '1' in report:
            return report['1']['f1-score']
        elif '1.0' in report:
            return report['1.0']['f1-score']
        else:
            return 0.0
    

