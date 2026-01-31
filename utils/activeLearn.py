
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import modAL
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
import modAL.uncertainty
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils.wraps import print_return

class BasicActiveLearner:

    def __init__(self, estimator=None, query_strategy='uncertainty', random_state=42):
        """
        初始化主动学习器

        :param estimator: 基础分类器
        :param query_strategy: 查询策略 ('uncertainty', 'margin', 'entropy')
        :param random_state: 随机种子
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.estimator = estimator if estimator is not None else RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        )
        
        query_strategy_dict = {
            'uncertainty': modAL.uncertainty.uncertainty_sampling,
            'margin': modAL.uncertainty.margin_sampling,
            'entropy': modAL.uncertainty.entropy_sampling
        }
        self.query_strategy = query_strategy_dict.get(query_strategy, modAL.uncertainty.uncertainty_sampling)
            
        self.learner = None
        self.X_labeled = None
        self.y_labeled = None
        self.X_unlabeled = None
        self.y_unlabeled = None
        self.X_pool = None
        self.y_pool = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self, X, y, test_size=0.5, imbalance_handle='None', **kwargs):
        """
        加载数据并划分测试集

        :param X: 特征数据
        :param y: 标签数据
        :param test_size: 测试集比例
        :param imbalance_handle: 进行不平衡处理 ('None', 'SMOTE', 'SMOTEENN')
        """
        self.X_pool, self.X_test, self.y_pool, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )
        im_dict = {
            'SMOTE': SMOTE,
            'SMOTEENN': SMOTEENN
        }
        if imbalance_handle in im_dict:
            smote = im_dict.get(imbalance_handle, SMOTE)(random_state=self.random_state, **kwargs)
            self.X_pool, self.y_pool = smote.fit_resample(self.X_pool, self.y_pool)
            
        print(f"训练池样本数: {len(self.X_pool)}")
        print(f"测试集样本数: {len(self.X_test)}")
        return self.X_pool, self.y_pool, self.X_test, self.y_test
    
        
    def initialize_data(self, initial_size=10):# -> tuple[Any, Any, Any, Any]:
        self.X_labeled, self.X_unlabeled, self.y_labeled, self.y_unlabeled = train_test_split(
            self.X_pool, self.y_pool,
            train_size=initial_size,
            stratify=self.y_pool,
            random_state=self.random_state
        )
        print(f"初始已标注样本数: {len(self.X_labeled)}")
        print(f"初始未标注样本数: {len(self.X_unlabeled)}")
        return self.X_labeled, self.y_labeled, self.X_unlabeled, self.y_unlabeled
    
    
    def create_learner(self):
        self.learner = ActiveLearner(
            estimator=self.estimator,
            query_strategy=self.query_strategy,
            X_training=self.X_labeled,
            y_training=self.y_labeled
        )
        return self.learner
    
    
    @logger.catch()
    def active_learning_cycle(self, n_queries=50, query_size=1, threshold=0.5, **kwargs):
        """
        执行主动学习循环
        
        :param n_queries: 查询轮次
        :param query_size: 每轮查询样本数
        :param threshold: 分类阈值
        """
        if self.learner is None:
            self.create_learner()
            
        metrics = {"acc": [], "confusion": [], "report": []}
        
        for i in range(n_queries):
            
            if self.X_unlabeled is None or len(self.X_unlabeled) < query_size:
                print(f'Stopping at iteration {i + 1}: Only {len(self.X_unlabeled)} samples left in pool (need {query_size})')
                break
                
            # 查询最不确定的样本
            query_idx, query_X = self.learner.query(
                X_pool=self.X_unlabeled, 
                n_instances=query_size
            )
            query_idx = np.asarray(query_idx).reshape(-1)
            query_y = self.y_unlabeled[query_idx]
            
            self.learner.teach(
                X=query_X,
                y=query_y
            )
            
            ratio = self._update_ratio()
            _update_ensemble_weights(self.learner.estimator, ratio)
            
            self.X_unlabeled = np.delete(self.X_unlabeled, query_idx, axis=0)
            self.y_unlabeled = np.delete(self.y_unlabeled, query_idx, axis=0)
            
            y_prob = self.learner.predict_proba(self.X_test)[:, 1]
            y_pred = (y_prob > threshold).astype(int)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            metrics["acc"].append(accuracy)
            metrics["confusion"].append(confusion_matrix(self.y_test, y_pred))
            metrics["report"].append(classification_report(self.y_test, y_pred, output_dict=True))
            
            print(f'Accuracy after query {i + 1}: {accuracy:.4f}')
            print(confusion_matrix(self.y_test, y_pred))
            print(classification_report(self.y_test, y_pred))
            
        return metrics
    
    def plot_precision_recall_curve(self, metrics):
        precisions = [report['1.0']['precision'] for report in metrics['report']]
        recalls = [report['1.0']['recall'] for report in metrics['report']]
        
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.show()


    @print_return
    def _update_ratio(self):
        y = self.learner.y_training
        neg_cnt = np.sum(y == 0)
        pos_cnt = np.sum(y == 1)
        return neg_cnt / pos_cnt if pos_cnt > 0 else 1.0

    
def _update_ensemble_weights(model, ratio):
    weight_dict = {0: 1.0, 1: ratio}
    if isinstance(model, XGBClassifier):
        model.set_params(scale_pos_weight=ratio)
    elif isinstance(model, LGBMClassifier):
        model.set_params(scale_pos_weight=ratio)
    elif hasattr(model, 'class_weight'):
        model.set_params(class_weight=weight_dict)
    elif hasattr(model, 'estimators'):
        estimators = getattr(model, 'estimators', [])
        if isinstance(estimators, list):
            for _, est in estimators:
                _update_ensemble_weights(est, ratio)
        final_est = getattr(model, 'final_estimator', None)
        if final_est:
            _update_ensemble_weights(final_est, ratio)
            

        
@logger.catch()
def instant_launch(X, y,estimator=None, query_strategy='uncertainty', random_state=42,
                   initial_size=10, n_queries=50, query_size=1, threshold=0.5,
                   test_size=0.5, imbalance_handle='None', sampling_strategy=0.8, **kwargs):
    """
    一键启动主动学习流程
    
    :param X: 样本
    :param y: 标签
    :param estimator: 分类器
    :param query_strategy: 查询策略  ('uncertainty', 'margin', 'entropy')
    :param random_state:   随机种子
    :param initial_size: 初始样本数量
    :param n_queries: 查询次数
    :param query_size: 每次查询的样本数量
    :param threshold: 预测阈值
    :param test_size: 测试集比例
    :param imbalance_handle: 不平衡处理 ('None', 'SMOTE', 'SMOTEENN')
    :param sampling_strategy: 采样策略
    :param kwargs: 其他参数
    """
    train = BasicActiveLearner(estimator=estimator, query_strategy=query_strategy, random_state=random_state)
    train.load_data(X.values, y.values, test_size=test_size, imbalance_handle=imbalance_handle,  sampling_strategy=sampling_strategy)
    train.initialize_data(initial_size=initial_size)
    metrics = train.active_learning_cycle(n_queries=n_queries, query_size=query_size, threshold=threshold, **kwargs)
   
    return metrics, train
