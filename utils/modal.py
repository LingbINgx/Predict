
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


class BasicActiveLearner:
    """
    基础的主动学习模板类
    """
    def __init__(self, estimator=None, query_strategy='uncertainty', random_state=42):
        """
        初始化主动学习器
        
        参数:
        estimator: 基础分类器，默认使用随机森林
        query_strategy: 查询策略 ('uncertainty', 'margin', 'entropy')
        random_state: 随机种子
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.estimator = estimator if estimator is not None else RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        )
        
        params_dict = {
            'uncertainty': modAL.uncertainty.uncertainty_sampling,
            'margin': modAL.uncertainty.margin_sampling,
            'entropy': modAL.uncertainty.entropy_sampling
        }
        self.query_strategy = params_dict.get(query_strategy, modAL.uncertainty.uncertainty_sampling)
            
        self.learner = None
        self.X_labeled = None
        self.y_labeled = None
        self.X_unlabeled = None
        self.y_unlabeled = None
        self.X_pool = None
        self.y_pool = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self, X, y, test_size=0.5):
        """
        加载数据并划分测试集
        
        参数:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例
        """
        self.X_pool, self.X_test, self.y_pool, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )
        print(f"训练池样本数: {len(self.X_pool)}")
        print(f"测试集样本数: {len(self.X_test)}")
        return self.X_pool, self.y_pool, self.X_test, self.y_test
        
    def initialize_data(self, initial_size=10):# -> tuple[Any, Any, Any, Any]:
        """
        初始化数据，划分已标注和未标注数据
        """
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
        """创建主动学习器"""
        self.learner = ActiveLearner(
            estimator=self.estimator,
            query_strategy=self.query_strategy,
            X_training=self.X_labeled,
            y_training=self.y_labeled
        )
        return self.learner
    
    def active_learning_cycle(self, n_queries=50, query_size=1, th=0.5):
        """
        执行主动学习循环
        
        参数:
        n_queries: 查询轮次
        query_size: 每轮查询样本数
        th: 分类阈值
        
        返回:
        metrics: 包含每轮次性能指标的字典
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
            
            self.X_unlabeled = np.delete(self.X_unlabeled, query_idx, axis=0)
            self.y_unlabeled = np.delete(self.y_unlabeled, query_idx, axis=0)
            
            # 评估性能
            y_prob = self.learner.predict_proba(self.X_test)[:, 1]
            y_pred = (y_prob > th).astype(int)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            metrics["acc"].append(accuracy)
            metrics["confusion"].append(confusion_matrix(self.y_test, y_pred))
            metrics["report"].append(classification_report(self.y_test, y_pred, output_dict=True))
            
            print(f'Accuracy after query {i + 1}: {accuracy:.4f}')
            print(confusion_matrix(self.y_test, y_pred))
            print(classification_report(self.y_test, y_pred))
            
        return metrics
    
    def plot_precision_recall_curve(self, metrics):
        """
        绘制精确率-召回率曲线
        """
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


# def visualize_results(active_performance, random_performance):
#     """
#     可视化主动学习与随机采样的对比
#     """
#     plt.figure(figsize=(12, 6))
    
#     # 准确率曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(active_performance, 'b-', label='主动学习', linewidth=2)
#     plt.plot(random_performance, 'r--', label='随机采样', linewidth=2)
#     plt.xlabel('查询轮次')
#     plt.ylabel('测试集准确率')
#     plt.title('主动学习 vs 随机采样')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # 性能提升对比
#     plt.subplot(1, 2, 2)
#     active_cumulative = np.cumsum(active_performance)
#     random_cumulative = np.cumsum(random_performance)
#     plt.bar(['主动学习', '随机采样'], 
#             [np.mean(active_performance), np.mean(random_performance)],
#             color=['blue', 'red'], alpha=0.7)
#     plt.ylabel('平均准确率')
#     plt.title('平均性能对比')
    
#     plt.tight_layout()
#     plt.show()
    
#     # 打印最终性能
#     print(f"\n最终性能对比:")
#     print(f"主动学习最终准确率: {active_performance[-1]:.4f}")
#     print(f"随机采样最终准确率: {random_performance[-1]:.4f}")
#     print(f"性能提升: {active_performance[-1] - random_performance[-1]:.4f}")