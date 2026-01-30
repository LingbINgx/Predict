import utils.activeLearn 
import datetime
from sklearn.metrics import precision_recall_curve
from loguru import logger


@logger.catch
def write_results_to_file(model, X_test, y_test):
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = f1.argmax()
        th = thresholds[best_idx]
        y_pred = (y_prob > th).astype(int)
        
        with open(f'../out/{model.__class__.__name__}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
            
            f.write(f'Best threshold: {th}\n')
            f.write(f'Precision: {precision[best_idx]}\n')
            f.write(f'Recall: {recall[best_idx]}\n')
            f.write(f'F1: {f1[best_idx]}\n')
            f.write(f'Confusion Matrix:\n{utils.activeLearn.confusion_matrix(y_test, y_pred)}\n')
            f.write(f'Classification Report:\n{utils.activeLearn.classification_report(y_test, y_pred)}\n')
    except Exception as e:
        logger.error(f"Error in write_results_to_file: {e}")        
    