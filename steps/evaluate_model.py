import logging
import pandas as pd
from zenml import step
from src.evaluation import Metrics
import numpy as np
from mlflow import start_run

@step(enable_cache = False)
def model_evaluation(y_test:pd.Series, y_pred:np.ndarray, y_proba:np.ndarray) -> dict:
    '''
    Evaluate the model 

    Args:
       y_test: Testing labels 
       y_pred: Predicted labels 
    Return:
    dict{
        f1 metrics
        recall metrics
        precision metrics
        roc_auc metrics}
    '''
    try:
        logging.info('Metrics Calculation has Started')
        
        with start_run(nested=True):
            
            metric = Metrics()
            metric_result = metric.calculate_scores(y_test, y_pred, y_proba)
            
        logging.info('Metrics Calculation Complete')
        
        return metric_result
        
    except Exception as e:
        
        logging.error(f" An Error Occurred during metrics calculation {e}")
        raise e
