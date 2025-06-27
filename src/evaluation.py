from abc import ABC, abstractmethod 
import logging 
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import numpy as np
import pandas as pd
import mlflow


class Evaluation(ABC):
    '''
    Abstract class defining the strategy
    '''

    @abstractmethod 
    def calculate_scores(self, y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray):
        ''' Calculate the metrics for the model 
        Args:
            y_test: true labels 
            y_pred: predicted labels 
        '''
        pass


class Metrics(Evaluation):
    '''
    Evaluation strategy that uses F1 scores, Recall_scores, Precision_score and Roc_Auc_score 
    '''
    def calculate_scores(self, y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        try: 
            
            metric = {}         
            f1 = f1_score(y_test,y_pred)
            recall = recall_score (y_test, y_pred)
            precision = precision_score(y_test,y_pred)
            roc_auc = roc_auc_score(y_test,y_proba)
            metric.update({'F1':f1, 'Recall':recall, 'Precision':precision, 'Roc_Auc':roc_auc})       
            for name, value in metric.items():
                mlflow.log_metric(name, value)
            return metric
            
        except Exception as e:
            logging.error('Error Calculating metrics')
            raise e