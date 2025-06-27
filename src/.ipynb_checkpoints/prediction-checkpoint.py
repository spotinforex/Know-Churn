from steps.train_model import train_model
from src.encoder import Encoder 
import logging 
import pandas as pd
import numpy as np
import joblib
from abc import ABC, abstractmethod


class Pred(ABC):
    '''
    Abstract class for model prediction 
    '''
    @abstractmethod 
    def predict(self, X_test:pd.DataFrame,file_path, encoder_path):
        '''
        Makes prediction for model 
        
        Args:
            X_test: prediction dataframe
        '''
        pass

class Prediction(Pred):
    def predict(self, X_test:pd.DataFrame, file_path:str, encoder_path:str) -> np.ndarray:
        try:
           
            model = joblib.load(file_path)
            encoder = joblib.load(encoder_path)
            logging.info(" Data Transformation in Progress")
            X_test = encoder.transform(X_test)
            logging.info(" Data Transformation Complete")
            logging.info(" Prediction in Progress")
            y_prob = model.predict_proba(X_test)[ : , 1]
            y_pred = (y_prob > 0.37).astype(int)
            logging.info("Prediction Completed")
            return y_pred.ravel(), y_prob.ravel()

        except Exception as e:
            logging.error(f"An Error occured during prediction {e}")
            raise e
        
        
        
        