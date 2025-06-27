import logging 
import pandas as pd 
from zenml import step
from src.prediction import Prediction
import numpy as np
from typing import Tuple
from typing_extensions import Annotated

@step(enable_cache = False)
def predict_data(X_test:pd.DataFrame, file_path, encoder_path) -> Tuple[
    Annotated[np.ndarray, 'y_pred'],
    Annotated[np.ndarray, 'y_proba']
    ]:
    '''
    Predicts future data 
    
    Args:
    X_test: Test data
    
    Returns:
    y_pred: predicted labels
    '''
    try:
        logging.info("Prediction has Started")
        pred = Prediction()
        y_pred, y_proba = pred.predict(X_test, file_path, encoder_path)
        logging.info(" Prediction has Finished")
    
        return y_pred, y_proba
    
    except Exception as e:
        logging.error(f" Error Occured during predicting model {e}")
        raise e