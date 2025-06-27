import logging 
import pandas as pd
from zenml import step 
from src.model_dev import CatBoost
from steps.Config import ModelNameConfig
from typing import Tuple
from typing_extensions import Annotated


@step(enable_cache = False)
def train_model(X_train: pd.DataFrame,
                y_train: pd.Series ) -> Tuple[
        Annotated[str, 'model_path'],
        Annotated[str, 'encoder_path']
        ]:
    '''
    Training the model
    
    Args:
        X_train: training data 
        y_train: training labels
    Returns:
        model_path
        encoder_path
    '''
    try:
        logging.info("Model Training Started")
        config = ModelNameConfig()
        model = CatBoost()
        file_path, encoder_path = model.train(X_train, y_train,**config.to_dict())
        logging.info("Model Training Completed")
        return file_path, encoder_path
        
    except Exception as e:
        logging.error(f" Error during training model {e}")
        raise e
