from zenml import step
import pandas as pd
from typing import Tuple
import logging
from catboost import CatBoostClassifier
from src.encoder import Encoder 
from src.data_cleaning import DataHandler, DataDivideStrategy, DataPreProcessStrategy

@step
def make_prediction(
    dataframe: pd.DataFrame,
    model: CatBoostClassifier,
    encoder: Encoder
) -> pd.DataFrame:
    '''
    Predicts on inference data.

    Args:
        dataframe (pd.DataFrame): Input features.
        model (CatBoostClassifier): Trained CatBoost model.
        encoder (Encoder): Fitted encoder object.

    Returns:
        pd.DataFrame: Input dataframe with predictions and optional probabilities.
    '''
    try:
        logging.info("Prediction in progress.")

        # Preprocess
        data_cleaning = DataHandler(data=dataframe, strategy=DataPreProcessStrategy())
        preprocessed_data = data_cleaning.handle_data()
        df_encoded = encoder.transform(preprocessed_data)

        # Predict
        predictions = model.predict(df_encoded)
        try:
            probabilities = model.predict_proba(df_encoded)[:, 1]
        except Exception as e:
            logging.warning(f"Could not compute probabilities: {e}")
            probabilities = None

        # Combine and return
        dataframe["Prediction"] = predictions
        if probabilities is not None:
            dataframe["Probability"] = probabilities

        logging.info("Prediction successful.")
        return dataframe

    except Exception as e:
        logging.exception(f"An error occurred during prediction: {e}")
        raise e
